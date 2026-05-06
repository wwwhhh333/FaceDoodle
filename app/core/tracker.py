import glob
import os
import cv2
import numpy as np
import threading
import time
import queue
from multiprocessing import Queue, Process, Event
from app.ai.agent import FaceDoodleAgent
from app.ai.generator import ComfyClient
from app.core.face_mesh import FaceDetector
from app.core.renderer import (render_scene, render_loading_progress, render_face_mesh,
                                apply_head_pose_skew, _build_location_quad)
from app.core.face_draw import FaceDrawCanvas
from app.utils.image_proc import load_rgba_sticker
from app.utils import storage

ai_state = {
    "is_generating": False,
    "current_sticker": None,
    "loading_sticker": None
}


def _handle_img2img(cmd, result_queue, mock):
    global ai_state
    try:
        prompt_text = cmd.get("prompt_text", "")
        image_path = cmd.get("image_path", "")
        target_location = cmd.get("target_location", "forehead_top")
        scale = float(cmd.get("scale", 1.0))

        # Preprocess: composite onto white bg for ControlNet (needs dark-on-light)
        if image_path and os.path.exists(image_path):
            raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if raw is not None and raw.shape[2] == 4:
                canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
                h, w = raw.shape[:2]
                sf = min(900 / max(w, h), 1.0)
                new_w, new_h = int(w * sf), int(h * sf)
                resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
                y_off = (1024 - new_h) // 2
                x_off = (1024 - new_w) // 2
                alpha = resized[:, :, 3:4] / 255.0
                rgb = resized[:, :, :3]
                roi = canvas[y_off:y_off + new_h, x_off:x_off + new_w]
                blended = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = blended
                preprocessed_path = os.path.join(os.path.dirname(image_path), "preprocessed_for_controlnet.png")
                cv2.imwrite(preprocessed_path, canvas)
                image_path = preprocessed_path

        print(f"[AI Worker] img2img 模式: prompt='{prompt_text}', image='{image_path}'")

        if mock:
            mock_path = "assets/static/loading.png"
            temp_files = glob.glob("assets/temp/*.png")
            if temp_files:
                temp_files.sort(key=os.path.getmtime, reverse=True)
                mock_path = temp_files[0]
            new_sticker = load_rgba_sticker(mock_path)
        else:
            comfy_client = ComfyClient()
            generated_path = comfy_client.generate_sync(
                prompt_text=prompt_text,
                workflow_name="img2img_controlnet_workflow_api.json",
                input_image_path=image_path,
            )
            if generated_path and os.path.exists(generated_path):
                new_sticker = load_rgba_sticker(generated_path)
            else:
                new_sticker = None
                print("[AI Worker] img2img 错误：ComfyUI 未生成有效文件")

        if new_sticker is not None:
            result_data = {
                "sticker": new_sticker,
                "location": target_location,
                "scale": scale,
                "prompt": cmd.get("display_name", prompt_text),
                "positive_prompt": prompt_text,
                "timestamp": time.time()
            }
            result_queue.put(result_data)
            print(f"[AI Worker] img2img 任务完成, 位置: {target_location}")
        else:
            result_queue.put({"error": "ComfyUI 未能生成有效图片，请检查服务是否正常运行"})

    except Exception as e:
        print(f"[AI Worker] img2img 异常: {str(e)}")
        result_queue.put({"error": str(e)})
    finally:
        ai_state["is_generating"] = False


def ai_worker_thread(user_command, result_queue, api_key, mock=False):
    global ai_state

    if isinstance(user_command, dict) and user_command.get("type") == "img2img":
        _handle_img2img(user_command, result_queue, mock)
        return

    agent = FaceDoodleAgent(api_key=api_key)

    try:
        print(f"[AI Worker] 正在解析指令: {user_command}")
        task_info = agent.parse_command(user_command)

        if not task_info or "positive_prompt" not in task_info:
            raise ValueError("Agent 解析指令失败")

        if mock:
            mock_path = "assets/static/loading.png"
            temp_files = glob.glob("assets/temp/*.png")
            if temp_files:
                temp_files.sort(key=os.path.getmtime, reverse=True)
                mock_path = temp_files[0]
            print(f"[AI Worker] Mock 模式: 跳过 ComfyUI, 使用 {mock_path}")
            new_sticker = load_rgba_sticker(mock_path)
        else:
            comfy_client = ComfyClient()
            print(f"[AI Worker] 正在调用 ComfyUI 生成: {task_info['positive_prompt']}")
            image_path = comfy_client.generate_sync(
                prompt_text=task_info["positive_prompt"],
                workflow_name=task_info["workflow"]
            )
            if image_path and os.path.exists(image_path):
                new_sticker = load_rgba_sticker(image_path)
            else:
                new_sticker = None
                print("[AI Worker] 错误：ComfyUI 未生成有效文件")

        if new_sticker is not None:
            result_data = {
                "sticker": new_sticker,
                "location": task_info["target_location"],
                "scale": task_info.get("scale", 1.0),
                "prompt": user_command,
                "positive_prompt": task_info.get("positive_prompt", ""),
                "timestamp": time.time()
            }
            result_queue.put(result_data)
            print(f"[AI Worker] 任务完成，位置: {task_info['target_location']}")
        else:
            result_queue.put({"error": "ComfyUI 未能生成有效图片，请检查服务是否正常运行"})

    except Exception as e:
        print(f"[AI Worker] 发生异常: {str(e)}")
        result_queue.put({"error": str(e)})

    finally:
        ai_state["is_generating"] = False


def producer(frame_queue, stop_event):
    from app.utils.config_loader import get_config
    cfg = get_config()
    cam_cfg = cfg.get("camera", {})

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Producer] 错误：无法打开摄像头 (index 0)")
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))

    while not stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        try:
            frame_queue.put(frame, timeout=0.1)
        except Exception:
            pass

    cap.release()
    print("[Producer] 已退出")


def consumer(in_queue, display_queue, command_queue, adjustment_queue, gallery_queue, draw_queue, api_key, stop_event, mock=False):
    detector = FaceDetector()
    result_queue = queue.Queue()

    active_content = None
    pending_command = None

    adjustment = {
        "offset_x": 0.0,
        "offset_y": 0.0,
        "rotation": 0.0,
        "scale_mult": 1.0,
        "edit_mode": False,
    }
    _sticker_adjustments = {}  # sticker_id -> adjustment dict

    face_canvas = FaceDrawCanvas()
    face_draw_active = False
    face_draw_region = "full_face"

    if mock:
        temp_files = glob.glob("assets/temp/*.png")
        if temp_files:
            temp_files.sort(key=os.path.getmtime, reverse=True)
            mock_sticker = load_rgba_sticker(temp_files[0])
            if mock_sticker is not None:
                active_content = {"sticker": mock_sticker, "location": "forehead", "scale": 1.0}
                print(f"[Consumer] Mock 模式: 已加载 {temp_files[0]}")

    try:
        while not stop_event.is_set():
            try:
                frame = in_queue.get(block=True, timeout=0.1)
            except Exception:
                continue

            if not command_queue.empty():
                pending_command = command_queue.get()

            if pending_command and not ai_state["is_generating"]:
                ai_state["is_generating"] = True
                ai_state["generation_start_time"] = time.time()
                cmd = pending_command
                pending_command = None
                threading.Thread(
                    target=ai_worker_thread,
                    args=(cmd, result_queue, api_key, mock),
                    daemon=True
                ).start()

            try:
                new_content = result_queue.get(block=False)
                if "error" in new_content:
                    display_queue.put({"action": "generation_failed", "error": new_content["error"]})
                else:
                    old_sid = active_content.get("sticker_id") if active_content else None
                    if old_sid:
                        saved = dict(adjustment)
                        _sticker_adjustments[old_sid] = saved
                        storage.save_sticker_adjustments(old_sid, saved)
                    active_content = new_content
                    adjustment["offset_x"] = 0.0
                    adjustment["offset_y"] = 0.0
                    adjustment["rotation"] = 0.0
                    adjustment["scale_mult"] = 1.0
                    try:
                        sticker_id = storage.save_sticker(
                            new_content['sticker'],
                            {"prompt": new_content.get("prompt", ""),
                             "location": new_content.get("location", "forehead"),
                             "scale": new_content.get("scale", 1.0)}
                        )
                        display_queue.put({"action": "sticker_saved", "sticker_id": sticker_id})
                    except Exception:
                        pass
            except queue.Empty:
                pass

            face_data = detector.get_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw_face_w = float(face_data.get("face_width", 200.0)) if face_data else 200.0
            face_w = max(raw_face_w, 1.0)

            # 处理调整指令（在人脸检测之后，用 face_w 做相对转换）
            while not adjustment_queue.empty():
                try:
                    msg = adjustment_queue.get(block=False)
                except Exception:
                    break
                action = msg.get("action")
                if action == "toggle_edit":
                    adjustment["edit_mode"] = not adjustment["edit_mode"]
                elif action == "move":
                    adjustment["offset_x"] += msg.get("dx", 0.0) / face_w
                    adjustment["offset_y"] += msg.get("dy", 0.0) / face_w
                elif action == "rotate":
                    adjustment["rotation"] += msg.get("d_angle", 0.0)
                elif action == "scale":
                    adjustment["scale_mult"] *= msg.get("multiplier", 1.0)
                    adjustment["scale_mult"] = max(0.05, adjustment["scale_mult"])
                elif action == "reset":
                    adjustment["offset_x"] = 0.0
                    adjustment["offset_y"] = 0.0
                    adjustment["rotation"] = 0.0
                    adjustment["scale_mult"] = 1.0

            # 处理画廊指令（加载已有贴纸）
            while not gallery_queue.empty():
                try:
                    gmsg = gallery_queue.get(block=False)
                except Exception:
                    break
                if gmsg.get("action") == "load_template":
                    t = gmsg.get("template")
                    if t and t.get("image") is not None:
                        old_sid = active_content.get("sticker_id") if active_content else None
                        if old_sid:
                            saved = dict(adjustment)
                            _sticker_adjustments[old_sid] = saved
                            storage.save_sticker_adjustments(old_sid, saved)
                        active_content = {
                            "sticker": t["image"],
                            "location": t.get("region", "forehead_top"),
                            "scale": 1.0,
                            "sticker_id": t["id"],
                            "prompt": t.get("name", "模板"),
                        }
                        adjustment["offset_x"] = 0.0
                        adjustment["offset_y"] = 0.0
                        adjustment["rotation"] = 0.0
                        adjustment["scale_mult"] = 1.0
                    else:
                        active_content = None
                        adjustment["offset_x"] = 0.0
                        adjustment["offset_y"] = 0.0
                        adjustment["rotation"] = 0.0
                        adjustment["scale_mult"] = 1.0
                elif gmsg.get("action") == "load_sticker":
                    sid = gmsg.get("sticker_id")
                    # 保存当前贴纸的调整状态
                    old_sid = active_content.get("sticker_id") if active_content else None
                    if old_sid:
                        saved = dict(adjustment)
                        _sticker_adjustments[old_sid] = saved
                        storage.save_sticker_adjustments(old_sid, saved)
                    if sid:
                        loaded, meta = storage.get_sticker(sid)
                        if loaded is not None and meta is not None:
                            active_content = {
                                "sticker": loaded,
                                "location": meta.get("region", "forehead_top"),
                                "scale": meta.get("scale", 1.0),
                                "sticker_id": sid,
                                "prompt": meta.get("prompt", ""),
                            }
                            # 恢复该贴纸的历史调整：优先内存，其次磁盘，否则重置
                            saved = _sticker_adjustments.get(sid) or storage.get_sticker_adjustments(sid)
                            if saved:
                                adjustment["offset_x"] = saved.get("offset_x", 0.0)
                                adjustment["offset_y"] = saved.get("offset_y", 0.0)
                                adjustment["rotation"] = saved.get("rotation", 0.0)
                                adjustment["scale_mult"] = saved.get("scale_mult", 1.0)
                                _sticker_adjustments[sid] = saved
                            else:
                                adjustment["offset_x"] = 0.0
                                adjustment["offset_y"] = 0.0
                                adjustment["rotation"] = 0.0
                                adjustment["scale_mult"] = 1.0
                    else:
                        active_content = None
                        adjustment["offset_x"] = 0.0
                        adjustment["offset_y"] = 0.0
                        adjustment["rotation"] = 0.0
                        adjustment["scale_mult"] = 1.0

            # 处理面部绘制指令
            while not draw_queue.empty():
                try:
                    msg = draw_queue.get(block=False)
                except Exception:
                    break
                action = msg.get("action")
                if action == "toggle_draw_mode":
                    face_draw_active = not face_draw_active
                    if face_draw_active:
                        adjustment["edit_mode"] = False
                elif action == "set_region":
                    face_draw_region = msg.get("region", "forehead_full")
                elif action == "set_brush":
                    face_canvas.set_brush_size(msg.get("brush_size", 12))
                    face_canvas.set_brush_color(msg.get("brush_color", (0, 0, 0, 255)))
                elif action == "toggle_eraser":
                    face_canvas.set_eraser_mode(msg.get("eraser_mode", True))
                elif action == "set_brush_type":
                    face_canvas.set_brush_type(msg.get("brush_id", "hard_round"))
                elif action == "set_pressure_mode":
                    face_canvas.set_pressure_mode(msg.get("mode", "both"))
                elif action == "set_spacing":
                    face_canvas.set_spacing(msg.get("coef", 0.3))
                elif action == "set_scatter":
                    face_canvas.set_scatter(msg.get("px", 0.0))
                elif action == "undo":
                    face_canvas.undo()
                elif action == "clear":
                    face_canvas.clear()
                elif action == "stroke_begin" and face_draw_active and face_data:
                    quad = _build_location_quad(face_data, face_draw_region, (512, 512, 4), scale=1.5)
                    if quad is not None:
                        quad = apply_head_pose_skew(quad, face_data)
                        face_canvas.begin_stroke(quad)

                elif action == "stroke_point" and face_draw_active:
                    pt = msg.get("point")
                    if pt is not None:
                        face_canvas.set_pressure(msg.get("pressure", 1.0))
                        face_canvas.add_stroke_point(pt)

                elif action == "stroke_end":
                    face_canvas.end_stroke()
                elif action == "save":
                    result = face_canvas.get_result()
                    if result is not None:
                        sid = storage.save_sticker(result, {
                            "prompt": "全脸手绘",
                            "location": face_draw_region,
                            "scale": 1.0,
                        })
                        display_queue.put({"action": "sticker_saved", "sticker_id": sid})

            if face_data:
                if ai_state["is_generating"]:
                    frame = render_loading_progress(frame, face_data, ai_state)

                if ai_state["is_generating"] or adjustment.get("edit_mode") or face_draw_active:
                    frame = render_face_mesh(frame, face_data)

                if active_content is not None:
                    frame = render_scene(frame, face_data, active_content, adjustment)

                if face_draw_active and face_canvas.has_content:
                    draw_content = {
                        "sticker": face_canvas.canvas,
                        "location": face_draw_region,
                        "scale": 1.5,
                    }
                    frame = render_scene(frame, face_data, draw_content, None)

            try:
                display_queue.put(frame, timeout=0.05)
            except Exception:
                pass
    finally:
        detector.close()
        print("[Consumer] 已退出")
