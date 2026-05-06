import glob
import os
import uuid
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
                                apply_head_pose_skew, _build_location_quad,
                                composite_stickers_to_merged)
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

    active_content = None  # backward compat for AI generation result
    pending_command = None

    active_stickers = []          # list[dict], z-order: index 0 = bottom
    edit_target_id = None         # instance_id currently receiving edit commands
    adjustments = {}              # instance_id -> {offset_x, offset_y, rotation, scale_mult}
    _sticker_adjustments = {}     # sticker_id -> default adjustment dict (for restore)
    MAX_STICKERS = 20
    cached_face_data = None       # last good face data for merge fallback

    face_canvas = FaceDrawCanvas()
    face_draw_active = False
    face_draw_region = "full_face"

    if mock:
        temp_files = glob.glob("assets/temp/*.png")
        if temp_files:
            temp_files.sort(key=os.path.getmtime, reverse=True)
            mock_sticker = load_rgba_sticker(temp_files[0])
            if mock_sticker is not None:
                instance_id = str(uuid.uuid4())
                active_stickers.append({
                    "instance_id": instance_id, "sticker_id": None,
                    "sticker": mock_sticker, "location": "forehead",
                    "scale": 1.0, "prompt": "Mock",
                })
                adjustments[instance_id] = {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0}
                edit_target_id = instance_id
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
                    try:
                        if len(active_stickers) >= MAX_STICKERS:
                            print(f"[Consumer] 贴纸数量已达上限 ({MAX_STICKERS})，跳过自动添加")
                            continue
                        sticker_id = storage.save_sticker(
                            new_content['sticker'],
                            {"prompt": new_content.get("prompt", ""),
                             "location": new_content.get("location", "forehead"),
                             "scale": new_content.get("scale", 1.0)}
                        )
                        instance_id = str(uuid.uuid4())
                        # Load default adjustment from previous instances of same sticker
                        saved_adj = _sticker_adjustments.get(sticker_id) or storage.get_sticker_adjustments(sticker_id)
                        adj = dict(saved_adj) if saved_adj else {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0}
                        adjustments[instance_id] = adj
                        active_stickers.append({
                            "instance_id": instance_id,
                            "sticker_id": sticker_id,
                            "sticker": new_content['sticker'],
                            "location": new_content.get("location", "forehead"),
                            "scale": new_content.get("scale", 1.0),
                            "prompt": new_content.get("prompt", ""),
                        })
                        edit_target_id = instance_id
                        display_queue.put({"action": "sticker_saved", "sticker_id": sticker_id})
                        active_content = new_content  # backward compat
                    except Exception as e:
                        print(f"[Consumer] 保存贴纸失败: {e}")
            except queue.Empty:
                pass

            face_data = detector.get_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if face_data and "nose_tip" in face_data:
                cached_face_data = face_data
            raw_face_w = float(face_data.get("face_width", 200.0)) if face_data else 200.0
            face_w = max(raw_face_w, 1.0)

            # 处理调整指令（在人脸检测之后，用 face_w 做相对转换）
            while not adjustment_queue.empty():
                try:
                    msg = adjustment_queue.get(block=False)
                except Exception:
                    break
                action = msg.get("action")
                if edit_target_id is None:
                    continue
                if action == "toggle_edit":
                    pass
                elif action == "move":
                    dx = msg.get("dx", 0.0) / face_w
                    dy = msg.get("dy", 0.0) / face_w
                    if edit_target_id in adjustments:
                        adjustments[edit_target_id]["offset_x"] += dx
                        adjustments[edit_target_id]["offset_y"] += dy
                elif action == "rotate":
                    d_angle = msg.get("d_angle", 0.0)
                    if edit_target_id in adjustments:
                        adjustments[edit_target_id]["rotation"] += d_angle
                elif action == "scale":
                    mult = msg.get("multiplier", 1.0)
                    if edit_target_id in adjustments:
                        adjustments[edit_target_id]["scale_mult"] *= mult
                        adjustments[edit_target_id]["scale_mult"] = max(0.05, adjustments[edit_target_id]["scale_mult"])
                elif action == "reset":
                    if edit_target_id in adjustments:
                        adjustments[edit_target_id]["offset_x"] = 0.0
                        adjustments[edit_target_id]["offset_y"] = 0.0
                        adjustments[edit_target_id]["rotation"] = 0.0
                        adjustments[edit_target_id]["scale_mult"] = 1.0

            # 处理画廊指令（加载已有贴纸 / 多贴纸管理）
            while not gallery_queue.empty():
                try:
                    gmsg = gallery_queue.get(block=False)
                except Exception:
                    break
                action = gmsg.get("action")

                def _add_sticker_instance(sticker_img, sid, location, scale, prompt):
                    """Helper: add a sticker instance to active_stickers."""
                    instance_id = str(uuid.uuid4())
                    saved = _sticker_adjustments.get(sid) or storage.get_sticker_adjustments(sid)
                    adj = dict(saved) if saved else {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0}
                    adjustments[instance_id] = adj
                    active_stickers.append({
                        "instance_id": instance_id, "sticker_id": sid,
                        "sticker": sticker_img, "location": location,
                        "scale": scale, "prompt": prompt,
                    })
                    return instance_id

                if action == "add_sticker":
                    if len(active_stickers) >= MAX_STICKERS:
                        print(f"[Consumer] 贴纸数量已达上限 ({MAX_STICKERS})，忽略添加请求")
                        continue
                    sid = gmsg.get("sticker_id")
                    if sid:
                        loaded, meta = storage.get_sticker(sid)
                        if loaded is not None and meta is not None:
                            iid = _add_sticker_instance(loaded, sid,
                                meta.get("region", "forehead_top"), meta.get("scale", 1.0),
                                meta.get("prompt", ""))
                            edit_target_id = iid
                            # backward compat
                            active_content = {"sticker": loaded, "location": meta.get("region", "forehead_top"),
                                              "scale": meta.get("scale", 1.0), "sticker_id": sid, "prompt": meta.get("prompt", "")}
                elif action == "remove_sticker":
                    iid = gmsg.get("instance_id")
                    removed_sticker_id = None
                    for s in active_stickers:
                        if s["instance_id"] == iid:
                            removed_sticker_id = s.get("sticker_id")
                            break
                    active_stickers = [s for s in active_stickers if s["instance_id"] != iid]
                    adjustments.pop(iid, None)
                    if edit_target_id == iid:
                        edit_target_id = active_stickers[-1]["instance_id"] if active_stickers else None
                    if active_content and active_content.get("sticker_id") == removed_sticker_id:
                        active_content = None
                elif action == "select_edit_target":
                    iid = gmsg.get("instance_id")
                    if iid and any(s["instance_id"] == iid for s in active_stickers):
                        edit_target_id = iid
                    elif not iid:
                        edit_target_id = None
                elif action == "load_template":
                    t = gmsg.get("template")
                    if t and t.get("image") is not None:
                        if len(active_stickers) >= MAX_STICKERS:
                            print(f"[Consumer] 贴纸数量已达上限 ({MAX_STICKERS})，忽略添加请求")
                            continue
                        iid = _add_sticker_instance(t["image"], t["id"],
                            t.get("region", "forehead_top"), 1.0, t.get("name", "模板"))
                        edit_target_id = iid
                        active_content = {"sticker": t["image"], "location": t.get("region", "forehead_top"),
                                          "scale": 1.0, "sticker_id": t["id"], "prompt": t.get("name", "模板")}
                    else:
                        active_stickers.clear()
                        adjustments.clear()
                        edit_target_id = None
                        active_content = None
                elif action == "load_sticker":
                    sid = gmsg.get("sticker_id")
                    if sid:
                        if len(active_stickers) >= MAX_STICKERS:
                            print(f"[Consumer] 贴纸数量已达上限 ({MAX_STICKERS})，忽略添加请求")
                            continue
                        loaded, meta = storage.get_sticker(sid)
                        if loaded is not None and meta is not None:
                            iid = _add_sticker_instance(loaded, sid,
                                meta.get("region", "forehead_top"), meta.get("scale", 1.0),
                                meta.get("prompt", ""))
                            edit_target_id = iid
                            active_content = {"sticker": loaded, "location": meta.get("region", "forehead_top"),
                                              "scale": meta.get("scale", 1.0), "sticker_id": sid, "prompt": meta.get("prompt", "")}
                    else:
                        active_stickers.clear()
                        adjustments.clear()
                        edit_target_id = None
                        active_content = None
                elif action == "merge_group":
                    iids = set(gmsg.get("instance_ids", []))
                    merge_face = face_data if (face_data and "nose_tip" in face_data) else cached_face_data
                    if len(iids) >= 2 and merge_face:
                        to_merge = [s for s in active_stickers if s["instance_id"] in iids]
                        if len(to_merge) >= 2:
                            merged_img, merged_location, merged_scale, mrg_ox, mrg_oy = composite_stickers_to_merged(
                                to_merge, adjustments, merge_face)
                            if merged_img is not None:
                                prompts = [s.get("prompt", "") for s in to_merge if s.get("prompt")]
                                merged_prompt = " + ".join(prompts[:3])
                                sid = storage.save_sticker(merged_img, {
                                    "prompt": merged_prompt or "合并贴纸",
                                    "location": merged_location,
                                    "scale": merged_scale,
                                })
                                # Remove merged instances
                                for s in to_merge:
                                    iid = s["instance_id"]
                                    active_stickers = [x for x in active_stickers if x["instance_id"] != iid]
                                    adjustments.pop(iid, None)
                                # Add merged result
                                merged_instance_id = str(uuid.uuid4())
                                adjustments[merged_instance_id] = {
                                    "offset_x": mrg_ox, "offset_y": mrg_oy,
                                    "rotation": 0.0, "scale_mult": 1.0,
                                }
                                active_stickers.append({
                                    "instance_id": merged_instance_id,
                                    "sticker_id": sid,
                                    "sticker": merged_img,
                                    "location": merged_location,
                                    "scale": merged_scale,
                                    "prompt": merged_prompt,
                                })
                                edit_target_id = merged_instance_id
                                active_content = {"sticker": merged_img, "location": merged_location,
                                                  "scale": merged_scale, "sticker_id": sid, "prompt": merged_prompt}
                                display_queue.put({"action": "sticker_saved", "sticker_id": sid})

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
                        edit_target_id = None
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

                if ai_state["is_generating"] or edit_target_id is not None or face_draw_active:
                    frame = render_face_mesh(frame, face_data)

                # Render all active stickers in z-order (bottom to top)
                for instance in active_stickers:
                    if instance["sticker"] is None:
                        continue
                    adj = adjustments.get(instance["instance_id"], {
                        "offset_x": 0.0, "offset_y": 0.0,
                        "rotation": 0.0, "scale_mult": 1.0,
                    })
                    adj["edit_mode"] = (instance["instance_id"] == edit_target_id)
                    content = {
                        "sticker": instance["sticker"],
                        "location": instance["location"],
                        "scale": instance["scale"],
                    }
                    frame = render_scene(frame, face_data, content, adj)

                if face_draw_active and face_canvas.has_content:
                    draw_content = {
                        "sticker": face_canvas.canvas,
                        "location": face_draw_region,
                        "scale": 1.5,
                    }
                    frame = render_scene(frame, face_data, draw_content, None)

                # Sync active stickers state to UI
                instances_info = []
                for s in active_stickers:
                    instances_info.append({
                        "instance_id": s["instance_id"],
                        "sticker_id": s["sticker_id"],
                        "region": s["location"],
                    })
                display_queue.put({
                    "action": "active_stickers_changed",
                    "active_count": len(active_stickers),
                    "instances": instances_info,
                    "edit_target_id": edit_target_id,
                })

            try:
                display_queue.put(frame, timeout=0.05)
            except Exception:
                pass
    finally:
        detector.close()
        print("[Consumer] 已退出")
