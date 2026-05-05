import glob
import os
import cv2
import threading
import time
import queue
from multiprocessing import Queue, Process, Event
from app.ai.agent import FaceDoodleAgent
from app.ai.generator import ComfyClient
from app.core.face_mesh import FaceDetector
from app.core.renderer import render_scene
from app.utils.image_proc import load_rgba_sticker

ai_state = {
    "is_generating": False,
    "current_sticker": None,
    "loading_sticker": None
}


def ai_worker_thread(user_command, result_queue, api_key, mock=False):
    global ai_state

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
                "timestamp": time.time()
            }
            result_queue.put(result_data)
            print(f"[AI Worker] 任务完成，位置: {task_info['target_location']}")
        else:
            print("[AI Worker] 错误：无法将生成结果转换为 RGBA 格式")

    except Exception as e:
        print(f"[AI Worker] 发生异常: {str(e)}")

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


def consumer(in_queue, display_queue, command_queue, adjustment_queue, api_key, stop_event, mock=False):
    detector = FaceDetector()
    result_queue = queue.Queue()

    loading_sticker = load_rgba_sticker("assets/static/loading.png")
    active_content = None
    pending_command = None

    adjustment = {
        "offset_x": 0.0,
        "offset_y": 0.0,
        "rotation": 0.0,
        "scale_mult": 1.0,
        "edit_mode": False,
    }

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
                cmd = pending_command
                pending_command = None
                threading.Thread(
                    target=ai_worker_thread,
                    args=(cmd, result_queue, api_key, mock),
                    daemon=True
                ).start()

            try:
                new_content = result_queue.get(block=False)
                active_content = new_content
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
                    adjustment["scale_mult"] = max(0.2, min(5.0, adjustment["scale_mult"]))
                elif action == "reset":
                    adjustment["offset_x"] = 0.0
                    adjustment["offset_y"] = 0.0
                    adjustment["rotation"] = 0.0
                    adjustment["scale_mult"] = 1.0

            if face_data:
                if ai_state["is_generating"] and loading_sticker is not None:
                    frame = render_scene(frame, face_data,
                                         {'sticker': loading_sticker, 'location': 'forehead', 'scale': 1.0},
                                         adjustment)
                elif active_content is not None:
                    frame = render_scene(frame, face_data, active_content, adjustment)

            try:
                display_queue.put(frame, timeout=0.05)
            except Exception:
                pass
    finally:
        detector.close()
        print("[Consumer] 已退出")
