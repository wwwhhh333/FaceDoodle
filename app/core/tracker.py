# 异步视频处理逻辑


import os
import cv2
import threading
import time
from multiprocessing import Queue, Process
from app.ai.agent import FaceDoodleAgent
from app.ai.generator import ComfyClient
from app.core.face_mesh import FaceDetector
from app.core.renderer import render_scene
from app.utils.image_proc import load_rgba_sticker


# 全局状态，用于协调视频流和AI任务
ai_state = {
    "is_generating": False,
    "current_sticker": None,
    "loading_sticker": None
}


def ai_worker_thread(user_command, result_queue, api_key):
    """
    后台线程：处理从指令解析到图像生成的全流程，不阻塞主视频流
    """
    global ai_state

    # 1. 初始化 Agent 和 ComfyUI 客户端
    agent = FaceDoodleAgent(api_key=api_key)
    comfy_client = ComfyClient(server_address="127.0.0.1:8188")

    try:
        # --- 步骤 1: LLM 意图解析 ---
        # 得到 {'positive_prompt': '...', 'target_location': 'eyes', 'workflow': '...'}
        print(f"[AI Worker] 正在解析指令: {user_command}")
        task_info = agent.parse_command(user_command)

        if not task_info or "positive_prompt" not in task_info:
            raise ValueError("Agent 解析指令失败")

        # --- 步骤 2: 调用 ComfyUI 生成透明贴纸 ---
        print(f"[AI Worker] 正在调用 ComfyUI 生成: {task_info['positive_prompt']}")

        # 传入 Prompt 和 工作流文件名
        image_path = comfy_client.generate_sync(
            prompt_text=task_info["positive_prompt"],
            workflow_name=task_info["workflow"]
        )

        # --- 步骤 3: 加载并交付结果 ---
        if image_path and os.path.exists(image_path):
            new_sticker = load_rgba_sticker(image_path)

            if new_sticker is not None:
                # 将图片数据和 Agent 决定的位置信息打包
                result_data = {
                    "sticker": new_sticker,
                    "location": task_info["target_location"],
                    "timestamp": time.time()
                }
                result_queue.put(result_data)
                print(f"[AI Worker] 任务完成，位置: {task_info['target_location']}")
            else:
                print("[AI Worker] 错误：无法将生成结果转换为 RGBA 格式")
        else:
            print("[AI Worker] 错误：ComfyUI 未生成有效文件")

    except Exception as e:
        print(f"[AI Worker] 发生异常: {str(e)}")

    finally:
        # 无论成功失败，必须重置状态，允许下一次生成
        ai_state["is_generating"] = False


def producer(queue):
    cap = cv2.VideoCapture(0)
    # 设置较高的分辨率以确保视觉效果
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 水平翻转画面，符合照镜子的直觉
        frame = cv2.flip(frame, 1)

        if not queue.full():
            queue.put(frame)
    cap.release()


def consumer(in_queue, display_queue, command_queue, api_key):
    detector = FaceDetector()
    result_queue = Queue()

    # 占位图
    loading_sticker = load_rgba_sticker("assets/static/loading.png")
    active_content = None  # 保存 {'sticker': img, 'location': '...'}

    while True:
        if not in_queue.empty():
            frame = in_queue.get()

            # 1. 检查是否有来自 UI 的新指令
            if not command_queue.empty():
                user_command = command_queue.get()
                if not ai_state["is_generating"]:
                    ai_state["is_generating"] = True
                    # 启动后台 AI 线程
                    threading.Thread(
                        target=ai_worker_thread,
                        args=(user_command, result_queue, api_key)
                    ).start()

            # 2. 检查是否有新生成的贴纸
            if not result_queue.empty():
                active_content = result_queue.get()

            # 3. 视觉追踪与 AR 渲染
            face_data = detector.get_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if face_data:
                if ai_state["is_generating"] and loading_sticker is not None:
                    # 正在生成时，在额头显示加载动画
                    temp_content = {'sticker': loading_sticker, 'location': 'forehead'}
                    frame = render_scene(frame, face_data, temp_content)
                elif active_content is not None:
                    # 渲染最终结果
                    frame = render_scene(frame, face_data, active_content)

            # 4. 将处理好的帧发送给 UI 进程
            # 如果 display_queue 满了，说明 UI 渲染跟不上，丢弃当前帧以保证实时性
            if not display_queue.full():
                display_queue.put(frame)


if __name__ == "__main__":
    frame_queue = Queue(maxsize=2)

    p1 = Process(target=producer, args=(frame_queue,))
    p2 = Process(target=consumer, args=(frame_queue,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()