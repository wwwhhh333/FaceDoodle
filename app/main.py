import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Process, Queue, Event
from PyQt5.QtWidgets import QApplication
from app.core.tracker import producer, consumer
from app.ui.main_window import FaceDoodleWindow
from app.utils.config_loader import load_config

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("MODELSCOPE_API_KEY")
MOCK_MODE = "--mock" in sys.argv


def main():
    config = load_config()
    queue_cfg = config.get("queue", {})

    frame_queue = Queue(maxsize=queue_cfg.get("frame_maxsize", 5))
    display_queue = Queue(maxsize=queue_cfg.get("display_maxsize", 5))
    command_queue = Queue(maxsize=queue_cfg.get("command_maxsize", 5))
    adjustment_queue = Queue(maxsize=20)
    stop_event = Event()

    print(f"[System] 正在启动视频流与 AI 消费者进程... {'(Mock 模式: 跳过 ComfyUI)' if MOCK_MODE else ''}")

    p_producer = Process(target=producer, args=(frame_queue, stop_event))
    p_consumer = Process(target=consumer, args=(
        frame_queue,
        display_queue,
        command_queue,
        adjustment_queue,
        API_KEY,
        stop_event,
        MOCK_MODE,
    ))

    p_producer.start()
    p_consumer.start()

    print("[System] 正在启动图形界面...")
    app = QApplication(sys.argv)
    window = FaceDoodleWindow(display_queue, command_queue, adjustment_queue)
    window.show()

    exit_code = app.exec_()

    print("[System] 正在关闭系统，清理子进程...")
    stop_event.set()

    p_producer.join(timeout=5)
    p_consumer.join(timeout=5)

    if p_producer.is_alive():
        p_producer.terminate()
    if p_consumer.is_alive():
        p_consumer.terminate()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
