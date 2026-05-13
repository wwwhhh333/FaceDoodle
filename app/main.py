import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Process, Queue, Event
from PyQt5.QtWidgets import QApplication
from app.core.tracker import producer, consumer
from app.ui.main_window import FaceDoodleWindow
from app.utils.config_loader import load_config
from app.utils.storage import load_preferences

MOCK_MODE = "--mock" in sys.argv


def _get_video_path(config):
    """Resolve video path from --video CLI arg or config default."""
    for i, arg in enumerate(sys.argv):
        if arg == "--video":
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                return sys.argv[i + 1]
            # No explicit path — fall back to config default
            return config["video"]["path"] or None
    return None


def _resolve_api_key(config):
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("MODELSCOPE_API_KEY")
    if key:
        return key
    key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_key.txt")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return config.get("api_key", "")


def main():
    config = load_config()
    api_key = _resolve_api_key(config)
    queue_cfg = config.get("queue", {})
    prefs = load_preferences()
    video_path = _get_video_path(config)

    if video_path:
        print(f"[System] 视频文件模式: {video_path}")

    frame_queue = Queue(maxsize=queue_cfg.get("frame_maxsize", 5))
    display_queue = Queue(maxsize=queue_cfg.get("display_maxsize", 5))
    command_queue = Queue(maxsize=queue_cfg.get("command_maxsize", 5))
    adjustment_queue = Queue(maxsize=20)
    gallery_queue = Queue(maxsize=10)
    draw_queue = Queue(maxsize=50)
    animation_queue = Queue(maxsize=20)
    stop_event = Event()

    print(f"[System] 正在启动视频流与 AI 消费者进程... {'(Mock 模式: 跳过 ComfyUI)' if MOCK_MODE else ''}")

    p_producer = Process(target=producer, args=(frame_queue, stop_event, video_path))
    p_consumer = Process(target=consumer, args=(
        frame_queue,
        display_queue,
        command_queue,
        adjustment_queue,
        gallery_queue,
        draw_queue,
        animation_queue,
        api_key,
        stop_event,
        MOCK_MODE,
    ))

    p_producer.start()
    p_consumer.start()

    print("[System] 正在启动图形界面...")
    app = QApplication(sys.argv)
    window = FaceDoodleWindow(display_queue, command_queue, adjustment_queue, gallery_queue, draw_queue, animation_queue)

    w = prefs.get("window_width", 1280)
    h = prefs.get("window_height", 800)
    window.resize(w, h)
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
