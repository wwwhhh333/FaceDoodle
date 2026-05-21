import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Process, Queue, Event
from PySide6.QtWidgets import QApplication
from app.core.tracker import producer, consumer
from app.ui.main_window import FaceDoodleWindow
from app.utils.config_loader import load_config, is_first_run
from app.utils.storage import load_preferences
from app.utils.logging_config import setup_logging

import logging
log = logging.getLogger(__name__)

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


def _show_first_run_setup(current_key):
    """Show first-run dialog for API key and ComfyUI setup. Returns (possibly updated) api_key."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout
    from PySide6.QtCore import Qt

    dlg = QDialog()
    dlg.setWindowTitle("欢迎使用 FaceDoodle — 首次设置")
    dlg.setMinimumWidth(460)
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(24, 20, 24, 20)
    layout.setSpacing(14)

    title = QLabel("欢迎使用 FaceDoodle AI 贴纸工坊")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title.setStyleSheet("font-size: 16px; font-weight: 700; color: #333;")
    layout.addWidget(title)

    desc = QLabel("首次运行需要配置 API Key 和 ComfyUI 地址。\n这些设置之后也可以在应用内修改。")
    desc.setWordWrap(True)
    desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
    desc.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 8px;")
    layout.addWidget(desc)

    form = QFormLayout()
    form.setSpacing(8)

    key_edit = QLineEdit(current_key or "")
    key_edit.setPlaceholderText("sk-...")
    key_edit.setEchoMode(QLineEdit.EchoMode.Password)
    form.addRow("DeepSeek API Key", key_edit)

    addr_edit = QLineEdit("127.0.0.1:8188")
    addr_edit.setPlaceholderText("127.0.0.1:8188")
    form.addRow("ComfyUI 地址", addr_edit)
    layout.addLayout(form)

    note = QLabel("也可以将 Key 写入项目根目录的 api_key.txt 文件（程序只读不写）\n"
                  "或设置环境变量 DEEPSEEK_API_KEY。\n"
                  "使用 --mock 参数可跳过 ComfyUI 以测试界面。")
    note.setWordWrap(True)
    note.setStyleSheet("color: #aaa; font-size: 11px;")
    layout.addWidget(note)

    btn_row = QHBoxLayout()
    btn_row.addStretch()

    skip_btn = QPushButton("跳过")
    skip_btn.setFixedWidth(80)
    skip_btn.clicked.connect(dlg.reject)
    btn_row.addWidget(skip_btn)

    save_btn = QPushButton("保存并开始")
    save_btn.setFixedWidth(110)
    save_btn.setStyleSheet(
        "QPushButton { background: #4F46E5; color: #fff; border: none; "
        "border-radius: 6px; padding: 8px 16px; font-weight: 600; }"
        "QPushButton:hover { background: #4338CA; }"
    )
    btn_row.addWidget(save_btn)
    layout.addLayout(btn_row)

    def on_save():
        key = key_edit.text().strip()
        if key:
            key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_key.txt")
            try:
                with open(key_file, "w", encoding="utf-8") as f:
                    f.write(key)
                log.info("API Key 已保存到 %s", key_file)
            except OSError:
                pass
        from app.utils.config_loader import get_config, save_config
        cfg = get_config()
        addr = addr_edit.text().strip()
        if addr:
            cfg["comfyui"]["server_address"] = addr
            save_config(cfg)
        dlg.accept()

    save_btn.clicked.connect(on_save)
    key_edit.returnPressed.connect(on_save)

    result = dlg.exec()
    if result == QDialog.Accepted:
        return _resolve_api_key(get_config())
    return current_key


def _resolve_api_key(config):
    key = os.getenv("DEEPSEEK_API_KEY")
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
    setup_logging(verbose="--verbose" in sys.argv)
    from app.ai.generator import cleanup_temp_files
    cleanup_temp_files()

    config = load_config()
    api_key = _resolve_api_key(config)
    queue_cfg = config.get("queue", {})
    prefs = load_preferences()
    video_path = _get_video_path(config)

    if video_path:
        log.info("视频文件模式: %s", video_path)

    # Must create QApplication before any QWidget
    app = QApplication(sys.argv)

    # First-run setup
    if is_first_run() or not api_key:
        api_key = _show_first_run_setup(api_key)

    frame_queue = Queue(maxsize=queue_cfg.get("frame_maxsize", 5))
    display_queue = Queue(maxsize=queue_cfg.get("display_maxsize", 5))
    command_queue = Queue(maxsize=queue_cfg.get("command_maxsize", 5))
    adjustment_queue = Queue(maxsize=20)
    gallery_queue = Queue(maxsize=10)
    draw_queue = Queue(maxsize=50)
    animation_queue = Queue(maxsize=20)
    stop_event = Event()

    log.info("正在启动视频流与 AI 消费者进程...%s", ' (Mock 模式: 跳过 ComfyUI)' if MOCK_MODE else '')

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

    from app.ai.comfy_manager import ComfyUIManager
    comfy_cfg = config.get("comfyui", {})
    comfy_mgr = ComfyUIManager(
        install_path=comfy_cfg.get("install_path", ""),
        server_address=comfy_cfg.get("server_address", "127.0.0.1:8188"),
    )
    if not MOCK_MODE:
        comfy_mgr.start()

    log.info("正在启动图形界面...")
    window = FaceDoodleWindow(display_queue, command_queue, adjustment_queue, gallery_queue, draw_queue, animation_queue)

    w = prefs.get("window_width", 1280)
    h = prefs.get("window_height", 800)
    window.resize(w, h)
    window.show()

    exit_code = app.exec()

    log.info("正在关闭系统，清理子进程...")
    stop_event.set()

    p_producer.join(timeout=5)
    p_consumer.join(timeout=5)

    if p_producer.is_alive():
        p_producer.terminate()
    if p_consumer.is_alive():
        p_consumer.terminate()

    comfy_mgr.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
