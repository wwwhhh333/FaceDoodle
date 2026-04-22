import sys
import os
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication
from app.core.tracker import producer, consumer
from app.ui.main_window import FaceDoodleWindow

MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY")

def main():
    # 1. 初始化进程间通信队列
    # frame_queue: 生产者 -> 消费者 (原始画面)
    frame_queue = Queue(maxsize=2)
    # display_queue: 消费者 -> UI (带特效的画面)
    display_queue = Queue(maxsize=2)
    # command_queue: UI -> 消费者 (用户文字指令)
    command_queue = Queue(maxsize=5)

    # 2. 启动视频流水线子进程
    print("[System] 正在启动视频流与 AI 消费者进程...")

    p_producer = Process(target=producer, args=(frame_queue,))
    p_consumer = Process(target=consumer, args=(
        frame_queue,
        display_queue,
        command_queue,
        MODELSCOPE_API_KEY
    ))

    p_producer.start()
    p_consumer.start()

    # 3. 启动 PyQt5 主界面 (运行在主进程)
    print("[System] 正在启动图形界面...")
    app = QApplication(sys.argv)
    window = FaceDoodleWindow(display_queue, command_queue)
    window.show()

    # 阻塞，直到关闭窗口
    exit_code = app.exec_()

    # 4. 安全退出：清理子进程
    print("[System] 正在关闭系统，清理子进程...")
    p_producer.terminate()
    p_consumer.terminate()
    p_producer.join()
    p_consumer.join()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()