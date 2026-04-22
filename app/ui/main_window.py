# PyQt5 界面主程序


import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class VideoUpdateThread(QThread):
    """
    专门用于从 display_queue 读取画面并发送给 UI 的线程
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, display_queue):
        super().__init__()
        self.display_queue = display_queue
        self._run_flag = True

    def run(self):
        while self._run_flag:
            if not self.display_queue.empty():
                cv_img = self.display_queue.get()
                self.change_pixmap_signal.emit(cv_img)

    def stop(self):
        self._run_flag = False
        self.wait()


class FaceDoodleWindow(QMainWindow):
    def __init__(self, display_queue, command_queue):
        super().__init__()
        self.display_queue = display_queue
        self.command_queue = command_queue

        self.setWindowTitle("FaceDoodle AI - 智能人脸涂鸦系统")
        self.resize(1280, 800)
        self.init_ui()

        # 启动视频更新线程
        self.video_thread = VideoUpdateThread(self.display_queue)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 1. 视频显示区域
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(800, 600)
        layout.addWidget(self.video_label, stretch=1)

        # 2. 交互控制区
        control_layout = QHBoxLayout()

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("请输入你的创意，例如：给我画一个搞怪的海盗眼罩...")
        self.input_box.setStyleSheet("font-size: 16px; padding: 10px;")
        # 支持回车发送
        self.input_box.returnPressed.connect(self.send_command)
        control_layout.addWidget(self.input_box)

        self.send_btn = QPushButton("生成贴纸 ✨", self)
        self.send_btn.setStyleSheet("font-size: 16px; padding: 10px; font-weight: bold;")
        self.send_btn.clicked.connect(self.send_command)
        control_layout.addWidget(self.send_btn)

        layout.addLayout(control_layout)

    def send_command(self):
        text = self.input_box.text().strip()
        if text:
            # 将用户指令放入队列，传给 Consumer 进程
            self.command_queue.put(text)
            self.input_box.clear()
            self.input_box.setPlaceholderText(f"已发送指令: {text} (请稍候...)")

    def update_image(self, cv_img):
        """将 OpenCV 的 BGR 图像转换为 PyQt 的 QPixmap 并显示"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        # 转换为 RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 按比例缩放以适应 Label 大小
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()