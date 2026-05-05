# PyQt5 界面主程序


import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QShortcut)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QEvent


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
            try:
                cv_img = self.display_queue.get(block=True, timeout=0.1)
                self.change_pixmap_signal.emit(cv_img)
            except Exception:
                pass

    def stop(self):
        self._run_flag = False
        self.wait()


class FaceDoodleWindow(QMainWindow):
    def __init__(self, display_queue, command_queue, adjustment_queue):
        super().__init__()
        self.display_queue = display_queue
        self.command_queue = command_queue
        self.adjustment_queue = adjustment_queue

        self._edit_mode = False
        self._mouse_down = False
        self._mouse_button = None
        self._last_mouse_pos = None
        self._frame_size = None

        self.setWindowTitle("FaceDoodle AI - 智能人脸涂鸦系统")
        self.resize(1280, 800)
        self.init_ui()

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
        self.video_label.setMouseTracking(True)
        self.video_label.installEventFilter(self)
        layout.addWidget(self.video_label, stretch=1)

        # 编辑模式指示器
        self.edit_indicator = QLabel("编辑模式: 关闭", self)
        self.edit_indicator.setStyleSheet(
            "color: white; background: rgba(0,0,0,150); font-size: 14px; padding: 4px 10px; border-radius: 4px;"
        )
        self.edit_indicator.setVisible(False)
        self.edit_indicator.setAlignment(Qt.AlignCenter)

        # 2. 交互控制区
        control_layout = QHBoxLayout()

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("请输入你的创意，例如：给我画一个搞怪的海盗眼罩...")
        self.input_box.setStyleSheet("font-size: 16px; padding: 10px;")
        self.input_box.returnPressed.connect(self.send_command)
        control_layout.addWidget(self.input_box)

        self.send_btn = QPushButton("生成贴纸", self)
        self.send_btn.setStyleSheet("font-size: 16px; padding: 10px; font-weight: bold;")
        self.send_btn.clicked.connect(self.send_command)
        control_layout.addWidget(self.send_btn)

        self.edit_btn = QPushButton("编辑", self)
        self.edit_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.edit_btn.setCheckable(True)
        self.edit_btn.clicked.connect(self._toggle_edit_mode)
        control_layout.addWidget(self.edit_btn)

        layout.addLayout(control_layout)

        # Ctrl+E 快捷键
        self.edit_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        self.edit_shortcut.activated.connect(self._toggle_edit_mode)

        # 状态提示标签
        self.status_label = QLabel("Ctrl+E 切换编辑 | 左键移动 右键旋转 滚轮缩放 双击重置", self)
        self.status_label.setStyleSheet("color: gray; font-size: 12px; padding: 2px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def _toggle_edit_mode(self):
        self._edit_mode = not self._edit_mode
        self.adjustment_queue.put({"action": "toggle_edit"})
        self.edit_btn.setChecked(self._edit_mode)
        self.edit_btn.setText("编辑中" if self._edit_mode else "编辑")
        self.edit_indicator.setText(
            f"编辑模式: {'开启' if self._edit_mode else '关闭'}"
        )
        if self._edit_mode:
            self.edit_indicator.setStyleSheet(
                "color: #00ff00; background: rgba(0,0,0,180); font-size: 14px; padding: 4px 10px; border-radius: 4px;"
            )
            self.edit_indicator.setVisible(True)
            self._position_indicator()
            self.input_box.setEnabled(False)
            self.send_btn.setEnabled(False)
        else:
            self.edit_indicator.setStyleSheet(
                "color: white; background: rgba(0,0,0,150); font-size: 14px; padding: 4px 10px; border-radius: 4px;"
            )
            QTimer.singleShot(1000, lambda: self.edit_indicator.setVisible(False))
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)
            self._mouse_down = False

    def _position_indicator(self):
        """将编辑指示器放在视频区域左上角"""
        label_pos = self.video_label.pos()
        margin = 12
        self.edit_indicator.move(label_pos.x() + margin, label_pos.y() + margin)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_indicator()

    def _label_to_frame_delta(self, dx, dy):
        """将标签坐标系的像素增量转换为帧坐标系的像素增量"""
        if self._frame_size is None:
            return dx, dy
        fw, fh = self._frame_size
        lw = self.video_label.width()
        lh = self.video_label.height()
        if lw <= 0 or lh <= 0:
            return dx, dy
        scale_x = fw / lw
        scale_y = fh / lh
        return dx * scale_x, dy * scale_y

    # ---- 鼠标事件（在 video_label 上通过 eventFilter 捕获）----

    def eventFilter(self, obj, event):
        if obj is self.video_label and self._edit_mode:
            t = event.type()
            if t == QEvent.MouseButtonPress:
                self._on_mouse_press(event)
                return True
            elif t == QEvent.MouseMove:
                self._on_mouse_move(event)
                return True
            elif t == QEvent.MouseButtonRelease:
                self._on_mouse_release(event)
                return True
            elif t == QEvent.Wheel:
                self._on_wheel(event)
                return True
            elif t == QEvent.MouseButtonDblClick:
                self._on_double_click(event)
                return True
        return super().eventFilter(obj, event)

    def _on_mouse_press(self, event):
        self._mouse_down = True
        self._mouse_button = event.button()
        self._last_mouse_pos = event.pos()

    def _on_mouse_move(self, event):
        if not self._mouse_down or self._last_mouse_pos is None:
            return
        dx_px = event.pos().x() - self._last_mouse_pos.x()
        dy_px = event.pos().y() - self._last_mouse_pos.y()
        self._last_mouse_pos = event.pos()

        dx_f, dy_f = self._label_to_frame_delta(dx_px, dy_px)

        if self._mouse_button == Qt.LeftButton:
            self.adjustment_queue.put({"action": "move", "dx": dx_f, "dy": dy_f})
        elif self._mouse_button == Qt.RightButton:
            self.adjustment_queue.put({"action": "rotate", "d_angle": dx_f * 0.3})

    def _on_mouse_release(self, event):
        self._mouse_down = False
        self._mouse_button = None
        self._last_mouse_pos = None

    def _on_wheel(self, event):
        delta = event.angleDelta().y()
        factor = 1.0 + delta * 0.0005
        self.adjustment_queue.put({"action": "scale", "multiplier": factor})

    def _on_double_click(self, event):
        self.adjustment_queue.put({"action": "reset"})

    # ---- 命令发送 ----

    def send_command(self):
        text = self.input_box.text().strip()
        if text:
            self.command_queue.put(text)
            self.input_box.clear()
            self.input_box.setPlaceholderText("生成中，请稍候...")
            self.input_box.setEnabled(False)
            self.send_btn.setEnabled(False)
            QTimer.singleShot(5000, self._reenable_input)

    def _reenable_input(self):
        if not self._edit_mode:
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)
        self.input_box.setPlaceholderText("请输入你的创意，例如：给我画一个搞怪的海盗眼罩...")

    def update_image(self, cv_img):
        """将 OpenCV 的 BGR 图像转换为 PyQt 的 QPixmap 并显示"""
        h, w, ch = cv_img.shape
        self._frame_size = (w, h)
        bytes_per_line = ch * w
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()
