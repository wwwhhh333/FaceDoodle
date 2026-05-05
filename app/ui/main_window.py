import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QShortcut, QMessageBox,
                             QFileDialog, QDialog, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QPainter, QColor
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QEvent

from app.ui.widgets import (ThumbnailCard, StyledButton, GradientBar,
                            GalleryScrollArea, DrawingDialog, _bgra_to_qpixmap)
from app.utils import storage


class VideoUpdateThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    sticker_saved_signal = pyqtSignal()

    def __init__(self, display_queue, gallery_queue):
        super().__init__()
        self.display_queue = display_queue
        self.gallery_queue = gallery_queue
        self._run_flag = True

    def run(self):
        while self._run_flag:
            try:
                item = self.display_queue.get(block=True, timeout=0.1)
                if isinstance(item, np.ndarray):
                    self.change_pixmap_signal.emit(item)
                elif isinstance(item, dict) and item.get("action") == "sticker_saved":
                    self.sticker_saved_signal.emit()
            except Exception:
                pass

    def stop(self):
        self._run_flag = False
        self.wait()


class FaceDoodleWindow(QMainWindow):
    def __init__(self, display_queue, command_queue, adjustment_queue, gallery_queue):
        super().__init__()
        self.display_queue = display_queue
        self.command_queue = command_queue
        self.adjustment_queue = adjustment_queue
        self.gallery_queue = gallery_queue

        self._edit_mode = False
        self._mouse_down = False
        self._mouse_button = None
        self._last_mouse_pos = None
        self._frame_size = None
        self._current_sticker_id = None
        self._gallery_items = {}

        self.setWindowTitle("FaceDoodle AI - 智能贴纸工坊")
        self.resize(1440, 860)

        self._init_stylesheet()
        self._init_ui()
        self._load_gallery()

        self.video_thread = VideoUpdateThread(self.display_queue, self.gallery_queue)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.sticker_saved_signal.connect(self._on_sticker_saved)
        self.video_thread.start()

    def _init_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background: #f5f5f8; }
            QWidget { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }
            QLineEdit {
                background: #fff;
                color: #333;
                border: 2px solid #ddd;
                padding: 12px 18px;
                font-size: 16px;
            }
            QLineEdit:focus { border-color: #667eea; }
            QLineEdit:disabled { background: #f0f0f0; color: #bbb; }
            QLabel { color: #444; }
            QMessageBox { background: #fff; }
            QMessageBox QLabel { color: #333; }
            QMessageBox QPushButton {
                background: #667eea; color: white;
                padding: 8px 24px; font-size: 14px;
            }
        """)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 1. 顶栏 ──
        top_bar = GradientBar("FaceDoodle AI 贴纸工坊")
        settings_btn = QPushButton("⚙️")
        settings_btn.setFixedSize(40, 40)
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.15); color: white; border: none;
                font-size: 20px;
            }
            QPushButton:hover { background: rgba(255,255,255,0.3); }
        """)
        settings_btn.clicked.connect(self._show_settings)
        top_bar.add_right_widget(settings_btn)
        root.addWidget(top_bar)

        # ── 2. 中间区域: 视频 + 右侧贴纸库 ──
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #e8e8ee; border: none;")
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setMouseTracking(True)
        self.video_label.installEventFilter(self)
        content_row.addWidget(self.video_label, stretch=1)

        # 右侧面板 — 贴纸库
        right_panel = QWidget()
        right_panel.setFixedWidth(210)
        right_panel.setStyleSheet("background: #fafafa; border: none;")
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(8, 12, 8, 8)
        rp_layout.setSpacing(6)

        gallery_label = QLabel("我的贴纸")
        gallery_label.setStyleSheet(
            "color: #555; font-size: 15px; font-weight: bold; background: transparent; border: none;"
        )
        rp_layout.addWidget(gallery_label)

        self.gallery = GalleryScrollArea()
        rp_layout.addWidget(self.gallery, stretch=1)

        self.sticker_count_label = QLabel("")
        self.sticker_count_label.setStyleSheet(
            "color: #888; font-size: 11px; background: transparent; border: none;"
        )
        rp_layout.addWidget(self.sticker_count_label)

        content_row.addWidget(right_panel)
        root.addLayout(content_row, stretch=1)

        self.edit_indicator = QLabel("编辑模式: 开启", self)
        self.edit_indicator.setStyleSheet(
            "color: #059669; background: rgba(255,255,255,230); font-size: 14px; "
            "padding: 6px 14px; border: 1px solid #10b981;"
        )
        self.edit_indicator.setAlignment(Qt.AlignCenter)
        self.edit_indicator.setVisible(False)

        # ── 3. 输入区 ──
        input_row = QWidget()
        input_row.setStyleSheet("background: #fff; border-top: 1px solid #eee;")
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(16, 12, 16, 12)
        input_layout.setSpacing(12)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("描述你的创意，例如：一副赛博朋克风格的护目镜...")
        self.input_box.returnPressed.connect(self.send_command)
        input_layout.addWidget(self.input_box)

        self.send_btn = StyledButton("生成贴纸", "#2c2c2c", "#444")
        self.send_btn.clicked.connect(self.send_command)
        input_layout.addWidget(self.send_btn)

        root.addWidget(input_row)

        # ── 4. 操作按钮区 ──
        action_row = QWidget()
        action_row.setStyleSheet("background: #fff; border-top: 1px solid #eee;")
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(16, 8, 16, 10)
        action_layout.setSpacing(10)

        self.edit_btn = StyledButton("编辑", "#7c3aed", "#a855f7")
        self.edit_btn.setCheckable(True)
        self.edit_btn.clicked.connect(self._toggle_edit_mode)
        action_layout.addWidget(self.edit_btn)

        self.reset_btn = StyledButton("重置位置", "#94a3b8", "#b0bec5")
        self.reset_btn.clicked.connect(lambda: self.adjustment_queue.put({"action": "reset"}))
        action_layout.addWidget(self.reset_btn)

        self.fav_btn = StyledButton("收藏", "#f59e0b", "#fbbf24")
        self.fav_btn.clicked.connect(self._toggle_favorite)
        action_layout.addWidget(self.fav_btn)

        self.del_btn = StyledButton("删除", "#ef4444", "#f87171")
        self.del_btn.clicked.connect(self._delete_current_sticker)
        action_layout.addWidget(self.del_btn)

        action_layout.addStretch()

        self.import_btn = StyledButton("导入", "#06b6d4", "#22d3ee")
        self.import_btn.clicked.connect(self._import_image)
        action_layout.addWidget(self.import_btn)

        self.draw_btn = StyledButton("绘制", "#10b981", "#34d399")
        self.draw_btn.clicked.connect(self._open_drawing_dialog)
        action_layout.addWidget(self.draw_btn)

        self.edit_sticker_btn = StyledButton("编辑贴纸", "#6366f1", "#818cf8")
        self.edit_sticker_btn.clicked.connect(self._open_edit_sticker_dialog)
        action_layout.addWidget(self.edit_sticker_btn)

        root.addWidget(action_row)

        # ── 5. 状态栏 ──
        self.status_label = QLabel("Ctrl+E 切换编辑 | 左键移动 右键旋转 滚轮缩放 双击重置")
        self.status_label.setStyleSheet(
            "color: #999; font-size: 12px; padding: 5px; background: #f0f0f5; border: none;"
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status_label)

        # 快捷键
        self.edit_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        self.edit_shortcut.activated.connect(self._toggle_edit_mode)

    # ── 画廊管理 ──

    def _load_gallery(self):
        self.gallery.clear_cards()
        self._gallery_items.clear()
        stickers = storage.load_gallery()
        for s in stickers:
            thumb = storage.get_sticker_thumb(s["id"])
            card = ThumbnailCard(s["id"], thumb, s.get("prompt", ""), s.get("favorite", False))
            card.clicked.connect(self._on_gallery_click)
            self.gallery.add_card(card)
            self._gallery_items[s["id"]] = card
        self._update_gallery_info()

    def _update_gallery_info(self):
        n = len(self._gallery_items)
        self.sticker_count_label.setText(f"共 {n} 枚贴纸")

    def _on_gallery_click(self, sticker_id):
        self._current_sticker_id = sticker_id
        for sid, card in self._gallery_items.items():
            card.set_selected(sid == sticker_id)
        self.gallery_queue.put({"action": "load_sticker", "sticker_id": sticker_id})

    def _on_sticker_saved(self):
        self._load_gallery()

    def _toggle_favorite(self):
        if not self._current_sticker_id:
            return
        stickers = storage.load_gallery()
        entry = next((s for s in stickers if s["id"] == self._current_sticker_id), None)
        if entry is None:
            return
        new_fav = not entry.get("favorite", False)
        storage.set_favorite(self._current_sticker_id, new_fav)
        self._load_gallery()

    def _delete_current_sticker(self):
        if not self._current_sticker_id:
            return
        reply = QMessageBox.question(
            self, "确认删除", "确定要删除这枚贴纸吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            storage.delete_sticker(self._current_sticker_id)
            self._current_sticker_id = None
            self.gallery_queue.put({"action": "load_sticker", "sticker_id": None})
            self._load_gallery()

    def _import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp);;所有文件 (*.*)"
        )
        if not path:
            return
        from app.utils.image_proc import load_rgba_sticker
        sticker = load_rgba_sticker(path)
        if sticker is None:
            QMessageBox.warning(self, "导入失败", "无法读取该图片，请确认格式正确。")
            return

        from app.ui.widgets import REGION_OPTIONS
        region, ok = self._choose_region_dialog()
        if not ok:
            return

        sid = storage.save_sticker(sticker, {
            "prompt": f"导入: {path.split('/')[-1].split(chr(92))[-1]}",
            "location": region,
            "scale": 1.0,
        })
        self.gallery_queue.put({"action": "load_sticker", "sticker_id": sid})
        self._load_gallery()

    def _choose_region_dialog(self):
        from app.ui.widgets import REGION_OPTIONS
        dlg = QDialog(self)
        dlg.setWindowTitle("选择位置")
        dlg.setFixedSize(280, 120)
        dlg.setStyleSheet("QDialog { background: #f5f5f8; }")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("请选择贴纸应用位置:"))
        combo = QComboBox()
        for label, value in REGION_OPTIONS:
            combo.addItem(label, value)
        layout.addWidget(combo)
        btn_layout = QHBoxLayout()
        ok_btn = StyledButton("确定", "#667eea", "#818cf8")
        cancel_btn = StyledButton("取消", "#94a3b8", "#b0bec5")
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        if dlg.exec_() == QDialog.Accepted:
            return combo.currentData(), True
        return None, False

    def _open_drawing_dialog(self):
        dlg = DrawingDialog(self, self.gallery_queue, self.command_queue)
        if dlg.exec_() == DrawingDialog.Accepted:
            self._load_gallery()

    def _open_edit_sticker_dialog(self):
        if not self._current_sticker_id:
            QMessageBox.information(self, "提示", "请先在画廊中选择一枚贴纸，或先 AI 生成一枚贴纸。")
            return
        sticker_img, _ = storage.get_sticker(self._current_sticker_id)
        if sticker_img is None:
            QMessageBox.warning(self, "错误", "无法加载该贴纸，文件可能已被删除。")
            return
        dlg = DrawingDialog(self, self.gallery_queue, self.command_queue, initial_sticker=sticker_img)
        if dlg.exec_() == DrawingDialog.Accepted:
            self._load_gallery()

    def _show_settings(self):
        from app.utils.config_loader import get_config
        cfg = get_config()
        prefs = cfg.get("preferences", {})
        recent = prefs.get("recent_prompts", [])
        info = (
            f"ComfyUI 地址: {cfg['comfyui']['server_address']}\n"
            f"LoRA: {cfg['model']['lora']['name']}\n"
            f"AI 模型: {cfg['agent']['model_id']}\n"
            f"最近指令: {', '.join(recent[:5]) if recent else '无'}"
        )
        QMessageBox.information(self, "设置", info)

    # ── 编辑模式 ──

    def _toggle_edit_mode(self):
        self._edit_mode = not self._edit_mode
        self.adjustment_queue.put({"action": "toggle_edit"})
        self.edit_btn.setChecked(self._edit_mode)
        self.edit_btn.setText("编辑中" if self._edit_mode else "编辑")
        if self._edit_mode:
            self.edit_indicator.setVisible(True)
            self._position_indicator()
            self.input_box.setEnabled(False)
            self.send_btn.setEnabled(False)
        else:
            self.edit_indicator.setVisible(False)
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)
            self._mouse_down = False

    def _position_indicator(self):
        label_pos = self.video_label.pos()
        margin = 12
        self.edit_indicator.move(label_pos.x() + margin, label_pos.y() + margin)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_indicator()

    def _label_to_frame_delta(self, dx, dy):
        if self._frame_size is None:
            return dx, dy
        fw, fh = self._frame_size
        lw = self.video_label.width()
        lh = self.video_label.height()
        if lw <= 0 or lh <= 0:
            return dx, dy
        return dx * fw / lw, dy * fh / lh

    # ── 鼠标事件 ──

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

    # ── 命令发送 ──

    def send_command(self):
        text = self.input_box.text().strip()
        if not text:
            return
        self.command_queue.put(text)
        storage.add_recent_prompt(text)
        self.input_box.clear()
        self.input_box.setPlaceholderText("AI 正在生成中，请稍候...")
        self.input_box.setEnabled(False)
        self.send_btn.setEnabled(False)
        QTimer.singleShot(5000, self._reenable_input)

    def _reenable_input(self):
        if not self._edit_mode:
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)
        self.input_box.setPlaceholderText("描述你的创意，例如：一副赛博朋克风格的护目镜...")

    # ── 视频显示 ──

    def update_image(self, cv_img):
        h, w, ch = cv_img.shape
        self._frame_size = (w, h)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(),
                          Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        from app.utils.storage import save_preferences
        save_preferences({
            "window_width": self.width(),
            "window_height": self.height(),
        })
        self.video_thread.stop()
        event.accept()
