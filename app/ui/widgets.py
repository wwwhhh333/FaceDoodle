from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton, QScrollArea,
                             QHBoxLayout, QVBoxLayout, QSlider, QColorDialog,
                             QComboBox, QDialog, QGridLayout)
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QColor,
                         QPen, QMouseEvent)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QTimer
import numpy as np
import cv2
import os
import subprocess


def _bgra_to_qpixmap(bgra):
    if bgra is None:
        return None
    h, w = bgra.shape[:2]
    if bgra.shape[2] == 4:
        fmt = QImage.Format_RGBA8888
        rgb = bgra[:, :, [2, 1, 0, 3]]
    else:
        fmt = QImage.Format_RGB888
        rgb = bgra[:, :, [2, 1, 0]]
    qimg = QImage(rgb.data.tobytes(), w, h, rgb.shape[2] * w, fmt)
    return QPixmap.fromImage(qimg)


class ThumbnailCard(QWidget):
    clicked = pyqtSignal(str)  # sticker_id

    def __init__(self, sticker_id, thumb_bgra, prompt, is_favorite=False, parent=None):
        super().__init__(parent)
        self.sticker_id = sticker_id
        self._selected = False
        self._active = False
        self._fav = is_favorite
        self.setFixedSize(100, 120)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(96, 96)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet(
            "background: #f0f0f5; border: 2px solid #ddd;"
        )
        if thumb_bgra is not None:
            pix = _bgra_to_qpixmap(thumb_bgra)
            if pix is not None:
                self.thumb_label.setPixmap(
                    pix.scaled(92, 92, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        layout.addWidget(self.thumb_label, alignment=Qt.AlignCenter)

        label_text = prompt[:10] + ".." if len(prompt) > 10 else prompt
        self.name_label = QLabel(label_text)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet(
            "color: #666; font-size: 11px; background: transparent; border: none;"
        )
        layout.addWidget(self.name_label)

    def set_selected(self, sel):
        self._selected = sel
        self._update_border()

    def set_active(self, act):
        self._active = act
        self._update_border()

    def _update_border(self):
        if self._selected:
            color = "#333"
        elif self._active:
            color = "#10b981"
        else:
            color = "#ddd"
        self.thumb_label.setStyleSheet(
            f"background: #f0f0f5; border: 2px solid {color};"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.sticker_id)


class StyledButton(QPushButton):
    def __init__(self, text, color="#7c3aed", hover="#a855f7", text_color="white", parent=None):
        super().__init__(text, parent)
        self._color = color
        self._hover = hover
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: {text_color};
                border: none;
                padding: 14px 28px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
            QPushButton:pressed {{
                background: {color};
            }}
            QPushButton:disabled {{
                background: #ddd;
                color: #999;
            }}
        """)


class GradientBar(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet("background: #2c2c2c;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 16, 0)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            "color: white; font-size: 20px; font-weight: bold; background: transparent; border: none;"
        )
        layout.addWidget(self.title_label)

        self.right_widget = None
        layout.addStretch()

    def add_right_widget(self, widget):
        self.right_widget = widget
        layout = self.layout()
        if layout:
            layout.addWidget(widget)


class GalleryScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(210)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #f5f5f5; width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #ccc; min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(8, 4, 8, 4)
        self.layout.setSpacing(6)

        self._placeholder = QLabel("还没有贴纸\n输入描述来生成吧")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "color: #bbb; font-size: 13px; background: transparent; border: none; padding: 40px 12px;"
        )
        self._placeholder.setVisible(False)
        self.layout.addWidget(self._placeholder)
        self.layout.addStretch()
        self.setWidget(self.container)

    def show_placeholder(self, show):
        self._placeholder.setVisible(show)

    def add_card(self, card):
        self.layout.insertWidget(self.layout.count() - 1, card)

    def clear_cards(self):
        for i in range(self.layout.count() - 1, -1, -1):
            w = self.layout.itemAt(i).widget()
            if w is not None and w is not self._placeholder:
                w.deleteLater()


# ── 手绘贴纸组件 ──

PRESET_COLORS = [
    ("黑", (0, 0, 0, 255)),
    ("白", (255, 255, 255, 255)),
    ("红", (0, 0, 255, 255)),
    ("橙", (0, 165, 255, 255)),
    ("黄", (0, 255, 255, 255)),
    ("绿", (0, 255, 0, 255)),
    ("蓝", (255, 0, 0, 255)),
    ("紫", (255, 0, 255, 255)),
    ("粉", (203, 192, 255, 255)),
    ("棕", (19, 69, 139, 255)),
]

REGION_OPTIONS = [
    ("全脸", "full_face"),
    ("头顶", "head_top"),
    ("额头", "forehead_top"),
    ("眼睛", "eyes"),
    ("鼻子", "nose"),
    ("嘴巴", "mouth"),
    ("左脸颊", "cheek_left"),
    ("右脸颊", "cheek_right"),
    ("下巴", "chin"),
]


class DrawingCanvas(QWidget):
    CANVAS_SIZE = 512

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        self._drawing = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 4), dtype=np.uint8)
        self._undo_stack = []
        self._brush_size = 12
        self._brush_color = (0, 0, 0, 255)
        self._eraser_mode = False
        self._mirror_mode = False
        self._last_pos = None
        self._drawing_active = False

        self._checker = self._make_checkerboard()

    def _make_checkerboard(self):
        grid = 8
        rows = self.CANVAS_SIZE // grid + 1
        cols = self.CANVAS_SIZE // grid + 1
        img = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 3), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                color = 220 if (r + c) % 2 == 0 else 255
                y1, y2 = r * grid, min((r + 1) * grid, self.CANVAS_SIZE)
                x1, x2 = c * grid, min((c + 1) * grid, self.CANVAS_SIZE)
                img[y1:y2, x1:x2] = color
        h, w = img.shape[:2]
        qimg = QImage(img.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def load_image(self, bgra):
        if bgra is None:
            return
        h, w = bgra.shape[:2]
        scale = min(self.CANVAS_SIZE / max(w, h), 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            bgra = cv2.resize(bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = bgra.shape[:2]
        self._drawing = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 4), dtype=np.uint8)
        y_off = (self.CANVAS_SIZE - h) // 2
        x_off = (self.CANVAS_SIZE - w) // 2
        self._drawing[y_off:y_off + h, x_off:x_off + w] = bgra
        self._undo_stack.clear()
        self.update()

    def get_result(self):
        alpha = self._drawing[:, :, 3]
        nonzero = np.where(alpha > 0)
        if len(nonzero[0]) == 0:
            return self._drawing
        y1, y2 = max(nonzero[0].min() - 10, 0), min(nonzero[0].max() + 10, self.CANVAS_SIZE)
        x1, x2 = max(nonzero[1].min() - 10, 0), min(nonzero[1].max() + 10, self.CANVAS_SIZE)
        return self._drawing[y1:y2, x1:x2].copy()

    def set_brush_size(self, size):
        self._brush_size = max(1, min(50, size))

    def set_brush_color(self, color):
        self._brush_color = color
        self._eraser_mode = False

    def set_eraser_mode(self, on):
        self._eraser_mode = on

    def set_mirror_mode(self, on):
        self._mirror_mode = on
        self.update()

    def toggle_mirror(self):
        self._mirror_mode = not self._mirror_mode
        self.update()
        return self._mirror_mode

    def decrease_brush(self):
        self._brush_size = max(1, self._brush_size - 2)

    def increase_brush(self):
        self._brush_size = min(50, self._brush_size + 2)

    def undo(self):
        if self._undo_stack:
            self._drawing = self._undo_stack.pop()
            self.update()

    def clear_canvas(self):
        self._undo_stack.append(self._drawing.copy())
        self._drawing = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 4), dtype=np.uint8)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(0, 0, self._checker)
        pix = _bgra_to_qpixmap(self._drawing)
        if pix is not None:
            painter.drawPixmap(0, 0, pix)
        if self._mirror_mode:
            pen = QPen(QColor(102, 126, 234, 180), 1, Qt.DashLine)
            painter.setPen(pen)
            cx = self.CANVAS_SIZE // 2
            painter.drawLine(cx, 0, cx, self.CANVAS_SIZE)

    def _push_undo(self):
        self._undo_stack.append(self._drawing.copy())
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)

    def _mirror_x(self, x):
        return self.CANVAS_SIZE - x

    def _draw_stroke(self, p1, p2):
        color = (0, 0, 0, 0) if self._eraser_mode else self._brush_color
        self._draw_stroke_at(p1.x(), p1.y(), p2.x(), p2.y(), color)

    def _draw_stroke_at(self, x1, y1, x2, y2, color):
        cv2.line(self._drawing, (x1, y1), (x2, y2), color, self._brush_size, cv2.LINE_AA)
        cv2.circle(self._drawing, (x2, y2), self._brush_size // 2, color, -1, cv2.LINE_AA)
        if self._mirror_mode:
            mx1, mx2 = self._mirror_x(x1), self._mirror_x(x2)
            cv2.line(self._drawing, (mx1, y1), (mx2, y2), color, self._brush_size, cv2.LINE_AA)
            cv2.circle(self._drawing, (mx2, y2), self._brush_size // 2, color, -1, cv2.LINE_AA)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._push_undo()
            self._drawing_active = True
            self._last_pos = event.pos()
            self._draw_stroke(event.pos(), event.pos())
            self.update()

    def mouseMoveEvent(self, event):
        if self._drawing_active and self._last_pos is not None:
            self._draw_stroke(self._last_pos, event.pos())
            self._last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing_active = False
            self._last_pos = None


class DrawingDialog(QDialog):
    def __init__(self, parent, gallery_queue, command_queue=None, initial_sticker=None):
        super().__init__(parent)
        self.gallery_queue = gallery_queue
        self.command_queue = command_queue
        self.setWindowTitle("绘制贴纸")
        self.setFixedSize(660, 780)
        self.setStyleSheet("""
            QDialog { background: #f5f5f8; }
            QLabel { color: #444; font-size: 14px; }
            QSlider::groove:horizontal {
                background: #ddd; height: 6px;
            }
            QSlider::handle:horizontal {
                background: #667eea; width: 16px; height: 16px;
                margin: -5px 0;
            }
            QComboBox {
                background: #fff; color: #333; border: 2px solid #ddd;
                padding: 6px 12px; font-size: 14px;
            }
            QComboBox::drop-down { border: none; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(12)

        # 画布
        self.canvas = DrawingCanvas()
        if initial_sticker is not None:
            self.canvas.load_image(initial_sticker)
        canvas_container = QWidget()
        canvas_container.setStyleSheet("background: #fff; padding: 8px;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        layout.addWidget(canvas_container, alignment=Qt.AlignCenter)

        # 笔刷大小
        size_row = QHBoxLayout()
        size_label = QLabel("画笔大小:")
        size_row.addWidget(size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 50)
        self.size_slider.setValue(12)
        self.size_slider.valueChanged.connect(lambda v: self.canvas.set_brush_size(v))
        size_row.addWidget(self.size_slider)
        self.size_value_label = QLabel("12px")
        self.size_slider.valueChanged.connect(lambda v: self.size_value_label.setText(f"{v}px"))
        size_row.addWidget(self.size_value_label)
        layout.addLayout(size_row)

        # 颜色
        color_label = QLabel("颜色:")
        layout.addWidget(color_label)
        color_grid = QGridLayout()
        color_grid.setSpacing(4)
        for i, (name, bgra) in enumerate(PRESET_COLORS):
            btn = QPushButton()
            btn.setFixedSize(32, 32)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(name)
            r, g, b, a = bgra[2], bgra[1], bgra[0], bgra[3]
            btn.setStyleSheet(
                f"background: rgba({r},{g},{b},{a}); border: 2px solid #ccc;"
            )
            btn.clicked.connect(lambda _, c=bgra: self.canvas.set_brush_color(c))
            color_grid.addWidget(btn, 0, i)

        custom_btn = QPushButton("+")
        custom_btn.setFixedSize(32, 32)
        custom_btn.setCursor(Qt.PointingHandCursor)
        custom_btn.setToolTip("自定义颜色")
        custom_btn.setStyleSheet("background: #fff; border: 2px dashed #aaa; font-size: 14px;")
        custom_btn.clicked.connect(self._pick_custom_color)
        color_grid.addWidget(custom_btn, 0, len(PRESET_COLORS))
        layout.addLayout(color_grid)

        # 工具
        tool_row = QHBoxLayout()
        tool_row.setSpacing(8)

        self.brush_btn = StyledButton("画笔", "#667eea", "#818cf8")
        self.brush_btn.clicked.connect(self._set_brush_mode)
        tool_row.addWidget(self.brush_btn)

        self.eraser_btn = StyledButton("橡皮", "#94a3b8", "#b0bec5")
        self.eraser_btn.clicked.connect(self._set_eraser_mode)
        tool_row.addWidget(self.eraser_btn)

        undo_btn = StyledButton("撤销", "#f59e0b", "#fbbf24")
        undo_btn.clicked.connect(self.canvas.undo)
        tool_row.addWidget(undo_btn)

        clear_btn = StyledButton("清除画布", "#ef4444", "#f87171")
        clear_btn.clicked.connect(self.canvas.clear_canvas)
        tool_row.addWidget(clear_btn)

        self.mirror_btn = StyledButton("镜像", "#8b5cf6", "#a78bfa")
        self.mirror_btn.setCheckable(True)
        self.mirror_btn.clicked.connect(self._toggle_mirror)
        tool_row.addWidget(self.mirror_btn)

        tool_row.addStretch()

        import_btn = StyledButton("导入", "#06b6d4", "#22d3ee")
        import_btn.clicked.connect(self._import_image)
        tool_row.addWidget(import_btn)

        export_btn = StyledButton("导出", "#14b8a6", "#2dd4bf")
        export_btn.clicked.connect(self._export_image)
        tool_row.addWidget(export_btn)

        self.ext_edit_btn = StyledButton("外部编辑", "#f97316", "#fb923c")
        self.ext_edit_btn.clicked.connect(self._open_external_editor)
        tool_row.addWidget(self.ext_edit_btn)

        self._ext_watch_timer = None
        self._ext_file_mtime = None
        self._ext_file_path = None

        layout.addLayout(tool_row)

        # 快捷键提示
        hint_label = QLabel("快捷键: B 画笔 | E 橡皮 | M 镜像 | Ctrl+Z 撤销 | [ ] 调大小")
        hint_label.setStyleSheet("color: #aaa; font-size: 11px; background: transparent; border: none;")
        hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_label)

        # AI 精炼
        refine_row = QHBoxLayout()
        refine_row.setSpacing(8)
        refine_row.addWidget(QLabel("AI 精炼:"))

        self.refine_input = QLineEdit()
        self.refine_input.setPlaceholderText("描述风格，如：水彩画风、赛博朋克...")
        self.refine_input.returnPressed.connect(self._ai_refine)
        refine_row.addWidget(self.refine_input, stretch=1)

        self.refine_btn = StyledButton("AI 精炼", "#8b5cf6", "#a78bfa")
        self.refine_btn.clicked.connect(self._ai_refine)
        self.refine_btn.setEnabled(self.command_queue is not None)
        refine_row.addWidget(self.refine_btn)

        layout.addLayout(refine_row)

        # 位置 + 应用
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        bottom_row.addWidget(QLabel("位置:"))
        self.location_combo = QComboBox()
        for label, value in REGION_OPTIONS:
            self.location_combo.addItem(label, value)
        bottom_row.addWidget(self.location_combo)

        bottom_row.addStretch()

        apply_btn = StyledButton("应用到人脸", "#10b981", "#34d399")
        apply_btn.clicked.connect(self._apply)
        bottom_row.addWidget(apply_btn)

        layout.addLayout(bottom_row)

    def _set_brush_mode(self):
        self.canvas.set_eraser_mode(False)
        self.canvas.set_brush_color((0, 0, 0, 255))

    def _set_eraser_mode(self):
        self.canvas.set_eraser_mode(True)

    def _toggle_mirror(self):
        on = self.canvas.toggle_mirror()
        self.mirror_btn.setChecked(on)
        self.mirror_btn.setText("镜像中" if on else "镜像")

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        if key == Qt.Key_B:
            self._set_brush_mode()
        elif key == Qt.Key_E:
            self._set_eraser_mode()
        elif key == Qt.Key_M:
            self._toggle_mirror()
        elif key == Qt.Key_Z and mods == Qt.ControlModifier:
            self.canvas.undo()
        elif key == Qt.Key_BracketLeft:
            self.canvas.decrease_brush()
            self.size_slider.setValue(self.canvas._brush_size)
        elif key == Qt.Key_BracketRight:
            self.canvas.increase_brush()
            self.size_slider.setValue(self.canvas._brush_size)
        else:
            super().keyPressEvent(event)

    def _import_image(self):
        from PyQt5.QtWidgets import QFileDialog
        from app.utils.image_proc import load_rgba_sticker
        path, _ = QFileDialog.getOpenFileName(
            self, "导入图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp);;所有文件 (*.*)"
        )
        if not path:
            return
        sticker = load_rgba_sticker(path)
        if sticker is not None:
            self.canvas.load_image(sticker)

    def _export_image(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "导出贴纸", "sticker.png",
            "PNG (*.png);;所有文件 (*.*)"
        )
        if not path:
            return
        result = self.canvas.get_result()
        cv2.imwrite(path, result)

    def _open_external_editor(self):
        import os as _os
        result = self.canvas.get_result()
        if result is None:
            return

        temp_dir = "assets/temp"
        _os.makedirs(temp_dir, exist_ok=True)
        self._ext_file_path = _os.path.join(temp_dir, "edit_export.png")
        cv2.imwrite(self._ext_file_path, result)
        self._ext_file_mtime = _os.path.getmtime(self._ext_file_path)

        from app.utils.config_loader import get_config
        cfg = get_config()
        editor_cfg = cfg.get("external_editor", {})
        editor_path = editor_cfg.get("path", "").strip()
        editor_args = editor_cfg.get("args", "").strip()

        try:
            if editor_path:
                cmd = [editor_path]
                if editor_args:
                    cmd.extend(editor_args.split())
                cmd.append(self._ext_file_path)
                subprocess.Popen(cmd, shell=False)
            else:
                _os.startfile(self._ext_file_path)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "启动失败",
                f"无法启动外部编辑器。\n\n请确认已安装 PS/SAI2 等软件，"
                f"或在 config.json 中设置 external_editor.path。\n\n错误: {e}")
            return

        self.ext_edit_btn.setText("等待保存...")
        self.ext_edit_btn.setEnabled(False)

        self._ext_watch_timer = QTimer(self)
        self._ext_watch_timer.timeout.connect(self._check_external_file)
        self._ext_watch_timer.start(1000)

    def _check_external_file(self):
        if not self._ext_file_path or not os.path.exists(self._ext_file_path):
            return
        try:
            mtime = os.path.getmtime(self._ext_file_path)
        except OSError:
            return
        if self._ext_file_mtime is not None and abs(mtime - self._ext_file_mtime) > 0.1:
            self._ext_file_mtime = mtime
            from app.utils.image_proc import load_rgba_sticker
            updated = load_rgba_sticker(self._ext_file_path)
            if updated is not None:
                self.canvas.load_image(updated)
                self.ext_edit_btn.setText("已同步")
        else:
            return

    def closeEvent(self, event):
        if self._ext_watch_timer:
            self._ext_watch_timer.stop()
        super().closeEvent(event)

    def _pick_custom_color(self):
        qcolor = QColorDialog.getColor(parent=self)
        if qcolor.isValid():
            bgra = (qcolor.blue(), qcolor.green(), qcolor.red(), 255)
            self.canvas.set_brush_color(bgra)

    def _ai_refine(self):
        if self.command_queue is None:
            return
        refine_text = self.refine_input.text().strip()
        result = self.canvas.get_result()
        if result is None:
            return
        temp_dir = "assets/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "doodle_img2img_input.png")
        cv2.imwrite(temp_path, result)
        style_prefix = "flat vector sticker, front view, flat lay, clean outline, solid white background, icon design"
        full_prompt = f"{style_prefix}, {refine_text}" if refine_text else style_prefix
        region = self.location_combo.currentData()
        self.command_queue.put({
            "type": "img2img",
            "prompt_text": full_prompt,
            "image_path": os.path.abspath(temp_path),
            "target_location": region,
            "scale": 1.0,
            "display_name": refine_text if refine_text else "简笔画精炼",
        })
        self.refine_btn.setText("精炼中...")
        self.refine_btn.setEnabled(False)
        self.refine_input.setEnabled(False)
        QTimer.singleShot(300, self.accept)

    def _apply(self):
        result = self.canvas.get_result()
        if result is None:
            return

        region = self.location_combo.currentData()
        from app.utils import storage
        sid = storage.save_sticker(result, {
            "prompt": "手绘贴纸",
            "location": region,
            "scale": 1.0,
        })
        self.gallery_queue.put({"action": "load_sticker", "sticker_id": sid})
        self.accept()
