"""Hand-drawn sticker canvas and dialog."""

import cv2
import numpy as np
import os
import subprocess

from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton,
                             QHBoxLayout, QVBoxLayout, QSlider, QColorDialog,
                             QComboBox, QDialog, QGridLayout, QScrollArea)
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QEvent

from app.core.brush import BrushStateMixin, load_brush_config, stamp_brush, stamp_line
from app.core.protocol import CmdImg2Img, GalLoadSticker
from app.utils.config_loader import build_positive_prompt
from app.ui.widgets import _bgra_to_qpixmap, PRESET_COLORS, REGION_OPTIONS, StyledButton
from app.ui.theme import (PRIMARY, CANVAS, PARCHMENT, INK, INK_MUTED_48,
                          INK_MUTED_80, HAIRLINE, font_css, ROUNDED)


class DrawingCanvas(QWidget, BrushStateMixin):
    CANVAS_SIZE = 800

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        BrushStateMixin.__init__(self)
        self.setMinimumSize(400, 400)
        self.setMaximumSize(2400, 2400)
        self.resize(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        self._drawing = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 4), dtype=np.uint8)
        self._undo_stack = []
        self._init_brush_state()

        self._mirror_mode = False
        self._last_pos = None
        self._drawing_active = False
        self._tablet_in_use = False
        self._zoom = 1.0
        self._space_held = False
        self._panning = False
        self._pan_start = None
        self._scroll_area = None

        self._checker = self._make_checkerboard()
        self.setTabletTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

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
        self._reset_zoom()
        self.update()

    def get_result(self):
        alpha = self._drawing[:, :, 3]
        nonzero = np.where(alpha > 0)
        if len(nonzero[0]) == 0:
            return self._drawing
        y1, y2 = max(nonzero[0].min() - 10, 0), min(nonzero[0].max() + 10, self.CANVAS_SIZE)
        x1, x2 = max(nonzero[1].min() - 10, 0), min(nonzero[1].max() + 10, self.CANVAS_SIZE)
        return self._drawing[y1:y2, x1:x2].copy()

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
        self._reset_zoom()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        _, draw_size, ox, oy = self._get_transform()
        painter.drawPixmap(ox, oy, draw_size, draw_size, self._checker)
        pix = _bgra_to_qpixmap(self._drawing)
        if pix is not None:
            painter.drawPixmap(ox, oy, draw_size, draw_size, pix)
        if self._mirror_mode:
            c = QColor(PRIMARY)
            c.setAlpha(180)
            pen = QPen(c, 1, Qt.DashLine)
            painter.setPen(pen)
            cx = ox + draw_size // 2
            painter.drawLine(cx, oy, cx, oy + draw_size)

    def _push_undo(self):
        self._undo_stack.append(self._drawing.copy())
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)

    def _mirror_x(self, x):
        return self.CANVAS_SIZE - x

    def _draw_stroke(self, p1, p2):
        color = (0, 0, 0, 0) if self._eraser_mode else self._brush_color
        cx1, cy1 = self._widget_to_canvas(p1)
        cx2, cy2 = self._widget_to_canvas(p2)
        self._draw_stroke_at(cx1, cy1, cx2, cy2, color)

    def _get_transform(self):
        w, h = self.width(), self.height()
        scale = min(w, h) / self.CANVAS_SIZE
        draw_size = int(self.CANVAS_SIZE * scale)
        ox = (w - draw_size) // 2
        oy = (h - draw_size) // 2
        return scale, draw_size, ox, oy

    def _widget_to_canvas(self, pos):
        scale, draw_size, ox, oy = self._get_transform()
        if scale <= 0:
            return int(pos.x()), int(pos.y())
        cx = int((pos.x() - ox) / scale)
        cy = int((pos.y() - oy) / scale)
        return cx, cy

    def _draw_stroke_at(self, x1, y1, x2, y2, color):
        spacing_px, scatter_px, opacity_scale = self._resolve_spacing_scatter()
        self._ensure_tip()
        stamp_line(self._drawing, (x1, y1), (x2, y2), self._brush_tip,
                   color, opacity_scale, spacing_px, scatter=scatter_px,
                   eraser=self._eraser_mode)

        if self._mirror_mode:
            mx1, mx2 = self._mirror_x(x1), self._mirror_x(x2)
            stamp_line(self._drawing, (mx1, y1), (mx2, y2), self._brush_tip,
                       color, opacity_scale, spacing_px, scatter=scatter_px,
                       eraser=self._eraser_mode)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._tablet_in_use:
            if self._space_held:
                self._panning = True
                self._pan_start = event.globalPos()
                self.setCursor(Qt.ClosedHandCursor)
            else:
                self._push_undo()
                self._drawing_active = True
                self._last_pos = event.pos()
                self._draw_stroke(event.pos(), event.pos())
                self.update()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.globalPos() - self._pan_start
            self._pan_start = event.globalPos()
            if self._scroll_area:
                self._scroll_area.horizontalScrollBar().setValue(
                    self._scroll_area.horizontalScrollBar().value() - delta.x())
                self._scroll_area.verticalScrollBar().setValue(
                    self._scroll_area.verticalScrollBar().value() - delta.y())
        elif self._drawing_active and self._last_pos is not None and not self._tablet_in_use:
            self._draw_stroke(self._last_pos, event.pos())
            self._last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._panning:
                self._panning = False
                self._pan_start = None
                self._update_cursor()
            elif not self._tablet_in_use:
                self._drawing_active = False
                self._last_pos = None

    def set_scroll_area(self, scroll_area):
        self._scroll_area = scroll_area

    def _update_cursor(self):
        if self._space_held:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def tabletEvent(self, event):
        t = event.type()
        if self._space_held:
            if t == QEvent.TabletPress:
                self._tablet_in_use = True
                self._panning = True
                self._pan_start = event.globalPos()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
            elif t == QEvent.TabletMove and self._panning:
                delta = event.globalPos() - self._pan_start
                self._pan_start = event.globalPos()
                if self._scroll_area:
                    self._scroll_area.horizontalScrollBar().setValue(
                        self._scroll_area.horizontalScrollBar().value() - delta.x())
                    self._scroll_area.verticalScrollBar().setValue(
                        self._scroll_area.verticalScrollBar().value() - delta.y())
                event.accept()
            elif t == QEvent.TabletRelease:
                self._panning = False
                self._pan_start = None
                self._tablet_in_use = False
                self._update_cursor()
                event.accept()
            return
        if t == QEvent.TabletPress:
            self._tablet_in_use = True
            self.set_pressure(event.pressure())
            self._push_undo()
            self._drawing_active = True
            self._last_pos = event.pos()
            self._draw_stroke(event.pos(), event.pos())
            self.update()
            event.accept()
        elif t == QEvent.TabletMove and self._tablet_in_use:
            self.set_pressure(event.pressure())
            if event.pressure() > 0 and self._drawing_active and self._last_pos is not None:
                self._draw_stroke(self._last_pos, event.pos())
                self._last_pos = event.pos()
                self.update()
            elif event.pressure() == 0:
                self._drawing_active = False
                self._last_pos = None
                self._tablet_in_use = False
            event.accept()
        elif t == QEvent.TabletRelease:
            self._drawing_active = False
            self._last_pos = None
            self._tablet_in_use = False
            event.accept()
        else:
            super().tabletEvent(event)

    def _reset_zoom(self):
        self._zoom = 1.0
        self.resize(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self.update()

    def wheelEvent(self, event):
        if self._drawing_active:
            return  # ignore scroll during drawing to avoid coordinate shift
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._zoom = max(0.25, min(4.0, self._zoom * factor))
        size = int(self.CANVAS_SIZE * self._zoom)
        self.resize(size, size)
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self._update_cursor()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._update_cursor()
        else:
            super().keyReleaseEvent(event)


class DrawingDialog(QDialog):
    def __init__(self, parent, gallery_queue, command_queue=None, initial_sticker=None):
        super().__init__(parent)
        self.gallery_queue = gallery_queue
        self.command_queue = command_queue
        self.setWindowTitle("绘制贴纸")
        self.resize(1100, 720)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.setStyleSheet(f"""
            QDialog {{ background: {CANVAS}; }}
            QLabel {{ color: {INK}; {font_css('caption')} }}
            QSlider::groove:horizontal {{
                background: {HAIRLINE}; height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {PRIMARY}; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }}
            QComboBox {{
                background: {CANVAS}; color: {INK}; border: 1px solid {HAIRLINE};
                padding: 5px 8px; {font_css('caption')} border-radius: {ROUNDED['sm']};
            }}
            QComboBox:hover {{ border-color: {PRIMARY}; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {CANVAS}; color: {INK};
                selection-background-color: {PRIMARY}; selection-color: {CANVAS};
                border: 1px solid {HAIRLINE};
            }}
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(8)

        # ── 主体：左侧笔刷属性 + 画布 + 右侧工具栏 ──
        body_row = QHBoxLayout()
        body_row.setSpacing(10)

        # 画布 (先创建, 以便侧边栏按钮可以引用)
        self.canvas = DrawingCanvas()
        if initial_sticker is not None:
            self.canvas.load_image(initial_sticker)
        self._selected_color_btn = None
        self._color_btns = []

        # ── 左侧属性栏：笔刷参数 ──
        left_sidebar = QWidget()
        left_sidebar.setFixedWidth(190)
        left_sidebar.setStyleSheet(f"background: {PARCHMENT}; border: 1px solid {HAIRLINE}; border-radius: {ROUNDED['sm']};")
        left_layout = QVBoxLayout(left_sidebar)
        left_layout.setContentsMargins(8, 10, 8, 10)
        left_layout.setSpacing(6)

        row_style = f"color: {INK_MUTED_80}; {font_css('caption')} background: transparent; border: none;"
        value_style = f"color: {INK_MUTED_48}; {font_css('fine-print')} background: transparent; border: none;"

        # 笔刷类型 — 左右布局
        brush_row = QHBoxLayout()
        brush_label = QLabel("笔刷")
        brush_label.setFixedWidth(30)
        brush_label.setStyleSheet(row_style)
        brush_row.addWidget(brush_label)
        self.brush_type_combo = QComboBox()
        brushes = load_brush_config()
        for b in brushes:
            self.brush_type_combo.addItem(b["name"], b["id"])
        self.brush_type_combo.currentIndexChanged.connect(self._on_brush_type_changed)
        brush_row.addWidget(self.brush_type_combo, stretch=1)
        left_layout.addLayout(brush_row)

        # 压感模式 — 左右布局
        pressure_row = QHBoxLayout()
        pressure_label = QLabel("压感")
        pressure_label.setFixedWidth(30)
        pressure_label.setStyleSheet(row_style)
        pressure_row.addWidget(pressure_label)
        self.pressure_mode_combo = QComboBox()
        for label, mode in [("大小+浓度", "both"), ("仅大小", "size"),
                             ("仅浓度", "opacity"), ("无压感", "none")]:
            self.pressure_mode_combo.addItem(label, mode)
        self.pressure_mode_combo.currentIndexChanged.connect(self._on_pressure_mode_changed)
        pressure_row.addWidget(self.pressure_mode_combo, stretch=1)
        left_layout.addLayout(pressure_row)

        # 画笔大小 — 拉条
        size_label = QLabel("大小")
        size_label.setStyleSheet(row_style)
        left_layout.addWidget(size_label)
        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 50)
        self.size_slider.setValue(12)
        self.size_slider.valueChanged.connect(lambda v: self.canvas.set_brush_size(v))
        size_row.addWidget(self.size_slider, stretch=1)
        self.size_value_label = QLabel("12")
        self.size_value_label.setFixedWidth(24)
        self.size_value_label.setStyleSheet(value_style)
        self.size_slider.valueChanged.connect(lambda v: self.size_value_label.setText(str(v)))
        size_row.addWidget(self.size_value_label)
        left_layout.addLayout(size_row)

        # 间距 — 拉条
        spacing_label = QLabel("间距")
        spacing_label.setStyleSheet(row_style)
        left_layout.addWidget(spacing_label)
        spacing_row = QHBoxLayout()
        spacing_row.setSpacing(6)
        self.spacing_slider = QSlider(Qt.Horizontal)
        self.spacing_slider.setRange(3, 200)
        self.spacing_slider.setValue(30)
        self.spacing_slider.valueChanged.connect(self._on_spacing_changed)
        spacing_row.addWidget(self.spacing_slider, stretch=1)
        self.spacing_value_label = QLabel("0.30")
        self.spacing_value_label.setFixedWidth(28)
        self.spacing_value_label.setStyleSheet(value_style)
        spacing_row.addWidget(self.spacing_value_label)
        left_layout.addLayout(spacing_row)

        # 散射 — 拉条
        scatter_label = QLabel("散射")
        scatter_label.setStyleSheet(row_style)
        left_layout.addWidget(scatter_label)
        scatter_row = QHBoxLayout()
        scatter_row.setSpacing(6)
        self.scatter_slider = QSlider(Qt.Horizontal)
        self.scatter_slider.setRange(0, 30)
        self.scatter_slider.setValue(0)
        self.scatter_slider.valueChanged.connect(self._on_scatter_changed)
        scatter_row.addWidget(self.scatter_slider, stretch=1)
        self.scatter_value_label = QLabel("0")
        self.scatter_value_label.setFixedWidth(20)
        self.scatter_value_label.setStyleSheet(value_style)
        scatter_row.addWidget(self.scatter_value_label)
        left_layout.addLayout(scatter_row)

        # 颜色选择
        color_label = QLabel("颜色")
        color_label.setStyleSheet(row_style)
        left_layout.addWidget(color_label)
        color_grid = QGridLayout()
        color_grid.setSpacing(3)
        cols = 5
        self._color_btns = []
        for i, (name, bgra) in enumerate(PRESET_COLORS):
            btn = QPushButton()
            btn.setFixedSize(26, 26)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(name)
            r, g, b, a = bgra[2], bgra[1], bgra[0], bgra[3]
            btn.setStyleSheet(
                f"QPushButton {{ background: rgba({r},{g},{b},{a}); border: 1px solid {HAIRLINE}; border-radius: {ROUNDED['full']}; }}"
            )
            btn.clicked.connect(lambda _, c=bgra, b=btn: self._on_color_clicked(c, b))
            color_grid.addWidget(btn, i // cols, i % cols)
            self._color_btns.append(btn)

        custom_btn = QPushButton("+")
        custom_btn.setFixedSize(26, 26)
        custom_btn.setCursor(Qt.PointingHandCursor)
        custom_btn.setToolTip("自定义颜色")
        custom_btn.setStyleSheet(f"QPushButton {{ background: {CANVAS}; border: 2px dashed {INK_MUTED_48}; font-size: 13px; border-radius: {ROUNDED['full']}; }}")
        custom_btn.clicked.connect(self._pick_custom_color)
        ci = len(PRESET_COLORS)
        color_grid.addWidget(custom_btn, ci // cols, ci % cols)
        left_layout.addLayout(color_grid)

        # 默认选中第一个颜色（黑）
        if self._color_btns:
            self._on_color_clicked(PRESET_COLORS[0][1], self._color_btns[0])

        left_layout.addStretch()
        body_row.addWidget(left_sidebar)

        canvas_scroll = QScrollArea()
        canvas_scroll.setStyleSheet(f"QScrollArea {{ background: {CANVAS}; border: 1px solid {HAIRLINE}; border-radius: {ROUNDED['sm']}; }}")
        canvas_scroll.setWidgetResizable(False)
        canvas_scroll.setWidget(self.canvas)
        canvas_scroll.setAlignment(Qt.AlignCenter)
        self.canvas.set_scroll_area(canvas_scroll)
        body_row.addWidget(canvas_scroll, stretch=1)

        # ── 右侧工具栏 ──
        right_sidebar = QWidget()
        right_sidebar.setFixedWidth(160)
        right_sidebar.setStyleSheet(f"background: {PARCHMENT}; border: 1px solid {HAIRLINE}; border-radius: {ROUNDED['sm']};")
        right_layout = QVBoxLayout(right_sidebar)
        right_layout.setContentsMargins(8, 10, 8, 10)
        right_layout.setSpacing(6)

        section_label_style = f"color: {INK_MUTED_48}; {font_css('fine-print')} background: transparent; border: none;"

        draw_tools_label = QLabel("绘制")
        draw_tools_label.setStyleSheet(section_label_style)
        right_layout.addWidget(draw_tools_label)

        self.brush_btn = StyledButton("画笔", "utility")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.brush_btn.clicked.connect(self._set_brush_mode)
        right_layout.addWidget(self.brush_btn)

        self.eraser_btn = StyledButton("橡皮", "utility")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(self._set_eraser_mode)
        right_layout.addWidget(self.eraser_btn)

        sep_left1 = QWidget()
        sep_left1.setFixedHeight(1)
        sep_left1.setStyleSheet(f"background: {HAIRLINE}; border: none;")
        right_layout.addWidget(sep_left1)

        self.mirror_btn = StyledButton("镜像", "checkable")
        self.mirror_btn.setCheckable(True)
        self.mirror_btn.clicked.connect(self._toggle_mirror)
        right_layout.addWidget(self.mirror_btn)

        sep_left2 = QWidget()
        sep_left2.setFixedHeight(1)
        sep_left2.setStyleSheet(f"background: {HAIRLINE}; border: none;")
        right_layout.addWidget(sep_left2)

        undo_label = QLabel("历史")
        undo_label.setStyleSheet(section_label_style)
        right_layout.addWidget(undo_label)

        undo_btn = StyledButton("撤销", "utility")
        undo_btn.clicked.connect(self.canvas.undo)
        right_layout.addWidget(undo_btn)

        clear_btn = StyledButton("清除", "utility-danger")
        clear_btn.clicked.connect(self.canvas.clear_canvas)
        right_layout.addWidget(clear_btn)

        right_layout.addStretch()

        file_label = QLabel("文件")
        file_label.setStyleSheet(section_label_style)
        right_layout.addWidget(file_label)

        import_btn = StyledButton("导入", "ghost")
        import_btn.clicked.connect(self._import_image)
        right_layout.addWidget(import_btn)

        export_btn = StyledButton("导出", "ghost")
        export_btn.clicked.connect(self._export_image)
        right_layout.addWidget(export_btn)

        self.ext_edit_btn = StyledButton("外部编辑", "ghost")
        self.ext_edit_btn.clicked.connect(self._open_external_editor)
        right_layout.addWidget(self.ext_edit_btn)

        self._ext_watch_timer = None
        self._ext_file_mtime = None
        self._ext_file_path = None

        body_row.addWidget(right_sidebar)
        root_layout.addLayout(body_row, stretch=1)

        # ── 底部栏：AI精炼 + 位置 + 应用 ──
        bottom_widget = QWidget()
        bottom_widget.setStyleSheet(f"background: {CANVAS}; border-top: 1px solid {HAIRLINE};")
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(10, 8, 10, 6)
        bottom_layout.setSpacing(6)

        refine_row = QHBoxLayout()
        refine_row.setSpacing(8)
        refine_row.addWidget(QLabel("AI 精炼:"))
        self.refine_input = QLineEdit()
        self.refine_input.setPlaceholderText("描述风格，如：水彩画风、赛博朋克...")
        self.refine_input.returnPressed.connect(self._ai_refine)
        refine_row.addWidget(self.refine_input, stretch=1)
        self.refine_btn = StyledButton("AI 精炼", "primary")
        self.refine_btn.clicked.connect(self._ai_refine)
        self.refine_btn.setEnabled(self.command_queue is not None)
        refine_row.addWidget(self.refine_btn)
        bottom_layout.addLayout(refine_row)

        # ── 精炼参数 ──
        params_row = QHBoxLayout()
        params_row.setSpacing(8)

        params_row.addWidget(QLabel("ControlNet:"))
        self.cn_slider = QSlider(Qt.Horizontal)
        self.cn_slider.setRange(0, 100)
        self.cn_slider.setValue(85)
        self.cn_slider.setFixedWidth(100)
        self.cn_slider.valueChanged.connect(self._on_cn_slider_changed)
        params_row.addWidget(self.cn_slider)
        self.cn_value_label = QLabel("0.85")
        self.cn_value_label.setStyleSheet(f"color: {INK_MUTED_80}; {font_css('fine-print')}; min-width: 28px;")
        params_row.addWidget(self.cn_value_label)

        params_row.addWidget(QLabel("降噪:"))
        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(5, 100)
        self.denoise_slider.setValue(100)
        self.denoise_slider.setFixedWidth(100)
        self.denoise_slider.valueChanged.connect(self._on_denoise_slider_changed)
        params_row.addWidget(self.denoise_slider)
        self.denoise_value_label = QLabel("1.00")
        self.denoise_value_label.setStyleSheet(f"color: {INK_MUTED_80}; {font_css('fine-print')}; min-width: 28px;")
        params_row.addWidget(self.denoise_value_label)

        params_row.addStretch()
        bottom_layout.addLayout(params_row)

        apply_row = QHBoxLayout()
        apply_row.setSpacing(8)
        apply_row.addWidget(QLabel("位置:"))
        self.location_combo = QComboBox()
        for label, value in REGION_OPTIONS:
            self.location_combo.addItem(label, value)
        self.location_combo.setStyleSheet(f"QComboBox {{ {font_css('body')} }}")
        apply_row.addWidget(self.location_combo)
        apply_row.addStretch()
        apply_btn = StyledButton("应用到人脸", "primary")
        apply_btn.clicked.connect(self._apply)
        apply_row.addWidget(apply_btn)
        bottom_layout.addLayout(apply_row)

        hint_label = QLabel("B 画笔  E 橡皮  M 镜像  Ctrl+Z 撤销  [ ] 大小")
        hint_label.setStyleSheet(f"color: {INK_MUTED_48}; {font_css('fine-print')} background: transparent; border: none;")
        hint_label.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(hint_label)

        root_layout.addWidget(bottom_widget)

    def _set_brush_mode(self):
        self.canvas.set_eraser_mode(False)
        self.canvas.set_brush_color((0, 0, 0, 255))
        self.brush_btn.setChecked(True)
        self.eraser_btn.setChecked(False)

    def _set_eraser_mode(self):
        self.canvas.set_eraser_mode(True)
        self.brush_btn.setChecked(False)
        self.eraser_btn.setChecked(True)

    def _on_brush_type_changed(self, idx):
        brush_id = self.brush_type_combo.currentData()
        if brush_id:
            self.canvas.set_brush_type(brush_id)

    def _on_pressure_mode_changed(self, idx):
        mode = self.pressure_mode_combo.currentData()
        if mode:
            self.canvas.set_pressure_mode(mode)

    def _on_spacing_changed(self, value):
        coef = value / 100.0
        self.spacing_value_label.setText(f"{coef:.2f}")
        self.canvas.set_spacing(coef)

    def _on_scatter_changed(self, value):
        self.scatter_value_label.setText(str(value))
        self.canvas.set_scatter(float(value))

    def _on_cn_slider_changed(self, value):
        self.cn_value_label.setText(f"{value / 100:.2f}")

    def _on_denoise_slider_changed(self, value):
        self.denoise_value_label.setText(f"{value / 100:.2f}")

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
        result = self.canvas.get_result()
        if result is None:
            return

        temp_dir = "assets/temp"
        os.makedirs(temp_dir, exist_ok=True)
        self._ext_file_path = os.path.join(temp_dir, "edit_export.png")
        cv2.imwrite(self._ext_file_path, result)
        self._ext_file_mtime = os.path.getmtime(self._ext_file_path)

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
                os.startfile(self._ext_file_path)
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
            self._deselect_all_colors()

    def _on_color_clicked(self, bgra, btn):
        self.canvas.set_brush_color(bgra)
        self._deselect_all_colors()
        r, g, b, a = bgra[2], bgra[1], bgra[0], bgra[3]
        btn.setStyleSheet(
            f"QPushButton {{ background: rgba({r},{g},{b},{a}); "
            f"border: 2px solid {PRIMARY}; border-radius: {ROUNDED['full']}; }}"
        )
        self._selected_color_btn = btn

    def _deselect_all_colors(self):
        if self._selected_color_btn is not None:
            # 恢复默认样式
            for btn in self._color_btns:
                # 找到 btn 的原始颜色来还原
                idx = self._color_btns.index(btn)
                name, bgra = PRESET_COLORS[idx]
                r, g, b, a = bgra[2], bgra[1], bgra[0], bgra[3]
                btn.setStyleSheet(
                    f"QPushButton {{ background: rgba({r},{g},{b},{a}); "
                    f"border: 1px solid {HAIRLINE}; border-radius: {ROUNDED['full']}; }}"
                )
            self._selected_color_btn = None

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
        full_prompt = build_positive_prompt(refine_text)
        region = self.location_combo.currentData()
        cn_strength = self.cn_slider.value() / 100.0
        denoise = self.denoise_slider.value() / 100.0
        self.command_queue.put(CmdImg2Img(
            prompt_text=full_prompt,
            image_path=os.path.abspath(temp_path),
            target_location=region,
            scale=1.0,
            display_name=refine_text if refine_text else "简笔画精炼",
            controlnet_strength=cn_strength,
            denoise=denoise,
        ))
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
        self.gallery_queue.put(GalLoadSticker(sticker_id=sid))
        self.accept()
