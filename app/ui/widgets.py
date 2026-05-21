from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QScrollArea,
                             QHBoxLayout, QVBoxLayout, QSizePolicy, QLayout)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, QRect, QSize
import numpy as np

from app.ui.theme import (PRIMARY, CANVAS, PARCHMENT, INK,
                          INK_MUTED_48, INK_MUTED_80, HAIRLINE, DIVIDER_SOFT,
                          DESTRUCTIVE, font_css, ROUNDED, rgba,
                          pill_button_style, ghost_pill_button_style,
                          ghost_destructive_button_style, checkable_pill_button_style,
                          utility_button_style, utility_danger_button_style)


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
    clicked = Signal(str)  # sticker_id

    def __init__(self, sticker_id, thumb_bgra, prompt, is_favorite=False, parent=None, is_animated=False):
        super().__init__(parent)
        self.sticker_id = sticker_id
        self._selected = False
        self._active = False
        self._fav = is_favorite
        self._animated = is_animated
        self.setFixedSize(100, 120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(96, 96)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet(
            f"background: {DIVIDER_SOFT}; border: 2px solid {HAIRLINE}; border-radius: {ROUNDED['xs']};"
        )
        if thumb_bgra is not None:
            pix = _bgra_to_qpixmap(thumb_bgra)
            if pix is not None:
                self.thumb_label.setPixmap(
                    pix.scaled(92, 92, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.fav_star = QLabel("★", self)
        self.fav_star.setStyleSheet(f"color: {PRIMARY}; font-size: 14px; background: transparent; border: none;")
        self.fav_star.setVisible(self._fav)
        self.fav_star.move(78, 2)

        self.anim_badge = QLabel("▶", self)
        self.anim_badge.setStyleSheet(f"color: {PRIMARY}; font-size: 12px; background: transparent; border: none; font-weight: bold;")
        self.anim_badge.setVisible(self._animated)
        self.anim_badge.move(4, 2)

        label_text = prompt[:10] + ".." if len(prompt) > 10 else prompt
        self.name_label = QLabel(label_text)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet(
            f"color: {INK_MUTED_80}; {font_css('caption')} background: transparent; border: none;"
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
            color = PRIMARY
        elif self._active:
            color = "#10b981"
        else:
            color = HAIRLINE
        self.thumb_label.setStyleSheet(
            f"background: {DIVIDER_SOFT}; border: 2px solid {color}; border-radius: {ROUNDED['xs']};"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.sticker_id)


class StyledButton(QPushButton):
    """使用设计预设的按钮组件。

    预设键: 'primary' (pill CTA), 'ghost' (线框 pill),
    'ghost-destructive' (红色线框 pill), 'checkable' (pill 切换),
    'utility' (紧凑矩形).
    """

    PRESETS = {}

    @classmethod
    def _init_presets(cls):
        if cls.PRESETS:
            return
        cls.PRESETS['primary'] = pill_button_style()
        cls.PRESETS['ghost'] = ghost_pill_button_style()
        cls.PRESETS['ghost-destructive'] = ghost_destructive_button_style()
        cls.PRESETS['checkable'] = checkable_pill_button_style()
        cls.PRESETS['utility'] = utility_button_style()
        cls.PRESETS['utility-danger'] = utility_danger_button_style()

    def __init__(self, text, preset="primary", parent=None):
        super().__init__(text, parent)
        self._init_presets()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(StyledButton.PRESETS.get(preset, StyledButton.PRESETS['primary']))


class TitleBar(QWidget):
    """简洁的标题栏 — 米白背景配底部细线分割。"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setStyleSheet(
            f"background: {PARCHMENT}; border-bottom: 1px solid {HAIRLINE};"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 16, 0)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"color: {INK}; {font_css('tagline')} background: transparent; border: none;"
        )
        layout.addWidget(self.title_label)

        self.right_widget = None
        layout.addStretch()

    def add_right_widget(self, widget):
        self.right_widget = widget
        layout = self.layout()
        if layout:
            layout.addWidget(widget)


# Backward-compatible alias
GradientBar = TitleBar


class GallerySectionHeader(QWidget):
    toggled = Signal(str, bool)           # section_id, expanded
    loadGroupRequested = Signal(str)      # group_id

    def __init__(self, section_id, name, count, group_id=None, parent=None):
        super().__init__(parent)
        self.section_id = section_id
        self.group_id = group_id or ""
        self._expanded = True
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 4, 4)
        layout.setSpacing(4)

        self._chevron = QLabel("▾")
        self._chevron.setFixedWidth(10)
        self._chevron.setStyleSheet(
            f"color: {INK_MUTED_48}; font-size: 10px; background: transparent; border: none;"
        )
        layout.addWidget(self._chevron)

        self._full_name = name
        self._name_label = QLabel(name)
        self._name_label.setStyleSheet(
            f"color: {INK_MUTED_80}; {font_css('caption-strong')} background: transparent; border: none;"
        )
        self._name_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._name_label.setMinimumWidth(20)
        layout.addWidget(self._name_label, stretch=1)

        if group_id:
            self._add_btn = QPushButton("+")
            self._add_btn.setFixedSize(22, 20)
            self._add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._add_btn.setToolTip("整组添加到面部")
            self._add_btn.setStyleSheet(
                f"QPushButton {{ color: #fff; background: {PRIMARY}; "
                f"border-radius: {ROUNDED['sm']}; font-size: 13px; font-weight: 700; "
                f"border: 1px solid {PRIMARY}; padding: 0; }}"
                f"QPushButton:hover {{ background: #fff; color: {PRIMARY}; }}"
            )
            self._add_btn.clicked.connect(lambda: self.loadGroupRequested.emit(self.group_id))
            layout.addWidget(self._add_btn)
        else:
            self._add_btn = None

        self._count_badge = QLabel(str(count))
        self._count_badge.setFixedSize(22, 18)
        self._count_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_badge.setStyleSheet(
            f"color: {INK_MUTED_48}; background: {DIVIDER_SOFT}; border-radius: {ROUNDED['xs']}; "
            f"font-size: 9px; border: none;"
        )
        layout.addWidget(self._count_badge)

    def set_expanded(self, expanded):
        self._expanded = expanded
        self._chevron.setText("▾" if expanded else "▸")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._add_btn and self._add_btn.geometry().contains(event.pos()):
                super().mousePressEvent(event)
                return
            new_state = not self._expanded
            self.set_expanded(new_state)
            self.toggled.emit(self.section_id, new_state)


class GalleryScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(140)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: transparent; width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: {rgba(INK, 0.15)}; min-height: 30px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {rgba(INK, 0.25)};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(8, 4, 8, 4)
        self.layout.setSpacing(6)

        self._placeholder = QLabel("还没有贴纸\n输入描述来生成吧")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {INK_MUTED_48}; {font_css('caption')} background: transparent; border: none; padding: 40px 12px;"
        )
        self._placeholder.setVisible(False)
        self.layout.addWidget(self._placeholder)
        self.layout.addStretch()
        self.setWidget(self.container)

        self._section_headers = {}
        self._section_flows = {}

    def show_placeholder(self, show):
        self._placeholder.setVisible(show)

    def get_section_header(self, section_id):
        return self._section_headers.get(section_id)

    def add_section_header(self, section_id, name, count, group_id=None):
        header = GallerySectionHeader(section_id, name, count, group_id=group_id)
        header.toggled.connect(self._on_section_toggle)
        self.layout.insertWidget(self.layout.count() - 1, header)
        self._section_headers[section_id] = header
        flow_container = QWidget()
        flow_container.setStyleSheet("background: transparent;")
        flow_layout = FlowLayout(flow_container, spacing=4)
        flow_layout.setContentsMargins(0, 2, 0, 6)
        self.layout.insertWidget(self.layout.count() - 1, flow_container)
        self._section_flows[section_id] = (flow_container, flow_layout)

    def add_card(self, card, section_id=None):
        if section_id and section_id in self._section_flows:
            _, flow_layout = self._section_flows[section_id]
            flow_layout.addWidget(card)
        else:
            self.layout.insertWidget(self.layout.count() - 1, card)

    def _on_section_toggle(self, section_id, expanded):
        if section_id in self._section_flows:
            flow_container, _ = self._section_flows[section_id]
            flow_container.setVisible(expanded)

    def clear_cards(self):
        for i in range(self.layout.count() - 1, -1, -1):
            item = self.layout.itemAt(i)
            w = item.widget()
            if w is not None and w is not self._placeholder:
                self.layout.takeAt(i)
                w.deleteLater()
        self._section_headers.clear()
        self._section_flows.clear()


# ── FlowLayout ──

class FlowLayout(QLayout):
    """Wrapping horizontal layout: cards flow left-to-right, wrap when narrow."""

    def __init__(self, parent=None, spacing=4):
        super().__init__(parent)
        self._items = []
        self._h_spacing = spacing
        self._v_spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), apply_geometry=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, apply_geometry=True)

    def sizeHint(self):
        h = self.heightForWidth(200)
        return QSize(200, h)

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, apply_geometry):
        margins = self.contentsMargins()
        x = rect.x() + margins.left()
        y = rect.y() + margins.top()
        line_height = 0
        max_width = rect.width() - margins.left() - margins.right()

        for item in self._items:
            size_hint = item.sizeHint()
            w = min(size_hint.width(), max_width)
            h = size_hint.height()

            if x + w > rect.x() + margins.left() + max_width and line_height > 0:
                x = rect.x() + margins.left()
                y += line_height + self._v_spacing
                line_height = 0

            if apply_geometry:
                item.setGeometry(QRect(x, y, w, h))

            x += w + self._h_spacing
            line_height = max(line_height, h)

        return y + line_height - rect.y() + margins.bottom()


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
