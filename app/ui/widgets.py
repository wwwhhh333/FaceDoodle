from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QScrollArea,
                             QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
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
    clicked = pyqtSignal(str)  # sticker_id

    def __init__(self, sticker_id, thumb_bgra, prompt, is_favorite=False, parent=None, is_animated=False):
        super().__init__(parent)
        self.sticker_id = sticker_id
        self._selected = False
        self._active = False
        self._fav = is_favorite
        self._animated = is_animated
        self.setFixedSize(100, 120)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(96, 96)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet(
            f"background: {DIVIDER_SOFT}; border: 2px solid {HAIRLINE}; border-radius: {ROUNDED['xs']};"
        )
        if thumb_bgra is not None:
            pix = _bgra_to_qpixmap(thumb_bgra)
            if pix is not None:
                self.thumb_label.setPixmap(
                    pix.scaled(92, 92, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        layout.addWidget(self.thumb_label, alignment=Qt.AlignCenter)

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
        self.name_label.setAlignment(Qt.AlignCenter)
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
        if event.button() == Qt.LeftButton:
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
        self.setCursor(Qt.PointingHandCursor)
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


class GalleryScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(210)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {INK_MUTED_48}; {font_css('caption')} background: transparent; border: none; padding: 40px 12px;"
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
