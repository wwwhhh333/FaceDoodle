"""Active stickers panel — shows which stickers are on the face."""

from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QScrollArea,
                             QHBoxLayout, QVBoxLayout)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal

from app.ui.widgets import _bgra_to_qpixmap
from app.ui.theme import (PRIMARY, CANVAS, PARCHMENT, INK, INK_MUTED_48,
                          HAIRLINE, DIVIDER_SOFT, DESTRUCTIVE,
                          font_css, ROUNDED, rgba)


class ActiveStickerCard(QWidget):
    """Small card representing a sticker currently on face."""
    clicked = Signal(str)       # instance_id — select as edit target
    removed = Signal(str)       # instance_id — remove from face

    def __init__(self, instance_id, thumb_bgra, region_label, parent=None):
        super().__init__(parent)
        self.instance_id = instance_id
        self._is_edit_target = False
        self.setFixedSize(84, 100)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(72, 72)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet(
            f"background: {DIVIDER_SOFT}; border: 2px solid {HAIRLINE}; border-radius: {ROUNDED['xs']};"
        )
        if thumb_bgra is not None:
            pix = _bgra_to_qpixmap(thumb_bgra)
            if pix is not None:
                self.thumb_label.setPixmap(
                    pix.scaled(68, 68, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Region label
        self.region_label = QLabel(region_label)
        self.region_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.region_label.setStyleSheet(
            f"color: {INK_MUTED_48}; {font_css('micro-legal')} background: transparent; border: none;"
        )
        layout.addWidget(self.region_label)

        # Remove button
        self._remove_btn = QPushButton("×", self)
        self._remove_btn.setFixedSize(16, 16)
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.setStyleSheet(
            f"QPushButton {{ background: {rgba(INK, 0.5)}; color: {CANVAS}; border: none; "
            f"font-size: 14px; font-weight: bold; border-radius: {ROUNDED['full']}; }}"
            f"QPushButton:hover {{ background: {DESTRUCTIVE}; }}"
        )
        self._remove_btn.move(66, 0)
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self.instance_id))

    def set_edit_target(self, is_target):
        self._is_edit_target = is_target
        if is_target:
            self.thumb_label.setStyleSheet(
                f"background: {DIVIDER_SOFT}; border: 3px solid {PRIMARY}; border-radius: {ROUNDED['xs']};"
            )
        else:
            self.thumb_label.setStyleSheet(
                f"background: {DIVIDER_SOFT}; border: 2px solid {HAIRLINE}; border-radius: {ROUNDED['xs']};"
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            if self._remove_btn.geometry().contains(pos):
                return
            self.clicked.emit(self.instance_id)


class ActiveStickersPanel(QWidget):
    """Vertical scrollable panel showing stickers currently on face."""

    select_edit_target = Signal(str)  # instance_id
    remove_sticker = Signal(str)       # instance_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(96)
        self.setStyleSheet(f"background: {PARCHMENT}; border: none;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {rgba(INK, 0.15)}; min-height: 20px; border-radius: 3px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._card_layout = QVBoxLayout(self._container)
        self._card_layout.setContentsMargins(2, 2, 2, 2)
        self._card_layout.setSpacing(4)
        self._card_layout.addStretch()

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

        self._cards = {}  # instance_id -> ActiveStickerCard

    def sync(self, instance_ids, thumbs_info, edit_target_id):
        """Sync panel with current active stickers on face.
        thumbs_info: dict instance_id -> {"thumb": bgra, "region": str}
        """
        current_ids = set(instance_ids)
        existing_ids = set(self._cards.keys())

        # Remove cards no longer on face
        for iid in existing_ids - current_ids:
            card = self._cards.pop(iid)
            self._card_layout.removeWidget(card)
            card.deleteLater()

        # Add/update cards
        for iid in instance_ids:
            info = thumbs_info.get(iid, {})
            if iid in self._cards:
                card = self._cards[iid]
            else:
                card = ActiveStickerCard(
                    iid, info.get("thumb"), info.get("region", ""), self
                )
                card.clicked.connect(lambda checked, iid=iid: self.select_edit_target.emit(iid))
                card.removed.connect(lambda iid=iid: self.remove_sticker.emit(iid))
                self._cards[iid] = card
                self._card_layout.insertWidget(self._card_layout.count() - 1, card)
            card.set_edit_target(iid == edit_target_id)

        self.setVisible(len(self._cards) > 0)

    def clear_all(self):
        for iid, card in list(self._cards.items()):
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self.setVisible(False)
