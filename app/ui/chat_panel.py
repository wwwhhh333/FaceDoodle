"""Compact chat message panel — sits above the input row, shows user/agent dialog."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel
from PyQt5.QtCore import Qt, QTimer

from app.ui.theme import PARCHMENT, INK, PRIMARY, HAIRLINE, font_css, rgba


class ChatMessagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages = []
        self.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setMaximumHeight(150)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
        )

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(16, 6, 16, 2)
        self._container_layout.setSpacing(3)
        self._container_layout.addStretch()

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

    def add_user_message(self, text):
        bubble = _make_bubble(text, is_user=True)
        self._insert_bubble(bubble)

    def add_agent_message(self, text, status="done"):
        icon_map = {"generating": "\U0001f504 ", "done": "✓ ", "failed": "✗ ", "ask": "❓ "}
        icon = icon_map.get(status, "")
        bubble = _make_bubble(icon + text if icon else text, is_user=False)
        self._insert_bubble(bubble)

    def _insert_bubble(self, bubble):
        self._container_layout.takeAt(self._container_layout.count() - 1)
        self._container_layout.addWidget(bubble)
        self._container_layout.addStretch()
        self._messages.append(bubble)

        if len(self._messages) > 20:
            oldest = self._messages.pop(0)
            oldest.deleteLater()

        self.setVisible(True)
        QTimer.singleShot(30, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self):
        while self._messages:
            msg = self._messages.pop(0)
            msg.deleteLater()
        self.setVisible(False)


def _make_bubble(text, is_user):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    label.setCursor(Qt.PointingHandCursor)

    if is_user:
        bubble_css = f"""
            QLabel {{
                background: {rgba(PRIMARY, 0.1)};
                color: {INK};
                border: 1px solid {rgba(PRIMARY, 0.2)};
                border-radius: 10px;
                padding: 5px 12px;
                {font_css("caption")}
            }}
        """
    else:
        bubble_css = f"""
            QLabel {{
                background: {PARCHMENT};
                color: {INK};
                border: 1px solid {HAIRLINE};
                border-radius: 10px;
                padding: 5px 12px;
                {font_css("caption")}
            }}
        """

    label.setStyleSheet(bubble_css)
    return label
