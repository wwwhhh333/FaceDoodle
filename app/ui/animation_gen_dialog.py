"""Dialog for submitting an AI texture animation generation request.

Stays open during generation to show progress and result.
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QPushButton, QLineEdit, QSpinBox, QLabel,
                              QProgressBar)
from PySide6.QtCore import Qt

from app.utils.config_loader import get_config
from app.ui.theme import (PRIMARY, INK_MUTED_48, INK_MUTED_80,
                          HAIRLINE, ERROR, font_css, label_css)
from app.ui.widgets import StyledButton

_ERROR_STYLESHEET = label_css('caption', ERROR)


class AnimationGenDialog(QDialog):
    def __init__(self, sticker_id, animation_queue, parent=None):
        super().__init__(parent)
        self._sticker_id = sticker_id
        self._animation_queue = animation_queue
        self._running = False
        self.setWindowTitle("AI 纹理动画")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setSpacing(8)

        self._prompt_edit = QLineEdit()
        self._prompt_edit.setPlaceholderText("描述你想要的动态效果，如：猫耳轻轻飘动")
        form.addRow("运动描述:", self._prompt_edit)

        self._duration_spin = QSpinBox()
        self._duration_spin.setRange(1, 5)
        self._duration_spin.setValue(2)
        self._duration_spin.setSuffix(" 秒")
        self._duration_spin.valueChanged.connect(self._update_frame_label)
        form.addRow("时长:", self._duration_spin)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(4, 15)
        self._fps_spin.setValue(8)
        self._fps_spin.valueChanged.connect(self._update_frame_label)
        form.addRow("FPS:", self._fps_spin)

        self._frame_label = QLabel("16 帧")
        self._frame_label.setStyleSheet(f"color: {INK_MUTED_48};")
        form.addRow("总帧数:", self._frame_label)

        cfg = get_config()
        presets = cfg.get("style", {}).get("presets", {})
        selected = cfg.get("style", {}).get("selected_preset", "pixel_art")
        preset_name = presets.get(selected, {}).get("name", selected)
        style_label = QLabel(f"风格: {preset_name}")
        style_label.setStyleSheet(f"color: {INK_MUTED_48}; {font_css('fine-print')}")
        form.addRow("", style_label)

        layout.addLayout(form)

        layout.addSpacing(12)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                border: none; background: {HAIRLINE}; height: 4px; border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {PRIMARY}; border-radius: 2px;
            }}
        """)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {INK_MUTED_80}; {font_css('caption')}")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addSpacing(12)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._cancel_btn = StyledButton("取消", "utility")
        self._cancel_btn.clicked.connect(self._on_close)
        btn_layout.addWidget(self._cancel_btn)

        self._gen_btn = StyledButton("生成动画", "primary")
        self._gen_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._gen_btn)

        layout.addLayout(btn_layout)

    def _update_frame_label(self):
        total = self._duration_spin.value() * self._fps_spin.value()
        self._frame_label.setText(f"{total} 帧")

    def _on_generate(self):
        if self._running:
            return
        text = self._prompt_edit.text().strip()
        if not text:
            self._status_label.setText("请输入运动描述")
            self._status_label.setStyleSheet(_ERROR_STYLESHEET)
            return

        self._running = True
        self._gen_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("正在生成...")
        self._status_label.setStyleSheet(f"color: {INK_MUTED_80}; {font_css('caption')}")

        from app.core.protocol import AnimGenTexture
        self._animation_queue.put(AnimGenTexture(
            sticker_id=self._sticker_id,
            motion_prompt=text,
            frame_count=self._duration_spin.value() * self._fps_spin.value(),
            fps=self._fps_spin.value(),
        ))

    def _on_close(self):
        if self._running:
            # Don't close during generation — let cancel act as future cancel
            return
        self.reject()

    def set_progress(self, value):
        self._progress.setVisible(True)
        self._progress.setValue(int(value * 100))

    def on_done(self, error=None):
        self._running = False
        self._gen_btn.setEnabled(False)
        self._cancel_btn.setText("关闭")
        if error:
            self._status_label.setText(f"失败: {error}")
            self._status_label.setStyleSheet(_ERROR_STYLESHEET)
            self._gen_btn.setEnabled(True)
            self._gen_btn.setText("重试")
            self._running = False
            self._cancel_btn.setText("关闭")
        else:
            self._status_label.setText("生成完成！")
            self._status_label.setStyleSheet(f"color: {PRIMARY}; {font_css('caption')}")
            self._progress.setValue(100)
            self._cancel_btn.setText("关闭")
            self._cancel_btn.clicked.disconnect()
            self._cancel_btn.clicked.connect(self.accept)
