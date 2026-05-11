"""Dialog for submitting an AI texture animation generation request."""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QPushButton, QLineEdit, QSpinBox, QLabel,
                              QProgressBar)
from PyQt5.QtCore import Qt

from app.utils.config_loader import get_config


class AnimationGenDialog(QDialog):
    def __init__(self, sticker_id, parent=None):
        super().__init__(parent)
        self._sticker_id = sticker_id
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
        self._frame_label.setStyleSheet("color: #aaa;")
        form.addRow("总帧数:", self._frame_label)

        cfg = get_config()
        presets = cfg.get("style", {}).get("presets", {})
        selected = cfg.get("style", {}).get("selected_preset", "flat_vector")
        preset_name = presets.get(selected, {}).get("name", selected)
        style_label = QLabel(f"风格: {preset_name}")
        style_label.setStyleSheet("color: #aaa; font-size: 11px;")
        form.addRow("", style_label)

        layout.addLayout(form)

        layout.addSpacing(12)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888; font-size: 12px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addSpacing(12)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)

        self._gen_btn = QPushButton("生成动画")
        self._gen_btn.setStyleSheet(
            "QPushButton { background: #7c3aed; color: white; padding: 8px 20px; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #a855f7; }"
        )
        self._gen_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._gen_btn)

        layout.addLayout(btn_layout)

    def _update_frame_label(self):
        total = self._duration_spin.value() * self._fps_spin.value()
        self._frame_label.setText(f"{total} 帧")

    def _on_generate(self):
        if not self._prompt_edit.text().strip():
            self._status_label.setText("请输入运动描述")
            return
        self.accept()

    def set_progress(self, value):
        self._progress.setVisible(True)
        self._progress.setValue(int(value * 100))
        if value >= 1.0:
            self._status_label.setText("生成完成！")
            self._gen_btn.setEnabled(False)
            self._cancel_btn.setText("关闭")

    def set_error(self, error_text):
        self._status_label.setText(f"错误: {error_text}")
        self._status_label.setStyleSheet("color: #e53e3e; font-size: 12px;")
        self._gen_btn.setEnabled(True)

    def result(self):
        return {
            "sticker_id": self._sticker_id,
            "motion_prompt": self._prompt_edit.text().strip(),
            "frame_count": self._duration_spin.value() * self._fps_spin.value(),
            "fps": self._fps_spin.value(),
        }
