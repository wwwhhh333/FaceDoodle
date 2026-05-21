"""Dialog for creating, editing, and deleting style presets.

The dialog presents a two-pane layout:
- Left:  QListWidget listing all presets (built-in marked "(内置)")
- Right: QFormLayout for editing the selected preset's fields
- Bottom: Save / Reset to Defaults / Delete / Close buttons
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QListWidget, QListWidgetItem, QLineEdit, QDoubleSpinBox,
    QPushButton, QLabel, QMessageBox,
)
from PySide6.QtCore import Qt
from app.ui.theme import (
    PRIMARY, INK, INK_MUTED_48, INK_MUTED_80, CANVAS, PARCHMENT,
    HAIRLINE, DESTRUCTIVE, font_css, global_stylesheet,
)
from app.ui.widgets import StyledButton
from app.utils.config_loader import (
    get_config, is_builtin_preset,
    add_preset, update_preset, delete_preset, reset_preset,
)


class StylePresetManagerDialog(QDialog):
    """Dialog to manage style presets (prompt + LoRA combinations)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_key = None
        self._setup_ui()
        self._populate_list()

    # ── UI setup ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("管理风格预设")
        self.setMinimumSize(620, 480)
        self.setStyleSheet(global_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # ── Top: Add button ──
        self.add_btn = QPushButton("+ 新增预设")
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.setStyleSheet(
            f"QPushButton {{ background: {PRIMARY}; color: #fff; border: none; "
            f"border-radius: 6px; padding: 6px 14px; font-weight: 600; {font_css('caption')} }}"
            f"QPushButton:hover {{ background: #0071e3; }}"
        )
        self.add_btn.clicked.connect(self._on_add)
        layout.addWidget(self.add_btn)

        # ── Body: two-pane ──
        body = QHBoxLayout()
        body.setSpacing(12)

        # Left pane: list
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(180)
        self.list_widget.setMaximumWidth(260)
        self.list_widget.setStyleSheet(
            f"QListWidget {{ background: {PARCHMENT}; border: 1px solid {HAIRLINE}; "
            f"border-radius: 6px; padding: 4px; {font_css('caption')} }}"
            f"QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}"
            f"QListWidget::item:selected {{ background: {PRIMARY}; color: #fff; }}"
        )
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        body.addWidget(self.list_widget)

        # Right pane: edit form
        self._build_form(body)

        layout.addLayout(body, 1)

        # ── Bottom buttons ──
        self._build_buttons(layout)

    def _build_form(self, parent_layout):
        grp = QGroupBox("预设详情")
        grp.setStyleSheet(
            f"QGroupBox {{ font-weight: 600; {font_css('caption')} border: 1px solid {HAIRLINE}; "
            f"border-radius: 6px; margin-top: 10px; padding: 14px 10px 10px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}"
        )
        form = QFormLayout(grp)
        form.setSpacing(8)
        form.setContentsMargins(4, 8, 4, 4)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("像素风格")
        form.addRow("显示名称", self.name_edit)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("art of {prompt}, ...")
        self.prompt_edit.setMinimumWidth(280)
        form.addRow("提示词模板", self.prompt_edit)

        tip_label = QLabel('使用 {prompt} 作为用户输入占位符')
        tip_label.setStyleSheet(f"color: {INK_MUTED_48}; {font_css('fine-print')}")
        form.addRow("", tip_label)

        self.lora_edit = QLineEdit()
        self.lora_edit.setPlaceholderText("留空则禁用 LoRA")
        form.addRow("LoRA 文件名", self.lora_edit)

        self.model_spin = self._make_spinbox()
        form.addRow("Model 强度", self.model_spin)

        self.clip_spin = self._make_spinbox()
        form.addRow("CLIP 强度", self.clip_spin)

        parent_layout.addWidget(grp, 1)

    def _build_buttons(self, parent_layout):
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.save_btn = QPushButton("保存更改")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setStyleSheet(
            f"QPushButton {{ background: {PRIMARY}; color: #fff; border: none; "
            f"border-radius: 6px; padding: 6px 16px; font-weight: 600; {font_css('caption')} }}"
            f"QPushButton:hover {{ background: #0071e3; }}"
        )
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)

        self.reset_btn = QPushButton("重置为默认")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setStyleSheet(
            f"QPushButton {{ color: {INK}; background: {PARCHMENT}; border: 1px solid {HAIRLINE}; "
            f"border-radius: 6px; padding: 6px 12px; {font_css('caption')} }}"
        )
        self.reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self.reset_btn)

        self.delete_btn = QPushButton("删除预设")
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setStyleSheet(
            f"QPushButton {{ color: #dc2626; background: rgba(220,38,38,0.08); "
            f"border: 1px solid rgba(220,38,38,0.25); border-radius: 6px; "
            f"padding: 6px 12px; {font_css('caption')} }}"
        )
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)

        btn_row.addStretch()

        close_btn = StyledButton("关闭", "utility")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        parent_layout.addLayout(btn_row)

    @staticmethod
    def _make_spinbox(default=0.0):
        s = QDoubleSpinBox()
        s.setRange(0.0, 3.0)
        s.setSingleStep(0.1)
        s.setDecimals(2)
        s.setValue(default)
        return s

    # ── list population ──

    def _populate_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        cfg = get_config()
        presets = cfg.get("style", {}).get("presets", {})
        for key, p in presets.items():
            name = p.get("name", key)
            if is_builtin_preset(key):
                name += "  (内置)"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, key)
            self.list_widget.addItem(item)

        self.list_widget.blockSignals(False)

        # Select first item
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        else:
            self._clear_form()

    # ── selection handling ──

    def _on_selection_changed(self, row):
        if row < 0:
            self._clear_form()
            return
        item = self.list_widget.item(row)
        if item is None:
            return
        key = item.data(Qt.UserRole)
        self._current_key = key

        cfg = get_config()
        p = cfg.get("style", {}).get("presets", {}).get(key, {})
        self.name_edit.setText(p.get("name", ""))
        self.prompt_edit.setText(p.get("positive_prefix", ""))
        self.lora_edit.setText(p.get("lora_name", ""))
        self.model_spin.setValue(float(p.get("lora_strength_model", 0.0)))
        self.clip_spin.setValue(float(p.get("lora_strength_clip", 0.0)))

        builtin = is_builtin_preset(key)
        self.reset_btn.setEnabled(builtin)
        self.reset_btn.setVisible(True)
        self.delete_btn.setEnabled(not builtin)
        self.save_btn.setEnabled(True)

    def _clear_form(self):
        self._current_key = None
        self.name_edit.clear()
        self.prompt_edit.clear()
        self.lora_edit.clear()
        self.model_spin.setValue(0.0)
        self.clip_spin.setValue(0.0)
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)

    # ── actions ──

    def _on_add(self):
        """Add a new blank preset, select it in the list."""
        key = add_preset({
            "name": "新预设",
            "positive_prefix": "art of {prompt}, clean design",
            "lora_name": "",
        })
        if key is None:
            QMessageBox.warning(self, "保存失败", "无法保存新预设，请检查 config.json 是否可写。")
            return
        self._populate_list()
        # Select the new item
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole) == key:
                self.list_widget.setCurrentRow(i)
                break

    def _on_save(self):
        if not self._current_key:
            return
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "验证失败", "显示名称不能为空。")
            self.name_edit.setFocus()
            return

        data = {
            "name": name,
            "positive_prefix": self.prompt_edit.text(),
            "lora_name": self.lora_edit.text().strip(),
            "lora_strength_model": self.model_spin.value(),
            "lora_strength_clip": self.clip_spin.value(),
        }
        if update_preset(self._current_key, data):
            # Refresh list item text
            item = self._find_item(self._current_key)
            if item:
                suffix = "  (内置)" if is_builtin_preset(self._current_key) else ""
                item.setText(name + suffix)
        else:
            QMessageBox.warning(self, "保存失败", "无法保存预设，请检查 config.json 是否可写。")

    def _on_delete(self):
        if not self._current_key or is_builtin_preset(self._current_key):
            return
        name = self.name_edit.text().strip() or self._current_key
        ret = QMessageBox.question(
            self, "确认删除", f"确定删除预设「{name}」？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        delete_preset(self._current_key)
        self._populate_list()

    def _on_reset(self):
        if not self._current_key or not is_builtin_preset(self._current_key):
            return
        reset_preset(self._current_key)
        self._on_selection_changed(self.list_widget.currentRow())

    # ── helpers ──

    def _find_item(self, key):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole) == key:
                return item
        return None
