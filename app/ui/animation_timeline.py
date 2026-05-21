"""Animation timeline editor widget — keyframe track, property panel, export dialog."""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QDoubleSpinBox, QComboBox, QDialog,
                             QFormLayout, QSpinBox, QLineEdit, QFileDialog,
                             QCheckBox)
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PySide6.QtCore import Qt, QPointF, Signal

from app.core.protocol import (
    AnimPlay, AnimPause, AnimStop, AnimSetLoop, AnimSeek,
    AnimAddKeyframe, AnimRemoveKeyframe, AnimUpdateKeyframe, AnimExport,
)
from app.core.animation import EASING_FUNCTIONS
from app.ui.theme import (PRIMARY, CANVAS, INK, SURFACE_TILE_1,
                          font_css, ROUNDED, rgba, transport_button_style)
from app.ui.widgets import StyledButton

EASING_OPTIONS = list(EASING_FUNCTIONS.keys())

TRACK_HEIGHT = 50
KEYFRAME_SIZE = 10
PLAYHEAD_WIDTH = 2
TICK_HEIGHT = 8


class TimeAxisTrack(QWidget):
    """Custom-painted timeline track with draggable keyframes and playhead."""

    time_clicked = Signal(float)
    keyframe_selected = Signal(int)   # index
    keyframe_dragged = Signal(int, float)  # index, new_time
    keyframe_deselected = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(TRACK_HEIGHT + 24)
        self.setMouseTracking(True)
        self._duration = 2.0
        self._playhead = 0.0
        self._keyframes = []      # list of (time,)
        self._selected_idx = -1
        self._dragging = False
        self._hover_idx = -1

    def set_state(self, duration, playhead, keyframes, selected_idx=-1):
        self._duration = max(duration, 0.1)
        self._playhead = playhead
        self._keyframes = [(k["time"],) for k in keyframes]
        self._selected_idx = selected_idx
        self.update()

    # ── helpers ──

    def _time_to_x(self, t):
        margin = 30
        w = self.width() - 2 * margin
        return margin + (t / self._duration) * w if self._duration > 0 else margin

    def _x_to_time(self, x):
        margin = 30
        w = self.width() - 2 * margin
        return max(0.0, min(self._duration, ((x - margin) / w) * self._duration)) if w > 0 else 0.0

    def _kf_rect(self, t):
        x = self._time_to_x(t)
        y = TRACK_HEIGHT // 2
        s = KEYFRAME_SIZE // 2
        return QPointF(x, y), s

    # ── paint ──

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor(SURFACE_TILE_1))

        # Time ruler ticks
        pen = QPen(QColor(100, 100, 100))
        p.setPen(pen)
        p.setFont(QFont("Segoe UI Variable", 9))
        y_base = TRACK_HEIGHT
        for i in range(int(self._duration) + 1):
            x = self._time_to_x(float(i))
            p.drawLine(int(x), y_base, int(x), y_base + TICK_HEIGHT)
            p.drawText(int(x) - 10, y_base + TICK_HEIGHT + 12, f"{i}s")

        # Center line
        p.setPen(QPen(QColor(60, 60, 60)))
        p.drawLine(30, TRACK_HEIGHT // 2, self.width() - 30, TRACK_HEIGHT // 2)

        # Keyframe diamonds
        for i, (t,) in enumerate(self._keyframes):
            x = self._time_to_x(t)
            y = TRACK_HEIGHT // 2
            s = KEYFRAME_SIZE // 2
            diamond = QPolygonF([
                QPointF(x, y - s), QPointF(x + s, y),
                QPointF(x, y + s), QPointF(x - s, y),
            ])
            if i == self._selected_idx:
                p.setBrush(QColor(255, 180, 50))
                p.setPen(QPen(QColor(255, 200, 100), 2))
            elif i == self._hover_idx:
                p.setBrush(QColor(200, 140, 30))
                p.setPen(QPen(QColor(220, 160, 60), 1))
            else:
                p.setBrush(QColor(100, 180, 255))
                p.setPen(QPen(QColor(140, 200, 255), 1))
            p.drawPolygon(diamond)

        # Playhead
        ph_x = self._time_to_x(self._playhead)
        p.setPen(QPen(QColor(255, 80, 80), PLAYHEAD_WIDTH))
        p.drawLine(int(ph_x), 4, int(ph_x), TRACK_HEIGHT + TICK_HEIGHT)

    # ── mouse ──

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            px = event.pos().x()
            py = event.pos().y()
            # Check keyframe hit
            for i, (t,) in enumerate(self._keyframes):
                cx, s = self._kf_rect(t)
                if abs(px - cx.x()) < s + 3 and abs(py - cx.y()) < s + 3:
                    self._selected_idx = i
                    self._dragging = True
                    self.keyframe_selected.emit(i)
                    self.update()
                    return
            # Click on track background → move playhead
            self._selected_idx = -1
            self.keyframe_deselected.emit()
            t = self._x_to_time(px)
            self.time_clicked.emit(t)
            self.update()

    def mouseMoveEvent(self, event):
        px = event.pos().x()
        py = event.pos().y()
        if self._dragging and self._selected_idx >= 0:
            new_t = self._x_to_time(px)
            self.keyframe_dragged.emit(self._selected_idx, new_t)
            self.update()
            return
        # Hover detection
        self._hover_idx = -1
        for i, (t,) in enumerate(self._keyframes):
            cx, s = self._kf_rect(t)
            if abs(px - cx.x()) < s + 3 and abs(py - cx.y()) < s + 3:
                self._hover_idx = i
                break
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False


class AnimationTimeline(QWidget):
    """Complete timeline widget: toolbar + track + property panel."""

    def __init__(self, animation_queue):
        super().__init__()
        self._queue = animation_queue
        self._instance_id = ""
        self._duration = 2.0
        self._playing = False
        self._loop = False
        self._playhead = 0.0
        self._keyframes = []        # list of Keyframe dicts
        self._selected_idx = -1
        self._clip_data = None
        self._last_pushed_kf = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # ── toolbar ──
        toolbar = QHBoxLayout()

        _t_style = transport_button_style()

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(32, 28)
        self._play_btn.setStyleSheet(_t_style)
        self._play_btn.clicked.connect(self._on_play)
        toolbar.addWidget(self._play_btn)

        self._stop_btn = QPushButton("⏹")
        self._stop_btn.setFixedSize(32, 28)
        self._stop_btn.setStyleSheet(_t_style)
        self._stop_btn.clicked.connect(self._on_stop)
        toolbar.addWidget(self._stop_btn)

        self._loop_btn = QPushButton("🔁")
        self._loop_btn.setFixedSize(32, 28)
        self._loop_btn.setCheckable(True)
        self._loop_btn.setStyleSheet(_t_style)
        self._loop_btn.clicked.connect(self._on_loop)
        toolbar.addWidget(self._loop_btn)

        toolbar.addSpacing(12)
        self._time_label = QLabel("0.00s / 2.00s")
        self._time_label.setStyleSheet(f"color: {INK}; {font_css('caption')}")
        toolbar.addWidget(self._time_label)

        toolbar.addStretch()
        add_kf_btn = StyledButton("添加关键帧", "utility")
        add_kf_btn.clicked.connect(self._on_add_keyframe)
        toolbar.addWidget(add_kf_btn)

        export_btn = StyledButton("导出", "ghost")
        export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(export_btn)

        layout.addLayout(toolbar)

        # ── track ──
        self._track = TimeAxisTrack()
        self._track.time_clicked.connect(self._on_time_clicked)
        self._track.keyframe_selected.connect(self._on_keyframe_selected)
        self._track.keyframe_dragged.connect(self._on_keyframe_dragged)
        self._track.keyframe_deselected.connect(self._on_keyframe_deselected)
        layout.addWidget(self._track)

        # ── property panel ──
        prop_layout = QHBoxLayout()
        prop_layout.setSpacing(8)

        prop_layout.addWidget(QLabel("offset_x:"))
        self._spin_ox = QDoubleSpinBox()
        self._spin_ox.setRange(-2.0, 2.0)
        self._spin_ox.setDecimals(4)
        self._spin_ox.setSingleStep(0.01)
        self._spin_ox.valueChanged.connect(self._on_property_changed)
        prop_layout.addWidget(self._spin_ox)

        prop_layout.addWidget(QLabel("offset_y:"))
        self._spin_oy = QDoubleSpinBox()
        self._spin_oy.setRange(-2.0, 2.0)
        self._spin_oy.setDecimals(4)
        self._spin_oy.setSingleStep(0.01)
        self._spin_oy.valueChanged.connect(self._on_property_changed)
        prop_layout.addWidget(self._spin_oy)

        prop_layout.addWidget(QLabel("rotation:"))
        self._spin_rot = QDoubleSpinBox()
        self._spin_rot.setRange(-180.0, 180.0)
        self._spin_rot.setDecimals(1)
        self._spin_rot.setSingleStep(5.0)
        self._spin_rot.valueChanged.connect(self._on_property_changed)
        prop_layout.addWidget(self._spin_rot)

        prop_layout.addWidget(QLabel("scale:"))
        self._spin_scale = QDoubleSpinBox()
        self._spin_scale.setRange(0.05, 5.0)
        self._spin_scale.setDecimals(2)
        self._spin_scale.setSingleStep(0.1)
        self._spin_scale.setValue(1.0)
        self._spin_scale.valueChanged.connect(self._on_property_changed)
        prop_layout.addWidget(self._spin_scale)

        prop_layout.addWidget(QLabel("opacity:"))
        self._spin_opacity = QDoubleSpinBox()
        self._spin_opacity.setRange(0.0, 1.0)
        self._spin_opacity.setDecimals(2)
        self._spin_opacity.setSingleStep(0.1)
        self._spin_opacity.setValue(1.0)
        self._spin_opacity.valueChanged.connect(self._on_property_changed)
        prop_layout.addWidget(self._spin_opacity)

        prop_layout.addWidget(QLabel("easing:"))
        self._easing_combo = QComboBox()
        self._easing_combo.addItems(EASING_OPTIONS)
        self._easing_combo.currentTextChanged.connect(self._on_property_changed)
        prop_layout.addWidget(self._easing_combo)

        del_kf_btn = StyledButton("删除关键帧", "ghost-destructive")
        del_kf_btn.clicked.connect(self._on_delete_keyframe)
        prop_layout.addWidget(del_kf_btn)

        prop_layout.addStretch()
        layout.addLayout(prop_layout)

        self._enable_property_panel(False)

    # ── public API ──

    def set_instance(self, instance_id):
        self._instance_id = instance_id
        self._last_pushed_kf = None

    def update_clip(self, clip_data):
        """Called when AnimClipUpdated arrives from consumer."""
        self._clip_data = clip_data
        self._last_pushed_kf = None
        self._duration = clip_data.get("duration", 2.0)
        self._loop = clip_data.get("loop", False)
        self._keyframes = clip_data.get("keyframes", [])
        self._loop_btn.setChecked(self._loop)
        self._update_display()

    def update_playback(self, playing, time_val, duration):
        self._playing = playing
        self._playhead = time_val
        self._duration = max(duration, self._duration)
        self._play_btn.setText("⏸" if playing else "▶")
        self._update_display()

    # ── internal ──

    def _update_display(self):
        self._time_label.setText(f"{self._playhead:.2f}s / {self._duration:.2f}s")
        self._track.set_state(self._duration, self._playhead, self._keyframes, self._selected_idx)
        if self._selected_idx >= 0 and self._selected_idx < len(self._keyframes):
            self._load_keyframe_to_panel(self._keyframes[self._selected_idx])

    def _load_keyframe_to_panel(self, kf):
        for w in (self._spin_ox, self._spin_oy, self._spin_rot,
                  self._spin_scale, self._spin_opacity):
            w.blockSignals(True)
        self._easing_combo.blockSignals(True)
        self._spin_ox.setValue(kf.get("offset_x", 0.0))
        self._spin_oy.setValue(kf.get("offset_y", 0.0))
        self._spin_rot.setValue(kf.get("rotation", 0.0))
        self._spin_scale.setValue(kf.get("scale_mult", 1.0))
        self._spin_opacity.setValue(kf.get("opacity", 1.0))
        idx = EASING_OPTIONS.index(kf.get("easing", "linear"))
        self._easing_combo.setCurrentIndex(idx if idx >= 0 else 0)
        for w in (self._spin_ox, self._spin_oy, self._spin_rot,
                  self._spin_scale, self._spin_opacity):
            w.blockSignals(False)
        self._easing_combo.blockSignals(False)

    def _enable_property_panel(self, enabled):
        for w in (self._spin_ox, self._spin_oy, self._spin_rot,
                  self._spin_scale, self._spin_opacity, self._easing_combo):
            w.setEnabled(enabled)

    # ── slots ──

    def _on_play(self):
        if not self._instance_id:
            return
        if self._playing:
            self._queue.put(AnimPause(instance_id=self._instance_id))
        else:
            self._queue.put(AnimPlay(instance_id=self._instance_id))

    def _on_stop(self):
        if not self._instance_id:
            return
        self._queue.put(AnimStop(instance_id=self._instance_id))

    def _on_loop(self, checked):
        if not self._instance_id:
            return
        self._queue.put(AnimSetLoop(instance_id=self._instance_id, loop=checked))

    def _on_time_clicked(self, t):
        if not self._instance_id:
            return
        self._playhead = t
        self._queue.put(AnimSeek(instance_id=self._instance_id, time=t))

    def _on_keyframe_selected(self, idx):
        self._selected_idx = idx
        self._enable_property_panel(True)
        self._update_display()

    def _on_keyframe_deselected(self):
        self._selected_idx = -1
        self._enable_property_panel(False)
        self._update_display()

    def _push_keyframe_update(self, idx):
        if idx < 0 or idx >= len(self._keyframes):
            return
        kf = self._keyframes[idx]
        # Skip if unchanged from last push (rate-limits 60 Hz drag / spinbox events)
        key = (idx, kf.get("time"), kf.get("offset_x"), kf.get("offset_y"),
               kf.get("rotation"), kf.get("scale_mult"), kf.get("opacity"), kf.get("easing"))
        if self._last_pushed_kf == key:
            return
        self._last_pushed_kf = key
        self._queue.put(AnimUpdateKeyframe(
            instance_id=self._instance_id,
            keyframe_index=idx,
            time=kf.get("time", 0.0),
            offset_x=kf.get("offset_x", 0.0),
            offset_y=kf.get("offset_y", 0.0),
            rotation=kf.get("rotation", 0.0),
            scale_mult=kf.get("scale_mult", 1.0),
            opacity=kf.get("opacity", 1.0),
            easing=kf.get("easing", "linear"),
        ))

    def _on_keyframe_dragged(self, idx, new_time):
        if idx >= len(self._keyframes):
            return
        self._keyframes[idx]["time"] = new_time
        self._push_keyframe_update(idx)
        self._update_display()

    def _on_add_keyframe(self):
        if not self._instance_id:
            return
        easing = self._easing_combo.currentText()
        self._queue.put(AnimAddKeyframe(
            instance_id=self._instance_id,
            time=self._playhead,
            easing=easing,
        ))

    def _on_delete_keyframe(self):
        if not self._instance_id or self._selected_idx < 0:
            return
        self._queue.put(AnimRemoveKeyframe(
            instance_id=self._instance_id,
            keyframe_index=self._selected_idx,
        ))

    def _on_property_changed(self):
        if self._selected_idx < 0 or self._selected_idx >= len(self._keyframes):
            return
        kf = self._keyframes[self._selected_idx]
        kf["offset_x"] = self._spin_ox.value()
        kf["offset_y"] = self._spin_oy.value()
        kf["rotation"] = self._spin_rot.value()
        kf["scale_mult"] = self._spin_scale.value()
        kf["opacity"] = self._spin_opacity.value()
        kf["easing"] = self._easing_combo.currentText()
        self._push_keyframe_update(self._selected_idx)

    def _on_export(self):
        if not self._instance_id:
            return
        dlg = ExportDialog(self)
        if dlg.exec() == QDialog.Accepted:
            fmt, fps, path = dlg.result()
            self._queue.put(AnimExport(
                instance_id=self._instance_id,
                format=fmt, fps=fps, output_path=path,
            ))


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出动画")
        self.setMinimumWidth(360)

        layout = QFormLayout(self)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["mp4", "gif"])
        layout.addRow("格式:", self._format_combo)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(24)
        layout.addRow("FPS:", self._fps_spin)

        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("选择保存路径...")
        path_layout.addWidget(self._path_edit)
        browse_btn = StyledButton("浏览", "utility")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)
        layout.addRow("保存到:", path_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = StyledButton("取消", "utility")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        ok_btn = StyledButton("导出", "primary")
        ok_btn.clicked.connect(self._validate)
        btn_layout.addWidget(ok_btn)
        layout.addRow(btn_layout)

    def _browse(self):
        fmt = self._format_combo.currentText()
        filter_str = f"*.{fmt}"
        path, _ = QFileDialog.getSaveFileName(self, "保存动画", "", f"{fmt.upper()} Files ({filter_str})")
        if path:
            self._path_edit.setText(path)

    def _validate(self):
        if not self._path_edit.text().strip():
            return
        self.accept()

    def result(self):
        return (
            self._format_combo.currentText(),
            self._fps_spin.value(),
            self._path_edit.text().strip(),
        )
