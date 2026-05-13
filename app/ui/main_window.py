import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QShortcut, QMessageBox,
                             QFileDialog, QDialog, QComboBox, QSlider, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QPainter, QColor
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QEvent

from app.ui.widgets import (ThumbnailCard, StyledButton, GradientBar,
                            GalleryScrollArea, REGION_OPTIONS, PRESET_COLORS)
from app.ui.theme import (PRIMARY, CANVAS, PARCHMENT, INK,
                          INK_MUTED_48, INK_MUTED_80, HAIRLINE,
                          DESTRUCTIVE, font_css, ROUNDED, rgba,
                          global_stylesheet)
from app.ui.drawing_widgets import DrawingDialog
from app.ui.sticker_panel import ActiveStickersPanel
from app.ui.animation_timeline import AnimationTimeline
from app.ui.animation_gen_dialog import AnimationGenDialog
from app.ui.chat_panel import ChatMessagePanel
from app.utils import storage
from app.utils.config_loader import get_config, get_style_preset_items
from app.core.brush import load_brush_config
from app.core.templates import load_templates
from app.core.protocol import (
    AdjMove, AdjRotate, AdjScale, AdjReset,
    GalAddSticker, GalRemoveSticker, GalSelectEditTarget,
    GalLoadTemplate, GalLoadSticker, GalMergeGroup,
    DrawToggleDrawMode, DrawSetRegion, DrawSetBrush, DrawToggleEraser,
    DrawSetBrushType, DrawSetPressureMode, DrawSetSpacing, DrawSetScatter,
    DrawUndo, DrawClear, DrawStrokeBegin, DrawStrokePoint, DrawStrokeEnd, DrawSave,
    DispStickerSaved, DispGenerationFailed, DispActiveStickersChanged,
    DispGenProgress, DispAgentMessage, DispAgentQuestion,
    AnimExportProgress, AnimClipUpdated, AnimPlaybackState,
    AnimGenTexture, AnimGenProgress,
)


class VideoUpdateThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    sticker_saved_signal = pyqtSignal(str)
    generation_failed_signal = pyqtSignal(str)
    active_stickers_signal = pyqtSignal(object)
    gen_progress_signal = pyqtSignal(object)
    agent_message_signal = pyqtSignal(str)
    agent_question_signal = pyqtSignal(str)
    anim_export_progress_signal = pyqtSignal(object)
    anim_clip_updated_signal = pyqtSignal(object)
    anim_playback_state_signal = pyqtSignal(object)
    anim_gen_progress_signal = pyqtSignal(object)

    def __init__(self, display_queue, gallery_queue):
        super().__init__()
        self.display_queue = display_queue
        self.gallery_queue = gallery_queue
        self._run_flag = True

    def run(self):
        while self._run_flag:
            try:
                item = self.display_queue.get(block=True, timeout=0.1)
                if isinstance(item, np.ndarray):
                    self.change_pixmap_signal.emit(item)
                elif isinstance(item, DispStickerSaved):
                    self.sticker_saved_signal.emit(item.sticker_id)
                elif isinstance(item, DispGenerationFailed):
                    self.generation_failed_signal.emit(item.error)
                elif isinstance(item, DispActiveStickersChanged):
                    self.active_stickers_signal.emit(item)
                elif isinstance(item, DispGenProgress):
                    self.gen_progress_signal.emit(item)
                elif isinstance(item, DispAgentMessage):
                    self.agent_message_signal.emit(item.text)
                elif isinstance(item, DispAgentQuestion):
                    self.agent_question_signal.emit(item.text)
                elif isinstance(item, AnimExportProgress):
                    self.anim_export_progress_signal.emit(item)
                elif isinstance(item, AnimClipUpdated):
                    self.anim_clip_updated_signal.emit(item)
                elif isinstance(item, AnimPlaybackState):
                    self.anim_playback_state_signal.emit(item)
                elif isinstance(item, AnimGenProgress):
                    self.anim_gen_progress_signal.emit(item)
            except Exception:
                pass

    def stop(self):
        self._run_flag = False
        self.wait()


class FaceDoodleWindow(QMainWindow):
    def __init__(self, display_queue, command_queue, adjustment_queue, gallery_queue, draw_queue, animation_queue):
        super().__init__()
        self.display_queue = display_queue
        self.command_queue = command_queue
        self.adjustment_queue = adjustment_queue
        self.gallery_queue = gallery_queue
        self.draw_queue = draw_queue
        self.animation_queue = animation_queue

        self._edit_mode = False
        self._mouse_down = False
        self._mouse_button = None
        self._last_mouse_pos = None
        self._frame_size = None
        self._current_sticker_id = None
        self._active_instance_ids = []      # instance_ids currently on face
        self._edit_target_id = None
        self._gallery_selected_ids = set()  # multi-select in gallery
        self._gallery_items = {}
        self._gallery_filter = "stickers"
        self._template_cards = {}

        self._face_draw_mode = False
        self._face_draw_region = "full_face"
        self._face_draw_brush_size = 12
        self._face_draw_brush_color = (0, 0, 0, 255)
        self._face_draw_brush_type = "hard_round"
        self._face_draw_pressure_mode = "both"
        self._face_draw_eraser = False
        self._tablet_in_use = False
        self._face_draw_mouse_down = False
        self._face_draw_stroke_points = []
        self._label_scale = 1.0
        self._label_offset_x = 0.0
        self._label_offset_y = 0.0

        self.setWindowTitle("FaceDoodle AI - 智能贴纸工坊")
        self.resize(1440, 860)

        self._init_stylesheet()
        self._init_ui()
        self._load_gallery()

        self.video_thread = VideoUpdateThread(self.display_queue, self.gallery_queue)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.sticker_saved_signal.connect(self._on_sticker_saved)
        self.video_thread.generation_failed_signal.connect(self._on_generation_failed)
        self.video_thread.active_stickers_signal.connect(self._on_active_stickers_changed)
        self.video_thread.gen_progress_signal.connect(self._on_gen_progress)
        self.video_thread.agent_message_signal.connect(self._on_agent_message)
        self.video_thread.agent_question_signal.connect(self._on_agent_question)
        self.video_thread.anim_clip_updated_signal.connect(self._on_anim_clip_updated)
        self.video_thread.anim_playback_state_signal.connect(self._on_anim_playback_state)
        self.video_thread.anim_export_progress_signal.connect(self._on_anim_export_progress)
        self.video_thread.anim_gen_progress_signal.connect(self._on_anim_gen_progress)
        self.video_thread.start()

    def _init_stylesheet(self):
        self.setStyleSheet(global_stylesheet())

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 1. 顶栏 ──
        top_bar = GradientBar("FaceDoodle AI 贴纸工坊")
        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(36, 36)
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {INK_MUTED_80}; border: none;
                font-size: 18px;
            }}
            QPushButton:hover {{ background: {rgba(INK, 0.08)}; border-radius: {ROUNDED['full']}; }}
        """)
        settings_btn.clicked.connect(self._show_settings)
        top_bar.add_right_widget(settings_btn)
        root.addWidget(top_bar)

        # ── 2. 中间区域: 左侧活动贴纸 + 视频 + 右侧贴纸库 ──
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        # 左侧边栏 — 已添加到面部的贴纸
        left_panel = QWidget()
        left_panel.setFixedWidth(100)
        left_panel.setStyleSheet(f"background: {PARCHMENT}; border: none;")
        lp_layout = QVBoxLayout(left_panel)
        lp_layout.setContentsMargins(4, 8, 4, 8)
        lp_layout.setSpacing(4)

        left_title = QLabel("面部贴纸")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet(
            f"color: {INK_MUTED_48}; {font_css('caption-strong')} background: transparent; border: none; padding-bottom: 4px;"
        )
        lp_layout.addWidget(left_title)

        self.active_panel = ActiveStickersPanel()
        self.active_panel.select_edit_target.connect(
            lambda iid: self.gallery_queue.put(GalSelectEditTarget(instance_id=iid))
        )
        self.active_panel.remove_sticker.connect(
            lambda iid: self.gallery_queue.put(GalRemoveSticker(instance_id=iid))
        )
        lp_layout.addWidget(self.active_panel, stretch=1)
        content_row.addWidget(left_panel)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"background: {CANVAS}; border: none;")
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setMouseTracking(True)
        self.video_label.setTabletTracking(True)
        self.video_label.installEventFilter(self)
        content_row.addWidget(self.video_label, stretch=1)

        # 右侧面板 — 贴纸库
        right_panel = QWidget()
        right_panel.setFixedWidth(210)
        right_panel.setStyleSheet(f"background: {PARCHMENT}; border: none;")
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(8, 12, 8, 8)
        rp_layout.setSpacing(6)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(0)
        self._filter_btns = {}
        for key, label in [("templates", "模板"), ("stickers", "贴纸"), ("favorites", "收藏")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {INK_MUTED_48};
                    border: none; padding: 6px 0; {font_css('caption-strong')}
                }}
                QPushButton:hover {{ color: {INK}; }}
                QPushButton:checked {{ background: {PRIMARY}; color: {CANVAS}; border-radius: {ROUNDED['pill']}; }}
            """)
            btn.clicked.connect(lambda checked, k=key: self._on_filter_changed(k))
            filter_row.addWidget(btn, stretch=1)
            self._filter_btns[key] = btn
        self._filter_btns["stickers"].setChecked(True)
        rp_layout.addLayout(filter_row)

        self.gallery = GalleryScrollArea()
        rp_layout.addWidget(self.gallery, stretch=1)

        self.sticker_count_label = QLabel("")
        self.sticker_count_label.setStyleSheet(
            f"color: {INK_MUTED_48}; {font_css('fine-print')} background: transparent; border: none;"
        )
        rp_layout.addWidget(self.sticker_count_label)

        self.add_to_face_btn = StyledButton("添加到面部", "primary")
        self.add_to_face_btn.clicked.connect(self._on_add_selected_to_face)
        self.add_to_face_btn.setEnabled(False)
        rp_layout.addWidget(self.add_to_face_btn)

        self.ai_anim_btn = StyledButton("AI 动画", "primary")
        self.ai_anim_btn.clicked.connect(self._on_ai_animate)
        self.ai_anim_btn.setEnabled(False)
        rp_layout.addWidget(self.ai_anim_btn)

        content_row.addWidget(right_panel)
        root.addLayout(content_row, stretch=1)

        self.edit_indicator = QLabel("编辑模式: 开启", self)
        self.edit_indicator.setStyleSheet(
            f"color: {PRIMARY}; background: {CANVAS}; {font_css('caption')} "
            f"padding: 6px 14px; border: 1px solid {PRIMARY}; border-radius: {ROUNDED['pill']};"
        )
        self.edit_indicator.setAlignment(Qt.AlignCenter)
        self.edit_indicator.setVisible(False)

        # ── Animation timeline ──
        self.anim_timeline = AnimationTimeline(self.animation_queue)
        self.anim_timeline.setVisible(False)
        root.addWidget(self.anim_timeline)

        # ── Chat message panel ──
        self.chat_panel = ChatMessagePanel()
        self.chat_panel.setStyleSheet(f"background: {CANVAS}; border-top: 1px solid {HAIRLINE};")
        root.addWidget(self.chat_panel)

        # ── 3. 输入区 ──
        input_row = QWidget()
        input_row.setStyleSheet(f"background: {CANVAS}; border-top: 1px solid {HAIRLINE};")
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(16, 12, 16, 12)
        input_layout.setSpacing(12)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("描述你的创意，例如：一副赛博朋克风格的护目镜...")
        self.input_box.returnPressed.connect(self.send_command)
        input_layout.addWidget(self.input_box)

        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(110)
        self.style_combo.setToolTip("选择生成风格预设")
        self.style_combo.setStyleSheet(f"QComboBox {{ {font_css('body')} }}")
        self._populate_style_combo()
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        input_layout.addWidget(self.style_combo)

        self.symmetry_check = QCheckBox("对称模式")
        self.symmetry_check.setToolTip("启用后在提示词中添加对称性关键词")
        self.symmetry_check.setCursor(Qt.PointingHandCursor)
        self.symmetry_check.setStyleSheet(f"QCheckBox {{ {font_css('body')} }}")
        self.symmetry_check.stateChanged.connect(self._on_symmetry_toggled)
        input_layout.addWidget(self.symmetry_check)

        cfg = get_config()
        self.symmetry_check.setChecked(cfg.get("generation", {}).get("symmetry_enabled", False))

        self.send_btn = StyledButton("生成贴纸", "primary")
        self.send_btn.clicked.connect(self.send_command)
        input_layout.addWidget(self.send_btn)

        root.addWidget(input_row)

        # ── 4. 操作按钮区 ──
        action_row = QWidget()
        action_row.setStyleSheet(f"background: {CANVAS}; border-top: 1px solid {HAIRLINE};")
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(16, 8, 16, 10)
        action_layout.setSpacing(10)

        self.merge_btn = StyledButton("合并", "ghost")
        self.merge_btn.setEnabled(False)
        self.merge_btn.clicked.connect(self._on_merge_group)
        action_layout.addWidget(self.merge_btn)

        sep_merge = QWidget()
        sep_merge.setFixedWidth(1)
        sep_merge.setStyleSheet(f"background: {HAIRLINE}; border: none; margin: 0 6px;")
        action_layout.addWidget(sep_merge)

        self.edit_btn = StyledButton("编辑", "checkable")
        self.edit_btn.setCheckable(True)
        self.edit_btn.clicked.connect(self._toggle_edit_mode)
        action_layout.addWidget(self.edit_btn)

        self.face_draw_btn = StyledButton("面部绘制", "checkable")
        self.face_draw_btn.setCheckable(True)
        self.face_draw_btn.clicked.connect(self._toggle_face_draw_mode)
        action_layout.addWidget(self.face_draw_btn)

        self.reset_btn = StyledButton("重置", "utility")
        self.reset_btn.clicked.connect(lambda: self.adjustment_queue.put(AdjReset()))
        action_layout.addWidget(self.reset_btn)

        sep1 = QWidget()
        sep1.setFixedWidth(1)
        sep1.setStyleSheet(f"background: {HAIRLINE}; border: none; margin: 0 6px;")
        action_layout.addWidget(sep1)

        self.fav_btn = StyledButton("收藏", "ghost")
        self.fav_btn.clicked.connect(self._toggle_favorite)
        action_layout.addWidget(self.fav_btn)

        self.del_btn = StyledButton("删除", "ghost-destructive")
        self.del_btn.clicked.connect(self._delete_current_sticker)
        action_layout.addWidget(self.del_btn)

        sep2 = QWidget()
        sep2.setFixedWidth(1)
        sep2.setStyleSheet(f"background: {HAIRLINE}; border: none; margin: 0 6px;")
        action_layout.addWidget(sep2)

        action_layout.addStretch()

        self.import_btn = StyledButton("导入图片", "ghost")
        self.import_btn.clicked.connect(self._import_image)
        action_layout.addWidget(self.import_btn)

        self.draw_btn = StyledButton("绘制贴纸", "primary")
        self.draw_btn.clicked.connect(self._open_drawing_dialog)
        action_layout.addWidget(self.draw_btn)

        self.edit_sticker_btn = StyledButton("编辑贴纸", "ghost")
        self.edit_sticker_btn.clicked.connect(self._open_edit_sticker_dialog)
        action_layout.addWidget(self.edit_sticker_btn)

        root.addWidget(action_row)

        # ── 4.5 面部绘制工具栏 ──
        self._init_face_draw_toolbar()
        root.addWidget(self.face_draw_toolbar)

        # ── 5. 状态栏 ──
        self.status_label = QLabel("Ctrl+E 编辑 | Ctrl+D 面部绘制 | 左键移动 右键旋转 滚轮缩放 双击重置")
        self.status_label.setStyleSheet(
            f"color: {INK_MUTED_48}; {font_css('fine-print')} padding: 5px; background: {CANVAS}; border-top: 1px solid {HAIRLINE};"
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status_label)

        # 快捷键
        self.edit_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        self.edit_shortcut.activated.connect(self._toggle_edit_mode)
        self.draw_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self.draw_shortcut.activated.connect(self._toggle_face_draw_mode)

        # 面部绘制工具快捷键（默认禁用，仅在面部绘制模式下启用）
        self._draw_tool_shortcuts = []
        for key, handler in [
            ("B", lambda: self._on_draw_quick_brush()),
            ("E", lambda: self._on_draw_quick_eraser()),
            ("Ctrl+Z", lambda: self._on_draw_undo()),
            ("[", lambda: self._on_draw_size_delta(-2)),
            ("]", lambda: self._on_draw_size_delta(2)),
            ("C", lambda: self._on_draw_clear()),
            ("S", lambda: self._on_draw_save()),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(handler)
            sc.setEnabled(False)
            self._draw_tool_shortcuts.append(sc)

    def _init_face_draw_toolbar(self):

        self.face_draw_toolbar = QWidget()
        self.face_draw_toolbar.setVisible(False)
        self.face_draw_toolbar.setStyleSheet(f"background: {PARCHMENT}; border-top: 1px solid {HAIRLINE};")
        tb_layout = QHBoxLayout(self.face_draw_toolbar)
        tb_layout.setContentsMargins(12, 6, 12, 6)
        tb_layout.setSpacing(8)

        label_style = f"color: {INK_MUTED_80}; {font_css('caption')} background: transparent; border: none;"
        value_label_style = f"color: {INK_MUTED_80}; {font_css('fine-print')} background: transparent; border: none;"

        # Region 选择
        region_label = QLabel("位置:")
        region_label.setStyleSheet(label_style)
        tb_layout.addWidget(region_label)

        self._draw_region_combo = QComboBox()
        for label, value in REGION_OPTIONS:
            self._draw_region_combo.addItem(label, value)
        self._draw_region_combo.currentIndexChanged.connect(self._on_draw_region_changed)
        tb_layout.addWidget(self._draw_region_combo)

        tb_layout.addSpacing(8)

        # 笔刷大小
        size_label = QLabel("粗细:")
        size_label.setStyleSheet(label_style)
        tb_layout.addWidget(size_label)

        self._draw_brush_slider = QSlider(Qt.Horizontal)
        self._draw_brush_slider.setRange(1, 50)
        self._draw_brush_slider.setValue(12)
        self._draw_brush_slider.setFixedWidth(100)
        self._draw_brush_slider.valueChanged.connect(self._on_draw_brush_size_changed)
        tb_layout.addWidget(self._draw_brush_slider)

        self._brush_size_label = QLabel("12px")
        self._brush_size_label.setStyleSheet(value_label_style + " min-width: 32px;")
        tb_layout.addWidget(self._brush_size_label)

        tb_layout.addSpacing(8)

        # 间距
        spacing_label = QLabel("间距:")
        spacing_label.setStyleSheet(label_style)
        tb_layout.addWidget(spacing_label)

        self._draw_spacing_slider = QSlider(Qt.Horizontal)
        self._draw_spacing_slider.setRange(3, 200)
        self._draw_spacing_slider.setValue(30)
        self._draw_spacing_slider.setFixedWidth(80)
        self._draw_spacing_slider.valueChanged.connect(self._on_draw_spacing_changed)
        tb_layout.addWidget(self._draw_spacing_slider)

        self._spacing_value_label = QLabel("0.30")
        self._spacing_value_label.setStyleSheet(value_label_style + " min-width: 28px;")
        tb_layout.addWidget(self._spacing_value_label)

        tb_layout.addSpacing(8)

        # 散射
        scatter_label = QLabel("散射:")
        scatter_label.setStyleSheet(label_style)
        tb_layout.addWidget(scatter_label)

        self._draw_scatter_slider = QSlider(Qt.Horizontal)
        self._draw_scatter_slider.setRange(0, 30)
        self._draw_scatter_slider.setValue(0)
        self._draw_scatter_slider.setFixedWidth(80)
        self._draw_scatter_slider.valueChanged.connect(self._on_draw_scatter_changed)
        tb_layout.addWidget(self._draw_scatter_slider)

        self._scatter_value_label = QLabel("0")
        self._scatter_value_label.setStyleSheet(value_label_style + " min-width: 20px;")
        tb_layout.addWidget(self._scatter_value_label)

        tb_layout.addSpacing(8)

        # 颜色预设
        color_label = QLabel("颜色:")
        color_label.setStyleSheet(label_style)
        tb_layout.addWidget(color_label)

        for name, bgra in PRESET_COLORS:
            btn = QPushButton()
            btn.setFixedSize(22, 22)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(name)
            r, g, b = bgra[2], bgra[1], bgra[0]
            btn.setStyleSheet(
                f"QPushButton {{ background: rgb({r},{g},{b}); border: 1px solid {HAIRLINE}; border-radius: {ROUNDED['full']}; }}"
                f"QPushButton:hover {{ border: 2px solid {INK}; }}"
            )
            btn.clicked.connect(lambda checked, c=bgra: self._on_draw_color_clicked(c))
            tb_layout.addWidget(btn)

        tb_layout.addSpacing(8)

        # 笔刷类型
        brush_label = QLabel("笔刷:")
        brush_label.setStyleSheet(label_style)
        tb_layout.addWidget(brush_label)

        self._draw_brush_combo = QComboBox()
        brushes = load_brush_config()
        for b in brushes:
            self._draw_brush_combo.addItem(b["name"], b["id"])
        self._draw_brush_combo.currentIndexChanged.connect(self._on_draw_brush_type_changed)
        tb_layout.addWidget(self._draw_brush_combo)

        tb_layout.addSpacing(8)

        # 压感模式
        pressure_label = QLabel("压感:")
        pressure_label.setStyleSheet(label_style)
        tb_layout.addWidget(pressure_label)

        self._draw_pressure_combo = QComboBox()
        for label, mode in [("大小+浓度", "both"), ("仅大小", "size"), ("仅浓度", "opacity"), ("无压感", "none")]:
            self._draw_pressure_combo.addItem(label, mode)
        self._draw_pressure_combo.currentIndexChanged.connect(self._on_draw_pressure_mode_changed)
        tb_layout.addWidget(self._draw_pressure_combo)

        tb_layout.addSpacing(8)

        # 橡皮擦
        self._draw_eraser_btn = StyledButton("橡皮", "utility")
        self._draw_eraser_btn.setCheckable(True)
        self._draw_eraser_btn.clicked.connect(self._on_draw_eraser_toggled)
        tb_layout.addWidget(self._draw_eraser_btn)

        tb_layout.addSpacing(4)

        # 撤销
        undo_btn = StyledButton("撤销", "utility")
        undo_btn.clicked.connect(self._on_draw_undo)
        tb_layout.addWidget(undo_btn)

        # 清除
        clear_btn = StyledButton("清除", "utility-danger")
        clear_btn.clicked.connect(self._on_draw_clear)
        tb_layout.addWidget(clear_btn)

        save_btn = StyledButton("保存", "primary")
        save_btn.clicked.connect(self._on_draw_save)
        tb_layout.addWidget(save_btn)

        tb_layout.addStretch()

    # ── 画廊管理 ──

    def _on_filter_changed(self, key):
        self._gallery_filter = key
        for k, btn in self._filter_btns.items():
            btn.setChecked(k == key)
        self._load_gallery()

    def _load_gallery(self):
        self.gallery.clear_cards()
        self._gallery_items.clear()
        self._template_cards.clear()

        if self._gallery_filter == "templates":
            templates = load_templates()
            for t in templates:
                tid = t["id"]
                card = ThumbnailCard(tid, t["thumb"], t["name"], False)
                card.clicked.connect(self._on_template_click)
                card.set_selected(tid in self._gallery_selected_ids)
                card.set_active(False)
                self.gallery.add_card(card)
                self._template_cards[tid] = card
            self._update_gallery_info()
            return

        stickers = storage.load_gallery()
        groups = storage.load_groups()

        grouped = {}
        ungrouped = []
        for s in stickers:
            if self._gallery_filter == "favorites" and not s.get("favorite", False):
                continue
            gid = s.get("group_id")
            if gid and any(grp["id"] == gid for grp in groups):
                grouped.setdefault(gid, []).append(s)
            else:
                ungrouped.append(s)

        def _create_card(s):
            sid = s["id"]
            thumb = storage.get_sticker_thumb(sid)
            card = ThumbnailCard(sid, thumb, s.get("prompt", ""), s.get("favorite", False),
                                is_animated=s.get("is_animated", False))
            card.clicked.connect(self._on_gallery_click)
            card.set_selected(sid in self._gallery_selected_ids)
            card.set_active(False)
            self._gallery_items[sid] = card
            return card

        for g in groups:
            members = grouped.get(g["id"], [])
            if not members:
                continue
            section_id = f"__grp_{g['id']}"
            self.gallery.add_section_header(section_id, g.get("name", "未命名"), len(members))
            for s in members:
                card = _create_card(s)
                self.gallery.add_card(card, section_id=section_id)

        if ungrouped:
            self.gallery.add_section_header("__ungrouped", "其他贴纸", len(ungrouped))
            for s in ungrouped:
                card = _create_card(s)
                self.gallery.add_card(card, section_id="__ungrouped")

        self._update_gallery_info()

    def _update_gallery_info(self):
        if self._gallery_filter == "templates":
            n = len(self._template_cards)
            label = f"共 {n} 个模板" if n else "暂无模板"
        elif self._gallery_filter == "favorites":
            n = len(self._gallery_items)
            label = f"共 {n} 个收藏" if n else "暂无收藏贴纸"
        else:
            n = len(self._gallery_items)
            if n == 0:
                label = "还没有贴纸，输入描述来生成第一枚吧"
            else:
                label = f"共 {n} 枚贴纸"
        self.sticker_count_label.setText(label)
        self.gallery.show_placeholder(len(self._gallery_items) == 0 and len(self._template_cards) == 0)

    def _on_template_click(self, template_id):
        # Multi-select: toggle in selection set
        if template_id in self._gallery_selected_ids:
            self._gallery_selected_ids.discard(template_id)
        else:
            self._gallery_selected_ids.add(template_id)
        self._current_sticker_id = template_id
        self._update_gallery_selection_visuals()

    def _on_gallery_click(self, sticker_id):
        # Multi-select: toggle in selection set
        if sticker_id in self._gallery_selected_ids:
            self._gallery_selected_ids.discard(sticker_id)
        else:
            self._gallery_selected_ids.add(sticker_id)
        self._current_sticker_id = sticker_id
        self._update_gallery_selection_visuals()

    def _update_gallery_selection_visuals(self):
        for sid, card in self._gallery_items.items():
            card.set_selected(sid in self._gallery_selected_ids)
            card.set_active(False)
        for tid, card in self._template_cards.items():
            card.set_selected(tid in self._gallery_selected_ids)
            card.set_active(False)
        self.add_to_face_btn.setEnabled(len(self._gallery_selected_ids) > 0)

        # Enable AI Animation only when a single non-template, non-animated sticker is selected
        can_animate = False
        if len(self._gallery_selected_ids) == 1:
            sid = next(iter(self._gallery_selected_ids))
            card = self._gallery_items.get(sid)
            if card and not card._animated:
                can_animate = True
        self.ai_anim_btn.setEnabled(can_animate)

    def _on_add_selected_to_face(self):
        if not self._gallery_selected_ids:
            return
        for sid in self._gallery_selected_ids:
            sid_type = "template" if sid in self._template_cards else "sticker"
            if sid_type == "template":
                templates = load_templates()
                for t in templates:
                    if t["id"] == sid:
                        self.gallery_queue.put(GalLoadTemplate(template=t))
                        break
            else:
                self.gallery_queue.put(GalAddSticker(sticker_id=sid))
        self._gallery_selected_ids.clear()
        self._update_gallery_selection_visuals()

    def _on_sticker_saved(self, sticker_id):
        self._current_sticker_id = sticker_id
        self._load_gallery()
        # Don't re-enable here — DispGenProgress(done=True) handles it

    def _on_generation_failed(self, error_msg):
        self.chat_panel.add_agent_message(f"生成失败: {error_msg}", "failed")
        self._reenable_input()
        QMessageBox.warning(self, "生成失败", f"贴纸生成失败：\n{error_msg}")

    def _on_gen_progress(self, msg):
        if msg.done:
            self.chat_panel.add_agent_message(msg.message, "done")
            self._load_gallery()
            self._reenable_input()
        else:
            self.status_label.setText(msg.message)

    def _on_agent_message(self, text):
        self.chat_panel.add_agent_message(text, "done")
        self.status_label.setText(text)

    def _on_agent_question(self, text):
        self.chat_panel.add_agent_message(text, "ask")
        self.status_label.setText(text)
        self._reenable_input()

    def _toggle_favorite(self):
        if not self._current_sticker_id:
            return
        stickers = storage.load_gallery()
        entry = next((s for s in stickers if s["id"] == self._current_sticker_id), None)
        if entry is None:
            return
        new_fav = not entry.get("favorite", False)
        storage.set_favorite(self._current_sticker_id, new_fav)
        self._load_gallery()

    def _delete_current_sticker(self):
        if not self._current_sticker_id:
            return
        reply = QMessageBox.question(
            self, "确认删除", "确定要删除这枚贴纸吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            storage.delete_sticker(self._current_sticker_id)
            self._current_sticker_id = None
            self.gallery_queue.put(GalLoadSticker(sticker_id=None))
            self._load_gallery()

    def _import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp);;所有文件 (*.*)"
        )
        if not path:
            return
        from app.utils.image_proc import load_rgba_sticker
        sticker = load_rgba_sticker(path)
        if sticker is None:
            QMessageBox.warning(self, "导入失败", "无法读取该图片，请确认格式正确。")
            return

        region, ok = self._choose_region_dialog()
        if not ok:
            return

        sid = storage.save_sticker(sticker, {
            "prompt": f"导入: {path.split('/')[-1].split(chr(92))[-1]}",
            "location": region,
            "scale": 1.0,
        })
        self.gallery_queue.put(GalLoadSticker(sticker_id=sid))
        self._load_gallery()

    def _choose_region_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("选择位置")
        dlg.setFixedSize(280, 120)
        dlg.setStyleSheet(f"QDialog {{ background: {CANVAS}; }}")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("请选择贴纸应用位置:"))
        combo = QComboBox()
        for label, value in REGION_OPTIONS:
            combo.addItem(label, value)
        layout.addWidget(combo)
        btn_layout = QHBoxLayout()
        ok_btn = StyledButton("确定", "primary")
        cancel_btn = StyledButton("取消", "utility")
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        if dlg.exec_() == QDialog.Accepted:
            return combo.currentData(), True
        return None, False

    def _open_drawing_dialog(self):
        dlg = DrawingDialog(self, self.gallery_queue, self.command_queue)
        if dlg.exec_() == DrawingDialog.Accepted:
            self._load_gallery()

    def _open_edit_sticker_dialog(self):
        if not self._current_sticker_id:
            QMessageBox.information(self, "提示", "请先在画廊中选择一枚贴纸，或先 AI 生成一枚贴纸。")
            return
        sticker_img, _ = storage.get_sticker(self._current_sticker_id)
        if sticker_img is None:
            QMessageBox.warning(self, "错误", "无法加载该贴纸，文件可能已被删除。")
            return
        dlg = DrawingDialog(self, self.gallery_queue, self.command_queue, initial_sticker=sticker_img)
        if dlg.exec_() == DrawingDialog.Accepted:
            self._load_gallery()

    def _populate_style_combo(self):
        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        items = get_style_preset_items()
        cfg = get_config()
        selected_key = cfg.get("style", {}).get("selected_preset", "flat_vector")
        selected_idx = 0
        for idx, (key, name) in enumerate(items):
            self.style_combo.addItem(name, key)
            if key == selected_key:
                selected_idx = idx
        self.style_combo.setCurrentIndex(selected_idx)
        self.style_combo.blockSignals(False)

    def _on_style_changed(self, idx):
        preset_key = self.style_combo.currentData()
        if not preset_key:
            return
        cfg = get_config()
        cfg.setdefault("style", {})["selected_preset"] = preset_key

        # Update LoRA from preset if specified
        preset = cfg.get("style", {}).get("presets", {}).get(preset_key, {})
        if "lora_name" in preset:
            lora = cfg.setdefault("model", {}).setdefault("lora", {})
            if preset["lora_name"]:
                lora["name"] = preset["lora_name"]
                lora["strength_model"] = preset.get("lora_strength_model", 0.8)
                lora["strength_clip"] = preset.get("lora_strength_clip", 0.8)
            else:
                # Empty lora_name means disable LoRA
                lora["strength_model"] = 0.0
                lora["strength_clip"] = 0.0
        # If preset has no lora_name key at all, don't touch the global lora

        # Persist both style selection and lora changes
        self._save_style_and_lora_to_disk(preset_key)

    def _update_config_json(self, update_fn):
        import json, os
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    disk_cfg = json.load(f)
            else:
                disk_cfg = {}
            update_fn(disk_cfg)
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(disk_cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[UI] 保存 config.json 失败: {e}")

    def _save_style_and_lora_to_disk(self, preset_key):
        def _update(cfg):
            cfg.setdefault("style", {})["selected_preset"] = preset_key
            preset = cfg.get("style", {}).get("presets", {}).get(preset_key, {})
            if "lora_name" in preset:
                lora = cfg.setdefault("model", {}).setdefault("lora", {})
                if preset["lora_name"]:
                    lora["name"] = preset["lora_name"]
                    lora["strength_model"] = preset.get("lora_strength_model", 0.8)
                    lora["strength_clip"] = preset.get("lora_strength_clip", 0.8)
                else:
                    lora["strength_model"] = 0.0
                    lora["strength_clip"] = 0.0
        self._update_config_json(_update)

    def _on_symmetry_toggled(self, state):
        enabled = state == Qt.Checked
        cfg = get_config()
        cfg.setdefault("generation", {})["symmetry_enabled"] = enabled
        self._save_symmetry_to_disk(enabled)

    def _save_symmetry_to_disk(self, enabled):
        self._update_config_json(lambda cfg: cfg.setdefault("generation", {}).update(symmetry_enabled=enabled))

    def _show_settings(self):
        from app.utils.config_loader import get_config
        cfg = get_config()
        prefs = cfg.get("preferences", {})
        recent = prefs.get("recent_prompts", [])
        info = (
            f"ComfyUI 地址: {cfg['comfyui']['server_address']}\n"
            f"LoRA: {cfg['model']['lora']['name']}\n"
            f"AI 模型: {cfg['agent']['model_id']}\n"
            f"最近指令: {', '.join(recent[:5]) if recent else '无'}"
        )
        QMessageBox.information(self, "设置", info)

    # ── 编辑模式 ──

    def _toggle_edit_mode(self):
        self._edit_mode = not self._edit_mode
        self.edit_btn.setChecked(self._edit_mode)
        self.edit_btn.setText("编辑中" if self._edit_mode else "编辑")
        if self._edit_mode:
            if self._face_draw_mode:
                self._toggle_face_draw_mode()
            # Select last active sticker as edit target, or leave None
            if self._active_instance_ids:
                self.gallery_queue.put(GalSelectEditTarget(instance_id=self._active_instance_ids[-1]))
            self.edit_indicator.setVisible(True)
            self._position_indicator()
            self.input_box.setEnabled(False)
            self.send_btn.setEnabled(False)
        else:
            self.gallery_queue.put(GalSelectEditTarget(instance_id=None))
            self.edit_indicator.setVisible(False)
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)
            self._mouse_down = False

    def _position_indicator(self):
        label_pos = self.video_label.pos()
        margin = 12
        self.edit_indicator.move(label_pos.x() + margin, label_pos.y() + margin)

    # ── 活动贴纸状态 ──

    def _on_active_stickers_changed(self, data):
        instances = data.instances
        edit_target_id = data.edit_target_id

        thumbs_info = {}
        templates = load_templates()
        for inst in instances:
            iid = inst["instance_id"]
            sid = inst.get("sticker_id")
            region = inst.get("region", "")
            thumb = None
            if sid:
                thumb = storage.get_sticker_thumb(sid)
                if thumb is None:
                    for t in templates:
                        if t["id"] == sid:
                            thumb = t.get("thumb")
                            break
            thumbs_info[iid] = {"thumb": thumb, "region": region}

        self._active_instance_ids = [inst["instance_id"] for inst in instances]
        self._edit_target_id = edit_target_id
        self._sync_active_panel(thumbs_info)

        if edit_target_id:
            self.anim_timeline.set_instance(edit_target_id)
            self.anim_timeline.setVisible(True)
        else:
            self.anim_timeline.setVisible(False)

    def _sync_active_panel(self, thumbs_info=None):
        if not self._active_instance_ids:
            self.active_panel.clear_all()
            self._update_merge_button()
            return
        if thumbs_info is None:
            thumbs_info = {}
        self.active_panel.sync(self._active_instance_ids, thumbs_info, self._edit_target_id)
        self._update_merge_button()

    def _update_merge_button(self):
        self.merge_btn.setEnabled(len(self._active_instance_ids) >= 2)

    def _on_merge_group(self):
        if len(self._active_instance_ids) < 2:
            return
        self.gallery_queue.put(GalMergeGroup(instance_ids=list(self._active_instance_ids)))

    def _on_anim_clip_updated(self, msg):
        self.anim_timeline.update_clip(msg.clip_data)

    def _on_anim_playback_state(self, msg):
        self.anim_timeline.update_playback(msg.playing, msg.time, msg.duration)

    def _on_anim_export_progress(self, msg):
        if msg.done:
            if msg.output_path:
                print(f"[UI] 导出完成: {msg.output_path}")
            else:
                print("[UI] 导出失败")

    def _on_ai_animate(self):
        if len(self._gallery_selected_ids) != 1:
            return
        sid = next(iter(self._gallery_selected_ids))
        dlg = AnimationGenDialog(sid, self)
        self._ai_anim_dlg = dlg
        if dlg.exec_() == QDialog.Accepted:
            params = dlg.result()
            self.animation_queue.put(AnimGenTexture(
                sticker_id=params["sticker_id"],
                motion_prompt=params["motion_prompt"],
                frame_count=params["frame_count"],
                fps=params["fps"],
            ))
            self._gallery_selected_ids.clear()
            self._update_gallery_selection_visuals()

    def _on_anim_gen_progress(self, msg):
        dlg = getattr(self, '_ai_anim_dlg', None)
        if dlg and (not msg.done):
            dlg.set_progress(msg.progress)
        elif dlg and msg.done:
            dlg.set_progress(1.0)
            if msg.error:
                dlg.set_error(msg.error)
            else:
                # Refresh gallery to show new animated sticker
                self._refresh_gallery()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_indicator()

    def _label_to_frame_delta(self, dx, dy):
        if self._frame_size is None:
            return dx, dy
        fw, fh = self._frame_size
        lw = self.video_label.width()
        lh = self.video_label.height()
        if lw <= 0 or lh <= 0:
            return dx, dy
        return dx * fw / lw, dy * fh / lh

    # ── 鼠标事件 ──

    def eventFilter(self, obj, event):
        if obj is self.video_label:
            if self._edit_mode:
                t = event.type()
                if t == QEvent.MouseButtonPress:
                    self._on_mouse_press(event)
                    return True
                elif t == QEvent.MouseMove:
                    self._on_mouse_move(event)
                    return True
                elif t == QEvent.MouseButtonRelease:
                    self._on_mouse_release(event)
                    return True
                elif t == QEvent.Wheel:
                    self._on_wheel(event)
                    return True
                elif t == QEvent.MouseButtonDblClick:
                    self._on_double_click(event)
                    return True
            elif self._face_draw_mode:
                t = event.type()
                if t == QEvent.TabletPress:
                    self._tablet_in_use = True
                    self._handle_draw_press(event.pos(), event.pressure())
                    return True
                elif t == QEvent.TabletMove and self._tablet_in_use:
                    if event.pressure() > 0:
                        self._handle_draw_move(event.pos(), event.pressure())
                    else:
                        self._handle_draw_release()
                        self._tablet_in_use = False
                    return True
                elif t == QEvent.TabletRelease:
                    self._handle_draw_release()
                    self._tablet_in_use = False
                    return True
                elif not self._tablet_in_use:
                    if t == QEvent.MouseButtonPress:
                        self._on_draw_mouse_press(event)
                        return True
                    elif t == QEvent.MouseMove:
                        self._on_draw_mouse_move(event)
                        return True
                    elif t == QEvent.MouseButtonRelease:
                        self._on_draw_mouse_release(event)
                        return True
        return super().eventFilter(obj, event)

    def _on_mouse_press(self, event):
        self._mouse_down = True
        self._mouse_button = event.button()
        self._last_mouse_pos = event.pos()

    def _on_mouse_move(self, event):
        if not self._mouse_down or self._last_mouse_pos is None:
            return
        dx_px = event.pos().x() - self._last_mouse_pos.x()
        dy_px = event.pos().y() - self._last_mouse_pos.y()
        self._last_mouse_pos = event.pos()
        dx_f, dy_f = self._label_to_frame_delta(dx_px, dy_px)
        if self._mouse_button == Qt.LeftButton:
            self.adjustment_queue.put(AdjMove(dx=dx_f, dy=dy_f))
        elif self._mouse_button == Qt.RightButton:
            self.adjustment_queue.put(AdjRotate(d_angle=dx_f * 0.3))

    def _on_mouse_release(self, event):
        self._mouse_down = False
        self._mouse_button = None
        self._last_mouse_pos = None

    def _on_wheel(self, event):
        delta = event.angleDelta().y()
        factor = 1.0 + delta * 0.0005
        self.adjustment_queue.put(AdjScale(multiplier=factor))

    def _on_double_click(self, event):
        self.adjustment_queue.put(AdjReset())

    # ── 面部绘制模式 ──

    def _toggle_face_draw_mode(self):
        self._face_draw_mode = not self._face_draw_mode
        self.draw_queue.put(DrawToggleDrawMode())
        self.face_draw_btn.setChecked(self._face_draw_mode)
        self.face_draw_btn.setText("绘制中" if self._face_draw_mode else "面部绘制")
        for sc in self._draw_tool_shortcuts:
            sc.setEnabled(self._face_draw_mode)
        if self._face_draw_mode:
            if self._edit_mode:
                self._toggle_edit_mode()
            self.face_draw_toolbar.setVisible(True)
            self.edit_btn.setEnabled(False)
            self.input_box.setEnabled(False)
            self.send_btn.setEnabled(False)
            self._face_draw_stroke_points = []
            self.status_label.setText("B 画笔  E 橡皮  [ ] 粗细  Ctrl+Z 撤销  C 清除  S 保存")
        else:
            self.face_draw_toolbar.setVisible(False)
            self.edit_btn.setEnabled(True)
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)
            self._face_draw_mouse_down = False
            self.status_label.setText("Ctrl+E 编辑 | Ctrl+D 面部绘制 | 左键移动 右键旋转 滚轮缩放 双击重置")

    def _label_point_to_frame(self, label_pos):
        if self._frame_size is None:
            return label_pos.x(), label_pos.y()
        fx = (label_pos.x() - self._label_offset_x) / self._label_scale
        fy = (label_pos.y() - self._label_offset_y) / self._label_scale
        return fx, fy

    # ── shared draw handlers (used by both mouse and tablet) ──

    def _handle_draw_press(self, pos, pressure):
        fx, fy = self._label_point_to_frame(pos)
        self._face_draw_mouse_down = True
        self._face_draw_stroke_points = [(fx, fy)]
        self.draw_queue.put(DrawStrokeBegin())
        self.draw_queue.put(DrawStrokePoint(point=(fx, fy), pressure=pressure))

    def _handle_draw_move(self, pos, pressure):
        if not self._face_draw_mouse_down:
            return
        fx, fy = self._label_point_to_frame(pos)
        self._face_draw_stroke_points.append((fx, fy))
        self.draw_queue.put(DrawStrokePoint(point=(fx, fy), pressure=pressure))

    def _handle_draw_release(self):
        if self._face_draw_mouse_down:
            self._face_draw_mouse_down = False
            self._face_draw_stroke_points = []
            self.draw_queue.put(DrawStrokeEnd())

    def _on_draw_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self._face_draw_mouse_down = True
            self._face_draw_stroke_points = []
            fx, fy = self._label_point_to_frame(event.pos())
            self._face_draw_stroke_points.append((fx, fy))
            self.draw_queue.put(DrawStrokeBegin())
            self.draw_queue.put(DrawStrokePoint(point=(fx, fy)))

    def _on_draw_mouse_move(self, event):
        if not self._face_draw_mouse_down:
            return
        fx, fy = self._label_point_to_frame(event.pos())
        self._face_draw_stroke_points.append((fx, fy))
        self.draw_queue.put(DrawStrokePoint(point=(fx, fy)))

    def _on_draw_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self._face_draw_mouse_down = False
            self._face_draw_stroke_points = []
            self.draw_queue.put(DrawStrokeEnd())

    def _on_draw_region_changed(self, idx):
        region = self._draw_region_combo.currentData()
        self._face_draw_region = region
        self.draw_queue.put(DrawSetRegion(region=region))

    def _on_draw_brush_size_changed(self, value):
        self._face_draw_brush_size = value
        self._brush_size_label.setText(f"{value}px")
        self.draw_queue.put(DrawSetBrush(brush_size=value, brush_color=self._face_draw_brush_color))

    def _on_draw_color_clicked(self, bgra):
        self._face_draw_brush_color = bgra
        self._face_draw_eraser = False
        self._draw_eraser_btn.setChecked(False)
        self.draw_queue.put(DrawSetBrush(brush_color=bgra, brush_size=self._face_draw_brush_size))

    def _on_draw_eraser_toggled(self, checked):
        self._face_draw_eraser = checked
        self.draw_queue.put(DrawToggleEraser(eraser_mode=checked))

    def _on_draw_undo(self):
        self.draw_queue.put(DrawUndo())

    def _on_draw_clear(self):
        self.draw_queue.put(DrawClear())

    def _on_draw_save(self):
        self.draw_queue.put(DrawSave())

    def _on_draw_brush_type_changed(self, idx):
        brush_id = self._draw_brush_combo.currentData()
        if brush_id:
            self._face_draw_brush_type = brush_id
            self.draw_queue.put(DrawSetBrushType(brush_id=brush_id))

    def _on_draw_pressure_mode_changed(self, idx):
        mode = self._draw_pressure_combo.currentData()
        if mode:
            self._face_draw_pressure_mode = mode
            self.draw_queue.put(DrawSetPressureMode(mode=mode))

    def _on_draw_spacing_changed(self, value):
        coef = value / 100.0
        self._spacing_value_label.setText(f"{coef:.2f}")
        self.draw_queue.put(DrawSetSpacing(coef=coef))

    def _on_draw_scatter_changed(self, value):
        self._scatter_value_label.setText(str(value))
        self.draw_queue.put(DrawSetScatter(px=float(value)))

    def _on_draw_quick_brush(self):
        self._face_draw_eraser = False
        self._draw_eraser_btn.setChecked(False)
        self.draw_queue.put(DrawToggleEraser(eraser_mode=False))

    def _on_draw_quick_eraser(self):
        self._face_draw_eraser = True
        self._draw_eraser_btn.setChecked(True)
        self.draw_queue.put(DrawToggleEraser(eraser_mode=True))

    def _on_draw_size_delta(self, delta):
        new_size = max(1, min(50, self._face_draw_brush_size + delta))
        self._face_draw_brush_size = new_size
        self._draw_brush_slider.setValue(new_size)
        self._brush_size_label.setText(f"{new_size}px")
        self.draw_queue.put(DrawSetBrush(brush_size=new_size, brush_color=self._face_draw_brush_color))

    # ── 命令发送 ──

    def send_command(self):
        text = self.input_box.text().strip()
        if not text:
            return
        self.chat_panel.add_user_message(text)
        self.command_queue.put(text)
        storage.add_recent_prompt(text)
        self.input_box.clear()
        self.input_box.setPlaceholderText("AI 正在生成中，请稍候...")
        self.input_box.setEnabled(False)
        self.send_btn.setEnabled(False)

    def _reenable_input(self):
        if not self._edit_mode:
            self.input_box.setEnabled(True)
            self.send_btn.setEnabled(True)
        self.input_box.setPlaceholderText("描述你的创意，例如：一副赛博朋克风格的护目镜...")

    # ── 视频显示 ──

    def update_image(self, cv_img):
        h, w, ch = cv_img.shape
        self._frame_size = (w, h)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.width(), self.video_label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._label_scale = scaled.width() / max(w, 1)
        self._label_offset_x = (self.video_label.width() - scaled.width()) / 2.0
        self._label_offset_y = (self.video_label.height() - scaled.height()) / 2.0
        self.video_label.setPixmap(scaled)

    def closeEvent(self, event):
        from app.utils.storage import save_preferences
        save_preferences({
            "window_width": self.width(),
            "window_height": self.height(),
        })
        self.video_thread.stop()
        event.accept()
