"""Typed message protocol for all inter-process queues.

Replaces bare dicts with dataclass instances.  Each message type is a
separate class — field names are explicit, defaults are declared once,
and ``isinstance`` replaces stringly-typed ``msg.get("action")`` checks.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any

# ══════════════════════════════════════════════════════════════════════════════
# Action constants (for code that still needs string matching)
# ══════════════════════════════════════════════════════════════════════════════

class Adj:
    MOVE   = "move"
    ROTATE = "rotate"
    SCALE  = "scale"
    RESET  = "reset"


class Gal:
    ADD_STICKER        = "add_sticker"
    REMOVE_STICKER     = "remove_sticker"
    SELECT_EDIT_TARGET = "select_edit_target"
    LOAD_TEMPLATE      = "load_template"
    LOAD_STICKER       = "load_sticker"
    MERGE_GROUP        = "merge_group"


class Draw:
    TOGGLE_DRAW_MODE  = "toggle_draw_mode"
    SET_REGION        = "set_region"
    SET_BRUSH         = "set_brush"
    TOGGLE_ERASER     = "toggle_eraser"
    SET_BRUSH_TYPE    = "set_brush_type"
    SET_PRESSURE_MODE = "set_pressure_mode"
    SET_SPACING       = "set_spacing"
    SET_SCATTER       = "set_scatter"
    UNDO              = "undo"
    CLEAR             = "clear"
    STROKE_BEGIN      = "stroke_begin"
    STROKE_POINT       = "stroke_point"
    STROKE_END        = "stroke_end"
    SAVE              = "save"
    SET_TEXT          = "set_text"


class Result:
    GENERATION_PROGRESS = "generation_progress"
    GENERATION_RESULT   = "generation_result"
    GENERATION_DONE     = "generation_done"
    AGENT_QUESTION      = "agent_question"
    ERROR               = "error"


class Disp:
    STICKER_SAVED           = "sticker_saved"
    GENERATION_FAILED       = "generation_failed"
    ACTIVE_STICKERS_CHANGED = "active_stickers_changed"
    GEN_PROGRESS            = "gen_progress"
    AGENT_MESSAGE           = "agent_message"
    AGENT_QUESTION          = "agent_question"


class Anim:
    PLAY            = "anim_play"
    PAUSE           = "anim_pause"
    STOP            = "anim_stop"
    SET_CLIP        = "anim_set_clip"
    ADD_KEYFRAME    = "anim_add_keyframe"
    REMOVE_KEYFRAME = "anim_remove_keyframe"
    UPDATE_KEYFRAME = "anim_update_keyframe"
    SET_LOOP        = "anim_set_loop"
    SEEK            = "anim_seek"
    EXPORT          = "anim_export"
    EXPORT_PROGRESS = "anim_export_progress"
    CLIP_UPDATED    = "anim_clip_updated"
    PLAYBACK_STATE  = "anim_playback_state"
    GEN_TEXTURE     = "anim_gen_texture"
    GEN_PROGRESS    = "anim_gen_progress"


# ══════════════════════════════════════════════════════════════════════════════
# adjustment_queue  (UI → Consumer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdjMove:
    action: str = Adj.MOVE
    dx: float = 0.0
    dy: float = 0.0


@dataclass
class AdjRotate:
    action: str = Adj.ROTATE
    d_angle: float = 0.0


@dataclass
class AdjScale:
    action: str = Adj.SCALE
    multiplier: float = 1.0


@dataclass
class AdjReset:
    action: str = Adj.RESET


# Union type for convenience
AdjustmentMsg = AdjMove | AdjRotate | AdjScale | AdjReset


# ══════════════════════════════════════════════════════════════════════════════
# gallery_queue  (UI → Consumer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GalAddSticker:
    action: str = Gal.ADD_STICKER
    sticker_id: str = ""


@dataclass
class GalRemoveSticker:
    action: str = Gal.REMOVE_STICKER
    instance_id: str = ""


@dataclass
class GalSelectEditTarget:
    action: str = Gal.SELECT_EDIT_TARGET
    instance_id: Optional[str] = None


@dataclass
class GalLoadTemplate:
    action: str = Gal.LOAD_TEMPLATE
    template: Optional[dict] = None


@dataclass
class GalLoadSticker:
    action: str = Gal.LOAD_STICKER
    sticker_id: Optional[str] = None


@dataclass
class GalMergeGroup:
    action: str = Gal.MERGE_GROUP
    instance_ids: list = field(default_factory=list)


GalleryMsg = (GalAddSticker | GalRemoveSticker | GalSelectEditTarget
              | GalLoadTemplate | GalLoadSticker | GalMergeGroup)


# ══════════════════════════════════════════════════════════════════════════════
# draw_queue  (UI → Consumer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrawToggleDrawMode:
    action: str = Draw.TOGGLE_DRAW_MODE


@dataclass
class DrawSetRegion:
    action: str = Draw.SET_REGION
    region: str = "forehead_full"


@dataclass
class DrawSetBrush:
    action: str = Draw.SET_BRUSH
    brush_size: int = 12
    brush_color: Tuple[int, int, int, int] = (0, 0, 0, 255)


@dataclass
class DrawToggleEraser:
    action: str = Draw.TOGGLE_ERASER
    eraser_mode: bool = True


@dataclass
class DrawSetBrushType:
    action: str = Draw.SET_BRUSH_TYPE
    brush_id: str = "hard_round"


@dataclass
class DrawSetPressureMode:
    action: str = Draw.SET_PRESSURE_MODE
    mode: str = "both"


@dataclass
class DrawSetSpacing:
    action: str = Draw.SET_SPACING
    coef: float = 0.3


@dataclass
class DrawSetScatter:
    action: str = Draw.SET_SCATTER
    px: float = 0.0


@dataclass
class DrawUndo:
    action: str = Draw.UNDO


@dataclass
class DrawClear:
    action: str = Draw.CLEAR


@dataclass
class DrawStrokeBegin:
    action: str = Draw.STROKE_BEGIN


@dataclass
class DrawStrokePoint:
    action: str = Draw.STROKE_POINT
    point: Optional[Tuple[float, float]] = None
    pressure: float = 1.0


@dataclass
class DrawStrokeEnd:
    action: str = Draw.STROKE_END


@dataclass
class DrawSave:
    action: str = Draw.SAVE


@dataclass
class DrawText:
    action: str = Draw.SET_TEXT
    text: str = ""
    pos_x: float = 0.5       # 0–1 相对于脸宽的位置 (以鼻尖为原点)
    pos_y: float = -0.3      # 正=下，负=上
    font_scale: float = 1.0
    color_bgr: tuple = (255, 255, 255)  # BGR
    thickness: int = 2
    clear: bool = False      # True 时清除文字


DrawMsg = (DrawToggleDrawMode | DrawSetRegion | DrawSetBrush | DrawToggleEraser
           | DrawSetBrushType | DrawSetPressureMode | DrawSetSpacing
           | DrawSetScatter | DrawUndo | DrawClear | DrawStrokeBegin
           | DrawStrokePoint | DrawStrokeEnd | DrawSave | DrawText)


# ══════════════════════════════════════════════════════════════════════════════
# animation_queue  (UI → Consumer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnimPlay:
    action: str = Anim.PLAY
    instance_id: str = ""


@dataclass
class AnimPause:
    action: str = Anim.PAUSE
    instance_id: str = ""


@dataclass
class AnimStop:
    action: str = Anim.STOP
    instance_id: str = ""


@dataclass
class AnimSetClip:
    action: str = Anim.SET_CLIP
    instance_id: str = ""
    clip_id: str = ""


@dataclass
class AnimAddKeyframe:
    action: str = Anim.ADD_KEYFRAME
    instance_id: str = ""
    time: float = 0.0
    easing: str = "linear"


@dataclass
class AnimRemoveKeyframe:
    action: str = Anim.REMOVE_KEYFRAME
    instance_id: str = ""
    keyframe_index: int = 0


@dataclass
class AnimUpdateKeyframe:
    action: str = Anim.UPDATE_KEYFRAME
    instance_id: str = ""
    keyframe_index: int = 0
    time: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    scale_mult: float = 1.0
    opacity: float = 1.0
    easing: str = "linear"


@dataclass
class AnimSetLoop:
    action: str = Anim.SET_LOOP
    instance_id: str = ""
    loop: bool = False


@dataclass
class AnimSeek:
    action: str = Anim.SEEK
    instance_id: str = ""
    time: float = 0.0


@dataclass
class AnimExport:
    action: str = Anim.EXPORT
    instance_id: str = ""
    format: str = "mp4"
    fps: int = 24
    output_path: str = ""


@dataclass
class AnimGenTexture:
    action: str = Anim.GEN_TEXTURE
    sticker_id: str = ""
    motion_prompt: str = ""
    frame_count: int = 16
    fps: int = 8


AnimationMsg = (AnimPlay | AnimPause | AnimStop | AnimSetClip
                | AnimAddKeyframe | AnimRemoveKeyframe | AnimUpdateKeyframe
                | AnimSetLoop | AnimSeek | AnimExport | AnimGenTexture)


# ══════════════════════════════════════════════════════════════════════════════
# display_queue  (Consumer → UI)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DispStickerSaved:
    action: str = Disp.STICKER_SAVED
    sticker_id: str = ""


@dataclass
class DispGenerationFailed:
    action: str = Disp.GENERATION_FAILED
    error: str = ""


@dataclass
class DispActiveStickersChanged:
    action: str = Disp.ACTIVE_STICKERS_CHANGED
    active_count: int = 0
    instances: list = field(default_factory=list)
    edit_target_id: Optional[str] = None
    face_center_x: float = 0.0
    face_center_y: float = 0.0
    face_width: float = 0.0


@dataclass
class DispGenProgress:
    action: str = Disp.GEN_PROGRESS
    current: int = 0
    total: int = 0
    step: int = 0
    total_steps: int = 0
    preview_path: str = ""
    message: str = ""
    done: bool = False


@dataclass
class DispAgentMessage:
    action: str = Disp.AGENT_MESSAGE
    text: str = ""


@dataclass
class DispAgentQuestion:
    action: str = Disp.AGENT_QUESTION
    text: str = ""
    options: list = field(default_factory=list)



@dataclass
class AnimExportProgress:
    action: str = Anim.EXPORT_PROGRESS
    progress: float = 0.0
    done: bool = False
    output_path: str = ""


@dataclass
class AnimClipUpdated:
    action: str = Anim.CLIP_UPDATED
    clip_data: dict = field(default_factory=dict)


@dataclass
class AnimPlaybackState:
    action: str = Anim.PLAYBACK_STATE
    instance_id: str = ""
    playing: bool = False
    time: float = 0.0
    duration: float = 0.0


@dataclass
class AnimGenProgress:
    action: str = Anim.GEN_PROGRESS
    sticker_id: str = ""
    progress: float = 0.0
    done: bool = False
    error: str = ""
    result_sticker_id: str = ""


DisplayStatusMsg = (DispStickerSaved | DispGenerationFailed | DispActiveStickersChanged
                    | DispGenProgress | DispAgentMessage | DispAgentQuestion
                    | AnimExportProgress | AnimClipUpdated | AnimPlaybackState
                    | AnimGenProgress)


# ══════════════════════════════════════════════════════════════════════════════
# command_queue  (UI → Consumer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CmdImg2Img:
    """Sent from DrawingDialog._ai_refine or future img2img flows."""
    type: str = "img2img"
    prompt_text: str = ""
    image_path: str = ""
    target_location: str = "forehead_top"
    scale: float = 1.0
    display_name: str = ""
    controlnet_strength: float = 0.85
    denoise: float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Domain model types (shared across ConsumerProcessor, mixins, and tests)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Adjustment:
    """Manual edit state bound to a StickerInstance."""
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    scale_mult: float = 1.0


@dataclass
class StickerInstance:
    """An active sticker placed on the face, with placement and animation metadata.

    Replaces the bare ``dict`` previously used in ``self.active_stickers``.
    """
    instance_id: str
    sticker_id: str = ""
    sticker: Any = None        # np.ndarray — BGRA image
    location: str = "forehead_top"
    scale: float = 1.0
    prompt: str = ""
    is_animated: bool = False
    frame_count: int = 0
    frame_cols: int = 0
    frame_rows: int = 0
    fps: int = 8
