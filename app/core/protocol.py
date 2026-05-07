"""Typed message protocol for all inter-process queues.

Replaces bare dicts with dataclass instances.  Each message type is a
separate class — field names are explicit, defaults are declared once,
and ``isinstance`` replaces stringly-typed ``msg.get("action")`` checks.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

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


class Disp:
    STICKER_SAVED           = "sticker_saved"
    GENERATION_FAILED       = "generation_failed"
    ACTIVE_STICKERS_CHANGED = "active_stickers_changed"


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


DrawMsg = (DrawToggleDrawMode | DrawSetRegion | DrawSetBrush | DrawToggleEraser
           | DrawSetBrushType | DrawSetPressureMode | DrawSetSpacing
           | DrawSetScatter | DrawUndo | DrawClear | DrawStrokeBegin
           | DrawStrokePoint | DrawStrokeEnd | DrawSave)


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


DisplayStatusMsg = DispStickerSaved | DispGenerationFailed | DispActiveStickersChanged


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
