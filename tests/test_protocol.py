"""Test all message dataclasses and action constants."""

import dataclasses

from app.core.protocol import (
    Adj, Gal, Draw, Disp, Anim,
    AdjMove, AdjRotate, AdjScale, AdjReset,
    GalAddSticker, GalRemoveSticker, GalSelectEditTarget,
    GalLoadTemplate, GalLoadSticker, GalMergeGroup,
    DrawToggleDrawMode, DrawSetRegion, DrawSetBrush, DrawToggleEraser,
    DrawSetBrushType, DrawSetPressureMode, DrawSetSpacing, DrawSetScatter,
    DrawUndo, DrawClear, DrawStrokeBegin, DrawStrokePoint, DrawStrokeEnd, DrawSave,
    AnimPlay, AnimPause, AnimStop, AnimSetClip, AnimAddKeyframe,
    AnimRemoveKeyframe, AnimSetLoop, AnimSeek, AnimExport,
    DispStickerSaved, DispGenerationFailed, DispActiveStickersChanged,
    AnimExportProgress, AnimClipUpdated, AnimPlaybackState,
    CmdImg2Img,
    AdjustmentMsg, GalleryMsg, DrawMsg, AnimationMsg, DisplayStatusMsg,
)


# ── Action constants ──

def test_adj_constants():
    assert Adj.MOVE == "move"
    assert Adj.ROTATE == "rotate"
    assert Adj.SCALE == "scale"
    assert Adj.RESET == "reset"


def test_gal_constants():
    assert Gal.ADD_STICKER == "add_sticker"
    assert Gal.REMOVE_STICKER == "remove_sticker"
    assert Gal.SELECT_EDIT_TARGET == "select_edit_target"
    assert Gal.LOAD_TEMPLATE == "load_template"
    assert Gal.LOAD_STICKER == "load_sticker"
    assert Gal.MERGE_GROUP == "merge_group"


def test_draw_constants():
    assert Draw.TOGGLE_DRAW_MODE == "toggle_draw_mode"
    assert Draw.STROKE_BEGIN == "stroke_begin"
    assert Draw.STROKE_POINT == "stroke_point"
    assert Draw.STROKE_END == "stroke_end"
    assert Draw.SAVE == "save"


def test_disp_constants():
    assert Disp.STICKER_SAVED == "sticker_saved"
    assert Disp.GENERATION_FAILED == "generation_failed"
    assert Disp.ACTIVE_STICKERS_CHANGED == "active_stickers_changed"


def test_anim_constants():
    assert Anim.PLAY == "anim_play"
    assert Anim.PAUSE == "anim_pause"
    assert Anim.STOP == "anim_stop"
    assert Anim.SET_CLIP == "anim_set_clip"
    assert Anim.ADD_KEYFRAME == "anim_add_keyframe"
    assert Anim.REMOVE_KEYFRAME == "anim_remove_keyframe"
    assert Anim.SET_LOOP == "anim_set_loop"
    assert Anim.SEEK == "anim_seek"
    assert Anim.EXPORT == "anim_export"
    assert Anim.EXPORT_PROGRESS == "anim_export_progress"
    assert Anim.CLIP_UPDATED == "anim_clip_updated"
    assert Anim.PLAYBACK_STATE == "anim_playback_state"


# ── Adjustment messages ──

def test_adj_move_defaults():
    m = AdjMove()
    assert m.action == "move"
    assert m.dx == 0.0 and m.dy == 0.0


def test_adj_move_values():
    m = AdjMove(dx=0.1, dy=-0.2)
    assert m.dx == 0.1 and m.dy == -0.2


def test_adj_rotate():
    m = AdjRotate(d_angle=15.0)
    assert m.action == "rotate"
    assert m.d_angle == 15.0


def test_adj_scale():
    m = AdjScale(multiplier=1.5)
    assert m.action == "scale"
    assert m.multiplier == 1.5


def test_adj_reset():
    m = AdjReset()
    assert m.action == "reset"


# ── Gallery messages ──

def test_gal_add_sticker():
    m = GalAddSticker(sticker_id="abc123")
    assert m.action == "add_sticker"
    assert m.sticker_id == "abc123"


def test_gal_remove_sticker():
    m = GalRemoveSticker(instance_id="inst-1")
    assert m.action == "remove_sticker"
    assert m.instance_id == "inst-1"


def test_gal_select_edit_target():
    m = GalSelectEditTarget(instance_id="inst-2")
    assert m.instance_id == "inst-2"
    assert GalSelectEditTarget().instance_id is None


def test_gal_load_template():
    t = {"id": "t1", "image": None}
    m = GalLoadTemplate(template=t)
    assert m.template is t


def test_gal_load_sticker():
    m = GalLoadSticker(sticker_id="s1")
    assert m.sticker_id == "s1"


def test_gal_merge_group():
    m = GalMergeGroup(instance_ids=["a", "b"])
    assert m.instance_ids == ["a", "b"]
    assert GalMergeGroup().instance_ids == []


# ── Draw messages ──

def test_draw_toggle_draw_mode():
    m = DrawToggleDrawMode()
    assert m.action == "toggle_draw_mode"


def test_draw_set_region():
    m = DrawSetRegion(region="eyes")
    assert m.region == "eyes"


def test_draw_set_brush():
    m = DrawSetBrush(brush_size=20, brush_color=(255, 0, 0, 255))
    assert m.brush_size == 20
    assert m.brush_color == (255, 0, 0, 255)


def test_draw_toggle_eraser():
    m = DrawToggleEraser(eraser_mode=False)
    assert not m.eraser_mode


def test_draw_set_brush_type():
    m = DrawSetBrushType(brush_id="soft_round")
    assert m.brush_id == "soft_round"


def test_draw_set_pressure_mode():
    m = DrawSetPressureMode(mode="size")
    assert m.mode == "size"


def test_draw_set_spacing():
    m = DrawSetSpacing(coef=0.5)
    assert m.coef == 0.5


def test_draw_set_scatter():
    m = DrawSetScatter(px=2.0)
    assert m.px == 2.0


def test_draw_undo():
    m = DrawUndo()
    assert m.action == "undo"


def test_draw_clear():
    m = DrawClear()
    assert m.action == "clear"


def test_draw_stroke_begin():
    m = DrawStrokeBegin()
    assert m.action == "stroke_begin"


def test_draw_stroke_point():
    m = DrawStrokePoint(point=(100.5, 200.3), pressure=0.8)
    assert m.point == (100.5, 200.3)
    assert m.pressure == 0.8


def test_draw_stroke_end():
    m = DrawStrokeEnd()
    assert m.action == "stroke_end"


def test_draw_save():
    m = DrawSave()
    assert m.action == "save"


# ── Animation queue messages ──

def test_anim_play():
    m = AnimPlay(instance_id="inst1")
    assert m.action == "anim_play"
    assert m.instance_id == "inst1"


def test_anim_pause():
    m = AnimPause(instance_id="inst1")
    assert m.action == "anim_pause"
    assert m.instance_id == "inst1"


def test_anim_stop():
    m = AnimStop(instance_id="inst1")
    assert m.action == "anim_stop"
    assert m.instance_id == "inst1"


def test_anim_set_clip():
    m = AnimSetClip(instance_id="inst1", clip_id="clip1")
    assert m.action == "anim_set_clip"
    assert m.instance_id == "inst1"
    assert m.clip_id == "clip1"


def test_anim_add_keyframe():
    m = AnimAddKeyframe(instance_id="inst1", time=1.5, easing="ease-in-out")
    assert m.action == "anim_add_keyframe"
    assert m.instance_id == "inst1"
    assert m.time == 1.5
    assert m.easing == "ease-in-out"


def test_anim_add_keyframe_defaults():
    m = AnimAddKeyframe()
    assert m.time == 0.0
    assert m.easing == "linear"


def test_anim_remove_keyframe():
    m = AnimRemoveKeyframe(instance_id="inst1", keyframe_index=2)
    assert m.action == "anim_remove_keyframe"
    assert m.instance_id == "inst1"
    assert m.keyframe_index == 2


def test_anim_set_loop():
    m = AnimSetLoop(instance_id="inst1", loop=True)
    assert m.action == "anim_set_loop"
    assert m.instance_id == "inst1"
    assert m.loop is True


def test_anim_seek():
    m = AnimSeek(instance_id="inst1", time=2.5)
    assert m.action == "anim_seek"
    assert m.instance_id == "inst1"
    assert m.time == 2.5


def test_anim_export():
    m = AnimExport(instance_id="inst1", format="gif", fps=15, output_path="/tmp/out.gif")
    assert m.action == "anim_export"
    assert m.instance_id == "inst1"
    assert m.format == "gif"
    assert m.fps == 15
    assert m.output_path == "/tmp/out.gif"


def test_anim_export_defaults():
    m = AnimExport()
    assert m.format == "mp4"
    assert m.fps == 24
    assert m.output_path == ""


# ── Animation display messages ──

def test_anim_export_progress():
    m = AnimExportProgress(progress=0.5, done=False, output_path="/tmp/out.mp4")
    assert m.action == "anim_export_progress"
    assert m.progress == 0.5
    assert not m.done
    assert m.output_path == "/tmp/out.mp4"


def test_anim_export_progress_done():
    m = AnimExportProgress(progress=1.0, done=True, output_path="/tmp/out.mp4")
    assert m.progress == 1.0
    assert m.done


def test_anim_clip_updated():
    m = AnimClipUpdated(clip_data={"id": "c1", "name": "bounce", "duration": 2.0})
    assert m.action == "anim_clip_updated"
    assert m.clip_data["id"] == "c1"
    assert m.clip_data["name"] == "bounce"


def test_anim_clip_updated_defaults():
    m = AnimClipUpdated()
    assert m.clip_data == {}


def test_anim_playback_state():
    m = AnimPlaybackState(instance_id="inst1", playing=True, time=1.5, duration=3.0)
    assert m.action == "anim_playback_state"
    assert m.instance_id == "inst1"
    assert m.playing is True
    assert m.time == 1.5
    assert m.duration == 3.0


def test_anim_playback_state_defaults():
    m = AnimPlaybackState()
    assert m.playing is False
    assert m.time == 0.0
    assert m.duration == 0.0


# ── Display messages ──

def test_disp_sticker_saved():
    m = DispStickerSaved(sticker_id="abc")
    assert m.sticker_id == "abc"


def test_disp_generation_failed():
    m = DispGenerationFailed(error="timeout")
    assert m.error == "timeout"


def test_disp_active_stickers_changed():
    m = DispActiveStickersChanged(
        active_count=3,
        instances=[{"instance_id": "i1"}, {"instance_id": "i2"}],
        edit_target_id="i1",
    )
    assert m.active_count == 3
    assert len(m.instances) == 2
    assert m.edit_target_id == "i1"


# ── Command messages ──

def test_cmd_img2img():
    m = CmdImg2Img(
        prompt_text="cat ears",
        image_path="/tmp/test.png",
        target_location="head_top",
        scale=1.2,
        display_name="cat_ears",
    )
    assert m.type == "img2img"
    assert m.prompt_text == "cat ears"
    assert m.image_path == "/tmp/test.png"
    assert m.target_location == "head_top"
    assert m.scale == 1.2
    assert m.display_name == "cat_ears"


# ── Union types ──

def test_adj_move_is_adjustment_msg():
    assert isinstance(AdjMove(), AdjustmentMsg)


def test_adj_rotate_is_adjustment_msg():
    assert isinstance(AdjRotate(), AdjustmentMsg)


def test_gal_add_is_gallery_msg():
    assert isinstance(GalAddSticker(), GalleryMsg)


def test_draw_stroke_point_is_draw_msg():
    assert isinstance(DrawStrokePoint(), DrawMsg)


def test_anim_play_is_animation_msg():
    assert isinstance(AnimPlay(), AnimationMsg)


def test_anim_add_keyframe_is_animation_msg():
    assert isinstance(AnimAddKeyframe(), AnimationMsg)


def test_anim_export_is_animation_msg():
    assert isinstance(AnimExport(), AnimationMsg)


def test_disp_sticker_saved_is_display_status_msg():
    assert isinstance(DispStickerSaved(), DisplayStatusMsg)


def test_anim_export_progress_is_display_status_msg():
    assert isinstance(AnimExportProgress(), DisplayStatusMsg)


def test_anim_clip_updated_is_display_status_msg():
    assert isinstance(AnimClipUpdated(), DisplayStatusMsg)


def test_anim_playback_state_is_display_status_msg():
    assert isinstance(AnimPlaybackState(), DisplayStatusMsg)


# ── All dataclasses are actually dataclasses ──

_ALL_MSG_TYPES = [
    AdjMove, AdjRotate, AdjScale, AdjReset,
    GalAddSticker, GalRemoveSticker, GalSelectEditTarget,
    GalLoadTemplate, GalLoadSticker, GalMergeGroup,
    DrawToggleDrawMode, DrawSetRegion, DrawSetBrush, DrawToggleEraser,
    DrawSetBrushType, DrawSetPressureMode, DrawSetSpacing, DrawSetScatter,
    DrawUndo, DrawClear, DrawStrokeBegin, DrawStrokePoint, DrawStrokeEnd, DrawSave,
    AnimPlay, AnimPause, AnimStop, AnimSetClip, AnimAddKeyframe,
    AnimRemoveKeyframe, AnimSetLoop, AnimSeek, AnimExport,
    DispStickerSaved, DispGenerationFailed, DispActiveStickersChanged,
    AnimExportProgress, AnimClipUpdated, AnimPlaybackState,
    CmdImg2Img,
]


def test_all_types_are_dataclasses():
    for cls in _ALL_MSG_TYPES:
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"
