"""Tests for app/core/tracker_animation.py — AnimationProcessor mixin."""

import queue

import pytest

from app.core.animation import AnimationEngine, AnimationClip, Keyframe, TextureAnimator
from app.core.protocol import Adjustment, StickerInstance
from app.core.protocol import (
    AnimPlay, AnimPause, AnimStop, AnimSetClip,
    AnimAddKeyframe, AnimRemoveKeyframe, AnimUpdateKeyframe,
    AnimSetLoop, AnimSeek, AnimExport, AnimGenTexture,
)
from app.core.sticker_registry import StickerRegistry
from app.core.tracker_animation import AnimationProcessor


class _TestAnimationProcessor(AnimationProcessor):
    """Minimal wrapper providing attributes ConsumerProcessor normally sets."""

    def __init__(self, display_queue=None, animation_queue=None):
        self.anim_engine = AnimationEngine()
        self.animation_queue = animation_queue or queue.Queue()
        self.display_queue = display_queue or queue.Queue()
        self.registry = StickerRegistry(max_stickers=20)
        self._anim_evaluations = {}
        self._adj_is_delta = set()
        self._pending_export = None
        self._pending_texture_gen = None
        self._texture_gen_running = False
        self.texture_animator = TextureAnimator()
        self.cached_face_data = None


@pytest.fixture
def proc():
    return _TestAnimationProcessor()


@pytest.fixture
def proc_with_sticker(proc):
    """Add a sticker instance and return (proc, instance_id)."""
    iid = "test-instance-001"
    inst = StickerInstance(
        instance_id=iid, sticker_id="sid-001",
        sticker=None, location="forehead_top",
        scale=1.0, prompt="test",
    )
    adj = Adjustment()
    proc.registry.add(inst, adj)
    return proc, iid


# ── _adj_to_delta / _adj_to_absolute ──

class TestAdjConversion:
    def _set_adj(self, proc, iid, ox, oy, rot, sm):
        adj = proc.registry.get_adj(iid)
        adj.offset_x = ox
        adj.offset_y = oy
        adj.rotation = rot
        adj.scale_mult = sm

    def test_delta_subtracts_anim_values(self, proc):
        iid = "test-iid"
        self._set_adj(proc, iid, 10.0, 5.0, 2.0, 2.0)
        proc._anim_evaluations[iid] = {"offset_x": 3.0, "offset_y": 1.0, "rotation": 0.5, "scale_mult": 1.5}
        proc._adj_to_delta(iid)
        adj = proc.registry.get_adj(iid)
        assert adj.offset_x == pytest.approx(7.0)
        assert adj.offset_y == pytest.approx(4.0)
        assert adj.rotation == pytest.approx(1.5)
        assert adj.scale_mult == pytest.approx(2.0 / 1.5)

    def test_absolute_adds_anim_values(self, proc):
        iid = "test-iid"
        self._set_adj(proc, iid, 7.0, 4.0, 1.5, 1.33333)
        proc._anim_evaluations[iid] = {"offset_x": 3.0, "offset_y": 1.0, "rotation": 0.5, "scale_mult": 1.5}
        proc._adj_to_absolute(iid)
        adj = proc.registry.get_adj(iid)
        assert adj.offset_x == pytest.approx(10.0)
        assert adj.offset_y == pytest.approx(5.0)
        assert adj.rotation == pytest.approx(2.0)
        assert adj.scale_mult == pytest.approx(2.0, rel=0.01)

    def test_delta_no_anim_no_op(self, proc):
        iid = "test-iid"
        self._set_adj(proc, iid, 5.0, 0.0, 0.0, 1.0)
        proc._adj_to_delta(iid)
        adj = proc.registry.get_adj(iid)
        assert adj.offset_x == 5.0

    def test_delta_zero_scale_mult_no_division_by_zero(self, proc):
        iid = "test-iid"
        self._set_adj(proc, iid, 0.0, 0.0, 0.0, 2.0)
        proc._anim_evaluations[iid] = {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 0.0}
        proc._adj_to_delta(iid)
        adj = proc.registry.get_adj(iid)
        assert adj.scale_mult == 1.0


# ── _process_animation_queue ──

class TestProcessAnimationQueue:
    def test_play_sets_delta_mode(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        proc.animation_queue.put(AnimPlay(instance_id=iid))
        # Register a clip so it doesn't silently fail
        clip = AnimationClip(name="Test Clip", sticker_id=iid)
        proc.anim_engine.register_clip(clip)
        proc.anim_engine.set_clip(iid, clip.id)
        proc._process_animation_queue()
        assert iid in proc._adj_is_delta

    def test_pause_switches_to_absolute(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        clip = AnimationClip(name="Test Clip", sticker_id=iid)
        proc.anim_engine.register_clip(clip)
        proc.anim_engine.set_clip(iid, clip.id)
        # Need to have an anim evaluation for pause to work
        proc.anim_engine.tick(iid)  # advance a bit
        proc.anim_engine.pause(iid)
        proc._adj_is_delta.add(iid)
        proc.animation_queue.put(AnimPause(instance_id=iid))
        proc._process_animation_queue()
        assert iid not in proc._adj_is_delta

    def test_stop_switches_to_absolute(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        proc._adj_is_delta.add(iid)
        proc.animation_queue.put(AnimStop(instance_id=iid))
        proc._process_animation_queue()
        assert iid not in proc._adj_is_delta

    def test_set_clip(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        clip = AnimationClip(name="Test Clip", sticker_id=iid)
        proc.anim_engine.register_clip(clip)
        proc.animation_queue.put(AnimSetClip(instance_id=iid, clip_id=clip.id))
        proc._process_animation_queue()
        bound = proc.anim_engine.get_bound_clip(iid)
        assert bound is not None
        assert bound.id == clip.id

    def test_set_loop(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        proc.animation_queue.put(AnimSetLoop(instance_id=iid, loop=True))
        proc._process_animation_queue()
        # Loop setting is internal to engine; just verify no crash

    def test_seek(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        proc.animation_queue.put(AnimSeek(instance_id=iid, time=1.5))
        proc._process_animation_queue()
        # Seek is internal; just verify no crash

    def test_export_sets_pending(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        proc.animation_queue.put(AnimExport(instance_id=iid, format="mp4", fps=24, output_path="/tmp/test.mp4"))
        proc._process_animation_queue()
        assert proc._pending_export is not None
        assert proc._pending_export[0] == iid

    def test_gen_texture_sets_pending(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        proc.animation_queue.put(AnimGenTexture(sticker_id="sid-001", motion_prompt="wave", frame_count=16, fps=8))
        proc._process_animation_queue()
        assert proc._pending_texture_gen is not None
        assert proc._pending_texture_gen.sticker_id == "sid-001"


# ── _handle_add_keyframe ──

class TestHandleAddKeyframe:
    def test_creates_new_clip_if_none_bound(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        msg = AnimAddKeyframe(instance_id=iid, time=0.0, easing="linear")
        proc._handle_add_keyframe(msg)
        clip = proc.anim_engine.get_bound_clip(iid)
        assert clip is not None
        assert len(clip.keyframes) == 1

    def test_adds_keyframe_to_existing_clip(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        clip = AnimationClip(name="Test", sticker_id=iid)
        clip.add_keyframe(Keyframe(time=0.0))
        clip.add_keyframe(Keyframe(time=1.0))
        proc.anim_engine.register_clip(clip)
        proc.anim_engine.set_clip(iid, clip.id)
        msg = AnimAddKeyframe(instance_id=iid, time=0.5, easing="ease-in")
        proc._handle_add_keyframe(msg)
        assert len(clip.keyframes) == 3  # two default + one added

    def test_sends_clip_updated_to_display(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        initial_count = proc.display_queue.qsize()
        msg = AnimAddKeyframe(instance_id=iid, time=0.0, easing="linear")
        proc._handle_add_keyframe(msg)
        assert proc.display_queue.qsize() == initial_count + 1


# ── _handle_remove_keyframe ──

class TestHandleRemoveKeyframe:
    def test_removes_keyframe(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        clip = AnimationClip(name="Test", sticker_id=iid)
        clip.add_keyframe(Keyframe(time=0.0))
        clip.add_keyframe(Keyframe(time=1.0))
        proc.anim_engine.register_clip(clip)
        proc.anim_engine.set_clip(iid, clip.id)
        initial_count = len(clip.keyframes)
        msg = AnimRemoveKeyframe(instance_id=iid, keyframe_index=0)
        proc._handle_remove_keyframe(msg)
        assert len(clip.keyframes) == initial_count - 1

    def test_no_clip_does_nothing(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        msg = AnimRemoveKeyframe(instance_id=iid, keyframe_index=0)
        # Should not raise
        proc._handle_remove_keyframe(msg)


# ── _handle_update_keyframe ──

class TestHandleUpdateKeyframe:
    def test_updates_keyframe_values(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        clip = AnimationClip(name="Test", sticker_id=iid)
        clip.add_keyframe(Keyframe(time=0.0))
        clip.add_keyframe(Keyframe(time=1.0))
        proc.anim_engine.register_clip(clip)
        proc.anim_engine.set_clip(iid, clip.id)
        msg = AnimUpdateKeyframe(
            instance_id=iid, keyframe_index=0, time=0.5,
            offset_x=10.0, offset_y=20.0, rotation=30.0, scale_mult=2.0,
            opacity=0.8, easing="ease-out",
        )
        proc._handle_update_keyframe(msg)
        kf = clip.keyframes[0]
        assert kf.time == 0.5
        assert kf.offset_x == 10.0
        assert kf.easing == "ease-out"

    def test_invalid_index_no_op(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        msg = AnimUpdateKeyframe(
            instance_id=iid, keyframe_index=99, time=0.0,
            offset_x=0.0, offset_y=0.0, rotation=0.0, scale_mult=1.0,
            opacity=1.0, easing="linear",
        )
        # Should not raise
        proc._handle_update_keyframe(msg)


# ── _evaluate_animations ──

class TestEvaluateAnimations:
    def test_no_animations_does_nothing(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        # No clip bound — should just iterate without error
        proc._evaluate_animations()
        assert iid not in proc._anim_evaluations

    def test_evaluates_bound_clip(self, proc_with_sticker):
        proc, iid = proc_with_sticker
        clip = AnimationClip(name="Test", sticker_id=iid)
        clip.add_keyframe(Keyframe(time=0.0))
        clip.add_keyframe(Keyframe(time=1.0))
        proc.anim_engine.register_clip(clip)
        proc.anim_engine.set_clip(iid, clip.id)
        proc.anim_engine.play(iid)
        proc._evaluate_animations()
        # After tick+evaluate, _anim_evaluations should have an entry
        assert iid in proc._anim_evaluations
