"""Test animation engine — easing, evaluate_clip, AnimationEngine, serialization."""

import math
import pytest

from app.core.animation import (
    linear, ease_in, ease_out, ease_in_out, EASING_FUNCTIONS,
    Keyframe, AnimationClip, evaluate_clip, AnimationEngine,
)


# ── Easing functions ──

def test_linear():
    assert linear(0.0) == 0.0
    assert linear(0.5) == 0.5
    assert linear(1.0) == 1.0


def test_ease_in_endpoints():
    assert ease_in(0.0) == 0.0
    assert ease_in(1.0) == 1.0


def test_ease_in_below_diagonal():
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert ease_in(t) <= t


def test_ease_out_endpoints():
    assert ease_out(0.0) == 0.0
    assert ease_out(1.0) == 1.0


def test_ease_out_above_diagonal():
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert ease_out(t) >= t


def test_ease_in_out_endpoints():
    assert ease_in_out(0.0) == 0.0
    assert pytest.approx(ease_in_out(0.5)) == 0.5
    assert ease_in_out(1.0) == 1.0


def test_ease_in_out_symmetric():
    assert pytest.approx(ease_in_out(0.2)) == 1.0 - ease_in_out(0.8)


def test_all_easing_in_dict():
    assert "linear" in EASING_FUNCTIONS
    assert "ease-in" in EASING_FUNCTIONS
    assert "ease-out" in EASING_FUNCTIONS
    assert "ease-in-out" in EASING_FUNCTIONS


# ── Keyframe ──

def test_keyframe_defaults():
    kf = Keyframe(time=1.0)
    assert kf.time == 1.0
    assert kf.offset_x == 0.0
    assert kf.offset_y == 0.0
    assert kf.rotation == 0.0
    assert kf.scale_mult == 1.0
    assert kf.opacity == 1.0
    assert kf.easing == "linear"


def test_keyframe_to_from_dict():
    kf = Keyframe(time=2.0, offset_x=0.1, offset_y=-0.2, rotation=15.0,
                  scale_mult=1.3, opacity=0.8, easing="ease-in-out")
    d = kf.to_dict()
    kf2 = Keyframe.from_dict(d)
    assert kf2.time == 2.0
    assert kf2.offset_x == 0.1
    assert kf2.offset_y == -0.2
    assert kf2.rotation == 15.0
    assert kf2.scale_mult == 1.3
    assert kf2.opacity == 0.8
    assert kf2.easing == "ease-in-out"


# ── AnimationClip ──

def test_clip_generates_id():
    clip = AnimationClip(name="test")
    assert clip.id != ""


def test_clip_add_keyframe_sorted():
    clip = AnimationClip(duration=3.0)
    clip.add_keyframe(Keyframe(time=2.0))
    clip.add_keyframe(Keyframe(time=0.5))
    clip.add_keyframe(Keyframe(time=1.0))
    times = [k.time for k in clip.keyframes]
    assert times == [0.5, 1.0, 2.0]


def test_clip_remove_keyframe():
    clip = AnimationClip()
    clip.add_keyframe(Keyframe(time=0.0))
    clip.add_keyframe(Keyframe(time=1.0))
    clip.remove_keyframe(0)
    assert len(clip.keyframes) == 1
    assert clip.keyframes[0].time == 1.0


def test_clip_to_from_dict():
    clip = AnimationClip(name="bounce", duration=2.0, loop=True, sticker_id="s1")
    clip.add_keyframe(Keyframe(time=0.0, offset_y=0.0, easing="ease-in-out"))
    clip.add_keyframe(Keyframe(time=1.0, offset_y=-0.1, easing="ease-in-out"))
    clip.add_keyframe(Keyframe(time=2.0, offset_y=0.0, easing="ease-in-out"))

    d = clip.to_dict()
    clip2 = AnimationClip.from_dict(d)
    assert clip2.name == "bounce"
    assert clip2.duration == 2.0
    assert clip2.loop is True
    assert len(clip2.keyframes) == 3
    assert clip2.keyframes[1].offset_y == -0.1


# ── evaluate_clip ──

def _make_clip(keyframes, loop=False, duration=3.0):
    clip = AnimationClip(loop=loop, duration=duration)
    for kf in keyframes:
        clip.add_keyframe(kf)
    return clip


def test_evaluate_empty_clip():
    clip = AnimationClip()
    adj = evaluate_clip(clip, 0.5)
    assert adj["offset_x"] == 0.0 and adj["scale_mult"] == 1.0


def test_evaluate_single_keyframe():
    clip = _make_clip([Keyframe(time=0.5, offset_x=0.3)])
    adj = evaluate_clip(clip, 0.0)
    assert adj["offset_x"] == 0.3


def test_evaluate_at_exact_keyframe():
    clip = _make_clip([
        Keyframe(time=0.0, scale_mult=1.0),
        Keyframe(time=1.0, scale_mult=2.0),
    ])
    adj = evaluate_clip(clip, 1.0)
    assert adj["scale_mult"] == 2.0


def test_evaluate_midpoint():
    clip = _make_clip([
        Keyframe(time=0.0, offset_x=0.0),
        Keyframe(time=2.0, offset_x=1.0),
    ])
    adj = evaluate_clip(clip, 1.0)
    assert adj["offset_x"] == pytest.approx(0.5)


def test_evaluate_before_first_keyframe():
    clip = _make_clip([
        Keyframe(time=0.5, offset_x=0.3),
        Keyframe(time=1.5, offset_x=0.9),
    ])
    adj = evaluate_clip(clip, 0.0)
    assert adj["offset_x"] == 0.3


def test_evaluate_past_end_no_loop():
    clip = _make_clip([
        Keyframe(time=0.0, offset_x=0.0),
        Keyframe(time=1.0, offset_x=1.0),
    ], loop=False, duration=1.0)
    adj = evaluate_clip(clip, 5.0)
    assert adj["offset_x"] == 1.0


def test_evaluate_looping():
    clip = _make_clip([
        Keyframe(time=0.0, offset_x=0.0),
        Keyframe(time=1.0, offset_x=1.0),
    ], loop=True, duration=1.0)
    # t=1.5 should wrap to 0.5 → midpoint
    adj = evaluate_clip(clip, 1.5)
    assert adj["offset_x"] == pytest.approx(0.5)


def test_evaluate_with_easing():
    clip = _make_clip([
        Keyframe(time=0.0, offset_y=0.0, easing="ease-in"),
        Keyframe(time=1.0, offset_y=1.0, easing="ease-in"),
    ])
    adj_linear = evaluate_clip(clip, 0.5)
    # ease-in at t=0.5 should be < linear 0.5
    assert adj_linear["offset_y"] == pytest.approx(0.25)  # ease_in(0.5) = 0.25


def test_evaluate_rotation_interpolation():
    clip = _make_clip([
        Keyframe(time=0.0, rotation=0.0),
        Keyframe(time=1.0, rotation=90.0),
    ])
    adj = evaluate_clip(clip, 0.5)
    assert adj["rotation"] == pytest.approx(45.0)


# ── AnimationEngine ──

@pytest.fixture
def engine():
    eng = AnimationEngine()
    clip = AnimationClip(id="c1", name="bounce", duration=2.0, loop=True)
    clip.add_keyframe(Keyframe(time=0.0, offset_y=0.0))
    clip.add_keyframe(Keyframe(time=1.0, offset_y=-0.1))
    clip.add_keyframe(Keyframe(time=2.0, offset_y=0.0))
    eng.register_clip(clip)
    eng.set_clip("inst1", "c1")
    return eng


def test_engine_set_clip(engine):
    assert engine.get_bound_clip("inst1") is not None
    assert engine.get_bound_clip("inst1").name == "bounce"


def test_engine_set_nonexistent_clip(engine):
    assert engine.set_clip("inst2", "nonexistent") is False


def test_engine_play_pause_stop(engine):
    info = engine.get_playback_info("inst1")
    assert not info["playing"]

    engine.play("inst1")
    assert engine.is_playing("inst1")

    engine.pause("inst1")
    assert not engine.is_playing("inst1")

    engine.stop("inst1")
    assert not engine.is_playing("inst1")
    assert engine.get_playback_info("inst1")["time"] == 0.0


def test_engine_evaluate_returns_none_for_unbound():
    eng = AnimationEngine()
    assert eng.evaluate("no_such_instance") is None


def test_engine_evaluate_returns_values(engine):
    adj = engine.evaluate("inst1")
    assert adj is not None
    assert "offset_x" in adj
    assert "offset_y" in adj


def test_engine_tick_advances_time(engine):
    engine.play("inst1")
    import time
    time.sleep(0.05)
    engine.tick("inst1")
    info = engine.get_playback_info("inst1")
    assert info["time"] > 0.0


def test_engine_tick_no_advance_when_paused(engine):
    engine.tick("inst1")
    info = engine.get_playback_info("inst1")
    assert info["time"] == pytest.approx(0.0)


def test_engine_register_and_all_clips(engine):
    clips = engine.all_clips()
    assert len(clips) == 1
    assert clips[0].id == "c1"


def test_engine_seek(engine):
    engine.seek("inst1", 1.5)
    info = engine.get_playback_info("inst1")
    assert info["time"] == 1.5


def test_engine_loop_wraps(engine):
    engine.play("inst1")
    engine.seek("inst1", 2.0)
    engine.tick("inst1")
    info = engine.get_playback_info("inst1")
    assert info["time"] == pytest.approx(0.0, abs=0.1)
