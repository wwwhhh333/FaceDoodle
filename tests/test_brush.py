"""Test brush stamping, tip generation, and BrushStateMixin."""

import numpy as np
import pytest

from app.core.brush import (
    stamp_brush, stamp_line, _generate_default_tip,
    BrushStateMixin, PRESSURE_MIN_RATIO,
    load_brush_config, load_brush_tip, ensure_default_brushes, get_brush_by_id,
)


# ── _generate_default_tip ──

def test_generate_hard_tip_shape():
    tip = _generate_default_tip(64, soft=False)
    assert tip.shape == (64, 64, 4)
    assert tip.dtype == np.uint8


def test_generate_soft_tip_shape():
    tip = _generate_default_tip(32, soft=True)
    assert tip.shape == (32, 32, 4)


def test_hard_tip_has_nonzero_alpha():
    tip = _generate_default_tip(64, soft=False)
    assert np.any(tip[:, :, 3] > 0)


def test_hard_tip_is_binary_like():
    """Hard tip alpha should be mostly 0 or 255."""
    tip = _generate_default_tip(128, soft=False)
    alpha = tip[:, :, 3]
    # At least 90% of pixels are 0 or 255
    extreme = (alpha == 0) | (alpha == 255)
    assert extreme.mean() > 0.9


def test_soft_tip_has_smooth_falloff():
    tip = _generate_default_tip(128, soft=True)
    alpha = tip[:, :, 3]
    # Soft tip has intermediate values
    mid = (alpha > 10) & (alpha < 245)
    assert mid.sum() > 50


def test_tip_rgb_is_white():
    tip = _generate_default_tip(64, soft=False)
    assert np.all(tip[:, :, :3] == 255)


# ── stamp_brush ──

def test_stamp_brush_center(blank_canvas, hard_tip):
    stamp_brush(blank_canvas, 50, 50, hard_tip, (255, 0, 0, 255), 1.0)
    assert np.any(blank_canvas[:, :, 3] > 0)


def test_stamp_brush_color_applied(blank_canvas, hard_tip):
    stamp_brush(blank_canvas, 50, 50, hard_tip, (0, 0, 255, 255), 1.0)
    # Pixels under tip should be red (BGR: 0,0,255)
    mask = blank_canvas[:, :, 3] > 0
    assert mask.sum() > 0
    assert np.all(blank_canvas[mask, 2] > 200)  # red channel


def test_stamp_brush_opacity(blank_canvas, hard_tip):
    canvas1 = blank_canvas.copy()
    canvas2 = blank_canvas.copy()
    stamp_brush(canvas1, 50, 50, hard_tip, (255, 255, 255, 255), 0.3)
    stamp_brush(canvas2, 50, 50, hard_tip, (255, 255, 255, 255), 1.0)
    # Full opacity should have higher alpha
    assert canvas2[:, :, 3].max() > canvas1[:, :, 3].max()


def test_stamp_brush_eraser(blank_canvas, hard_tip):
    # First add some color
    stamp_brush(blank_canvas, 50, 50, hard_tip, (255, 255, 255, 255), 1.0)
    before = blank_canvas[:, :, 3].sum()
    # Then erase
    stamp_brush(blank_canvas, 50, 50, hard_tip, (0, 0, 0, 0), 1.0, eraser=True)
    after = blank_canvas[:, :, 3].sum()
    assert after < before


def test_stamp_brush_out_of_bounds(blank_canvas, hard_tip):
    """Should not raise when stamping way outside canvas."""
    stamp_brush(blank_canvas, -1000, -1000, hard_tip, (255, 0, 0, 255), 1.0)
    stamp_brush(blank_canvas, 9999, 9999, hard_tip, (255, 0, 0, 255), 1.0)


def test_stamp_brush_partial_overlap(blank_canvas, hard_tip):
    """Stamp at edge where only part of tip overlaps canvas."""
    stamp_brush(blank_canvas, 0, 0, hard_tip, (255, 0, 0, 255), 1.0)
    assert np.any(blank_canvas[:, :, 3] > 0)


# ── stamp_line ──

def test_stamp_line_draws_multiple_points(blank_canvas, hard_tip):
    canvas = blank_canvas.copy()
    stamp_line(canvas, (10, 50), (90, 50), hard_tip, (255, 0, 0, 255), 1.0, spacing_px=10.0)
    assert np.any(canvas[:, :, 3] > 0)


def test_stamp_line_no_scatter_is_straight(blank_canvas, hard_tip):
    canvas = blank_canvas.copy()
    stamp_line(canvas, (10, 50), (90, 50), hard_tip, (255, 255, 255, 255), 1.0, spacing_px=5.0, scatter=0.0)
    # With zero scatter and horizontal line, all non-zero pixels should be around y=50
    ys = np.where(canvas[:, :, 3] > 0)[0]
    assert abs(ys.mean() - 50) < 15


def test_stamp_line_zero_spacing_handled(blank_canvas, hard_tip):
    """Should not crash with zero/negative spacing."""
    stamp_line(blank_canvas, (10, 50), (90, 50), hard_tip, (255, 0, 0, 255), 1.0, spacing_px=0.0)


# ── BrushStateMixin ──

class _TestBrush(BrushStateMixin):
    """Minimal class to exercise the mixin in isolation."""
    def __init__(self):
        self._init_brush_state()


@pytest.fixture
def brush():
    return _TestBrush()


def test_brush_init_defaults(brush):
    assert brush._brush_size == 12
    assert brush._brush_color == (0, 0, 0, 255)
    assert not brush._eraser_mode
    assert brush._pressure == 1.0
    assert brush._pressure_mode == "both"


def test_set_brush_size_clamps(brush):
    brush.set_brush_size(0)
    assert brush._brush_size == 1
    brush.set_brush_size(100)
    assert brush._brush_size == 50
    brush.set_brush_size(20)
    assert brush._brush_size == 20


def test_set_brush_color(brush):
    brush.set_brush_color((255, 0, 0, 255))
    assert brush._brush_color == (255, 0, 0, 255)
    assert not brush._eraser_mode  # setting color disables eraser


def test_set_eraser_mode(brush):
    brush.set_eraser_mode(True)
    assert brush._eraser_mode
    brush.set_eraser_mode(0)
    assert not brush._eraser_mode


def test_set_pressure_clamps(brush):
    brush.set_pressure(-0.5)
    assert brush._pressure == 0.0
    brush.set_pressure(1.5)
    assert brush._pressure == 1.0
    brush.set_pressure(0.5)
    assert brush._pressure == 0.5


def test_set_spacing_clamps(brush):
    brush.set_spacing(0.0)
    assert brush._spacing_override == 0.03
    brush.set_spacing(5.0)
    assert brush._spacing_override == 2.0


def test_set_scatter_clamps(brush):
    brush.set_scatter(-5.0)
    assert brush._scatter_override == 0.0
    brush.set_scatter(100.0)
    assert brush._scatter_override == 30.0


def test_pressure_scales_none(brush):
    brush._pressure_mode = "none"
    s, o = brush._pressure_scales()
    assert s == 1.0 and o == 1.0


def test_pressure_scales_size(brush):
    brush._pressure_mode = "size"
    brush._pressure = 0.0
    s, o = brush._pressure_scales()
    assert s == PRESSURE_MIN_RATIO
    assert o == 1.0


def test_pressure_scales_opacity(brush):
    brush._pressure_mode = "opacity"
    brush._pressure = 0.0
    s, o = brush._pressure_scales()
    assert s == 1.0
    assert o == PRESSURE_MIN_RATIO


def test_pressure_scales_both(brush):
    brush._pressure_mode = "both"
    brush._pressure = 0.0
    s, o = brush._pressure_scales()
    assert s == PRESSURE_MIN_RATIO
    assert o == PRESSURE_MIN_RATIO


def test_effective_size(brush):
    brush._brush_size = 20
    brush._pressure = 1.0
    brush._pressure_mode = "none"
    assert brush._effective_size() == 20


def test_resolve_spacing_scatter_defaults(brush):
    brush._brush_size = 10
    brush._pressure_mode = "none"
    spacing, scatter, opacity = brush._resolve_spacing_scatter()
    assert spacing >= 1.0
    assert scatter >= 0.0
    assert opacity == 1.0


def test_resolve_spacing_scatter_override(brush):
    brush._brush_size = 10
    brush._pressure_mode = "none"
    brush._spacing_override = 0.5
    brush._scatter_override = 2.0
    spacing, scatter, opacity = brush._resolve_spacing_scatter()
    assert spacing == 5.0  # 10 * 0.5
    assert scatter == 20.0  # 10 * 2.0


# ── File I/O tests (Phase 3, but included here for convenience) ──

def test_ensure_default_brushes_creates_files(tmp_path, monkeypatch):
    import app.core.brush as brush_mod
    monkeypatch.setattr(brush_mod, "BRUSH_DIR", str(tmp_path / "brushes"))
    monkeypatch.setattr(brush_mod, "CONFIG_PATH", str(tmp_path / "brushes" / "brushes.json"))

    ensure_default_brushes()
    assert (tmp_path / "brushes" / "hard_round.png").exists()
    assert (tmp_path / "brushes" / "soft_round.png").exists()
    assert (tmp_path / "brushes" / "brushes.json").exists()


def test_load_brush_config_returns_list(tmp_path, monkeypatch):
    import app.core.brush as brush_mod
    monkeypatch.setattr(brush_mod, "BRUSH_DIR", str(tmp_path / "brushes"))
    monkeypatch.setattr(brush_mod, "CONFIG_PATH", str(tmp_path / "brushes" / "brushes.json"))

    ensure_default_brushes()
    brushes = load_brush_config()
    assert isinstance(brushes, list)
    assert any(b["id"] == "hard_round" for b in brushes)


def test_get_brush_by_id_found(tmp_path, monkeypatch):
    import app.core.brush as brush_mod
    monkeypatch.setattr(brush_mod, "BRUSH_DIR", str(tmp_path / "brushes"))
    monkeypatch.setattr(brush_mod, "CONFIG_PATH", str(tmp_path / "brushes" / "brushes.json"))

    ensure_default_brushes()
    b = get_brush_by_id("soft_round")
    assert b is not None
    assert b["id"] == "soft_round"


def test_get_brush_by_id_not_found():
    assert get_brush_by_id("nonexistent") is None


def test_load_brush_tip_returns_array(tmp_path, monkeypatch):
    import app.core.brush as brush_mod
    monkeypatch.setattr(brush_mod, "BRUSH_DIR", str(tmp_path / "brushes"))
    monkeypatch.setattr(brush_mod, "CONFIG_PATH", str(tmp_path / "brushes" / "brushes.json"))
    # Clear global cache between tests
    brush_mod._tip_cache.clear()

    ensure_default_brushes()
    tip = load_brush_tip("hard_round.png", 6)
    assert tip is not None
    assert tip.shape[2] == 4  # RGBA
    assert tip.shape[0] == 13  # 6*2+1


def test_load_brush_tip_cached(tmp_path, monkeypatch):
    import app.core.brush as brush_mod
    monkeypatch.setattr(brush_mod, "BRUSH_DIR", str(tmp_path / "brushes"))
    monkeypatch.setattr(brush_mod, "CONFIG_PATH", str(tmp_path / "brushes" / "brushes.json"))
    brush_mod._tip_cache.clear()

    ensure_default_brushes()
    tip1 = load_brush_tip("hard_round.png", 6)
    tip2 = load_brush_tip("hard_round.png", 6)
    assert tip1 is tip2  # same object from cache


def test_load_brush_tip_missing_file():
    result = load_brush_tip("__nonexistent__.png", 5)
    assert result is None
