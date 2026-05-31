"""Test FaceDrawCanvas — undo/redo, coordinate mapping, stroke flow."""

import numpy as np
import pytest

from app.core.face_draw import FaceDrawCanvas, CANVAS_SIZE, MAX_UNDO


@pytest.fixture
def canvas(tmp_path, monkeypatch):
    """FaceDrawCanvas with brush I/O redirected to tmp_path."""
    import app.core.brush as brush_mod
    import app.core.face_draw as fd_mod

    # Redirect brush paths so _init_brush_state can find brush files
    brush_dir = str(tmp_path / "brushes")
    config_path = str(tmp_path / "brushes" / "brushes.json")
    monkeypatch.setattr(brush_mod, "BRUSH_DIR", brush_dir)
    monkeypatch.setattr(brush_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(fd_mod, "CANVAS_SIZE", CANVAS_SIZE)

    # Ensure default brushes exist so _init_brush_state succeeds
    from app.core.brush import ensure_default_brushes
    ensure_default_brushes()

    return FaceDrawCanvas()


def test_init_blank_canvas(canvas):
    assert canvas._canvas.shape == (CANVAS_SIZE, CANVAS_SIZE, 4)
    assert not canvas.has_content


def test_clear_empty_canvas(canvas):
    canvas.clear()
    assert not canvas.has_content


def test_push_undo_and_restore(canvas):
    # Directly modify canvas to simulate drawing
    canvas._push_undo()
    canvas._canvas[:, :, 3] = 255
    assert canvas.has_content

    canvas.undo()
    assert not canvas.has_content  # restored to blank


def test_push_undo_fifo_eviction(canvas):
    """After MAX_UNDO pushes, oldest entry is evicted."""
    # MAX_UNDO = 20, push MAX_UNDO+1 times
    for i in range(MAX_UNDO + 1):
        canvas._push_undo()
        canvas._canvas[10, 10 + i, 3] = 255

    # Should still have only MAX_UNDO entries
    assert len(canvas._undo_stack) == MAX_UNDO


def test_undo_empty_stack(canvas):
    """Undo on empty stack should not crash."""
    canvas.undo()
    assert not canvas.has_content


def test_clear_pushes_undo(canvas):
    assert len(canvas._undo_stack) == 0
    canvas.clear()
    assert len(canvas._undo_stack) == 1


def test_has_content_detects_alpha(canvas):
    assert not canvas.has_content
    canvas._canvas[256, 256, 0] = 255  # Blue channel, no alpha
    assert not canvas.has_content
    canvas._canvas[256, 256, 1] = 255  # Green, still no alpha
    assert not canvas.has_content
    canvas._canvas[100, 100, 3] = 128  # Alpha channel (BGRA index 3)
    assert canvas.has_content


# ── _frame_to_canvas ──

def test_frame_to_canvas_maps_center(canvas):
    """Map a face quad centered on the canvas to verify the mapping."""
    quad = np.array([
        [100, 100], [400, 100], [400, 400], [100, 400]
    ], dtype=np.float32)
    canvas.begin_stroke(quad)
    # Center of the face quad (250, 250) → center of canvas (CANVAS_SIZE/2)
    half = CANVAS_SIZE // 2
    cx, cy = canvas._frame_to_canvas((250, 250))
    assert half - 60 < cx < half + 60
    assert half - 60 < cy < half + 60


def test_frame_to_canvas_clamps(canvas):
    quad = np.array([
        [0, 0], [200, 0], [200, 200], [0, 200]
    ], dtype=np.float32)
    canvas.begin_stroke(quad)
    # Point way outside should be clamped
    cx, cy = canvas._frame_to_canvas((-9999, -9999))
    assert 0 <= cx < CANVAS_SIZE
    assert 0 <= cy < CANVAS_SIZE


# ── begin/point/end stroke flow ──

def test_begin_stroke_sets_transform(canvas):
    quad = np.array([
        [50, 50], [400, 60], [390, 400], [60, 390]
    ], dtype=np.float32)
    canvas.begin_stroke(quad)
    assert canvas._M_inv is not None
    assert canvas._prev_canvas_pt is None


def test_end_stroke_clears_state(canvas):
    quad = np.array([
        [50, 50], [400, 60], [390, 400], [60, 390]
    ], dtype=np.float32)
    canvas.begin_stroke(quad)
    canvas.end_stroke()
    assert canvas._M_inv is None
    assert canvas._prev_canvas_pt is None


def test_stroke_leaves_mark(canvas):
    """A complete stroke should leave marks on the canvas."""
    quad = np.array([
        [100, 100], [400, 100], [400, 400], [100, 400]
    ], dtype=np.float32)
    canvas.begin_stroke(quad)
    # Draw a line across the face
    for t in range(10):
        x = 150 + t * 20
        y = 250
        canvas.add_stroke_point((x, y))
    canvas.end_stroke()
    assert canvas.has_content


def test_add_stroke_point_without_begin(canvas):
    """Should not crash if no begin_stroke called."""
    canvas.add_stroke_point((200, 200))
    # No crash, no content
    assert not canvas.has_content


# ── get_result ──

def test_get_result_empty(canvas):
    assert canvas.get_result() is None


def test_get_result_after_stroke(canvas):
    quad = np.array([
        [100, 100], [400, 100], [400, 400], [100, 400]
    ], dtype=np.float32)
    canvas.begin_stroke(quad)
    for t in range(10):
        canvas.add_stroke_point((150 + t * 20, 250))
    canvas.end_stroke()

    result = canvas.get_result()
    assert result is not None
    assert result.shape[2] == 4  # RGBA
    # Result should be cropped tight (smaller than full canvas)
    assert result.shape[0] < CANVAS_SIZE or result.shape[1] < CANVAS_SIZE
