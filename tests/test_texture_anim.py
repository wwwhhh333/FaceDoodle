"""Tests for sprite sheet packing/extraction and TextureAnimator."""
import time

import numpy as np
import pytest

from app.core.animation import (
    compute_grid,
    pack_frames_to_sprite_sheet,
    extract_sprite_frame,
    TextureAnimator,
)


def _make_frame(h, w, value):
    return np.full((h, w, 4), value, dtype=np.uint8)


class TestComputeGrid:
    def test_square(self):
        assert compute_grid(16) == (4, 4)
        assert compute_grid(9) == (3, 3)

    def test_rectangular(self):
        cols, rows = compute_grid(10)
        assert cols * rows >= 10
        assert cols == 4
        assert rows == 3

    def test_single_frame(self):
        assert compute_grid(1) == (1, 1)

    def test_odd_count(self):
        cols, rows = compute_grid(7)
        assert cols * rows >= 7
        assert cols == 3
        assert rows == 3


class TestPackExtract:
    def test_pack_single_frame(self):
        frame = _make_frame(32, 32, 100)
        sheet, cols, rows = pack_frames_to_sprite_sheet([frame])
        assert cols == 1
        assert rows == 1
        assert sheet.shape == (32, 32, 4)
        np.testing.assert_array_equal(sheet, frame)

    def test_pack_multiple_frames(self):
        frames = [_make_frame(16, 16, i) for i in range(4)]
        sheet, cols, rows = pack_frames_to_sprite_sheet(frames)
        assert cols == 2
        assert rows == 2
        assert sheet.shape == (32, 32, 4)

    def test_extract_all_frames(self):
        frames = [_make_frame(16, 16, i + 10) for i in range(16)]
        sheet, cols, rows = pack_frames_to_sprite_sheet(frames)

        for i in range(16):
            extracted = extract_sprite_frame(sheet, i, cols, rows)
            assert extracted.shape == (16, 16, 4)
            np.testing.assert_array_equal(extracted, frames[i])

    def test_extract_wraps_modulo(self):
        frames = [_make_frame(16, 16, i) for i in range(4)]
        sheet, cols, rows = pack_frames_to_sprite_sheet(frames)

        frame_4 = extract_sprite_frame(sheet, 4, cols, rows)
        np.testing.assert_array_equal(frame_4, frames[0])

    def test_empty_frames_raises(self):
        with pytest.raises(ValueError):
            pack_frames_to_sprite_sheet([])

    def test_size_mismatch_raises(self):
        frames = [_make_frame(16, 16, 0), _make_frame(32, 32, 0)]
        with pytest.raises(ValueError):
            pack_frames_to_sprite_sheet(frames)


class TestTextureAnimator:
    def test_register_and_get_frame(self):
        ta = TextureAnimator()
        ta.register("a", frame_count=16, fps=10, cols=4, rows=4)
        time.sleep(0.05)
        idx = ta.get_frame_index("a")
        assert 0 <= idx < 16

    def test_returns_zero_for_unknown(self):
        ta = TextureAnimator()
        assert ta.get_frame_index("missing") == 0
        assert ta.get_frame_params("missing") == (0, 1, 1)

    def test_unregister(self):
        ta = TextureAnimator()
        ta.register("a", frame_count=10, fps=20, cols=4, rows=3)
        ta.unregister("a")
        assert ta.get_frame_index("a") == 0

    def test_non_looping_holds_last(self):
        ta = TextureAnimator()
        ta.register("b", frame_count=4, fps=100, cols=2, rows=2, loop=False)
        time.sleep(0.05)
        idx = ta.get_frame_index("b")
        assert idx == 3  # Last frame held

    def test_looping_wraps(self):
        ta = TextureAnimator()
        ta.register("c", frame_count=4, fps=100, cols=2, rows=2, loop=True)
        time.sleep(0.05)
        idx = ta.get_frame_index("c")
        # With fps=100, after >0.04s, frame_index >= 4, so it wraps to 0+ via modulo
        assert 0 <= idx < 4

    def test_get_frame_params(self):
        ta = TextureAnimator()
        ta.register("d", frame_count=16, fps=8, cols=4, rows=4)
        idx, cols, rows = ta.get_frame_params("d")
        assert 0 <= idx < 16
        assert cols == 4
        assert rows == 4

    def test_reset(self):
        ta = TextureAnimator()
        ta.register("e", frame_count=16, fps=100, cols=4, rows=4)
        time.sleep(0.05)
        before = ta.get_frame_index("e")
        assert before > 0
        ta.reset("e")
        assert ta.get_frame_index("e") <= 0

    def test_multiple_instances(self):
        ta = TextureAnimator()
        ta.register("x", frame_count=8, fps=5, cols=4, rows=2)
        ta.register("y", frame_count=4, fps=50, cols=2, rows=2)
        x_idx = ta.get_frame_index("x")
        y_idx = ta.get_frame_index("y")
        assert 0 <= x_idx < 8
        assert 0 <= y_idx < 4
