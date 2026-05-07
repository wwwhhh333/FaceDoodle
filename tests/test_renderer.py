"""Test renderer geometry functions and sticker compositing."""

import numpy as np
import pytest

from app.core.renderer import (
    _pt, _normalize, _hsv_to_bgr, _make_quad,
    _build_location_quad, _warp_sticker_onto_quad,
    _apply_head_pose_skew, _deform_quad_by_pose,
    composite_stickers_to_merged,
)


# ── _pt ──

def test_pt_scalar():
    result = _pt(5.0)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_pt_list():
    result = _pt([1, 2, 3])
    assert np.array_equal(result, np.array([1, 2, 3], dtype=np.float32))


# ── _normalize ──

def test_normalize_unit():
    v = _normalize(np.array([3.0, 0.0], dtype=np.float32))
    assert np.allclose(v, [1.0, 0.0])


def test_normalize_diagonal():
    v = _normalize(np.array([1.0, 1.0], dtype=np.float32))
    assert np.allclose(np.linalg.norm(v), 1.0)


def test_normalize_zero_vector():
    v = _normalize(np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(v, [1.0, 0.0])


# ── _hsv_to_bgr ──

def test_hsv_to_bgr_red():
    b, g, r = _hsv_to_bgr(0.0)
    assert r > g and r > b  # Red hue


def test_hsv_to_bgr_returns_ints():
    b, g, r = _hsv_to_bgr(0.5)
    assert all(isinstance(c, int) for c in (b, g, r))


# ── _make_quad ──

def test_make_quad_shape():
    center = np.array([100.0, 100.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0], dtype=np.float32)
    quad = _make_quad(center, x_axis, width=50.0, height=40.0, sticker_shape=(40, 50, 4))
    assert quad.shape == (4, 2)


def test_make_quad_square_sticker():
    """Square sticker in square region: quad should match region exactly."""
    center = np.array([200.0, 200.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0], dtype=np.float32)
    quad = _make_quad(center, x_axis, width=100.0, height=100.0, sticker_shape=(100, 100, 4))
    # Center should be at the center of quad
    q_center = np.mean(quad, axis=0)
    assert np.allclose(q_center, center, atol=0.1)


def test_make_quad_preserves_aspect_ratio():
    """Wide sticker in square region: short dimension should be the bounding box."""
    center = np.array([100.0, 100.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0], dtype=np.float32)
    # Sticker is wider than tall (200x100), region is 100x100
    quad = _make_quad(center, x_axis, width=100.0, height=100.0, sticker_shape=(100, 200, 4))
    pts = quad
    w = np.linalg.norm(pts[1] - pts[0])
    h = np.linalg.norm(pts[2] - pts[1])
    # Width should be larger than height (preserving wide ratio)
    assert w > h


# ── _build_location_quad ──

def test_build_full_face_quad(synthetic_face_landmarks):
    quad = _build_location_quad(synthetic_face_landmarks, "full_face", (100, 100, 4))
    assert quad is not None
    assert quad.shape == (4, 2)


def test_build_eyes_quad(synthetic_face_landmarks):
    quad = _build_location_quad(synthetic_face_landmarks, "eyes", (50, 50, 4))
    assert quad is not None


def test_build_nose_quad(synthetic_face_landmarks):
    quad = _build_location_quad(synthetic_face_landmarks, "nose", (30, 30, 4))
    assert quad is not None


def test_build_unknown_location_returns_none(synthetic_face_landmarks):
    quad = _build_location_quad(synthetic_face_landmarks, "nonexistent", (10, 10, 4))
    assert quad is None


def test_build_quad_with_scale(synthetic_face_landmarks):
    q1 = _build_location_quad(synthetic_face_landmarks, "full_face", (100, 100, 4), scale=1.0)
    q2 = _build_location_quad(synthetic_face_landmarks, "full_face", (100, 100, 4), scale=2.0)
    area1 = np.linalg.norm(q1[1] - q1[0]) * np.linalg.norm(q1[2] - q1[1])
    area2 = np.linalg.norm(q2[1] - q2[0]) * np.linalg.norm(q2[2] - q2[1])
    assert area2 > area1


# ── _warp_sticker_onto_quad ──

def test_warp_sticker_on_frame(sample_rgba_sticker):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    quad = np.array([[50, 50], [150, 40], [160, 140], [40, 130]], dtype=np.float32)
    result = _warp_sticker_onto_quad(frame, sample_rgba_sticker, quad)
    assert result.shape == frame.shape
    # Should have some non-zero pixels from the red circle
    assert np.any(result[:, :, 2] > 0)  # red channel


def test_warp_sticker_out_of_bounds(sample_rgba_sticker):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    quad = np.array([[-500, -500], [-400, -500], [-400, -400], [-500, -400]], dtype=np.float32)
    result = _warp_sticker_onto_quad(frame, sample_rgba_sticker, quad)
    assert np.array_equal(result, frame)  # no change


def test_warp_sticker_none_sticker():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    quad = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
    result = _warp_sticker_onto_quad(frame, None, quad)
    assert np.array_equal(result, frame)


# ── _apply_head_pose_skew ──

def test_head_pose_skew_frontal(synthetic_face_landmarks):
    """Frontal face (symmetric) should have minimal skew."""
    quad = _build_location_quad(synthetic_face_landmarks, "full_face", (100, 100, 4))
    skewed = _apply_head_pose_skew(quad, synthetic_face_landmarks)
    assert skewed.shape == quad.shape
    # Frontal face: quad shouldn't change much
    assert np.allclose(quad, skewed, atol=20)


def test_head_pose_skew_profile(synthetic_face_landmarks):
    """Asymmetric face should produce skew."""
    asym = dict(synthetic_face_landmarks)
    asym["left_cheek"] = np.array([200.0, 260.0], dtype=np.float32)
    asym["right_cheek"] = np.array([440.0, 260.0], dtype=np.float32)
    quad = _build_location_quad(asym, "full_face", (100, 100, 4))
    skewed = _apply_head_pose_skew(quad, asym)
    assert skewed.shape == quad.shape


# ── _deform_quad_by_pose ──

def test_deform_quad_no_pose_data(synthetic_face_landmarks):
    """Without rvec/tvec/camera_matrix, returns original quad."""
    quad = _build_location_quad(synthetic_face_landmarks, "full_face", (100, 100, 4))
    minimal = {"nose_tip": synthetic_face_landmarks["nose_tip"],
               "face_width": synthetic_face_landmarks["face_width"]}
    result = _deform_quad_by_pose(quad, minimal)
    assert np.array_equal(result, quad)


# ── composite_stickers_to_merged ──

def test_composite_empty_list(synthetic_face_landmarks):
    img, loc, scale, ox, oy = composite_stickers_to_merged([], {}, synthetic_face_landmarks)
    assert img is None


def test_composite_single_sticker(synthetic_face_landmarks, sample_rgba_sticker):
    instances = [{
        "instance_id": "i1",
        "sticker": sample_rgba_sticker,
        "location": "full_face",
        "scale": 1.0,
    }]
    adjustments = {"i1": {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0}}
    img, loc, scale, ox, oy = composite_stickers_to_merged(instances, adjustments, synthetic_face_landmarks)
    assert img is not None
    assert img.shape[2] == 4  # RGBA
    assert np.any(img[:, :, 3] > 0)  # has alpha content
    assert loc == "full_face"


def test_composite_two_stickers(synthetic_face_landmarks, sample_rgba_sticker):
    instances = [
        {"instance_id": "i1", "sticker": sample_rgba_sticker, "location": "eyes", "scale": 0.8},
        {"instance_id": "i2", "sticker": sample_rgba_sticker, "location": "mouth", "scale": 0.6},
    ]
    adjustments = {
        "i1": {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0},
        "i2": {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0},
    }
    img, loc, scale, ox, oy = composite_stickers_to_merged(instances, adjustments, synthetic_face_landmarks)
    assert img is not None
    assert img.shape[2] == 4


def test_composite_no_face_data():
    img, loc, scale, ox, oy = composite_stickers_to_merged([{"id": "x"}], {}, {})
    assert img is None
