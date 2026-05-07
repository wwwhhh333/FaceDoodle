import numpy as np
import pytest


@pytest.fixture
def blank_canvas():
    """100x100 BGRA zero canvas."""
    return np.zeros((100, 100, 4), dtype=np.uint8)


@pytest.fixture
def hard_tip():
    """Small hard-round brush tip for stamp tests."""
    size = 5
    d = size * 2 + 1
    tip = np.zeros((d, d, 4), dtype=np.uint8)
    tip[:, :, :3] = 255
    cy = cx = d // 2
    y, x = np.ogrid[-cy:d - cy, -cx:d - cx]
    mask = (x * x + y * y) <= (size * size)
    tip[:, :, 3] = (mask.astype(np.float32) * 255).astype(np.uint8)
    return tip


@pytest.fixture
def synthetic_face_landmarks():
    """Minimal face_landmarks dict mimicking MediaPipe output at 640x480."""
    w, h = 640, 480
    return {
        "face_width": 200.0,
        "model_face_width": 0.14,
        "nose_tip": np.array([320.0, 240.0], dtype=np.float32),
        "left_eye_center": np.array([280.0, 200.0], dtype=np.float32),
        "right_eye_center": np.array([360.0, 200.0], dtype=np.float32),
        "left_cheek": np.array([260.0, 260.0], dtype=np.float32),
        "right_cheek": np.array([380.0, 260.0], dtype=np.float32),
        "forehead": np.array([320.0, 160.0], dtype=np.float32),
        "chin": np.array([320.0, 340.0], dtype=np.float32),
        "nose_bridge": np.array([320.0, 220.0], dtype=np.float32),
        "landmark_rects": {
            "full_face": (210, 108, 220, 280),
            "head_top": (230, 68, 180, 60),
            "forehead_top": (240, 108, 160, 60),
            "forehead_full": (210, 108, 220, 100),
            "eyes": (230, 170, 180, 60),
            "nose": (274, 210, 92, 60),
            "mouth": (248, 280, 144, 80),
            "cheek_left": (190, 230, 100, 80),
            "cheek_right": (350, 230, 100, 80),
            "chin": (262, 340, 116, 60),
            "brows": (230, 162, 180, 30),
            "jaw": (210, 340, 220, 40),
        },
        "all_landmarks": [
            (320, 240) for _ in range(468)
        ],
        "mesh_connections": [(0, 1), (1, 2)],
        "rvec": np.zeros((3, 1), dtype=np.float32),
        "tvec": np.array([[0.0], [0.0], [1.0]], dtype=np.float32),
        "camera_matrix": np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1],
        ], dtype=np.float32),
    }


@pytest.fixture
def sample_rgba_sticker():
    """50x60 RGBA sticker with a red circle."""
    s = np.zeros((60, 50, 4), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.circle(s, (25, 30), 20, (0, 0, 255, 255), -1)
    return s
