# MediaPipe 468 关键点提取

import cv2
import mediapipe as mp
import numpy as np

SOLVEPNP_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.17, -0.04],
    [-0.08, -0.05, 0.04],
    [0.08, -0.05, 0.04],
    [-0.06, 0.10, 0.02],
    [0.06, 0.10, 0.02],
], dtype=np.float32)

SOLVEPNP_LM_INDICES = [1, 152, 33, 263, 61, 291]


class FaceDetector:
    def __init__(self):
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._miss_counter = 0
        self._cached_landmarks = None

    def get_landmarks(self, rgb_frame):
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            self._miss_counter += 1
            if self._miss_counter > 5:
                self._cached_landmarks = None
                return None
            return self._cached_landmarks

        self._miss_counter = 0

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = rgb_frame.shape

        def get_pt(idx):
            return np.array(
                [landmarks[idx].x * w, landmarks[idx].y * h],
                dtype=np.float32
            )

        def avg_pt(*indices):
            points = [get_pt(idx) for idx in indices]
            return np.mean(points, axis=0)

        forehead = get_pt(10)
        nose_tip = get_pt(1)
        nose_bridge = avg_pt(6, 168)
        nose_left = get_pt(98)
        nose_right = get_pt(327)

        left_eye_outer = get_pt(33)
        left_eye_inner = get_pt(133)
        right_eye_inner = get_pt(362)
        right_eye_outer = get_pt(263)
        left_eye_upper = avg_pt(159, 160)
        left_eye_lower = avg_pt(145, 144)
        right_eye_upper = avg_pt(386, 387)
        right_eye_lower = avg_pt(374, 373)
        left_eye_center = avg_pt(33, 133, 159, 145)
        right_eye_center = avg_pt(263, 362, 386, 374)
        mid_eyes = (left_eye_center + right_eye_center) / 2.0

        mouth_left = get_pt(61)
        mouth_right = get_pt(291)
        mouth_upper = avg_pt(13, 0)
        mouth_lower = avg_pt(14, 17)
        mouth_center = (mouth_upper + mouth_lower) / 2.0

        left_cheek = get_pt(234)
        right_cheek = get_pt(454)
        chin = get_pt(152)

        left_brow_outer = get_pt(70)
        left_brow_inner = get_pt(107)
        right_brow_inner = get_pt(336)
        right_brow_outer = get_pt(300)
        brow_left = (left_brow_outer + left_brow_inner) / 2.0
        brow_right = (right_brow_outer + right_brow_inner) / 2.0
        brow_center = (brow_left + brow_right) / 2.0

        face_width = np.linalg.norm(right_cheek - left_cheek)
        face_height = np.linalg.norm(chin - forehead)

        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(d_y, d_x))

        def _rect_from_indices(indices, margin=0.0):
            pts = np.array([get_pt(i) for i in indices])
            x, y = np.min(pts[:, 0]), np.min(pts[:, 1])
            w, h = np.max(pts[:, 0]) - x, np.max(pts[:, 1]) - y
            mw, mh = w * margin, h * margin
            return (float(x - mw), float(y - mh), float(w + mw * 2), float(h + mh * 2))

        LANDMARK_GROUPS = {
            "forehead_full": ([8, 9, 10, 67, 68, 69, 103, 104, 108, 109, 151, 299, 300, 301, 333, 334, 337, 338], 0.0),
            "forehead_top": ([8, 9, 10, 67, 68, 69, 108, 109, 151, 337, 338, 299, 300, 301], 0.1),
            "head_top": ([8, 9, 10, 108, 109, 151, 337, 338], 0.0),
            "brows": ([70, 105, 107, 336, 334, 300], 0.2),
            "eyes": ([33, 133, 159, 145, 362, 263, 386, 374, 70, 300], 0.3),
            "nose": ([1, 2, 4, 5, 6, 19, 98, 168, 195, 197, 327], 0.25),
            "mouth": ([0, 13, 14, 17, 37, 61, 78, 81, 267, 291, 308, 311, 402], 0.3),
            "cheek_left": ([36, 50, 101, 117, 118, 119, 121, 143, 203, 205, 207, 234], 0.15),
            "cheek_right": ([266, 280, 329, 330, 346, 347, 348, 349, 423, 425, 454], 0.15),
            "chin": ([152, 148, 149, 150, 175, 176, 377, 378, 379], 0.2),
            "jaw": ([172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397], 0.1),
        }

        rects = {}
        for name, (indices, margin) in LANDMARK_GROUPS.items():
            rects[name] = _rect_from_indices(indices, margin)

        # forehead 系列额外向上扩展用于猫耳/帽子
        for key in ("forehead_full", "forehead_top"):
            if key in rects:
                x, y, w, h = rects[key]
                rects[key] = (x, y - h * 0.5, w, h * 1.5)

        # head_top 用于头顶饰品 (猫耳/帽子/皇冠)，位置更高
        if "head_top" in rects:
            x, y, w, h = rects["head_top"]
            rects["head_top"] = (x, y - h * 1.2, w, h * 1.6)

        result = {
            "forehead": tuple(forehead),
            "landmark_rects": rects,
            "nose_tip": tuple(nose_tip),
            "nose_bridge": tuple(nose_bridge),
            "nose_left": tuple(nose_left),
            "nose_right": tuple(nose_right),
            "left_eye_outer": tuple(left_eye_outer),
            "left_eye_inner": tuple(left_eye_inner),
            "right_eye_inner": tuple(right_eye_inner),
            "right_eye_outer": tuple(right_eye_outer),
            "left_eye_upper": tuple(left_eye_upper),
            "left_eye_lower": tuple(left_eye_lower),
            "right_eye_upper": tuple(right_eye_upper),
            "right_eye_lower": tuple(right_eye_lower),
            "left_eye_center": tuple(left_eye_center),
            "right_eye_center": tuple(right_eye_center),
            "mid_eyes": tuple(mid_eyes),
            "mouth_left": tuple(mouth_left),
            "mouth_right": tuple(mouth_right),
            "mouth_upper": tuple(mouth_upper),
            "mouth_lower": tuple(mouth_lower),
            "mouth_center": tuple(mouth_center),
            "left_cheek": tuple(left_cheek),
            "right_cheek": tuple(right_cheek),
            "chin": tuple(chin),
            "left_brow_outer": tuple(left_brow_outer),
            "left_brow_inner": tuple(left_brow_inner),
            "right_brow_inner": tuple(right_brow_inner),
            "right_brow_outer": tuple(right_brow_outer),
            "brow_left": tuple(brow_left),
            "brow_right": tuple(brow_right),
            "brow_center": tuple(brow_center),
            "face_width": float(face_width),
            "face_height": float(face_height),
            "angle": float(angle)
        }

        image_points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in SOLVEPNP_LM_INDICES
        ], dtype=np.float32)

        focal = float(w)
        camera_matrix = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1],
        ], dtype=np.float32)

        try:
            ok, rvec, tvec = cv2.solvePnP(
                SOLVEPNP_MODEL_POINTS, image_points, camera_matrix,
                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                result["rvec"] = rvec
                result["tvec"] = tvec
                result["camera_matrix"] = camera_matrix
        except cv2.error:
            pass

        result["model_face_width"] = 0.14

        self._cached_landmarks = result
        return result

    def close(self):
        self.face_mesh.close()
