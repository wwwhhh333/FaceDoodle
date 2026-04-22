# MediaPipe 468 关键点提取

import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self):
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

    def get_landmarks(self, rgb_frame):
        """
        处理图像并返回 AR 渲染所需的关键面部数据
        """
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

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

        return {
            "forehead": tuple(forehead),
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
