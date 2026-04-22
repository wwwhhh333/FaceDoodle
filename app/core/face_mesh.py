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

        # 辅助函数：将归一化坐标转换为像素坐标
        def get_pt(idx):
            return np.array([int(landmarks[idx].x * w), int(landmarks[idx].y * h)])

        # 获取核心节点
        nose_tip = get_pt(1)      # 鼻尖
        forehead = get_pt(10)     # 额头/眉心
        left_eye = get_pt(33)     # 左眼外角
        right_eye = get_pt(263)   # 右眼外角
        left_cheek = get_pt(234)  # 左侧脸颊外侧点
        right_cheek = get_pt(454) # 右侧脸颊外侧点

        # 计算面部宽度，作为动态调整贴纸大小的基准距离
        face_width = np.linalg.norm(right_cheek - left_cheek)

        # 计算面部倾斜角度 (基于两眼中心线的倾斜角度)
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        return {
            "nose_tip": tuple(nose_tip),
            "forehead": tuple(forehead),
            "face_width": face_width,
            "angle": angle
        }