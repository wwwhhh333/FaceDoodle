# 区域透视贴合与贴纸 Alpha 融合渲染

import cv2
import numpy as np


def _pt(value):
    return np.array(value, dtype=np.float32)


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vec / norm


def _make_quad(center, x_axis, width, height, sticker_shape):
    x_axis = _normalize(x_axis)
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

    if sticker_shape[0] > 0 and sticker_shape[1] > 0:
        sticker_ratio = sticker_shape[1] / sticker_shape[0]
        quad_ratio = width / max(height, 1e-6)
        if quad_ratio > sticker_ratio:
            height = width / sticker_ratio
        else:
            width = height * sticker_ratio

    half_w = width / 2.0
    half_h = height / 2.0

    return np.array([
        center - x_axis * half_w - y_axis * half_h,
        center + x_axis * half_w - y_axis * half_h,
        center + x_axis * half_w + y_axis * half_h,
        center - x_axis * half_w + y_axis * half_h,
    ], dtype=np.float32)


def _build_location_quad(face_landmarks, location, sticker_shape):
    face_width = max(float(face_landmarks.get('face_width', 200.0)), 1.0)

    left_eye_center = _pt(face_landmarks['left_eye_center'])
    right_eye_center = _pt(face_landmarks['right_eye_center'])
    mid_eyes = _pt(face_landmarks['mid_eyes'])
    mouth_center = _pt(face_landmarks['mouth_center'])
    x_axis = right_eye_center - left_eye_center
    vertical_axis = _normalize(mouth_center - mid_eyes)

    if location == "forehead":
        brow_left = _pt(face_landmarks['brow_left'])
        brow_right = _pt(face_landmarks['brow_right'])
        brow_center = _pt(face_landmarks['brow_center'])
        forehead = _pt(face_landmarks['forehead'])
        center = brow_center + (forehead - brow_center) * 0.75
        width = np.linalg.norm(brow_right - brow_left) * 1.5
        height = max(np.linalg.norm(forehead - brow_center) * 2.4, face_width * 0.34)
        return _make_quad(center, brow_right - brow_left, width, height, sticker_shape)

    if location == "eyes":
        eye_span = np.linalg.norm(right_eye_center - left_eye_center)
        eye_top = (_pt(face_landmarks['left_eye_upper']) + _pt(face_landmarks['right_eye_upper'])) / 2.0
        eye_bottom = (_pt(face_landmarks['left_eye_lower']) + _pt(face_landmarks['right_eye_lower'])) / 2.0
        center = (eye_top + eye_bottom) / 2.0
        width = eye_span * 1.45
        height = max(np.linalg.norm(eye_bottom - eye_top) * 3.2, face_width * 0.24)
        return _make_quad(center, x_axis, width, height, sticker_shape)

    if location == "nose":
        nose_left = _pt(face_landmarks['nose_left'])
        nose_right = _pt(face_landmarks['nose_right'])
        nose_bridge = _pt(face_landmarks['nose_bridge'])
        nose_tip = _pt(face_landmarks['nose_tip'])
        center = (nose_bridge + nose_tip) / 2.0
        width = np.linalg.norm(nose_right - nose_left) * 1.6
        height = max(np.linalg.norm(nose_tip - nose_bridge) * 2.0, face_width * 0.20)
        return _make_quad(center, nose_right - nose_left, width, height, sticker_shape)

    if location == "mouth":
        mouth_left = _pt(face_landmarks['mouth_left'])
        mouth_right = _pt(face_landmarks['mouth_right'])
        mouth_upper = _pt(face_landmarks['mouth_upper'])
        mouth_lower = _pt(face_landmarks['mouth_lower'])
        center = (mouth_upper + mouth_lower) / 2.0
        width = np.linalg.norm(mouth_right - mouth_left) * 1.35
        height = max(np.linalg.norm(mouth_lower - mouth_upper) * 2.8, face_width * 0.26)
        return _make_quad(center, mouth_right - mouth_left, width, height, sticker_shape)

    if location == "cheek":
        center = _pt(face_landmarks['left_cheek']) + vertical_axis * (face_width * 0.02)
        width = face_width * 0.28
        height = face_width * 0.22
        return _make_quad(center, x_axis, width, height, sticker_shape)

    if location == "right_cheek":
        center = _pt(face_landmarks['right_cheek']) + vertical_axis * (face_width * 0.02)
        width = face_width * 0.28
        height = face_width * 0.22
        return _make_quad(center, x_axis, width, height, sticker_shape)

    return None


def _warp_sticker_onto_quad(frame, sticker, quad):
    if sticker is None or sticker.shape[2] != 4:
        return frame

    frame_h, frame_w = frame.shape[:2]
    sticker_h, sticker_w = sticker.shape[:2]

    src = np.array([
        [0, 0],
        [sticker_w - 1, 0],
        [sticker_w - 1, sticker_h - 1],
        [0, sticker_h - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(
        sticker,
        matrix,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    if np.max(alpha) <= 0:
        return frame

    overlay = warped[:, :, :3].astype(np.float32)
    base = frame.astype(np.float32)
    blended = overlay * alpha + base * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_scene(frame, face_landmarks, active_content):
    """
    主渲染调度函数
    :param frame: 原始摄像头画面 (BGR)
    :param face_landmarks: face_mesh.py 提取的关键点字典
    :param active_content: 从 result_queue 获取的数据包，含 'sticker' 和 'location'
    """
    if active_content is None or active_content['sticker'] is None:
        return frame

    sticker = active_content['sticker']
    location = active_content['location']

    quad = _build_location_quad(face_landmarks, location, sticker.shape[:2])
    if quad is None:
        return frame

    try:
        return _warp_sticker_onto_quad(frame, sticker, quad)
    except Exception as e:
        print(f"[Renderer] 渲染出错: {e}")
        return frame
