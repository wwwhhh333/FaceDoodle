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


def _deform_quad_by_pose(quad, face_landmarks):
    rvec = face_landmarks.get("rvec")
    tvec = face_landmarks.get("tvec")
    camera_matrix = face_landmarks.get("camera_matrix")
    if rvec is None or tvec is None or camera_matrix is None:
        return quad

    nose_pt = _pt(face_landmarks["nose_tip"])
    face_w = max(float(face_landmarks.get("face_width", 1.0)), 1.0)
    model_w = float(face_landmarks.get("model_face_width", 0.14))
    scale = model_w / face_w

    R, _ = cv2.Rodrigues(rvec)

    deformed = np.zeros_like(quad)
    for i in range(4):
        dx = float(quad[i][0] - nose_pt[0]) * scale
        dy = float(quad[i][1] - nose_pt[1]) * scale
        face_pt = np.array([dx, dy, 0.001], dtype=np.float32)
        cam_pt = tvec.ravel() + R @ face_pt
        uv = camera_matrix @ cam_pt
        deformed[i][0] = uv[0] / max(uv[2], 1e-6)
        deformed[i][1] = uv[1] / max(uv[2], 1e-6)

    if not np.isfinite(deformed).all():
        return quad

    # 合理性检查：变形后四边形的中心不能偏离原始位置太远
    orig_center = np.mean(quad, axis=0)
    deformed_center = np.mean(deformed, axis=0)
    if np.linalg.norm(deformed_center - orig_center) > face_w * 1.5:
        return quad

    w = camera_matrix[0, 2] * 2
    h = camera_matrix[1, 2] * 2
    deformed[:, 0] = np.clip(deformed[:, 0], -100, w + 100)
    deformed[:, 1] = np.clip(deformed[:, 1], -100, h + 100)

    # 近正脸用平面，转脸用透视，之间平滑过渡
    angle = float(np.linalg.norm(rvec))
    blend = float(np.clip((angle - 0.15) / 0.3, 0.0, 1.0))
    return quad * (1.0 - blend) + deformed * blend


def _apply_head_pose_skew(quad, face_landmarks):
    nose = _pt(face_landmarks["nose_tip"])
    left_cheek = _pt(face_landmarks["left_cheek"])
    right_cheek = _pt(face_landmarks["right_cheek"])
    forehead = _pt(face_landmarks["forehead"])
    chin = _pt(face_landmarks["chin"])
    face_w = max(float(face_landmarks.get("face_width", 1.0)), 1.0)

    left_dist = float(nose[0] - left_cheek[0])
    right_dist = float(right_cheek[0] - nose[0])
    total_h = left_dist + right_dist
    if total_h < 1.0:
        return quad
    yaw_ratio = (right_dist - left_dist) / total_h

    forehead_dist = float(nose[1] - forehead[1])
    chin_dist = float(chin[1] - nose[1])
    total_v = forehead_dist + chin_dist
    if total_v < 0.1:
        pitch_ratio = 0.0
    else:
        pitch_ratio = (forehead_dist - chin_dist) / total_v

    if abs(yaw_ratio) < 0.06 and abs(pitch_ratio) < 0.06:
        return quad

    nose_x = float(nose[0])
    nose_y = float(nose[1])

    # 透视模拟: 压缩远侧 + 朝面部朝向平移
    result = np.copy(quad)
    for i in range(4):
        rx = (float(quad[i][0]) - nose_x) / face_w
        ry = (float(quad[i][1]) - nose_y) / face_w

        # 偏航: 水平压缩
        result[i][0] -= rx * abs(yaw_ratio) * face_w * 0.35

        # 俯仰: 不对称透视 (近侧拉伸, 远侧压缩)
        # pitch_ratio < 0 → 抬头(下巴近): 上半压缩, 下半拉伸
        # pitch_ratio > 0 → 低头(额头近): 上半拉伸, 下半压缩
        stretch = 1.0 - pitch_ratio * np.sign(ry) * 0.35
        result[i][1] = nose_y + ry * face_w * stretch

    # 整体平移跟随面部朝向
    result[:, 0] -= yaw_ratio * face_w * 0.15
    result[:, 1] -= pitch_ratio * face_w * 0.05

    return result


def _make_quad(center, x_axis, width, height, sticker_shape):
    x_axis = _normalize(x_axis)
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

    sticker_h, sticker_w = sticker_shape[:2]
    sticker_ratio = sticker_w / max(float(sticker_h), 1.0)
    rect_ratio = width / max(float(height), 0.001)

    # 保持贴纸宽高比，用包围盒中比例更接近的维度来适配
    if rect_ratio > sticker_ratio:
        actual_h = height
        actual_w = height * sticker_ratio
    else:
        actual_w = width
        actual_h = width / max(sticker_ratio, 0.001)

    half_w = actual_w / 2.0
    half_h = actual_h / 2.0

    return np.array([
        center - x_axis * half_w - y_axis * half_h,
        center + x_axis * half_w - y_axis * half_h,
        center + x_axis * half_w + y_axis * half_h,
        center - x_axis * half_w + y_axis * half_h,
    ], dtype=np.float32)


def _build_location_quad(face_landmarks, location, sticker_shape, scale=1.0):
    left_eye_center = _pt(face_landmarks['left_eye_center'])
    right_eye_center = _pt(face_landmarks['right_eye_center'])
    x_axis = right_eye_center - left_eye_center

    rects = face_landmarks.get("landmark_rects", {})

    # location 直接就是关键点组名，支持旧的映射做兼容
    region = {
        "forehead": "forehead_full",
        "eyes": "eyes",
        "nose": "nose",
        "mouth": "mouth",
        "cheek": "cheek_left",
        "right_cheek": "cheek_right",
    }.get(location, location)

    rect = rects.get(region)
    if rect is None:
        return None

    ox, oy, ow, oh = rect
    cx = ox + ow / 2.0
    cy = oy + oh / 2.0

    return _make_quad(
        np.array([cx, cy], dtype=np.float32),
        x_axis,
        ow * scale,
        oh * scale,
        sticker_shape
    )


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

    min_x = max(int(np.floor(np.min(quad[:, 0]))) - 1, 0)
    max_x = min(int(np.ceil(np.max(quad[:, 0]))) + 1, frame_w)
    min_y = max(int(np.floor(np.min(quad[:, 1]))) - 1, 0)
    max_y = min(int(np.ceil(np.max(quad[:, 1]))) + 1, frame_h)

    if min_x >= max_x or min_y >= max_y:
        return frame

    shifted_quad = quad.copy()
    shifted_quad[:, 0] -= min_x
    shifted_quad[:, 1] -= min_y

    roi_w = max_x - min_x
    roi_h = max_y - min_y

    matrix_roi = cv2.getPerspectiveTransform(src, shifted_quad.astype(np.float32))
    sticker_roi = cv2.warpPerspective(
        sticker,
        matrix_roi,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    alpha = sticker_roi[:, :, 3:4].astype(np.float32) / 255.0
    if np.max(alpha) <= 0:
        return frame

    roi = frame[min_y:max_y, min_x:max_x].astype(np.float32)
    blended_roi = sticker_roi[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    frame = frame.copy()
    frame[min_y:max_y, min_x:max_x] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    return frame


def render_scene(frame, face_landmarks, active_content, adjustment=None):
    """
    主渲染调度函数
    :param frame: 原始摄像头画面 (BGR)
    :param face_landmarks: face_mesh.py 提取的关键点字典
    :param active_content: 从 result_queue 获取的数据包，含 'sticker' 和 'location'
    :param adjustment: 手动调整参数 dict {offset_x(相对脸宽), offset_y(相对脸宽), rotation(deg), scale_mult, edit_mode}
    """
    if active_content is None or active_content['sticker'] is None:
        return frame

    sticker = active_content['sticker']
    location = active_content['location']
    scale = float(active_content.get('scale', 1.0))

    quad = _build_location_quad(face_landmarks, location, sticker.shape[:2], scale)
    if quad is None:
        return frame

    quad = _apply_head_pose_skew(quad, face_landmarks)

    # 手动调整：偏移/旋转/缩放
    if adjustment:
        face_w = max(float(face_landmarks.get("face_width", 1.0)), 1.0)
        ox = adjustment.get("offset_x", 0.0) * face_w
        oy = adjustment.get("offset_y", 0.0) * face_w
        quad = quad + np.array([ox, oy], dtype=np.float32)

        angle = adjustment.get("rotation", 0.0)
        if abs(angle) > 0.01:
            center = np.mean(quad, axis=0)
            rad = np.radians(angle)
            c, s = np.cos(rad), np.sin(rad)
            for i in range(4):
                dx = quad[i][0] - center[0]
                dy = quad[i][1] - center[1]
                quad[i][0] = center[0] + dx * c - dy * s
                quad[i][1] = center[1] + dx * s + dy * c

        sm = adjustment.get("scale_mult", 1.0)
        if abs(sm - 1.0) > 0.001:
            center = np.mean(quad, axis=0)
            quad = center + (quad - center) * sm

    try:
        frame = _warp_sticker_onto_quad(frame, sticker, quad)
    except Exception as e:
        print(f"[Renderer] 渲染出错: {e}")
        return frame

    # 编辑模式：绘制贴纸轮廓
    if adjustment and adjustment.get("edit_mode"):
        pts = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cx, cy = np.mean(quad, axis=0).astype(np.int32)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)

    return frame
