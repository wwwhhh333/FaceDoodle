# 区域透视贴合与贴纸 Alpha 融合渲染

import logging
import math
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # DEBUG fires per-frame → keep at INFO


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
        "全脸": "full_face",
        "full_face": "full_face",
        "头顶": "head_top",
        "head_top": "head_top",
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


def _warp_sticker_onto_quad(frame, sticker, quad, opacity=1.0):
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
    if opacity < 1.0:
        alpha *= opacity
    if np.max(alpha) <= 0:
        return frame

    roi = frame[min_y:max_y, min_x:max_x].astype(np.float32)
    blended_roi = sticker_roi[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    frame = frame.copy()
    frame[min_y:max_y, min_x:max_x] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    return frame


def _apply_manual_adjustment(quad, adjustment, face_w):
    """Apply translate, rotate, scale from an adjustment dict onto *quad* in place.

    Returns the modified quad for convenience.
    """
    ox = adjustment.get("offset_x", 0.0) * face_w
    oy = adjustment.get("offset_y", 0.0) * face_w
    quad += np.array([ox, oy], dtype=np.float32)

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
        quad[:] = center + (quad - center) * sm

    return quad


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

    quad = apply_head_pose_skew(quad, face_landmarks)

    # 手动调整：偏移/旋转/缩放
    if adjustment:
        face_w = max(float(face_landmarks.get("face_width", 1.0)), 1.0)
        _apply_manual_adjustment(quad, adjustment, face_w)

    try:
        opacity = adjustment.get("opacity", 1.0) if adjustment else 1.0
        frame = _warp_sticker_onto_quad(frame, sticker, quad, opacity)
    except Exception as e:
        log.error("渲染出错: %s", e)
        return frame

    # 编辑模式：绘制贴纸轮廓
    if adjustment and adjustment.get("edit_mode"):
        pts = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cx, cy = np.mean(quad, axis=0).astype(np.int32)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)

    return frame


def _hsv_to_bgr(h, s=0.8, v=0.9):
    hsv = np.uint8([[[int(h * 179) % 180, int(s * 255), int(v * 255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2]))


def render_loading_progress(frame, face_data, gen_state):
    elapsed = gen_state.get_elapsed()
    if elapsed <= 0:
        return frame
    if not face_data or "face_width" not in face_data:
        return frame

    t = time.time()

    left_cheek = face_data.get("left_cheek")
    right_cheek = face_data.get("right_cheek")
    nose_bridge = face_data.get("nose_bridge")
    if not left_cheek or not right_cheek or not nose_bridge:
        return frame

    cx = int((left_cheek[0] + right_cheek[0]) / 2)
    cy = int(nose_bridge[1])
    radius = int(face_data["face_width"] * 0.55)
    if radius < 10:
        return frame

    hue = (t * 0.25) % 1.0
    color = _hsv_to_bgr(hue)

    pulse = 0.6 + 0.4 * math.sin(t * math.pi)
    alpha = pulse * 0.7

    start_angle = (t * 90) % 360
    end_angle = start_angle + 300

    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy), (radius, radius), 0,
                start_angle, end_angle, color, thickness=3, lineType=cv2.LINE_AA)
    frame = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

    end_rad = math.radians(end_angle)
    dot_x = cx + radius * math.cos(end_rad)
    dot_y = cy - radius * math.sin(end_rad)
    cv2.circle(frame, (int(dot_x), int(dot_y)), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(dot_x), int(dot_y)), 9, color, 2, cv2.LINE_AA)

    text = f"{int(elapsed)}s"
    text_y = cy + radius + 28
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx = cx - tw // 2
    cv2.putText(frame, text, (tx, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def render_face_mesh(frame, face_data):
    if "all_landmarks" not in face_data or "mesh_connections" not in face_data:
        return frame

    t = time.time()
    pts = face_data["all_landmarks"]
    connections = face_data["mesh_connections"]

    hue = (t * 0.25) % 1.0
    color = _hsv_to_bgr(hue, s=0.6, v=1.0)

    alpha = 0.25 + 0.15 * (math.sin(t * math.pi) * 0.5 + 0.5)

    overlay = frame.copy()
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            pt1 = (int(pts[a][0]), int(pts[a][1]))
            pt2 = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(overlay, pt1, pt2, color, 1, cv2.LINE_AA)

    frame = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
    return frame


def apply_head_pose_skew(quad, face_landmarks):
    """Apply head-pose deformation to a sticker quad.

    Prefers the accurate solvePnP-based 3D projection when rvec/tvec/camera_matrix
    are present in *face_landmarks*.  Falls back to a 2D yaw/pitch heuristic
    otherwise (e.g. when solvePnP failed or the camera frame has poor lighting).
    """
    # Try the accurate 3D pose method first
    if face_landmarks.get("rvec") is not None:
        deformed = _deform_quad_by_pose(np.copy(quad), face_landmarks)
        if not np.array_equal(quad, deformed):
            return deformed
    return _apply_head_pose_skew(quad, face_landmarks)


def blend_sticker_views(view_a, view_b, t):
    """Cross-fade two RGBA images. t=0 → all a, t=1 → all b."""
    return (view_a.astype(np.float32) * (1.0 - t) + view_b.astype(np.float32) * t).astype(np.uint8)


def select_view_sticker(instance, yaw_deg):
    """Select or blend sticker views based on head yaw angle.

    yaw_deg > 0: head turned right → left side of sticker visible → use left_45
    yaw_deg < 0: head turned left → right side of sticker visible → use right_45
    """
    views = instance.get("views")
    if not views:
        return instance["sticker"]

    front = views.get("front", instance["sticker"])
    ayaw = abs(yaw_deg)

    if ayaw < 15:
        return front
    elif ayaw < 45:
        t = (ayaw - 15.0) / 30.0
        side_key = "left_45" if yaw_deg > 0 else "right_45"
        side = views.get(side_key, front)
        return blend_sticker_views(front, side, t)
    else:
        side_key = "left_45" if yaw_deg > 0 else "right_45"
        return views.get(side_key, front)


def composite_stickers_to_merged(active_stickers, adjustments, face_data):
    """Merge multiple sticker instances into a single composite RGBA image.

    Two-pass approach:
    1. Compute all canvas quads via full_face perspective, find bounding box.
    2. Expand canvas to encompass all quads (maintaining full_face aspect ratio),
       shift quads, warp and composite.

    Returns (merged_image, location, placement_scale) where placement_scale
    accounts for canvas expansion so content aligns correctly when placed.
    """
    if not active_stickers:
        log.debug("没有贴纸实例")
        return None, None, 1.0, 0.0, 0.0
    if not face_data or "nose_tip" not in face_data:
        log.debug("无人脸数据或缺少 nose_tip")
        return None, None, 1.0, 0.0, 0.0

    face_w = max(float(face_data.get("face_width", 1.0)), 1.0)

    full_rect = face_data.get("landmark_rects", {}).get("full_face")
    if not full_rect:
        log.debug("缺少 full_face 区域")
        return None, None, 1.0, 0.0, 0.0
    _, _, fw, fh = full_rect
    full_aspect = fw / max(float(fh), 0.001)

    canvas_scale = 3.0
    init_canvas_w = int(fw * canvas_scale)
    init_canvas_h = int(fh * canvas_scale)

    full_quad = _build_location_quad(face_data, "full_face", (init_canvas_h, init_canvas_w, 4), 1.0)
    if full_quad is None:
        log.warning("无法构建 full_face quad")
        return None, None, 1.0, 0.0, 0.0
    full_quad = apply_head_pose_skew(full_quad, face_data)

    dst_rect = np.array([
        [0, 0], [init_canvas_w - 1, 0],
        [init_canvas_w - 1, init_canvas_h - 1], [0, init_canvas_h - 1],
    ], dtype=np.float32)
    frame_to_canvas = cv2.getPerspectiveTransform(full_quad.astype(np.float32), dst_rect)

    # ── Pass 1: compute all canvas quads, collect sticker data ──
    sticker_data = []  # (sticker, canvas_quad)

    for instance in active_stickers:
        sticker = instance.get("sticker")
        if sticker is None or sticker.shape[2] != 4:
            continue

        location = instance.get("location", "forehead_top")
        quad = _build_location_quad(face_data, location, sticker.shape[:2], instance.get("scale", 1.0))
        if quad is None:
            continue

        adj = adjustments.get(instance["instance_id"],
                              {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0})

        try:
            quad = apply_head_pose_skew(quad, face_data)
            _apply_manual_adjustment(quad, adj, face_w)

            quad_h = np.hstack([quad, np.ones((4, 1), dtype=np.float32)])
            canvas_pts = (frame_to_canvas @ quad_h.T).T
            canvas_quad = np.zeros((4, 2), dtype=np.float32)
            canvas_quad[:, 0] = canvas_pts[:, 0] / np.maximum(canvas_pts[:, 2], 1e-6)
            canvas_quad[:, 1] = canvas_pts[:, 1] / np.maximum(canvas_pts[:, 2], 1e-6)

            sticker_data.append((sticker, canvas_quad))
        except Exception as e:
            log.error("贴纸预处理异常: %s", e)

    if not sticker_data:
        log.debug("没有有效的贴纸数据")
        return None, None, 1.0, 0.0, 0.0

    # Find bounding box of all canvas quads
    all_pts = np.vstack([cq for _, cq in sticker_data])
    min_x = float(np.min(all_pts[:, 0]))
    min_y = float(np.min(all_pts[:, 1]))
    max_x = float(np.max(all_pts[:, 0]))
    max_y = float(np.max(all_pts[:, 1]))

    # Expand bbox to match full_face aspect ratio
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    if bbox_w / max(bbox_h, 0.001) < full_aspect:
        new_w = bbox_h * full_aspect
        expand = (new_w - bbox_w) / 2.0
        min_x -= expand
        max_x += expand
    else:
        new_h = bbox_w / max(full_aspect, 0.001)
        expand = (new_h - bbox_h) / 2.0
        min_y -= expand
        max_y += expand

    # Small padding
    pad = max(max_x - min_x, max_y - min_y) * 0.02
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    final_canvas_w = max(int(max_x - min_x + 1), 1)
    final_canvas_h = max(int(max_y - min_y + 1), 1)

    # Placement scale: makes quad large enough for all content
    placement_scale = final_canvas_h / max(init_canvas_h, 1)

    # Compute offset so full_face content maps to the correct position.
    # In the merged sticker, full_face content occupies rows [-min_y, -min_y+init_canvas_h],
    # but after scaling by placement_scale, the content center shifts.
    # offset = fh * (scale/2 - 1/2 + scale * min / final_canvas_wh) / face_w
    offset_x = (fw / face_w) * (placement_scale / 2.0 - 0.5 + placement_scale * min_x / max(final_canvas_w, 1))
    offset_y = (fh / face_w) * (placement_scale / 2.0 - 0.5 + placement_scale * min_y / max(final_canvas_h, 1))

    # Shift all quads into the expanded canvas
    shift = np.array([min_x, min_y], dtype=np.float32)
    for i in range(len(sticker_data)):
        sticker_data[i] = (sticker_data[i][0], sticker_data[i][1] - shift)

    # ── Pass 2: warp and composite onto expanded canvas ──
    merged = np.zeros((final_canvas_h, final_canvas_w, 4), dtype=np.uint8)
    composited_count = 0

    for sticker, canvas_quad in sticker_data:
        try:
            sh, sw = sticker.shape[:2]
            src = np.array([
                [0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1],
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(src, canvas_quad.astype(np.float32))
            warped = cv2.warpPerspective(
                sticker, matrix, (final_canvas_w, final_canvas_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )

            fg_alpha = warped[:, :, 3].astype(np.float32) / 255.0
            merged_f = merged.astype(np.float32)
            warped_f = warped.astype(np.float32)
            for c in range(3):
                merged_f[:, :, c] = warped_f[:, :, c] * fg_alpha + merged_f[:, :, c] * (1.0 - fg_alpha)
            merged_f[:, :, 3] = warped_f[:, :, 3] + merged_f[:, :, 3] * (1.0 - fg_alpha)
            merged = np.clip(merged_f, 0, 255).astype(np.uint8)

            composited_count += 1
        except Exception as e:
            log.error("贴纸合成异常: %s", e)

    if composited_count == 0:
        log.warning("没有成功合成任何贴纸")
        return None, None, 1.0, 0.0, 0.0

    log.debug("合成完成: %d 枚 → %dx%d (scale=%.2f, offset=(%.3f,%.3f))",
              composited_count, final_canvas_w, final_canvas_h,
              placement_scale, offset_x, offset_y)
    return merged, "full_face", placement_scale, offset_x, offset_y
