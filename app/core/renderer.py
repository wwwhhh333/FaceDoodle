# 仿射变换与贴纸 Alpha 融合渲染

import cv2
import numpy as np


def overlay_sticker(frame, sticker, anchor, angle, scale):
    """
    底层渲染函数：处理贴纸的缩放、旋转和 Alpha 融合
    """
    # 1. 基础缩放：根据人脸宽度调整贴纸大小
    h, w = sticker.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w <= 0 or new_h <= 0: return frame

    resized_sticker = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. 旋转：根据人脸倾斜角度旋转贴纸
    center = (new_w // 2, new_h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 使用透明填充旋转后的空白区域
    rotated_sticker = cv2.warpAffine(resized_sticker, matrix, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))

    # 3. 计算坐标：将锚点对齐到贴纸中心
    x_offset = int(anchor[0] - new_w // 2)
    y_offset = int(anchor[1] - new_h // 2)

    # 4. Alpha 混合 (RGBA 贴合到 BGR 帧)
    for c in range(0, 3):
        # 提取贴纸的 RGB 通道和 Alpha 通道 (0-1 归一化)
        alpha = rotated_sticker[:, :, 3] / 255.0
        sticker_color = rotated_sticker[:, :, c]

        # 确定在主帧上的有效绘制区域（防止贴纸超出边界导致溢出错误）
        y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + new_h)
        x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + new_w)

        # 裁剪对应区域的贴纸数据
        st_y1, st_y2 = max(0, -y_offset), min(new_h, frame.shape[0] - y_offset)
        st_x1, st_x2 = max(0, -x_offset), min(new_w, frame.shape[1] - x_offset)

        if x1 < x2 and y1 < y2:
            frame[y1:y2, x1:x2, c] = (
                    alpha[st_y1:st_y2, st_x1:st_x2] * sticker_color[st_y1:st_y2, st_x1:st_x2] +
                    (1 - alpha[st_y1:st_y2, st_x1:st_x2]) * frame[y1:y2, x1:x2, c]
            )

    return frame


def render_scene(frame, face_landmarks, active_content):
    """
    主渲染调度函数
    :param frame: 原始摄像头画面 (BGR)
    :param face_landmarks: face_mesh.py 提取的关键点字典 (含 'forehead', 'nose_tip', 'angle', 'face_width' 等)
    :param active_content: 从 result_queue 获取的数据包，含 'sticker' 和 'location'
    """
    if active_content is None or active_content['sticker'] is None:
        return frame

    sticker = active_content['sticker']
    location = active_content['location']

    # 提取全局人脸状态
    angle = face_landmarks.get('angle', 0)
    face_width = face_landmarks.get('face_width', 200)

    # --- 空间匹配逻辑：根据 Agent 决策选择锚点和缩放比例 ---
    anchor = None
    scale_factor = 1.0

    if location == "forehead":
        # 适用于：皇冠、帽子、发卡
        anchor = face_landmarks['forehead']
        scale_factor = (face_width / sticker.shape[1]) * 1.5  # 贴纸宽度约为脸宽 1.5 倍

    elif location == "eyes":
        # 适用于：眼罩、眼镜
        # 取双眼中心点作为锚点
        anchor = face_landmarks['mid_eyes']
        scale_factor = (face_width / sticker.shape[1]) * 1.1

    elif location == "nose":
        # 适用于：小丑红鼻子、猪鼻子
        anchor = face_landmarks['nose_tip']
        scale_factor = (face_width / sticker.shape[1]) * 0.4  # 鼻子贴纸通常较小

    elif location == "mouth":
        # 适用于：胡须、口罩、搞怪嘴巴
        anchor = face_landmarks['mouth_center']
        scale_factor = (face_width / sticker.shape[1]) * 0.8

    elif location == "cheek":
        # 适用于：腮红、面纹
        anchor = face_landmarks['left_cheek']  # 默认贴左脸，或由 Agent 细分左右
        scale_factor = (face_width / sticker.shape[1]) * 0.3

    # --- 执行渲染 ---
    if anchor is not None:
        try:
            frame = overlay_sticker(frame, sticker, anchor, angle, scale_factor)
        except Exception as e:
            print(f"[Renderer] 渲染出错: {e}")

    return frame