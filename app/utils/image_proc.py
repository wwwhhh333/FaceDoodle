# 图像平滑 格式转换 等

import os

import cv2
import numpy as np


def load_rgba_sticker(filepath):
    """
    加载透明背景贴纸
    返回带有 4 个通道的 BGRA 图像
    """
    if not os.path.exists(filepath):
        print(f"Error: 找不到贴纸文件 {filepath}")
        return None

    raw = np.fromfile(filepath, dtype=np.uint8)
    sticker = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

    if sticker is None:
        print(f"Warning: 无法读取贴纸文件 {filepath}")
        return None

    if len(sticker.shape) == 2:
        sticker = cv2.cvtColor(sticker, cv2.COLOR_GRAY2BGRA)
        sticker[:, :, 3] = 255
        return sticker

    if sticker.shape[2] == 4:
        return sticker

    if sticker.shape[2] == 3:
        # ComfyUI 预览图通常是白底 RGB，这里把接近白色的背景转成透明，
        # 让白底贴纸也能进入 AR 渲染流程。
        bgr = sticker.astype(np.uint8)
        min_channel = np.min(bgr, axis=2).astype(np.float32)
        alpha = ((255.0 - min_channel) / 35.0) * 255.0
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)

        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        if np.max(alpha) == 0:
            print(f"Warning: {filepath} 背景抠图失败，图片可能几乎全白。")
            return None

        return bgra

    print(f"Warning: {filepath} 不是支持的贴纸图像格式。")
    return None
