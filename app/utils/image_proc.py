# 图像平滑 格式转换 等

import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)


def load_rgba_sticker(filepath):
    """
    加载透明背景贴纸
    返回带有 4 个通道的 BGRA 图像
    """
    if not os.path.exists(filepath):
        log.error("找不到贴纸文件 %s", filepath)
        return None

    raw = np.fromfile(filepath, dtype=np.uint8)
    sticker = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

    if sticker is None:
        log.warning("无法读取贴纸文件 %s", filepath)
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
            log.warning("%s 背景抠图失败，图片可能几乎全白。", filepath)
            return None

        return bgra

    log.warning("%s 不是支持的贴纸图像格式。", filepath)
    return None


def list_system_fonts():
    """Return list of (font_name, font_path) for common Chinese/English fonts."""
    import platform
    fonts = []
    if platform.system() == "Windows":
        font_dir = r"C:\Windows\Fonts"
        candidates = [
            ("微软雅黑", "msyh.ttc"),
            ("微软雅黑 Bold", "msyhbd.ttc"),
            ("黑体", "simhei.ttf"),
            ("宋体", "simsun.ttc"),
            ("楷体", "simkai.ttf"),
            ("仿宋", "simfang.ttf"),
            ("Arial", "arial.ttf"),
            ("Times New Roman", "times.ttf"),
            ("Courier New", "cour.ttf"),
            ("Impact", "impact.ttf"),
            ("Comic Sans MS", "comic.ttf"),
        ]
        for name, filename in candidates:
            path = os.path.join(font_dir, filename)
            if os.path.exists(path):
                fonts.append((name, path))
    # Fallback: system default
    if not fonts:
        fonts.append(("Default", None))
    return fonts


def render_text_sticker(text, font_path, font_size, color_bgr, stroke_width=0, stroke_color_bgr=None):
    """Render text to a tight-cropped BGRA sticker image using PIL.

    Returns a BGRA numpy array, or None on failure.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.error("Pillow 未安装，无法渲染文字贴纸")
        return None

    if not text or not text.strip():
        return None

    # Convert BGR → RGB for PIL
    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
    color_rgb = (r, g, b)
    stroke_rgb = None
    if stroke_width > 0 and stroke_color_bgr is not None:
        stroke_rgb = (int(stroke_color_bgr[2]), int(stroke_color_bgr[1]), int(stroke_color_bgr[0]))

    # Load font
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Measure text
    dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0] + stroke_width * 2 + 4
    text_h = bbox[3] - bbox[1] + stroke_width * 2 + 4

    # Render
    img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((stroke_width + 2 - bbox[0], stroke_width + 2 - bbox[1]),
              text, font=font, fill=color_rgb, stroke_width=stroke_width,
              stroke_fill=stroke_rgb)

    # Convert PIL RGBA → BGRA numpy
    rgba = np.array(img, dtype=np.uint8)
    # PIL is RGBA, OpenCV expects BGRA
    bgra = np.zeros_like(rgba)
    bgra[:, :, 0] = rgba[:, :, 2]  # B ← R
    bgra[:, :, 1] = rgba[:, :, 1]  # G ← G
    bgra[:, :, 2] = rgba[:, :, 0]  # R ← B
    bgra[:, :, 3] = rgba[:, :, 3]  # A ← A
    return bgra
