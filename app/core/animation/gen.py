"""Orchestrate AI texture animation generation: ComfyUI → rembg → sprite sheet."""

import logging
import os
import tempfile

import cv2
import numpy as np

from app.core.animation.texture import pack_frames_to_sprite_sheet
from app.utils.config_loader import build_positive_prompt

log = logging.getLogger(__name__)


def _preprocess_sticker(sticker_bgra, canvas_size=1024):
    """Composite RGBA sticker onto white canvas for ComfyUI input.

    Returns path to the temporary preprocessed PNG.
    """
    h, w = sticker_bgra.shape[:2]
    sf = min((canvas_size - 100) / max(w, h), 1.0)
    new_w, new_h = int(w * sf), int(h * sf)
    resized = cv2.resize(sticker_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2

    alpha = resized[:, :, 3:4] / 255.0
    rgb = resized[:, :, :3]
    roi = canvas[y_off:y_off + new_h, x_off:x_off + new_w]
    blended = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = blended

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="assets/temp")
    cv2.imwrite(tmp.name, canvas)
    return tmp.name


def _remove_background(frame_bgr):
    """Remove background from an RGB/BGR frame using rembg.  Returns BGRA."""
    try:
        from rembg import remove
        rgba = remove(frame_bgr, alpha_matting=False)
        if rgba.shape[2] == 3:
            alpha = np.ones((rgba.shape[0], rgba.shape[1], 1), dtype=np.uint8) * 255
            rgba = np.concatenate([rgba, alpha], axis=2)
        return rgba
    except ImportError:
        log.info("rembg 未安装，动画帧将使用白色背景（不影响功能）")
        h, w = frame_bgr.shape[:2]
        alpha = np.ones((h, w, 1), dtype=np.uint8) * 255
        if frame_bgr.shape[2] == 4:
            return frame_bgr
        return np.concatenate([frame_bgr, alpha], axis=2)


def generate_animated_sticker(sticker_bgra, client, motion_prompt,
                               frame_count=16, fps=8, seed=None,
                               progress_callback=None):
    """Run the full AI texture animation pipeline.

    Args:
        sticker_bgra: source RGBA sticker (uint8 numpy array)
        client: ComfyClient instance
        motion_prompt: text description of the motion (e.g. "猫耳轻轻飘动")
        frame_count: number of frames to generate
        fps: playback speed
        seed: optional ComfyUI seed for reproducibility
        progress_callback: callable(0.0..1.0) for progress reporting

    Returns:
        (sprite_sheet_bgra, anim_meta) or (None, None) on failure
    """
    # 1. Preprocess
    if progress_callback:
        progress_callback(0.05)
    preprocessed_path = _preprocess_sticker(sticker_bgra)

    # 2. Call ComfyUI AnimateDiff
    if progress_callback:
        progress_callback(0.1)
    try:
        frame_paths = client.generate_animated_frames(
            prompt_text=build_positive_prompt(motion_prompt),
            workflow_name="animatediff_workflow_api.json",
            input_image_path=preprocessed_path,
            frame_count=frame_count,
            fps=fps,
            seed=seed,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(preprocessed_path)
        except OSError:
            pass

    if not frame_paths:
        if progress_callback:
            progress_callback(1.0)
        return None, None

    # 3. Post-process each frame
    rgba_frames = []
    total = len(frame_paths)
    for i, path in enumerate(frame_paths):
        raw = np.fromfile(path, dtype=np.uint8)
        frame_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        rgba = _remove_background(frame_bgr)
        rgba_frames.append(rgba)
        if progress_callback:
            progress_callback(0.1 + 0.7 * (i + 1) / total)

    if not rgba_frames:
        return None, None

    # Resize all frames to match the first frame's dimensions
    target_h, target_w = rgba_frames[0].shape[:2]
    aligned = []
    for f in rgba_frames:
        if f.shape[:2] != (target_h, target_w):
            f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
        aligned.append(f)

    # 4. Pack into sprite sheet
    if progress_callback:
        progress_callback(0.85)
    sprite_sheet, cols, rows = pack_frames_to_sprite_sheet(aligned)

    anim_meta = {
        "frame_count": len(aligned),
        "fps": fps,
        "cols": cols,
        "rows": rows,
        "motion_prompt": motion_prompt,
    }

    if progress_callback:
        progress_callback(1.0)

    return sprite_sheet, anim_meta
