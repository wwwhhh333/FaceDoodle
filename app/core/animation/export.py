"""Export animation clips to MP4/GIF files."""

import logging
import os
import cv2
import numpy as np

from app.core.animation.clip import evaluate_clip
from app.core.renderer import render_scene

log = logging.getLogger(__name__)


def export_animation(clip, sticker, fps, output_path, face_data,
                     location, sticker_scale, progress_callback=None, format="mp4",
                     manual_adj=None):
    """Render every frame of *clip* onto *face_data* and encode to video/gif.

    Args:
        clip: AnimationClip
        sticker: RGBA numpy array
        fps: frames per second
        output_path: destination file path
        face_data: dict of MediaPipe landmarks (static — no tracking)
        location: face region string (e.g. "full_face")
        sticker_scale: base scale float
        progress_callback: callable(0.0..1.0)
        format: "mp4" or "gif"
        manual_adj: optional per-instance manual adjustment dict
    """
    total_frames = max(1, int(clip.duration * fps))
    if total_frames < 1:
        return

    frames = []
    frame_h, frame_w = 480, 640  # default render size

    for i in range(total_frames):
        t = i / fps
        anim = evaluate_clip(clip, t)
        if manual_adj:
            adj = {
                "offset_x": anim["offset_x"] + manual_adj.get("offset_x", 0.0),
                "offset_y": anim["offset_y"] + manual_adj.get("offset_y", 0.0),
                "rotation": anim["rotation"] + manual_adj.get("rotation", 0.0),
                "scale_mult": anim["scale_mult"] * manual_adj.get("scale_mult", 1.0),
                "opacity": anim.get("opacity", 1.0) * manual_adj.get("opacity", 1.0),
            }
        else:
            adj = anim
        adj["edit_mode"] = False

        # Build a single-frame face canvas
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        content = {"sticker": sticker, "location": location, "scale": sticker_scale}
        rendered = render_scene(canvas, face_data, content, adj)

        if i == 0:
            frame_h, frame_w = rendered.shape[:2]

        frames.append(rendered)

        if progress_callback:
            progress_callback((i + 1) / total_frames)

    if format == "gif":
        _write_gif(frames, output_path, fps)
    else:
        _write_mp4(frames, output_path, fps, (frame_w, frame_h))

    if progress_callback:
        progress_callback(1.0)


def _write_mp4(frames, output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        # Fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for f in frames:
        writer.write(f)
    writer.release()


def _write_gif(frames, output_path, fps):
    try:
        import imageio
        duration = 1.0 / fps
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        imageio.mimsave(output_path, rgb_frames, duration=duration, loop=0)
    except ImportError:
        log.warning("imageio 未安装，回退到 mp4")
        _write_mp4(frames, output_path.replace(".gif", ".mp4"), fps,
                    (frames[0].shape[1], frames[0].shape[0]))
