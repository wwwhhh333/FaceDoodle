"""Sprite-sheet texture animation engine — pack/extract frames, per-instance playback."""

import math
import time

import numpy as np


def compute_grid(frame_count):
    """Near-square grid dimensions for *frame_count* frames.

    Returns (cols, rows).  Example: 16 → (4, 4), 10 → (4, 3).
    """
    cols = math.ceil(math.sqrt(frame_count))
    rows = math.ceil(frame_count / cols)
    return cols, rows


def pack_frames_to_sprite_sheet(frames):
    """Pack a list of BGRA numpy arrays into a single sprite sheet.

    All frames must be the same size (H, W).  Frames are laid out
    left-to-right, top-to-bottom in a near-square grid.

    Returns (sprite_sheet, cols, rows) — sheet is a BGRA uint8 array.
    """
    if not frames:
        raise ValueError("frames list is empty")

    h, w = frames[0].shape[:2]
    channels = frames[0].shape[2]
    cols, rows = compute_grid(len(frames))

    sheet = np.zeros((rows * h, cols * w, channels), dtype=np.uint8)

    for i, frame in enumerate(frames):
        if frame.shape[:2] != (h, w):
            raise ValueError(f"Frame {i} has shape {frame.shape}, expected ({h}, {w})")
        r = i // cols
        c = i % cols
        y0, y1 = r * h, r * h + h
        x0, x1 = c * w, c * w + w
        sheet[y0:y1, x0:x1] = frame

    return sheet, cols, rows


def extract_sprite_frame(sheet, frame_index, cols, rows):
    """Extract the *frame_index*-th frame from a sprite sheet.

    Frame 0 is top-left; frames increment left-to-right, top-to-bottom.
    Frames beyond the count wrap around (modulo).
    """
    total_frames = cols * rows
    idx = frame_index % total_frames
    r = idx // cols
    c = idx % cols

    sheet_h, sheet_w = sheet.shape[:2]
    cell_h = sheet_h // rows
    cell_w = sheet_w // cols

    y0, y1 = r * cell_h, r * cell_h + cell_h
    x0, x1 = c * cell_w, c * cell_w + cell_w
    return sheet[y0:y1, x0:x1].copy()


class TextureAnimator:
    """Tracks per-instance textured animation playback.

    Each registered instance has a start timestamp, frame count, FPS,
    and sprite-sheet grid layout.  ``get_frame_params()`` returns the
    current ``(frame_index, cols, rows)`` tuple for the caller to use
    with ``extract_sprite_frame()``.

    When *loop* is True (default) the animation loops forever.  When
    False the last frame is held after the duration elapses.
    """

    def __init__(self):
        self._entries = {}   # instance_id → dict

    def register(self, instance_id, frame_count, fps, cols, rows, loop=True):
        self._entries[instance_id] = {
            "start": time.perf_counter(),
            "frame_count": frame_count,
            "fps": float(fps),
            "cols": cols,
            "rows": rows,
            "loop": loop,
        }

    def unregister(self, instance_id):
        self._entries.pop(instance_id, None)

    def get_frame_index(self, instance_id):
        """Return current frame index (0 … frame_count-1)."""
        entry = self._entries.get(instance_id)
        if entry is None:
            return 0
        elapsed = time.perf_counter() - entry["start"]
        frame = int(elapsed * entry["fps"])
        if entry["loop"]:
            return frame % entry["frame_count"]
        return min(frame, entry["frame_count"] - 1)

    def get_frame_params(self, instance_id):
        """Return (frame_index, cols, rows) for the current tick."""
        entry = self._entries.get(instance_id)
        if entry is None:
            return 0, 1, 1
        return self.get_frame_index(instance_id), entry["cols"], entry["rows"]

    def reset(self, instance_id):
        entry = self._entries.get(instance_id)
        if entry:
            entry["start"] = time.perf_counter()

    def seek(self, instance_id, t):
        """Move the texture animation playhead to time *t* (seconds).

        Adjusts the internal start timestamp so that ``get_frame_index()``
        returns the frame at *t* on the next call.
        """
        entry = self._entries.get(instance_id)
        if entry is None:
            return
        frame = int(t * entry["fps"])
        if not entry["loop"]:
            frame = min(frame, entry["frame_count"] - 1)
        else:
            frame = frame % entry["frame_count"]
        entry["start"] = time.perf_counter() - (frame / entry["fps"])
