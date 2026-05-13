import json
import os
import uuid
import shutil
import threading
from datetime import datetime, timezone

import cv2
import numpy as np

GALLERY_DIR = "assets/gallery"
INDEX_PATH = os.path.join(GALLERY_DIR, "index.json")
THUMB_SIZE = 120

_index_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(GALLERY_DIR, exist_ok=True)


def _load_index():
    _ensure_dir()
    if not os.path.exists(INDEX_PATH):
        return {"stickers": []}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"stickers": []}


def _save_index(data):
    _ensure_dir()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_sticker(sticker_bgra, metadata):
    sticker_id = str(uuid.uuid4())
    image_name = f"{sticker_id}.png"
    thumb_name = f"{sticker_id}_thumb.png"
    image_path = os.path.join(GALLERY_DIR, image_name)
    thumb_path = os.path.join(GALLERY_DIR, thumb_name)

    _ensure_dir()
    cv2.imwrite(image_path, sticker_bgra)

    thumb = _make_thumbnail(sticker_bgra)
    cv2.imwrite(thumb_path, thumb)

    entry = {
        "id": sticker_id,
        "prompt": metadata.get("prompt", ""),
        "region": metadata.get("location", "forehead_top"),
        "scale": metadata.get("scale", 1.0),
        "image": image_name,
        "thumb": thumb_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "favorite": False,
    }
    group_id = metadata.get("group_id")
    if group_id:
        entry["group_id"] = group_id

    with _index_lock:
        index = _load_index()
        index["stickers"].append(entry)
        _save_index(index)

    return sticker_id


def load_gallery():
    return _load_index().get("stickers", [])


def load_index_data():
    """Return (stickers, groups) from a single disk read."""
    index = _load_index()
    return index.get("stickers", []), index.get("groups", [])


def save_group(group_name, member_ids, group_id=None):
    """Create or update a sticker group. Returns group_id."""
    if group_id is None:
        group_id = str(uuid.uuid4())
    with _index_lock:
        index = _load_index()
        groups = index.setdefault("groups", [])
        existing = next((g for g in groups if g["id"] == group_id), None)
        if existing:
            existing["name"] = group_name
            existing["member_ids"] = list(member_ids)
        else:
            groups.append({
                "id": group_id,
                "name": group_name,
                "member_ids": list(member_ids),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        _save_index(index)
    return group_id


def load_groups():
    """Return list of group dicts from index.json."""
    return _load_index().get("groups", [])


def get_group(group_id):
    """Return a single group dict or None."""
    for g in load_groups():
        if g["id"] == group_id:
            return g
    return None


def get_sticker(sticker_id):
    index = _load_index()
    for s in index.get("stickers", []):
        if s["id"] == sticker_id:
            path = os.path.join(GALLERY_DIR, s["image"])
            if os.path.exists(path):
                return cv2.imread(path, cv2.IMREAD_UNCHANGED), s
    return None, None


def get_sticker_thumb(sticker_id):
    index = _load_index()
    for s in index.get("stickers", []):
        if s["id"] == sticker_id:
            path = os.path.join(GALLERY_DIR, s["thumb"])
            if os.path.exists(path):
                return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None


def delete_sticker(sticker_id):
    with _index_lock:
        index = _load_index()
        entry = next((s for s in index["stickers"] if s["id"] == sticker_id), None)
        if entry is None:
            return False
        for key in ("image", "thumb"):
            p = os.path.join(GALLERY_DIR, entry.get(key, ""))
            if os.path.exists(p):
                os.remove(p)
        index["stickers"] = [s for s in index["stickers"] if s["id"] != sticker_id]
        groups = index.get("groups", [])
        for g in groups:
            g["member_ids"] = [m for m in g.get("member_ids", []) if m != sticker_id]
        index["groups"] = [g for g in groups if len(g.get("member_ids", [])) > 0]
        _save_index(index)
    return True


def set_favorite(sticker_id, fav):
    with _index_lock:
        index = _load_index()
        for s in index["stickers"]:
            if s["id"] == sticker_id:
                s["favorite"] = bool(fav)
                break
        _save_index(index)


def _make_thumbnail(bgra):
    h, w = bgra.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((THUMB_SIZE, THUMB_SIZE, 4), dtype=np.uint8)
    scale = THUMB_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((THUMB_SIZE, THUMB_SIZE, 4), dtype=np.uint8)
    y_off = (THUMB_SIZE - new_h) // 2
    x_off = (THUMB_SIZE - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def save_preferences(prefs):
    from app.utils.config_loader import get_config, load_config
    cfg = load_config()
    existing = cfg.get("preferences", {})
    existing.update(prefs)
    cfg["preferences"] = existing
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_preferences():
    from app.utils.config_loader import get_config
    return get_config().get("preferences", {})


def save_sticker_adjustments(sticker_id, adjustment):
    with _index_lock:
        index = _load_index()
        for s in index["stickers"]:
            if s["id"] == sticker_id:
                s["offset_x"] = adjustment.get("offset_x", 0.0)
                s["offset_y"] = adjustment.get("offset_y", 0.0)
                s["rotation"] = adjustment.get("rotation", 0.0)
                s["scale_mult"] = adjustment.get("scale_mult", 1.0)
                break
        _save_index(index)


def get_sticker_adjustments(sticker_id):
    index = _load_index()
    for s in index["stickers"]:
        if s["id"] == sticker_id:
            return {
                "offset_x": s.get("offset_x", 0.0),
                "offset_y": s.get("offset_y", 0.0),
                "rotation": s.get("rotation", 0.0),
                "scale_mult": s.get("scale_mult", 1.0),
            }
    return None


def add_recent_prompt(prompt_text):
    prefs = load_preferences()
    recent = prefs.get("recent_prompts", [])
    if prompt_text in recent:
        recent.remove(prompt_text)
    recent.insert(0, prompt_text)
    prefs["recent_prompts"] = recent[:10]
    save_preferences(prefs)


# ══════════════════════════════════════════════════════════════════════════════
# Animation clip persistence
# ══════════════════════════════════════════════════════════════════════════════

ANIMATIONS_DIR = "assets/animations"
ANIMATIONS_INDEX = os.path.join(ANIMATIONS_DIR, "animations.json")


def _ensure_anim_dir():
    os.makedirs(ANIMATIONS_DIR, exist_ok=True)


def _load_anim_index():
    _ensure_anim_dir()
    if not os.path.exists(ANIMATIONS_INDEX):
        return {"clips": []}
    try:
        with open(ANIMATIONS_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"clips": []}


def _save_anim_index(data):
    _ensure_anim_dir()
    with open(ANIMATIONS_INDEX, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_animation_clip(clip_data):
    """Persist an animation clip dict (from AnimationClip.to_dict())."""
    with _index_lock:
        index = _load_anim_index()
        existing = next((c for c in index["clips"] if c["id"] == clip_data["id"]), None)
        if existing:
            existing.update(clip_data)
        else:
            index["clips"].append(clip_data)
        _save_anim_index(index)
    return clip_data["id"]


def load_animation_clips():
    """Return list of clip dicts."""
    return _load_anim_index().get("clips", [])


def delete_animation_clip(clip_id):
    with _index_lock:
        index = _load_anim_index()
        index["clips"] = [c for c in index["clips"] if c["id"] != clip_id]
        _save_anim_index(index)


# ══════════════════════════════════════════════════════════════════════════════
# Animated (sprite-sheet) sticker persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_animated_sticker(sprite_sheet_bgra, anim_meta, base_metadata):
    """Persist an animated sprite sheet sticker.

    Args:
        sprite_sheet_bgra: BGRA sprite sheet (uint8 numpy array)
        anim_meta: dict with keys frame_count, fps, cols, rows, motion_prompt
        base_metadata: dict with keys prompt, location, scale

    Returns:
        sticker_id (str)
    """
    sticker_id = str(uuid.uuid4())
    image_name = f"{sticker_id}.png"
    thumb_name = f"{sticker_id}_thumb.png"
    image_path = os.path.join(GALLERY_DIR, image_name)
    thumb_path = os.path.join(GALLERY_DIR, thumb_name)

    _ensure_dir()
    cv2.imwrite(image_path, sprite_sheet_bgra)

    # Extract first frame for thumbnail
    cols = anim_meta.get("cols", 1)
    rows = anim_meta.get("rows", 1)
    sheet_h, sheet_w = sprite_sheet_bgra.shape[:2]
    cell_h = sheet_h // rows
    cell_w = sheet_w // cols
    first_frame = sprite_sheet_bgra[0:cell_h, 0:cell_w].copy()
    thumb = _make_thumbnail(first_frame)
    cv2.imwrite(thumb_path, thumb)

    entry = {
        "id": sticker_id,
        "prompt": base_metadata.get("prompt", anim_meta.get("motion_prompt", "")),
        "region": base_metadata.get("location", "forehead_top"),
        "scale": base_metadata.get("scale", 1.0),
        "image": image_name,
        "thumb": thumb_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "favorite": False,
        "is_animated": True,
        "frame_count": anim_meta.get("frame_count", 16),
        "frame_cols": cols,
        "frame_rows": rows,
        "fps": anim_meta.get("fps", 8),
        "motion_prompt": anim_meta.get("motion_prompt", ""),
    }
    group_id = base_metadata.get("group_id")
    if group_id:
        entry["group_id"] = group_id

    with _index_lock:
        index = _load_index()
        index["stickers"].append(entry)
        _save_index(index)

    return sticker_id
