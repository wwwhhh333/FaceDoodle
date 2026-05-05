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

    with _index_lock:
        index = _load_index()
        index["stickers"].append(entry)
        _save_index(index)

    return sticker_id


def load_gallery():
    return _load_index().get("stickers", [])


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


def add_recent_prompt(prompt_text):
    prefs = load_preferences()
    recent = prefs.get("recent_prompts", [])
    if prompt_text in recent:
        recent.remove(prompt_text)
    recent.insert(0, prompt_text)
    prefs["recent_prompts"] = recent[:10]
    save_preferences(prefs)
