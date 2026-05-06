import json
import os

import cv2
import numpy as np

BRUSH_DIR = "assets/brushes"
CONFIG_PATH = os.path.join(BRUSH_DIR, "brushes.json")

_tip_cache = {}

DEFAULT_BRUSHES = [
    {
        "id": "hard_round",
        "name": "硬圆笔",
        "tip": "hard_round.png",
        "spacing": 0.3,
        "pressure_size": True,
        "pressure_opacity": False,
        "scatter": 0.0,
    },
    {
        "id": "soft_round",
        "name": "软圆笔",
        "tip": "soft_round.png",
        "spacing": 0.25,
        "pressure_size": True,
        "pressure_opacity": True,
        "scatter": 0.0,
    },
]

PRESSURE_MIN_RATIO = 0.2


def ensure_default_brushes():
    os.makedirs(BRUSH_DIR, exist_ok=True)

    for b in DEFAULT_BRUSHES:
        tip_path = os.path.join(BRUSH_DIR, b["tip"])
        if not os.path.exists(tip_path):
            soft = (b["id"] == "soft_round")
            tip_img = _generate_default_tip(128, soft=soft)
            cv2.imwrite(tip_path, tip_img)

    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"brushes": DEFAULT_BRUSHES}, f, ensure_ascii=False, indent=2)


def load_brush_config():
    ensure_default_brushes()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("brushes", [])
    except (json.JSONDecodeError, IOError):
        return list(DEFAULT_BRUSHES)


def get_brush_by_id(brush_id):
    for b in load_brush_config():
        if b["id"] == brush_id:
            return b
    return None


def load_brush_tip(tip_filename, size):
    cache_key = (tip_filename, size)
    if cache_key in _tip_cache:
        return _tip_cache[cache_key]

    path = os.path.join(BRUSH_DIR, tip_filename)
    if not os.path.exists(path):
        print(f"[Brush] 警告: 笔刷蒙版文件不存在: {path}")
        return None

    # 用 imdecode 代替 imread，解决 Windows 上 OpenCV 不支持中文路径的问题
    raw = np.fromfile(path, dtype=np.uint8)
    tip = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if tip is None:
        print(f"[Brush] 警告: 无法读取笔刷蒙版: {path}")
        return None
    if tip.ndim == 2:
        # 灰度图 → alpha 通道 (白=不透明, 黑=透明)
        alpha = tip
        tip = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
        tip[:, :, 3] = alpha
    elif tip.shape[2] == 2:
        # 灰度+alpha → 取第二通道作为 alpha
        alpha_ch = tip[:, :, 1]
        rgb = np.zeros((tip.shape[0], tip.shape[1], 4), dtype=np.uint8)
        rgb[:, :, 3] = alpha_ch
        tip = rgb
    elif tip.shape[2] == 3:
        # BGR → 补全不透明 alpha
        alpha = np.full((tip.shape[0], tip.shape[1], 1), 255, dtype=np.uint8)
        tip = np.dstack([tip, alpha])

    d = max(size * 2 + 1, 3)
    tip = cv2.resize(tip, (d, d), interpolation=cv2.INTER_AREA)
    _tip_cache[cache_key] = tip
    return tip


def stamp_brush(canvas, cx, cy, tip, color, opacity, eraser=False):
    """Alpha-composite brush tip onto canvas at (cx, cy).

    canvas: (H, W, 4) uint8 BGRA
    tip:    (D, D, 4) uint8 BGRA brush stamp
    color:  (B, G, R, A) uint8
    opacity: 0.0-1.0 overall strength multiplier
    eraser: if True, subtract alpha instead of adding color
    """
    d = tip.shape[0]
    half = d // 2
    ch, cw = canvas.shape[1], canvas.shape[0]

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + d
    y2 = y1 + d

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = d - max(0, x2 - cw)
    sy2 = d - max(0, y2 - ch)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(cw, x2)
    y2 = min(ch, y2)

    if x1 >= x2 or y1 >= y2 or sx1 >= sx2 or sy1 >= sy2:
        return

    tip_roi = tip[sy1:sy2, sx1:sx2]
    alpha = tip_roi[:, :, 3].astype(np.float32) / 255.0 * opacity
    roi = canvas[y1:y2, x1:x2]

    if eraser:
        roi[:, :, 3] = (roi[:, :, 3].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    else:
        alpha_3d = alpha[:, :, np.newaxis]
        roi[:, :, :3] = (np.array(color[:3], dtype=np.float32) * alpha_3d +
                         roi[:, :, :3].astype(np.float32) * (1.0 - alpha_3d)).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (alpha * 255.0).astype(np.uint8))


def stamp_line(canvas, p1, p2, tip, color, opacity, spacing_px, scatter=0.0, eraser=False):
    """Stamp brush tip at intervals along the line segment from p1 to p2."""
    if spacing_px <= 0:
        spacing_px = 1.0

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.hypot(dx, dy)

    steps = max(1, int(dist / spacing_px))
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0.5
        cx = int(p1[0] + dx * t)
        cy = int(p1[1] + dy * t)
        if scatter > 0:
            cx += int(np.random.normal(0, scatter))
            cy += int(np.random.normal(0, scatter))
        stamp_brush(canvas, cx, cy, tip, color, opacity, eraser=eraser)


def _generate_default_tip(size=128, soft=False):
    tip = np.zeros((size, size, 4), dtype=np.uint8)
    tip[:, :, :3] = 255
    cy, cx = size // 2, size // 2
    radius = size // 2 - 1

    y, x = np.ogrid[-cy:size - cy, -cx:size - cx]
    dist = np.sqrt(x * x + y * y)

    if soft:
        sigma = radius * 0.4
        alpha = np.exp(-0.5 * (dist / sigma) ** 2)
        alpha = np.clip(alpha, 0.0, 1.0)
    else:
        alpha = (dist <= radius).astype(np.float32)
        # Slight anti-aliasing at edge
        edge = np.abs(dist - radius) < 1.5
        alpha[edge] = np.clip((radius + 1.5 - dist[edge]) / 3.0, 0.0, 1.0)

    tip[:, :, 3] = (alpha * 255.0).astype(np.uint8)
    return tip
