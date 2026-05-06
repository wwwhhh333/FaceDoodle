import os
import cv2
import numpy as np

TEMPLATE_DIR = "assets/templates"
TEMPLATE_SIZE = 256

TEMPLATE_DEFS = [
    ("full_face", "面部轮廓", "full_face"),
    ("head_top", "小皇冠", "head_top"),
    ("forehead_top", "眉心点", "forehead_top"),
    ("eyes", "星星双眼", "eyes"),
    ("nose", "鼻影", "nose"),
    ("mouth", "微笑", "mouth"),
    ("cheek_left", "左腮红", "cheek_left"),
    ("cheek_right", "右腮红", "cheek_right"),
    ("chin", "下巴点", "chin"),
]

_templates_cache = None


def _new_canvas():
    return np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE, 4), dtype=np.uint8)


def _center():
    return TEMPLATE_SIZE // 2


def _gen_full_face():
    img = _new_canvas()
    cx, cy = _center(), _center()
    r = 80
    cv2.ellipse(img, (cx, cy), (r, int(r * 1.2)), 0, 0, 360, (60, 60, 60, 255), 2)
    return img


def _gen_head_top():
    img = _new_canvas()
    cx, cy = _center(), 70
    pts = np.array([[cx, cy - 50], [cx - 35, cy + 20], [cx + 35, cy + 20]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 180, 220, 255))
    cv2.rectangle(img, (cx - 40, cy + 20), (cx + 40, cy + 28), (0, 180, 220, 255), -1)
    return img


def _gen_forehead_top():
    img = _new_canvas()
    cx, cy = _center(), _center() - 20
    pts = np.array([[cx, cy - 18], [cx - 12, cy], [cx, cy + 18], [cx + 12, cy]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255, 255))
    return img


def _gen_eyes():
    img = _new_canvas()
    cy = _center() - 10
    for cx in [_center() - 40, _center() + 40]:
        pts = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            r = 22 if i % 2 == 0 else 9
            pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (0, 0, 0, 255))
    return img


def _gen_nose():
    img = _new_canvas()
    cx, cy = _center(), _center() + 10
    pts = np.array([[cx, cy - 15], [cx - 10, cy + 10], [cx + 10, cy + 10]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 120, 180, 180))
    return img


def _gen_mouth():
    img = _new_canvas()
    cx, cy = _center(), _center() + 10
    cv2.ellipse(img, (cx, cy), (28, 16), 0, 10, 170, (0, 0, 255, 255), 3)
    return img


def _gen_cheek_left():
    img = _new_canvas()
    cv2.circle(img, (_center() - 50, _center() + 30), 28, (160, 140, 255, 120), -1)
    return img


def _gen_cheek_right():
    img = _new_canvas()
    cv2.circle(img, (_center() + 50, _center() + 30), 28, (160, 140, 255, 120), -1)
    return img


def _gen_chin():
    img = _new_canvas()
    cv2.circle(img, (_center(), _center() + 60), 8, (60, 60, 60, 255), -1)
    return img


_GENERATORS = {
    "full_face": _gen_full_face,
    "head_top": _gen_head_top,
    "forehead_top": _gen_forehead_top,
    "eyes": _gen_eyes,
    "nose": _gen_nose,
    "mouth": _gen_mouth,
    "cheek_left": _gen_cheek_left,
    "cheek_right": _gen_cheek_right,
    "chin": _gen_chin,
}


def _make_thumb(bgra, size=96):
    h, w = bgra.shape[:2]
    sf = size / max(w, h)
    new_w, new_h = int(w * sf), int(h * sf)
    resized = cv2.resize(bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def ensure_templates():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    for tid, name, region in TEMPLATE_DEFS:
        path = os.path.join(TEMPLATE_DIR, f"{tid}.png")
        if not os.path.exists(path):
            gen = _GENERATORS.get(tid)
            if gen:
                img = gen()
                cv2.imwrite(path, img)


def load_templates():
    global _templates_cache
    if _templates_cache is not None:
        return _templates_cache
    ensure_templates()
    result = []
    for tid, name, region in TEMPLATE_DEFS:
        path = os.path.join(TEMPLATE_DIR, f"{tid}.png")
        if not os.path.exists(path):
            continue
        raw = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        thumb = _make_thumb(img)
        result.append({
            "id": tid,
            "name": name,
            "region": region,
            "image": img,
            "thumb": thumb,
            "template": True,
        })
    _templates_cache = result
    return result
