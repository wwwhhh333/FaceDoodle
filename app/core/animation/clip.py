"""Keyframe animation engine for sticker transforms.

Provides easing functions, keyframe interpolation, clip management,
and a per-instance playback engine for the consumer loop.
"""

import time
import uuid
from dataclasses import dataclass, field
from bisect import bisect_right


# ══════════════════════════════════════════════════════════════════════════════
# Easing functions
# ══════════════════════════════════════════════════════════════════════════════

def linear(t):
    return t


def ease_in(t):
    return t * t


def ease_out(t):
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out(t):
    return 3.0 * t * t - 2.0 * t * t * t


EASING_FUNCTIONS = {
    "linear": linear,
    "ease-in": ease_in,
    "ease-out": ease_out,
    "ease-in-out": ease_in_out,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Keyframe:
    time: float
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    scale_mult: float = 1.0
    opacity: float = 1.0
    easing: str = "linear"

    def to_dict(self):
        return {
            "time": self.time, "offset_x": self.offset_x,
            "offset_y": self.offset_y, "rotation": self.rotation,
            "scale_mult": self.scale_mult, "opacity": self.opacity,
            "easing": self.easing,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            time=float(d["time"]),
            offset_x=float(d.get("offset_x", 0)),
            offset_y=float(d.get("offset_y", 0)),
            rotation=float(d.get("rotation", 0)),
            scale_mult=float(d.get("scale_mult", 1)),
            opacity=float(d.get("opacity", 1)),
            easing=str(d.get("easing", "linear")),
        )


@dataclass
class AnimationClip:
    id: str = ""
    name: str = "New Clip"
    duration: float = 2.0
    loop: bool = False
    sticker_id: str = ""
    keyframes: list = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def add_keyframe(self, kf):
        i = bisect_right([k.time for k in self.keyframes], kf.time)
        self.keyframes.insert(i, kf)
        self._recalc_duration()

    def remove_keyframe(self, index):
        if 0 <= index < len(self.keyframes):
            del self.keyframes[index]
            self._recalc_duration()

    def _recalc_duration(self):
        if self.keyframes:
            last = self.keyframes[-1].time
            if last > self.duration:
                self.duration = last + 0.5

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "duration": self.duration,
            "loop": self.loop, "sticker_id": self.sticker_id,
            "keyframes": [k.to_dict() for k in self.keyframes],
        }

    @classmethod
    def from_dict(cls, d):
        clip = cls(
            id=d.get("id", ""),
            name=d.get("name", "New Clip"),
            duration=float(d.get("duration", 2.0)),
            loop=bool(d.get("loop", False)),
            sticker_id=d.get("sticker_id", ""),
        )
        clip.keyframes = [Keyframe.from_dict(k) for k in d.get("keyframes", [])]
        return clip


# ══════════════════════════════════════════════════════════════════════════════
# Clip evaluation
# ══════════════════════════════════════════════════════════════════════════════

_IDENTITY = {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0.0, "scale_mult": 1.0, "opacity": 1.0}


def evaluate_clip(clip, t):
    """Evaluate an AnimationClip at time *t* (seconds), returning an adjustment dict."""
    kfs = clip.keyframes
    if not kfs:
        return dict(_IDENTITY)
    if len(kfs) == 1:
        return _kf_to_adj(kfs[0])

    if clip.loop and clip.duration > 0:
        t = t % clip.duration
    elif not clip.loop and t >= kfs[-1].time:
        return _kf_to_adj(kfs[-1])
    if t <= kfs[0].time:
        return _kf_to_adj(kfs[0])

    # Find the segment
    times = [k.time for k in kfs]
    i = bisect_right(times, t) - 1
    i = max(0, min(i, len(kfs) - 2))

    a, b = kfs[i], kfs[i + 1]
    seg_dur = b.time - a.time
    if seg_dur <= 0:
        return _kf_to_adj(a)

    raw_t = (t - a.time) / seg_dur
    ease_fn = EASING_FUNCTIONS.get(a.easing, linear)
    eased = max(0.0, min(1.0, ease_fn(raw_t)))

    return {
        "offset_x": _lerp(a.offset_x, b.offset_x, eased),
        "offset_y": _lerp(a.offset_y, b.offset_y, eased),
        "rotation": _lerp_angle(a.rotation, b.rotation, eased),
        "scale_mult": _lerp(a.scale_mult, b.scale_mult, eased),
        "opacity": _lerp(a.opacity, b.opacity, eased),
    }


def _kf_to_adj(kf):
    return {
        "offset_x": kf.offset_x, "offset_y": kf.offset_y,
        "rotation": kf.rotation, "scale_mult": kf.scale_mult,
        "opacity": kf.opacity,
    }


def _lerp(a, b, t):
    return a + (b - a) * t


def _lerp_angle(a, b, t):
    """Shortest-arc linear interpolation for angles (degrees)."""
    diff = (b - a) % 360
    if diff > 180:
        diff -= 360
    return a + diff * t
