"""Per-instance keyframe animation playback engine."""

import time

from app.core.animation.clip import evaluate_clip


class AnimationEngine:
    """Manages clip playback per sticker instance (single-threaded, no locks)."""

    def __init__(self):
        self._clips = {}          # clip_id → AnimationClip
        self._bindings = {}       # instance_id → clip_id
        self._playback = {}       # instance_id → {playing, time, last_tick}
        self._manual_base = {}    # instance_id → dict (snapshot of manual offsets when playback started)

    # ── clip management ──

    def register_clip(self, clip):
        self._clips[clip.id] = clip

    def get_clip(self, clip_id):
        return self._clips.get(clip_id)

    def all_clips(self):
        return list(self._clips.values())

    def remove_clip(self, clip_id):
        self._clips.pop(clip_id, None)

    # ── binding ──

    def set_clip(self, instance_id, clip_id):
        clip = self._clips.get(clip_id)
        if clip is None:
            return False
        self._bindings[instance_id] = clip_id
        self._playback[instance_id] = {"playing": False, "time": 0.0, "last_tick": time.perf_counter()}
        return True

    def get_bound_clip(self, instance_id):
        cid = self._bindings.get(instance_id)
        return self._clips.get(cid) if cid else None

    # ── playback control ──

    def play(self, instance_id):
        pb = self._playback.get(instance_id)
        if pb:
            pb["playing"] = True
            pb["last_tick"] = time.perf_counter()

    def pause(self, instance_id):
        pb = self._playback.get(instance_id)
        if pb:
            pb["playing"] = False

    def stop(self, instance_id):
        pb = self._playback.get(instance_id)
        if pb:
            pb["playing"] = False
            pb["time"] = 0.0

    def is_playing(self, instance_id):
        pb = self._playback.get(instance_id)
        return pb["playing"] if pb else False

    def set_loop(self, instance_id, loop):
        clip = self.get_bound_clip(instance_id)
        if clip:
            clip.loop = loop

    def seek(self, instance_id, t):
        pb = self._playback.get(instance_id)
        if pb:
            clip = self.get_bound_clip(instance_id)
            if clip:
                pb["time"] = max(0.0, min(t, clip.duration))
            pb["last_tick"] = time.perf_counter()

    # ── per-frame ──

    def tick(self, instance_id):
        pb = self._playback.get(instance_id)
        if not pb or not pb["playing"]:
            return
        now = time.perf_counter()
        delta = now - pb["last_tick"]
        pb["last_tick"] = now
        clip = self.get_bound_clip(instance_id)
        if clip and not clip.loop and clip.duration > 0:
            pb["time"] = min(pb["time"] + delta, clip.duration)
        else:
            pb["time"] += delta

    def evaluate(self, instance_id):
        pb = self._playback.get(instance_id)
        if not pb:
            return None
        clip = self.get_bound_clip(instance_id)
        if clip is None:
            return None
        return evaluate_clip(clip, pb["time"])

    def get_playback_info(self, instance_id):
        pb = self._playback.get(instance_id)
        if not pb:
            return None
        clip = self.get_bound_clip(instance_id)
        return {
            "instance_id": instance_id,
            "playing": pb["playing"],
            "time": pb["time"],
            "duration": clip.duration if clip else 0.0,
            "loop": clip.loop if clip else False,
        }
