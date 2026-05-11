"""Animation queue processing and texture generation mixin for ConsumerProcessor."""

import threading

from app.core.animation import (
    AnimationClip, Keyframe,
)
from app.core.protocol import (
    AnimPlay, AnimPause, AnimStop, AnimSetClip,
    AnimAddKeyframe, AnimRemoveKeyframe, AnimSetLoop, AnimSeek, AnimExport,
    AnimGenTexture, AnimClipUpdated, AnimExportProgress, AnimGenProgress,
)
from app.utils import storage


class AnimationProcessor:
    """Mixin providing animation queue processing, export, and texture generation."""

    def _process_animation_queue(self):
        while not self.animation_queue.empty():
            try:
                msg = self.animation_queue.get(block=False)
            except Exception:
                break

            if isinstance(msg, AnimPlay):
                self.anim_engine.play(msg.instance_id)
                if msg.instance_id not in self._adj_is_delta:
                    self._adj_to_delta(msg.instance_id)
                    self._adj_is_delta.add(msg.instance_id)
            elif isinstance(msg, AnimPause):
                self.anim_engine.pause(msg.instance_id)
                if msg.instance_id in self._adj_is_delta:
                    self._adj_to_absolute(msg.instance_id)
                    self._adj_is_delta.discard(msg.instance_id)
            elif isinstance(msg, AnimStop):
                self.anim_engine.stop(msg.instance_id)
                if msg.instance_id in self._adj_is_delta:
                    self._adj_to_absolute(msg.instance_id)
                    self._adj_is_delta.discard(msg.instance_id)
            elif isinstance(msg, AnimSetClip):
                self.anim_engine.set_clip(msg.instance_id, msg.clip_id)
            elif isinstance(msg, AnimAddKeyframe):
                self._handle_add_keyframe(msg)
            elif isinstance(msg, AnimRemoveKeyframe):
                self._handle_remove_keyframe(msg)
            elif isinstance(msg, AnimSetLoop):
                self.anim_engine.set_loop(msg.instance_id, msg.loop)
            elif isinstance(msg, AnimSeek):
                self.anim_engine.seek(msg.instance_id, msg.time)
            elif isinstance(msg, AnimExport):
                self._pending_export = (msg.instance_id, msg.format, msg.fps, msg.output_path)
            elif isinstance(msg, AnimGenTexture):
                if not self._texture_gen_running:
                    self._pending_texture_gen = msg

    def _adj_to_delta(self, iid):
        adj = self.adjustments.get(iid)
        anim = self._anim_evaluations.get(iid)
        if adj is None or anim is None:
            return
        adj["offset_x"] -= anim["offset_x"]
        adj["offset_y"] -= anim["offset_y"]
        adj["rotation"] -= anim["rotation"]
        if anim["scale_mult"] > 0.001:
            adj["scale_mult"] /= anim["scale_mult"]

    def _adj_to_absolute(self, iid):
        adj = self.adjustments.get(iid)
        anim = self._anim_evaluations.get(iid)
        if adj is None or anim is None:
            return
        adj["offset_x"] += anim["offset_x"]
        adj["offset_y"] += anim["offset_y"]
        adj["rotation"] += anim["rotation"]
        adj["scale_mult"] *= anim["scale_mult"]

    def _handle_add_keyframe(self, msg):
        clip = self.anim_engine.get_bound_clip(msg.instance_id)
        if clip is None:
            cls = AnimationClip(name="New Clip", sticker_id=msg.instance_id)
            self.anim_engine.register_clip(cls)
            self.anim_engine.set_clip(msg.instance_id, cls.id)
            clip = cls

        adj = self.adjustments.get(msg.instance_id, {})
        anim = self._anim_evaluations.get(msg.instance_id, {})
        if self.anim_engine.is_playing(msg.instance_id) and anim:
            eff_ox = anim.get("offset_x", 0.0) + adj.get("offset_x", 0.0)
            eff_oy = anim.get("offset_y", 0.0) + adj.get("offset_y", 0.0)
            eff_rot = anim.get("rotation", 0.0) + adj.get("rotation", 0.0)
            eff_sm = anim.get("scale_mult", 1.0) * adj.get("scale_mult", 1.0)
        else:
            eff_ox = adj.get("offset_x", 0.0)
            eff_oy = adj.get("offset_y", 0.0)
            eff_rot = adj.get("rotation", 0.0)
            eff_sm = adj.get("scale_mult", 1.0)

        kf = Keyframe(
            time=msg.time,
            offset_x=eff_ox,
            offset_y=eff_oy,
            rotation=eff_rot,
            scale_mult=eff_sm,
            easing=msg.easing,
        )
        clip.add_keyframe(kf)
        self.display_queue.put(AnimClipUpdated(clip_data=clip.to_dict()))

    def _handle_remove_keyframe(self, msg):
        clip = self.anim_engine.get_bound_clip(msg.instance_id)
        if clip is None:
            return
        clip.remove_keyframe(msg.keyframe_index)
        self.display_queue.put(AnimClipUpdated(clip_data=clip.to_dict()))

    def _evaluate_animations(self):
        for instance in self.active_stickers:
            iid = instance["instance_id"]
            clip = self.anim_engine.get_bound_clip(iid)
            if clip is None:
                continue
            self.anim_engine.tick(iid)
            anim_adj = self.anim_engine.evaluate(iid)
            if anim_adj is not None:
                self._anim_evaluations[iid] = anim_adj

    def _process_export(self):
        if self._pending_export is None:
            return
        instance_id, fmt, fps, output_path = self._pending_export
        self._pending_export = None

        clip = self.anim_engine.get_bound_clip(instance_id)
        if clip is None:
            return

        instance = next((s for s in self.active_stickers if s["instance_id"] == instance_id), None)
        if instance is None:
            return

        face_data = self.cached_face_data
        if face_data is None:
            return

        threading.Thread(
            target=self._run_export,
            args=(clip, instance, face_data, fmt, fps, output_path),
            daemon=True,
        ).start()

    def _run_export(self, clip, instance, face_data, fmt, fps, output_path):
        try:
            from app.core.animation.export import export_animation
            export_animation(
                clip, instance["sticker"], fps, output_path, face_data,
                instance["location"], instance["scale"],
                progress_callback=lambda p: self.display_queue.put(
                    AnimExportProgress(progress=p, done=(p >= 1.0), output_path=output_path)),
                format=fmt,
            )
        except Exception as e:
            print(f"[Export] 导出失败: {e}")
            self.display_queue.put(AnimExportProgress(progress=0.0, done=True, output_path=""))

    def _process_texture_generation(self):
        if self._texture_gen_running or self._pending_texture_gen is None:
            return
        msg = self._pending_texture_gen
        self._pending_texture_gen = None
        self._texture_gen_running = True

        threading.Thread(
            target=self._run_texture_gen,
            args=(msg,),
            daemon=True,
        ).start()

    def _run_texture_gen(self, msg):
        try:
            sticker_bgra, meta = storage.get_sticker(msg.sticker_id)
            if sticker_bgra is None:
                self.display_queue.put(AnimGenProgress(
                    sticker_id=msg.sticker_id, done=True,
                    error="未找到贴纸",
                ))
                self._texture_gen_running = False
                return

            from app.ai.generator import ComfyClient
            from app.core.animation.gen import generate_animated_sticker

            client = ComfyClient()
            sprite_sheet, anim_meta = generate_animated_sticker(
                sticker_bgra=sticker_bgra,
                client=client,
                motion_prompt=msg.motion_prompt,
                frame_count=msg.frame_count,
                fps=msg.fps,
                progress_callback=lambda p: self.display_queue.put(AnimGenProgress(
                    sticker_id=msg.sticker_id,
                    progress=p,
                    done=False,
                )),
            )

            if sprite_sheet is None:
                self.display_queue.put(AnimGenProgress(
                    sticker_id=msg.sticker_id, done=True,
                    error="生成失败，ComfyUI 未返回任何帧",
                ))
                self._texture_gen_running = False
                return

            base_meta = meta if meta else {}
            base_meta["prompt"] = base_meta.get("prompt", "") + " [动画]"
            result_id = storage.save_animated_sticker(sprite_sheet, anim_meta, base_meta)
            self.display_queue.put(AnimGenProgress(
                sticker_id=msg.sticker_id,
                progress=1.0,
                done=True,
                result_sticker_id=result_id,
            ))
        except Exception as e:
            print(f"[TextureGen] 纹理动画生成失败: {e}")
            self.display_queue.put(AnimGenProgress(
                sticker_id=msg.sticker_id, done=True,
                error=str(e),
            ))
        finally:
            self._texture_gen_running = False
