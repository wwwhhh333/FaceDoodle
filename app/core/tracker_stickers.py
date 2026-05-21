"""Sticker lifecycle management mixin for ConsumerProcessor.

Uses ``StickerRegistry`` (``self.registry``) for all instance storage and
relies on the registry's ``"removed"`` event for cross-domain cleanup
(texture animator, animation evaluation state) — registered once in
``ConsumerProcessor.__init__``.
"""

import logging
import uuid

from app.core.protocol import (
    GalAddSticker, GalRemoveSticker, GalSelectEditTarget,
    GalLoadTemplate, GalLoadSticker, GalMergeGroup,
    Adjustment, StickerInstance,
    DispStickerSaved,
)

log = logging.getLogger(__name__)
from app.utils import storage
from app.core.renderer import composite_stickers_to_merged


class StickerManager:
    """Mixin providing sticker add/remove/load/merge methods."""

    def _add_sticker_instance(self, sticker_img, sid, location, scale, prompt, anim_meta=None):
        instance_id = str(uuid.uuid4())
        anim_meta = anim_meta or {}
        inst = StickerInstance(
            instance_id=instance_id,
            sticker_id=sid,
            sticker=sticker_img,
            location=location,
            scale=scale,
            prompt=prompt,
            is_animated=anim_meta.get("is_animated", False),
            frame_count=anim_meta.get("frame_count", 0),
            frame_cols=anim_meta.get("frame_cols", 0),
            frame_rows=anim_meta.get("frame_rows", 0),
            fps=anim_meta.get("fps", 8),
        )
        saved = (self._sticker_adjustments.get(sid)
                 or storage.get_sticker_adjustments(sid))
        adj = Adjustment(**saved) if saved else Adjustment()
        self.registry.add(inst, adj)
        self._had_stickers = True
        return instance_id

    def _handle_add_sticker(self, gmsg):
        if self.registry.is_full:
            log.warning("贴纸数量已达上限 (%d)，忽略添加请求", self.registry.max_stickers)
            return
        sid = gmsg.sticker_id
        if not sid:
            return
        loaded, meta = storage.get_sticker(sid)
        if loaded is None or meta is None:
            return
        anim_meta = None
        if meta.get("is_animated"):
            anim_meta = {
                "is_animated": True,
                "frame_count": meta.get("frame_count", 16),
                "frame_cols": meta.get("frame_cols", 1),
                "frame_rows": meta.get("frame_rows", 1),
                "fps": meta.get("fps", 8),
            }
        iid = self._add_sticker_instance(
            loaded, sid,
            meta.get("region", "forehead_top"),
            meta.get("scale", 1.0),
            meta.get("prompt", ""),
            anim_meta=anim_meta,
        )
        if anim_meta:
            self.texture_animator.register(iid, anim_meta["frame_count"],
                                           anim_meta["fps"],
                                           anim_meta["frame_cols"],
                                           anim_meta["frame_rows"])
        self.edit_target_id = iid
        self.active_content = {
            "sticker": loaded,
            "location": meta.get("region", "forehead_top"),
            "scale": meta.get("scale", 1.0),
            "sticker_id": sid,
            "prompt": meta.get("prompt", ""),
        }

    def _handle_remove_sticker(self, gmsg):
        iid = gmsg.instance_id
        inst = self.registry.get(iid)
        removed_sticker_id = inst.sticker_id if inst else None
        self.registry.remove(iid)
        if self.edit_target_id == iid:
            all_instances = self.registry.all
            self.edit_target_id = all_instances[-1].instance_id if all_instances else None
        if self.active_content and self.active_content.get("sticker_id") == removed_sticker_id:
            self.active_content = None

    def _handle_select_edit_target(self, gmsg):
        iid = gmsg.instance_id
        if iid and self.registry.has(iid):
            self.edit_target_id = iid
        elif not iid:
            self.edit_target_id = None

    def _handle_load_template(self, gmsg):
        t = gmsg.template
        if t and t.get("image") is not None:
            if self.registry.is_full:
                log.warning("贴纸数量已达上限 (%d)，忽略添加请求", self.registry.max_stickers)
                return
            iid = self._add_sticker_instance(
                t["image"], t["id"],
                t.get("region", "forehead_top"), 1.0,
                t.get("name", "模板"),
            )
            self.edit_target_id = iid
            self.active_content = {
                "sticker": t["image"],
                "location": t.get("region", "forehead_top"),
                "scale": 1.0,
                "sticker_id": t["id"],
                "prompt": t.get("name", "模板"),
            }
        else:
            for s in self.registry.all:
                self.texture_animator.unregister(s.instance_id)
            self.registry.clear()
            self.edit_target_id = None
            self.active_content = None

    def _handle_load_sticker(self, gmsg):
        sid = gmsg.sticker_id
        if sid:
            if self.registry.is_full:
                log.warning("贴纸数量已达上限 (%d)，忽略添加请求", self.registry.max_stickers)
                return
            loaded, meta = storage.get_sticker(sid)
            if loaded is not None and meta is not None:
                anim_meta = None
                if meta.get("is_animated"):
                    anim_meta = {
                        "is_animated": True,
                        "frame_count": meta.get("frame_count", 16),
                        "frame_cols": meta.get("frame_cols", 1),
                        "frame_rows": meta.get("frame_rows", 1),
                        "fps": meta.get("fps", 8),
                    }
                iid = self._add_sticker_instance(
                    loaded, sid,
                    meta.get("region", "forehead_top"),
                    meta.get("scale", 1.0),
                    meta.get("prompt", ""),
                    anim_meta=anim_meta,
                )
                if anim_meta:
                    self.texture_animator.register(iid, anim_meta["frame_count"],
                                                   anim_meta["fps"],
                                                   anim_meta["frame_cols"],
                                                   anim_meta["frame_rows"])
                self.edit_target_id = iid
                self.active_content = {
                    "sticker": loaded,
                    "location": meta.get("region", "forehead_top"),
                    "scale": meta.get("scale", 1.0),
                    "sticker_id": sid,
                    "prompt": meta.get("prompt", ""),
                }
        else:
            for s in self.registry.all:
                self.texture_animator.unregister(s.instance_id)
            self.registry.clear()
            self.edit_target_id = None
            self.active_content = None

    def _handle_merge_group(self, gmsg, face_data):
        iids = set(gmsg.instance_ids)
        merge_face = face_data if (face_data and "nose_tip" in face_data) else self.cached_face_data
        if len(iids) < 2 or not merge_face:
            return

        # Build the old-style dict list that composite_stickers_to_merged expects
        to_merge = [s for s in self.registry.all if s.instance_id in iids]
        if len(to_merge) < 2:
            return

        # Convert to old dict format for the renderer (it reads dict keys)
        old_style_instances = [
            {
                "instance_id": s.instance_id,
                "sticker_id": s.sticker_id,
                "sticker": s.sticker,
                "location": s.location,
                "scale": s.scale,
                "prompt": s.prompt,
            }
            for s in to_merge
        ]
        adjustments_dict = {
            s.instance_id: {
                "offset_x": self.registry.get_adj(s.instance_id).offset_x,
                "offset_y": self.registry.get_adj(s.instance_id).offset_y,
                "rotation": self.registry.get_adj(s.instance_id).rotation,
                "scale_mult": self.registry.get_adj(s.instance_id).scale_mult,
            }
            for s in to_merge
        }

        merged_img, merged_location, merged_scale, mrg_ox, mrg_oy = \
            composite_stickers_to_merged(old_style_instances, adjustments_dict, merge_face)
        if merged_img is None:
            return

        prompts = [s.prompt for s in to_merge if s.prompt]
        merged_prompt = " + ".join(prompts[:3])
        sid = storage.save_sticker(merged_img, {
            "prompt": merged_prompt or "合并贴纸",
            "location": merged_location,
            "scale": merged_scale,
        })

        for s in to_merge:
            self.registry.remove(s.instance_id)

        merged_instance_id = str(uuid.uuid4())
        merged_inst = StickerInstance(
            instance_id=merged_instance_id,
            sticker_id=sid,
            sticker=merged_img,
            location=merged_location,
            scale=merged_scale,
            prompt=merged_prompt,
        )
        merged_adj = Adjustment(offset_x=mrg_ox, offset_y=mrg_oy, rotation=0.0, scale_mult=1.0)
        self.registry.add(merged_inst, merged_adj)
        storage.save_sticker_adjustments(sid, {
            "offset_x": mrg_ox, "offset_y": mrg_oy,
            "rotation": 0.0, "scale_mult": 1.0,
        })
        self.edit_target_id = merged_instance_id
        self.active_content = {
            "sticker": merged_img, "location": merged_location,
            "scale": merged_scale, "sticker_id": sid,
            "prompt": merged_prompt,
        }
        self.display_queue.put(DispStickerSaved(sticker_id=sid))
