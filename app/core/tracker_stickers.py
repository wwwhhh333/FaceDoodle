"""Sticker lifecycle management mixin for ConsumerProcessor."""

import logging
import uuid

from app.core.protocol import (
    GalAddSticker, GalRemoveSticker, GalSelectEditTarget,
    GalLoadTemplate, GalLoadSticker, GalMergeGroup,
    DispStickerSaved,
)

log = logging.getLogger(__name__)
from app.utils import storage
from app.core.renderer import composite_stickers_to_merged


class StickerManager:
    """Mixin providing sticker add/remove/load/merge methods."""

    def _add_sticker_instance(self, sticker_img, sid, location, scale, prompt, anim_meta=None):
        instance_id = str(uuid.uuid4())
        saved = (self._sticker_adjustments.get(sid)
                 or storage.get_sticker_adjustments(sid))
        adj = dict(saved) if saved else {
            "offset_x": 0.0, "offset_y": 0.0,
            "rotation": 0.0, "scale_mult": 1.0,
        }
        self.adjustments[instance_id] = adj
        instance = {
            "instance_id": instance_id, "sticker_id": sid,
            "sticker": sticker_img, "location": location,
            "scale": scale, "prompt": prompt,
        }
        if anim_meta and anim_meta.get("is_animated"):
            instance["is_animated"] = True
            instance["frame_count"] = anim_meta.get("frame_count", 16)
            instance["frame_cols"] = anim_meta.get("frame_cols", 1)
            instance["frame_rows"] = anim_meta.get("frame_rows", 1)
            instance["fps"] = anim_meta.get("fps", 8)
        self.active_stickers.append(instance)
        self._had_stickers = True
        return instance_id

    def _handle_add_sticker(self, gmsg):
        if len(self.active_stickers) >= self.MAX_STICKERS:
            log.warning("贴纸数量已达上限 (%d)，忽略添加请求", self.MAX_STICKERS)
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
        removed_sticker_id = None
        for s in self.active_stickers:
            if s["instance_id"] == iid:
                removed_sticker_id = s.get("sticker_id")
                break
        self.active_stickers = [s for s in self.active_stickers if s["instance_id"] != iid]
        self.adjustments.pop(iid, None)
        self._anim_evaluations.pop(iid, None)
        self._adj_is_delta.discard(iid)
        self.texture_animator.unregister(iid)
        if self.edit_target_id == iid:
            self.edit_target_id = self.active_stickers[-1]["instance_id"] if self.active_stickers else None
        if self.active_content and self.active_content.get("sticker_id") == removed_sticker_id:
            self.active_content = None

    def _handle_select_edit_target(self, gmsg):
        iid = gmsg.instance_id
        if iid and any(s["instance_id"] == iid for s in self.active_stickers):
            self.edit_target_id = iid
        elif not iid:
            self.edit_target_id = None

    def _handle_load_template(self, gmsg):
        t = gmsg.template
        if t and t.get("image") is not None:
            if len(self.active_stickers) >= self.MAX_STICKERS:
                log.warning("贴纸数量已达上限 (%d)，忽略添加请求", self.MAX_STICKERS)
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
            for s in self.active_stickers:
                self.texture_animator.unregister(s["instance_id"])
            self.active_stickers.clear()
            self.adjustments.clear()
            self._anim_evaluations.clear()
            self._adj_is_delta.clear()
            self.edit_target_id = None
            self.active_content = None

    def _handle_load_sticker(self, gmsg):
        sid = gmsg.sticker_id
        if sid:
            if len(self.active_stickers) >= self.MAX_STICKERS:
                log.warning("贴纸数量已达上限 (%d)，忽略添加请求", self.MAX_STICKERS)
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
            for s in self.active_stickers:
                self.texture_animator.unregister(s["instance_id"])
            self.active_stickers.clear()
            self.adjustments.clear()
            self._anim_evaluations.clear()
            self._adj_is_delta.clear()
            self.edit_target_id = None
            self.active_content = None

    def _handle_merge_group(self, gmsg, face_data):
        iids = set(gmsg.instance_ids)
        merge_face = face_data if (face_data and "nose_tip" in face_data) else self.cached_face_data
        if len(iids) < 2 or not merge_face:
            return

        to_merge = [s for s in self.active_stickers if s["instance_id"] in iids]
        if len(to_merge) < 2:
            return

        merged_img, merged_location, merged_scale, mrg_ox, mrg_oy = \
            composite_stickers_to_merged(to_merge, self.adjustments, merge_face)
        if merged_img is None:
            return

        prompts = [s.get("prompt", "") for s in to_merge if s.get("prompt")]
        merged_prompt = " + ".join(prompts[:3])
        sid = storage.save_sticker(merged_img, {
            "prompt": merged_prompt or "合并贴纸",
            "location": merged_location,
            "scale": merged_scale,
        })

        for s in to_merge:
            iid = s["instance_id"]
            self.active_stickers = [x for x in self.active_stickers if x["instance_id"] != iid]
            self.adjustments.pop(iid, None)
            self._anim_evaluations.pop(iid, None)
            self._adj_is_delta.discard(iid)
            self.texture_animator.unregister(iid)

        merged_instance_id = str(uuid.uuid4())
        self.adjustments[merged_instance_id] = {
            "offset_x": mrg_ox, "offset_y": mrg_oy,
            "rotation": 0.0, "scale_mult": 1.0,
        }
        storage.save_sticker_adjustments(sid, self.adjustments[merged_instance_id])
        self.active_stickers.append({
            "instance_id": merged_instance_id,
            "sticker_id": sid,
            "sticker": merged_img,
            "location": merged_location,
            "scale": merged_scale,
            "prompt": merged_prompt,
        })
        self.edit_target_id = merged_instance_id
        self.active_content = {
            "sticker": merged_img, "location": merged_location,
            "scale": merged_scale, "sticker_id": sid,
            "prompt": merged_prompt,
        }
        self.display_queue.put(DispStickerSaved(sticker_id=sid))
