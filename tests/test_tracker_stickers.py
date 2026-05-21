"""Tests for app/core/tracker_stickers.py — StickerManager mixin."""

import queue
import uuid

import numpy as np
import pytest

from app.core.protocol import (
    GalAddSticker, GalRemoveSticker, GalSelectEditTarget,
    GalLoadTemplate, GalLoadSticker, GalMergeGroup,
)

# Need to import texture animator for the mixin
from app.core.animation import TextureAnimator
from app.core.sticker_registry import StickerRegistry
from app.core.tracker_stickers import StickerManager


class _TestStickerManager(StickerManager):
    """Minimal wrapper providing attributes ConsumerProcessor normally sets."""

    def __init__(self, display_queue=None):
        self.registry = StickerRegistry(max_stickers=20)
        self._anim_evaluations = {}
        self._adj_is_delta = set()
        self.texture_animator = TextureAnimator()
        self.edit_target_id = None
        self.active_content = {}
        self._sticker_adjustments = {}
        self.cached_face_data = None
        self.display_queue = display_queue or queue.Queue()
        self._had_stickers = False


@pytest.fixture
def mgr():
    return _TestStickerManager()


@pytest.fixture
def sample_sticker_img():
    return np.zeros((40, 50, 4), dtype=np.uint8)


# ── _add_sticker_instance ──

class TestAddStickerInstance:
    def test_adds_to_active_stickers(self, mgr, sample_sticker_img):
        sid = "test-id-123"
        iid = mgr._add_sticker_instance(sample_sticker_img, sid, "forehead_top", 1.0, "test")
        assert len(mgr.registry.all) == 1
        assert mgr.registry.all[0].instance_id == iid
        assert mgr.registry.all[0].sticker_id == sid
        assert mgr.registry.all[0].location == "forehead_top"
        assert mgr._had_stickers is True

    def test_creates_adjustment_entry(self, mgr, sample_sticker_img):
        iid = mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        adj = mgr.registry.get_adj(iid)
        assert adj.offset_x == 0.0
        assert adj.offset_y == 0.0
        assert adj.rotation == 0.0
        assert adj.scale_mult == 1.0

    def test_with_anim_meta(self, mgr, sample_sticker_img):
        anim_meta = {"is_animated": True, "frame_count": 16, "frame_cols": 4, "frame_rows": 4, "fps": 8}
        iid = mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test", anim_meta=anim_meta)
        inst = mgr.registry.all[0]
        assert inst.is_animated is True
        assert inst.frame_count == 16
        assert inst.frame_cols == 4
        assert inst.frame_rows == 4
        assert inst.fps == 8

    def test_capacity_check(self, mgr, sample_sticker_img):
        # _add_sticker_instance does not enforce cap; _handle_add_sticker does.
        # But the registry is initialized with max_stickers.
        assert mgr.registry.max_stickers == 20


# ── _handle_add_sticker ──

class TestHandleAddSticker:
    def test_adds_sticker_from_gallery(self, mgr, mock_storage):
        from app.utils import storage
        # Save a sticker first
        img = np.zeros((30, 30, 4), dtype=np.uint8)
        sid = storage.save_sticker(img, {"prompt": "test", "location": "eyes", "scale": 1.0})
        msg = GalAddSticker(sticker_id=sid)
        mgr._handle_add_sticker(msg)
        assert len(mgr.registry.all) == 1
        assert mgr.edit_target_id is not None

    def test_empty_sticker_id_does_nothing(self, mgr):
        msg = GalAddSticker(sticker_id="")
        mgr._handle_add_sticker(msg)
        assert len(mgr.registry.all) == 0

    def test_unknown_sticker_id_does_nothing(self, mgr):
        msg = GalAddSticker(sticker_id="nonexistent-id")
        mgr._handle_add_sticker(msg)
        assert len(mgr.registry.all) == 0

    def test_respects_capacity(self, mgr, mock_storage):
        mgr.registry._max_stickers = 1
        from app.utils import storage
        # First sticker — should succeed
        sid1 = storage.save_sticker(np.zeros((10, 10, 4), dtype=np.uint8),
                                     {"prompt": "a", "location": "eyes", "scale": 1.0})
        mgr._handle_add_sticker(GalAddSticker(sticker_id=sid1))
        assert len(mgr.registry.all) == 1
        # Second sticker — should be rejected (at capacity)
        sid2 = storage.save_sticker(np.zeros((10, 10, 4), dtype=np.uint8),
                                     {"prompt": "b", "location": "nose", "scale": 1.0})
        mgr._handle_add_sticker(GalAddSticker(sticker_id=sid2))
        assert len(mgr.registry.all) == 1


# ── _handle_remove_sticker ──

class TestHandleRemoveSticker:
    def test_removes_active_sticker(self, mgr, sample_sticker_img):
        iid = mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        msg = GalRemoveSticker(instance_id=iid)
        mgr._handle_remove_sticker(msg)
        assert len(mgr.registry.all) == 0
        assert mgr.registry.get(iid) is None

    def test_removes_nonexistent_does_nothing(self, mgr, sample_sticker_img):
        mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        msg = GalRemoveSticker(instance_id="nonexistent-iid")
        mgr._handle_remove_sticker(msg)
        assert len(mgr.registry.all) == 1

    def test_clears_edit_target(self, mgr, sample_sticker_img):
        iid = mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        mgr.edit_target_id = iid
        msg = GalRemoveSticker(instance_id=iid)
        mgr._handle_remove_sticker(msg)
        assert mgr.edit_target_id is None


# ── _handle_select_edit_target ──

class TestHandleSelectEditTarget:
    def test_selects_sticker(self, mgr, sample_sticker_img):
        iid = mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        msg = GalSelectEditTarget(instance_id=iid)
        mgr._handle_select_edit_target(msg)
        assert mgr.edit_target_id == iid

    def test_select_nonexistent_does_not_set(self, mgr, sample_sticker_img):
        mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        msg = GalSelectEditTarget(instance_id="bad-iid")
        mgr._handle_select_edit_target(msg)
        assert mgr.edit_target_id is None

    def test_clear_edit_target(self, mgr, sample_sticker_img):
        iid = mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        mgr.edit_target_id = iid
        msg = GalSelectEditTarget(instance_id=None)
        mgr._handle_select_edit_target(msg)
        assert mgr.edit_target_id is None


# ── _handle_load_template ──

class TestHandleLoadTemplate:
    def test_loads_template_with_image(self, mgr, sample_sticker_img):
        template = {"image": sample_sticker_img, "id": "tmpl-1", "region": "forehead_top", "name": "测试模板"}
        msg = GalLoadTemplate(template=template)
        mgr._handle_load_template(msg)
        assert len(mgr.registry.all) == 1
        assert mgr.edit_target_id is not None

    def test_empty_template_clears_stickers(self, mgr, sample_sticker_img):
        mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        msg = GalLoadTemplate(template=None)
        mgr._handle_load_template(msg)
        assert len(mgr.registry.all) == 0

    def test_template_without_image_clears(self, mgr, sample_sticker_img):
        mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        msg = GalLoadTemplate(template={"id": "tmpl"})
        mgr._handle_load_template(msg)
        assert len(mgr.registry.all) == 0

    def test_template_respects_capacity(self, mgr, sample_sticker_img):
        mgr.registry._max_stickers = 1
        mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        template = {"image": np.zeros((10, 10, 4), dtype=np.uint8), "id": "tmpl-2",
                     "region": "eyes", "name": "新模板"}
        msg = GalLoadTemplate(template=template)
        mgr._handle_load_template(msg)
        assert len(mgr.registry.all) == 1  # stays at 1, not 2


# ── _handle_load_sticker ──

class TestHandleLoadSticker:
    def test_loads_sticker_from_storage(self, mgr, mock_storage):
        from app.utils import storage
        img = np.zeros((30, 30, 4), dtype=np.uint8)
        sid = storage.save_sticker(img, {"prompt": "test", "location": "nose", "scale": 1.0})
        msg = GalLoadSticker(sticker_id=sid)
        mgr._handle_load_sticker(msg)
        assert len(mgr.registry.all) == 1
        assert mgr.edit_target_id is not None

    def test_empty_sid_clears_stickers(self, mgr, sample_sticker_img):
        mgr._add_sticker_instance(sample_sticker_img, "sid", "forehead", 1.0, "test")
        msg = GalLoadSticker(sticker_id=None)
        mgr._handle_load_sticker(msg)
        assert len(mgr.registry.all) == 0

    def test_unknown_sid_does_nothing(self, mgr, sample_sticker_img):
        msg = GalLoadSticker(sticker_id="nonexistent-sid")
        mgr._handle_load_sticker(msg)
        assert len(mgr.registry.all) == 0
