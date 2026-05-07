"""Test storage module — save/load/delete/favorite with tmp_path."""

import numpy as np
import pytest

import app.utils.storage as storage_mod


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    """Redirect GALLERY_DIR and INDEX_PATH to tmp_path."""
    gdir = str(tmp_path / "gallery")
    ipath = str(tmp_path / "gallery" / "index.json")
    monkeypatch.setattr(storage_mod, "GALLERY_DIR", gdir)
    monkeypatch.setattr(storage_mod, "INDEX_PATH", ipath)
    return tmp_path


@pytest.fixture
def sample_sticker():
    """A small RGBA sticker image."""
    s = np.zeros((40, 50, 4), dtype=np.uint8)
    s[10:30, 15:35, :] = (0, 0, 255, 255)  # red square
    return s


def test_save_and_load_gallery(gallery, sample_sticker):
    sid = storage_mod.save_sticker(sample_sticker, {"prompt": "test sticker"})
    assert sid is not None

    stickers = storage_mod.load_gallery()
    assert len(stickers) == 1
    assert stickers[0]["id"] == sid
    assert stickers[0]["prompt"] == "test sticker"


def test_save_multiple_stickers(gallery, sample_sticker):
    ids = []
    for i in range(3):
        sid = storage_mod.save_sticker(sample_sticker, {"prompt": f"sticker {i}"})
        ids.append(sid)

    stickers = storage_mod.load_gallery()
    assert len(stickers) == 3


def test_get_sticker(gallery, sample_sticker):
    sid = storage_mod.save_sticker(sample_sticker, {"prompt": "test", "location": "eyes", "scale": 1.5})
    img, meta = storage_mod.get_sticker(sid)
    assert img is not None
    assert img.shape[2] == 4  # RGBA
    assert meta["prompt"] == "test"
    assert meta["region"] == "eyes"
    assert meta["scale"] == 1.5


def test_get_sticker_nonexistent(gallery):
    img, meta = storage_mod.get_sticker("nonexistent-id")
    assert img is None
    assert meta is None


def test_get_sticker_thumb(gallery, sample_sticker):
    sid = storage_mod.save_sticker(sample_sticker, {"prompt": "thumb test"})
    thumb = storage_mod.get_sticker_thumb(sid)
    assert thumb is not None
    assert thumb.shape[0] <= storage_mod.THUMB_SIZE
    assert thumb.shape[1] <= storage_mod.THUMB_SIZE


def test_delete_sticker(gallery, sample_sticker):
    sid = storage_mod.save_sticker(sample_sticker, {"prompt": "delete me"})
    assert len(storage_mod.load_gallery()) == 1

    result = storage_mod.delete_sticker(sid)
    assert result is True
    assert len(storage_mod.load_gallery()) == 0


def test_delete_nonexistent(gallery):
    result = storage_mod.delete_sticker("fake-id")
    assert result is False


def test_set_favorite(gallery, sample_sticker):
    sid = storage_mod.save_sticker(sample_sticker, {"prompt": "fav test"})
    storage_mod.set_favorite(sid, True)

    stickers = storage_mod.load_gallery()
    assert stickers[0]["favorite"] is True

    storage_mod.set_favorite(sid, False)
    stickers = storage_mod.load_gallery()
    assert stickers[0]["favorite"] is False


def test_save_and_get_adjustments(gallery, sample_sticker):
    sid = storage_mod.save_sticker(sample_sticker, {"prompt": "adj test"})
    adj = {"offset_x": 0.1, "offset_y": -0.2, "rotation": 15.0, "scale_mult": 1.3}
    storage_mod.save_sticker_adjustments(sid, adj)

    loaded = storage_mod.get_sticker_adjustments(sid)
    assert loaded is not None
    assert loaded["offset_x"] == 0.1
    assert loaded["offset_y"] == -0.2
    assert loaded["rotation"] == 15.0
    assert loaded["scale_mult"] == 1.3


def test_get_adjustments_nonexistent(gallery):
    assert storage_mod.get_sticker_adjustments("no-such-id") is None


def test_save_adjustments_partial(gallery, sample_sticker):
    """Only some adjustment fields provided, others get defaults."""
    sid = storage_mod.save_sticker(sample_sticker, {"prompt": "partial"})
    storage_mod.save_sticker_adjustments(sid, {"offset_x": 0.5})

    loaded = storage_mod.get_sticker_adjustments(sid)
    assert loaded["offset_x"] == 0.5
    assert loaded["offset_y"] == 0.0  # default
    assert loaded["rotation"] == 0.0
    assert loaded["scale_mult"] == 1.0


# save_preferences / load_preferences / add_recent_prompt are thin wrappers
# around config_loader that write to hardcoded "config.json".  These are
# better tested as integration tests; the config_loader._deep_merge logic
# is already covered in test_config_loader.py.
