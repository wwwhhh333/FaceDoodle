"""Test procedural template generation."""

import numpy as np
import pytest

from app.core.templates import (
    _gen_full_face, _gen_head_top, _gen_forehead_top, _gen_eyes,
    _gen_nose, _gen_mouth, _gen_cheek_left, _gen_cheek_right, _gen_chin,
    _make_thumb, _GENERATORS, TEMPLATE_DEFS, TEMPLATE_SIZE,
    ensure_templates, load_templates,
)


_all_generators = [
    ("full_face", _gen_full_face),
    ("head_top", _gen_head_top),
    ("forehead_top", _gen_forehead_top),
    ("eyes", _gen_eyes),
    ("nose", _gen_nose),
    ("mouth", _gen_mouth),
    ("cheek_left", _gen_cheek_left),
    ("cheek_right", _gen_cheek_right),
    ("chin", _gen_chin),
]


@pytest.mark.parametrize("name,gen_func", _all_generators)
def test_template_shape(name, gen_func):
    img = gen_func()
    assert img.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE, 4), f"{name} shape mismatch"
    assert img.dtype == np.uint8


@pytest.mark.parametrize("name,gen_func", _all_generators)
def test_template_has_alpha(name, gen_func):
    img = gen_func()
    assert np.any(img[:, :, 3] > 0), f"{name} has no visible alpha"


def test_all_defs_have_generators():
    for tid, name, region in TEMPLATE_DEFS:
        assert tid in _GENERATORS, f"Missing generator for {tid}"


# ── _make_thumb ──

def test_make_thumb_shape():
    src = np.zeros((200, 100, 4), dtype=np.uint8)
    src[:, :, 3] = 255
    thumb = _make_thumb(src, size=96)
    assert thumb.shape == (96, 96, 4)


def test_make_thumb_centers_content():
    """Non-square source: short dimension gets centered with transparent borders."""
    src = np.zeros((15, 30, 4), dtype=np.uint8)
    src[:, :, 3] = 255
    thumb = _make_thumb(src, size=96)
    # Short dimension (15 * 3.2 = 48) leaves borders
    assert thumb[0, 48, 3] == 0   # top border transparent
    assert thumb[48, 48, 3] == 255  # center has content


# ── File I/O tests (Phase 3) ──

def test_ensure_and_load_templates(tmp_path, monkeypatch):
    import app.core.templates as tmod
    monkeypatch.setattr(tmod, "TEMPLATE_DIR", str(tmp_path / "templates"))
    # reset cache
    monkeypatch.setattr(tmod, "_templates_cache", None)

    ensure_templates()
    templates = load_templates()
    assert len(templates) == len(TEMPLATE_DEFS)
    for t in templates:
        assert "id" in t
        assert "image" in t
        assert "thumb" in t
        assert t["template"] is True
        assert t["image"].shape == (TEMPLATE_SIZE, TEMPLATE_SIZE, 4)
        assert t["thumb"].shape == (96, 96, 4)


def test_ensure_templates_idempotent(tmp_path, monkeypatch):
    import app.core.templates as tmod
    monkeypatch.setattr(tmod, "TEMPLATE_DIR", str(tmp_path / "templates"))
    monkeypatch.setattr(tmod, "_templates_cache", None)

    ensure_templates()
    # Second call should not raise
    ensure_templates()
