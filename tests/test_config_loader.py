"""Test config merging and loading."""

from app.utils.config_loader import _deep_merge, DEFAULT_CONFIG


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99}}
    _deep_merge(base, override)
    assert base["a"]["x"] == 1  # preserved
    assert base["a"]["y"] == 99  # overridden
    assert base["b"] == 3


def test_deep_merge_new_key():
    base = {"a": 1}
    override = {"b": 2}
    _deep_merge(base, override)
    assert base["b"] == 2


def test_deep_merge_scalar_override():
    base = {"a": 1}
    override = {"a": 100}
    _deep_merge(base, override)
    assert base["a"] == 100


def test_deep_merge_empty_override():
    base = {"a": 1, "b": {"c": 2}}
    _deep_merge(base, {})
    assert base == {"a": 1, "b": {"c": 2}}


def test_deep_merge_three_levels():
    base = {"x": {"y": {"z": 1, "w": 2}}}
    override = {"x": {"y": {"z": 99}}}
    _deep_merge(base, override)
    assert base["x"]["y"]["z"] == 99
    assert base["x"]["y"]["w"] == 2


def test_default_config_has_required_keys():
    assert "comfyui" in DEFAULT_CONFIG
    assert "camera" in DEFAULT_CONFIG
    assert "queue" in DEFAULT_CONFIG
    assert "agent" in DEFAULT_CONFIG
    assert "generation" in DEFAULT_CONFIG
    assert "preferences" in DEFAULT_CONFIG


def test_default_config_comfyui():
    assert "server_address" in DEFAULT_CONFIG["comfyui"]
    assert "generate_timeout" in DEFAULT_CONFIG["comfyui"]
