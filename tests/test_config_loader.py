"""Test config merging and loading."""

import copy
import os
import tempfile

import pytest

from app.utils.config_loader import (
    _deep_merge, DEFAULT_CONFIG,
    BUILTIN_PRESET_KEYS, is_builtin_preset, _generate_preset_key,
    add_preset, update_preset, delete_preset, reset_preset,
    get_config,
)


@pytest.fixture(autouse=True)
def _isolate_config():
    """Backup and restore config.json so tests never pollute the real file."""
    cfg_path = "config.json"
    bak = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            bak = f.read()
    yield
    if bak is not None:
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(bak)
    elif os.path.exists(cfg_path):
        os.remove(cfg_path)


# ──────────────────────────────────────────────
# Preset CRUD helpers
# ──────────────────────────────────────────────


def _reload_config_with_fresh_state():
    """Force a fresh in-memory config from DEFAULT_CONFIG (no disk I/O)."""
    import app.utils.config_loader as cl
    cl._config = copy.deepcopy(DEFAULT_CONFIG)
    return cl._config


def test_builtin_preset_keys_match_default():
    assert BUILTIN_PRESET_KEYS == DEFAULT_CONFIG["style"]["presets"].keys()


def test_is_builtin_preset_true_for_builtin():
    assert is_builtin_preset("pixel_art") is True
    assert is_builtin_preset("vector_art") is True
    assert is_builtin_preset("cartoon_style") is True
    assert is_builtin_preset("semi_realistic") is True


def test_is_builtin_preset_false_for_custom():
    assert is_builtin_preset("my_custom_style") is False
    assert is_builtin_preset("custom_abc123") is False


def test_generate_preset_key_not_builtin():
    key = _generate_preset_key()
    assert key.startswith("custom_")
    assert is_builtin_preset(key) is False


def test_generate_preset_key_unique():
    keys = {_generate_preset_key() for _ in range(100)}
    assert len(keys) == 100  # no collisions


def test_add_preset_returns_key():
    cfg = _reload_config_with_fresh_state()
    key = add_preset({
        "name": "Test Style",
        "positive_prefix": "test art of {prompt}",
        "lora_name": "test.safetensors",
        "lora_strength_model": 1.0,
        "lora_strength_clip": 1.0,
    })
    assert key is not None
    assert key in cfg["style"]["presets"]
    assert cfg["style"]["presets"][key]["name"] == "Test Style"


def test_add_preset_minimal_fields():
    cfg = _reload_config_with_fresh_state()
    key = add_preset({"name": "Minimal"})
    assert key is not None
    p = cfg["style"]["presets"][key]
    assert p["name"] == "Minimal"
    assert p["lora_name"] == ""


def test_update_preset_modifies_fields():
    cfg = _reload_config_with_fresh_state()
    key = add_preset({"name": "Before"})
    assert update_preset(key, {"name": "After", "lora_name": "after.safetensors"}) is True
    assert cfg["style"]["presets"][key]["name"] == "After"
    assert cfg["style"]["presets"][key]["lora_name"] == "after.safetensors"


def test_update_preset_missing_key_returns_false():
    assert update_preset("nonexistent", {"name": "Nope"}) is False


def test_delete_preset_refuses_builtin():
    assert delete_preset("pixel_art") is False


def test_delete_preset_removes_custom():
    cfg = _reload_config_with_fresh_state()
    key = add_preset({"name": "DeleteMe"})
    assert key in cfg["style"]["presets"]
    assert delete_preset(key) is True
    assert key not in cfg["style"]["presets"]


def test_delete_preset_falls_back_selected_preset():
    cfg = _reload_config_with_fresh_state()
    key = add_preset({"name": "Custom"})
    cfg["style"]["selected_preset"] = key
    delete_preset(key)
    assert cfg["style"]["selected_preset"] != key
    assert cfg["style"]["selected_preset"] in BUILTIN_PRESET_KEYS


def test_reset_preset_restores_default():
    cfg = _reload_config_with_fresh_state()
    original = copy.deepcopy(DEFAULT_CONFIG["style"]["presets"]["pixel_art"])
    cfg["style"]["presets"]["pixel_art"]["name"] = "改过的"
    assert reset_preset("pixel_art") is True
    assert cfg["style"]["presets"]["pixel_art"] == original


def test_reset_preset_refuses_custom():
    cfg = _reload_config_with_fresh_state()
    key = add_preset({"name": "Custom"})
    assert reset_preset(key) is False


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
