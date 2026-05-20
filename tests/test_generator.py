"""Tests for app/ai/generator.py — pure functions and isolated logic."""

import json
import os
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── _make_slug ──

class TestMakeSlug:
    def test_ascii_prompt_underscores(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        slug = client._make_slug("hello world, foo bar")
        assert isinstance(slug, str)
        assert len(slug) > 0
        # Slug should contain no spaces
        assert " " not in slug
        # Uses first 2 comma parts → ascii runs first 3
        assert "hello" in slug and "world" in slug and "foo" in slug

    def test_ascii_short_slug_subset(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        # Only first 2 words used, first 3 ascii runs
        slug = client._make_slug("cat ears, fluffy tail, big eyes")
        assert "cat_ears" in slug

    def test_chinese_prompt_fallback_to_random(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        # Pure Chinese — no ASCII runs of >=2 chars
        slug = client._make_slug("赛博朋克护目镜")
        assert len(slug) == 8  # uuid4 hex[:8]
        assert all(c in "0123456789abcdef" for c in slug)

    def test_empty_prompt_fallback(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        slug = client._make_slug("")
        # Falls back to "sticker" when input is empty
        assert slug == "sticker"

    def test_comma_separated_uses_first_two(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        slug = client._make_slug("red, green, blue")
        assert "red" in slug
        assert "green" in slug

    def test_mixed_chinese_ascii_extracts_ascii(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        slug = client._make_slug("赛博朋克 cyber goggles 2024")
        assert "cyber" in slug
        assert "goggles" in slug


# ── _score_output_node ──

class TestScoreOutputNode:
    def test_save_image_with_alpha_highest_score(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        # SaveImage feeds FROM JoinImageWithAlpha → uses alpha
        workflow = {
            "5": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
            "6": {"class_type": "JoinImageWithAlpha", "inputs": {}},
        }
        score = client._score_output_node("5", workflow)
        assert score == 0

    def test_save_image_no_alpha_medium_score(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "5": {"class_type": "SaveImage", "inputs": {}},
        }
        score = client._score_output_node("5", workflow)
        assert score == 1

    def test_preview_with_alpha_second_best(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        # PreviewImage feeds FROM JoinImageWithAlpha → uses alpha
        workflow = {
            "7": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
            "6": {"class_type": "JoinImageWithAlpha", "inputs": {}},
        }
        score = client._score_output_node("7", workflow)
        assert score == 2

    def test_unknown_type_max_score(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "9": {"class_type": "UnknownNode", "inputs": {}},
        }
        score = client._score_output_node("9", workflow)
        assert score == 4


# ── _node_uses_alpha_output ──

class TestNodeUsesAlphaOutput:
    def test_join_alpha_node_returns_true(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {"6": {"class_type": "JoinImageWithAlpha", "inputs": {}}}
        assert client._node_uses_alpha_output("6", workflow) is True

    def test_save_node_upstream_of_alpha_returns_true(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "5": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
            "6": {"class_type": "JoinImageWithAlpha", "inputs": {}},
        }
        assert client._node_uses_alpha_output("5", workflow) is True

    def test_save_node_no_alpha_chain_returns_false(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "5": {"class_type": "SaveImage", "inputs": {}},
        }
        assert client._node_uses_alpha_output("5", workflow) is False

    def test_caching_prevents_infinite_loop(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        # Circular reference should not infinite loop
        workflow = {
            "1": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
            "2": {"class_type": "JoinImageWithAlpha", "inputs": {"images": ["1", 0]}},
            "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
        }
        cache = {}
        assert client._node_uses_alpha_output("1", workflow, cache) is True


# ── _bypass_lora_nodes ──

class TestBypassLoraNodes:
    def test_no_lora_node_no_change(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        }
        original = json.dumps(workflow, sort_keys=True)
        client._bypass_lora_nodes(workflow)
        assert json.dumps(workflow, sort_keys=True) == original

    def test_lora_node_rewired_to_checkpoint(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
            "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "test.safetensors", "model": ["1", 0]}},
        }
        client._bypass_lora_nodes(workflow)
        assert "LoraLoader" not in {n.get("class_type") for n in workflow.values()}
        assert len(workflow) == 1

    def test_multiple_lora_nodes_all_bypassed(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
            "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "a.safetensors", "model": ["1", 0]}},
            "3": {"class_type": "LoraLoader", "inputs": {"lora_name": "b.safetensors", "model": ["1", 0]}},
        }
        client._bypass_lora_nodes(workflow)
        lora_count = sum(1 for n in workflow.values() if n.get("class_type") == "LoraLoader")
        assert lora_count == 0


# ── _inject_workflow (no HTTP) ──

class TestInjectWorkflow:
    def test_prompt_injection(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "CLIP Text Encode (Negative)"},
            },
        }
        monkeypatch.setattr(client, "_cfg", {"model": {"lora": {}}})
        client._inject_workflow(workflow, "cat ears", negative_prompt="bad quality")
        assert workflow["2"]["inputs"]["text"] == "cat ears"
        # Negative prompt is built via build_negative_prompt; should at least contain the override
        assert "bad quality" in workflow["3"]["inputs"]["text"]

    def test_seed_injection(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {"seed": 0, "steps": 20},
                "_meta": {"title": "KSampler"},
            },
        }
        monkeypatch.setattr(client, "_cfg", {"model": {"lora": {}}})
        client._inject_workflow(workflow, "test", seed=12345)
        assert workflow["4"]["inputs"]["seed"] == 12345

    def test_denoise_injection(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "CLIP Text Encode (Prompt)"}},
            "4": {"class_type": "KSampler", "inputs": {"seed": 0}, "_meta": {"title": "KSampler"}},
        }
        monkeypatch.setattr(client, "_cfg", {"model": {"lora": {}}})
        client._inject_workflow(workflow, "test", denoise=0.7)
        assert workflow["4"]["inputs"]["denoise"] == 0.7

    def test_filename_prefix_injection(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "CLIP Text Encode (Prompt)"}},
            "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI"}, "_meta": {"title": "Save Image"}},
        }
        monkeypatch.setattr(client, "_cfg", {"model": {"lora": {}}})
        client._inject_workflow(workflow, "test cat", filename_prefix="custom_prefix")
        assert "custom_prefix" in workflow["5"]["inputs"]["filename_prefix"]

    def test_missing_prompt_node_raises(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "5": {"class_type": "SaveImage", "inputs": {}, "_meta": {"title": "Save Image"}},
        }
        monkeypatch.setattr(client, "_cfg", {"model": {"lora": {}}})
        with pytest.raises(ValueError, match="未找到"):
            client._inject_workflow(workflow, "test")

    def test_controlnet_strength(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "CLIP Text Encode (Prompt)"}},
            "8": {"class_type": "ControlNetApply", "inputs": {"strength": 0.5}, "_meta": {"title": "ControlNet Apply"}},
        }
        monkeypatch.setattr(client, "_cfg", {"model": {"lora": {}}})
        client._inject_workflow(workflow, "test", controlnet_strength=0.85)
        assert workflow["8"]["inputs"]["strength"] == 0.85


# ── cleanup_temp_files ──

class TestCleanupTempFiles:
    def test_under_max_keeps_all(self, tmp_path):
        from app.ai.generator import cleanup_temp_files
        for i in range(3):
            (tmp_path / f"file_{i}.png").write_text("dummy")
        cleanup_temp_files(str(tmp_path), max_files=5)
        remaining = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
        assert len(remaining) == 3

    def test_over_max_removes_oldest(self, tmp_path):
        from app.ai.generator import cleanup_temp_files
        for i in range(10):
            p = tmp_path / f"file_{i:03d}.png"
            p.write_text("dummy")
            # Set mtime to be in order of creation
            atime = os.path.getatime(tmp_path)
            os.utime(p, (atime, atime + i))
        cleanup_temp_files(str(tmp_path), max_files=3)
        remaining = sorted(os.listdir(tmp_path))
        # Should keep the 3 most recent (highest mtime = highest index)
        assert len(remaining) == 3
        # The ones with highest index (newest) should remain
        for fn in remaining:
            idx = int(fn.split("_")[1].split(".")[0])
            assert idx >= 7  # 7, 8, 9 are the newest

    def test_nonexistent_directory_no_error(self, tmp_path):
        from app.ai.generator import cleanup_temp_files
        # Should not raise
        cleanup_temp_files(str(tmp_path / "nonexistent"), max_files=5)

    def test_non_image_files_excluded(self, tmp_path):
        from app.ai.generator import cleanup_temp_files
        (tmp_path / "data.txt").write_text("not an image")
        (tmp_path / "image.png").write_text("actual image")
        cleanup_temp_files(str(tmp_path), max_files=0)
        # .txt should not be counted or cleaned
        assert os.path.exists(tmp_path / "data.txt")
        assert not os.path.exists(tmp_path / "image.png")


# ── _iter_preferred_output_nodes ──

class TestIterPreferredOutputNodes:
    def test_orders_by_score_ascending(self, monkeypatch):
        from app.ai.generator import ComfyClient
        client = ComfyClient(server_address="127.0.0.1:8188")
        workflow = {
            "a": {"class_type": "PreviewImage", "inputs": {}},
            "b": {"class_type": "SaveImage", "inputs": {
                "images": ["c", 0],
            }},
            "c": {"class_type": "JoinImageWithAlpha", "inputs": {}},
        }
        outputs = {
            "a": {"images": []},
            "b": {"images": []},
        }
        ids = list(client._iter_preferred_output_nodes(outputs, workflow))
        # b has alpha (score 0) should come before a (no alpha, score 4)
        assert ids == [("b", {"images": []}), ("a", {"images": []})]
