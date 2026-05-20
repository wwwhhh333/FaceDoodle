"""Tests for app/main.py — entry point helper functions."""

import os
import sys

import pytest


# ── _get_video_path ──

class TestGetVideoPath:
    def test_no_video_arg_returns_none(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py"])
        from app.main import _get_video_path
        result = _get_video_path({})
        assert result is None

    def test_video_with_path(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--video", "test_data/test.mp4"])
        from app.main import _get_video_path
        result = _get_video_path({})
        assert result == "test_data/test.mp4"

    def test_video_without_path_uses_config(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--video"])
        config = {"video": {"path": "config_video.mp4"}}
        from app.main import _get_video_path
        result = _get_video_path(config)
        assert result == "config_video.mp4"

    def test_video_arg_with_no_config_path_returns_none(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--video"])
        config = {"video": {"path": ""}}
        from app.main import _get_video_path
        result = _get_video_path(config)
        assert result is None

    def test_multiple_args(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--verbose", "--video", "vid.mp4", "--mock"])
        from app.main import _get_video_path
        result = _get_video_path({})
        assert result == "vid.mp4"

    def test_mock_arg_before_video(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--mock", "--video", "test.mp4"])
        from app.main import _get_video_path
        result = _get_video_path({})
        assert result == "test.mp4"

    def test_video_as_last_arg(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--video"])
        from app.main import _get_video_path
        # No value after --video (it's the last arg)
        config = {"video": {"path": None}}
        result = _get_video_path(config)
        assert result is None


# ── _resolve_api_key ──

class TestResolveApiKey:
    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key-123")
        from app.main import _resolve_api_key
        result = _resolve_api_key({"api_key": "config-key"})
        assert result == "env-key-123"

    def test_config_last_resort(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        # Mock api_key.txt existence check to force config fallback
        real_exists = os.path.exists

        def patched_exists(path):
            if "api_key.txt" in path:
                return False
            return real_exists(path)

        monkeypatch.setattr("os.path.exists", patched_exists)
        from app.main import _resolve_api_key
        result = _resolve_api_key({"api_key": "cfg-key-789"})
        assert result == "cfg-key-789"

    def test_empty_config_api_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        real_exists = os.path.exists

        def patched_exists(path):
            if "api_key.txt" in path:
                return False
            return real_exists(path)

        monkeypatch.setattr("os.path.exists", patched_exists)
        from app.main import _resolve_api_key
        result = _resolve_api_key({"api_key": ""})
        assert result == ""
