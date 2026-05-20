"""Tests for app/ai/comfy_manager.py — subprocess launch resolution and socket check."""

import os
import socket
import sys
import time

import pytest


# ── _resolve_launch ──

class TestResolveLaunch:
    def test_bat_file_direct(self, tmp_path):
        from app.ai.comfy_manager import _resolve_launch
        bat = tmp_path / "run_nvidia_gpu.bat"
        bat.write_text("echo hello")
        result = _resolve_launch(str(bat))
        assert result is not None
        exe, args, cwd = result
        assert exe == str(bat)
        assert args == []
        assert cwd == str(tmp_path)

    def test_bat_in_dir_nvidia_preferred(self, tmp_path):
        from app.ai.comfy_manager import _resolve_launch
        (tmp_path / "run_cpu.bat").write_text("echo cpu")
        (tmp_path / "run_nvidia_gpu.bat").write_text("echo nvidia")
        result = _resolve_launch(str(tmp_path))
        assert result is not None
        exe, args, _ = result
        assert "nvidia" in exe.lower()

    def test_bat_in_dir_fallback(self, tmp_path):
        from app.ai.comfy_manager import _resolve_launch
        (tmp_path / "main.bat").write_text("echo main")
        result = _resolve_launch(str(tmp_path))
        assert result is not None
        exe, args, _ = result
        assert exe.endswith(".bat") or exe.endswith(".cmd")

    def test_main_py_in_dir(self, tmp_path):
        from app.ai.comfy_manager import _resolve_launch
        main_py = tmp_path / "main.py"
        main_py.write_text("print('hello')")
        result = _resolve_launch(str(tmp_path))
        assert result is not None
        exe, args, cwd = result
        assert exe == sys.executable
        assert args == [str(main_py)]
        assert cwd == str(tmp_path)

    def test_main_py_in_subdir(self, tmp_path):
        from app.ai.comfy_manager import _resolve_launch
        sub = tmp_path / "ComfyUI"
        sub.mkdir()
        (sub / "main.py").write_text("print('hello')")
        result = _resolve_launch(str(tmp_path))
        assert result is not None
        exe, args, cwd = result
        assert args == [str(sub / "main.py")]

    def test_nonexistent_path_returns_none(self, tmp_path):
        from app.ai.comfy_manager import _resolve_launch
        result = _resolve_launch(str(tmp_path / "nonexistent"))
        assert result is None

    def test_no_launchable_file_returns_none(self, tmp_path):
        from app.ai.comfy_manager import _resolve_launch
        (tmp_path / "readme.txt").write_text("hello")
        result = _resolve_launch(str(tmp_path))
        assert result is None


# ── check_comfy_ready ──

class TestCheckComfyReady:
    def test_socket_connects_returns_true(self, monkeypatch):
        from app.ai.comfy_manager import check_comfy_ready

        def mock_create_connection(addr, timeout=2):
            return MagicMock()

        monkeypatch.setattr(socket, "create_connection", mock_create_connection)
        assert check_comfy_ready("127.0.0.1:8188", timeout=1, interval=0.1) is True

    def test_socket_refused_returns_false(self, monkeypatch):
        from app.ai.comfy_manager import check_comfy_ready

        call_count = 0

        def mock_create_connection(addr, timeout=2):
            nonlocal call_count
            call_count += 1
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(socket, "create_connection", mock_create_connection)
        # Short timeout should return False after retries
        result = check_comfy_ready("127.0.0.1:8188", timeout=1, interval=0.1)
        assert result is False
        # Should have retried multiple times
        assert call_count > 1

    def test_socket_timeout_returns_false(self, monkeypatch):
        from app.ai.comfy_manager import check_comfy_ready

        def mock_create_connection(addr, timeout=2):
            raise socket.timeout("timed out")

        monkeypatch.setattr(socket, "create_connection", mock_create_connection)
        assert check_comfy_ready("127.0.0.1:8188", timeout=1, interval=0.1) is False

    def test_port_parsing_default(self, monkeypatch):
        from app.ai.comfy_manager import check_comfy_ready

        captured = []

        def mock_create_connection(addr, timeout=2):
            captured.append(addr)
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(socket, "create_connection", mock_create_connection)
        check_comfy_ready("192.168.1.1", timeout=0.2, interval=0.1)
        # Should default to port 8188
        assert len(captured) > 0
        assert captured[0] == ("192.168.1.1", 8188)


# ── ComfyUIManager ──

class TestComfyUIManager:
    def test_enabled_with_valid_path(self, tmp_path):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager(install_path=str(tmp_path))
        assert mgr.enabled is True

    def test_disabled_with_none_path(self):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager(install_path=None)
        assert mgr.enabled is False

    def test_disabled_with_empty_path(self):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager(install_path="")
        assert mgr.enabled is False

    def test_disabled_with_nonexistent_path(self):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager(install_path="/nonexistent/path")
        assert mgr.enabled is False

    def test_start_disabled_returns_false(self):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager(install_path=None)
        assert mgr.start() is False

    def test_default_server_address(self):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager()
        assert mgr.server_address == "127.0.0.1:8188"

    def test_custom_server_address(self):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager(server_address="192.168.1.1:8080")
        assert mgr.server_address == "192.168.1.1:8080"

    def test_stop_with_no_process(self):
        from app.ai.comfy_manager import ComfyUIManager
        mgr = ComfyUIManager()
        # Should not raise
        mgr.stop()


class MagicMock:
    """Minimal mock for socket object — no external dependencies."""
    def close(self):
        pass
