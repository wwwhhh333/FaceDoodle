import subprocess
import time
import socket
import os
import sys
import logging

log = logging.getLogger(__name__)


def _resolve_launch(install_path):
    """Resolve how to launch ComfyUI: returns (executable, args, cwd) or None."""
    # Case 1: install_path points to a .bat or .cmd file directly
    if os.path.isfile(install_path) and install_path.lower().endswith((".bat", ".cmd")):
        return install_path, [], os.path.dirname(install_path)

    if not os.path.isdir(install_path):
        return None

    # Case 2: directory — look for a .bat first (portable version), then main.py
    fallback_bat = None
    for entry in os.scandir(install_path):
        if entry.is_file() and entry.name.lower().endswith((".bat", ".cmd")):
            if "nvidia" in entry.name.lower():
                return entry.path, [], install_path
            if fallback_bat is None:
                fallback_bat = entry.path
    if fallback_bat:
        return fallback_bat, [], install_path

    # Case 3: standard ComfyUI install — look for main.py
    for c in [os.path.join(install_path, "main.py"), os.path.join(install_path, "ComfyUI", "main.py")]:
        if os.path.exists(c):
            return sys.executable, [c], os.path.dirname(c)

    return None


def check_comfy_ready(server_address, timeout=500, interval=2.0):
    """Wait until ComfyUI responds on *server_address*. Returns True if ready."""
    host, _, port = server_address.partition(":")
    port = int(port) if port else 8188
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(interval)
    return False


class ComfyUIManager:
    """Manages a ComfyUI subprocess lifecycle."""

    def __init__(self, install_path=None, server_address="127.0.0.1:8188"):
        self.install_path = install_path
        self.server_address = server_address
        self._process = None

    @property
    def enabled(self):
        return bool(self.install_path and os.path.exists(self.install_path))

    def start(self):
        if not self.enabled:
            log.info("未配置 ComfyUI 路径，跳过自动启动")
            return False

        launch = _resolve_launch(self.install_path)
        if not launch:
            log.warning("在 %s 中未找到可启动的文件", self.install_path)
            return False

        executable, args, cwd = launch
        cmd = [executable] + args
        log.info("启动 ComfyUI: %s", ' '.join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            log.error("ComfyUI 启动失败: %s", e)
            return False

        log.info("等待 ComfyUI 就绪...")
        if check_comfy_ready(self.server_address):
            log.info("ComfyUI 已就绪")
            return True
        else:
            log.warning("ComfyUI 启动超时，请手动检查")
            return False

    def stop(self):
        if self._process and self._process.poll() is None:
            log.info("正在关闭 ComfyUI...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            log.info("ComfyUI 已关闭")
