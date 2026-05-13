import subprocess
import time
import socket
import os
import sys


def _find_comfy_main(install_path):
    """Find main.py inside a ComfyUI installation directory."""
    candidates = [
        os.path.join(install_path, "main.py"),
        os.path.join(install_path, "ComfyUI", "main.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def check_comfy_ready(server_address, timeout=30, interval=1.0):
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
        return bool(self.install_path and os.path.isdir(self.install_path))

    def start(self):
        if not self.enabled:
            print("[ComfyManager] 未配置 ComfyUI 路径，跳过自动启动")
            return False

        main_py = _find_comfy_main(self.install_path)
        if not main_py:
            print(f"[ComfyManager] 在 {self.install_path} 中未找到 ComfyUI main.py")
            return False

        print(f"[ComfyManager] 启动 ComfyUI: {main_py}")
        try:
            self._process = subprocess.Popen(
                [sys.executable, main_py],
                cwd=os.path.dirname(main_py),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            print(f"[ComfyManager] 启动失败: {e}")
            return False

        print("[ComfyManager] 等待 ComfyUI 就绪...")
        if check_comfy_ready(self.server_address):
            print("[ComfyManager] ComfyUI 已就绪")
            return True
        else:
            print("[ComfyManager] ComfyUI 启动超时，请手动检查")
            return False

    def stop(self):
        if self._process and self._process.poll() is None:
            print("[ComfyManager] 正在关闭 ComfyUI...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            print("[ComfyManager] ComfyUI 已关闭")
