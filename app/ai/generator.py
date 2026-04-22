# Stable Diffusion API 调用


import json
import os
import time
import uuid
from urllib.parse import quote

import requests


class ComfyClient:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.output_dir = "assets/temp"  # 确保此目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_local_output_path(self, image_info):
        filename = image_info.get("filename")
        if not filename:
            return None

        subfolder = image_info.get("subfolder", "")
        image_type = image_info.get("type", "output")
        candidate_roots = []

        env_root = os.getenv("COMFYUI_OUTPUT_DIR")
        if env_root:
            candidate_roots.append(env_root)

        cwd = os.getcwd()
        candidate_roots.extend([
            os.path.join(cwd, "output"),
            os.path.join(cwd, "ComfyUI", "output"),
            os.path.join("D:/ComfyUI", image_type),
            os.path.join("D:/ComfyUI", "output"),
        ])

        for root in candidate_roots:
            if not root:
                continue
            candidate = os.path.join(root, subfolder, filename) if subfolder else os.path.join(root, filename)
            if os.path.exists(candidate):
                return candidate

        return None

    def _download_image(self, image_info, prompt_id):
        filename = image_info.get("filename")
        if not filename:
            return None

        subfolder = image_info.get("subfolder", "")
        image_type = image_info.get("type", "output")
        params = [
            f"filename={quote(filename)}",
            f"subfolder={quote(subfolder)}",
            f"type={quote(image_type)}",
        ]
        url = f"http://{self.server_address}/view?{'&'.join(params)}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            safe_name = f"{prompt_id}_{os.path.basename(filename)}"
            local_path = os.path.join(self.output_dir, safe_name)
            with open(local_path, "wb") as file_obj:
                file_obj.write(response.content)
            return local_path
        except requests.RequestException as exc:
            print(f"[ComfyClient] 下载生成图片失败: {exc}")
            return None

    def _snapshot_temp_files(self):
        try:
            return {
                entry.path
                for entry in os.scandir(self.output_dir)
                if entry.is_file() and entry.name.lower().endswith((".png", ".webp", ".jpg", ".jpeg"))
            }
        except FileNotFoundError:
            return set()

    def _find_new_temp_file(self, previous_files):
        current_files = self._snapshot_temp_files()
        new_files = [path for path in current_files - previous_files if os.path.exists(path)]
        if not new_files:
            return None

        new_files.sort(key=os.path.getmtime, reverse=True)
        return new_files[0]

    def generate_sync(self, prompt_text, workflow_name, timeout=30):
        """
        同步生成逻辑：提交任务 -> 轮询状态 -> 返回结果路径
        """
        # 1. 加载工作流 JSON
        workflow_path = os.path.join(os.getcwd(), "app", "workflows", workflow_name)
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        # 2. 只更新正向提示词节点，避免把负向提示词也覆盖掉
        prompt_updated = False
        for node_id, node in workflow.items():
            if node.get("_meta", {}).get("title") != "CLIP Text Encode (Prompt)":
                continue

            existing_text = str(node.get("inputs", {}).get("text", "")).strip()
            if existing_text or node_id == "8":
                node["inputs"]["text"] = prompt_text
                prompt_updated = True
                break

        if not prompt_updated:
            raise ValueError("工作流中未找到可用的正向提示词节点")

        sampler_node = workflow.get("11")
        if sampler_node and sampler_node.get("class_type") == "KSampler":
            sampler_node.setdefault("inputs", {})["seed"] = uuid.uuid4().int & ((1 << 63) - 1)

        previous_temp_files = self._snapshot_temp_files()

        # 3. 提交任务给 ComfyUI
        p = {"prompt": workflow}
        res = requests.post(
            f"http://{self.server_address}/prompt",
            json=p,
            timeout=30,
        ).json()
        prompt_id = res['prompt_id']

        # 4. 轮询历史记录，等待任务完成
        start_time = time.time()
        while time.time() - start_time < timeout:
            history_res = requests.get(
                f"http://{self.server_address}/history/{prompt_id}",
                timeout=30,
            ).json()

            if prompt_id in history_res:
                # 任务完成，提取文件名
                outputs = history_res[prompt_id]['outputs']
                for node_id in outputs:
                    if 'images' in outputs[node_id]:
                        for image_info in outputs[node_id]['images']:
                            downloaded_path = self._download_image(image_info, prompt_id)
                            if downloaded_path and os.path.exists(downloaded_path):
                                return downloaded_path

                            local_path = self._resolve_local_output_path(image_info)
                            if local_path and os.path.exists(local_path):
                                return local_path

                if history_res[prompt_id].get("status", {}).get("completed"):
                    fallback_file = self._find_new_temp_file(previous_temp_files)
                    if fallback_file:
                        print(f"[ComfyClient] 使用本地缓存文件兜底: {fallback_file}")
                        return fallback_file

            time.sleep(0.5)  # 每 0.5 秒轮询一次

        return None
