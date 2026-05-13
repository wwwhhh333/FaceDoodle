# Stable Diffusion API 调用


import json

from app.utils.config_loader import build_negative_prompt
import os
import re
import time
import uuid
from urllib.parse import quote

import requests


TEMP_MAX_FILES = 50


def cleanup_temp_files(output_dir="assets/temp", max_files=TEMP_MAX_FILES):
    """Remove oldest temp files, keeping at most *max_files* most recent ones."""
    try:
        files = []
        for entry in os.scandir(output_dir):
            if entry.is_file() and entry.name.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
                files.append((entry.stat().st_mtime, entry.path))
        if len(files) <= max_files:
            return
        files.sort(key=lambda x: x[0], reverse=True)
        for _, path in files[max_files:]:
            os.remove(path)
        print(f"[ComfyClient] 清理临时文件: 保留 {max_files}/{len(files) + max_files} 个")
    except FileNotFoundError:
        pass


class ComfyClient:
    def __init__(self, server_address=None):
        from app.utils.config_loader import get_config
        self._cfg = get_config()
        if server_address is None:
            server_address = self._cfg.get("comfyui", {}).get("server_address", "127.0.0.1:8188")
        self.server_address = server_address
        self.output_dir = "assets/temp"
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
            os.path.join(os.path.expanduser("~"), "ComfyUI", image_type),
            os.path.join(os.path.expanduser("~"), "ComfyUI", "output"),
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

            safe_name = os.path.basename(filename)
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

    def _upload_image(self, filepath):
        url = f"http://{self.server_address}/upload/image"
        try:
            with open(filepath, 'rb') as f:
                files = {'image': (os.path.basename(filepath), f, 'image/png')}
                response = requests.post(url, files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("name")
        except requests.RequestException as e:
            raise RuntimeError(f"图片上传到 ComfyUI 失败: {e}")

    def _node_uses_alpha_output(self, node_id, workflow, cache=None):
        if cache is None:
            cache = {}
        if node_id in cache:
            return cache[node_id]

        node = workflow.get(str(node_id), {})
        class_type = node.get("class_type")

        if class_type == "JoinImageWithAlpha":
            cache[node_id] = True
            return True

        inputs = node.get("inputs", {})
        for value in inputs.values():
            if isinstance(value, list) and value:
                parent_id = str(value[0])
                if parent_id in workflow and self._node_uses_alpha_output(parent_id, workflow, cache):
                    cache[node_id] = True
                    return True

        cache[node_id] = False
        return False

    def _iter_preferred_output_nodes(self, outputs, workflow):
        def score(node_id):
            node = workflow.get(str(node_id), {})
            class_type = node.get("class_type", "")
            uses_alpha = self._node_uses_alpha_output(str(node_id), workflow)

            if class_type == "SaveImage" and uses_alpha:
                return 0
            if class_type == "SaveImage":
                return 1
            if class_type == "PreviewImage" and uses_alpha:
                return 2
            if uses_alpha:
                return 3
            return 4

        ordered_ids = sorted(outputs.keys(), key=score)
        for node_id in ordered_ids:
            yield node_id, outputs[node_id]

    def generate_sync(self, prompt_text, workflow_name, negative_prompt=None, timeout=None, input_image_path=None, seed=None):
        """
        同步生成逻辑：提交任务 -> 轮询状态 -> 返回结果路径
        """
        if timeout is None:
            timeout = self._cfg.get("comfyui", {}).get("generate_timeout", 120)

        # 加载工作流 JSON
        workflow_path = os.path.join(os.getcwd(), "assets", "workflows", workflow_name)
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        # 注入提示词
        prompt_updated = False
        negative_updated = False
        for node_id, node in workflow.items():
            title = node.get("_meta", {}).get("title", "")
            if title == "CLIP Text Encode (Prompt)":
                node["inputs"]["text"] = prompt_text
                prompt_updated = True
            elif title == "CLIP Text Encode (Negative)":
                base_neg = negative_prompt or str(node.get("inputs", {}).get("text", "")).strip() or None
                node["inputs"]["text"] = build_negative_prompt(override=base_neg)
                negative_updated = True

        if not prompt_updated:
            raise ValueError("工作流中未找到可用的正向提示词节点")

        # LoRA 名称覆盖
        lora_cfg = self._cfg.get("model", {}).get("lora", {})
        if lora_cfg:
            for node in workflow.values():
                if node.get("class_type") == "LoraLoader":
                    if lora_cfg.get("name"):
                        node["inputs"]["lora_name"] = lora_cfg["name"]
                        print(f"[ComfyClient] 使用 LoRA: {lora_cfg['name']}")
                    if lora_cfg.get("strength_model") is not None:
                        node["inputs"]["strength_model"] = lora_cfg["strength_model"]
                    if lora_cfg.get("strength_clip") is not None:
                        node["inputs"]["strength_clip"] = lora_cfg["strength_clip"]
                    print(f"[ComfyClient] LoRA 强度: model={node['inputs'].get('strength_model')}, clip={node['inputs'].get('strength_clip')}")
                    break
        else:
            print("[ComfyClient] 未配置 LoRA，使用工作流默认值")

        # Upload input image and inject into LoadImage node (img2img mode)
        if input_image_path and os.path.exists(input_image_path):
            uploaded_name = self._upload_image(input_image_path)
            if uploaded_name:
                load_image_set = False
                for node in workflow.values():
                    if node.get("class_type") == "LoadImage":
                        node["inputs"]["image"] = uploaded_name
                        load_image_set = True
                        print(f"[ComfyClient] img2img: 已上传并设置 LoadImage -> {uploaded_name}")
                        break
                if not load_image_set:
                    print("[ComfyClient] 警告: 工作流中未找到 LoadImage 节点，input_image_path 被忽略")
            else:
                print("[ComfyClient] 警告: 图片上传失败，将按 text-to-image 模式运行")

        # 根据 prompt 生成输出文件名
        slug_parts = [p.strip() for p in prompt_text.split(",") if p.strip()]
        short_prompt = "_".join(slug_parts[:2]) if len(slug_parts) >= 2 else slug_parts[0] if slug_parts else "sticker"
        slug = re.sub(r'[^\w]', '_', short_prompt.lower())
        slug = re.sub(r'_+', '_', slug).strip('_')[:50]
        for node in workflow.values():
            if node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = f"FaceDoodle/{slug}"
                break

        sampler_node = None
        for nid, node in workflow.items():
            if node.get("class_type") == "KSampler":
                sampler_node = node
                break
        if sampler_node:
            if seed is not None:
                sampler_node.setdefault("inputs", {})["seed"] = seed
            else:
                sampler_node.setdefault("inputs", {})["seed"] = uuid.uuid4().int & ((1 << 63) - 1)

        previous_temp_files = self._snapshot_temp_files()

        # 提交任务给 ComfyUI
        p = {"prompt": workflow}
        try:
            http_res = requests.post(
                f"http://{self.server_address}/prompt",
                json=p,
                timeout=30,
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"无法连接 ComfyUI ({self.server_address})，请确认 ComfyUI 已启动。"
                f"\n  原始错误: {e}"
            )

        if http_res.status_code != 200:
            raise RuntimeError(
                f"ComfyUI 返回 HTTP {http_res.status_code}。"
                f"\n  响应内容: {http_res.text[:500]}"
            )

        res = http_res.json()

        if "prompt_id" not in res:
            error_msg = res.get("error", {})
            if error_msg:
                detail = error_msg.get("message", str(error_msg))
                extra = error_msg.get("extra_info", "")
                raise RuntimeError(
                    f"ComfyUI 工作流提交失败: {detail}"
                    + (f"\n  详细信息: {extra}" if extra else "")
                    + "\n  可能原因: LoRA 文件缺失、节点配置错误、或模型文件不匹配"
                )
            raise RuntimeError(
                f"ComfyUI 返回了异常的响应: {json.dumps(res, ensure_ascii=False)[:500]}"
            )

        prompt_id = res['prompt_id']

        # 轮询历史记录，等待任务完成
        start_time = time.time()
        while time.time() - start_time < timeout:
            history_res = requests.get(
                f"http://{self.server_address}/history/{prompt_id}",
                timeout=30,
            ).json()

            if prompt_id in history_res:
                # 任务完成，提取文件名
                outputs = history_res[prompt_id]['outputs']
                for node_id, output_node in self._iter_preferred_output_nodes(outputs, workflow):
                    if 'images' in output_node:
                        for image_info in output_node['images']:
                            local_path = self._resolve_local_output_path(image_info)
                            if local_path and os.path.exists(local_path):
                                return local_path

                            downloaded_path = self._download_image(image_info, prompt_id)
                            if downloaded_path and os.path.exists(downloaded_path):
                                return downloaded_path

                if history_res[prompt_id].get("status", {}).get("completed"):
                    fallback_file = self._find_new_temp_file(previous_temp_files)
                    if fallback_file:
                        print(f"[ComfyClient] 使用本地缓存文件兜底: {fallback_file}")
                        return fallback_file

            time.sleep(0.5)

        return None

    def generate_animated_frames(self, prompt_text, workflow_name,
                                  input_image_path, frame_count=16, fps=8,
                                  seed=None, negative_prompt=None, timeout=None):
        """Submit an AnimateDiff I2V workflow, download all output frames.

        Returns a list of local file paths sorted by frame order.
        """
        if timeout is None:
            timeout = self._cfg.get("comfyui", {}).get("generate_timeout", 300)

        workflow_path = os.path.join(os.getcwd(), "assets", "workflows", workflow_name)
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        # Inject prompt, negative, seed, frame_count
        for node_id, node in workflow.items():
            title = node.get("_meta", {}).get("title", "")
            class_type = node.get("class_type", "")
            if title == "CLIP Text Encode (Prompt)":
                node["inputs"]["text"] = prompt_text
            elif title == "CLIP Text Encode (Negative)":
                base_neg = negative_prompt or str(node.get("inputs", {}).get("text", "")).strip() or None
                node["inputs"]["text"] = build_negative_prompt(override=base_neg)
            elif class_type == "KSampler":
                if seed is not None:
                    node.setdefault("inputs", {})["seed"] = int(seed)
                else:
                    node.setdefault("inputs", {})["seed"] = uuid.uuid4().int & ((1 << 63) - 1)
            elif class_type == "EmptyLatentImage":
                node.setdefault("inputs", {})["batch_size"] = frame_count
            elif class_type == "SaveImage":
                slug = re.sub(r'[^\w]', '_', prompt_text.lower())[:40]
                node["inputs"]["filename_prefix"] = f"FaceDoodle/anim_{slug}"

        # Inject LoRA config
        lora_cfg = self._cfg.get("model", {}).get("lora", {})
        if lora_cfg and lora_cfg.get("name"):
            for node in workflow.values():
                if node.get("class_type") == "LoraLoader":
                    node["inputs"]["lora_name"] = lora_cfg["name"]
                    if "strength_model" in lora_cfg:
                        node["inputs"]["strength_model"] = lora_cfg["strength_model"]
                    if "strength_clip" in lora_cfg:
                        node["inputs"]["strength_clip"] = lora_cfg["strength_clip"]
                    break

        # Upload input image
        if input_image_path and os.path.exists(input_image_path):
            uploaded_name = self._upload_image(input_image_path)
            if uploaded_name:
                for node in workflow.values():
                    if node.get("class_type") == "LoadImage":
                        node["inputs"]["image"] = uploaded_name
                        break

        previous_temp_files = self._snapshot_temp_files()

        p = {"prompt": workflow}
        try:
            http_res = requests.post(
                f"http://{self.server_address}/prompt",
                json=p, timeout=30,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"无法连接 ComfyUI: {e}")

        if http_res.status_code != 200:
            try:
                error_body = http_res.json()
            except Exception:
                error_body = http_res.text[:500]
            raise RuntimeError(
                f"ComfyUI 返回 HTTP {http_res.status_code}\n  响应内容: {json.dumps(error_body, ensure_ascii=False)[:1000]}"
            )

        res = http_res.json()
        if "prompt_id" not in res:
            error_msg = res.get("error", {})
            detail = error_msg.get("message", str(error_msg)) if error_msg else "unknown"
            raise RuntimeError(f"ComfyUI 工作流提交失败: {detail}")

        prompt_id = res["prompt_id"]

        start_time = time.time()
        while time.time() - start_time < timeout:
            history_res = requests.get(
                f"http://{self.server_address}/history/{prompt_id}",
                timeout=30,
            ).json()

            if prompt_id in history_res:
                outputs = history_res[prompt_id].get("outputs", {})
                all_frames = []
                for node_id, output_node in self._iter_preferred_output_nodes(outputs, workflow):
                    if "images" in output_node:
                        for image_info in output_node["images"]:
                            local = (self._resolve_local_output_path(image_info)
                                     or self._download_image(image_info, prompt_id))
                            if local and os.path.exists(local):
                                all_frames.append(local)

                if all_frames:
                    all_frames.sort(key=lambda p: (
                        os.path.basename(p),
                        os.path.getmtime(p),
                    ))
                    return all_frames

                if history_res[prompt_id].get("status", {}).get("completed"):
                    return []

            time.sleep(0.5)

        return []
