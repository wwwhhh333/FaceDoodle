# Stable Diffusion API 调用


import json
import logging
import os
import re
import time
import uuid
from urllib.parse import quote

import requests

from app.utils.config_loader import build_negative_prompt

log = logging.getLogger(__name__)


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
        log.debug("清理临时文件: 保留 %d/%d 个", max_files, len(files))
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
            log.warning("下载生成图片失败: %s", exc)
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
        ordered_ids = sorted(outputs.keys(), key=lambda nid: self._score_output_node(nid, workflow))
        for node_id in ordered_ids:
            yield node_id, outputs[node_id]

    def _extract_image(self, image_info, prompt_id):
        return (self._resolve_local_output_path(image_info)
                or self._download_image(image_info, prompt_id))

    def _score_output_node(self, node_id, workflow):
        node = workflow.get(str(node_id), {})
        ct = node.get("class_type", "")
        uses_alpha = self._node_uses_alpha_output(str(node_id), workflow)
        if ct == "SaveImage" and uses_alpha:
            return 0
        if ct == "SaveImage":
            return 1
        if ct == "PreviewImage" and uses_alpha:
            return 2
        if uses_alpha:
            return 3
        return 4

    # ── workflow injection helpers ──

    def _bypass_lora_nodes(self, workflow):
        """Remove LoraLoader nodes and rewire downstream references to the checkpoint.

        When no LoRA is configured, hardcoded placeholder names in the workflow
        would cause ComfyUI to error.  Bypassing the node entirely avoids that.
        """
        checkpoint_id = None
        lora_ids = []
        for nid, node in workflow.items():
            ct = node.get("class_type", "")
            if ct == "CheckpointLoaderSimple":
                checkpoint_id = nid
            elif ct == "LoraLoader":
                lora_ids.append(nid)

        if not lora_ids or checkpoint_id is None:
            return

        # Build reverse-reference index: node_id → [(node, key, list_index), ...]
        refs = {}
        for nid, node in workflow.items():
            for key, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) >= 2:
                    target = str(value[0])
                    refs.setdefault(target, []).append((node, key))

        for lora_id in lora_ids:
            for node, key in refs.get(lora_id, []):
                node["inputs"][key][0] = checkpoint_id
            del workflow[lora_id]
            log.debug("已绕过 LoRA 节点 %s，直接连接 checkpoint %s", lora_id, checkpoint_id)

    @staticmethod
    def _make_slug(prompt_text):
        """Build a safe filename slug from a prompt, falling back to a random id.

        Extracts ASCII word runs so Chinese/emoji prompts still produce
        readable filenames (e.g. "一副赛博朋克护目镜" → random hex).
        """
        slug_parts = [p.strip() for p in prompt_text.split(",") if p.strip()]
        short = "_".join(slug_parts[:2]) if len(slug_parts) >= 2 else slug_parts[0] if slug_parts else "sticker"
        ascii_words = re.findall(r'[a-zA-Z0-9]{2,}', short.lower())
        if ascii_words:
            slug = "_".join(ascii_words[:3])[:50]
        else:
            slug = uuid.uuid4().hex[:8]
        return slug

    def _post_prompt(self, workflow):
        """Submit a populated workflow to ComfyUI and return the prompt_id."""
        try:
            http_res = requests.post(
                f"http://{self.server_address}/prompt",
                json={"prompt": workflow}, timeout=30,
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"无法连接 ComfyUI ({self.server_address})，请确认 ComfyUI 已启动。\n  原始错误: {e}"
            )

        if http_res.status_code != 200:
            raise RuntimeError(
                f"ComfyUI 返回 HTTP {http_res.status_code}。\n  响应内容: {http_res.text[:500]}"
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

        return res["prompt_id"]

    def _inject_workflow(self, workflow, prompt_text, negative_prompt=None,
                         input_image_path=None, seed=None, filename_prefix=None,
                         controlnet_strength=None, denoise=None):
        """Inject prompt, LoRA, image, filename, seed, and optional overrides into a workflow.

        Modifies *workflow* in place.  Returns the workflow for chaining.
        """
        prompt_found = False
        has_load_image = any(
            node.get("class_type") == "LoadImage" for node in workflow.values()
        )
        need_image = (input_image_path and os.path.exists(input_image_path)
                      and has_load_image)
        slug = self._make_slug(filename_prefix or prompt_text)
        lora_cfg = self._cfg.get("model", {}).get("lora", {})
        have_lora_node = False
        need_lora = bool(lora_cfg and lora_cfg.get("name"))

        if need_image:
            uploaded_name = self._upload_image(input_image_path)

        for node in workflow.values():
            ct = node.get("class_type")
            title = node.get("_meta", {}).get("title", "")

            if title == "CLIP Text Encode (Prompt)":
                node["inputs"]["text"] = prompt_text
                prompt_found = True
            elif title == "CLIP Text Encode (Negative)":
                base_neg = negative_prompt or str(node.get("inputs", {}).get("text", "")).strip() or None
                node["inputs"]["text"] = build_negative_prompt(override=base_neg)
            elif ct == "SaveImage":
                node["inputs"]["filename_prefix"] = f"FaceDoodle/{slug}"
            elif ct == "KSampler":
                node.setdefault("inputs", {})["seed"] = (
                    seed if seed is not None else uuid.uuid4().int & ((1 << 63) - 1)
                )
                if denoise is not None:
                    node["inputs"]["denoise"] = denoise
            elif ct == "ControlNetApply" and controlnet_strength is not None:
                node["inputs"]["strength"] = controlnet_strength
            elif ct == "LoadImage" and need_image and uploaded_name:
                node["inputs"]["image"] = uploaded_name
                need_image = False  # Only set the first LoadImage
            elif ct == "LoraLoader":
                have_lora_node = True
                if need_lora:
                    node["inputs"]["lora_name"] = lora_cfg["name"]
                    if lora_cfg.get("strength_model") is not None:
                        node["inputs"]["strength_model"] = lora_cfg["strength_model"]
                    if lora_cfg.get("strength_clip") is not None:
                        node["inputs"]["strength_clip"] = lora_cfg["strength_clip"]
                    need_lora = False  # Only inject the first LoraLoader

        if not prompt_found:
            raise ValueError("工作流中未找到可用的正向提示词节点")
        if have_lora_node and not (lora_cfg and lora_cfg.get("name")):
            self._bypass_lora_nodes(workflow)

        return workflow

    # ── submission ──

    def _submit_workflow(self, prompt_text, workflow_name, negative_prompt=None,
                          input_image_path=None, seed=None,
                          controlnet_strength=None, denoise=None):
        """Load workflow, inject prompt/LoRA/seed/image, submit, return (workflow, prompt_id)."""
        workflow_path = os.path.join(os.path.dirname(__file__), "workflows", workflow_name)
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        self._inject_workflow(workflow, prompt_text,
                              negative_prompt=negative_prompt,
                              input_image_path=input_image_path,
                              seed=seed,
                              controlnet_strength=controlnet_strength,
                              denoise=denoise)
        prompt_id = self._post_prompt(workflow)
        return workflow, prompt_id

    # ── generation (polling) ──

    def generate_sync(self, prompt_text, workflow_name, negative_prompt=None, timeout=None, input_image_path=None, seed=None,
                        controlnet_strength=None, denoise=None):
        """
        同步生成逻辑：提交任务 -> 轮询状态 -> 返回结果路径
        """
        if timeout is None:
            timeout = self._cfg.get("comfyui", {}).get("generate_timeout", 120)

        workflow, prompt_id = self._submit_workflow(
            prompt_text, workflow_name,
            negative_prompt=negative_prompt,
            input_image_path=input_image_path,
            seed=seed,
            controlnet_strength=controlnet_strength,
            denoise=denoise,
        )

        previous_temp_files = self._snapshot_temp_files()

        # 轮询历史记录，等待任务完成
        start_time = time.time()
        while time.time() - start_time < timeout:
            history_res = requests.get(
                f"http://{self.server_address}/history/{prompt_id}",
                timeout=30,
            ).json()

            if prompt_id in history_res:
                outputs = history_res[prompt_id]['outputs']
                for node_id, output_node in self._iter_preferred_output_nodes(outputs, workflow):
                    if 'images' in output_node:
                        for image_info in output_node['images']:
                            path = self._extract_image(image_info, prompt_id)
                            if path:
                                return path

                if history_res[prompt_id].get("status", {}).get("completed"):
                    fallback_file = self._find_new_temp_file(previous_temp_files)
                    if fallback_file:
                        log.info("使用本地缓存文件兜底: %s", fallback_file)
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

        workflow_path = os.path.join(os.path.dirname(__file__), "workflows", workflow_name)
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        # Use shared injection for prompt, LoRA, image, seed, filename
        self._inject_workflow(workflow, prompt_text,
                              negative_prompt=negative_prompt,
                              input_image_path=input_image_path,
                              seed=seed)

        # AnimateDiff-specific: set batch_size on EmptyLatentImage
        for node in workflow.values():
            if node.get("class_type") == "EmptyLatentImage":
                node.setdefault("inputs", {})["batch_size"] = frame_count
                break

        prompt_id = self._post_prompt(workflow)
        previous_temp_files = self._snapshot_temp_files()

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
                            path = self._extract_image(image_info, prompt_id)
                            if path:
                                all_frames.append(path)

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
