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

_ws_available = True  # cached availability of websockets package

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

    def _submit_workflow(self, prompt_text, workflow_name, negative_prompt=None,
                          input_image_path=None, seed=None):
        """Load workflow, inject prompt/LoRA/seed/image, submit to ComfyUI, return (workflow, prompt_id)."""
        workflow_path = os.path.join(os.path.dirname(__file__), "workflows", workflow_name)
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        # Inject prompt
        prompt_updated = False
        for node_id, node in workflow.items():
            title = node.get("_meta", {}).get("title", "")
            if title == "CLIP Text Encode (Prompt)":
                node["inputs"]["text"] = prompt_text
                prompt_updated = True
            elif title == "CLIP Text Encode (Negative)":
                base_neg = negative_prompt or str(node.get("inputs", {}).get("text", "")).strip() or None
                node["inputs"]["text"] = build_negative_prompt(override=base_neg)

        if not prompt_updated:
            raise ValueError("工作流中未找到可用的正向提示词节点")

        # LoRA
        lora_cfg = self._cfg.get("model", {}).get("lora", {})
        if lora_cfg:
            for node in workflow.values():
                if node.get("class_type") == "LoraLoader":
                    if lora_cfg.get("name"):
                        node["inputs"]["lora_name"] = lora_cfg["name"]
                        log.debug("使用 LoRA: %s", lora_cfg['name'])
                    if lora_cfg.get("strength_model") is not None:
                        node["inputs"]["strength_model"] = lora_cfg["strength_model"]
                    if lora_cfg.get("strength_clip") is not None:
                        node["inputs"]["strength_clip"] = lora_cfg["strength_clip"]
                    log.debug("LoRA 强度: model=%s, clip=%s",
                              node['inputs'].get('strength_model'), node['inputs'].get('strength_clip'))
                    break
        else:
            log.debug("未配置 LoRA，使用工作流默认值")

        # Upload input image (img2img)
        if input_image_path and os.path.exists(input_image_path):
            uploaded_name = self._upload_image(input_image_path)
            if uploaded_name:
                load_image_set = False
                for node in workflow.values():
                    if node.get("class_type") == "LoadImage":
                        node["inputs"]["image"] = uploaded_name
                        load_image_set = True
                        log.debug("img2img: 已上传并设置 LoadImage -> %s", uploaded_name)
                        break
                if not load_image_set:
                    log.warning("工作流中未找到 LoadImage 节点，input_image_path 被忽略")
            else:
                log.warning("图片上传失败，将按 text-to-image 模式运行")

        # Filename prefix
        slug_parts = [p.strip() for p in prompt_text.split(",") if p.strip()]
        short_prompt = "_".join(slug_parts[:2]) if len(slug_parts) >= 2 else slug_parts[0] if slug_parts else "sticker"
        slug = re.sub(r'[^\w]', '_', short_prompt.lower())
        slug = re.sub(r'_+', '_', slug).strip('_')[:50]
        for node in workflow.values():
            if node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = f"FaceDoodle/{slug}"
                break

        # Seed
        for nid, node in workflow.items():
            if node.get("class_type") == "KSampler":
                if seed is not None:
                    node.setdefault("inputs", {})["seed"] = seed
                else:
                    node.setdefault("inputs", {})["seed"] = uuid.uuid4().int & ((1 << 63) - 1)
                break

        # Submit
        p = {"prompt": workflow}
        try:
            http_res = requests.post(
                f"http://{self.server_address}/prompt",
                json=p, timeout=30,
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

        return workflow, res['prompt_id']

    def generate_sync_ws(self, prompt_text, workflow_name, negative_prompt=None,
                          timeout=None, input_image_path=None, seed=None,
                          progress_callback=None):
        """WebSocket-based generation with real-time step progress.

        Falls back to polling ``generate_sync()`` when the ``websockets``
        package is unavailable.
        """
        global _ws_available
        if _ws_available:
            try:
                from websockets.sync.client import connect as ws_connect
            except ImportError:
                _ws_available = False
                log.warning("websockets 未安装，回退到轮询模式")

        if not _ws_available:
            log.debug("回退到轮询模式")
            return self.generate_sync(prompt_text, workflow_name,
                                       negative_prompt=negative_prompt,
                                       timeout=timeout,
                                       input_image_path=input_image_path,
                                       seed=seed)

        if timeout is None:
            timeout = self._cfg.get("comfyui", {}).get("generate_timeout", 120)

        ws_url = f"ws://{self.server_address}/ws?clientId={uuid.uuid4().hex}"
        result_path = None
        result_score = 99
        start_time = time.time()

        try:
            ws = ws_connect(ws_url, timeout=timeout)
        except Exception as e:
            log.warning("WebSocket 连接失败，回退到轮询: %s", e)
            return self.generate_sync(prompt_text, workflow_name,
                                       negative_prompt=negative_prompt,
                                       timeout=timeout,
                                       input_image_path=input_image_path,
                                       seed=seed)

        # Submit prompt AFTER WS connects, so no execution events are missed
        try:
            workflow, prompt_id = self._submit_workflow(
                prompt_text, workflow_name,
                negative_prompt=negative_prompt,
                input_image_path=input_image_path,
                seed=seed,
            )
        except Exception as e:
            ws.close()
            _ws_available = False
            log.warning("WebSocket 提交失败，回退到轮询: %s", e)
            return self.generate_sync(prompt_text, workflow_name,
                                       negative_prompt=negative_prompt,
                                       timeout=timeout,
                                       input_image_path=input_image_path,
                                       seed=seed)

        previous_temp_files = self._snapshot_temp_files()

        try:
            ws.socket.settimeout(timeout)
            while time.time() - start_time < timeout:
                try:
                    message = json.loads(ws.recv())
                except TimeoutError:
                    continue
                msg_type = message.get("type")
                data = message.get("data", {})

                if data.get("prompt_id") != prompt_id:
                    continue

                if msg_type == "progress" and progress_callback:
                    progress_callback(data.get("value", 0), data.get("max", 1))

                elif msg_type == "executed":
                    node_id = data.get("node", "")
                    node_output = data.get("output", {})
                    if "images" in node_output:
                        score = self._score_output_node(node_id, workflow)
                        for image_info in node_output["images"]:
                            path = self._extract_image(image_info, prompt_id)
                            if path and score < result_score:
                                result_path = path
                                result_score = score
                                break

                elif msg_type == "execution_cached":
                    nodes = data.get("nodes", [])
                    if nodes:
                        log.debug("ComfyUI 缓存命中: %d 个节点", len(nodes))

                elif msg_type == "executing":
                    if data.get("node") is None:
                        break  # prompt complete (node=None = all done)
                elif msg_type in ("execution_success", "execution_complete"):
                    break

                elif msg_type == "execution_error":
                    raise RuntimeError(
                        f"ComfyUI 执行错误: {data.get('node_id')} "
                        f"-- {data.get('exception_message', '')}"
                    )

        except Exception as e:
            _ws_available = False
            log.warning("WebSocket 异常，回退到轮询: %s", e)
            return self.generate_sync(prompt_text, workflow_name,
                                       negative_prompt=negative_prompt,
                                       timeout=timeout,
                                       input_image_path=input_image_path,
                                       seed=seed)
        finally:
            ws.close()

        if not result_path:
            log.info("WebSocket 完成，但未捕获到结果，尝试本地文件和回退...")
            fallback = self._find_new_temp_file(previous_temp_files)
            if fallback:
                log.info("WebSocket 未捕获结果，使用本地缓存文件: %s", fallback)
                return fallback
            log.info("WebSocket 未捕获结果，回退轮询")
            return self.generate_sync(prompt_text, workflow_name,
                                       negative_prompt=negative_prompt,
                                       timeout=max(5, timeout - (time.time() - start_time)),
                                       input_image_path=input_image_path,
                                       seed=seed)

        if result_path:
            log.info("WebSocket 成功捕获结果: %s", result_path)
        else:
            log.info("WebSocket 完成，未获取到结果")
        return result_path

    def generate_sync(self, prompt_text, workflow_name, negative_prompt=None, timeout=None, input_image_path=None, seed=None):
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


def generate_view_variant(client, output_dir, base_image_path, view_angle,
                          prompt_text, negative_prompt=None):
    """Generate a side-view variant of a sticker via ComfyUI img2img.

    Args:
        client: ComfyClient instance
        output_dir: path to save temp files
        base_image_path: path to the front-facing sticker PNG
        view_angle: string like "left 45" or "right 45"
        prompt_text: original generation prompt
        negative_prompt: optional negative prompt

    Returns:
        RGBA numpy array on success, None on failure.
    """
    from app.utils.image_proc import load_rgba_sticker

    direction = "left" if "left" in view_angle else "right"
    view_prompt = (
        f"{prompt_text}, side view, viewed from the {direction} at 45 degrees, "
        f"same style, same sticker, same design, consistent"
    )

    try:
        result_path = client.generate_sync(
            prompt_text=view_prompt,
            workflow_name="img2img_controlnet_workflow_api.json",
            input_image_path=base_image_path,
            negative_prompt=negative_prompt,
        )
        if result_path:
            return load_rgba_sticker(result_path)
    except Exception as e:
        log.error("生成视角变体失败 (%s): %s", view_angle, e)

    return None
