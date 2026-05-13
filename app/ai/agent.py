# 基于 LLM 的任务编排引擎


import json
import re

from openai import OpenAI

from app.utils.config_loader import build_positive_prompt


REGION_GROUPS = {
    "head_top", "forehead_top", "forehead_full", "brows", "eyes",
    "nose", "mouth", "cheek_left", "cheek_right", "chin", "jaw"
}

REGION_ALIASES = {
    "forehead": "forehead_top",
    "头顶": "head_top",
    "cheek": "cheek_left",
    "right_cheek": "cheek_right",
    "左脸颊": "cheek_left",
    "右脸颊": "cheek_right",
    "下巴": "chin",
    "额头": "forehead_top",
    "眼睛": "eyes", "眼部": "eyes",
    "鼻子": "nose",
    "嘴巴": "mouth", "嘴部": "mouth",
    "脸颊": "cheek_left", "脸部": "cheek_left",
    "下颌": "jaw", "下颚": "jaw",
}


def _estimate_scale(region, keyword):
    large = {"猫耳", "兔耳", "耳朵", "帽子", "皇冠", "光环", "犄角", "触角", "王冠", "发带", "头巾", "护目镜"}
    small = {"鼻子", "猪鼻", "小丑鼻", "红鼻子", "雀斑", "爱心", "星星", "睫毛", "眉毛", "美瞳"}
    if keyword in large:
        return 1.4
    if keyword in small:
        return 0.5
    return 1.0


KEYWORD_REGION_MAP = {
    "head_top": ["猫耳", "兔耳", "耳朵", "帽子", "皇冠", "光环", "王冠", "犄角"],
    "forehead_top": ["触角", "发箍", "头箍"],
    "forehead_full": ["发饰", "头饰", "发带", "蝴蝶结", "头巾", "角"],
    "eyes": ["眼镜", "墨镜", "眼罩", "眼影", "眼线", "睫毛", "护目镜", "太阳镜", "美瞳", "眼"],
    "brows": ["眉毛"],
    "nose": ["鼻子", "猪鼻", "小丑鼻", "红鼻子", "狗鼻", "猫鼻", "鼻"],
    "mouth": ["胡子", "口罩", "嘴唇", "口红", "牙齿", "舌头", "嘴", "獠牙", "虎牙", "龅牙", "口"],
    "cheek_left": ["腮红", "面纹", "伤疤", "雀斑", "脸红", "纹身", "刀疤", "爱心", "星星", "脸", "面"],
}

ADJUST_KEYWORDS = {
    "scale_down": ["太大", "缩小", "小一点", "变小"],
    "scale_up": ["太小", "放大", "大一点", "变大"],
    "move_left": ["往左", "左移", "左一点"],
    "move_right": ["往右", "右移", "右一点"],
    "move_up": ["往上", "上移", "上一点"],
    "move_down": ["往下", "下移", "下一点"],
    "rotate_cw": ["顺时针", "右转"],
    "rotate_ccw": ["逆时针", "左转", "转一下"],
    "regenerate": ["太暗", "太亮", "换个颜色", "换个", "改成"],
    "remove": ["去掉", "删除", "不要"],
}

BASE_SYSTEM_PROMPT = """你是一个 AR 滤镜设计师助手。根据用户描述判断意图，输出 JSON。

## 输出格式

### 生成新贴纸 (generate):
{"action": "generate", "message": "给你加了一副墨镜~", "tasks": [{"prompt": "英文提示词", "region": "eyes", "scale": 1.0}]}

### 调整现有贴纸 (adjust):
{"action": "adjust", "target_index": 0, "adjustments": [{"type": "scale_mult", "value": 0.8}], "message": "已经缩小了"}

adjust type: offset_x(-0.15~0.15 左负右正), offset_y(-0.15~0.15 上负下正), rotation(-45~45 度 逆正顺负), scale_mult(0.5~1.5)

### 移除贴纸 (remove):
{"action": "adjust", "remove": true, "target_index": 0, "message": "已移除"}

### 需要澄清 (ask):
{"action": "ask", "message": "海盗主题通常包含眼罩和帽子，你想要哪些？"}

## message 规则
- generate 和 adjust 必须带 message，简短中文描述做了什么，例如"给你加了墨镜"、"把眼罩换成蓝色了"、"海盗帽缩小了"
- 多贴纸时描述所有贴纸内容，例如"给你加了海盗眼罩和帽子"
- message 是对话历史的一部分，后续对话会依赖它理解上下文

## 规则
- 单张贴纸也用 tasks 数组(一个元素)
- prompt 用英文，front view, flat lay, icon style，≤20词
- region: head_top/forehead_top/forehead_full/brows/eyes/nose/mouth/cheek_left/cheek_right/chin/jaw
- scale 0.3~2.0, 大型=1.3~1.8, 中型=0.9~1.2, 小型=0.3~0.7
- target_index 对应当前面部贴纸的序号(0开始), 用户没指定默认 0
- 能拆成独立贴纸的用 generate + 多 tasks
- 移动/旋转/缩放已有贴纸用 adjust
- 要求修改外观(颜色/形状)用 generate 重新生成
- 只有确实不确定时才用 ask"""


class FaceDoodleAgent:
    def __init__(self, api_key, model_id=None):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=api_key,
            )

        if model_id is None:
            from app.utils.config_loader import get_config
            model_id = get_config().get("agent", {}).get(
                "model_id", "deepseek-chat"
            )
        self.model_id = model_id
        self.system_prompt = BASE_SYSTEM_PROMPT

    # ── context builder ──

    def _build_sticker_context(self, active_stickers):
        if not active_stickers:
            return "当前面部无贴纸。"
        lines = ["当前面部贴纸："]
        for i, s in enumerate(active_stickers):
            prompt = s.get("prompt", "未知")
            location = s.get("location", "?")
            lines.append(f"  [{i}] {prompt} (位置: {location})")
        return "\n".join(lines)

    # ── response parsing ──

    def _extract_content_text(self, response):
        message = response.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()

        return str(content).strip()

    def _parse_json_text(self, raw_text):
        if not raw_text:
            raise ValueError("模型返回为空")

        decoder = json.JSONDecoder()

        try:
            obj, _ = decoder.raw_decode(raw_text)
            return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"无法从模型响应中提取 JSON: {raw_text}")

        try:
            obj, _ = decoder.raw_decode(match.group(0))
            return obj
        except json.JSONDecodeError:
            raise ValueError(f"无法解析提取的 JSON: {match.group(0)[:100]}")

    def _normalize_region(self, region):
        r = str(region or "").strip().lower()
        if r in REGION_GROUPS:
            return r
        mapped = REGION_ALIASES.get(str(region or "").strip(), None)
        if mapped:
            return mapped
        for alias, target in REGION_ALIASES.items():
            if alias in str(region or "").strip():
                return target
        return "eyes"

    def _validate_tasks(self, tasks):
        clean = []
        for t in tasks:
            prompt = str(t.get("prompt", "")).strip()
            if not prompt:
                continue
            region = self._normalize_region(t.get("region", "eyes"))
            try:
                scale = float(t.get("scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            scale = max(0.3, min(2.0, scale))
            clean.append({
                "prompt": build_positive_prompt(prompt),
                "region": region,
                "scale": scale,
            })
        return clean

    def _validate_adjustments(self, adjustments):
        valid_types = {"offset_x", "offset_y", "rotation", "scale_mult"}
        clean = []
        for adj in adjustments:
            t = adj.get("type", "")
            if t not in valid_types:
                continue
            try:
                v = float(adj.get("value", 0))
            except (TypeError, ValueError):
                continue
            if t == "scale_mult":
                v = max(0.3, min(3.0, v))
            elif t in ("offset_x", "offset_y"):
                v = max(-0.5, min(0.5, v))
            clean.append({"type": t, "value": v})
        return clean

    # ── keyword / heuristic fallbacks (no API key or API error) ──

    def _keyword_fallback(self, user_input):
        for region, keywords in KEYWORD_REGION_MAP.items():
            for kw in keywords:
                if kw in user_input:
                    print(f"[Agent] 关键词匹配: {kw} -> {region}")
                    return region, kw
        return None, None

    def _build_fallback_result(self, user_input):
        region, kw = self._keyword_fallback(user_input)
        if region is None:
            region = "eyes"
        scale = _estimate_scale(region, kw) if kw else 1.0
        print(f"[Agent] 使用{'关键词' if kw else '默认'} fallback: {region}, scale={scale}")
        return {
            "action": "generate",
            "message": f"已生成: {user_input}",
            "tasks": [{
                "prompt": build_positive_prompt(user_input),
                "region": region,
                "scale": scale,
            }],
            "workflow": "transparent_workflow_api.json"
        }

    _ADJUSTMENT_STEPS = [
        ("scale_down", "scale_mult", 0.8),
        ("scale_up", "scale_mult", 1.25),
        ("move_left", "offset_x", -0.08),
        ("move_right", "offset_x", 0.08),
        ("move_up", "offset_y", -0.08),
        ("move_down", "offset_y", 0.08),
        ("rotate_cw", "rotation", 15),
        ("rotate_ccw", "rotation", -15),
    ]

    def _adjustment_fallback(self, user_input, active_stickers):
        if not active_stickers:
            return None

        for kw in ADJUST_KEYWORDS["remove"]:
            if kw in user_input:
                return {
                    "action": "adjust",
                    "message": "好的，已移除贴纸",
                    "remove": True,
                    "target_index": 0,
                }

        adjustments = []
        for category, adj_type, value in self._ADJUSTMENT_STEPS:
            for kw in ADJUST_KEYWORDS[category]:
                if kw in user_input:
                    adjustments.append({"type": adj_type, "value": value})
                    break

        if adjustments:
            return {
                "action": "adjust",
                "message": "已调整",
                "adjustments": adjustments,
            }

        for kw in ADJUST_KEYWORDS["regenerate"]:
            if kw in user_input:
                return {
                    "action": "generate",
                    "message": "正在重新生成...",
                    "tasks": [{
                        "prompt": build_positive_prompt(user_input),
                        "region": active_stickers[0].get("location", "eyes"),
                        "scale": active_stickers[0].get("scale", 1.0),
                    }],
                    "workflow": "transparent_workflow_api.json"
                }

        return None

    # ── main API ──

    def chat(self, user_message, conversation_history=None, active_stickers=None):
        """Multi-turn chat interface.

        Returns dict with keys: action, message, and action-specific fields.
        """
        context = self._build_sticker_context(active_stickers or [])

        if not self.client:
            print("[Agent] 未配置 API Key")
            if active_stickers:
                adj_result = self._adjustment_fallback(user_message, active_stickers)
                if adj_result:
                    return adj_result
            return self._build_fallback_result(user_message)

        try:
            messages = [
                {"role": "system", "content": self.system_prompt + "\n\n" + context}
            ]
            if conversation_history:
                messages.extend(conversation_history[-6:])
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            raw_content = self._extract_content_text(response)
            print(f"[Agent] LLM 原始响应 ({len(raw_content)} 字符): {raw_content[:200]}")
            data = self._parse_json_text(raw_content)
            action = data.get("action", "generate")
            message = data.get("message", "")

            if action == "adjust":
                if data.get("remove"):
                    active = active_stickers or []
                    target_idx = int(data.get("target_index", 0))
                    target_instance = active[target_idx] if 0 <= target_idx < len(active) else (active[0] if active else None)
                    print(f"[Agent] 解析成功: action=remove, target_idx={target_idx}")
                    return {
                        "action": "adjust",
                        "message": message or "已移除",
                        "remove": True,
                        "target_instance": target_instance,
                    }

                adjustments = self._validate_adjustments(data.get("adjustments", []))
                if not adjustments:
                    print("[Agent] adjust 无有效调整项，降级为 ask")
                    return {"action": "ask", "message": message or "请告诉我具体怎么调整？"}

                active = active_stickers or []
                target_idx = int(data.get("target_index", 0))
                target_instance = active[target_idx] if 0 <= target_idx < len(active) else (active[0] if active else None)

                print(f"[Agent] 解析成功: action=adjust, target={target_idx}, adjustments={adjustments}")
                return {
                    "action": "adjust",
                    "message": message or "已调整",
                    "adjustments": adjustments,
                    "target_instance": target_instance,
                }

            elif action == "generate":
                tasks = self._validate_tasks(data.get("tasks", [data]))
                if not tasks:
                    prompt = str(data.get("prompt", "")).strip() or user_message
                    region = self._normalize_region(data.get("region", "eyes"))
                    try:
                        scale = float(data.get("scale", 1.0))
                    except (TypeError, ValueError):
                        scale = 1.0
                    scale = max(0.3, min(2.0, scale))
                    tasks = [{"prompt": build_positive_prompt(prompt), "region": region, "scale": scale}]

                if not message:
                    locations = [t.get("region", "?") for t in tasks]
                    if len(tasks) == 1:
                        message = f"已生成贴纸 (位置: {locations[0]})"
                    else:
                        message = f"已生成 {len(tasks)} 个贴纸 (位置: {', '.join(locations)})"
                print(f"[Agent] 解析成功: action=generate, tasks={len(tasks)}")
                return {
                    "action": "generate",
                    "message": message,
                    "tasks": tasks,
                    "workflow": "transparent_workflow_api.json",
                }

            elif action == "ask":
                print(f"[Agent] 反问: {message}")
                return {"action": "ask", "message": message or "请再描述一下你想要的效果？"}

            else:
                print(f"[Agent] 未知 action '{action}'，降级为 generate")
                return self._build_fallback_result(user_message)

        except Exception as e:
            print(f"[Agent] API 调用失败: {e}")
            if active_stickers:
                adj_result = self._adjustment_fallback(user_message, active_stickers)
                if adj_result:
                    return adj_result
            return self._build_fallback_result(user_message)

    # ── backward-compat wrapper ──

    def parse_command(self, user_input):
        """Legacy wrapper returning old dict format for existing callers."""
        result = self.chat(user_input)
        tasks = result.get("tasks", [])
        if tasks:
            task = tasks[0]
        else:
            task = {
                "prompt": build_positive_prompt(user_input),
                "region": "eyes",
                "scale": 1.0,
            }
        return {
            "positive_prompt": task["prompt"],
            "target_location": task["region"],
            "scale": task["scale"],
            "workflow": result.get("workflow", "transparent_workflow_api.json")
        }
