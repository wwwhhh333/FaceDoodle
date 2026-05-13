# 基于 LLM 的任务编排引擎


import json
import time

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

REGION_DISPLAY = {
    "head_top": "头顶", "forehead_top": "额头", "forehead_full": "额头",
    "brows": "眉毛", "eyes": "眼睛", "nose": "鼻子", "mouth": "嘴巴",
    "cheek_left": "左脸", "cheek_right": "右脸", "chin": "下巴", "jaw": "下颌",
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

BASE_SYSTEM_PROMPT = """你是一个 AR 滤镜设计师助手。根据用户描述判断意图，输出 JSON。

## 输出格式

### 生成新贴纸 (generate):
{"action": "generate", "message": "给你加了一副墨镜~", "tasks": [{"prompt": "英文提示词", "region": "eyes", "scale": 1.0}]}

### 需要澄清 (ask):
{"action": "ask", "message": "海盗主题通常包含眼罩和帽子，你想要哪些？"}

## message 规则
- generate 必须带 message，简短中文描述做了什么，例如"给你加了墨镜"
- 多贴纸时描述所有贴纸内容，例如"给你加了海盗眼罩和帽子"
- message 是对话历史的一部分，后续对话会依赖它理解上下文

## 规则
- 单张贴纸也用 tasks 数组(一个元素)
- prompt 用英文，front view, flat lay, icon style，≤20词
- region: head_top/forehead_top/forehead_full/brows/eyes/nose/mouth/cheek_left/cheek_right/chin/jaw
- scale 0.3~2.0, 大型=1.3~1.8, 中型=0.9~1.2, 小型=0.3~0.7
- 能拆成独立贴纸的用 generate + 多 tasks
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

    def _extract_json_object(self, text):
        """Extract the first balanced JSON object from text, handling nesting."""
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _parse_json_text(self, raw_text):
        if not raw_text:
            raise ValueError("模型返回为空")

        decoder = json.JSONDecoder()

        try:
            obj, _ = decoder.raw_decode(raw_text)
            return obj
        except json.JSONDecodeError:
            pass

        json_str = self._extract_json_object(raw_text)
        if json_str is None:
            raise ValueError(f"无法从模型响应中提取 JSON: {raw_text}")

        try:
            obj, _ = decoder.raw_decode(json_str)
            return obj
        except json.JSONDecodeError:
            raise ValueError(f"无法解析提取的 JSON: {json_str[:100]}")

    def _normalize_region(self, region):
        r = str(region or "").strip().lower()
        if r in REGION_GROUPS:
            return r
        mapped = REGION_ALIASES.get(r, None)
        if mapped:
            return mapped
        for alias, target in REGION_ALIASES.items():
            if alias in r:
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
        display_region = REGION_DISPLAY.get(region, region)
        print(f"[Agent] 使用{'关键词' if kw else '默认'} fallback: {region}, scale={scale}")

        if kw:
            messages = [
                f"帮你加上了{kw}，放在{display_region}位置~",
                f"好的，给你画上{kw}啦",
                f"{kw}安排上了！",
            ]
            msg = messages[hash(user_input) % len(messages)]
        else:
            msg = f"帮你生成了「{user_input}」的贴纸"

        return {
            "action": "generate",
            "message": msg,
            "tasks": [{
                "prompt": build_positive_prompt(user_input),
                "region": region,
                "scale": scale,
            }],
            "workflow": "transparent_workflow_api.json"
        }

    # ── main API ──

    def _call_api(self, messages, retries=2):
        """Call DeepSeek API with retry on transient failures."""
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                return response
            except Exception as e:
                last_error = e
                if attempt < retries:
                    delay = 1.0 * (2 ** attempt)
                    print(f"[Agent] API 调用失败 (尝试 {attempt + 1}/{retries + 1}): {e}，{delay}s 后重试")
                    time.sleep(delay)
        raise last_error

    def chat(self, user_message, conversation_history=None, active_stickers=None):
        """Multi-turn chat interface.

        Returns dict with keys: action, message, and action-specific fields.
        """
        context = self._build_sticker_context(active_stickers or [])

        if not self.client:
            print("[Agent] 未配置 API Key，使用关键词回退")
            return self._build_fallback_result(user_message)

        try:
            messages = [
                {"role": "system", "content": self.system_prompt + "\n\n" + context}
            ]
            if conversation_history:
                messages.extend(conversation_history[-6:])
            messages.append({"role": "user", "content": user_message})

            response = self._call_api(messages)

            raw_content = self._extract_content_text(response)
            print(f"[Agent] LLM 原始响应 ({len(raw_content)} 字符): {raw_content[:200]}")
            data = self._parse_json_text(raw_content)
            action = data.get("action", "generate")
            message = data.get("message", "")

            if action == "generate":
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
                    locations = [REGION_DISPLAY.get(t.get("region", ""), t.get("region", "?")) for t in tasks]
                    if len(tasks) == 1:
                        message = f"帮你生成了一张贴纸，放在了{locations[0]}位置~"
                    else:
                        message = f"帮你生成了{len(tasks)}张贴纸，分别放在：{'、'.join(locations)}"
                print(f"[Agent] 解析成功: action=generate, tasks={len(tasks)}")
                return {
                    "action": "generate",
                    "message": message,
                    "tasks": tasks,
                    "workflow": "transparent_workflow_api.json",
                }

            elif action == "ask":
                print(f"[Agent] 反问: {message}")
                if not message:
                    message = "不太确定你想要什么效果，能再具体描述一下吗？"
                return {"action": "ask", "message": message}

            else:
                print(f"[Agent] 未知 action '{action}'，降级为 generate")
                return self._build_fallback_result(user_message)

        except Exception as e:
            print(f"[Agent] API 调用失败，使用关键词回退: {e}")
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
