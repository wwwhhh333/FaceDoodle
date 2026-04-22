# 基于 LLM 的任务编排引擎


import json
import re

from openai import OpenAI


# 位置映射表：将 AI 返回的关键词映射到 Renderer 能识别的面部区域
LOCATION_MAPPING = {
    "forehead": "额头",   # 适用于：帽子、发饰、光环
    "eyes": "眼睛",       # 适用于：眼镜、眼罩、瞳色
    "nose": "鼻子",       # 适用于：小丑红鼻子、胡须
    "mouth": "嘴部",      # 适用于：口罩、牙齿、口红
    "cheek": "脸颊",      # 适用于：面纹、腮红
}

LOCATION_ALIASES = {
    "额头": "forehead",
    "眼睛": "eyes",
    "眼部": "eyes",
    "鼻子": "nose",
    "嘴巴": "mouth",
    "嘴部": "mouth",
    "脸颊": "cheek",
    "脸部": "cheek",
}


class FaceDoodleAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.client = OpenAI(
                base_url="https://api-inference.modelscope.cn/v1",
                api_key=api_key,
            )
        self.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


        # 核心逻辑：定义结构化输出
        self.system_prompt = """
你是一个 AR 滤镜设计师。你的任务是解析用户需求，并且只输出一个 JSON 对象。

输出格式示例：
{"prompt": "pirate eyepatch, black leather", "location": "eyes"}

要求：
1. 只能输出 JSON，不要输出解释。
2. location 只能是以下值之一：
["forehead", "eyes", "nose", "mouth", "cheek"]
3. prompt 用简洁英文描述贴纸内容，不要重复 location。
"""

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

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"无法从模型响应中提取 JSON: {raw_text}")

        return json.loads(match.group(0))

    def _normalize_location(self, location):
        normalized = str(location or "").strip().lower()
        if normalized in LOCATION_MAPPING:
            return normalized

        return LOCATION_ALIASES.get(str(location or "").strip(), "eyes")

    def parse_command(self, user_input):
        fallback_result = {
            "positive_prompt": f"{user_input}, sticker style, white background",
            "target_location": "eyes",  # 默认挂载到眼睛
            "workflow": "transparent_workflow_api.json"
        }

        if not self.client:
            print("[Agent] 未配置 MODELSCOPE_API_KEY，使用默认配置。")
            return fallback_result

        request_kwargs = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"用户指令：{user_input}"}
            ],
            "temperature": 0.2,
        }

        try:
            try:
                response = self.client.chat.completions.create(
                    **request_kwargs,
                    response_format={"type": "json_object"}
                )
            except Exception as first_error:
                print(f"[Agent] 结构化输出请求失败，尝试兼容模式。错误信息: {first_error}")
                response = self.client.chat.completions.create(**request_kwargs)

            raw_content = self._extract_content_text(response)
            data = self._parse_json_text(raw_content)
            prompt = str(data.get("prompt", "")).strip() or user_input
            location = self._normalize_location(data.get("location"))

            print(f"[Agent] 解析成功: 贴纸内容->{prompt}, 挂载位置->{location}")

            return {
                "positive_prompt": f"{prompt}, sticker style, white background",
                "target_location": location,
                "workflow": "transparent_workflow_api.json"
            }
        except Exception as e:
            print(f"[Agent] API 调用失败，使用默认配置。错误信息: {e}")
            return fallback_result
