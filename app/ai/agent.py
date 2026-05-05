# 基于 LLM 的任务编排引擎


import json
import re

from openai import OpenAI


# 位置映射表：将 AI 返回的关键词映射到 Renderer 能识别的面部区域
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

        self.system_prompt = """
你是一个 AR 滤镜设计师。你的任务是解析用户需求，并且只输出一个 JSON 对象。

输出格式示例：
{"prompt": "pirate eyepatch, black leather", "region": "eyes", "scale": 1.0}

要求：
1. 只能输出 JSON，不要输出解释。
2. prompt 用英文描述贴纸内容，必须包含"正面平铺、无透视、像一枚徽章或图标"，加上材质、颜色，不超过20词。例如： "glasses front view, flat lay, icon style, black thick frame, symmetric" 而不是 "a pair of glasses"。
3. region 从以下人脸关键点组中选择最合适的：
   head_top(头顶,猫耳/帽子/皇冠/兔耳), forehead_top(发际线,触角/发箍), forehead_full(整个额头,头巾/绷带), brows(眉毛),
   eyes(眼睛,眼镜/眼罩), nose(鼻子,鼻环/红鼻头), mouth(嘴部,口罩/胡子/嘴唇),
   cheek_left(左脸颊,腮红/伤疤), cheek_right(右脸颊), chin(下巴), jaw(下颌线)
4. scale 是贴纸缩放系数，0.3到2.0，默认1.0。根据贴纸类型判断：大型饰品(帽子/猫耳/翅膀)=1.3~1.8，中型(眼镜/口罩)=0.9~1.2，小型(鼻环/雀斑/星星)=0.3~0.7。
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

    def _normalize_region(self, region):
        r = str(region or "").strip().lower()
        if r in REGION_GROUPS:
            return r
        mapped = REGION_ALIASES.get(str(region or "").strip(), None)
        if mapped:
            return mapped
        # 中文直接匹配
        for alias, target in REGION_ALIASES.items():
            if alias in str(region or "").strip():
                return target
        return "eyes"

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
            "positive_prompt": f"flat vector sticker of {user_input}, front view, flat lay, clean outline, solid white background, icon design",
            "target_location": region,
            "scale": scale,
            "workflow": "transparent_workflow_api.json"
        }

    def parse_command(self, user_input):
        if not self.client:
            print("[Agent] 未配置 API Key，使用关键词匹配。")
            return self._build_fallback_result(user_input)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"用户指令：{user_input}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            raw_content = self._extract_content_text(response)
            data = self._parse_json_text(raw_content)
            prompt = str(data.get("prompt", "")).strip() or user_input
            region = self._normalize_region(data.get("region") or data.get("location"))
            try:
                scale = float(data.get("scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            scale = max(0.3, min(2.0, scale))

            print(f"[Agent] 解析成功: 贴纸内容->{prompt}, 区域->{region}, 缩放->{scale}")

            return {
                "positive_prompt": f"flat vector sticker of {prompt}, front view, flat lay, clean outline, solid white background, icon design",
                "target_location": region,
                "scale": scale,
                "workflow": "transparent_workflow_api.json"
            }
        except Exception as e:
            print(f"[Agent] API 调用失败，使用关键词匹配。错误信息: {e}")
            return self._build_fallback_result(user_input)
