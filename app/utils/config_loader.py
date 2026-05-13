import copy
import json
import os

DEFAULT_CONFIG = {
    "comfyui": {
        "server_address": "127.0.0.1:8188",
        "generate_timeout": 120
    },
    "camera": {
        "width": 1280,
        "height": 720
    },
    "video": {
        "path": "",
        "loop": True
    },
    "queue": {
        "frame_maxsize": 5,
        "display_maxsize": 5,
        "command_maxsize": 5
    },
    "agent": {
        "model_id": "deepseek-chat"
    },
    "model": {
        "lora": {
            "name": "game20icon20research.gW7e.safetensors",
            "strength_model": 0.8,
            "strength_clip": 0.8
        }
    },
    "generation": {
        "negative_prompt": "photo, realistic, 3D render, shadow, complex background, blur, noisy edges, text, watermark, signature, low quality, jpeg artifacts",
        "symmetry_enabled": False,
        "symmetry_positive_suffix": "symmetrical design, perfectly mirrored, balanced composition",
        "symmetry_negative_suffix": "asymmetrical, uneven, unbalanced, lopsided"
    },
    "preferences": {
        "default_region": "forehead_top",
        "default_scale": 1.0,
        "recent_prompts": [],
        "window_width": 1440,
        "window_height": 860
    },
    "external_editor": {
        "path": "D:/software/sai2/sai2.exe",
        "args": ""
    },
    "style": {
        "selected_preset": "pixel_art",
        "presets": {
            "pixel_art": {
                "name": "像素风格",
                "positive_prefix": "game icon institute, pixel art of {prompt}, retro game sprite, 16-bit style, clean pixel edges, iconic design",
                "lora_name": "gmic icon_Pixel style-000012.safetensors",
                "lora_strength_model": 0.8,
                "lora_strength_clip": 0.8
            },
            "vector_art": {
                "name": "矢量风格",
                "positive_prefix": "vector art of {prompt}, clean flat vector illustration, simple background, crisp edges, minimalist icon design",
                "lora_name": "vector_art_IL_MIX_V01.safetensors",
                "lora_strength_model": 2.0,
                "lora_strength_clip": 2.0
            },
            "cartoon_style": {
                "name": "卡通风格",
                "positive_prefix": "gmic icon \\(2dkat\\), cartoon illustration of {prompt}, vibrant cel shading, charming character design, clean edges, game item icon, masterpiece, best quality, good quality",
                "lora_name": "gmic icon_2d cartoon icon.safetensors",
                "lora_strength_model": 0.85,
                "lora_strength_clip": 0.85
            },
            "semi_realistic": {
                "name": "半写实",
                "positive_prefix": "highly detailed digital illustration of {prompt}, professional rendering, intricate details, sharp focus, clean composition, studio quality",
                "lora_name": "add-detail-xl.safetensors",
                "lora_strength_model": 1.5,
                "lora_strength_clip": 1.5
            }
        }
    }
}

_config = None


def load_config(config_path="config.json"):
    global _config
    config = copy.deepcopy(DEFAULT_CONFIG)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
            _deep_merge(config, file_config)

    config["comfyui"]["server_address"] = os.getenv(
        "COMFYUI_SERVER", config["comfyui"]["server_address"]
    )

    _config = config
    return config


def get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def save_config(cfg=None, config_path="config.json"):
    """Persist in-memory config to disk. Returns True on success."""
    if cfg is None:
        cfg = get_config()
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except (OSError, IOError) as e:
        print(f"[Config] 保存失败: {e}")
        return False


def _deep_merge(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def build_styled_prompt(prompt_text):
    """Wrap user prompt in the currently-selected style preset template.

    Falls back to the original hardcoded template when no style config exists.
    """
    import re
    preset = get_selected_preset()

    if preset and "positive_prefix" in preset:
        template = preset["positive_prefix"]
    else:
        template = "flat vector sticker of {prompt}, front view, flat lay, clean outline, solid white background, icon design"

    prompt = prompt_text.strip() if prompt_text else ""
    if prompt:
        return template.replace("{prompt}", prompt)
    return re.sub(r'\s*of\s*\{prompt\}\s*,?\s*', ', ', template).strip().rstrip(',').strip()


def get_style_preset_items():
    """Return list of (preset_key, display_name) tuples for UI population."""
    config = get_config()
    presets = config.get("style", {}).get("presets", {})
    return [(key, p.get("name", key)) for key, p in presets.items()]


def get_selected_preset():
    """Return the full dict of the currently-selected style preset."""
    config = get_config()
    style_cfg = config.get("style", {})
    presets = style_cfg.get("presets", {})
    key = style_cfg.get("selected_preset", "pixel_art")
    return presets.get(key, {})


def build_positive_prompt(prompt_text):
    """Wrap user prompt in style template, then append symmetry keywords if enabled."""
    prompt = build_styled_prompt(prompt_text)
    config = get_config()
    gen = config.get("generation", {})
    if gen.get("symmetry_enabled", False):
        suffix = gen.get("symmetry_positive_suffix", "").strip()
        if suffix and prompt:
            prompt = prompt.rstrip().rstrip(',') + ", " + suffix
    return prompt


def build_negative_prompt(override=None):
    """Build negative prompt, appending anti-symmetry suffix if symmetry is enabled.

    Args:
        override: If provided, used as the base instead of config default.
    """
    config = get_config()
    gen = config.get("generation", {})
    base = override if override else gen.get(
        "negative_prompt",
        "photo, realistic, 3D render, shadow, complex background, blur, noisy edges, text, watermark, signature, low quality, jpeg artifacts",
    )
    if gen.get("symmetry_enabled", False):
        suffix = gen.get("symmetry_negative_suffix", "").strip()
        if suffix and base:
            base = base.rstrip().rstrip(',') + ", " + suffix
    return base
