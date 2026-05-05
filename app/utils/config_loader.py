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
        "negative_prompt": "photo, realistic, 3D render, shadow, complex background, blur, noisy edges, text, watermark, signature, low quality, jpeg artifacts"
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


def _deep_merge(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
