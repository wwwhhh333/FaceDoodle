# FaceDoodle AI — AR 面部贴纸生成与编辑

摄像头前实时生成、绘制、编辑面部贴纸。自然语言描述需求，DeepSeek 多轮对话解析意图，AI 自动生成透明 PNG 贴纸并贴合到脸上；支持手绘、简笔画精炼、贴纸动画、直接在脸上绘制。

## 功能

### AI 贴纸生成（聊天式交互）
- 底部聊天面板，自然语言描述需求（例如"海盗风格，给我眼罩和帽子"）
- DeepSeek 多轮对话解析意图，支持三种操作：
  - **生成** — 单/多贴纸生成，自动选面部区域和缩放
  - **反问** — 需求不明确时主动澄清
- ComfyUI SDXL + Layer Diffusion 生成透明背景贴纸
- 无 API Key 时自动降级为关键词匹配（50+ 中文关键词覆盖 12 个面部区域）

### 贴纸编辑
- `Ctrl+E` 进入编辑模式，左键拖拽移动、右键拖拽旋转、滚轮缩放、双击重置
- 贴纸的偏移/旋转/缩放数据持久化，下次启动自动恢复
- 面部网格线辅助定位

### 贴纸动画
- 关键帧动画系统：每张贴纸可绑定动画片段
- 支持的属性：offset_x、offset_y、rotation、scale_mult
- 5 种缓动函数：linear、ease-in、ease-out、ease-in-out
- 支持循环播放、Seek、导出 GIF/MP4
- 纹理动画：AI 驱动生成动态贴纸（motion prompt → ComfyUI 精灵表）
- 动画时间轴面板可视化编辑

### 面部绘制
- `Ctrl+D` 进入面部绘制模式，直接用鼠标或数位板在脸上画
- 笔触实时跟随面部移动，支持画笔/橡皮/撤销/清除
- 笔刷系统：PNG 蒙版定义笔刷尖端形状，支持硬圆笔/软圆笔及自定义笔刷
- 数位板压感映射到笔刷大小和/或浓度
- 快捷键：`B` 画笔 `E` 橡皮 `[` `]` 粗细 `Ctrl+Z` 撤销 `C` 清除 `S` 保存

### 简笔画精炼 (img2img + ControlNet)
- 独立画板手绘简笔画 + 文字描述
- ControlNet Scribble 将草图精炼为精致贴纸
- 支持镜像绘制、导入外部图片、调用外部编辑器

### 系统模板
- 9 张预设模板贴纸（面部轮廓、皇冠、眉心点、星星眼、鼻影、微笑、腮红、下巴点）
- 首次启动自动生成，与用户贴纸统一管理

### 视频文件测试
- `--video <path>` 用本地视频文件代替摄像头输入，方便无摄像头时调试
- 自动读取视频 FPS 控制播放速度，支持循环播放
- 可与 `--mock` 结合使用，完全离线测试

### 画廊管理
- 三个过滤标签：`模板` | `贴纸` | `收藏`
- 点击贴纸应用到脸上，再次点击取消

## 自定义笔刷

将 PNG 蒙版文件放入 `assets/brushes/`，编辑 `assets/brushes/brushes.json` 添加配置：

```json
{
  "brushes": [
    {
      "id": "star",
      "name": "星形笔",
      "tip": "star.png",
      "spacing": 0.25,
      "pressure_size": true,
      "pressure_opacity": true,
      "scatter": 0.0
    }
  ]
}
```

Windows 下 OpenCV 不支持中文路径，PNG 文件名请用英文，笔刷名称可以用中文。

## 系统架构

```
摄像头/视频文件 → Producer 进程 → Consumer 进程 → 渲染帧 → Qt UI
                         DeepSeek API    ComfyUI
                         (多轮对话)     (图片生成)
```

- **Producer**: 摄像头采集或视频文件读取，帧推送到 `frame_queue`（视频模式按 FPS 节流）
- **Consumer**: MediaPipe 468 点面部检测 + 贴纸渲染 + AI 调度 + 动画引擎。Mixin 架构：`ConsumerProcessor(StickerManager, AnimationProcessor)`
- **队列通信**: 7 个多进程队列 + 1 个内部 result_queue，所有消息使用 typed dataclass (`app/core/protocol.py`)
- **UI**: PySide6 界面（dark-first 现代主题），`VideoUpdateThread` 拉取 `display_queue` 帧和状态消息
- **ComfyUI**: SDXL + Layer Diffusion 生成透明 PNG（工作流 API）

## 环境要求

| 组件 | 说明 |
|------|------|
| Python | 3.10+ |
| 摄像头 | 系统默认摄像头（可用 `--video` 替代） |
| ComfyUI | 本地运行，需 LayerDiffusion + ControlNet 节点 |
| DeepSeek API | 自然语言解析（设置 `DEEPSEEK_API_KEY` 环境变量） |
| 数位板（可选） | Windows Ink 兼容数位板，支持压感 |

### Python 依赖

完整列表见 `requirements.txt`，核心依赖：`opencv-python` `mediapipe` `PySide6` `openai` `aiohttp` `numpy`

### ComfyUI 依赖

| 模型/节点 | 用途 | 下载 |
|-----------|------|------|
| sdXL_v10VAEFix.safetensors | SDXL 底模 | [HuggingFace](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |
| LayerDiffusion | 生成透明 PNG | [GitHub](https://github.com/layerdiffusion/LayerDiffuse) |
| ControlNet Scribble | img2img 精炼 | [HuggingFace](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0) |
| gmic icon_Pixel style | 像素风格 LoRA | [CivitAI](https://civitai.com/models/160165/gmic-icon-pixel-style) |
| vector_art_IL_MIX_V01 | 矢量风格 LoRA | [CivitAI](https://civitai.com/models/208905/vector-art-il-mix) |
| gmic icon_2d cartoon icon | 卡通风格 LoRA | [CivitAI](https://civitai.com/models/159956/gmic-icon-2d-cartoon-icon) |
| add-detail-xl | 半写实风格 LoRA | [CivitAI](https://civitai.com/models/135867/add-detail-xl) |

LoRA 文件放入 ComfyUI 的 `models/loras/` 目录。

### ComfyUI 自动启动

在设置对话框中配置 ComfyUI 安装路径后，FaceDoodle 启动时自动拉起 ComfyUI，退出时自动关闭。留空则需手动启动。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 FaceDoodle（首次运行会自动弹出设置向导）
python app/main.py

# 3. 在设置向导中填入 API Key 和 ComfyUI 地址即可开始使用
#    配置 ComfyUI 安装路径后，FaceDoodle 会自动启动/关闭 ComfyUI

# Mock 模式（跳过 ComfyUI，用缓存图片测试 UI）
python app/main.py --mock

# 视频文件模式（代替摄像头，可结合 --mock 使用）
python app/main.py --video test_data/face_test.mp4
python app/main.py --video test_data/face_test.mp4 --mock
```

## 配置

`config.json` 主要配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `comfyui.server_address` | `127.0.0.1:8188` | ComfyUI 服务地址 |
| `comfyui.generate_timeout` | `120` | 生成超时（秒） |
| `comfyui.install_path` | (空) | ComfyUI 安装目录，设定后自动启动 |
| `camera.width` / `camera.height` | `1280` / `720` | 摄像头分辨率 |
| `video.path` | `""` | 视频文件模式默认路径（`--video` 无参时使用） |
| `video.loop` | `true` | 视频播放结束后循环 |
| `agent.model_id` | `deepseek-chat` | LLM 模型 |
| `model.lora.name` | 贴纸 LoRA | ComfyUI LoRA 文件名 |
| `external_editor.path` | — | 外部图片编辑器路径 |

## 测试与代码质量

```bash
# 运行全量测试（401 个）
python -m pytest tests/ -v

# 单文件
python -m pytest tests/test_agent.py -v

# Pre-commit 验证关卡（语法 + 测试）
pre-commit run --all-files
```

每次 git commit 时自动触发 pre-commit hooks（语法检查 + 测试套件）。

## 快捷键

### 全局
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+E` | 切换编辑模式 |
| `Ctrl+D` | 切换面部绘制模式 |
| `Enter` | 发送 AI 生成指令 |

### 编辑模式
| 操作 | 功能 |
|------|------|
| 左键拖拽 | 移动贴纸 |
| 右键拖拽 | 旋转贴纸 |
| 滚轮 | 缩放贴纸 |
| 双击 | 重置位置 |

### 面部绘制模式
| 快捷键 | 功能 |
|--------|------|
| `B` | 画笔 |
| `E` | 橡皮 |
| `[` `]` | 减小/增大笔刷 |
| `Ctrl+Z` | 撤销 |
| `C` | 清除画布 |
| `S` | 保存为贴纸 |

## 项目结构

```
FaceDoodle/
├── app/
│   ├── main.py                  # 入口，进程与队列初始化
│   ├── ai/
│   │   ├── agent.py             # DeepSeek 多轮对话解析（generate/adjust/ask）
│   │   └── generator.py         # ComfyUI API 客户端
│   ├── core/
│   │   ├── animation/           # 动画系统（clip/engine/texture/gen/export）
│   │   ├── brush.py             # 笔刷引擎
│   │   ├── face_mesh.py         # MediaPipe 468 点检测
│   │   ├── face_draw.py         # 面部绘制画布 + 坐标映射
│   │   ├── protocol.py          # 所有队列消息的 typed dataclass 定义
│   │   ├── renderer.py          # 贴纸透视贴合 + 加载动画 + 面部网格
│   │   ├── templates.py         # 系统模板贴纸生成
│   │   ├── tracker.py           # Consumer 主循环，AI 调度与渲染编排
│   │   ├── tracker_stickers.py  # StickerManager mixin（贴纸增删改查）
│   │   └── tracker_animation.py # AnimationProcessor mixin（动画队列处理）
│   ├── ui/
│   │   ├── main_window.py       # 主窗口 UI + 事件处理 + 画廊管理
│   │   ├── chat_panel.py        # 聊天消息面板（多轮对话气泡）
│   │   ├── widgets.py           # 组件：画廊卡片/画板/按钮
│   │   ├── sticker_panel.py     # 贴纸面板组件
│   │   ├── drawing_widgets.py   # 绘制相关组件
│   │   ├── animation_timeline.py # 动画时间轴面板
│   │   ├── animation_gen_dialog.py # 动画生成对话框
│   │   └── theme.py             # 主题色板与字体
│   ├── utils/
│   │   ├── config_loader.py     # 配置文件加载
│   │   ├── image_proc.py        # 图像加载与预处理
│   │   └── storage.py           # 贴纸持久化存储
│   └── workflows/               # ComfyUI 工作流 JSON
├── assets/
│   ├── brushes/                 # 笔刷蒙版 PNG + 配置
│   ├── templates/               # 系统模板（自动生成）
│   └── gallery/                 # 用户贴纸
├── scripts/
│   └── check_syntax.py          # Pre-commit 语法检查脚本
├── tests/                       # pytest 测试套件（401 个用例）
├── config.json                  # 应用配置
├── requirements.txt              # Python 依赖
├── .pre-commit-config.yaml      # Git 提交前验证关卡
└── README.md
```
