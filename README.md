# FaceDoodle AI — 智能 AR 贴纸工坊

在摄像头前实时生成、绘制、编辑面部贴纸。自然语言描述需求，AI 自动生成透明 PNG 贴纸并贴合到脸上；也支持手绘、简笔画精炼、直接在脸上绘制。

## 功能

### AI 贴纸生成
- 输入自然语言描述（例如"一副赛博朋克护目镜"），DeepSeek 解析意图，ComfyUI 生成透明背景贴纸
- 自动选择最佳面部位置（额头/眼睛/鼻子/嘴巴/脸颊等），也可手动指定

### 面部绘制
- `Ctrl+D` 进入面部绘制模式，直接用鼠标或数位板在脸上画
- 笔触实时跟随面部移动，支持画笔/橡皮/撤销/清除
- 笔刷系统：PNG 蒙版定义笔刷尖端形状（SAI2 风格），支持硬圆笔/软圆笔及自定义笔刷
- 数位板压感支持：压力映射到笔刷大小和/或浓度，四种压感模式可切换
- 笔刷间距、散射参数可实时调节
- 快捷键：`B` 画笔 `E` 橡皮 `[` `]` 粗细 `Ctrl+Z` 撤销 `C` 清除 `S` 保存
- 保存后自动入库

### 简笔画精炼 (img2img + ControlNet)
- 在独立画板中手绘简笔画，搭配文字描述
- 通过 ControlNet Scribble 模型将草图精炼为精致贴纸
- 双栏布局：左侧工具面板 + 右侧属性面板，操作区域宽裕
- 支持镜像绘制、导入外部图片、调用外部编辑器（如 SAI2/Photoshop）

### 系统模板
- 9 张预设模板贴纸，覆盖全部面部区域（面部轮廓、皇冠、眉心点、星星眼、鼻影、微笑、腮红、下巴点）
- 首次启动自动生成到 `assets/templates/`
- 与用户贴纸统一管理

### 贴纸编辑
- `Ctrl+E` 进入编辑模式，左键拖拽移动、右键拖拽旋转、滚轮缩放、双击重置
- 贴纸的偏移/旋转/缩放数据持久化，下次启动自动恢复
- 面部网格线辅助定位

### 画廊管理
- 右侧面板三个过滤标签：`模板` | `贴纸` | `收藏`
- 贴纸卡片支持收藏（星标标识），收藏视图可筛选
- 点击贴纸应用到脸上，再次点击取消使用
- 绿色边框 = 正在使用，深色边框 = 已选中

## 自定义笔刷

将 PNG 蒙版文件放入 `assets/brushes/` 目录，编辑 `assets/brushes/brushes.json` 添加笔刷配置即可。PNG 的 alpha 通道作为笔刷形状（alpha=255 完全着色，alpha=0 透明）。

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

> 注意：Windows 下 OpenCV 不支持中文路径，PNG 文件名请使用英文，笔刷名称（name）可以使用中文。

## 系统架构

```
摄像头 → Producer 进程 → Consumer 进程 → 渲染帧 → Qt UI
                DeepSeek API   ComfyUI
                (指令解析)    (图片生成)
```

- **Producer**: 摄像头采集，30fps 帧推送到队列
- **Consumer**: MediaPipe 468 点面部检测 + 贴纸渲染 + AI 调度
- **UI**: PyQt5 界面，多进程队列通信
- **ComfyUI**: SDXL + Layer Diffusion 生成透明 PNG（工作流 API 模式）

## 环境要求

| 组件 | 说明 |
|------|------|
| Python | 3.10+ |
| 摄像头 | 系统默认摄像头 |
| ComfyUI | 本地运行，需安装 LayerDiffusion + ControlNet 节点 |
| DeepSeek API | 自然语言解析（设置 `DEEPSEEK_API_KEY` 环境变量） |
| 数位板（可选） | Windows Ink 兼容数位板，支持压感 |

### Python 依赖

```
opencv-python
mediapipe
PyQt5
openai
aiohttp
numpy
```

完整列表见 `requirement.txt`。

### ComfyUI 依赖

确保 ComfyUI 已安装以下模型/节点：

| 模型/节点 | 用途 |
|-----------|------|
| sdXL_v10VAEFix.safetensors | SDXL 底模 |
| LayerDiffusion (LayeredDiffusionApply / LayeredDiffusionDecode) | 生成透明 PNG |
| ControlNet (ControlNetLoader / ControlNetApply) | img2img 精炼 |
| controlnet-scribble-sdxl-1.0.safetensors | 简笔画 ControlNet |
| LoRA 贴纸模型 | 贴纸风格 |

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url>
cd FaceDoodle

# 2. 安装 Python 依赖
pip install -r requirement.txt

# 3. 设置 API Key
$env:DEEPSEEK_API_KEY = "sk-your-key-here"

# 4. 启动 ComfyUI（另一个终端）
cd path/to/ComfyUI
python main.py

# 5. 启动 FaceDoodle
python app/main.py

# 可选：Mock 模式（跳过 ComfyUI，使用本地缓存图片测试 UI）
python app/main.py --mock
```

## 配置

`config.json` 主要配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `comfyui.server_address` | `127.0.0.1:8188` | ComfyUI 服务地址 |
| `comfyui.generate_timeout` | `120` | 生成超时（秒） |
| `camera.width` / `camera.height` | `1280` / `720` | 摄像头分辨率 |
| `agent.model_id` | `deepseek-chat` | LLM 模型 |
| `model.lora.name` | 贴纸 LoRA | ComfyUI 使用的 LoRA 文件名 |
| `external_editor.path` | — | 外部图片编辑器路径 |
| `external_editor.args` | — | 启动参数（空格分隔） |

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

### 绘制贴纸窗口
| 快捷键 | 功能 |
|--------|------|
| `B` | 画笔模式 |
| `E` | 橡皮模式 |
| `M` | 切换镜像 |
| `Ctrl+Z` | 撤销 |
| `[` `]` | 减小/增大笔刷 |

## 项目结构

```
FaceDoodle/
├── app/
│   ├── main.py              # 入口，进程与队列初始化
│   ├── ai/
│   │   ├── agent.py         # DeepSeek 自然语言解析
│   │   └── generator.py     # ComfyUI API 客户端
│   ├── core/
│   │   ├── brush.py         # 笔刷引擎：蒙版加载、stamp 渲染、压感
│   │   ├── face_mesh.py     # MediaPipe 468 点检测 + 关键点分组
│   │   ├── face_draw.py     # 面部绘制画布 + 坐标映射 + 笔刷管线
│   │   ├── renderer.py      # 贴纸透视贴合 + 加载动画 + 面部网格
│   │   ├── templates.py     # 系统模板贴纸生成与加载
│   │   └── tracker.py       # Consumer 主循环，AI 调度与渲染编排
│   ├── ui/
│   │   ├── main_window.py   # 主窗口 UI + 事件处理 + 画廊管理
│   │   └── widgets.py       # 组件：画廊卡片/画板/按钮/绘制对话框
│   ├── utils/
│   │   ├── config_loader.py # 配置文件加载
│   │   ├── image_proc.py    # 图像加载与预处理
│   │   └── storage.py       # 贴纸持久化存储
│   └── workflows/
│       ├── transparent_workflow_api.json         # 文生贴纸工作流
│       ├── img2img_workflow_api.json             # 图生贴纸工作流
│       └── img2img_controlnet_workflow_api.json  # ControlNet 精炼工作流
├── assets/
│   ├── brushes/             # 笔刷蒙版 PNG + brushes.json 配置
│   ├── templates/           # 系统模板贴纸（自动生成）
│   └── gallery/             # 用户贴纸存储目录
├── config.json              # 应用配置
├── requirement.txt          # Python 依赖
└── README.md
```
