# ADR 0001: FaceDoodle 从桌面应用迁移至 Web 架构

**日期**: 2026-05-21
**状态**: 已接受

## 背景

FaceDoodle 当前为 PySide6 桌面应用，三进程架构（Producer → Consumer → Qt UI）。用户希望将 UI 层改为浏览器端，保留 Python 后端在本地运行。

## 决策

将应用改为浏览器客户端 + 本地 Python 后端的 Web 架构。

### 决策 1：面部检测 — MediaPipe JS（浏览器端）

**备选**：保持 MediaPipe Python（服务器端检测）。

**选择浏览器端**。面部检测在本地 Canvas 帧上执行，不需要上传原始帧到服务器。MediaPipe JS 的 WASM 运行时首次加载约 5-8MB，但随后缓存在浏览器中。服务器不做检测，CPU 全用于 AI 生成。WASM 的延迟 <1ms，而帧上传走 WebSocket 至少 5-10ms。

### 决策 2：通信协议 — WebSocket 主通道 + HTTP 文件上传

**备选**：纯 HTTP 轮询、单一 WebSocket 承载一切、WebRTC。

**选择 WebSocket + HTTP 分离**。WebSocket 替换当前 8 条 `multiprocessing.Queue`，承载 `display_queue` 帧推送、`command_queue` 生成指令、`adjustment_queue` 编辑操作等所有实时双向消息。HTTP 专用于文件上传（Canvas 手绘保存、外部编辑器回传），因为 WebSocket 对大文件没有流式支持。

WebRTC 被排除——当前不需要多人协作或超低延迟视频通话，增加信令服务器得不偿失。

### 决策 3：前端框架 — Vue 3 + Canvas 视频层

**备选**：纯 HTML/JS、React、Flutter Web。

**选择 Vue 3**。Vue 3 的 Composition API 对独立开发者最友好，单文件组件结构与当前 `main_window.py` 的自包含风格一致。视频帧通过 `<canvas>` 逐帧绘制 `ImageData`，不走 DOM。UI 组件（设置面板、画廊、时间轴、聊天面板）用 Vue 组合式组件。

React 未被选入——JSX 对模板类 UI（表单、列表、面板）的直观性不如 Vue 的 `<template>` 语法。纯 HTML/JS 在组件数量超过 10 个后维护成本失控。

### 决策 4：摄像头帧 — 640×480 WebP 压缩

**备选**：1280×720 原始 BGR、JPEG 压缩、跳帧策略。

**选择 640×480 WebP**。MediaPipe FaceMesh 在 480p 下检测精度与 720p 基本一致（官方 benchmark 显示 480p 的 468 个关键点误差 <2px）。WebP 在浏览器端原生支持 `canvas.toBlob("image/webp")`，压缩率约 5:1，640×480 每帧约 20-30KB，15fps 约 300-450KB/s，WebSocket 完全够。

1280×720 每帧 2.76MB 的原始 BGR 被排除——15fps 约 40MB/s 超过 WebSocket 单通道承受能力。JPEG 被排除是因为 WebP 在同等质量下体积小 25-34%。

### 决策 5：贴纸渲染 — 浏览器端 Canvas 2D API

**备选**：服务器端渲染后推送完整帧。

**选择浏览器端渲染**。`renderer.py` 的透视变换和 Alpha 合成逻辑迁移到 Canvas 2D API（`ctx.setTransform()` → `ctx.drawImage()`）。渲染在浏览器主线程上运行，延迟为 0。

`renderer.py` 中的 `solvePnP` 3D 头部变形无法直接平移——Canvas 2D 没有 3D 投影 API。降级为 `renderer.py` 已有的 2D 启发式方法（根据 yaw_ratio/pitch_ratio 缩放和位移四边形）。

**成本**：浏览器渲染代码约 300-500 行 JS（对应 `renderer.py` 566 行），需要重写。但收益是服务器 CPU 全用于 ComfyUI，渲染不占用推理资源。

### 决策 6：部署 — 本地运行

**备选**：部署到云服务器。

**选择本地运行**。保留当前核心能力：`subprocess.Popen` 启动外部编辑器（PS/SAI2）、直接读写 `assets/gallery/`、本地 ComfyUI GPU 推理、`api_key.txt` 文件读取。云部署将失去所有这些能力，且需要额外认证系统、对象存储和 GPU 云实例。

本地运行意味着：
- `localhost:5000` 作为后端地址
- 浏览器仅作 UI 层，无权访问文件系统
- 多设备访问不在范围——用户必须在本机使用

## 架构变化

```
迁移前（桌面应用）:
  摄像头 → Producer 进程 ──frame_queue──→ Consumer 进程 ──display_queue──→ PySide6 UI
                DeepSeek API    ComfyUI                        Qt widgets (4,573 行)

迁移后（Web 应用）:
  浏览器: MediaPipe JS → Canvas 渲染 ←→ WebSocket ←→ Python 后端 (Consumer + AI)
          Vue 3 UI 组件                               本地文件系统 / ComfyUI
```

## 代码影响

| 代码层 | 变化 |
|--------|------|
| `app/core/` (tracker, renderer, protocol) | `renderer.py` 的透视贴合逻辑迁移到 JS，其余协议保持 |
| `app/ai/` (agent, generator, comfy_manager) | 不变——仍由 Python 后端调用 |
| `app/ui/` (main_window, theme, widgets...) | **完全移除**，由 Vue 组件替代 |
| `app/main.py` | 替换为 Flask/Quart WebSocket 服务器入口 |
| 新增 | `web/` 目录：Vue 前端、Canvas 渲染器、WebSocket 客户端 |

## 后果

**正面**：
- UI 开发效率提升——Vue 模板比 QSS 声明式，热重载即时看到效果
- 摄像头和渲染不再服务端负担
- 未来可逐步支持移动端（Vue 组件天然响应式）

**负面**：
- `renderer.py` 566 行的透视贴合逻辑需要 JS 重写
- 数位板压感精度下降——`PointerEvent.pressure` 不如 `QTabletEvent` 精确
- 首次加载需要下载 MediaPipe WASM（~5-8MB）
- 仅为单用户/本地使用设计，不支持远程部署
