# FaceDoodle 项目术语表

## 领域对象

### Sticker
一张由 ComfyUI 生成的透明 PNG 图片，持久化存储在 `assets/gallery/` 中。每个 Sticker 有 UUID、prompt、region、scale 等元数据。是"素材"层面的概念——存储在磁盘上，可被多次实例化到面部。

### StickerInstance
一个 Sticker 在面部上的活跃实例。通过 `StickerRegistry.add()` 注册，通过 `StickerRegistry.remove()` 移除。每个 instance 有独立的编辑状态（位置、旋转、缩放）。允许多次实例化同一张 Sticker。

### Adjustment
用户对手动编辑状态的抽象——包括 offset_x/y、rotation、scale_mult。每个 StickerInstance 绑定一个 Adjustment。动画播放时 Adjustment 在 delta 和 absolute 模式之间切换。

### StickerRegistry
所有活跃 StickerInstance 及其 Adjustment 的中央仓库。提供 add/remove/get/iterate 操作。当实例被移除时通过回调通知其他域（如 AnimationProcessor）。

### Gesture
一组预设的面部区域（"额头"、"左脸颊"、"鼻子"等），AI Agent 返回的 region 值映射到 Gesture 上，Sticker 通过 Gesture 绑定到面部对应位置。

## 架构角色

### Producer
独立进程，仅负责摄像头/视频帧采集。与 Consumer 进程通过 `frame_queue` 通信。

### ConsumerProcessor
中央编排器，12 步帧循环的主控。通过 Mixin 组合获得 StickerManager 和 AnimationProcessor 的能力。

### StickerManager
Mixin，封装 StickerInstance 的增删改查和模板/分组加载。

### AnimationProcessor
Mixin，封装关键帧动画播放、导出和 AI 纹理动画生成。

## Web 架构迁移（2026-05-21 决议）

### 迁移目标
将 FaceDoodle 从 PySide6 桌面应用改造为 Web 应用（浏览器客户端 + 本地 Python 后端）。

### 已确定的架构决策

| 决策点 | 选择 | 理由 |
|-------|------|------|
| 部署方式 | 本地运行 | 保留外部编辑器、本地文件系统、ComfyUI 直连 |
| 面部检测 | MediaPipe JS（浏览器端） | 零延迟、帧不上传、服务器不负担检测 |
| 通信协议 | WebSocket（主通道）+ HTTP（文件上传） | 双向实时 + 大文件专用 |
| 前端框架 | Vue 3 + Canvas | 组件化 UI（设置面板、画廊、时间轴）+ Canvas 视频渲染 |
| 摄像头帧传输 | 640×480 WebP 压缩 | MediaPipe 精度够，带宽 ~5-8MB/s |
| 贴纸渲染 | 浏览器端 Canvas 2D API | 零帧延迟、服务器不碰渲染 |
