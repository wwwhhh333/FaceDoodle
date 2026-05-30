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
中央编排器，14 步帧循环的主控。通过 Mixin 组合获得 StickerManager 和 AnimationProcessor 的能力。

### StickerManager
Mixin，封装 StickerInstance 的增删改查和模板/分组加载。

### AnimationProcessor
Mixin，封装关键帧动画播放、导出和 AI 纹理动画生成。