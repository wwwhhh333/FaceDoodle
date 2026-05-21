import glob
import logging
import os
import uuid
import cv2
import numpy as np
import threading
import time
import queue
from multiprocessing import Event
from app.ai.agent import FaceDoodleAgent
from app.ai.generator import ComfyClient
from app.core.face_mesh import FaceDetector
from app.core.renderer import (render_scene, render_loading_progress, render_face_mesh,
                                apply_head_pose_skew, _build_location_quad)
from app.core.face_draw import FaceDrawCanvas
from app.core.animation import AnimationEngine
from app.core.animation import TextureAnimator, extract_sprite_frame
from app.core.protocol import (
    CmdImg2Img,
    AdjMove, AdjRotate, AdjScale, AdjReset,
    GalAddSticker, GalRemoveSticker, GalSelectEditTarget,
    GalLoadTemplate, GalLoadSticker, GalMergeGroup,
    DrawToggleDrawMode, DrawSetRegion, DrawSetBrush, DrawToggleEraser,
    DrawSetBrushType, DrawSetPressureMode, DrawSetSpacing, DrawSetScatter,
    DrawUndo, DrawClear, DrawStrokeBegin, DrawStrokePoint, DrawStrokeEnd, DrawSave,
    DispStickerSaved, DispGenerationFailed, DispActiveStickersChanged,
    DispGenProgress, DispAgentMessage, DispAgentQuestion,
    AnimPlaybackState,
    Result,
)
from app.utils.image_proc import load_rgba_sticker
from app.utils import storage
from app.core.tracker_stickers import StickerManager
from app.core.tracker_animation import AnimationProcessor
from app.core.sticker_registry import StickerRegistry
from app.core.protocol import Adjustment, StickerInstance

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Thread-safe generation state
# ══════════════════════════════════════════════════════════════════════════════

class GenerationState:
    """Tracks AI generation lifecycle — all access is locked for cross-thread visibility."""

    def __init__(self):
        self._lock = threading.Lock()
        self._is_generating = False
        self._start_time = 0.0

    @property
    def is_generating(self):
        with self._lock:
            return self._is_generating

    def get_elapsed(self):
        """Atomically read (is_generating, elapsed_seconds). Returns 0 when idle."""
        with self._lock:
            if not self._is_generating:
                return 0.0
            return time.time() - self._start_time

    def start(self):
        with self._lock:
            self._is_generating = True
            self._start_time = time.time()

    def finish(self):
        with self._lock:
            self._is_generating = False


# ══════════════════════════════════════════════════════════════════════════════
# AI worker functions (run in daemon threads)
# ══════════════════════════════════════════════════════════════════════════════

def _handle_img2img(cmd, result_queue, mock, gen_state):
    try:
        prompt_text = cmd.prompt_text
        image_path = cmd.image_path
        target_location = cmd.target_location
        scale = cmd.scale
        controlnet_strength = cmd.controlnet_strength
        denoise = cmd.denoise

        if image_path and os.path.exists(image_path):
            raw = np.fromfile(image_path, dtype=np.uint8)
            raw = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if raw is not None and raw.shape[2] == 4:
                canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
                h, w = raw.shape[:2]
                sf = min(900 / max(w, h), 1.0)
                new_w, new_h = int(w * sf), int(h * sf)
                resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
                y_off = (1024 - new_h) // 2
                x_off = (1024 - new_w) // 2
                alpha = resized[:, :, 3:4] / 255.0
                rgb = resized[:, :, :3]
                roi = canvas[y_off:y_off + new_h, x_off:x_off + new_w]
                blended = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = blended
                preprocessed_path = os.path.join(os.path.dirname(image_path), "preprocessed_for_img2img.png")
                cv2.imwrite(preprocessed_path, canvas)
                image_path = preprocessed_path

        log.debug("img2img 模式: prompt='%s', image='%s'", prompt_text, image_path)

        if mock:
            mock_path = "assets/ui/loading.png"
            temp_files = glob.glob("assets/temp/*.png")
            if temp_files:
                temp_files.sort(key=os.path.getmtime, reverse=True)
                mock_path = temp_files[0]
            new_sticker = load_rgba_sticker(mock_path)
        else:
            comfy_client = ComfyClient()
            generated_path = comfy_client.generate_sync(
                prompt_text=prompt_text,
                workflow_name="img2img_controlnet_workflow_api.json",
                input_image_path=image_path,
                denoise=denoise,
                controlnet_strength=controlnet_strength,
            )
            if generated_path and os.path.exists(generated_path):
                new_sticker = load_rgba_sticker(generated_path)
            else:
                new_sticker = None
                log.error("img2img 错误：ComfyUI 未生成有效文件")

        if new_sticker is not None:
            result_data = {
                "sticker": new_sticker,
                "location": target_location,
                "scale": scale,
                "prompt": cmd.display_name or prompt_text,
                "positive_prompt": prompt_text,
                "timestamp": time.time()
            }
            result_queue.put(result_data)
            log.info("img2img 任务完成, 位置: %s", target_location)
        else:
            result_queue.put({"error": "ComfyUI 未能生成有效图片，请检查服务是否正常运行"})

    except Exception as e:
        log.error("img2img 异常: %s", e)
        result_queue.put({"error": str(e)})
    finally:
        gen_state.finish()


def ai_worker_thread(user_command, result_queue, api_key, mock, gen_state,
                     conversation_history=None, active_stickers=None):
    if isinstance(user_command, CmdImg2Img):
        _handle_img2img(user_command, result_queue, mock, gen_state)
        return

    agent = FaceDoodleAgent(api_key=api_key)

    try:
        log.info("正在处理: %s", user_command)
        chat_result = agent.chat(
            str(user_command),
            conversation_history=conversation_history,
            active_stickers=active_stickers,
        )
        action = chat_result.get("action")
        assistant_message = chat_result.get("message", "")

        if action == "generate":
            tasks = chat_result.get("tasks", [])
            workflow = chat_result.get("workflow", "transparent_workflow_api.json")
            log.info("生成任务: %d 个贴纸", len(tasks))

            group_id = None
            group_name = None
            if len(tasks) > 1:
                group_id = str(uuid.uuid4())
                raw_name = str(user_command)
                group_name = raw_name if len(raw_name) <= 50 else raw_name[:47] + "..."

            for i, task in enumerate(tasks):
                result_queue.put({
                    "type": Result.GENERATION_PROGRESS,
                    "current": i + 1,
                    "total": len(tasks),
                    "message": f"正在生成 ({i + 1}/{len(tasks)})...",
                })

                if mock:
                    mock_path = "assets/ui/loading.png"
                    temp_files = glob.glob("assets/temp/*.png")
                    if temp_files:
                        temp_files.sort(key=os.path.getmtime, reverse=True)
                        mock_path = temp_files[0]
                    log.debug("Mock 模式: 使用 %s", mock_path)
                    new_sticker = load_rgba_sticker(mock_path)
                else:
                    comfy_client = ComfyClient()
                    log.debug("生成提示词: %s", task['prompt'])
                    image_path = comfy_client.generate_sync(
                        prompt_text=task["prompt"],
                        workflow_name=workflow,
                    )
                    log.info("生成完成，返回路径: %s", image_path or "None")
                    if image_path and os.path.exists(image_path):
                        new_sticker = load_rgba_sticker(image_path)
                        log.info("贴纸加载: %s", "成功" if new_sticker is not None else "失败")
                    else:
                        new_sticker = None
                        log.error("错误：ComfyUI 未生成有效文件")

                if new_sticker is not None:
                    result_data = {
                        "type": Result.GENERATION_RESULT,
                        "sticker": new_sticker,
                        "location": task["region"],
                        "scale": task["scale"],
                        "prompt": str(user_command),
                        "positive_prompt": task["prompt"],
                        "timestamp": time.time(),
                    }
                    if group_id:
                        result_data["group_id"] = group_id
                        result_data["group_name"] = group_name
                    result_queue.put(result_data)
                else:
                    result_queue.put({
                        "type": Result.GENERATION_RESULT,
                        "error": f"任务 {i + 1}/{len(tasks)} 生成失败",
                    })

            # Signal completion
            result_queue.put({
                "type": Result.GENERATION_DONE,
                "assistant_message": assistant_message,
                "group_id": group_id,
                "group_name": group_name,
            })

        elif action == "ask":
            result_queue.put({
                "type": Result.AGENT_QUESTION,
                "message": chat_result.get("message", ""),
                "assistant_message": assistant_message,
            })

        else:
            log.warning("未知 action: %s", action)
            result_queue.put({"type": Result.ERROR, "error": f"未知操作: {action}"})

    except Exception as e:
        log.error("AI Worker 异常: %s", e)
        result_queue.put({"type": Result.ERROR, "error": str(e)})

    finally:
        gen_state.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Producer process
# ══════════════════════════════════════════════════════════════════════════════

def producer(frame_queue, stop_event, video_path=None):
    from app.utils.logging_config import setup_logging
    setup_logging()
    from app.utils.config_loader import get_config
    cfg = get_config()

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error("无法打开视频文件 %s", video_path)
            stop_event.set()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = max(fps, 1.0) if fps > 0 else 30.0
        frame_delay = 1.0 / fps
        loop = cfg["video"]["loop"]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("视频模式: %s (%.1f FPS, %d 帧, %s)",
                 os.path.basename(video_path), fps, total_frames, '循环' if loop else '单次')
    else:
        cam_cfg = cfg.get("camera", {})
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.error("无法打开摄像头 (index 0)")
            stop_event.set()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
        frame_delay = 0
        loop = False

    try:
        while not stop_event.is_set() and cap.isOpened():
            _frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if video_path and loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if not video_path:
                frame = cv2.flip(frame, 1)
            try:
                frame_queue.put(frame, timeout=0.1)
            except Exception:
                log.debug("frame_queue 已满，丢弃一帧")
                pass
            if frame_delay > 0:
                elapsed = time.perf_counter() - _frame_start
                remaining = frame_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()
        log.info("Producer 已退出")


# ══════════════════════════════════════════════════════════════════════════════
# Consumer processor
# ══════════════════════════════════════════════════════════════════════════════

class ConsumerProcessor(StickerManager, AnimationProcessor):
    """Central orchestrator for face detection, sticker rendering, AI generation."""

    MAX_STICKERS = 20

    def __init__(self, in_queue, display_queue, command_queue, adjustment_queue,
                 gallery_queue, draw_queue, animation_queue, api_key, stop_event, mock=False):
        self.in_queue = in_queue
        self.display_queue = display_queue
        self.command_queue = command_queue
        self.adjustment_queue = adjustment_queue
        self.gallery_queue = gallery_queue
        self.draw_queue = draw_queue
        self.animation_queue = animation_queue
        self.stop_event = stop_event
        self.api_key = api_key
        self.mock = mock

        self.detector = FaceDetector()
        self.result_queue = queue.Queue()
        self.gen_state = GenerationState()
        self.face_canvas = FaceDrawCanvas()

        self.registry = StickerRegistry(max_stickers=self.MAX_STICKERS)
        self.edit_target_id = None
        self._sticker_adjustments = {}  # sticker_id -> default adjustment (for restore)

        self.cached_face_data = None
        self.pending_command = None
        self.active_content = None      # backward compat
        self.face_draw_active = False
        self.face_draw_region = "full_face"

        self.anim_engine = AnimationEngine()
        self._anim_evaluations = {}     # instance_id -> dict (last evaluate result)
        self._adj_is_delta = set()      # instance_ids currently in delta mode
        self._pending_export = None     # (instance_id, format, fps, output_path)

        self.texture_animator = TextureAnimator()
        self._pending_texture_gen = None   # AnimGenTexture message
        self._texture_gen_running = False

        # Register cross-domain cleanup: when a sticker is removed, animation
        # processor must also clean up its state.  This replaces the old pattern
        # of StickerManager directly calling self.texture_animator.unregister()
        # and self._anim_evaluations.pop().
        self.registry.on("removed", self.texture_animator.unregister)
        def _cleanup_anim(iid):
            self._anim_evaluations.pop(iid, None)
            self._adj_is_delta.discard(iid)
        self.registry.on("removed", _cleanup_anim)

        self.conversation_history = []   # list[dict], multi-turn dialog state
        self.max_conversation_turns = 10  # keep last 10 messages (5 turns)

        self._last_synced_state = None   # fingerprint to skip no-op _sync_state_to_ui pushes
        self._last_face_fp = None        # (fc_x, fc_y, fw) for change detection
        self._pending_group = None       # dict: {group_id, group_name, member_ids: []}
        self._had_stickers = False       # track whether stickers were ever present (for auto-clear)

    # ── public API ──

    def run(self):
        self._init_mock()
        try:
            while not self.stop_event.is_set():
                frame = self._get_frame()
                if frame is None:
                    self._process_command_queue()
                    self._process_result_queue()
                    time.sleep(0.01)
                    continue

                self._process_command_queue()
                self._process_result_queue()

                face_data = self._detect_face(frame)
                face_w = self._get_face_width(face_data)

                self._process_adjustment_queue(face_w)
                self._process_gallery_queue(face_data)
                self._process_draw_queue(face_data)
                self._process_animation_queue()
                self._evaluate_animations()
                self._process_export()
                self._process_texture_generation()

                frame = self._render_frame(frame, face_data)
                self._sync_state_to_ui(face_data)
                self._push_frame(frame)
        finally:
            self.detector.close()
            log.info("Consumer 已退出")

    # ── initialization ──

    def _init_mock(self):
        if not self.mock:
            return
        temp_files = glob.glob("assets/temp/*.png")
        if temp_files:
            temp_files.sort(key=os.path.getmtime, reverse=True)
            mock_sticker = load_rgba_sticker(temp_files[0])
            if mock_sticker is not None:
                instance_id = str(uuid.uuid4())
                inst = StickerInstance(
                    instance_id=instance_id,
                    sticker_id="", sticker=mock_sticker,
                    location="forehead", scale=1.0, prompt="Mock",
                )
                self.registry.add(inst)
                log.info("Mock 模式: 已加载 %s", temp_files[0])

    # ── frame i/o ──

    def _get_frame(self):
        try:
            return self.in_queue.get(block=True, timeout=0.1)
        except Exception:
            return None

    def _push_frame(self, frame):
        try:
            self.display_queue.put(frame, timeout=0.05)
        except Exception:
            pass

    # ── face detection ──

    def _detect_face(self, frame):
        face_data = self.detector.get_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face_data and "nose_tip" in face_data:
            self.cached_face_data = face_data
        return face_data

    @staticmethod
    def _get_face_width(face_data):
        if face_data is None:
            return 200.0
        return max(float(face_data.get("face_width", 200.0)), 1.0)

    # ── command queue ──

    def _process_command_queue(self):
        if not self.command_queue.empty():
            self.pending_command = self.command_queue.get()

        if self.pending_command and not self.gen_state.is_generating:
            cmd = self.pending_command
            self.pending_command = None

            if isinstance(cmd, str):
                self._append_conversation("user", cmd)

            self._pending_group = None  # Reset any leaked state from interrupted generation
            self.gen_state.start()
            threading.Thread(
                target=ai_worker_thread,
                args=(cmd, self.result_queue, self.api_key, self.mock, self.gen_state,
                      list(self.conversation_history),
                      [{"prompt": s.prompt, "location": s.location}
                       for s in self.registry.all]),
                daemon=True
            ).start()

    # ── result queue (AI generation output) ──

    def _process_result_queue(self):
        try:
            new_content = self.result_queue.get(block=False)
        except queue.Empty:
            return

        msg_type = new_content.get("type", "")

        # ── old-format error (backward compat) ──
        if msg_type == "" and "error" in new_content:
            self.display_queue.put(DispGenerationFailed(error=new_content["error"]))
            return

        # ── new-format error ──
        if msg_type == Result.ERROR:
            self.display_queue.put(DispGenerationFailed(error=new_content.get("error", "未知错误")))
            return

        # ── generation progress ──
        if msg_type == Result.GENERATION_PROGRESS:
            self.display_queue.put(DispGenProgress(
                current=new_content.get("current", 0),
                total=new_content.get("total", 0),
                step=new_content.get("step", 0),
                total_steps=new_content.get("total_steps", 0),
                message=new_content.get("message", ""),
            ))
            return

        # ── generation done ──
        if msg_type == Result.GENERATION_DONE:
            self._append_conversation("assistant", new_content.get("assistant_message", ""))
            if self._pending_group is not None and len(self._pending_group["member_ids"]) > 1:
                storage.save_group(
                    group_name=self._pending_group["group_name"],
                    member_ids=self._pending_group["member_ids"],
                    group_id=self._pending_group["group_id"],
                )
                log.info("贴纸组已保存: %s (%d 成员)", self._pending_group['group_name'], len(self._pending_group['member_ids']))
            self._pending_group = None
            self.display_queue.put(DispGenProgress(done=True, message=new_content.get("assistant_message", "生成完成")))
            log.debug("生成完成, 历史: %d 条", len(self.conversation_history))
            return

        # ── agent question ──
        if msg_type == Result.AGENT_QUESTION:
            self._append_conversation("assistant", new_content.get("assistant_message", ""))
            self.display_queue.put(DispAgentQuestion(
                text=new_content.get("message", ""),
            ))
            log.debug("Agent 反问, 历史: %d 条", len(self.conversation_history))
            return

        # ── generation result (single sticker) ──
        if msg_type == Result.GENERATION_RESULT:
            if "error" in new_content:
                log.error("贴纸生成失败: %s", new_content['error'])
                self.display_queue.put(DispGenerationFailed(error=new_content['error']))
                return
            try:
                sticker_id = self._save_generated_sticker(new_content)
                group_id = new_content.get("group_id")
                if group_id and sticker_id is not None:
                    self._accumulate_group_member(
                        group_id,
                        new_content.get("group_name", ""),
                        sticker_id,
                    )
            except Exception as e:
                log.error("保存贴纸失败: %s", e)
            return

        # ── old-format success (backward compat, no type field) ──
        if msg_type == "" and "sticker" in new_content:
            try:
                if self._save_generated_sticker(new_content) is not None:
                    self.display_queue.put(DispGenProgress(done=True, message="生成完成"))
            except Exception as e:
                log.error("保存贴纸失败: %s", e)
            return

    # ── sticker persistence helper ──

    def _save_generated_sticker(self, new_content):
        """Save a generated sticker to disk and add it as an active instance.

        Returns sticker_id on success, None if at capacity.
        """
        if self.registry.count >= self.MAX_STICKERS:
            log.warning("贴纸数量已达上限 (%d)，跳过添加", self.MAX_STICKERS)
            return None

        metadata = {
            "prompt": new_content.get("prompt", ""),
            "location": new_content.get("location", "forehead"),
            "scale": new_content.get("scale", 1.0),
        }
        group_id = new_content.get("group_id")
        if group_id:
            metadata["group_id"] = group_id

        sticker_id = storage.save_sticker(
            new_content['sticker'], metadata,
        )
        instance_id = self._add_sticker_instance(
            new_content['sticker'], sticker_id,
            new_content.get("location", "forehead"),
            new_content.get("scale", 1.0),
            new_content.get("prompt", ""),
        )
        self.edit_target_id = instance_id
        self.display_queue.put(DispStickerSaved(sticker_id=sticker_id))
        self.active_content = new_content  # backward compat
        log.info("贴纸已保存: %s", sticker_id)
        return sticker_id

    def _accumulate_group_member(self, group_id, group_name, sticker_id):
        if self._pending_group is None:
            self._pending_group = {
                "group_id": group_id,
                "group_name": group_name,
                "member_ids": [],
            }
        if sticker_id:
            self._pending_group["member_ids"].append(sticker_id)

    # ── conversation helpers ──

    def _append_conversation(self, role, content):
        if not content:
            return
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_conversation_turns:
            self.conversation_history = self.conversation_history[-self.max_conversation_turns:]
        # Auto-clear only when stickers that were once present are all removed
        if self.registry.count == 0 and self._had_stickers:
            self.conversation_history.clear()
            self._had_stickers = False

    # ── adjustment queue ──

    def _process_adjustment_queue(self, face_w):
        dirty = False
        while not self.adjustment_queue.empty():
            try:
                msg = self.adjustment_queue.get(block=False)
            except Exception:
                break

            if self.edit_target_id is None:
                continue

            if not self.registry.has(self.edit_target_id):
                continue

            adj = self.registry.get_adj(self.edit_target_id)

            if isinstance(msg, AdjMove):
                adj.offset_x += msg.dx / face_w
                adj.offset_y += msg.dy / face_w
                dirty = True
            elif isinstance(msg, AdjRotate):
                adj.rotation += msg.d_angle
                dirty = True
            elif isinstance(msg, AdjScale):
                adj.scale_mult *= msg.multiplier
                adj.scale_mult = max(0.05, adj.scale_mult)
                dirty = True
            elif isinstance(msg, AdjReset):
                adj.offset_x = 0.0
                adj.offset_y = 0.0
                adj.rotation = 0.0
                adj.scale_mult = 1.0
                dirty = True

        if dirty:
            instance = self.registry.get(self.edit_target_id)
            if instance:
                storage.save_sticker_adjustments(instance.sticker_id, {
                    "offset_x": adj.offset_x, "offset_y": adj.offset_y,
                    "rotation": adj.rotation, "scale_mult": adj.scale_mult,
                })

    # ── gallery queue ──

    def _process_gallery_queue(self, face_data):
        while not self.gallery_queue.empty():
            try:
                gmsg = self.gallery_queue.get(block=False)
            except Exception:
                break

            if isinstance(gmsg, GalAddSticker):
                self._handle_add_sticker(gmsg)
            elif isinstance(gmsg, GalRemoveSticker):
                self._handle_remove_sticker(gmsg)
            elif isinstance(gmsg, GalSelectEditTarget):
                self._handle_select_edit_target(gmsg)
            elif isinstance(gmsg, GalLoadTemplate):
                self._handle_load_template(gmsg)
            elif isinstance(gmsg, GalLoadSticker):
                self._handle_load_sticker(gmsg)
            elif isinstance(gmsg, GalMergeGroup):
                self._handle_merge_group(gmsg, face_data)

    # ── draw queue ──

    def _process_draw_queue(self, face_data):
        while not self.draw_queue.empty():
            try:
                msg = self.draw_queue.get(block=False)
            except Exception:
                break

            if isinstance(msg, DrawToggleDrawMode):
                self.face_draw_active = not self.face_draw_active
                if self.face_draw_active:
                    self.edit_target_id = None

            elif isinstance(msg, DrawSetRegion):
                self.face_draw_region = msg.region

            elif isinstance(msg, DrawSetBrush):
                self.face_canvas.set_brush_size(msg.brush_size)
                self.face_canvas.set_brush_color(msg.brush_color)

            elif isinstance(msg, DrawToggleEraser):
                self.face_canvas.set_eraser_mode(msg.eraser_mode)

            elif isinstance(msg, DrawSetBrushType):
                self.face_canvas.set_brush_type(msg.brush_id)

            elif isinstance(msg, DrawSetPressureMode):
                self.face_canvas.set_pressure_mode(msg.mode)

            elif isinstance(msg, DrawSetSpacing):
                self.face_canvas.set_spacing(msg.coef)

            elif isinstance(msg, DrawSetScatter):
                self.face_canvas.set_scatter(msg.px)

            elif isinstance(msg, DrawUndo):
                self.face_canvas.undo()

            elif isinstance(msg, DrawClear):
                self.face_canvas.clear()

            elif isinstance(msg, DrawStrokeBegin) and self.face_draw_active and face_data:
                quad = _build_location_quad(face_data, self.face_draw_region, (512, 512, 4), scale=1.5)
                if quad is not None:
                    quad = apply_head_pose_skew(quad, face_data)
                    self.face_canvas.begin_stroke(quad)

            elif isinstance(msg, DrawStrokePoint) and self.face_draw_active:
                if msg.point is not None:
                    self.face_canvas.set_pressure(msg.pressure)
                    self.face_canvas.add_stroke_point(msg.point)

            elif isinstance(msg, DrawStrokeEnd):
                self.face_canvas.end_stroke()

            elif isinstance(msg, DrawSave):
                result = self.face_canvas.get_result()
                if result is not None:
                    sid = storage.save_sticker(result, {
                        "prompt": "全脸手绘",
                        "location": self.face_draw_region,
                        "scale": 1.0,
                    })
                    self.display_queue.put(DispStickerSaved(sticker_id=sid))

    # ── rendering ──

    def _render_frame(self, frame, face_data):
        if not face_data:
            return frame

        if self.gen_state.is_generating:
            frame = render_loading_progress(frame, face_data, self.gen_state)

        if self.gen_state.is_generating or self.edit_target_id is not None or self.face_draw_active:
            frame = render_face_mesh(frame, face_data)

        for instance in self.registry.all:
            if instance.sticker is None:
                continue
            iid = instance.instance_id
            manual = self.registry.get_adj(iid)
            adj = {
                "offset_x": manual.offset_x,
                "offset_y": manual.offset_y,
                "rotation": manual.rotation,
                "scale_mult": manual.scale_mult,
            }
            if self.anim_engine.is_playing(iid):
                anim = self._anim_evaluations.get(iid)
                if anim:
                    adj["offset_x"] = anim["offset_x"] + manual.offset_x
                    adj["offset_y"] = anim["offset_y"] + manual.offset_y
                    adj["rotation"] = anim["rotation"] + manual.rotation
                    adj["scale_mult"] = anim["scale_mult"] * manual.scale_mult
                    adj["opacity"] = anim.get("opacity", 1.0)
            adj["edit_mode"] = (iid == self.edit_target_id)
            if instance.is_animated:
                frame_idx, cols, rows = self.texture_animator.get_frame_params(iid)
                sticker = extract_sprite_frame(instance.sticker, frame_idx, cols, rows)
            else:
                sticker = instance.sticker
            content = {
                "sticker": sticker,
                "location": instance.location,
                "scale": instance.scale,
            }
            frame = render_scene(frame, face_data, content, adj)

        if self.face_draw_active and self.face_canvas.has_content:
            draw_content = {
                "sticker": self.face_canvas.canvas,
                "location": self.face_draw_region,
                "scale": 1.5,
            }
            frame = render_scene(frame, face_data, draw_content, None)

        return frame

    # ── UI state sync ──

    def _face_data_changed(self, fc_x, fc_y, fw, threshold=5.0):
        if self._last_face_fp is None:
            return True
        return (abs(fc_x - self._last_face_fp[0]) > threshold or
                abs(fc_y - self._last_face_fp[1]) > threshold or
                abs(fw - self._last_face_fp[2]) > threshold)

    def _sync_state_to_ui(self, face_data=None):
        instances_info = []
        for s in self.registry.all:
            adj = self.registry.get_adj(s.instance_id)
            info = {
                "instance_id": s.instance_id,
                "sticker_id": s.sticker_id,
                "region": s.location,
                "offset_x": adj.offset_x,
                "offset_y": adj.offset_y,
            }
            instances_info.append(info)

        fc_x, fc_y, fw = 0.0, 0.0, 0.0
        if face_data and face_data.get("landmark_rects"):
            rects = face_data["landmark_rects"]
            fh = rects.get("forehead_full")
            if fh and len(fh) >= 4:
                fc_x, fc_y = fh[0] + fh[2] / 2.0, fh[1] + fh[3] / 2.0
                fw = max(fh[2], 1.0)

        fingerprint = (
            self.registry.count,
            tuple((s.instance_id, s.sticker_id, s.location,
                   round(self.registry.get_adj(s.instance_id).offset_x, 3),
                   round(self.registry.get_adj(s.instance_id).offset_y, 3))
                  for s in self.registry.all),
            self.edit_target_id,
        )
        if fingerprint == self._last_synced_state and not self._face_data_changed(fc_x, fc_y, fw):
            return
        self._last_synced_state = fingerprint
        self._last_face_fp = (round(fc_x, 1), round(fc_y, 1), round(fw, 1))

        self.display_queue.put(DispActiveStickersChanged(
            active_count=self.registry.count,
            instances=instances_info,
            edit_target_id=self.edit_target_id,
            face_center_x=fc_x,
            face_center_y=fc_y,
            face_width=fw,
        ))

        if self.edit_target_id:
            info = self.anim_engine.get_playback_info(self.edit_target_id)
            if info:
                self.display_queue.put(AnimPlaybackState(
                    instance_id=info["instance_id"],
                    playing=info["playing"],
                    time=info["time"],
                    duration=info["duration"],
                ))


# ══════════════════════════════════════════════════════════════════════════════
# Consumer entry point (thin wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def consumer(in_queue, display_queue, command_queue, adjustment_queue,
             gallery_queue, draw_queue, animation_queue, api_key, stop_event, mock=False):
    from app.utils.logging_config import setup_logging
    setup_logging()
    processor = ConsumerProcessor(
        in_queue, display_queue, command_queue, adjustment_queue,
        gallery_queue, draw_queue, animation_queue, api_key, stop_event, mock,
    )
    processor.run()
