import glob
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

        if image_path and os.path.exists(image_path):
            raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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
                preprocessed_path = os.path.join(os.path.dirname(image_path), "preprocessed_for_controlnet.png")
                cv2.imwrite(preprocessed_path, canvas)
                image_path = preprocessed_path

        print(f"[AI Worker] img2img 模式: prompt='{prompt_text}', image='{image_path}'")

        if mock:
            mock_path = "assets/static/loading.png"
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
            )
            if generated_path and os.path.exists(generated_path):
                new_sticker = load_rgba_sticker(generated_path)
            else:
                new_sticker = None
                print("[AI Worker] img2img 错误：ComfyUI 未生成有效文件")

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
            print(f"[AI Worker] img2img 任务完成, 位置: {target_location}")
        else:
            result_queue.put({"error": "ComfyUI 未能生成有效图片，请检查服务是否正常运行"})

    except Exception as e:
        print(f"[AI Worker] img2img 异常: {str(e)}")
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
        print(f"[AI Worker] 正在处理: {user_command}")
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
            print(f"[AI Worker] 生成任务: {len(tasks)} 个贴纸")

            group_id = None
            group_name = None
            if len(tasks) > 1:
                group_id = str(uuid.uuid4())
                group_name = str(user_command)

            for i, task in enumerate(tasks):
                result_queue.put({
                    "type": Result.GENERATION_PROGRESS,
                    "current": i + 1,
                    "total": len(tasks),
                    "message": f"正在生成 ({i + 1}/{len(tasks)})...",
                })

                if mock:
                    mock_path = "assets/static/loading.png"
                    temp_files = glob.glob("assets/temp/*.png")
                    if temp_files:
                        temp_files.sort(key=os.path.getmtime, reverse=True)
                        mock_path = temp_files[0]
                    print(f"[AI Worker] Mock 模式: 使用 {mock_path}")
                    new_sticker = load_rgba_sticker(mock_path)
                else:
                    comfy_client = ComfyClient()
                    print(f"[AI Worker] ComfyUI 生成: {task['prompt'][:60]}...")
                    image_path = comfy_client.generate_sync(
                        prompt_text=task["prompt"],
                        workflow_name=workflow,
                    )
                    if image_path and os.path.exists(image_path):
                        new_sticker = load_rgba_sticker(image_path)
                    else:
                        new_sticker = None
                        print("[AI Worker] 错误：ComfyUI 未生成有效文件")

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

        elif action == "adjust":
            result_queue.put({
                "type": Result.ADJUSTMENT_RESULT,
                "adjustments": chat_result.get("adjustments", []),
                "target_instance": chat_result.get("target_instance"),
                "remove": chat_result.get("remove", False),
                "assistant_message": assistant_message,
            })

        elif action == "ask":
            result_queue.put({
                "type": Result.AGENT_QUESTION,
                "message": chat_result.get("message", ""),
                "assistant_message": assistant_message,
            })

        else:
            print(f"[AI Worker] 未知 action: {action}")
            result_queue.put({"type": Result.ERROR, "error": f"未知操作: {action}"})

    except Exception as e:
        print(f"[AI Worker] 异常: {e}")
        result_queue.put({"type": Result.ERROR, "error": str(e)})

    finally:
        gen_state.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Producer process
# ══════════════════════════════════════════════════════════════════════════════

def producer(frame_queue, stop_event, video_path=None):
    from app.utils.config_loader import get_config
    cfg = get_config()

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Producer] 错误：无法打开视频文件 {video_path}")
            stop_event.set()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = max(fps, 1.0) if fps > 0 else 30.0
        frame_delay = 1.0 / fps
        loop = cfg["video"]["loop"]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Producer] 视频模式: {os.path.basename(video_path)} ({fps:.1f} FPS, "
              f"{total_frames} 帧, {'循环' if loop else '单次'})")
    else:
        cam_cfg = cfg.get("camera", {})
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Producer] 错误：无法打开摄像头 (index 0)")
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
                print("[Producer] 警告：frame_queue 已满或消费者已退出，丢弃一帧")
                pass
            if frame_delay > 0:
                elapsed = time.perf_counter() - _frame_start
                remaining = frame_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()
        print("[Producer] 已退出")


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

        self.active_stickers = []       # list[dict], z-order: index 0 = bottom
        self.edit_target_id = None
        self.adjustments = {}           # instance_id -> {offset_x, offset_y, rotation, scale_mult}
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

        self.conversation_history = []   # list[dict], multi-turn dialog state
        self.max_conversation_turns = 6  # keep last 6 messages (3 turns)

        self._last_synced_state = None   # fingerprint to skip no-op _sync_state_to_ui pushes
        self._pending_group = None       # dict: {group_id, group_name, member_ids: []}
        self._had_stickers = False       # track whether stickers were ever present (for auto-clear)

    # ── public API ──

    def run(self):
        self._init_mock()
        try:
            while not self.stop_event.is_set():
                frame = self._get_frame()
                if frame is None:
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
                self._sync_state_to_ui()
                self._push_frame(frame)
        finally:
            self.detector.close()
            print("[Consumer] 已退出")

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
                self.active_stickers.append({
                    "instance_id": instance_id, "sticker_id": None,
                    "sticker": mock_sticker, "location": "forehead",
                    "scale": 1.0, "prompt": "Mock",
                })
                self.adjustments[instance_id] = {
                    "offset_x": 0.0, "offset_y": 0.0,
                    "rotation": 0.0, "scale_mult": 1.0,
                }
                print(f"[Consumer] Mock 模式: 已加载 {temp_files[0]}")

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

            self.gen_state.start()
            threading.Thread(
                target=ai_worker_thread,
                args=(cmd, self.result_queue, self.api_key, self.mock, self.gen_state,
                      list(self.conversation_history),
                      list(self.active_stickers)),
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
                print(f"[Consumer] 贴纸组已保存: {self._pending_group['group_name']} ({len(self._pending_group['member_ids'])} 成员)")
            self._pending_group = None
            self.display_queue.put(DispGenProgress(done=True, message="生成完成"))
            print(f"[Conversation] 生成完成, 历史: {len(self.conversation_history)} 条")
            return

        # ── agent question ──
        if msg_type == Result.AGENT_QUESTION:
            self._append_conversation("assistant", new_content.get("assistant_message", ""))
            self.display_queue.put(DispAgentQuestion(
                text=new_content.get("message", ""),
            ))
            print(f"[Conversation] Agent 反问, 历史: {len(self.conversation_history)} 条")
            return

        # ── adjustment result ──
        if msg_type == Result.ADJUSTMENT_RESULT:
            assistant_message = new_content.get("assistant_message", "")
            self._append_conversation("assistant", assistant_message)

            if new_content.get("remove"):
                target = new_content.get("target_instance")
                if target:
                    iid = target.get("instance_id")
                    self.active_stickers = [s for s in self.active_stickers if s["instance_id"] != iid]
                    self.adjustments.pop(iid, None)
                    self._anim_evaluations.pop(iid, None)
                    self._adj_is_delta.discard(iid)
                    self.texture_animator.unregister(iid)
                    if self.edit_target_id == iid:
                        self.edit_target_id = self.active_stickers[-1]["instance_id"] if self.active_stickers else None
                    print(f"[Consumer] 已移除贴纸: {iid}")
            else:
                target = new_content.get("target_instance")
                if target:
                    iid = target.get("instance_id")
                    adj = self.adjustments.get(iid, {
                        "offset_x": 0.0, "offset_y": 0.0,
                        "rotation": 0.0, "scale_mult": 1.0,
                    })
                    for a in new_content.get("adjustments", []):
                        t = a["type"]
                        v = a["value"]
                        if t in ("offset_x", "offset_y"):
                            adj[t] = adj.get(t, 0.0) + v
                        elif t == "rotation":
                            adj[t] = adj.get(t, 0.0) + v
                        elif t == "scale_mult":
                            adj[t] = adj.get(t, 1.0) * v
                    self.adjustments[iid] = adj
                    self._adj_is_delta.discard(iid)
                    print(f"[Consumer] 已调整贴纸: {iid}, {new_content.get('adjustments', [])}")

            self.display_queue.put(DispAgentMessage(text=assistant_message))
            print(f"[Conversation] 调整完成, 历史: {len(self.conversation_history)} 条")
            return

        # ── generation result (single sticker) ──
        if msg_type == Result.GENERATION_RESULT:
            if "error" in new_content:
                print(f"[Consumer] 贴纸生成失败: {new_content['error']}")
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
                print(f"[Consumer] 保存贴纸失败: {e}")
            return

        # ── old-format success (backward compat, no type field) ──
        if msg_type == "" and "sticker" in new_content:
            try:
                if self._save_generated_sticker(new_content) is not None:
                    self.display_queue.put(DispGenProgress(done=True, message="生成完成"))
            except Exception as e:
                print(f"[Consumer] 保存贴纸失败: {e}")
            return

    # ── sticker persistence helper ──

    def _save_generated_sticker(self, new_content):
        """Save a generated sticker to disk and add it as an active instance.

        Returns sticker_id on success, None if at capacity.
        """
        if len(self.active_stickers) >= self.MAX_STICKERS:
            print(f"[Consumer] 贴纸数量已达上限 ({self.MAX_STICKERS})，跳过添加")
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
        saved_adj = (self._sticker_adjustments.get(sticker_id)
                     or storage.get_sticker_adjustments(sticker_id))
        adj = dict(saved_adj) if saved_adj else {
            "offset_x": 0.0, "offset_y": 0.0,
            "rotation": 0.0, "scale_mult": 1.0,
        }
        instance_id = self._add_sticker_instance(
            new_content['sticker'], sticker_id,
            new_content.get("location", "forehead"),
            new_content.get("scale", 1.0),
            new_content.get("prompt", ""),
        )
        self.adjustments[instance_id] = adj
        self.edit_target_id = instance_id
        self.display_queue.put(DispStickerSaved(sticker_id=sticker_id))
        self.active_content = new_content  # backward compat
        print(f"[Consumer] 贴纸已保存: {sticker_id}")
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
        if not self.active_stickers and self._had_stickers:
            self.conversation_history.clear()
            self._had_stickers = False

    # ── adjustment queue ──

    def _process_adjustment_queue(self, face_w):
        while not self.adjustment_queue.empty():
            try:
                msg = self.adjustment_queue.get(block=False)
            except Exception:
                break

            if self.edit_target_id is None:
                continue

            adj = self.adjustments.get(self.edit_target_id)
            if adj is None:
                continue

            if isinstance(msg, AdjMove):
                adj["offset_x"] += msg.dx / face_w
                adj["offset_y"] += msg.dy / face_w
            elif isinstance(msg, AdjRotate):
                adj["rotation"] += msg.d_angle
            elif isinstance(msg, AdjScale):
                adj["scale_mult"] *= msg.multiplier
                adj["scale_mult"] = max(0.05, adj["scale_mult"])
            elif isinstance(msg, AdjReset):
                adj["offset_x"] = 0.0
                adj["offset_y"] = 0.0
                adj["rotation"] = 0.0
                adj["scale_mult"] = 1.0

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

        for instance in self.active_stickers:
            if instance["sticker"] is None:
                continue
            iid = instance["instance_id"]
            manual = self.adjustments.get(iid, {
                "offset_x": 0.0, "offset_y": 0.0,
                "rotation": 0.0, "scale_mult": 1.0,
            })
            adj = dict(manual)
            if self.anim_engine.is_playing(iid):
                anim = self._anim_evaluations.get(iid)
                if anim:
                    adj["offset_x"] = anim["offset_x"] + manual["offset_x"]
                    adj["offset_y"] = anim["offset_y"] + manual["offset_y"]
                    adj["rotation"] = anim["rotation"] + manual["rotation"]
                    adj["scale_mult"] = anim["scale_mult"] * manual["scale_mult"]
            adj["edit_mode"] = (iid == self.edit_target_id)
            if instance.get("is_animated"):
                frame_idx, cols, rows = self.texture_animator.get_frame_params(iid)
                sticker = extract_sprite_frame(instance["sticker"], frame_idx, cols, rows)
            else:
                sticker = instance["sticker"]
            content = {
                "sticker": sticker,
                "location": instance["location"],
                "scale": instance["scale"],
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

    def _sync_state_to_ui(self):
        instances_info = []
        for s in self.active_stickers:
            instances_info.append({
                "instance_id": s["instance_id"],
                "sticker_id": s["sticker_id"],
                "region": s["location"],
            })

        fingerprint = (
            len(self.active_stickers),
            tuple((i["instance_id"], i["sticker_id"], i["region"]) for i in instances_info),
            self.edit_target_id,
        )
        if fingerprint == self._last_synced_state:
            return
        self._last_synced_state = fingerprint

        self.display_queue.put(DispActiveStickersChanged(
            active_count=len(self.active_stickers),
            instances=instances_info,
            edit_target_id=self.edit_target_id,
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
    processor = ConsumerProcessor(
        in_queue, display_queue, command_queue, adjustment_queue,
        gallery_queue, draw_queue, animation_queue, api_key, stop_event, mock,
    )
    processor.run()
