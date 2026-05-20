# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run

```bash
# Normal mode (requires ComfyUI running locally)
python app/main.py

# Mock mode (bypasses ComfyUI, uses cached images for UI testing)
python app/main.py --mock
```

API key priority: env var `DEEPSEEK_API_KEY` → `api_key.txt` (project root) → `config.json` `api_key` field. `api_key.txt` is gitignored and never written by the app (unlike config.json which is rewritten on close). ComfyUI must be running at the address in `config.json` (`comfyui.server_address`, default `127.0.0.1:8188`).

## Logging

Each process calls `setup_logging()` from `app/utils/logging_config.py` once at startup. Configures the **root logger** so `logging.getLogger(__name__)` works everywhere. Output:
- **Console**: INFO+ (DEBUG in `--verbose` mode)
- **File**: `logs/error.log` — WARNING+ with rotation (5 MB × 3 backups)
- Noisy third-party loggers (`matplotlib`, `PIL`, `urllib3`, `asyncio`) are suppressed to WARNING

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_agent.py -v   # single file
```

17 test files under `tests/` (401 test cases), pytest with fixtures in `conftest.py`. Tests cover agent, brush, config, templates, renderer, face_draw, storage, animation, protocol, and texture_anim. `test_agent.py` tests the keyword-fallback path (no API key needed).

**After any code change**, run the full test suite. If the modified code has no test coverage, add tests before declaring the task complete. Passing unit tests do not guarantee correctness — runtime issues (Chinese path encoding, silent error drops, queue message handling) only surface when the app actually runs.

## Validation Gates

Every change must pass these gates before being considered complete:

1. **Syntax**: `python scripts/check_syntax.py` (or `pre-commit run python-syntax-check`)
2. **Tests**: `python -m pytest tests/ -v` (or `pre-commit run pytest`)
3. **Runtime smoke test**: For non-trivial changes, launch the app (`python app/main.py --mock`) and exercise the affected workflow

If any gate fails, fix the issue and re-run all gates from step 1. After all gates pass, **stage and commit** the changes with a conventional commit message summarizing what changed and why. If a gate fails 3 times without resolution, stop and explain the blocker.

Pre-commit hooks run automatically on `git commit`. Run `pre-commit run --all-files` to check all files manually.

## Workflow

**Before modifying 3+ files or refactoring**, present a written plan covering scope, files affected, and rollback strategy. Wait for explicit user approval before writing any code. Work incrementally — one file at a time with test validation after each, getting confirmation before proceeding to the next.

**After completing a feature or significant change**, update `README.md` to reflect the new reality — add new features, remove stale claims, update the project structure tree, adjust architecture descriptions. The README is the user-facing summary and must stay in sync with what the code actually does.

## Architecture

Multi-process application: **Producer** (camera capture) → queue → **Consumer** (face detection + sticker rendering + AI orchestration) → queue → **Qt UI** (display + input).

Seven queues connect the UI process to the consumer process, plus an internal `result_queue` (thread-safe `queue.Queue`) for AI worker threads to post results back to the consumer main loop:

| Queue | Direction | Purpose |
|-------|-----------|---------|
| `frame_queue` | Producer → Consumer | Raw camera frames |
| `display_queue` | Consumer → UI | Rendered frames + typed status messages (`Disp*` dataclasses) |
| `command_queue` | UI → Consumer | AI generation prompts (text or `CmdImg2Img` dataclass) |
| `adjustment_queue` | UI → Consumer | Sticker edit actions (`AdjMove/Rotate/Scale/Reset`) |
| `gallery_queue` | UI → Consumer | Load sticker/template/merge (`Gal*` dataclasses) |
| `draw_queue` | UI → Consumer | Face drawing actions (`Draw*` dataclasses) |
| `animation_queue` | UI → Consumer | Animation playback + keyframe editing + texture gen + export |
| `result_queue` | AI threads → Consumer | Internal thread-safe queue for AI generation results |

All inter-process messages are **typed dataclasses** defined in `app/core/protocol.py`. Each queue family has its own action-constant class (`Adj`, `Gal`, `Draw`, `Disp`, `Anim`, `Result`) and corresponding dataclasses. The `display_queue` carries either numpy frame arrays (for video) or `DisplayStatusMsg` dataclass instances.

### ConsumerProcessor (`app/core/tracker.py`)

The central orchestrator uses **mixin-based composition** — `ConsumerProcessor(StickerManager, AnimationProcessor)`. It runs a single main loop per frame:

1. `_get_frame()` — dequeue camera frame
2. `_process_command_queue()` — spawn AI worker thread when idle
3. `_process_result_queue()` — dispatch AI results (8 branches by `Result.*` type)
4. `_detect_face()` — MediaPipe 468 landmarks
5. `_process_adjustment_queue()` — apply manual edit deltas
6. `_process_gallery_queue()` — add/remove/load stickers & templates
7. `_process_draw_queue()` — face drawing stroke pipeline
8. `_process_animation_queue()` — animation playback & keyframe edits
9. `_evaluate_animations()` — evaluate active animation clips
10. `_render_frame()` — composite all stickers + face mesh onto frame
11. `_sync_state_to_ui()` — push `DispActiveStickersChanged` (fingerprint-guarded against no-op pushes)
12. `_push_frame()` — enqueue rendered frame

**Mixin responsibilities:**
- `StickerManager` (`tracker_stickers.py`): `_add_sticker_instance`, `_handle_add_sticker`, `_handle_remove_sticker`, `_handle_load_template`, `_handle_load_sticker`, `_handle_merge_group`
- `AnimationProcessor` (`tracker_animation.py`): animation queue dispatch, clip evaluation, GIF/MP4 export, texture generation

### AI agent (`app/ai/agent.py`)

`FaceDoodleAgent.chat(user_message, conversation_history, active_stickers)` returns structured dicts with `action` field:

- **`generate`** — `{action, message, tasks: [{prompt, region, scale}, ...], workflow}` — multi-sticker generation
- **`ask`** — `{action, message}` — clarification question for the user

Uses DeepSeek API (`openai` client pointed at `api.deepseek.com`) with `response_format: json_object`. Falls back to **keyword matching** when no API key is configured or API calls fail: `KEYWORD_REGION_MAP` maps Chinese keywords to face regions. API calls retry twice with exponential backoff on transient failures. `parse_command()` is a backward-compat wrapper returning the old flat dict format.

Conversation state lives in `ConsumerProcessor.conversation_history` (list of `{role, content}` dicts, max 6 messages). Auto-clears only when stickers that were once present are all removed (tracked via `_had_stickers` flag). User messages are appended in `_process_command_queue`, assistant messages in `_process_result_queue`.

### AI generation flow

1. UI sends text → `command_queue`
2. `_process_command_queue` appends user message to history, spawns `ai_worker_thread` daemon
3. Worker calls `agent.chat()` with conversation history + active stickers summary
4. Worker posts typed dicts to `result_queue`:
   - `Result.GENERATION_PROGRESS` — per-sticker progress
   - `Result.GENERATION_RESULT` — each completed sticker (or error)
   - `Result.GENERATION_DONE` — all tasks complete
   - `Result.AGENT_QUESTION` — clarification needed
   - `Result.ERROR` — exception
5. `_process_result_queue` dispatches: saves stickers via `_save_generated_sticker()`, forwards messages to `display_queue` as `DispGenProgress`/`DispAgentMessage`/`DispAgentQuestion`/`DispGenerationFailed`

`GenerationState` is a thread-safe gating class — `_process_command_queue` skips new commands while `gen_state.is_generating` is True. Multi-sticker `generate` tasks run **sequentially** in the worker thread (each ComfyUI call blocks).

### ComfyUI client (`app/ai/generator.py`)

`ComfyClient` wraps the ComfyUI REST API. Submits workflow JSON via REST and polls `/api/queue` until completion.

Workflow JSON files live in `app/ai/workflows/`:
- `transparent_workflow_api.json` — SDXL + Layer Diffusion for transparent PNG stickers
- `img2img_controlnet_workflow_api.json` — img2img with ControlNet Scribble refinement
- `animatediff_workflow_api.json` — AnimateDiff for sprite-sheet texture animation

`generate_animated_frames()` returns a list of BGRA frames from AnimateDiff output. Temp file cleanup (`cleanup_temp_files`) keeps `assets/temp/` at ≤50 files.

### ComfyUI auto-management (`app/ai/comfy_manager.py`)

`ComfyUIManager` manages the ComfyUI subprocess lifecycle. On startup, if `comfyui.install_path` is configured, resolves how to launch (`.bat`/`.cmd` → `main.py`), starts the subprocess, and waits for socket readiness. On application exit, terminates the subprocess (10s timeout then kill). Startup logs go to `assets/temp/comfyui_startup.log`.

### Animation system (`app/core/animation/`)

Package with 5 modules:
- `clip.py` — `AnimationClip` (keyframes + interpolation), `Keyframe` (time, easing, property values)
- `engine.py` — `AnimationEngine` (per-instance playback state, clip management, evaluation)
- `texture.py` — `TextureAnimator` (sprite-sheet frame extraction for animated stickers)
- `gen.py` — AI-driven texture generation (`generate_animated_sticker()`: ComfyUI AnimateDiff → rembg background removal → `pack_frames_to_sprite_sheet()`)
- `export.py` — GIF/MP4 export with progress reporting

Animations evaluate per-frame in `_evaluate_animations()`, merging animation-driven transforms with manual adjustments in `_render_frame()`.

### Style presets (`app/utils/config_loader.py` + `app/ui/style_preset_manager_dialog.py`)

Style presets bundle prompt templates with LoRA configurations. Built-in presets (marked `is_builtin_preset`) ship with the app; custom presets are user-defined. Functions: `build_styled_prompt()` wraps user text with the selected preset's prompt template; `add_preset()`/`update_preset()`/`delete_preset()`/`reset_preset()` manage the preset catalog. The `StylePresetManagerDialog` provides a two-pane Qt UI (preset list + editable fields).

## Key design patterns

### Protocol (`app/core/protocol.py`)
All inter-process queue messages are typed dataclasses. Action constants live in namespaced classes (`Adj.MOVE`, `Gal.ADD_STICKER`, `Draw.STROKE_BEGIN`, `Disp.STICKER_SAVED`, `Anim.PLAY`, `Result.GENERATION_RESULT`). Union types (`AdjustmentMsg`, `GalleryMsg`, `DrawMsg`, `DisplayStatusMsg`, `AnimationMsg`) document valid message shapes. New queue messages should follow this pattern — no bare dicts.

### Brush system (`app/core/brush.py`)
PNG-based brush tips rendered via alpha-composite stamping. `stamp_brush()` blends a brush tip onto a BGRA canvas; `stamp_line()` stamps at intervals along a line segment with configurable spacing and random scatter. Brush tips are cached by `(filename, size)` in `_tip_cache`. Default brushes are auto-generated on first run.

### Face drawing (`app/core/face_draw.py`)
`FaceDrawCanvas` maps frame coordinates to a 512×512 square canvas via a perspective transform (`cv2.getPerspectiveTransform`) from the detected face quad. Each stroke opens a new undo entry, transforms frame points to canvas coords, and stamps along segments.

### Pressure sensitivity
`QTabletEvent` provides `event.pressure()` (0.0–1.0). A `_tablet_in_use` flag suppresses mouse events during tablet use to prevent double-handling. Pressure mode (`none`/`size`/`opacity`/`both`) controls whether pressure affects brush size, opacity, or both. Minimum ratio is `PRESSURE_MIN_RATIO = 0.2`.

### Templates (`app/core/templates.py`)
9 region-specific template stickers are procedurally generated as PNGs on first run (`ensure_templates()`). They are displayed alongside user stickers in the gallery via filter tabs, and loaded through `gallery_queue` as `GalLoadTemplate`.

### Override pattern
Brush spacing/scatter have per-canvas overrides (`_spacing_override`/`_scatter_override`). When `None`, the per-brush config value from `brushes.json` is used. The UI sliders set the overrides; the canvas checks overrides first in `_draw_segment()`.

### Undo system
Both `FaceDrawCanvas` and `DrawingCanvas` maintain a `_undo_stack` (max 20 entries, FIFO eviction). `_push_undo()` copies the current canvas before any stroke begins.

### Sticker persistence (`app/utils/storage.py`)
Stickers are stored as PNG files in `assets/gallery/` with metadata in `index.json`. Each sticker gets a UUID-based filename and thumbnail. `save_sticker_adjustments()`/`get_sticker_adjustments()` persist per-sticker offset/rotation/scale so stickers restore their position across sessions. Thread-safe via `_index_lock`.

### Chat panel (`app/ui/chat_panel.py`)
`ChatMessagePanel` — a QScrollArea widget showing user/agent dialog bubbles above the input row. User bubbles use `rgba(PRIMARY, 0.1)` background; agent bubbles use `PARCHMENT`. Status icons (✓/✗/❓/🔄) prefix agent messages. Capped at 20 messages with auto-scroll.

### Chinese path handling
OpenCV's `cv2.imread()` does not support Unicode/Chinese paths on Windows. Always use `np.fromfile(path, dtype=np.uint8)` + `cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)` to load images from disk. `load_rgba_sticker()` in `image_proc.py` uses this pattern. Keep PNG filenames ASCII — Chinese text in prompts can leak into generated filenames via ComfyUI.

### Config management (`app/utils/config_loader.py`)
`get_config()` returns the canonical config dict (cached after first load). `save_config()` does a deep-merge against the current config — it never writes `config.json` if values haven't changed, and strips Python dicts from values to keep the JSON portable. The `_deep_merge()` helper recurses on dicts only; non-dict values (lists, scalars) are replaced wholesale.

## Debugging

Unit tests alone are insufficient — 401 passing tests can still miss runtime bugs (Chinese path encoding, silent error drops, aggressive auto-clear). When fixing a runtime bug, verify the fix by launching the actual application and exercising the workflow end-to-end. Check console output for warnings or errors that tests don't catch. Use `python app/main.py --mock` for faster UI-only iteration without ComfyUI.

## Project structure

```
FaceDoodle/
├── app/
│   ├── main.py                      # Entry point, process & queue initialization
│   ├── ai/
│   │   ├── agent.py                 # DeepSeek multi-turn conversation parser
│   │   ├── generator.py             # ComfyUI API client
│   │   ├── comfy_manager.py         # ComfyUI subprocess lifecycle management
│   │   └── workflows/               # ComfyUI workflow JSON templates (×3)
│   ├── core/
│   │   ├── animation/               # Animation system (clip/engine/texture/gen/export)
│   │   ├── brush.py                 # Brush engine (PNG tip stamping)
│   │   ├── face_mesh.py             # MediaPipe 468-landmark detection
│   │   ├── face_draw.py             # Face drawing canvas + coordinate mapping
│   │   ├── protocol.py              # All queue message typed dataclass definitions
│   │   ├── renderer.py              # Sticker perspective compositing + face mesh
│   │   ├── templates.py             # Built-in template sticker generation
│   │   ├── tracker.py               # Consumer main loop, AI scheduling & render orchestration
│   │   ├── tracker_stickers.py      # StickerManager mixin (CRUD operations)
│   │   └── tracker_animation.py     # AnimationProcessor mixin (animation queue dispatch)
│   ├── ui/
│   │   ├── main_window.py           # Main window + event handling + gallery management
│   │   ├── chat_panel.py            # Chat message bubble panel
│   │   ├── widgets.py               # Reusable widgets (gallery cards, canvas, buttons)
│   │   ├── sticker_panel.py         # Sticker panel components
│   │   ├── drawing_widgets.py       # Drawing-related widgets
│   │   ├── animation_timeline.py    # Animation timeline panel
│   │   ├── animation_gen_dialog.py  # Animation generation dialog
│   │   ├── style_preset_manager_dialog.py  # Style preset create/edit/delete dialog
│   │   └── theme.py                 # Theme palette & fonts
│   └── utils/
│       ├── config_loader.py         # Config loading, style presets, prompt building
│       ├── image_proc.py            # Image loading & preprocessing
│       ├── storage.py               # Sticker persistence (UUID filenames, index.json)
│       └── logging_config.py        # Centralized multi-process logging setup
├── assets/
│   ├── brushes/                     # Brush tip PNGs + brushes.json config
│   ├── templates/                   # Built-in templates (auto-generated)
│   └── gallery/                     # User stickers (PNG + index.json)
├── tests/                           # pytest suite (401 tests across 17 files)
├── scripts/
│   └── check_syntax.py              # Pre-commit syntax check script
├── config.json                      # Application configuration
├── requirements.txt
├── .pre-commit-config.yaml
└── README.md
```
