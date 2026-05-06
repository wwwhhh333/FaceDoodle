# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run

```bash
# Normal mode (requires ComfyUI running locally)
python app/main.py

# Mock mode (bypasses ComfyUI, uses cached images for UI testing)
python app/main.py --mock
```

Requires `DEEPSEEK_API_KEY` or `MODELSCOPE_API_KEY` environment variable. ComfyUI must be running at the address in `config.json` (`comfyui.server_address`, default `127.0.0.1:8188`).

No test suite or linting scripts exist.

## Architecture

Multi-process application: **Producer** (camera capture) → queue → **Consumer** (face detection + sticker rendering + AI orchestration) → queue → **Qt UI** (display + input).

Five multiprocessing queues connect the UI process to the consumer process:

| Queue | Direction | Purpose |
|-------|-----------|---------|
| `frame_queue` | Producer → Consumer | Raw camera frames |
| `display_queue` | Consumer → UI | Rendered frames + status messages (`sticker_saved`, `generation_failed`) |
| `command_queue` | UI → Consumer | AI generation prompts (text or `{"type":"img2img", ...}` dict) |
| `adjustment_queue` | UI → Consumer | Sticker edit actions (`move`, `rotate`, `scale`, `reset`, `toggle_edit`) |
| `gallery_queue` | UI → Consumer | Load sticker/template onto face (`load_sticker`, `load_template`) |
| `draw_queue` | UI → Consumer | Face drawing actions (`stroke_begin/point/end`, `set_brush`, `undo`, `clear`, `save`, etc.) |

**Consumer** (`app/core/tracker.py`) is the central orchestrator — it runs a single main loop that processes all queues per frame, runs face detection (MediaPipe 468 landmarks), renders the scene, and pushes frames to the display queue. AI generation runs in a daemon thread to avoid blocking the frame loop.

**UI** (`app/ui/main_window.py` `FaceDoodleWindow`) uses a `VideoUpdateThread` to pull from `display_queue` and update the video label. All drawing/sticker events go through the queues — the UI never directly manipulates canvas state.

## Key design patterns

### Brush system (`app/core/brush.py`)
PNG-based brush tips rendered via alpha-composite stamping. `stamp_brush()` blends a brush tip onto a BGRA canvas; `stamp_line()` stamps at intervals along a line segment with configurable spacing and random scatter. Brush tips are cached by `(filename, size)` in `_tip_cache`. Default brushes are auto-generated on first run.

**Windows critical**: OpenCV's `cv2.imread()` does not support Unicode/Chinese paths on Windows. Always use `np.fromfile(path, dtype=np.uint8)` + `cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)` to load images from disk.

### Face drawing (`app/core/face_draw.py`)
`FaceDrawCanvas` maps frame coordinates to a 512×512 square canvas via a perspective transform (`cv2.getPerspectiveTransform`) from the detected face quad. Each stroke opens a new undo entry, transforms frame points to canvas coords, and stamps along segments.

### Pressure sensitivity
`QTabletEvent` provides `event.pressure()` (0.0–1.0). A `_tablet_in_use` flag suppresses mouse events during tablet use to prevent double-handling. Pressure mode (`none`/`size`/`opacity`/`both`) controls whether pressure affects brush size, opacity, or both. Minimum ratio is `PRESSURE_MIN_RATIO = 0.2`.

### Templates (`app/core/templates.py`)
9 region-specific template stickers are procedurally generated as PNGs on first run (`ensure_templates()`). They are displayed alongside user stickers in the gallery via filter tabs, and loaded through `gallery_queue` as `{"action": "load_template", "template": {...}}` carrying the full image dict.

### Override pattern
Brush spacing/scatter have per-canvas overrides (`_spacing_override`/`_scatter_override`). When `None`, the per-brush config value from `brushes.json` is used. The UI sliders set the overrides; the canvas checks overrides first in `_draw_segment()`. This lets users tweak parameters without modifying the brush config file.

### Undo system
Both `FaceDrawCanvas` and `DrawingCanvas` maintain a `_undo_stack` (max 20 entries, FIFO eviction). `_push_undo()` copies the current canvas before any stroke begins.

### Sticker persistence (`app/utils/storage.py`)
Stickers are stored as PNG files in `assets/gallery/` with metadata in `index.json`. Each sticker gets a UUID-based filename and thumbnail. `save_sticker_adjustments()`/`get_sticker_adjustments()` persist per-sticker offset/rotation/scale so stickers restore their position across sessions. Thread-safe via `_index_lock`.

### Chinese path handling
Beyond the imread fix, `np.fromfile()` + `cv2.imdecode()` is the project convention for loading any image that may have a Chinese filename. `load_rgba_sticker()` in `image_proc.py` also uses this pattern.
