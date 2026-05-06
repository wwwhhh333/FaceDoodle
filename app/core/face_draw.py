import cv2
import numpy as np

from app.core.brush import (load_brush_config, get_brush_by_id, load_brush_tip,
                            stamp_brush, stamp_line, PRESSURE_MIN_RATIO)

CANVAS_SIZE = 512
MAX_UNDO = 20


class FaceDrawCanvas:
    def __init__(self):
        self._canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)
        self._undo_stack = []
        self._brush_size = 12
        self._brush_color = (0, 0, 0, 255)
        self._eraser_mode = False
        self._prev_canvas_pt = None
        self._M_inv = None
        self._stroke_undo_pushed = False

        brushes = load_brush_config()
        default = brushes[0] if brushes else None
        self._brush_type = default["id"] if default else "hard_round"
        self._brush_config = default
        self._brush_tip = None
        self._pressure = 1.0
        self._pressure_mode = "both"
        self._spacing_override = None
        self._scatter_override = None

        self._rebuild_tip()

    @property
    def canvas(self):
        return self._canvas

    @property
    def has_content(self):
        return bool(np.any(self._canvas[:, :, 3] > 0))

    def set_brush_size(self, size):
        self._brush_size = max(1, min(50, int(size)))
        self._rebuild_tip()

    def set_brush_color(self, bgra):
        self._brush_color = tuple(bgra)

    def set_eraser_mode(self, on):
        self._eraser_mode = bool(on)

    def set_brush_type(self, brush_id):
        cfg = get_brush_by_id(brush_id)
        if cfg is None:
            return
        self._brush_type = brush_id
        self._brush_config = cfg
        self._rebuild_tip()

    def set_pressure(self, p):
        self._pressure = max(0.0, min(1.0, float(p)))

    def set_pressure_mode(self, mode):
        self._pressure_mode = mode

    def set_spacing(self, coef):
        self._spacing_override = max(0.03, min(2.0, float(coef)))

    def set_scatter(self, px):
        self._scatter_override = max(0.0, min(30.0, float(px)))

    def clear(self):
        self._push_undo()
        self._canvas.fill(0)

    def undo(self):
        if self._undo_stack:
            self._canvas = self._undo_stack.pop()

    # ── internal ──

    def _push_undo(self):
        self._undo_stack.append(self._canvas.copy())
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)

    def _rebuild_tip(self):
        cfg = self._brush_config
        tip_file = cfg["tip"] if cfg else "hard_round.png"
        size = self._effective_size()
        tip = load_brush_tip(tip_file, size)
        if tip is None and tip_file != "hard_round.png":
            tip = load_brush_tip("hard_round.png", size)
        self._brush_tip = tip

    def _effective_size(self):
        size_scale, _ = self._pressure_scales()
        return max(1, int(self._brush_size * size_scale))

    def _pressure_scales(self):
        p = self._pressure
        if self._pressure_mode == "none":
            return 1.0, 1.0
        elif self._pressure_mode == "size":
            return PRESSURE_MIN_RATIO + (1.0 - PRESSURE_MIN_RATIO) * p, 1.0
        elif self._pressure_mode == "opacity":
            return 1.0, PRESSURE_MIN_RATIO + (1.0 - PRESSURE_MIN_RATIO) * p
        else:
            s = PRESSURE_MIN_RATIO + (1.0 - PRESSURE_MIN_RATIO) * p
            return s, s

    # ── incremental stroke API ──

    def begin_stroke(self, face_quad):
        self._push_undo()
        self._stroke_undo_pushed = True
        canvas_corners = np.array([
            [0, 0], [CANVAS_SIZE - 1, 0],
            [CANVAS_SIZE - 1, CANVAS_SIZE - 1], [0, CANVAS_SIZE - 1]
        ], dtype=np.float32)
        self._M_inv = cv2.getPerspectiveTransform(face_quad.astype(np.float32), canvas_corners)
        self._prev_canvas_pt = None

    def add_stroke_point(self, pt):
        if self._M_inv is None:
            return
        cc = self._frame_to_canvas(pt)
        if self._prev_canvas_pt is not None:
            self._draw_segment(self._prev_canvas_pt, cc)
        else:
            # First point: single stamp at position
            self._ensure_tip()
            _, opacity_scale = self._pressure_scales()
            stamp_brush(self._canvas, cc[0], cc[1], self._brush_tip,
                        self._brush_color, opacity_scale, eraser=self._eraser_mode)
        self._prev_canvas_pt = cc

    def add_stroke_segment(self, p1, p2):
        """Draw segment from p1 to p2 (frame coords), for batch usage."""
        if self._M_inv is None:
            return
        c1 = self._frame_to_canvas(p1)
        c2 = self._frame_to_canvas(p2)
        if self._prev_canvas_pt is not None:
            self._draw_segment(self._prev_canvas_pt, c1)
        self._draw_segment(c1, c2)
        self._prev_canvas_pt = c2

    def end_stroke(self):
        self._M_inv = None
        self._prev_canvas_pt = None
        self._stroke_undo_pushed = False

    def _ensure_tip(self):
        eff_size = self._effective_size()
        if self._brush_tip is None or self._brush_tip.shape[0] != eff_size * 2 + 1:
            self._rebuild_tip()

    def _draw_segment(self, c1, c2):
        cfg = self._brush_config
        spacing_coef = (self._spacing_override if self._spacing_override is not None
                        else (cfg.get("spacing", 0.3) if cfg else 0.3))
        scatter_coef = (self._scatter_override if self._scatter_override is not None
                        else (cfg.get("scatter", 0.0) if cfg else 0.0))

        eff_size = self._effective_size()
        spacing_px = max(1.0, eff_size * spacing_coef)
        scatter_px = scatter_coef * eff_size

        _, opacity_scale = self._pressure_scales()

        self._ensure_tip()

        stamp_line(self._canvas, c1, c2, self._brush_tip,
                   self._brush_color, opacity_scale, spacing_px, scatter=scatter_px,
                   eraser=self._eraser_mode)

    # ── coordinate mapping ──

    def _frame_to_canvas(self, pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self._M_inv)
        cx, cy = dst[0][0]
        cx = max(0, min(CANVAS_SIZE - 1, int(cx)))
        cy = max(0, min(CANVAS_SIZE - 1, int(cy)))
        return cx, cy

    def get_result(self):
        alpha = self._canvas[:, :, 3]
        rows, cols = np.where(alpha > 0)
        if len(rows) == 0:
            return None
        margin = 10
        y1 = max(rows.min() - margin, 0)
        y2 = min(rows.max() + margin + 1, CANVAS_SIZE)
        x1 = max(cols.min() - margin, 0)
        x2 = min(cols.max() + margin + 1, CANVAS_SIZE)
        return self._canvas[y1:y2, x1:x2].copy()
