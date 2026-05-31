import cv2
import numpy as np

from app.core.brush import BrushStateMixin, stamp_brush, stamp_line

CANVAS_SIZE = 1024
MAX_UNDO = 20


class FaceDrawCanvas(BrushStateMixin):
    def __init__(self):
        super().__init__()
        self._canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)
        self._undo_stack = []
        self._prev_canvas_pt = None
        self._M_inv = None
        self._stroke_undo_pushed = False
        self._init_brush_state()

    @property
    def canvas(self):
        return self._canvas

    @property
    def has_content(self):
        return bool(np.any(self._canvas[:, :, 3] > 0))

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

    def _draw_segment(self, c1, c2):
        spacing_px, scatter_px, opacity_scale = self._resolve_spacing_scatter()
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
