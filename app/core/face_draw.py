import cv2
import numpy as np

CANVAS_SIZE = 512
MAX_UNDO = 20


class FaceDrawCanvas:
    def __init__(self):
        self._canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)
        self._undo_stack = []
        self._brush_size = 12
        self._brush_color = (0, 0, 0, 255)
        self._eraser_mode = False
        self._prev_canvas_pt = None     # 当前 stroke 的上一个画布坐标
        self._M_inv = None              # 当前 stroke 的逆透视矩阵
        self._stroke_undo_pushed = False

    @property
    def canvas(self):
        return self._canvas

    @property
    def has_content(self):
        return bool(np.any(self._canvas[:, :, 3] > 0))

    def set_brush_size(self, size):
        self._brush_size = max(1, min(50, int(size)))

    def set_brush_color(self, bgra):
        self._brush_color = tuple(bgra)

    def set_eraser_mode(self, on):
        self._eraser_mode = bool(on)

    def clear(self):
        self._push_undo()
        self._canvas.fill(0)

    def undo(self):
        if self._undo_stack:
            self._canvas = self._undo_stack.pop()

    def _push_undo(self):
        self._undo_stack.append(self._canvas.copy())
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)

    # ── 增量 stroke API ──

    def begin_stroke(self, face_quad):
        self._push_undo()
        self._stroke_undo_pushed = True
        canvas_corners = np.array([
            [0, 0], [CANVAS_SIZE - 1, 0],
            [CANVAS_SIZE - 1, CANVAS_SIZE - 1], [0, CANVAS_SIZE - 1]
        ], dtype=np.float32)
        self._M_inv = cv2.getPerspectiveTransform(face_quad.astype(np.float32), canvas_corners)
        self._prev_canvas_pt = None

    def add_stroke_segment(self, p1, p2):
        """绘制从 p1 到 p2 的线段 (帧坐标)，实时追加到画布"""
        if self._M_inv is None:
            return
        color = (0, 0, 0, 0) if self._eraser_mode else self._brush_color
        c1 = self._frame_to_canvas(p1)
        c2 = self._frame_to_canvas(p2)
        # 从上一个点连到 c1（保证接续）
        if self._prev_canvas_pt is not None:
            cv2.line(self._canvas, self._prev_canvas_pt, c1, color, self._brush_size, cv2.LINE_AA)
        cv2.line(self._canvas, c1, c2, color, self._brush_size, cv2.LINE_AA)
        cv2.circle(self._canvas, c2, self._brush_size // 2, color, -1, cv2.LINE_AA)
        self._prev_canvas_pt = c2

    def end_stroke(self):
        self._M_inv = None
        self._prev_canvas_pt = None
        self._stroke_undo_pushed = False

    def add_stroke_point(self, pt):
        """添加单个帧坐标点，连接到上一个点"""
        if self._M_inv is None:
            return
        color = (0, 0, 0, 0) if self._eraser_mode else self._brush_color
        cc = self._frame_to_canvas(pt)
        if self._prev_canvas_pt is not None:
            cv2.line(self._canvas, self._prev_canvas_pt, cc, color, self._brush_size, cv2.LINE_AA)
        cv2.circle(self._canvas, cc, self._brush_size // 2, color, -1, cv2.LINE_AA)
        self._prev_canvas_pt = cc

    # ── 坐标映射 ──

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
