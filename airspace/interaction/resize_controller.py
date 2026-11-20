"""Window resizing: edge/corner handles and the two-hand stretch."""

from __future__ import annotations

from airspace.interaction.events import Grab, InteractionState, PointerEvent, TwoHandInfo
from airspace.workspace.geometry import Rect


def resize_rect(start: Rect, edges: str, dx: float, dy: float, min_w: float, min_h: float,
                max_w: float = 1e9, max_h: float = 1e9) -> Rect:
    """Resize ``start`` by dragging ``edges`` (any of 'n','s','e','w'); opposite edges stay fixed."""
    x0, y0, x1, y1 = start.x, start.y, start.right, start.bottom
    if "e" in edges:
        x1 = min(max(x1 + dx, x0 + min_w), x0 + max_w)
    if "w" in edges:
        x0 = max(min(x0 + dx, x1 - min_w), x1 - max_w)
    if "s" in edges:
        y1 = min(max(y1 + dy, y0 + min_h), y0 + max_h)
    if "n" in edges:
        y0 = max(min(y0 + dy, y1 - min_h), y1 - max_h)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def stretch_rect(start: Rect, start_info: TwoHandInfo, now: TwoHandInfo, min_w: float, min_h: float,
                 max_w: float = 1e9, max_h: float = 1e9) -> Rect:
    """Two-hand stretch: scale about the window centre by the change in hand
    separation; the window centre follows the hands' midpoint."""
    scale = now.distance / max(start_info.distance, 1.0)
    scale = max(0.25, min(4.0, scale))
    w = max(min_w, min(start.w * scale, max_w))
    h = max(min_h, min(start.h * scale, max_h))
    cx = start.cx + (now.cx - start_info.cx)
    cy = start.cy + (now.cy - start_info.cy)
    return Rect(cx - w / 2, cy - h / 2, w, h)


class WindowResizeGrab(Grab):
    kind = InteractionState.RESIZING_WINDOW

    def __init__(self, wm, win, edges: str, ev: PointerEvent) -> None:
        self.wm, self.win, self.edges = wm, win, edges
        self.start = win.rect.copy()
        self.sx, self.sy = ev.gx, ev.gy
        win.resizing = True
        wm.focus(win)

    def move(self, ev: PointerEvent) -> None:
        r = resize_rect(self.start, self.edges, ev.gx - self.sx, ev.gy - self.sy,
                        self.win.min_w, self.win.min_h, self.win.max_w, self.win.max_h)
        self.wm.set_rect(self.win, r)

    def release(self, ev: PointerEvent) -> None:
        self.win.resizing = False

    def cancel(self) -> None:
        self.win.resizing = False
        self.wm.set_rect(self.win, self.start)
