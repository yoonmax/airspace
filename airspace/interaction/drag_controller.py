"""Window dragging (with optional release momentum) and the two-hand resize."""

from __future__ import annotations

from collections import deque

from airspace.interaction.events import Grab, InteractionState, PointerEvent, TwoHandInfo
from airspace.interaction.resize_controller import stretch_rect


class WindowDragGrab(Grab):
    kind = InteractionState.DRAGGING_WINDOW
    supports_two_hand = True

    def __init__(self, wm, win, ev: PointerEvent, momentum: bool = True) -> None:
        self.wm, self.win = wm, win
        self.start = win.rect.copy()
        self.dx, self.dy = ev.gx - win.rect.x, ev.gy - win.rect.y
        self.momentum = momentum
        self._samples: deque[tuple[float, float, float]] = deque(maxlen=6)
        self._samples.append((ev.t, ev.gx, ev.gy))
        self._two_start = None
        self._two_rect = None
        self._rebase = False
        win.dragging = True
        wm.focus(win)

    def move(self, ev: PointerEvent) -> None:
        if self._two_start is not None:
            return  # two-hand mode owns geometry
        if self._rebase:
            self.rebase(ev)
            self._rebase = False
        self._samples.append((ev.t, ev.gx, ev.gy))
        self.wm.move_to(self.win, ev.gx - self.dx, ev.gy - self.dy)

    def release(self, ev: PointerEvent) -> None:
        self.win.dragging = False
        self.win.resizing = False
        if self.momentum and not ev.cancelled and self._two_start is None and len(self._samples) >= 3:
            t0, x0, y0 = self._samples[0]
            t1, x1, y1 = self._samples[-1]
            dt = t1 - t0
            if dt > 1e-3:
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
                speed = (vx * vx + vy * vy) ** 0.5
                if speed > 600:  # only a deliberate flick carries momentum
                    k = min(0.12, 160.0 / speed)  # gentle glide, bounded distance
                    target = self.win.rect.copy()
                    target.x += vx * k
                    target.y += vy * k
                    self.win.animate_to(self.wm.constrain(self.win, target))

    def cancel(self) -> None:
        self.win.dragging = False
        self.win.resizing = False
        self.wm.set_rect(self.win, self.start)

    # -------------------------------------------------- two-hand stretch
    def two_hand_begin(self, info: TwoHandInfo) -> bool:
        self._two_start = info
        self._two_rect = self.win.rect.copy()
        self.win.resizing = True
        self.kind = InteractionState.RESIZING_WINDOW
        return True

    def two_hand_update(self, start: TwoHandInfo, now: TwoHandInfo) -> None:
        if self._two_rect is None:
            return
        w = self.win
        self.wm.set_rect(w, stretch_rect(self._two_rect, start, now, w.min_w, w.min_h, w.max_w, w.max_h))

    def two_hand_end(self) -> None:
        self.win.resizing = False
        self._two_start = None
        self.kind = InteractionState.DRAGGING_WINDOW
        # Continue dragging from the new geometry without a jump.
        self._samples.clear()
        self._rebase = True

    def rebase(self, ev: PointerEvent) -> None:
        self.dx, self.dy = ev.gx - self.win.rect.x, ev.gy - self.win.rect.y
