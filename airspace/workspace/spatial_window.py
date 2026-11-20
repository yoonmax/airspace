"""A floating window: geometry, state, chrome hit-testing and animation."""

from __future__ import annotations

import itertools
from typing import Any

from airspace.ui.animations import Spring
from airspace.workspace.geometry import Rect

_ids = itertools.count(1)

RESIZE_BAND = 9.0  # px band along edges for mouse resizing
CORNER = 22.0  # px corner hot zone
HAND_CORNER = 34.0  # larger target for the visible bottom-right grip when using a hand


class SpatialWindow:
    def __init__(self, app: Any, rect: Rect, titlebar_h: float = 42.0) -> None:
        self.id = next(_ids)
        self.app = app
        self.app_type: str = app.app_type
        self.title: str = app.title
        self.rect = rect
        self.titlebar_h = titlebar_h
        self.min_w, self.min_h = app.min_size
        self.max_w, self.max_h = app.max_size or (10_000.0, 10_000.0)
        self.z = 0
        self.focused = False
        self.minimized = False
        self.dragging = False
        self.resizing = False
        self.closing = False
        self.hover_part: str | None = None
        # animation state
        self.appear = Spring(0.0, 300, damping=26)  # 0 = hidden, 1 = shown
        self.appear.target = 1.0
        self.lift = Spring(0.0, 400)  # selection elevation
        self.highlight = Spring(0.0, 500)  # active-interaction border glow
        self.anim_x = Spring(rect.x, 520)
        self.anim_y = Spring(rect.y, 520)
        self.animating_move = False
        self.velocity = (0.0, 0.0)  # px/s, for release momentum

    # ----------------------------------------------------------- geometry
    @property
    def content_rect(self) -> Rect:
        r = self.rect
        return Rect(r.x, r.y + self.titlebar_h, r.w, max(0.0, r.h - self.titlebar_h))

    def display_rect(self) -> Rect:
        """Where the window is drawn (accounts for move animation)."""
        if self.animating_move:
            return Rect(self.anim_x.value, self.anim_y.value, self.rect.w, self.rect.h)
        return self.rect

    @property
    def visible(self) -> bool:
        return not self.minimized and not (self.closing and self.appear.value <= 0.01)

    @property
    def interactive(self) -> bool:
        return not self.minimized and not self.closing

    def button_rects(self) -> dict[str, Rect]:
        r = self.rect
        size = self.titlebar_h * 0.62
        y = r.y + (self.titlebar_h - size) / 2
        right = r.right - self.titlebar_h * 0.22
        close = Rect(right - size, y, size, size)
        minimize = Rect(close.x - size - self.titlebar_h * 0.12, y, size, size)
        return {"close": close, "minimize": minimize}

    def part_at(self, x: float, y: float, hand: bool = False) -> str | None:
        """Which part of the window chrome/content is at (x, y), or None."""
        r = self.rect
        band = RESIZE_BAND
        corner = HAND_CORNER if hand else CORNER
        outer = Rect(r.x - band / 2, r.y - band / 2, r.w + band, r.h + band)
        if not outer.contains(x, y):
            return None
        # The visible grip in the bottom-right corner is the primary resize affordance.
        if x >= r.right - corner and y >= r.bottom - corner:
            return "resize:se"
        if not hand:
            on_l, on_r = x < r.x + band / 2, x > r.right - band / 2
            on_t, on_b = y < r.y + band / 2, y > r.bottom - band / 2
            near_l, near_r = x < r.x + CORNER, x > r.right - CORNER
            near_t, near_b = y < r.y + CORNER, y > r.bottom - CORNER
            if (on_t and near_l) or (on_l and near_t):
                return "resize:nw"
            if (on_t and near_r) or (on_r and near_t):
                return "resize:ne"
            if (on_b and near_l) or (on_l and near_b):
                return "resize:sw"
            if on_l:
                return "resize:w"
            if on_r:
                return "resize:e"
            if on_b:
                return "resize:s"
            if on_t:
                return "resize:n"
        slop = 4.0 if hand else 0.0
        for name, br in self.button_rects().items():
            if br.contains(x, y, slop):
                return name
        if r.y <= y < r.y + self.titlebar_h:
            return "titlebar"
        if r.contains(x, y):
            return "content"
        return None

    # ---------------------------------------------------------- animation
    def tick(self, dt: float) -> None:
        self.appear.step(dt)
        self.lift.target = 1.0 if (self.dragging or self.resizing) else (0.35 if self.focused else 0.0)
        self.lift.step(dt)
        self.highlight.target = 1.0 if (self.dragging or self.resizing) else 0.0
        self.highlight.step(dt)
        if self.animating_move:
            self.anim_x.target, self.anim_y.target = self.rect.x, self.rect.y
            self.anim_x.step(dt)
            self.anim_y.step(dt)
            if self.anim_x.settled and self.anim_y.settled:
                self.animating_move = False

    def animate_to(self, rect: Rect) -> None:
        """Move with a spring (used by arrange/restore, never during direct manipulation)."""
        cur = self.display_rect()
        self.anim_x.set(cur.x)
        self.anim_y.set(cur.y)
        self.rect = rect
        self.animating_move = True

    def stop_animation(self) -> None:
        self.animating_move = False
