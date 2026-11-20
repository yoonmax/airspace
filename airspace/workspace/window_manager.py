"""The airspace window manager: open, close, focus, z-order, minimise,
constraints and layout persistence. Qt-free so it can be unit tested."""

from __future__ import annotations

import logging
from typing import Any, Callable

from airspace.persistence.workspace_storage import WindowRecord, WorkspaceLayout
from airspace.workspace.geometry import Rect
from airspace.workspace.spatial_window import SpatialWindow

log = logging.getLogger(__name__)

AppFactory = Callable[[str], Any]
Placement = Callable[[float, float, list[SpatialWindow]], Rect]


class WindowManager:
    KEEP_VISIBLE = 96.0  # px of a window that must stay on screen horizontally

    def __init__(self, factory: AppFactory, bounds: Rect | None = None, top_inset: float = 60.0,
                 bottom_inset: float = 12.0, titlebar_h: float = 42.0) -> None:
        self.factory = factory
        self.bounds = bounds or Rect(0, 0, 1280, 720)
        self.top_inset = top_inset
        self.bottom_inset = bottom_inset
        self.titlebar_h = titlebar_h
        self.windows: list[SpatialWindow] = []  # bottom → top
        self.placement: Placement | None = None
        self.on_change: Callable[[], None] | None = None
        self.size_scale = 1.0

    # ------------------------------------------------------------ queries
    @property
    def visible_windows(self) -> list[SpatialWindow]:
        return [w for w in self.windows if w.visible]

    @property
    def focused(self) -> SpatialWindow | None:
        return next((w for w in reversed(self.windows) if w.focused and w.interactive), None)

    @property
    def minimized(self) -> list[SpatialWindow]:
        return [w for w in self.windows if w.minimized and not w.closing]

    def find(self, app_type: str) -> SpatialWindow | None:
        return next((w for w in self.windows if w.app_type == app_type and not w.closing), None)

    def by_id(self, win_id: int) -> SpatialWindow | None:
        return next((w for w in self.windows if w.id == win_id), None)

    def window_at(self, x: float, y: float, hand: bool = False) -> SpatialWindow | None:
        for w in reversed(self.windows):
            if w.interactive and w.part_at(x, y, hand) is not None:
                return w
        return None

    # -------------------------------------------------------- operations
    def open(self, app_type: str, rect: Rect | None = None, state: dict | None = None,
             focus: bool = True) -> SpatialWindow:
        existing = self.find(app_type)
        if existing is not None and getattr(existing.app, "single_instance", True):
            if existing.minimized:
                self.restore(existing)
            self.focus(existing)
            return existing
        app = self.factory(app_type)
        if rect is None:
            dw, dh = app.default_size
            dw, dh = dw * self.size_scale, dh * self.size_scale
            dw = min(dw, self.bounds.w - 24)
            dh = min(dh, self.bounds.h - self.top_inset - self.bottom_inset - 12)
            rect = self.placement(dw, dh, self.visible_windows) if self.placement else self._cascade(dw, dh)
        win = SpatialWindow(app, rect, self.titlebar_h)
        win.rect = self.constrain(win, rect)
        if state:
            try:
                app.restore_state(state)
            except Exception:  # noqa: BLE001 - a bad saved state must not block opening
                log.exception("Could not restore state for %s", app_type)
        app._close_request = lambda w=win: self.close(w)
        self.windows.append(win)
        self._sync_app_size(win)
        if focus:
            self.focus(win)
        self._changed()
        return win

    def close(self, win: SpatialWindow, animate: bool = True) -> None:
        if win not in self.windows or win.closing:
            return
        win.closing = True
        win.focused = False
        win.dragging = win.resizing = False
        win.appear.target = 0.0
        try:
            win.app.on_close()
        except Exception:  # noqa: BLE001
            log.exception("Error while closing %s", win.app_type)
        if not animate:
            self.windows.remove(win)
        self._focus_topmost()
        self._changed()

    def focus(self, win: SpatialWindow, raise_: bool = True) -> None:
        if win not in self.windows or win.closing:
            return
        for w in self.windows:
            if w.focused != (w is win):
                w.focused = w is win
                try:
                    w.app.on_focus(w.focused)
                except Exception:  # noqa: BLE001
                    log.exception("on_focus failed")
        if raise_:
            self.bring_to_front(win)

    def bring_to_front(self, win: SpatialWindow) -> None:
        if win in self.windows:
            self.windows.remove(win)
            self.windows.append(win)
            self._renumber()

    def minimize(self, win: SpatialWindow) -> None:
        if win.minimized or win.closing:
            return
        win.minimized = True
        win.dragging = win.resizing = False
        win.focused = False
        win.appear.target = 0.0
        self._focus_topmost()
        self._changed()

    def restore(self, win: SpatialWindow) -> None:
        if not win.minimized:
            return
        win.minimized = False
        win.appear.value = min(win.appear.value, 0.2)
        win.appear.target = 1.0
        win.rect = self.constrain(win, win.rect)
        self.focus(win)
        self._changed()

    def minimize_all(self) -> None:
        for w in list(self.windows):
            if not w.closing:
                self.minimize(w)

    def restore_all(self) -> None:
        for w in list(self.windows):
            self.restore(w)

    def close_all(self) -> None:
        for w in list(self.windows):
            self.close(w, animate=False)

    def move_to(self, win: SpatialWindow, x: float, y: float) -> None:
        win.stop_animation()
        win.rect = self.constrain(win, Rect(x, y, win.rect.w, win.rect.h))
        self._changed()

    def set_rect(self, win: SpatialWindow, rect: Rect) -> None:
        win.stop_animation()
        win.rect = self.constrain(win, self.clamp_size(win, rect))
        self._sync_app_size(win)
        self._changed()

    # -------------------------------------------------------- constraints
    def clamp_size(self, win: SpatialWindow, rect: Rect) -> Rect:
        max_w = min(win.max_w, self.bounds.w)
        max_h = min(win.max_h, self.bounds.h - self.top_inset)
        w = max(win.min_w, min(rect.w, max_w))
        h = max(win.min_h, min(rect.h, max_h))
        return Rect(rect.x, rect.y, w, h)

    def constrain(self, win: SpatialWindow, rect: Rect) -> Rect:
        """Keep the title bar reachable: never let a window escape the workspace."""
        b = self.bounds
        keep = min(self.KEEP_VISIBLE, rect.w)
        x = min(max(rect.x, b.x - rect.w + keep), b.right - keep)
        y = min(max(rect.y, b.y + self.top_inset), b.bottom - self.titlebar_h - self.bottom_inset)
        return Rect(x, y, rect.w, rect.h)

    def set_bounds(self, bounds: Rect) -> None:
        """Scale window positions proportionally when the workspace is resized."""
        old = self.bounds
        self.bounds = bounds
        if old.w <= 0 or old.h <= 0:
            return
        for w in self.windows:
            cx = (w.rect.cx - old.x) / old.w
            cy = (w.rect.cy - old.y) / old.h
            nw, nh = min(w.rect.w, bounds.w - 24), min(w.rect.h, bounds.h - self.top_inset - 12)
            r = Rect(bounds.x + cx * bounds.w - nw / 2, bounds.y + cy * bounds.h - nh / 2, nw, nh)
            w.rect = self.constrain(w, self.clamp_size(w, r))
            w.stop_animation()
            self._sync_app_size(w)

    def set_titlebar_height(self, h: float) -> None:
        self.titlebar_h = h
        for w in self.windows:
            w.titlebar_h = h
            self._sync_app_size(w)

    # ------------------------------------------------------------- frame
    def tick(self, dt: float) -> None:
        for w in list(self.windows):
            w.tick(dt)
            if w.closing and w.appear.value <= 0.01:
                self.windows.remove(w)
            elif not w.closing:
                self._sync_app_size(w)
                try:
                    w.app.tick(dt)
                except Exception:  # noqa: BLE001
                    log.exception("%s.tick failed", w.app_type)

    # ------------------------------------------------------- persistence
    def to_layout(self) -> WorkspaceLayout:
        b = self.bounds
        records = []
        for w in self.windows:
            if w.closing:
                continue
            try:
                state = w.app.save_state()
            except Exception:  # noqa: BLE001
                log.exception("save_state failed for %s", w.app_type)
                state = {}
            records.append(WindowRecord(
                app_type=w.app_type, nx=(w.rect.x - b.x) / b.w, ny=(w.rect.y - b.y) / b.h,
                width=w.rect.w, height=w.rect.h, z=w.z, minimized=w.minimized, state=state))
        return WorkspaceLayout(records)

    def apply_layout(self, layout: WorkspaceLayout, known_types: set[str] | None = None) -> None:
        self.close_all()
        b = self.bounds
        for rec in sorted(layout.windows, key=lambda r: r.z):
            if known_types is not None and rec.app_type not in known_types:
                continue
            rect = Rect(b.x + rec.nx * b.w, b.y + rec.ny * b.h, rec.width, rec.height)
            try:
                win = self.open(rec.app_type, rect, rec.state, focus=False)
            except Exception:  # noqa: BLE001
                log.exception("Could not reopen %s", rec.app_type)
                continue
            win.rect = self.constrain(win, self.clamp_size(win, rect))
            if rec.minimized:
                win.minimized = True
                win.appear.set(0.0)
        top = next((w for w in reversed(self.windows) if not w.minimized), None)
        if top:
            self.focus(top, raise_=False)  # keep the saved stacking order
        self._renumber()
        self._changed()

    # ----------------------------------------------------------- helpers
    def _cascade(self, w: float, h: float) -> Rect:
        n = len(self.visible_windows)
        return Rect(self.bounds.x + 40 + 32 * (n % 8), self.bounds.y + self.top_inset + 20 + 28 * (n % 8), w, h)

    def _renumber(self) -> None:
        for i, w in enumerate(self.windows):
            w.z = i

    def _focus_topmost(self) -> None:
        top = next((w for w in reversed(self.windows) if w.interactive), None)
        if top is not None and not top.focused:
            self.focus(top)

    def _sync_app_size(self, win: SpatialWindow) -> None:
        c = win.content_rect
        resize = getattr(win.app, "resize", None)
        if resize:
            resize(c.w, c.h)

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()
