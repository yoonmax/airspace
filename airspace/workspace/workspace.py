"""The workspace scene: input routing across layers.

Layer order for input (top first): modal overlays → launcher → HUD → windows
(top-most first) → background. Rendering uses the reverse order.
"""

from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QKeyEvent

from airspace.interaction.drag_controller import WindowDragGrab
from airspace.interaction.events import CallbackGrab, Grab, InteractionState, OffsetGrab, PointerEvent
from airspace.interaction.resize_controller import WindowResizeGrab
from airspace.ui.dialogs import Overlay
from airspace.workspace.hud import Hud
from airspace.workspace.launcher import Launcher
from airspace.workspace.spatial_window import SpatialWindow
from airspace.workspace.window_manager import WindowManager

log = logging.getLogger(__name__)


class _ChromeButton:
    """Hover target for title-bar buttons (so dwell-click works on them)."""

    dwell_clickable = True

    def __init__(self, win: SpatialWindow, part: str) -> None:
        self.win, self.part = win, part

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ChromeButton) and other.win is self.win and other.part == self.part

    def __hash__(self) -> int:
        return hash((self.win.id, self.part))


class Workspace:
    def __init__(self, wm: WindowManager, launcher: Launcher, hud: Hud, toast: Callable[[str], None]) -> None:
        self.wm = wm
        self.launcher = launcher
        self.hud = hud
        self.toast = toast
        self.overlays: list[Overlay] = []
        self.bounds = QRectF(0, 0, 1280, 720)
        self._hover_win: SpatialWindow | None = None
        self._hover_layer: object | None = None
        self.momentum = True

    # ------------------------------------------------------------- overlays
    def push_overlay(self, overlay: Overlay) -> Overlay:
        overlay.layout(self.bounds)
        self.overlays.append(overlay)
        return overlay

    def top_overlay(self) -> Overlay | None:
        for ov in reversed(self.overlays):
            if not ov.closing:
                return ov
        return None

    def remove_overlay_type(self, cls: type) -> None:
        for ov in self.overlays:
            if isinstance(ov, cls):
                ov.dismiss()

    def has_overlay(self, cls: type) -> bool:
        return any(isinstance(o, cls) and not o.closing for o in self.overlays)

    def modal_active(self) -> bool:
        return self.top_overlay() is not None

    # --------------------------------------------------------------- layout
    def set_bounds(self, bounds: QRectF) -> None:
        self.bounds = QRectF(bounds)
        self.launcher.bounds = QRectF(bounds)
        for ov in self.overlays:
            ov.layout(self.bounds)

    def tick(self, dt: float) -> None:
        self.launcher.tick(dt)
        self.hud.tick(dt)
        for ov in list(self.overlays):
            ov.tick(dt)
            if not ov.visible:
                self.overlays.remove(ov)
        self.wm.tick(dt)

    # ---------------------------------------------------------------- press
    def press(self, ev: PointerEvent) -> Grab | None:
        hand = ev.is_hand
        ov = self.top_overlay()
        if ov is not None:
            return ov.press(ev)
        if self.launcher.is_open:
            if self.launcher.contains(ev.x, ev.y):
                return self.launcher.press(ev)
            self.launcher.close()
            return CallbackGrab(InteractionState.CLICK_PENDING)  # swallow: closing is the action
        if self.hud.hit(ev.x, ev.y, hand):
            return self.hud.press(ev)
        win = self.wm.window_at(ev.x, ev.y, hand)
        if win is not None:
            return self._press_window(win, ev)
        return None

    def _press_window(self, win: SpatialWindow, ev: PointerEvent) -> Grab | None:
        part = win.part_at(ev.x, ev.y, ev.is_hand)
        if part in ("close", "minimize"):
            self.wm.focus(win)
            rect = win.button_rects()[part]
            slop = 8.0 if ev.is_hand else 0.0

            def release(e: PointerEvent, part=part, rect=rect) -> None:
                if not e.cancelled and rect.contains(e.gx, e.gy, slop):
                    (self.wm.close if part == "close" else self.wm.minimize)(win)

            return CallbackGrab(InteractionState.CLICK_PENDING, None, release)
        if part == "titlebar":
            if ev.button == 2:
                return None
            return WindowDragGrab(self.wm, win, ev, momentum=self.momentum)
        if part and part.startswith("resize:"):
            return WindowResizeGrab(self.wm, win, part.split(":")[1], ev)
        if part == "content":
            self.wm.focus(win)
            c = win.content_rect
            try:
                grab = win.app.press(ev.offset(c.x, c.y))
            except Exception:  # noqa: BLE001
                log.exception("%s.press failed", win.app_type)
                grab = None
            if grab is not None:
                return OffsetGrab(grab, lambda w=win: (w.content_rect.x, w.content_rect.y))
            return CallbackGrab(InteractionState.CLICK_PENDING)
        return None

    # ---------------------------------------------------------------- hover
    def hover(self, ev: PointerEvent):
        hand = ev.is_hand
        target = None
        ov = self.top_overlay()
        if ov is not None:
            self._leave_except(ov)
            self._hover_layer = ov
            return ov.hover(ev)
        if self.launcher.is_open:
            self._leave_except(self.launcher)
            self._hover_layer = self.launcher
            return self.launcher.hover(ev.x, ev.y)
        if self.hud.hit(ev.x, ev.y, hand):
            self._leave_except(self.hud)
            self._hover_layer = self.hud
            return self.hud.hover(ev)
        win = self.wm.window_at(ev.x, ev.y, hand)
        if win is not self._hover_win:
            self._leave_window()
            self._hover_win = win
        self._leave_except(win)
        self._hover_layer = win
        if win is None:
            return None
        part = win.part_at(ev.x, ev.y, hand)
        win.hover_part = part
        if part == "content":
            c = win.content_rect
            try:
                win.app.hover(ev.offset(c.x, c.y))
            except Exception:  # noqa: BLE001
                log.exception("hover failed")
            target = win.app.controls.hovered
        else:
            win.app.leave()
            if part in ("close", "minimize"):
                target = _ChromeButton(win, part)
            else:
                target = win
        return target

    def _leave_window(self) -> None:
        if self._hover_win is not None:
            self._hover_win.hover_part = None
            try:
                self._hover_win.app.leave()
            except Exception:  # noqa: BLE001
                pass
            self._hover_win = None

    def _leave_except(self, keep: object) -> None:
        if self._hover_layer is keep:
            return
        layer = self._hover_layer
        if layer is self.hud:
            self.hud.leave()
        elif layer is self.launcher:
            self.launcher.hover_index = None
            self.launcher.hub_hover = False
        elif isinstance(layer, Overlay):
            layer.leave()
        if not isinstance(keep, SpatialWindow) or keep is not self._hover_win:
            self._leave_window()

    def leave(self) -> None:
        self._leave_except(None)
        self._hover_layer = None

    # ------------------------------------------------------- gesture targets
    def launcher_is_open(self) -> bool:
        return self.launcher.is_open

    def open_launcher(self, x: float, y: float) -> None:
        if not self.modal_active():
            self.launcher.open(x, y)

    def close_launcher(self) -> None:
        self.launcher.close()

    def swipe_at(self, x: float, y: float, direction: str) -> bool:
        win = self.wm.window_at(x, y, True) or self.wm.focused
        if win is None or not getattr(win.app, "accepts_swipe", False):
            return False
        try:
            return bool(win.app.swipe(direction))
        except Exception:  # noqa: BLE001
            log.exception("swipe failed")
            return False

    def feedback(self, message: str) -> None:
        self.toast(message)

    # ------------------------------------------------------------- keyboard
    def key(self, event: QKeyEvent) -> bool:
        ov = self.top_overlay()
        if ov is not None:
            return ov.key(event) or True  # modal: swallow keys
        if self.launcher.is_open and event.key() == Qt.Key.Key_Escape:
            self.launcher.close()
            return True
        win = self.wm.focused
        if win is not None:
            try:
                return bool(win.app.key(event))
            except Exception:  # noqa: BLE001
                log.exception("key handler failed")
        return False

    def wheel(self, x: float, y: float, delta: float, modifiers: int = 0) -> bool:
        ov = self.top_overlay()
        if ov is not None:
            return ov.wheel(x, y, delta)
        win = self.wm.window_at(x, y)
        if win is None:
            return False
        part = win.part_at(x, y)
        if part != "content":
            return False
        c = win.content_rect
        try:
            return bool(win.app.wheel(x - c.x, y - c.y, delta, modifiers))
        except Exception:  # noqa: BLE001
            log.exception("wheel failed")
            return False
