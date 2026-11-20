"""Base class and service context for mini-apps.

A mini-app draws into its window's content area (local coordinates, origin at
the top-left of the content) and reacts to device-independent pointer events.
Apps never touch the camera, the tracker or other windows directly; they use
the :class:`AppContext` services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtGui import QKeyEvent, QPainter

from airspace.interaction.events import Grab, PointerEvent
from airspace.ui.controls import ControlSet
from airspace.ui.styles import Theme

if TYPE_CHECKING:
    from airspace.app.context import AppContext


class MiniApp:
    app_type = "base"
    title = "app"
    icon = "apps"
    default_size = (520.0, 380.0)
    min_size = (300.0, 220.0)
    max_size: tuple[float, float] | None = None
    accepts_swipe = False
    single_instance = True

    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.controls = ControlSet()
        self.width = 0.0
        self.height = 0.0
        self.focused = False
        self._close_request: Callable[[], None] | None = None

    # -------------------------------------------------------------- layout
    @property
    def theme(self) -> Theme:
        return self.ctx.theme

    def resize(self, w: float, h: float) -> None:
        if (w, h) != (self.width, self.height):
            self.width, self.height = w, h
            self.layout(w, h)

    def relayout(self) -> None:
        self.layout(self.width, self.height)

    def layout(self, w: float, h: float) -> None:
        pass

    # ---------------------------------------------------------- rendering
    def tick(self, dt: float) -> None:
        self.controls.tick(dt)

    def paint(self, p: QPainter, theme: Theme) -> None:
        self.paint_content(p, theme)
        self.controls.paint(p, theme)

    def paint_content(self, p: QPainter, theme: Theme) -> None:
        pass

    # --------------------------------------------------------------- input
    def hover(self, ev: PointerEvent) -> None:
        self.controls.hover(ev.x, ev.y, ev.is_hand)

    def leave(self) -> None:
        self.controls.leave()

    def press(self, ev: PointerEvent) -> Grab | None:
        grab = self.controls.press(ev)
        if grab is not None:
            return grab
        return self.content_press(ev)

    def content_press(self, ev: PointerEvent) -> Grab | None:
        return None

    def wheel(self, x: float, y: float, delta: float, modifiers: int = 0) -> bool:
        return self.controls.wheel(x, y, delta)

    def key(self, event: QKeyEvent) -> bool:
        return False

    def swipe(self, direction: str) -> bool:
        return False

    # ----------------------------------------------------------- lifecycle
    def save_state(self) -> dict[str, Any]:
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        pass

    def on_focus(self, focused: bool) -> None:
        self.focused = focused

    def on_close(self) -> None:
        pass

    def request_close(self) -> None:
        if self._close_request:
            self._close_request()

    def status_text(self) -> str:
        """Optional short text shown in the title bar."""
        return ""
