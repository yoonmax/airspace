"""Context-dependent routing of discrete gestures.

Gestures are not global: an open palm only opens the launcher when nothing
else is happening; a swipe only reaches a window that accepts swipes and only
when no drag is in progress; a fist closes the launcher or cancels the
current interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from airspace.vision.coordinate_mapper import CoordinateMapper
from airspace.vision.gesture_state_machine import GestureEvent, GestureType
from airspace.vision.tracking import TrackingFrame

if TYPE_CHECKING:
    from airspace.interaction.interaction_manager import InteractionManager


class GestureTargets(Protocol):
    def launcher_is_open(self) -> bool: ...
    def open_launcher(self, x: float, y: float) -> None: ...
    def close_launcher(self) -> None: ...
    def swipe_at(self, x: float, y: float, direction: str) -> bool: ...
    def modal_active(self) -> bool: ...
    def feedback(self, message: str) -> None: ...


SWIPES = {
    GestureType.SWIPE_LEFT: "left",
    GestureType.SWIPE_RIGHT: "right",
    GestureType.SWIPE_UP: "up",
    GestureType.SWIPE_DOWN: "down",
}


class GestureRouter:
    def __init__(self, manager: "InteractionManager", targets: GestureTargets) -> None:
        self.manager = manager
        self.targets = targets
        manager.router = self
        self.last_routed: str = ""

    def launcher_open(self) -> bool:
        return self.targets.launcher_is_open()

    def gesture(self, ev: GestureEvent, frame: TrackingFrame, mapper: CoordinateMapper, is_primary: bool) -> None:
        m, tg = self.manager, self.targets
        if ev.type == GestureType.PALM_OPEN:
            if m.busy("hand") or tg.modal_active() or tg.launcher_is_open():
                return  # holding a palm while it's already open must not toggle it closed
            # Hovering a control with a relaxed open hand is pointing, not a request for the launcher.
            target = m.hand_hover_target
            if target is not None and getattr(target, "dwell_clickable", False) or _is_control(target):
                return
            hand = next((h for h in frame.hands if h.id == ev.hand_id and h.features is not None), None)
            if hand is None:
                return
            x, y = mapper.norm_to_view(float(hand.features.palm_center[0]), float(hand.features.palm_center[1]))
            tg.open_launcher(x, y)
            self.last_routed = "launcher opened"
        elif ev.type == GestureType.FIST:
            if tg.launcher_is_open():
                tg.close_launcher()
                self.last_routed = "launcher closed"
            elif m.cancel("hand"):
                tg.feedback("Interaction cancelled")
                self.last_routed = "cancelled"
        elif ev.type in SWIPES:
            if not is_primary or m.busy() or tg.launcher_is_open() or tg.modal_active():
                return
            hp = m.pointers["hand"]
            if tg.swipe_at(hp.x, hp.y, SWIPES[ev.type]):
                self.last_routed = f"swipe {SWIPES[ev.type]}"


def _is_control(target) -> bool:
    from airspace.ui.controls import Control

    return isinstance(target, Control)
