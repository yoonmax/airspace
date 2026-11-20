"""Central ownership of interactions.

Mouse and hand pointers are routed through one manager. A press asks the
workspace for a :class:`Grab`; until release, *only* that grab receives
pointer motion from that source. Gesture events are routed by context
(see :mod:`gesture_router`). This prevents conflicting actions and stuck
states: every grab is released on pinch end, on tracking loss (after the
gesture engine's grace period), or cancelled explicitly.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from airspace.interaction.events import Grab, InteractionState, PointerEvent, TwoHandInfo
from airspace.vision.coordinate_mapper import CoordinateMapper
from airspace.vision.gesture_state_machine import GestureType
from airspace.vision.tracking import TrackedHand, TrackingFrame

log = logging.getLogger(__name__)


class InteractionRoot(Protocol):
    def press(self, ev: PointerEvent) -> Grab | None: ...
    def hover(self, ev: PointerEvent) -> Any: ...
    def leave(self) -> None: ...


@dataclass
class PointerState:
    x: float = 0.0
    y: float = 0.0
    visible: bool = False
    pressed: bool = False
    last_move: float = 0.0


@dataclass
class DwellState:
    target: Any = None
    since: float = 0.0
    x: float = 0.0
    y: float = 0.0
    fired: bool = False

    @property
    def active(self) -> bool:
        return self.target is not None and not self.fired


class InteractionManager:
    def __init__(self, root: InteractionRoot, clock: Callable[[], float] = time.perf_counter) -> None:
        self.root = root
        self.clock = clock
        self.grabs: dict[str, Grab] = {}
        self.pointers: dict[str, PointerState] = {"mouse": PointerState(), "hand": PointerState()}
        self.hover_source = "mouse"
        self.hand_grab_owner: int | None = None  # track id holding the hand grab
        self._two_hand: TwoHandInfo | None = None
        self._two_hand_owner: int | None = None
        self._cancelled_until = 0.0
        self.dwell = DwellState()
        self.dwell_enabled = False
        self.dwell_ms = 900.0
        self.hand_hover_target: Any = None
        self.router = None  # set by the application (GestureRouter)
        self.on_click_feedback: Callable[[str], None] | None = None
        self.last_gesture_latency_ms: float | None = None

    # ------------------------------------------------------------- state
    @property
    def state(self) -> InteractionState:
        grab = self.grabs.get("hand") or self.grabs.get("mouse")
        if grab is not None:
            return grab.kind
        if self.clock() < self._cancelled_until:
            return InteractionState.GESTURE_CANCELLED
        if self.router is not None and self.router.launcher_open():
            return InteractionState.LAUNCHER_OPEN
        if self.hand_hover_target is not None and self.pointers["hand"].visible:
            return InteractionState.HOVERING
        return InteractionState.IDLE

    def busy(self, source: str | None = None) -> bool:
        return bool(self.grabs) if source is None else source in self.grabs

    # ---------------------------------------------------------- pointers
    def _event(self, source: str, x: float, y: float, t: float, button: int = 1, modifiers: int = 0,
               cancelled: bool = False) -> PointerEvent:
        return PointerEvent(x, y, x, y, source, t, button, modifiers, cancelled)

    def pointer_move(self, source: str, x: float, y: float, t: float | None = None) -> None:
        t = self.clock() if t is None else t
        ps = self.pointers[source]
        ps.x, ps.y, ps.visible, ps.last_move = x, y, True, t
        ev = self._event(source, x, y, t)
        grab = self.grabs.get(source)
        if grab is not None:
            try:
                grab.move(ev)
            except Exception:  # noqa: BLE001
                log.exception("grab.move failed; releasing")
                self._drop(source)
            return
        if self.grabs and source == "hand" and "mouse" in self.grabs:
            return  # don't let hover churn under an active mouse drag
        self.hover_source = source
        target = self.root.hover(ev)
        if source == "hand":
            self.hand_hover_target = target
            self._update_dwell(target, x, y, t)

    def pointer_down(self, source: str, x: float, y: float, t: float | None = None, button: int = 1,
                     modifiers: int = 0) -> bool:
        t = self.clock() if t is None else t
        if source in self.grabs:  # should not happen; release defensively
            self.pointer_up(source, x, y, t, cancelled=True)
        ps = self.pointers[source]
        ps.x, ps.y, ps.pressed = x, y, True
        ev = self._event(source, x, y, t, button, modifiers)
        try:
            grab = self.root.press(ev)
        except Exception:  # noqa: BLE001
            log.exception("press failed")
            grab = None
        if grab is not None:
            self.grabs[source] = grab
            if self.on_click_feedback:
                self.on_click_feedback("press")
        return grab is not None

    def pointer_up(self, source: str, x: float, y: float, t: float | None = None, cancelled: bool = False) -> None:
        t = self.clock() if t is None else t
        ps = self.pointers[source]
        ps.pressed = False
        grab = self.grabs.pop(source, None)
        if source == "hand":
            self.hand_grab_owner = None
            self._end_two_hand(grab)
        if grab is None:
            return
        try:
            grab.release(self._event(source, x, y, t, cancelled=cancelled))
        except Exception:  # noqa: BLE001
            log.exception("grab.release failed")
        # Refresh hover after release so highlight state is accurate (unless the
        # release came from tracking loss — then there is no pointer to show).
        if not cancelled:
            self.pointer_move(source, x, y, t)

    def cancel(self, source: str | None = None) -> bool:
        """Abort active interactions without committing (fist / Escape)."""
        sources = [source] if source else list(self.grabs)
        did = False
        for s in sources:
            grab = self.grabs.pop(s, None)
            if grab is None:
                continue
            if s == "hand":
                self._end_two_hand(grab)
                self.hand_grab_owner = None
            try:
                grab.cancel()
            except Exception:  # noqa: BLE001
                log.exception("grab.cancel failed")
            did = True
        if did:
            self._cancelled_until = self.clock() + 0.6
        return did

    def _drop(self, source: str) -> None:
        self.grabs.pop(source, None)
        if source == "hand":
            self.hand_grab_owner = None
            self._two_hand = None

    def hide_hand(self) -> None:
        self.pointers["hand"].visible = False
        if self.hand_hover_target is not None and self.hover_source == "hand":
            self.root.leave()
        self.hand_hover_target = None
        self.dwell = DwellState()

    # ---------------------------------------------------- hand tracking
    def handle_tracking(self, frame: TrackingFrame, mapper: CoordinateMapper, capture_ts: float | None = None) -> None:
        t = frame.t
        primary = frame.primary
        # 1. Pointer motion for the primary hand.
        if primary is not None and primary.present and primary.pointer is not None:
            x, y = mapper.pointer_to_view(float(primary.pointer[0]), float(primary.pointer[1]))
            self.pointer_move("hand", x, y, t)
        elif primary is None or not primary.present:
            if "hand" not in self.grabs:
                self.hide_hand()

        # 2. Discrete gesture events.
        for ev in frame.events:
            is_primary = primary is not None and ev.hand_id == primary.id
            if ev.type == GestureType.PINCH_START:
                if is_primary and "hand" not in self.grabs:
                    hp = self.pointers["hand"]
                    if self.pointer_down("hand", hp.x, hp.y, t):
                        self.hand_grab_owner = ev.hand_id
                    if capture_ts is not None:
                        self.last_gesture_latency_ms = (time.perf_counter() - capture_ts) * 1000
                elif not is_primary and "hand" in self.grabs:
                    self._begin_two_hand(frame, mapper, ev.hand_id)
            elif ev.type == GestureType.PINCH_END:
                if ev.hand_id == self.hand_grab_owner:
                    hp = self.pointers["hand"]
                    self.pointer_up("hand", hp.x, hp.y, t, cancelled=ev.cancelled)
                elif ev.hand_id == self._two_hand_owner:
                    self._end_two_hand(self.grabs.get("hand"))
            elif self.router is not None:
                self.router.gesture(ev, frame, mapper, is_primary)

        # 3. Continuous two-hand update.
        if self._two_hand is not None:
            info = self._two_hand_info(frame, mapper)
            grab = self.grabs.get("hand")
            if info is None or grab is None:
                self._end_two_hand(grab)
            else:
                try:
                    grab.two_hand_update(self._two_hand, info)
                except Exception:  # noqa: BLE001
                    log.exception("two-hand update failed")
                    self._end_two_hand(grab)

    def _two_hand_info(self, frame: TrackingFrame, mapper: CoordinateMapper) -> TwoHandInfo | None:
        hands = [h for h in frame.hands if h.present and h.pointer is not None]
        if len(hands) < 2:
            return None
        a = next((h for h in hands if h.id == self.hand_grab_owner), None)
        b = next((h for h in hands if h.id == self._two_hand_owner), None)
        if a is None or b is None:
            return None
        return two_hand_info(a, b, mapper)

    def _begin_two_hand(self, frame: TrackingFrame, mapper: CoordinateMapper, other_id: int) -> None:
        grab = self.grabs.get("hand")
        if grab is None or not grab.supports_two_hand or self._two_hand is not None:
            return
        self._two_hand_owner = other_id
        info = self._two_hand_info(frame, mapper)
        if info is None:
            self._two_hand_owner = None
            return
        if grab.two_hand_begin(info):
            self._two_hand = info
        else:
            self._two_hand_owner = None

    def _end_two_hand(self, grab: Grab | None) -> None:
        if self._two_hand is not None and grab is not None:
            try:
                grab.two_hand_end()
            except Exception:  # noqa: BLE001
                log.exception("two_hand_end failed")
        self._two_hand = None
        self._two_hand_owner = None

    @property
    def two_hand_active(self) -> bool:
        return self._two_hand is not None

    # ------------------------------------------------------------- dwell
    def _update_dwell(self, target: Any, x: float, y: float, t: float) -> None:
        if not self.dwell_enabled or "hand" in self.grabs:
            self.dwell = DwellState()
            return
        clickable = target is not None and getattr(target, "dwell_clickable", False)
        d = self.dwell
        if not clickable:
            self.dwell = DwellState()
            return
        if target is not d.target or math.hypot(x - d.x, y - d.y) > 18:
            if target is d.target and d.fired:
                return  # must leave the control before it can fire again
            self.dwell = DwellState(target, t, x, y)
            return
        if not d.fired and (t - d.since) * 1000 >= self.dwell_ms:
            d.fired = True
            if self.pointer_down("hand", x, y, t):
                self.pointer_up("hand", x, y, t)

    @property
    def dwell_progress(self) -> float:
        d = self.dwell
        if not d.active:
            return 0.0
        return min(1.0, (self.clock() - d.since) * 1000 / self.dwell_ms)


def two_hand_info(a: TrackedHand, b: TrackedHand, mapper: CoordinateMapper) -> TwoHandInfo:
    ax, ay = mapper.pointer_to_view(float(a.pointer[0]), float(a.pointer[1]), clamp=False)
    bx, by = mapper.pointer_to_view(float(b.pointer[0]), float(b.pointer[1]), clamp=False)
    return TwoHandInfo((ax + bx) / 2, (ay + by) / 2, math.hypot(bx - ax, by - ay), math.atan2(by - ay, bx - ax))
