"""Device-independent pointer events and the *grab* protocol.

Mouse and hand input produce the same :class:`PointerEvent`. Pressing on a
target returns a :class:`Grab` that owns the interaction until release — this
is how airspace guarantees that one gesture drives exactly one action (a
window drag never also clicks a button or draws a stroke).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class InteractionState(str, Enum):
    IDLE = "idle"
    HOVERING = "hovering"
    PINCH_STARTED = "pinch started"
    CLICK_PENDING = "click pending"
    DRAGGING_WINDOW = "dragging window"
    RESIZING_WINDOW = "resizing window"
    DRAGGING_SLIDER = "adjusting"
    DRAWING = "drawing"
    MANIPULATING_OBJECT = "manipulating"
    PANNING = "panning"
    LAUNCHER_OPEN = "launcher"
    GESTURE_CANCELLED = "cancelled"


@dataclass(frozen=True)
class PointerEvent:
    x: float  # local coordinates of the receiving target
    y: float
    gx: float  # workspace coordinates
    gy: float
    source: str  # "mouse" | "hand"
    t: float
    button: int = 1  # 1 primary, 2 secondary, 3 middle
    modifiers: int = 0
    cancelled: bool = False  # release caused by tracking loss / cancellation

    def at(self, x: float, y: float) -> "PointerEvent":
        return replace(self, x=x, y=y)

    def offset(self, ox: float, oy: float) -> "PointerEvent":
        return replace(self, x=self.gx - ox, y=self.gy - oy)

    @property
    def is_hand(self) -> bool:
        return self.source == "hand"


@dataclass(frozen=True)
class TwoHandInfo:
    """Geometry of a two-hand gesture in workspace coordinates."""

    cx: float
    cy: float
    distance: float
    angle: float  # radians


class Grab:
    """An interaction in progress, owned by the target that was pressed."""

    kind: InteractionState = InteractionState.CLICK_PENDING
    supports_two_hand: bool = False

    def move(self, ev: PointerEvent) -> None:
        pass

    def release(self, ev: PointerEvent) -> None:
        pass

    def cancel(self) -> None:
        """Abort without committing (e.g. a fist gesture)."""

    def two_hand_begin(self, info: TwoHandInfo) -> bool:
        return False

    def two_hand_update(self, start: TwoHandInfo, now: TwoHandInfo) -> None:
        pass

    def two_hand_end(self) -> None:
        pass


class CallbackGrab(Grab):
    """A grab built from callables, convenient for controls."""

    def __init__(self, kind: InteractionState, on_move=None, on_release=None, on_cancel=None) -> None:
        self.kind = kind
        self._move, self._release, self._cancel = on_move, on_release, on_cancel

    def move(self, ev: PointerEvent) -> None:
        if self._move:
            self._move(ev)

    def release(self, ev: PointerEvent) -> None:
        if self._release:
            self._release(ev)

    def cancel(self) -> None:
        if self._cancel:
            self._cancel()


class OffsetGrab(Grab):
    """Wraps a grab created in local coordinates, translating later events.

    ``origin`` returns the current workspace position of the local origin, so
    the translation stays correct even if the container moves.
    """

    def __init__(self, inner: Grab, origin) -> None:
        self.inner = inner
        self.origin = origin
        self.kind = inner.kind
        self.supports_two_hand = inner.supports_two_hand

    def _local(self, ev: PointerEvent) -> PointerEvent:
        ox, oy = self.origin()
        return ev.offset(ox, oy)

    def move(self, ev: PointerEvent) -> None:
        self.inner.move(self._local(ev))

    def release(self, ev: PointerEvent) -> None:
        self.inner.release(self._local(ev))

    def cancel(self) -> None:
        self.inner.cancel()

    def two_hand_begin(self, info: TwoHandInfo) -> bool:
        ox, oy = self.origin()
        return self.inner.two_hand_begin(TwoHandInfo(info.cx - ox, info.cy - oy, info.distance, info.angle))

    def two_hand_update(self, start: TwoHandInfo, now: TwoHandInfo) -> None:
        ox, oy = self.origin()
        self.inner.two_hand_update(
            TwoHandInfo(start.cx - ox, start.cy - oy, start.distance, start.angle),
            TwoHandInfo(now.cx - ox, now.cy - oy, now.distance, now.angle),
        )

    def two_hand_end(self) -> None:
        self.inner.two_hand_end()
