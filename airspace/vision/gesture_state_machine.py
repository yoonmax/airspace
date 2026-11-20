"""Temporal gesture recognition.

Each gesture is an explicit state machine with entry conditions, hysteresis,
minimum durations, cooldowns and tracking-loss handling. A single noisy frame
can never produce an event.

Pinch                       ratio < enter          held >= min_ms & frames
    OPEN ───────────────▶ PRESSING ───────────────────────────────▶ PINCHED ─┐
      ▲                       │ ratio > exit                         │       │ held >= hold_ms
      │                       ▼                                      │       ▼ (PINCH_HOLD once)
      │                     OPEN                    ratio > exit     │
      └──────────── RELEASING ◀──────────────────────────────────────┘
        held >= release_ms     │ ratio < exit (bounce)
                               └──────────▶ PINCHED (no event)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from airspace.vision.gesture_classifier import HandFeatures, Pose


class GestureType(str, Enum):
    PINCH_START = "pinch_start"
    PINCH_HOLD = "pinch_hold"
    PINCH_END = "pinch_end"
    PALM_OPEN = "palm_open"
    FIST = "fist"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    TRACKING_LOST = "tracking_lost"


@dataclass(frozen=True)
class GestureEvent:
    type: GestureType
    hand_id: int
    t: float
    cancelled: bool = False  # True when a pinch ended because tracking was lost
    data: Any = None


@dataclass
class GestureConfig:
    pinch_enter: float = 0.30
    pinch_exit: float = 0.45
    pinch_min_ms: float = 25.0
    pinch_min_frames: int = 2
    release_min_ms: float = 25.0
    release_min_frames: int = 2
    pinch_debounce_ms: float = 90.0  # ignore a new press this soon after a release
    pinch_hold_ms: float = 260.0
    palm_hold_ms: float = 380.0
    palm_rearm_ms: float = 250.0
    palm_cooldown_ms: float = 900.0
    palm_max_speed: float = 0.9  # palm-scales per second; a moving hand is not "presenting"
    fist_hold_ms: float = 300.0
    fist_cooldown_ms: float = 700.0
    swipe_window_ms: float = 320.0
    swipe_min_distance: float = 0.20  # normalised frame width
    swipe_axis_ratio: float = 2.0
    swipe_cooldown_ms: float = 700.0
    loss_grace_ms: float = 500.0  # live sessions showed brief (<0.5 s) tracker dropouts mid-pinch


class PinchState(str, Enum):
    OPEN = "open"
    PRESSING = "pressing"
    PINCHED = "pinched"
    RELEASING = "releasing"


@dataclass
class _Pinch:
    state: PinchState = PinchState.OPEN
    since: float = 0.0
    frames: int = 0
    held_emitted: bool = False
    last_release: float = -1e9
    release_started: float = 0.0


@dataclass
class _Hold:
    """Generic stable-pose detector with re-arm and cooldown."""

    since: float | None = None
    armed: bool = True
    absent_since: float | None = None
    last_fire: float = -1e9


@dataclass
class HandGestureMachine:
    hand_id: int
    config: GestureConfig = field(default_factory=GestureConfig)

    def __post_init__(self) -> None:
        self.pinch = _Pinch()
        self.palm = _Hold()
        self.fist = _Hold()
        self._palm_hist: deque[tuple[float, np.ndarray]] = deque()
        self._last_swipe = -1e9
        self._lost_since: float | None = None
        self.last_features: HandFeatures | None = None
        self.palm_progress = 0.0  # 0..1 while an open palm is being held (for UI feedback)

    # ----------------------------------------------------------------- api
    @property
    def pinched(self) -> bool:
        return self.pinch.state in (PinchState.PINCHED, PinchState.RELEASING)

    @property
    def pinch_progress(self) -> float:
        """0 → open, 1 → at the enter threshold. Used for the cursor's activation ring."""
        f = self.last_features
        if f is None:
            return 0.0
        c = self.config
        span = max(c.pinch_exit * 2.2 - c.pinch_enter, 1e-3)
        return float(np.clip((c.pinch_exit * 2.2 - f.pinch_ratio) / span, 0.0, 1.0))

    def update(self, f: HandFeatures, t: float) -> list[GestureEvent]:
        """Feed one frame of features (``t`` in seconds). Returns emitted events."""
        self._lost_since = None
        self.last_features = f
        events: list[GestureEvent] = []
        events += self._update_pinch(f, t)
        events += self._update_palm(f, t)
        events += self._update_fist(f, t)
        events += self._update_swipe(f, t)
        return events

    def lost(self, t: float) -> list[GestureEvent]:
        """Call each frame the hand is missing. Releases a held pinch after a grace period."""
        if self._lost_since is None:
            self._lost_since = t
        self.palm.since = self.fist.since = None
        self.palm_progress = 0.0
        self._palm_hist.clear()
        if self.pinched and (t - self._lost_since) * 1000 >= self.config.loss_grace_ms:
            self.pinch = _Pinch(last_release=t)
            return [GestureEvent(GestureType.PINCH_END, self.hand_id, t, cancelled=True)]
        if self.pinch.state == PinchState.PRESSING:
            self.pinch.state = PinchState.OPEN
        return []

    def force_release(self, t: float) -> list[GestureEvent]:
        if self.pinched:
            self.pinch = _Pinch(last_release=t)
            return [GestureEvent(GestureType.PINCH_END, self.hand_id, t, cancelled=True)]
        return []

    # ------------------------------------------------------------- pinch
    def _update_pinch(self, f: HandFeatures, t: float) -> list[GestureEvent]:
        c, s = self.config, self.pinch
        r = f.pinch_ratio
        # A fist brings the thumb next to the index finger; an index tip tucked into
        # the palm is never a pinch.
        valid = f.tip_to_palm[1] > 0.6 or r < c.pinch_enter * 0.5
        ms = lambda: (t - s.since) * 1000.0  # noqa: E731
        out: list[GestureEvent] = []
        if s.state == PinchState.OPEN:
            if r < c.pinch_enter and valid and (t - s.last_release) * 1000 >= c.pinch_debounce_ms:
                s.state, s.since, s.frames = PinchState.PRESSING, t, 1
        elif s.state == PinchState.PRESSING:
            if r > c.pinch_exit or not valid:
                s.state = PinchState.OPEN
            else:
                s.frames += 1
                if ms() >= c.pinch_min_ms and s.frames >= c.pinch_min_frames:
                    s.state, s.since, s.held_emitted = PinchState.PINCHED, t, False
                    out.append(GestureEvent(GestureType.PINCH_START, self.hand_id, t))
        elif s.state == PinchState.PINCHED:
            if r > c.pinch_exit:
                s.state, s.frames, s.release_started = PinchState.RELEASING, 1, t
            elif not s.held_emitted and ms() >= c.pinch_hold_ms:
                s.held_emitted = True
                out.append(GestureEvent(GestureType.PINCH_HOLD, self.hand_id, t))
        elif s.state == PinchState.RELEASING:
            started = s.release_started
            if r <= c.pinch_exit:
                s.state = PinchState.PINCHED  # bounce: stay pinched, no event
            else:
                s.frames += 1
                if (t - started) * 1000 >= c.release_min_ms and s.frames >= c.release_min_frames:
                    self.pinch = _Pinch(last_release=t)
                    out.append(GestureEvent(GestureType.PINCH_END, self.hand_id, t))
        return out

    # -------------------------------------------------------- open palm
    def _palm_speed(self, f: HandFeatures, t: float) -> float:
        hist = self._palm_hist
        hist.append((t, f.palm_center.copy()))
        while hist and t - hist[0][0] > 0.45:
            hist.popleft()
        if len(hist) < 2:
            return 0.0
        (t0, p0) = hist[max(0, len(hist) - 5)]
        dt = max(t - t0, 1e-3)
        return float(np.linalg.norm(f.palm_center - p0) / dt / max(f.palm_scale, 1e-3))

    def _update_palm(self, f: HandFeatures, t: float) -> list[GestureEvent]:
        c, h = self.config, self.palm
        speed = self._palm_speed(f, t)
        is_palm = f.pose == Pose.OPEN_PALM and not self.pinched and speed < c.palm_max_speed
        return self._update_hold(h, is_palm, t, c.palm_hold_ms, c.palm_rearm_ms,
                                 c.palm_cooldown_ms, GestureType.PALM_OPEN, progress_attr="palm_progress")

    def _update_fist(self, f: HandFeatures, t: float) -> list[GestureEvent]:
        c = self.config
        is_fist = f.pose == Pose.FIST and not self.pinched and self.pinch.state != PinchState.PRESSING
        return self._update_hold(self.fist, is_fist, t, c.fist_hold_ms, 200.0,
                                 c.fist_cooldown_ms, GestureType.FIST)

    def _update_hold(self, h: _Hold, active: bool, t: float, hold_ms: float, rearm_ms: float,
                     cooldown_ms: float, kind: GestureType, progress_attr: str | None = None) -> list[GestureEvent]:
        out: list[GestureEvent] = []
        if active:
            h.absent_since = None
            if h.since is None:
                h.since = t
            held = (t - h.since) * 1000
            if progress_attr and h.armed:
                setattr(self, progress_attr, min(held / hold_ms, 1.0))
            if h.armed and held >= hold_ms and (t - h.last_fire) * 1000 >= cooldown_ms:
                h.armed = False
                h.last_fire = t
                out.append(GestureEvent(kind, self.hand_id, t))
        else:
            h.since = None
            if progress_attr:
                setattr(self, progress_attr, 0.0)
            if not h.armed:
                if h.absent_since is None:
                    h.absent_since = t
                elif (t - h.absent_since) * 1000 >= rearm_ms:
                    h.armed = True
        if not h.armed and progress_attr:
            setattr(self, progress_attr, 0.0)
        return out

    # ------------------------------------------------------------ swipe
    def _update_swipe(self, f: HandFeatures, t: float) -> list[GestureEvent]:
        c = self.config
        hist = self._palm_hist  # filled by _palm_speed this frame
        if self.pinched or f.extended_count < 3 or (t - self._last_swipe) * 1000 < c.swipe_cooldown_ms:
            return []
        window = [(ts, p) for ts, p in hist if (t - ts) * 1000 <= c.swipe_window_ms]
        if len(window) < 3:
            return []
        d = window[-1][1] - window[0][1]
        dx, dy = float(d[0]), float(d[1])
        kind: GestureType | None = None
        if abs(dx) >= c.swipe_min_distance and abs(dx) >= c.swipe_axis_ratio * abs(dy):
            kind = GestureType.SWIPE_RIGHT if dx > 0 else GestureType.SWIPE_LEFT
        elif abs(dy) >= c.swipe_min_distance * 0.9 and abs(dy) >= c.swipe_axis_ratio * abs(dx):
            kind = GestureType.SWIPE_DOWN if dy > 0 else GestureType.SWIPE_UP
        if kind is None:
            return []
        self._last_swipe = t
        hist.clear()
        # A swipe is deliberate motion: cancel any palm hold in progress.
        self.palm.since = None
        self.palm_progress = 0.0
        return [GestureEvent(kind, self.hand_id, t, data=(dx, dy))]
