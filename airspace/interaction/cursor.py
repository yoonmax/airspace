"""Fingertip pointer that stays put through a pinch.

The cursor sits on the index fingertip whenever the hand is not pinching, so
what you see on the camera backdrop and where the cursor is always agree.

Closing a pinch moves the fingertip towards the thumb, which would make a
naive cursor slide off small targets at the moment of the click. So when a
pinch starts, the pointer re-anchors to the index knuckle (which barely moves
while the fingers close) with the fingertip's offset *as it was just before
the fingers began closing*. Using the finger's position relative to the hand,
rather than an old absolute position, means a moving hand does not leave the
click behind. When the pinch ends the offset glides back so the cursor
returns to the fingertip without a jump.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from airspace.vision.filters import OneEuroFilter, OneEuroParams
from airspace.vision.gesture_classifier import HandFeatures


class PinchStablePointer:
    GLIDE_TAU = 0.08  # seconds for the anchor offset to decay after a release
    REWIND_S = 0.12  # how far back "before the fingers began closing" is
    HISTORY_S = 0.4

    def __init__(self, params: OneEuroParams) -> None:
        self.filter = OneEuroFilter(params)
        self._offset = np.zeros(2)
        self._mode = "tip"
        self._last_raw: np.ndarray | None = None
        self._last_t: float | None = None
        self._rel_history: deque[tuple[float, np.ndarray]] = deque()  # (t, tip - knuckle)
        self.value: np.ndarray | None = None
        self.velocity = np.zeros(2)  # normalised units / s (filtered)

    def set_params(self, params: OneEuroParams) -> None:
        self.filter.params = params

    def reset(self) -> None:
        self.filter.reset()
        self._offset[:] = 0
        self._mode = "tip"
        self._last_raw = None
        self._last_t = None
        self._rel_history.clear()
        self.value = None
        self.velocity[:] = 0

    def _relative_before(self, t: float) -> np.ndarray | None:
        """Fingertip-minus-knuckle at about ``t - REWIND_S`` (oldest sample if history is short)."""
        target = t - self.REWIND_S
        best = None
        for ts, rel in self._rel_history:
            if ts <= target:
                best = rel
            else:
                break
        if best is None and self._rel_history:
            best = self._rel_history[0][1]
        return best

    def update(self, f: HandFeatures, pinch_engaged: bool, t: float) -> np.ndarray:
        mode = "knuckle" if pinch_engaged else "tip"
        snap = False
        if mode == "knuckle" and self._mode == "tip":
            rel = self._relative_before(t)
            self._offset = rel.copy() if rel is not None else f.index_tip - f.knuckle
            snap = True  # the click must land exactly on the pre-closure fingertip
        elif mode == "tip" and self._mode == "knuckle" and self._last_raw is not None:
            self._offset = self._last_raw - f.index_tip
        elif mode == "tip" and self._last_t is not None:
            dt = max(t - self._last_t, 0.0)
            self._offset *= math.exp(-dt / self.GLIDE_TAU)
        if mode == "tip":
            self._rel_history.append((t, f.index_tip - f.knuckle))
            while self._rel_history and t - self._rel_history[0][0] > self.HISTORY_S:
                self._rel_history.popleft()
        self._mode = mode
        anchor = f.knuckle if mode == "knuckle" else f.index_tip
        raw = anchor + self._offset
        self._last_raw = raw
        prev = self.value
        if snap:
            self.filter.reset()
        self.value = self.filter(raw, t)
        if prev is not None and self._last_t is not None and t > self._last_t and not snap:
            self.velocity = 0.7 * self.velocity + 0.3 * (self.value - prev) / (t - self._last_t)
        self._last_t = t
        return self.value

    @property
    def mode(self) -> str:
        return self._mode
