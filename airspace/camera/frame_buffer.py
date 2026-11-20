"""A latest-frame buffer.

The capture thread overwrites the slot; consumers always receive the newest
frame. If processing falls behind, old frames are dropped instead of queueing,
which keeps end-to-end latency bounded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, unmirrored, as delivered by the camera
    timestamp: float  # time.perf_counter() when the frame was read
    index: int


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._dropped = 0
        self._consumed_index = -1
        self._closed = False
        self._counter = 0

    def next_index(self) -> int:
        """Monotonic frame index shared by every capture session using this buffer."""
        with self._cond:
            self._counter += 1
            return self._counter

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None and self._frame.index != self._consumed_index:
                self._dropped += 1
            self._frame = frame
            self._cond.notify_all()

    def get_newer(self, last_index: int, timeout: float = 0.1) -> Frame | None:
        """Return the newest frame with ``index > last_index`` or ``None`` on timeout."""
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._closed or (self._frame is not None and self._frame.index > last_index),
                timeout=timeout,
            ):
                return None
            if self._closed:
                return None
            frame = self._frame
            assert frame is not None
            self._consumed_index = frame.index
            return frame

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def reopen(self) -> None:
        with self._cond:
            self._closed = False
            self._frame = None
