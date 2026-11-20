"""Webcam capture on a dedicated thread.

The capture thread only reads frames and publishes them into a
:class:`LatestFrameBuffer`. It never touches Qt widgets. Temporary read
failures are retried; a sustained failure is reported as a disconnect and the
thread keeps trying to reopen the device with backoff.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import cv2

from airspace.camera.frame_buffer import Frame, LatestFrameBuffer

log = logging.getLogger(__name__)


class CameraState(str, Enum):
    STOPPED = "stopped"
    OPENING = "opening"
    RUNNING = "running"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass(frozen=True)
class CameraStatus:
    state: CameraState
    message: str = ""
    width: int = 0
    height: int = 0
    device_index: int = 0


def _backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY


def open_capture(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)  # fall back to OpenCV's default backend
        if not cap.isOpened():
            cap.release()
            return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise driver-side queueing where supported
    except cv2.error:
        pass
    return cap


class CaptureWorker:
    """Owns one ``cv2.VideoCapture`` and a reader thread."""

    MAX_CONSECUTIVE_FAILURES = 25
    FIRST_FRAME_TIMEOUT = 6.0

    def __init__(
        self,
        buffer: LatestFrameBuffer,
        device_index: int,
        size: tuple[int, int],
        fps: int,
        on_status: Callable[[CameraStatus], None],
    ) -> None:
        self.buffer = buffer
        self.device_index = device_index
        self.size = size
        self.fps = fps
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._times: deque[float] = deque(maxlen=60)
        self.actual_size = (0, 0)

    # ------------------------------------------------------------ control
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="airspace-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        self._thread = None

    @property
    def fps_measured(self) -> float:
        times = list(self._times)
        if len(times) < 2 or times[-1] - times[0] <= 0:
            return 0.0
        if time.perf_counter() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    # ------------------------------------------------------------- thread
    def _emit(self, state: CameraState, message: str = "") -> None:
        w, h = self.actual_size
        self._on_status(CameraStatus(state, message, w, h, self.device_index))

    def _run(self) -> None:
        backoff = 0.5
        first_attempt = True
        while not self._stop.is_set():
            self._emit(CameraState.OPENING if first_attempt else CameraState.RECOVERING,
                       f"Opening camera {self.device_index}…")
            cap = open_capture(self.device_index, self.size[0], self.size[1], self.fps)
            if cap is None:
                self._emit(
                    CameraState.FAILED,
                    "The camera could not be opened. It may be in use by another app, "
                    "disconnected, or camera access may not be granted.",
                )
                if self._stop.wait(min(backoff, 5.0)):
                    break
                backoff *= 1.6
                first_attempt = False
                continue
            try:
                ran = self._read_loop(cap)
            finally:
                cap.release()
            if ran:
                backoff = 0.5
            first_attempt = False
            if not self._stop.is_set():
                self._stop.wait(min(backoff, 5.0))
                backoff *= 1.6
        self._emit(CameraState.STOPPED, "Camera stopped")

    def _read_loop(self, cap: cv2.VideoCapture) -> bool:
        """Read until stopped or the device fails. Returns True if any frame arrived."""
        failures = 0
        got_any = False
        started = time.perf_counter()
        while not self._stop.is_set():
            ok, image = cap.read()
            now = time.perf_counter()
            if not ok or image is None or image.size == 0:
                failures += 1
                if not got_any and now - started > self.FIRST_FRAME_TIMEOUT:
                    self._emit(
                        CameraState.FAILED,
                        "The camera opened but delivered no frames. On macOS, check "
                        "System Settings → Privacy & Security → Camera.",
                    )
                    return False
                if got_any and failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self._emit(CameraState.RECOVERING, "Camera stopped responding — reconnecting…")
                    return True
                time.sleep(0.02)
                continue
            if not got_any:
                got_any = True
                self.actual_size = (image.shape[1], image.shape[0])
                self._emit(CameraState.RUNNING, "Camera running")
            elif failures:
                self._emit(CameraState.RUNNING, "Camera running")
            failures = 0
            self._times.append(now)
            self.buffer.put(Frame(image=image, timestamp=now, index=self.buffer.next_index()))
        return got_any
