"""Camera selection, lifecycle and status reporting for the UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from airspace.camera.capture import CameraState, CameraStatus, CaptureWorker
from airspace.camera.frame_buffer import LatestFrameBuffer
from airspace.persistence.settings import CAPTURE_PROFILES

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str


def enumerate_cameras() -> list[CameraInfo]:
    """List cameras without opening them.

    Names come from Qt Multimedia (which does not open the device). OpenCV
    addresses cameras by index; on macOS and Windows the platform ordering
    normally matches, but the mapping is best-effort, so the index is always
    shown alongside the name.
    """
    try:
        from PySide6.QtMultimedia import QMediaDevices

        devices = QMediaDevices.videoInputs()
        cams = [CameraInfo(i, d.description() or f"Camera {i}") for i, d in enumerate(devices)]
        if cams:
            return cams
    except Exception as exc:  # noqa: BLE001 - multimedia backend may be unavailable
        log.info("Qt camera enumeration unavailable: %s", exc)
    return [CameraInfo(0, "Camera 0")]


class CameraManager(QObject):
    """Owns the capture worker; status changes are delivered as Qt signals."""

    statusChanged = Signal(object)  # CameraStatus

    def __init__(self, buffer: LatestFrameBuffer) -> None:
        super().__init__()
        self.buffer = buffer
        self._worker: CaptureWorker | None = None
        self.status = CameraStatus(CameraState.STOPPED, "Camera off")
        self.device_index = 0
        self.profile = "balanced"
        # Receiver lives in the UI thread, so emissions from the capture
        # thread are queued and `status` is only ever written on the UI thread.
        self.statusChanged.connect(self._store_status)

    def _store_status(self, status: CameraStatus) -> None:
        self.status = status

    @property
    def running(self) -> bool:
        return self._worker is not None

    @property
    def fps(self) -> float:
        return self._worker.fps_measured if self._worker else 0.0

    def start(self, device_index: int, profile: str) -> None:
        self.stop()
        self.device_index = device_index
        self.profile = profile if profile in CAPTURE_PROFILES else "balanced"
        w, h, fps = CAPTURE_PROFILES[self.profile]
        self.buffer.reopen()
        self._worker = CaptureWorker(self.buffer, device_index, (w, h), fps, self._on_status)
        self._worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._set_status(CameraStatus(CameraState.STOPPED, "Camera off", device_index=self.device_index))

    def restart(self) -> None:
        self.start(self.device_index, self.profile)

    # Called from the capture thread: only emit a signal (queued to the UI thread).
    def _on_status(self, status: CameraStatus) -> None:
        if status.state in (CameraState.FAILED, CameraState.RECOVERING, CameraState.RUNNING):
            log.info("camera %d %s: %s (%dx%d)", status.device_index, status.state.value, status.message,
                     status.width, status.height)
        self.statusChanged.emit(status)

    def _set_status(self, status: CameraStatus) -> None:
        self.status = status
        self.statusChanged.emit(status)
