"""Camera permission handling.

On macOS, camera access is granted per *responsible app*: the .app bundle, or
for ``python -m airspace`` the terminal app it was started from. We ask
AVFoundation directly (through the Objective-C runtime via ctypes, so no extra
dependency) and poll the authorisation status until the user answers the
system prompt, then start capture. Qt's permission API and an OpenCV
main-thread open are kept as fallbacks. Once access is known, OpenCV's own
background-thread request is disabled via ``OPENCV_AVFOUNDATION_SKIP_AUTH``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Callable

log = logging.getLogger(__name__)

# AVAuthorizationStatus values.
NOT_DETERMINED, RESTRICTED, DENIED, AUTHORIZED = 0, 1, 2, 3

DENIED_HELP = ("Camera access is off for the app airspace was launched from. Turn it on in "
               "System Settings → Privacy & Security → Camera, then press Retry.")


class _AVFoundation:
    """Minimal ctypes bridge to ``AVCaptureDevice`` authorisation calls."""

    def __init__(self) -> None:
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        self._avf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        libsystem = ctypes.cdll.LoadLibrary("/usr/lib/libSystem.B.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        addr = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        self._status_fn = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(addr)
        self._request_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_void_p)(addr)
        self._cls = objc.objc_getClass(b"AVCaptureDevice")
        if not self._cls:
            raise OSError("AVCaptureDevice unavailable")
        self._sel_status = objc.sel_registerName(b"authorizationStatusForMediaType:")
        self._sel_request = objc.sel_registerName(b"requestAccessForMediaType:completionHandler:")
        self._media = ctypes.c_void_p.in_dll(self._avf, "AVMediaTypeVideo").value
        self._global_block_isa = ctypes.addressof(ctypes.c_void_p.in_dll(libsystem, "_NSConcreteGlobalBlock"))
        self._block_refs: list[object] = []  # blocks must outlive the asynchronous callback

    def status(self) -> int:
        return int(self._status_fn(self._cls, self._sel_status, self._media))

    def request(self) -> None:
        """Show the system prompt; the answer is observed by polling ``status``."""

        class Descriptor(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_ulong), ("size", ctypes.c_ulong)]

        invoke_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_bool)

        class Block(ctypes.Structure):
            _fields_ = [("isa", ctypes.c_void_p), ("flags", ctypes.c_int), ("reserved", ctypes.c_int),
                        ("invoke", invoke_t), ("descriptor", ctypes.POINTER(Descriptor))]

        def invoke(_block, granted) -> None:  # runs on a GCD thread; just log
            log.info("camera permission answered: %s", "granted" if granted else "denied")

        descriptor = Descriptor(0, ctypes.sizeof(Block))
        block = Block(self._global_block_isa, 1 << 28, 0, invoke_t(invoke), ctypes.pointer(descriptor))
        self._block_refs.extend([descriptor, block])
        self._request_fn(self._cls, self._sel_request, self._media, ctypes.addressof(block))


_bridge: _AVFoundation | None = None


def _avfoundation() -> _AVFoundation | None:
    global _bridge
    if _bridge is None and sys.platform == "darwin":
        try:
            _bridge = _AVFoundation()
        except Exception as exc:  # noqa: BLE001 - fall back to Qt/OpenCV
            log.info("AVFoundation bridge unavailable (%s)", exc)
    return _bridge


def authorization_status() -> int | None:
    """Current macOS camera authorisation status, or None when unknown/not macOS."""
    av = _avfoundation()
    if av is None:
        return None
    try:
        return av.status()
    except Exception:  # noqa: BLE001
        log.exception("camera authorisation status query failed")
        return None


def request_camera_access(parent, on_result: Callable[[bool, str], None], poll_ms: int = 250) -> None:
    """Ensure camera access; calls ``on_result(granted, detail)`` on the UI thread.

    When the user has not decided yet, the system prompt is shown and the
    result is delivered once they answer, however long that takes.
    """
    if sys.platform != "darwin":
        on_result(True, "")
        return
    status = authorization_status()
    if status is None:
        _request_via_qt(parent, on_result)
        return
    log.info("camera authorisation status: %d", status)
    if status == AUTHORIZED:
        _skip_opencv_auth()
        on_result(True, "")
        return
    if status in (DENIED, RESTRICTED):
        on_result(False, DENIED_HELP)
        return
    try:
        _avfoundation().request()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        log.exception("camera permission request failed; trying Qt")
        _request_via_qt(parent, on_result)
        return

    from PySide6.QtCore import QTimer

    timer = QTimer(parent)
    timer.setInterval(poll_ms)

    def poll() -> None:
        s = authorization_status()
        if s in (None, NOT_DETERMINED):
            return
        timer.stop()
        timer.deleteLater()
        if s == AUTHORIZED:
            _skip_opencv_auth()
            on_result(True, "")
        else:
            on_result(False, DENIED_HELP)

    timer.timeout.connect(poll)
    timer.start()


def _request_via_qt(parent, on_result: Callable[[bool, str], None]) -> None:
    try:
        from PySide6.QtCore import QCameraPermission, QCoreApplication, Qt

        app = QCoreApplication.instance()
        status = app.checkPermission(QCameraPermission())
        if status == Qt.PermissionStatus.Granted:
            _skip_opencv_auth()
            on_result(True, "")
            return
        if status == Qt.PermissionStatus.Denied:
            on_result(False, DENIED_HELP)
            return

        def done(permission) -> None:
            if app.checkPermission(QCameraPermission()) == Qt.PermissionStatus.Granted:
                _skip_opencv_auth()
                on_result(True, "")
            else:
                granted = _opencv_main_thread_request()
                on_result(granted, "" if granted else DENIED_HELP)

        app.requestPermission(QCameraPermission(), parent, done)
    except Exception as exc:  # noqa: BLE001 - permission API unavailable
        log.info("Qt permission API unavailable (%s); using OpenCV request", exc)
        granted = _opencv_main_thread_request()
        on_result(granted, "" if granted else DENIED_HELP)


def _skip_opencv_auth() -> None:
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"


def _opencv_main_thread_request() -> bool:
    """Open and immediately release the default camera on the main thread."""
    try:
        import cv2

        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        ok = cap.isOpened()
        cap.release()
        if ok:
            _skip_opencv_auth()
        return ok
    except Exception:  # noqa: BLE001
        log.exception("OpenCV permission request failed")
        return False
