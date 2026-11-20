"""Notice when the screen is locked so the camera is not left running behind it.

macOS keeps delivering camera frames to an app whose window is hidden behind
the lock screen. airspace would then keep acting on hand movements nobody can
see (such as typing a password at the lock screen), and an open camera keeps
the Mac from idle-sleeping. The watcher polls the login session once a second
(``CGSessionCopyCurrentDictionary`` is a cheap local call) and reports changes.

The session is treated as locked when the screen is locked or when another
user has the console (fast user switching). Where the state cannot be read —
not macOS, or no window-server session — it reports "unlocked" and never
pauses anything.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal

log = logging.getLogger(__name__)

_UTF8 = 0x08000100  # kCFStringEncodingUTF8


class _SessionProbe:
    """Reads the login session dictionary through CoreGraphics (macOS only)."""

    def __init__(self) -> None:
        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        vp = ctypes.c_void_p
        cg.CGSessionCopyCurrentDictionary.restype = vp
        cg.CGSessionCopyCurrentDictionary.argtypes = []
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFGetTypeID.restype = ctypes.c_ulong
        cf.CFGetTypeID.argtypes = [vp]
        cf.CFBooleanGetTypeID.restype = ctypes.c_ulong
        cf.CFBooleanGetValue.restype = ctypes.c_bool
        cf.CFBooleanGetValue.argtypes = [vp]
        cf.CFRelease.argtypes = [vp]
        self._cg, self._cf = cg, cf
        # Created once and kept for the life of the process.
        self._locked_key = cf.CFStringCreateWithCString(None, b"CGSSessionScreenIsLocked", _UTF8)
        self._console_key = cf.CFStringCreateWithCString(None, b"kCGSSessionOnConsoleKey", _UTF8)
        self._bool_type = cf.CFBooleanGetTypeID()

    def _flag(self, session: int, key: int) -> bool | None:
        value = self._cf.CFDictionaryGetValue(session, key)
        if not value:
            return None
        if self._cf.CFGetTypeID(value) != self._bool_type:
            return None
        return bool(self._cf.CFBooleanGetValue(value))

    def __call__(self) -> bool:
        session = self._cg.CGSessionCopyCurrentDictionary()
        if not session:
            return False  # no window-server session: nothing to judge by
        try:
            if self._flag(session, self._locked_key):
                return True  # the key is only present (and true) while locked
            return self._flag(session, self._console_key) is False
        finally:
            self._cf.CFRelease(session)


def default_probe() -> Callable[[], bool] | None:
    if sys.platform != "darwin":
        return None
    try:
        return _SessionProbe()
    except (OSError, AttributeError) as exc:
        log.warning("screen lock detection unavailable: %s", exc)
        return None


class ScreenLockWatcher(QObject):
    """Emits ``lockedChanged(bool)`` when the screen locks or unlocks."""

    lockedChanged = Signal(bool)
    POLL_MS = 1000

    def __init__(self, parent: QObject | None = None, probe: Callable[[], bool] | None = None) -> None:
        super().__init__(parent)
        self._probe = probe if probe is not None else default_probe()
        self.locked = False
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self.poll)

    @property
    def available(self) -> bool:
        return self._probe is not None

    def start(self) -> None:
        if self._probe is not None:
            self.poll()
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def poll(self) -> None:
        if self._probe is None:
            return
        try:
            locked = bool(self._probe())
        except Exception:  # noqa: BLE001 - a failed read must never pause the camera
            log.exception("screen lock check failed; disabling it")
            self._probe = None
            self._timer.stop()
            return
        if locked != self.locked:
            self.locked = locked
            self.lockedChanged.emit(locked)
