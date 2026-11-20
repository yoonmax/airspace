"""Run Python's cyclic garbage collector only on the UI thread, between frames.

Python 3.14.0's incremental collector walks every thread's stack. If it runs
on a worker thread (e.g. while MediaPipe is importing on the vision thread)
at the moment the UI thread is inside a PySide deallocation — which releases
the GIL part-way through freeing the object — it can read a half-freed object
and crash the process (two such crashes were captured in macOS crash reports,
both with ``mark_stacks`` on the ``airspace-vision`` thread).

Qt objects are only created and freed on the UI thread, so collecting from a
UI-thread timer (never inside paint) removes that overlap. Reference counting
still frees almost everything immediately; the collector only reclaims cycles.
Objects that survive start-up are frozen so full collections stay cheap.
"""

from __future__ import annotations

import gc
import logging
import time

from PySide6.QtCore import QObject, QTimer

log = logging.getLogger(__name__)


class UiThreadGarbageCollector(QObject):
    YOUNG_INTERVAL_MS = 500
    FULL_EVERY_S = 30.0
    FREEZE_AFTER_S = 20.0
    SLOW_MS = 8.0

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._started = time.monotonic()
        self._last_full = self._started
        self._frozen = False
        self._timer = QTimer(self)
        self._timer.setInterval(self.YOUNG_INTERVAL_MS)
        self._timer.timeout.connect(self.tick)

    def start(self) -> None:
        gc.disable()
        self._timer.start()
        log.info("garbage collection moved to the UI thread")

    def stop(self) -> None:
        self._timer.stop()
        gc.enable()

    def tick(self) -> None:
        now = time.monotonic()
        t0 = time.perf_counter()
        if not self._frozen and now - self._started >= self.FREEZE_AFTER_S:
            gc.collect()
            gc.freeze()  # long-lived start-up objects: never scanned again
            self._frozen = True
            self._last_full = now
            kind = "freeze"
        elif now - self._last_full >= self.FULL_EVERY_S:
            gc.collect()
            self._last_full = now
            kind = "full"
        else:
            gc.collect(0)
            kind = "young"
        ms = (time.perf_counter() - t0) * 1000
        if ms > self.SLOW_MS:
            log.info("gc %s collection took %.1f ms", kind, ms)
