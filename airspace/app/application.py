"""Application bootstrap: logging, Qt setup, error handling."""

from __future__ import annotations

import logging
import os
import signal
import threading
import sys
import traceback
from logging.handlers import RotatingFileHandler

from airspace.app.configuration import APP_NAME, APP_VERSION
from airspace.persistence.paths import data_dir


def configure_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.addHandler(stream)
    try:
        fh = RotatingFileHandler(data_dir() / "airspace.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    # MediaPipe / TFLite are chatty on stderr; keep their logs quiet.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    verbose = "--verbose" in argv
    configure_logging(verbose)
    log = logging.getLogger("airspace")

    if "--raster" in argv:
        os.environ["AIRSPACE_RENDERER"] = "raster"

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    from airspace.app.main_window import MainWindow
    from airspace.persistence.settings import SettingsStore

    store = SettingsStore()
    if "--reset" in argv:
        store.clear()
    if os.environ.get("AIRSPACE_RENDERER") == "raster":
        s = store.load()
        s.performance.renderer = "raster"
        store.save(s)

    def excepthook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            app.quit()
            return
        log.error("Unhandled exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        try:
            win.toast("Something went wrong — details were written to the log")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = excepthook

    def thread_excepthook(args) -> None:
        # Background threads (capture, vision) would otherwise only print to stderr,
        # which a packaged app discards.
        log.error("Unhandled exception in thread %s:\n%s", getattr(args.thread, "name", "?"),
                  "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

    threading.excepthook = thread_excepthook

    from airspace.app.gc_policy import UiThreadGarbageCollector

    # Before any worker thread starts: see gc_policy for the crash this prevents.
    collector = UiThreadGarbageCollector(app)
    collector.start()

    win = MainWindow(store, start_camera="--no-camera" not in argv)
    screen = app.primaryScreen().availableGeometry()
    w, h = int(screen.width() * 0.86), int(screen.height() * 0.86)
    win.resize(max(960, w), max(620, h))
    win.move(screen.center().x() - win.width() // 2, screen.center().y() - win.height() // 2)
    if "--fullscreen" in argv:
        win.showFullScreen()
    else:
        win.show()
    win.raise_()
    win.activateWindow()
    _schedule_screenshot(argv, win, app)
    # Ctrl-C in the terminal quits cleanly (saving layout) instead of being
    # reported as an error; the timer lets Python see the signal while Qt runs.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    from PySide6.QtCore import QTimer

    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(250)
    return app.exec()


def _schedule_screenshot(argv: list[str], win, app) -> None:
    """``--screenshot PATH [--after SECONDS]``: save a real frame of the running app, then quit."""
    if "--screenshot" not in argv:
        return
    from PySide6.QtCore import QTimer

    i = argv.index("--screenshot")
    path = argv[i + 1] if i + 1 < len(argv) else "airspace.png"
    after = float(argv[argv.index("--after") + 1]) if "--after" in argv else 4.0
    actions = argv[argv.index("--do") + 1].split(",") if "--do" in argv else []

    def run_actions() -> None:
        for act in actions:
            if act.startswith("open:"):
                win.open_app(act.split(":", 1)[1])
            elif act == "launcher":
                win.toggle_launcher()
            elif act == "calibrate":
                win.start_calibration()
            elif act.startswith("call:"):  # call:<app>:<method>[:arg] — presses an app's own button
                _, key, method, *arg = act.split(":")
                target = win.wm.find(key)
                if target is not None:
                    getattr(target.app, method)(*[int(a) if a.isdigit() else a for a in arg])
            elif act.startswith("focus:"):
                target = win.wm.find(act.split(":", 1)[1])
                if target is not None:
                    win.wm.focus(target)

    def shoot() -> None:
        img = win.grab_frame()
        img.save(path)
        logging.getLogger("airspace").warning("screenshot saved to %s (%dx%d)", path, img.width(), img.height())
        win.close()
        app.quit()

    QTimer.singleShot(int(after * 400), run_actions)
    QTimer.singleShot(int(after * 1000), shoot)
