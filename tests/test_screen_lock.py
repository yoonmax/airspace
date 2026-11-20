import sys

import pytest

from airspace.app.screen_lock import ScreenLockWatcher, default_probe


def test_watcher_reports_only_transitions(qapp):
    states = iter([False, False, True, True, False])
    w = ScreenLockWatcher(probe=lambda: next(states))
    seen = []
    w.lockedChanged.connect(seen.append)
    for _ in range(5):
        w.poll()
    assert seen == [True, False]
    assert w.locked is False


def test_failing_probe_disables_itself_and_never_reports_locked(qapp):
    def broken():
        raise RuntimeError("no session")

    w = ScreenLockWatcher(probe=broken)
    seen = []
    w.lockedChanged.connect(seen.append)
    w.poll()
    w.poll()
    assert seen == [] and not w.available and w.locked is False


@pytest.mark.skipif(sys.platform != "darwin", reason="CoreGraphics session probe is macOS-only")
def test_real_probe_returns_a_bool():
    probe = default_probe()
    assert probe is not None
    assert isinstance(probe(), bool)


def test_main_window_pauses_camera_while_locked(qapp, data_dir):
    from airspace.app.main_window import MainWindow
    from airspace.persistence.settings import SettingsStore

    win = MainWindow(SettingsStore(), start_camera=False)
    calls = []
    try:
        win.settings.camera.enabled = True
        win.camera._worker = object()  # pretend the camera is running
        win.camera.stop = lambda: (calls.append("stop"), setattr(win.camera, "_worker", None))
        win.camera.start = lambda *a: calls.append("start")
        win.latest_packet = object()
        win.ctx.perf.camera_fps = win.ctx.perf.tracking_fps = 30.0

        win._on_screen_lock(True)
        assert calls == ["stop"] and win._paused_for_lock
        assert win.latest_packet is None and win.tracking_frame is None
        assert win.ctx.perf.camera_fps == 0.0 and win.ctx.perf.tracking_fps == 0.0  # no stale rates in the log
        assert win.settings.camera.enabled  # the user's choice is untouched

        win._on_screen_lock(False)
        assert calls == ["stop", "start"] and not win._paused_for_lock

        # Camera switched off by the user: locking and unlocking leave it off.
        win.settings.camera.enabled = False
        win._on_screen_lock(True)
        win._on_screen_lock(False)
        assert calls == ["stop", "start"]
    finally:
        win.camera._worker = None
        win.close()
