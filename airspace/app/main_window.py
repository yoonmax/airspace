"""The airspace main window: wires every subsystem together.

Threads:
  * capture thread  — camera → LatestFrameBuffer
  * vision thread   — MediaPipe hands + segmentation → VisionPacket (Qt signal)
  * UI thread       — tracking/gestures, interaction, window manager, rendering
"""

from __future__ import annotations

import logging
import threading
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent, QKeySequence, QMouseEvent, QPainter, QShortcut, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QFileDialog, QMainWindow, QWidget

from airspace.app.context import AppContext
from airspace.app.screen_lock import ScreenLockWatcher
from airspace.camera.camera_manager import CameraManager, enumerate_cameras
from airspace.camera.capture import CameraState, CameraStatus
from airspace.camera.frame_buffer import LatestFrameBuffer
from airspace.camera.permissions import request_camera_access
from airspace.vision.gesture_stats import GestureStats
from airspace.interaction.gesture_router import GestureRouter
from airspace.interaction.interaction_manager import InteractionManager
from airspace.miniapps.registry import APPS, BY_KEY, load_class
from airspace.persistence.paths import data_dir
from airspace.persistence.settings import BACKGROUND_MODES, OCCLUSION_MODES, Settings, SettingsStore
from airspace.persistence.workspace_storage import WorkspaceStore
from airspace.ui import animations
from airspace.ui.calibration_wizard import CalibrationWizard
from airspace.ui.dialogs import ConfirmDialog, MessageCard
from airspace.ui.sounds import SoundBank
from airspace.ui.styles import Theme, make_theme
from airspace.vision.calibration import PinchCalibration
from airspace.vision.coordinate_mapper import CoordinateMapper
from airspace.vision.gesture_state_machine import GestureConfig, GestureType
from airspace.vision.model_files import download_missing, missing_models, model_path
from airspace.vision.tracking import HandTrackingProcessor, TrackingFrame
from airspace.vision.vision_worker import VisionConfig, VisionPacket, VisionWorker
from airspace.workspace.compositing import FrameLayers
from airspace.workspace.geometry import Rect
from airspace.workspace.hud import Hud, HudActions, StatusInfo
from airspace.workspace.launcher import Launcher, LauncherItem
from airspace.workspace.layout_manager import GRID_H, GRID_W, PlacementModel
from airspace.workspace.rendering import Renderer, RenderInputs
from airspace.workspace.window_manager import WindowManager
from airspace.workspace.workspace import Workspace

log = logging.getLogger(__name__)

DEFAULT_APPS = ("objects", "music")


# =================================================================== view
def make_view_class(opengl: bool):
    base = QOpenGLWidget if opengl else QWidget

    class WorkspaceView(base):  # type: ignore[misc, valid-type]
        def __init__(self, controller: "MainWindow") -> None:
            super().__init__()
            self.controller = controller
            self.setMouseTracking(True)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if opengl:
            def paintGL(self) -> None:  # noqa: N802
                p = QPainter(self)
                self.controller.paint(p)
                p.end()
        else:
            def paintEvent(self, event) -> None:  # noqa: N802
                p = QPainter(self)
                self.controller.paint(p)
                p.end()

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            self.controller.on_resize(self.width(), self.height())

        def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
            self.controller.mouse("down", e)

        def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
            self.controller.mouse("move", e)

        def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
            self.controller.mouse("up", e)

        def wheelEvent(self, e: QWheelEvent) -> None:  # noqa: N802
            self.controller.wheel(e)

        def leaveEvent(self, e: QEvent) -> None:  # noqa: N802
            self.controller.mouse_left()

        def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
            if not self.controller.key(e):
                super().keyPressEvent(e)

        def event(self, e: QEvent) -> bool:
            # Let Tab reach our key handler instead of moving widget focus.
            if e.type() == QEvent.Type.KeyPress and e.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                self.keyPressEvent(e)
                return True
            return super().event(e)

    return WorkspaceView


# ============================================================ main window
class MainWindow(QMainWindow):
    modelsDownloaded = Signal(bool, str)

    def __init__(self, settings_store: SettingsStore | None = None, workspace_store: WorkspaceStore | None = None,
                 start_camera: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("airspace")
        self.store = settings_store or SettingsStore()
        self.layout_store = workspace_store or WorkspaceStore()
        self.settings: Settings = self.store.load()
        self._theme: Theme = self._make_theme()
        self.sounds = SoundBank(data_dir() / "sounds")
        self.sounds.enabled = self.settings.interface.sounds

        # ---- vision pipeline
        self.buffer = LatestFrameBuffer()
        self.camera = CameraManager(self.buffer)
        self.camera.statusChanged.connect(self._on_camera_status)
        self.vision = VisionWorker(self.buffer, model_path("hand_landmarker.task"), model_path("selfie_segmenter.tflite"))
        self.vision.packetReady.connect(self._on_packet)
        self.vision.errorOccurred.connect(self._on_vision_error)
        self.gesture_stats = GestureStats()
        self.processor = HandTrackingProcessor(self._gesture_config(), self.settings.tracking.smoothing,
                                               self.settings.tracking.dominant_hand)
        self.mapper = CoordinateMapper(reach=self._reach())
        self.latest_packet: VisionPacket | None = None
        self.tracking_frame: TrackingFrame | None = None
        self.layers = FrameLayers()

        # ---- workspace
        self.ctx = AppContext(self.settings, lambda: self._theme, self, data_dir())
        self.wm = WindowManager(self._create_app, titlebar_h=self._theme.titlebar_height)
        self.wm.size_scale = self.settings.interface.window_scale
        self.placement = PlacementModel()
        self.wm.placement = self._place
        self.launcher = Launcher([LauncherItem(a.key, a.label, a.icon) for a in APPS], self.open_app)
        self.hud = Hud(HudActions(
            open_launcher=lambda: self.toggle_launcher(),
            arrange=self.arrange_windows,
            open_settings=lambda: self.open_app("settings"),
            toggle_fullscreen=self.toggle_fullscreen,
            is_fullscreen=self.isFullScreen,
            restore_window=self._restore_window,
            toggle_camera=self.toggle_camera,
            camera_enabled=lambda: self.settings.camera.enabled,
        ), self._status, self._minimized_items)
        self.workspace = Workspace(self.wm, self.launcher, self.hud, self.toast)
        self.manager = InteractionManager(self.workspace)
        self.router = GestureRouter(self.manager, self.workspace)
        self.manager.dwell_enabled = self.settings.accessibility.dwell_click
        self.manager.dwell_ms = self.settings.accessibility.dwell_ms
        self.manager.on_click_feedback = lambda kind: self.play_sound("click")
        self.renderer = Renderer()

        # ---- view
        View = make_view_class(self.settings.performance.renderer == "opengl")
        self.view = View(self)
        self.setCentralWidget(self.view)
        self.setMinimumSize(900, 600)

        # ---- state
        self._last_tick = time.perf_counter()
        self._frame_times: deque[float] = deque(maxlen=90)
        self._painted_packet = -1
        self._wizard: CalibrationWizard | None = None
        self._camera_card: MessageCard | None = None
        self._pending_defaults = True
        self._defaults_deadline = time.monotonic() + 2.5
        self._dialog_open = False
        self._hand_seen_at = 0.0
        self._hand_lost_at: float | None = None
        self._hand_present_since: float | None = None
        self._last_person_update = 0.0
        self._last_save = time.monotonic()
        self._layout_restored = False
        self._shutting_down = False
        self._camera_was_running = False
        self._permission_denied = False
        self._permission_pending = False
        self._paused_for_lock = False
        self._last_perf_log = time.monotonic()
        self._last_paint_at: float | None = None
        self.screen_lock = ScreenLockWatcher(self)
        self.screen_lock.lockedChanged.connect(self._on_screen_lock)

        # ---- timers & shortcuts
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / self.settings.performance.fps_target))
        for seq, fn in (("Ctrl+Q", self.close), ("F11", self.toggle_fullscreen),
                        ("Ctrl+Meta+F", self.toggle_fullscreen)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)
        self.modelsDownloaded.connect(self._on_models_downloaded)

        self._restore_layout_or_defaults()
        if start_camera:
            QTimer.singleShot(0, self._startup)

    # ============================================================ startup
    def _startup(self) -> None:
        if missing_models(required_only=True):
            self._show_models_missing()
            return
        self.vision.start()
        self._apply_vision_config()
        self.screen_lock.start()
        if self.settings.camera.enabled:
            self._permission_pending = True
            request_camera_access(self, self._on_camera_permission)
        if not self.settings.first_run_complete:
            QTimer.singleShot(600, self.start_calibration)

    def _on_camera_permission(self, granted: bool, detail: str) -> None:
        self._permission_pending = False
        if granted:
            self._permission_denied = False
            if self.screen_lock.locked:
                self._paused_for_lock = True  # start it when the screen is unlocked
            else:
                self.camera.start(self.settings.camera.device_index, self.settings.camera.profile)
        else:
            status = CameraStatus(CameraState.FAILED, detail, device_index=self.settings.camera.device_index)
            self.camera._set_status(status)
            self._permission_denied = True
            self._show_camera_error(status)

    def _restore_layout_or_defaults(self) -> None:
        layout = self.layout_store.load()
        if layout and layout.windows:
            self._pending_layout = layout
        else:
            self._pending_layout = None

    def _apply_initial_layout(self) -> None:
        """Runs once the workspace has its real size (first resize)."""
        if self._layout_restored:
            return
        self._layout_restored = True
        if self._pending_layout is not None:
            self.wm.apply_layout(self._pending_layout, set(BY_KEY))
            self._pending_defaults = False

    def _open_defaults_if_pending(self, force: bool = False) -> None:
        if not self._pending_defaults:
            return
        have_person = self.placement.person_cost is not None
        if have_person or force or time.monotonic() > self._defaults_deadline:
            self._pending_defaults = False
            for key in DEFAULT_APPS:
                try:
                    self.wm.open(key, focus=False)
                except Exception:  # noqa: BLE001
                    log.exception("Could not open default app %s", key)

    # ======================================================== app factory
    def _create_app(self, app_type: str):
        cls = load_class(app_type)
        return cls(self.ctx)

    def _place(self, w: float, h: float, windows) -> Rect:
        b = self.wm.bounds
        return self.placement.place(w, h, b, self.wm.top_inset, self.wm.bottom_inset + 56,
                                    [x.rect for x in windows])

    def open_app(self, key: str) -> None:
        if key not in BY_KEY:
            return
        try:
            self.wm.open(key)
            self.play_sound("open")
        except Exception as exc:  # noqa: BLE001
            log.exception("Could not open %s", key)
            self.toast(f"Could not open {key}: {exc}")

    def _restore_window(self, win_id: int) -> None:
        win = self.wm.by_id(win_id)
        if win:
            self.wm.restore(win)

    def _minimized_items(self) -> list[tuple[int, str, str]]:
        return [(w.id, w.title, w.app.icon) for w in self.wm.minimized]

    # ============================================================ settings
    def _make_theme(self) -> Theme:
        i, a = self.settings.interface, self.settings.accessibility
        animations.set_reduced_motion(a.reduced_motion or not i.animations)
        return make_theme(i.transparency, i.window_scale, a.large_buttons, a.high_contrast,
                          a.reduced_motion or not i.animations)

    def _gesture_config(self) -> GestureConfig:
        t = self.settings.tracking
        return GestureConfig(pinch_enter=t.pinch_enter, pinch_exit=t.pinch_exit, pinch_hold_ms=t.pinch_hold_ms,
                             palm_hold_ms=t.palm_hold_ms)

    def _reach(self) -> float:
        t = self.settings.tracking
        return 1.0 if t.precision_mode else t.pointer_reach

    def _apply_vision_config(self) -> None:
        s = self.settings
        self.vision.set_config(VisionConfig(
            mirror=s.camera.mirror, max_hands=s.tracking.max_hands,
            detection_confidence=s.tracking.detection_confidence,
            tracking_confidence=s.tracking.tracking_confidence,
            segmentation_interval=s.performance.segmentation_interval,
            occlusion=s.interface.occlusion,
            need_person_mask=True,
        ))

    def apply_settings(self, save: bool = True) -> None:
        """Re-apply every setting to the live subsystems."""
        s = self.settings
        s.validate()
        self._theme = self._make_theme()
        self.wm.set_titlebar_height(self._theme.titlebar_height)
        self.wm.size_scale = s.interface.window_scale
        self.processor.configure(self._gesture_config(),
                                 "steady" if s.tracking.precision_mode else s.tracking.smoothing,
                                 s.tracking.dominant_hand)
        self.mapper.configure(reach=self._reach())
        self.manager.dwell_enabled = s.accessibility.dwell_click
        self.manager.dwell_ms = s.accessibility.dwell_ms
        self.sounds.enabled = s.interface.sounds
        self._apply_vision_config()
        self.timer.setInterval(int(1000 / s.performance.fps_target))
        for w in self.wm.windows:
            w.app.relayout()
        if save:
            self.store.save(s)

    def set_camera(self, index: int, profile: str | None = None) -> None:
        s = self.settings.camera
        s.device_index = index
        if profile:
            s.profile = profile
        self.store.save(self.settings)
        if s.enabled:
            self.processor.reset()
            self.camera.start(s.device_index, s.profile)

    def set_mirror(self, mirror: bool) -> None:
        self.settings.camera.mirror = mirror
        self.processor.reset()
        self.apply_settings()

    def list_cameras(self) -> list[tuple[int, str]]:
        return [(c.index, c.name) for c in enumerate_cameras()]

    def toggle_camera(self) -> None:
        s = self.settings.camera
        s.enabled = not s.enabled
        self._paused_for_lock = False
        if s.enabled:
            self.camera.start(s.device_index, s.profile)
            self.toast("Camera on")
        else:
            self.camera.stop()
            self._clear_hand_input()
            self.toast("Camera off — mouse and keyboard remain available")
        self.store.save(self.settings)

    def _clear_hand_input(self) -> None:
        """Drop the camera backdrop and end any pinch, drag or hover the hand had in progress."""
        self.latest_packet = None
        self.layers.update(None)
        self.manager.cancel("hand")
        self.manager.hide_hand()
        self.processor.reset()
        self.tracking_frame = None
        # These are only refreshed by incoming frames; zero them so the log
        # does not keep reporting the last live rates while the camera is off.
        perf = self.ctx.perf
        perf.camera_fps = perf.tracking_fps = 0.0
        perf.capture_to_display_ms = 0.0
        perf.gesture_latency_ms = None

    def _on_screen_lock(self, locked: bool) -> None:
        """Pause the camera while the screen is locked; resume it on unlock.

        Behind the lock screen nobody can see airspace, so hand movements (such
        as typing a password) must not drive it, the camera should not stay on,
        and an open camera would keep the Mac from idle-sleeping.
        """
        if self._shutting_down:
            return
        if locked:
            if self.settings.camera.enabled and self.camera.running:
                self._paused_for_lock = True
                self.camera.stop()
                self._clear_hand_input()
                log.info("screen locked: camera paused")
        elif self._paused_for_lock:
            self._paused_for_lock = False
            if self.settings.camera.enabled:
                self.camera.start(self.settings.camera.device_index, self.settings.camera.profile)
                log.info("screen unlocked: camera resumed")

    def clear_saved_settings(self) -> None:
        def do() -> None:
            self.settings.__dict__.update(self.store.clear().__dict__)
            self.ctx.settings = self.settings
            self.settings.first_run_complete = True
            self.apply_settings()
            self.toast("Settings reset to defaults")

        self.workspace.push_overlay(ConfirmDialog("Reset all settings?",
                                                  "Every preference, including calibration, returns to its default.",
                                                  "Reset settings", do))

    # ========================================================= calibration
    def start_calibration(self) -> None:
        if self.workspace.has_overlay(CalibrationWizard):
            return
        self.launcher.close()
        wizard = CalibrationWizard(
            cameras=self.list_cameras(), current_camera=self.settings.camera.device_index,
            mirror=self.settings.camera.mirror, on_camera=lambda i: self.set_camera(i),
            on_mirror=self.set_mirror, on_calibrated=self._apply_calibration,
            on_finish=self._calibration_finished, pointer=lambda: self.manager.pointers["hand"])
        self._wizard = wizard
        self.workspace.push_overlay(wizard)

    def _apply_calibration(self, cal: PinchCalibration) -> None:
        if cal.valid:
            self.settings.tracking.pinch_enter = round(cal.enter, 3)
            self.settings.tracking.pinch_exit = round(cal.exit, 3)
            self.settings.calibrated = True
            self.apply_settings()

    def _calibration_finished(self, completed: bool) -> None:
        self.settings.first_run_complete = True
        self.store.save(self.settings)
        self._wizard = None
        self.toast("Calibration saved" if completed else "Using default gesture settings")
        self._open_defaults_if_pending(force=True)

    # ===================================================== workspace actions
    def toggle_launcher(self, at: QPointF | None = None) -> None:
        if self.launcher.is_open:
            self.launcher.close()
            return
        if at is None:
            mp = self.manager.pointers["mouse"]
            at = QPointF(mp.x, mp.y) if mp.visible else QPointF(self.view.width() / 2, self.view.height() / 2)
        self.workspace.open_launcher(at.x(), at.y())

    def arrange_windows(self) -> None:
        wins = [w for w in self.wm.windows if w.interactive]
        if not wins:
            self.toast("No windows to arrange")
            return
        rects = self.placement.arrange([(w.rect.w, w.rect.h) for w in wins], self.wm.bounds, self.wm.top_inset,
                                       self.wm.bottom_inset + 56)
        for w, r in zip(wins, rects):
            w.animate_to(self.wm.constrain(w, r))
        self.toast("Windows arranged")

    def save_layout(self, quiet: bool = False) -> None:
        try:
            self.layout_store.save(self.wm.to_layout())
            if not quiet:
                self.toast("Layout saved")
        except OSError as exc:
            log.warning("Could not save layout: %s", exc)

    def restore_layout(self) -> None:
        layout = self.layout_store.load()
        if not layout:
            self.toast("No saved layout yet")
            return
        self.wm.apply_layout(layout, set(BY_KEY))
        self.toast("Layout restored")

    def reset_workspace(self, confirm: bool = True) -> None:
        def do() -> None:
            self.wm.close_all()
            self.layout_store.clear()
            for key in DEFAULT_APPS:
                self.wm.open(key, focus=False)
            self.toast("Workspace reset")

        if confirm:
            self.workspace.push_overlay(ConfirmDialog(
                "Reset workspace?", "All windows close and the default layout returns. Saved drawings and "
                "imported files on disk are not affected.", "Reset", do))
        else:
            do()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toast(self, text: str) -> None:
        self.hud.toast(text)

    def play_sound(self, name: str) -> None:
        self.sounds.play(name)

    # =========================================================== dialogs
    def open_files(self, title: str, name_filter: str, multiple: bool = True) -> list[str]:
        self._begin_native_dialog()
        try:
            if multiple:
                files, _ = QFileDialog.getOpenFileNames(self, title, str(Path.home()), name_filter)
            else:
                f, _ = QFileDialog.getOpenFileName(self, title, str(Path.home()), name_filter)
                files = [f] if f else []
        finally:
            self._end_native_dialog()
        return files

    def save_file(self, title: str, default_name: str, name_filter: str) -> str | None:
        self._begin_native_dialog()
        try:
            f, _ = QFileDialog.getSaveFileName(self, title, str(Path.home() / default_name), name_filter)
        finally:
            self._end_native_dialog()
        return f or None

    def _begin_native_dialog(self) -> None:
        # Pause hand interaction while a native dialog is open (it runs a nested event loop).
        self._dialog_open = True
        self.manager.cancel("hand")
        self.manager.hide_hand()

    def _end_native_dialog(self) -> None:
        self._dialog_open = False
        self.view.setFocus()

    # ======================================================= vision input
    def _on_packet(self, pkt: VisionPacket) -> None:
        if self._shutting_down or not self.settings.camera.enabled or self._paused_for_lock:
            return
        self.latest_packet = pkt
        fw, fh = pkt.size
        if (fw, fh) != (self.mapper.frame_w, self.mapper.frame_h):
            self.mapper.configure(frame_w=fw, frame_h=fh)
        frame = self.processor.process(pkt.hands, pkt.capture_ts, fw / fh)
        self.tracking_frame = frame
        self.gesture_stats.add(frame)
        if not self._dialog_open:
            self.manager.handle_tracking(frame, self.mapper, capture_ts=pkt.capture_ts)
            if self.manager.last_gesture_latency_ms is not None:
                self.ctx.perf.gesture_latency_ms = self.manager.last_gesture_latency_ms
        for ev in frame.events:
            if ev.type == GestureType.PALM_OPEN and self.launcher.is_open:
                self.play_sound("open")
        if self._wizard is not None:
            self._wizard.on_tracking(frame)
        self._track_presence(frame)
        now = time.monotonic()
        if pkt.person_mask is not None and now - self._last_person_update > 0.5:
            self._last_person_update = now
            self.placement.update_person(self._person_grid(pkt.person_mask))
            self._open_defaults_if_pending()
        perf = self.ctx.perf
        perf.camera_fps = self.camera.fps
        perf.tracking_fps = self.vision.fps
        perf.hand_ms = pkt.hand_ms
        perf.seg_ms = pkt.seg_ms if pkt.seg_ms else perf.seg_ms
        perf.vision_ms = pkt.total_ms
        perf.dropped_frames = pkt.stats.get("dropped", 0)
        perf.frame_size = pkt.size

    def _person_grid(self, person: np.ndarray) -> np.ndarray:
        """Resample the person mask (camera space) to the placement grid (view space)."""
        ys = (np.arange(GRID_H) + 0.5) / GRID_H * self.mapper.view_h
        xs = (np.arange(GRID_W) + 0.5) / GRID_W * self.mapper.view_w
        u = (xs - self.mapper.image_rect.x) / self.mapper.image_rect.w
        v = (ys - self.mapper.image_rect.y) / self.mapper.image_rect.h
        h, w = person.shape
        ui = np.clip((u * w).astype(int), 0, w - 1)
        vi = np.clip((v * h).astype(int), 0, h - 1)
        return person[np.ix_(vi, ui)]

    def _track_presence(self, frame: TrackingFrame) -> None:
        now = time.monotonic()
        present = frame.present_count > 0
        if present:
            if self._hand_present_since is None:
                self._hand_present_since = now
            self._hand_seen_at = now
            self._hand_lost_at = None
        else:
            if self._hand_lost_at is None and self._hand_present_since is not None:
                self._hand_lost_at = now if now - self._hand_present_since > 1.0 else None
            self._hand_present_since = None

    def _on_vision_error(self, message: str) -> None:
        self.workspace.push_overlay(MessageCard(
            "Hand tracking stopped", message,
            ["Run  python scripts/download_models.py  to re-download the models.",
             "Mouse and keyboard control keep working."],
            [("Continue", lambda: self.workspace.remove_overlay_type(MessageCard), "primary")], icon="hand"))

    # ========================================================= camera status
    def _on_camera_status(self, status: CameraStatus) -> None:
        if status.state == CameraState.FAILED and self.settings.camera.enabled:
            if self._camera_card is None or self._camera_card.closing:
                self._show_camera_error(status)
        elif status.state == CameraState.RUNNING:
            if self._camera_card is not None:
                self._camera_card.dismiss()
                self._camera_card = None
                self.toast("Camera connected")
        elif status.state == CameraState.RECOVERING and self._camera_was_running:
            self.toast("Camera interrupted — reconnecting…")
        if status.state == CameraState.RUNNING:
            self._camera_was_running = True
        elif status.state in (CameraState.STOPPED, CameraState.FAILED):
            self._camera_was_running = False

    def _show_camera_error(self, status: CameraStatus) -> None:
        cams = self.list_cameras()
        card = MessageCard(
            "Camera unavailable",
            f"airspace couldn't get video from camera {status.device_index}. {status.message}",
            ["macOS: System Settings → Privacy & Security → Camera — allow the app running airspace "
             "(Terminal, your IDE, or airspace itself), then retry.",
             "Close other apps that may be using the camera (video calls, browsers).",
             "If you use an external webcam, check the cable or try another camera."],
            [("Continue without camera", self._continue_without_camera, "ghost"),
             ("Open Privacy Settings", self._open_camera_privacy, "default")
             if self._permission_denied and sys.platform == "darwin"
             else ("Next camera", lambda: self._next_camera(cams), "default"),
             ("Retry", self._retry_camera, "primary")],
            icon="camera", width=600)
        self._camera_card = card
        self.workspace.push_overlay(card)

    def _open_camera_privacy(self) -> None:
        """Show macOS's camera privacy pane; the user flips the switch, then presses Retry."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"))

    def _retry_camera(self) -> None:
        if self._camera_card:
            self._camera_card.dismiss()
            self._camera_card = None
        self._permission_pending = True
        request_camera_access(self, self._on_camera_permission)

    def _next_camera(self, cams: list[tuple[int, str]]) -> None:
        indices = [i for i, _ in cams] or [0]
        cur = self.settings.camera.device_index
        nxt = indices[(indices.index(cur) + 1) % len(indices)] if cur in indices else indices[0]
        if len(indices) == 1:
            nxt = (cur + 1) % 4  # probe the next index even if enumeration found only one
        if self._camera_card:
            self._camera_card.dismiss()
            self._camera_card = None
        self.set_camera(nxt)
        self.toast(f"Trying camera {nxt}…")

    def _continue_without_camera(self) -> None:
        if self._camera_card:
            self._camera_card.dismiss()
            self._camera_card = None
        if self.settings.camera.enabled:
            self.toggle_camera()
        self._open_defaults_if_pending(force=True)

    # ========================================================= models
    def _show_models_missing(self) -> None:
        names = ", ".join(f"{m.name} ({m.approx_size})" for m in missing_models())
        self.workspace.push_overlay(MessageCard(
            "Hand tracking models needed",
            "airspace uses Google's MediaPipe models (Apache 2.0) to track your hands on-device. "
            f"Missing: {names}. Nothing is uploaded — the models are only downloaded.",
            ["Or run  python scripts/download_models.py  from the project folder."],
            [("Use mouse only", self._models_declined, "ghost"),
             ("Download models", self._download_models, "primary")], icon="hand", width=600))

    def _models_declined(self) -> None:
        self.workspace.remove_overlay_type(MessageCard)
        self._open_defaults_if_pending(force=True)
        self.toast("Running without hand tracking")

    def _download_models(self) -> None:
        self.workspace.remove_overlay_type(MessageCard)
        self.toast("Downloading models…")

        def work() -> None:
            try:
                download_missing()
                self.modelsDownloaded.emit(True, "")
            except Exception as exc:  # noqa: BLE001
                self.modelsDownloaded.emit(False, str(exc))

        threading.Thread(target=work, daemon=True, name="airspace-models").start()

    def _on_models_downloaded(self, ok: bool, error: str) -> None:
        if ok:
            self.toast("Models ready")
            self._startup()
        else:
            self.workspace.push_overlay(MessageCard(
                "Download failed", f"The models could not be downloaded: {error}",
                ["Check your internet connection and try again."],
                [("Use mouse only", self._models_declined, "ghost"), ("Retry", self._download_models, "primary")],
                icon="info"))

    # ============================================================== status
    def _status(self) -> StatusInfo:
        s = self.camera.status
        if not self.settings.camera.enabled:
            cam_state, cam_label = "off", "camera off"
        elif self._permission_pending:
            cam_state, cam_label = "warn", "waiting for permission"
        elif s.state == CameraState.RUNNING:
            cam_state, cam_label = "ok", f"camera {s.width}×{s.height}"
        elif s.state in (CameraState.OPENING, CameraState.RECOVERING):
            cam_state, cam_label = "warn", "connecting"
        elif s.state == CameraState.FAILED:
            cam_state, cam_label = "error", "no access" if self._permission_denied else "camera error"
        else:
            cam_state, cam_label = "off", "camera off"
        frame = self.tracking_frame
        hands = frame.present_count if frame else 0
        primary = frame.primary if frame else None
        gesture = self.manager.state.value
        if primary is not None and primary.present and self.manager.state.value in ("idle", "hovering"):
            gesture = primary.pose.value
        lost = (self._hand_lost_at is not None and time.monotonic() - self._hand_lost_at < 2.5
                and self.settings.camera.enabled)
        return StatusInfo(cam_state, cam_label, hands, {0: "no hands", 1: "1 hand"}.get(hands, f"{hands} hands"),
                          gesture, self.ctx.perf.ui_fps, self.manager.state.value, lost)

    # ================================================================ frame
    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(0.1, now - self._last_tick)
        self._last_tick = now
        self._frame_times.append(now)
        if len(self._frame_times) > 2:
            span = self._frame_times[-1] - self._frame_times[0]
            self.ctx.perf.ui_fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
        self.layers.update(self.latest_packet if self.settings.camera.enabled else None)
        self.hud.layout(QRectF(0, 0, self.view.width(), self.view.height()), self._theme)
        self.workspace.tick(dt)
        if self._pending_defaults and self._layout_restored:
            self._open_defaults_if_pending()
        if time.monotonic() - self._last_perf_log > 5.0:
            self._last_perf_log = time.monotonic()
            self._log_perf()
        if time.monotonic() - self._last_save > 30:
            self._last_save = time.monotonic()
            self.save_layout(quiet=True)
        self.view.update()

    def _log_perf(self) -> None:
        """Write measured statistics to the local log (never sent anywhere)."""
        perf = self.ctx.perf
        frame = self.tracking_frame
        # macOS stops delivering paints to a hidden window (lock screen, other
        # Space, minimised); the last paint figures would then be stale.
        painting = self._last_paint_at is not None and time.perf_counter() - self._last_paint_at < 1.0
        logging.getLogger("airspace.perf").info(
            "perf ui_fps=%.1f paint_ms=%s camera_fps=%.1f tracking_fps=%.1f hand_ms=%.2f seg_ms=%.2f "
            "vision_ms=%.2f capture_to_display_ms=%s gesture_latency_ms=%s dropped=%d frame=%dx%d hands=%d state=%s%s",
            perf.ui_fps, f"{perf.paint_ms:.2f}" if painting else "n/a",
            perf.camera_fps, perf.tracking_fps, perf.hand_ms, perf.seg_ms, perf.vision_ms,
            f"{perf.capture_to_display_ms:.1f}" if painting else "n/a",
            "n/a" if perf.gesture_latency_ms is None else f"{perf.gesture_latency_ms:.1f}",
            perf.dropped_frames, perf.frame_size[0], perf.frame_size[1],
            frame.present_count if frame else 0, self.manager.state.value,
            ("" if painting else " painting=no") + (" camera=paused-locked" if self._paused_for_lock else ""))
        summary = self.gesture_stats.summary()
        if summary:
            t = self.settings.tracking
            logging.getLogger("airspace.perf").info("gestures %s pinch_enter=%.2f pinch_exit=%.2f", summary, t.pinch_enter, t.pinch_exit)
        self.gesture_stats.reset()

    def paint(self, p: QPainter) -> None:
        t0 = time.perf_counter()
        w, h = self.view.width(), self.view.height()
        packet = self.latest_packet if self.settings.camera.enabled else None
        live = packet is not None and time.perf_counter() - packet.processed_ts < 1.5
        ri = RenderInputs(self._theme, self.settings, self.workspace, self.manager, self.tracking_frame,
                          self.mapper, self.layers, live, w, h, self._wizard is not None)
        dt = 1.0 / max(self.ctx.perf.ui_fps, 30.0)
        self.renderer.render(p, ri, dt)
        if packet is not None and packet.frame_index != self._painted_packet:
            self._painted_packet = packet.frame_index
            lat = (time.perf_counter() - packet.capture_ts) * 1000
            perf = self.ctx.perf
            perf.capture_to_display_ms = lat if perf.capture_to_display_ms == 0 else 0.85 * perf.capture_to_display_ms + 0.15 * lat
        self._last_paint_at = time.perf_counter()
        self.ctx.perf.paint_ms = (self._last_paint_at - t0) * 1000

    def on_resize(self, w: int, h: int) -> None:
        bounds = Rect(0, 0, float(w), float(h))
        if not self._layout_restored:
            self.wm.bounds = bounds
        else:
            self.wm.set_bounds(bounds)
        self.wm.top_inset = self.hud.top_bar_height
        self.mapper.configure(view_w=w, view_h=h)
        self.workspace.set_bounds(QRectF(0, 0, w, h))
        self._apply_initial_layout()

    # ================================================================ input
    def mouse(self, kind: str, e: QMouseEvent) -> None:
        pos = e.position()
        x, y = pos.x(), pos.y()
        if not self.settings.accessibility.mouse_fallback and not self.workspace.modal_active():
            if kind == "down":
                self.toast("Mouse input is off — press ⌘⇧M to turn it back on")
            return
        button = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.RightButton: 2,
                  Qt.MouseButton.MiddleButton: 3}.get(e.button(), 1)
        if kind == "down":
            self.manager.pointer_down("mouse", x, y, button=button, modifiers=int(e.modifiers().value))
        elif kind == "move":
            self.manager.pointer_move("mouse", x, y)
        else:
            self.manager.pointer_up("mouse", x, y)

    def mouse_left(self) -> None:
        mp = self.manager.pointers["mouse"]
        mp.visible = False
        if self.manager.hover_source == "mouse" and not self.manager.busy("mouse"):
            self.workspace.leave()

    def wheel(self, e: QWheelEvent) -> None:
        pos = e.position()
        delta = e.angleDelta().y() or e.angleDelta().x()
        if delta:
            self.workspace.wheel(pos.x(), pos.y(), float(delta), int(e.modifiers().value))

    def key(self, e: QKeyEvent) -> bool:
        k = e.key()
        mods = e.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        # Global shortcuts that must always work, even with dialogs open.
        if ctrl and k == Qt.Key.Key_Q:
            self.close()
            return True
        if k == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            return True
        if ctrl and shift and k == Qt.Key.Key_M:
            a = self.settings.accessibility
            a.mouse_fallback = not a.mouse_fallback
            self.store.save(self.settings)
            self.toast(f"Mouse input {'on' if a.mouse_fallback else 'off'}")
            return True
        if self.workspace.modal_active():
            return self.workspace.key(e)
        if k == Qt.Key.Key_Escape:
            if self.launcher.is_open:
                self.launcher.close()
            elif self.manager.cancel():
                self.toast("Interaction cancelled")
            elif self.isFullScreen():
                self.showNormal()
            return True
        if ctrl:
            if k == Qt.Key.Key_O:
                self.toggle_launcher()
                return True
            if k == Qt.Key.Key_Comma:
                self.open_app("settings")
                return True
            if k == Qt.Key.Key_R:
                self.reset_workspace()
                return True
            if k == Qt.Key.Key_S and not shift:
                if not self.workspace.key(e):
                    self.save_layout()
                return True
            if k == Qt.Key.Key_W:
                if self.wm.focused:
                    self.wm.close(self.wm.focused)
                return True
            if k == Qt.Key.Key_M:
                if shift:
                    self.wm.restore_all()
                else:
                    self.wm.minimize_all()
                return True
            if k == Qt.Key.Key_D:
                t = self.settings.tracking
                t.show_skeleton = not t.show_skeleton
                t.show_zones = t.show_skeleton
                self.store.save(self.settings)
                self.toast(f"Tracking overlay {'on' if t.show_skeleton else 'off'}")
                return True
            if k == Qt.Key.Key_K:
                self.toggle_camera()
                return True
            if k == Qt.Key.Key_B:
                i = self.settings.interface
                i.background = BACKGROUND_MODES[(BACKGROUND_MODES.index(i.background) + 1) % len(BACKGROUND_MODES)]
                self.apply_settings()
                self.toast(f"Background: {i.background}")
                return True
            if k == Qt.Key.Key_L:
                i = self.settings.interface
                i.occlusion = OCCLUSION_MODES[(OCCLUSION_MODES.index(i.occlusion) + 1) % len(OCCLUSION_MODES)]
                self.apply_settings()
                self.toast(f"Occlusion: {i.occlusion}")
                return True
            if shift and k == Qt.Key.Key_C:
                self.start_calibration()
                return True
            if shift and k == Qt.Key.Key_A:
                self.arrange_windows()
                return True
            if Qt.Key.Key_1 <= k <= Qt.Key.Key_8:
                self.open_app(APPS[k - Qt.Key.Key_1].key)
                return True
        if k == Qt.Key.Key_Tab and not ctrl:
            wins = [w for w in self.wm.windows if w.interactive]
            if wins:
                self.wm.focus(wins[0] if not shift else wins[-2] if len(wins) > 1 else wins[0])
            return True
        if self.launcher.is_open:
            return True
        return self.workspace.key(e)

    def grab_frame(self):
        """Capture exactly what is on screen (used by --screenshot)."""
        if isinstance(self.view, QOpenGLWidget):
            return self.view.grabFramebuffer()
        return self.view.grab().toImage()

    # ============================================================ shutdown
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._shutting_down = True
        self.timer.stop()
        self.screen_lock.stop()
        try:
            self.save_layout(quiet=True)
            self.store.save(self.settings)
        except Exception:  # noqa: BLE001
            log.exception("Saving on exit failed")
        for w in list(self.wm.windows):
            try:
                w.app.on_close()
            except Exception:  # noqa: BLE001
                pass
        self.camera.stop()
        self.vision.stop()
        super().closeEvent(event)
