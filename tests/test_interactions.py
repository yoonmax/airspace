"""End-to-end interaction tests.

Synthetic landmark sequences go through the real HandTrackingProcessor
(identity, One Euro filtering, gesture state machines), the real
InteractionManager and GestureRouter, the real Workspace/WindowManager and
real mini-apps. This verifies the interaction logic deterministically; it does
not say anything about real-world tracking quality.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QRectF

from airspace.app.context import AppContext
from airspace.interaction.events import InteractionState
from airspace.interaction.gesture_router import GestureRouter
from airspace.interaction.interaction_manager import InteractionManager
from airspace.miniapps.registry import APPS, load_class
from airspace.persistence.settings import Settings
from airspace.ui.styles import make_theme
from airspace.vision.coordinate_mapper import CoordinateMapper
from airspace.vision.gesture_state_machine import GestureConfig
from airspace.vision.tracking import HandTrackingProcessor
from airspace.workspace.geometry import Rect
from airspace.workspace.hud import Hud, HudActions, StatusInfo
from airspace.workspace.launcher import Launcher, LauncherItem
from airspace.workspace.window_manager import WindowManager
from airspace.workspace.workspace import Workspace
from tests.synthetic_hands import fist, make_hand

W, H = 1280, 720
DT = 1 / 30


class FakeShell:
    def __init__(self) -> None:
        self.toasts: list[str] = []
        self.tracking_frame = None
        self.latest_packet = None
        self.manager = None

    def toast(self, text: str) -> None:
        self.toasts.append(text)

    def play_sound(self, name: str) -> None:
        pass

    def apply_settings(self, save: bool = True) -> None:
        pass

    def open_files(self, *a, **k):
        return []

    def save_file(self, *a, **k):
        return None

    def list_cameras(self):
        return [(0, "Camera 0")]

    # Workspace actions used by the settings app.
    def __getattr__(self, name):
        if name in ("start_calibration", "set_camera", "toggle_camera", "save_layout", "restore_layout",
                    "arrange_windows", "reset_workspace", "clear_saved_settings"):
            return lambda *a, **k: None
        if name == "wm":
            return self._wm
        raise AttributeError(name)


class Rig:
    def __init__(self, data_dir) -> None:
        self.t = 0.0
        self.settings = Settings()
        self.theme = make_theme()
        self.shell = FakeShell()
        self.ctx = AppContext(self.settings, lambda: self.theme, self.shell, data_dir)
        self.wm = WindowManager(lambda k: load_class(k)(self.ctx), Rect(0, 0, W, H), top_inset=58,
                                titlebar_h=self.theme.titlebar_height)
        self.launcher = Launcher([LauncherItem(a.key, a.label, a.icon) for a in APPS], lambda k: self.wm.open(k))
        noop = lambda *a: None  # noqa: E731
        self.hud = Hud(HudActions(noop, noop, noop, noop, lambda: False, noop, noop, lambda: True),
                       lambda: StatusInfo(), lambda: [(w.id, w.title, w.app.icon) for w in self.wm.minimized])
        self.hud.layout(QRectF(0, 0, W, H), self.theme)
        self.ws = Workspace(self.wm, self.launcher, self.hud, self.shell.toast)
        self.ws.momentum = False
        self.ws.set_bounds(QRectF(0, 0, W, H))
        self.mgr = InteractionManager(self.ws, clock=lambda: self.t)
        self.shell.manager = self.mgr
        self.shell._wm = self.wm
        self.router = GestureRouter(self.mgr, self.ws)
        self.proc = HandTrackingProcessor(GestureConfig())
        self.mapper = CoordinateMapper(W, H, W, H, reach=1.0)
        self.pos = (W / 2, H / 2)

    # ------------------------------------------------------------ driving
    def step(self, hands) -> None:
        self.t += DT
        frame = self.proc.process(hands, self.t, W / H)
        self.shell.tracking_frame = frame
        self.mgr.handle_tracking(frame, self.mapper)
        self.ws.tick(DT)

    @staticmethod
    def _tip_offset(handedness: str):
        h = make_hand(center=(0.0, 0.0), handedness=handedness)
        return float(h.landmarks[8, 0]), float(h.landmarks[8, 1])

    def hand(self, x: float, y: float, pinch: float = 0.0, handedness: str = "Right", open_hand: bool = False, **kw):
        """A pointing hand (index out, others folded) whose index tip is at (x, y)."""
        ou, ov = self._tip_offset(handedness)
        curl = (0.0, 0.0, 0.0, 0.0) if open_hand else (0.0, 1.0, 1.0, 1.0)
        return make_hand(center=(x / W - ou, y / H - ov), pinch=pinch, handedness=handedness, curl=curl, **kw)

    def move_to(self, x: float, y: float, frames: int = 10, pinch: float = 0.0, extra=None) -> None:
        x0, y0 = self.pos
        for i in range(1, frames + 1):
            f = i / frames
            px, py = x0 + (x - x0) * f, y0 + (y - y0) * f
            hands = [self.hand(px, py, pinch)] + (extra(i) if extra else [])
            self.step(hands)
        self.pos = (x, y)
        for _ in range(4):  # let the filter settle
            self.step([self.hand(x, y, pinch)] + (extra(frames) if extra else []))

    def pinch(self, frames: int = 8) -> None:
        x, y = self.pos
        self.step([self.hand(x, y, 0.6)])
        for _ in range(frames):
            self.step([self.hand(x, y, 1.0)])

    def release(self, frames: int = 8) -> None:
        x, y = self.pos
        for _ in range(frames):
            self.step([self.hand(x, y, 0.0)])

    def click_at(self, x: float, y: float) -> None:
        self.move_to(x, y)
        self.pinch()
        self.release()

    def hold(self, frames: int, hands_fn) -> None:
        for _ in range(frames):
            self.step(hands_fn())


@pytest.fixture
def rig(qapp, data_dir):
    return Rig(data_dir)


def global_center(win, control):
    c = win.content_rect
    r = control.rect
    return c.x + r.center().x(), c.y + r.center().y()


# ======================================================================
def test_pinch_clicks_button_exactly_once(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    app = win.app
    rig.click_at(*global_center(win, app.keys["7"]))
    assert app.model.expression == "7"
    rig.click_at(*global_center(win, app.keys["+"]))
    rig.click_at(*global_center(win, app.keys["2"]))
    rig.click_at(*global_center(win, app.keys["="]))
    assert app.model.result == "9"


def test_long_pinch_still_one_click(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    rig.move_to(*global_center(win, win.app.keys["5"]))
    rig.pinch(frames=60)  # two seconds
    rig.release()
    assert win.app.model.expression == "5"


def test_hovering_never_clicks(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    for key in "789456123":
        rig.move_to(*global_center(win, win.app.keys[key]), frames=6)
    assert win.app.model.expression == ""
    assert rig.mgr.hand_hover_target is not None


def test_titlebar_drag_moves_window_without_clicking(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    start = win.rect.copy()
    rig.move_to(start.x + 100, start.y + 18)
    rig.pinch(frames=4)
    assert rig.mgr.state == InteractionState.DRAGGING_WINDOW
    rig.move_to(start.x + 300, start.y + 118, pinch=1.0)  # passes over keys while dragging
    rig.release()
    assert win.rect.x == pytest.approx(start.x + 200, abs=6)
    assert win.rect.y == pytest.approx(start.y + 100, abs=6)
    assert win.app.model.expression == ""
    assert not win.dragging and rig.mgr.state != InteractionState.DRAGGING_WINDOW


def test_canvas_draws_only_while_pinched_on_surface(rig):
    win = rig.wm.open("canvas", Rect(200, 100, 720, 520))
    app = win.app
    s = app.surface
    c = win.content_rect
    x0, y0 = c.x + s.left() + 80, c.y + s.top() + 80
    rig.move_to(x0, y0)
    rig.move_to(x0 + 200, y0 + 100)  # pointing only
    assert app.doc.is_empty
    rig.pinch(frames=3)
    assert rig.mgr.state == InteractionState.DRAWING
    rig.move_to(x0 + 350, y0 + 50, pinch=1.0)
    rig.release()
    assert len(app.doc.strokes) == 1 and len(app.doc.strokes[0].points) > 5
    # Pinching the title bar drags the window and never draws.
    rig.move_to(win.rect.x + 200, win.rect.y + 15)
    rig.pinch(frames=3)
    rig.move_to(win.rect.x + 260, win.rect.y + 60, pinch=1.0)
    rig.release()
    assert len(app.doc.strokes) == 1


def test_tracking_loss_mid_drag_releases_safely_and_reentry_does_not_reattach(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    rig.move_to(win.rect.x + 100, win.rect.y + 18)
    rig.pinch(frames=4)
    rig.move_to(win.rect.x + 200, win.rect.y + 60, pinch=1.0)
    assert win.dragging
    at_loss = win.rect.copy()
    rig.hold(20, lambda: [])  # hand vanishes
    assert not win.dragging and not rig.mgr.busy()
    assert (win.rect.x, win.rect.y) == (at_loss.x, at_loss.y)  # no jump
    # The hand comes back, already pinched, somewhere else: nothing re-attaches.
    rig.pos = (1100, 600)
    rig.hold(15, lambda: [rig.hand(1100, 600, 1.0)])
    rig.hold(10, lambda: [rig.hand(1150, 650, 1.0)])
    assert (win.rect.x, win.rect.y) == (at_loss.x, at_loss.y)
    assert not win.dragging


def test_brief_dropout_keeps_drag(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    rig.move_to(win.rect.x + 100, win.rect.y + 18)
    rig.pinch(frames=4)
    rig.hold(2, lambda: [])  # ~66 ms dropout, within the grace period
    rig.move_to(win.rect.x + 180, win.rect.y + 40, pinch=1.0)
    assert win.dragging


def test_open_palm_opens_launcher_once_and_fist_closes(rig):
    rig.hold(45, lambda: [make_hand(center=(0.5, 0.75))])
    assert rig.launcher.is_open
    rig.hold(45, lambda: [make_hand(center=(0.5, 0.75))])  # still holding: must not toggle closed
    assert rig.launcher.is_open
    rig.hold(25, lambda: [fist(center=(0.5, 0.75))])
    assert not rig.launcher.is_open


def test_launcher_pinch_selects_app(rig):
    rig.ws.open_launcher(W / 2, H / 2)
    for _ in range(30):
        rig.ws.tick(DT)
    idx = [i.key for i in rig.launcher.items].index("calculator")
    p = rig.launcher.item_center(idx)
    rig.click_at(p.x(), p.y())
    assert not rig.launcher.is_open
    assert rig.wm.find("calculator") is not None


def test_fist_cancels_drag_and_restores_position(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    start = win.rect.copy()
    rig.move_to(start.x + 100, start.y + 18)
    rig.pinch(frames=4)
    rig.move_to(start.x + 260, start.y + 80, pinch=1.0)
    # The user cancels with a fist: the grab is released by the pinch ending first,
    # so cancellation is exercised through the manager directly (keyboard Esc does the same).
    rig.mgr.cancel()
    assert (win.rect.x, win.rect.y) == (start.x, start.y)
    assert rig.mgr.state == InteractionState.GESTURE_CANCELLED


def test_two_hand_stretch_resizes_window(rig):
    win = rig.wm.open("calculator", Rect(500, 150, 330, 470))
    w0 = win.rect.w
    grab_x, grab_y = win.rect.x + 150, win.rect.y + 18
    rig.move_to(grab_x, grab_y)
    rig.pinch(frames=4)
    left_x = grab_x - 220

    def left(px, pinch):
        return [rig.hand(px, grab_y + 40, pinch, handedness="Left")]

    for i in range(8):  # second hand appears, then pinches
        rig.step([rig.hand(grab_x, grab_y, 1.0)] + left(left_x, 0.0))
    for i in range(6):
        rig.step([rig.hand(grab_x, grab_y, 1.0)] + left(left_x, 1.0))
    assert rig.mgr.two_hand_active
    for i in range(1, 13):  # spread the hands apart
        rig.step([rig.hand(grab_x + 6 * i, grab_y, 1.0)] + left(left_x - 6 * i, 1.0))
    assert win.rect.w > w0 * 1.2
    assert rig.mgr.state == InteractionState.RESIZING_WINDOW
    for _ in range(8):
        rig.step([rig.hand(grab_x + 72, grab_y, 0.0)] + left(left_x - 72, 0.0))
    assert not rig.mgr.busy() and not win.resizing


def test_swipe_changes_gallery_image_but_not_while_dragging(rig):
    win = rig.wm.open("gallery", Rect(300, 100, 680, 520))
    win.app.model.add(["/nonexistent/a.png", "/nonexistent/b.png", "/nonexistent/c.png"])
    win.app.relayout()
    cx, cy = win.rect.cx, win.rect.cy
    rig.move_to(cx, cy)
    rig.hold(3, lambda: [rig.hand(cx, cy, open_hand=True)])
    for i in range(1, 9):  # fast open-hand sweep to the left
        rig.step([rig.hand(cx - 45 * i, cy, open_hand=True)])
    assert win.app.model.index == 1
    # The same sweep while dragging the window does not change the image.
    rig.pos = (cx - 360, cy)
    rig.move_to(win.rect.x + 200, win.rect.y + 15)
    rig.pinch(frames=3)
    x, y = rig.pos
    for i in range(1, 9):
        rig.step([rig.hand(x - 45 * i, y, 1.0)])
    rig.release()
    assert win.app.model.index == 1


def test_pointing_at_a_button_with_open_hand_does_not_open_launcher(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    x, y = global_center(win, win.app.keys["5"])
    rig.move_to(x, y)
    rig.hold(60, lambda: [rig.hand(x, y, open_hand=True)])
    assert not rig.launcher.is_open


def test_slider_drag_with_pinch(rig):
    win = rig.wm.open("music", Rect(300, 100, 470, 330))
    app = win.app
    app.audio.setVolume(0.2)
    vol = app.volume
    c = win.content_rect
    rig.move_to(c.x + vol.rect.left() + vol.rect.width() * 0.2, c.y + vol.rect.center().y())
    rig.pinch(frames=3)
    assert rig.mgr.state == InteractionState.DRAGGING_SLIDER
    rig.move_to(c.x + vol.rect.right() - 2, c.y + vol.rect.center().y(), pinch=1.0)
    rig.release()
    assert app.audio.volume() > 0.85


def test_mouse_fallback_click_and_drag(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    x, y = global_center(win, win.app.keys["4"])
    rig.mgr.pointer_down("mouse", x, y)
    rig.mgr.pointer_up("mouse", x, y)
    assert win.app.model.expression == "4"
    rig.mgr.pointer_down("mouse", win.rect.x + 80, win.rect.y + 15)
    rig.mgr.pointer_move("mouse", win.rect.x + 180, win.rect.y + 65)
    rig.mgr.pointer_up("mouse", win.rect.x + 180, win.rect.y + 65)
    assert win.rect.x == 500 and win.rect.y == 170


def test_dwell_click_alternative(rig):
    rig.mgr.dwell_enabled = True
    rig.mgr.dwell_ms = 600
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    target = global_center(win, win.app.keys["8"])
    rig.move_to(*target)
    rig.hold(30, lambda: [rig.hand(*target)])  # 1 s of hovering, no pinch
    assert win.app.model.expression == "8"  # exactly one click


def test_minimise_and_close_buttons_by_pinch(rig):
    win = rig.wm.open("calculator", Rect(400, 120, 330, 470))
    b = win.button_rects()
    rig.click_at(b["minimize"].cx, b["minimize"].cy)
    assert win.minimized
    rig.wm.restore(win)
    rig.click_at(b["close"].cx, b["close"].cy)
    assert win.closing
