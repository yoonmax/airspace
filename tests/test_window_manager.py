import pytest

from airspace.interaction.drag_controller import WindowDragGrab
from airspace.interaction.events import PointerEvent, TwoHandInfo
from airspace.interaction.resize_controller import WindowResizeGrab, resize_rect, stretch_rect
from airspace.persistence.workspace_storage import WorkspaceStore
from airspace.workspace.geometry import Rect
from airspace.workspace.window_manager import WindowManager


class DummyApp:
    single_instance = True

    def __init__(self, app_type):
        self.app_type = app_type
        self.title = app_type
        self.icon = "apps"
        self.default_size = (400.0, 300.0)
        self.min_size = (200.0, 150.0)
        self.max_size = (900.0, 700.0)
        self.state = {}
        self.closed = False
        self.size = (0, 0)

    def resize(self, w, h):
        self.size = (w, h)

    def tick(self, dt):
        pass

    def save_state(self):
        return dict(self.state)

    def restore_state(self, s):
        self.state = dict(s)

    def on_close(self):
        self.closed = True

    def on_focus(self, f):
        pass


def make_wm():
    return WindowManager(DummyApp, Rect(0, 0, 1600, 1000), top_inset=60, titlebar_h=40)


def ev(x, y, t=0.0, cancelled=False):
    return PointerEvent(x, y, x, y, "hand", t, cancelled=cancelled)


def test_open_focus_and_z_order():
    wm = make_wm()
    a = wm.open("a", Rect(100, 100, 400, 300))
    b = wm.open("b", Rect(200, 150, 400, 300))
    assert wm.focused is b and b.z > a.z
    wm.focus(a)
    assert wm.focused is a and a.z > b.z
    assert wm.window_at(250, 200) is a


def test_single_instance_refocuses_existing():
    wm = make_wm()
    a = wm.open("a")
    wm.open("b")
    again = wm.open("a")
    assert again is a and wm.focused is a and len(wm.windows) == 2


def test_close_animates_then_removes_and_calls_app():
    wm = make_wm()
    a = wm.open("a")
    wm.close(a)
    assert a.app.closed and not a.interactive
    for _ in range(120):
        wm.tick(1 / 60)
    assert a not in wm.windows


def test_minimize_and_restore():
    wm = make_wm()
    a = wm.open("a")
    b = wm.open("b")
    wm.minimize(b)
    assert b.minimized and wm.focused is a and wm.minimized == [b]
    wm.restore(b)
    assert not b.minimized and wm.focused is b
    wm.minimize_all()
    assert len(wm.minimized) == 2
    wm.restore_all()
    assert not wm.minimized


def test_drag_moves_window_and_drop_keeps_position():
    wm = make_wm()
    w = wm.open("a", Rect(100, 100, 400, 300))
    g = WindowDragGrab(wm, w, ev(150, 120, 0.0), momentum=False)
    assert w.dragging
    for i in range(1, 11):
        g.move(ev(150 + 30 * i, 120 + 10 * i, i / 30))
    g.release(ev(450, 220, 0.4))
    assert not w.dragging
    assert (w.rect.x, w.rect.y) == (400, 200)
    for _ in range(30):
        wm.tick(1 / 60)
    assert (w.rect.x, w.rect.y) == (400, 200)  # no snap-back


def test_drag_cancel_restores_original_position():
    wm = make_wm()
    w = wm.open("a", Rect(100, 100, 400, 300))
    g = WindowDragGrab(wm, w, ev(150, 120))
    g.move(ev(600, 500, 0.1))
    g.cancel()
    assert (w.rect.x, w.rect.y) == (100, 100)


def test_constrain_keeps_titlebar_reachable():
    wm = make_wm()
    w = wm.open("a", Rect(100, 100, 400, 300))
    wm.move_to(w, -5000, -5000)
    assert w.rect.x >= -400 + wm.KEEP_VISIBLE - 1e-9
    assert w.rect.y >= wm.top_inset
    wm.move_to(w, 99999, 99999)
    assert w.rect.x <= 1600 - wm.KEEP_VISIBLE
    assert w.rect.y <= 1000 - w.titlebar_h - wm.bottom_inset


def test_resize_rect_respects_min_and_opposite_edges():
    start = Rect(100, 100, 400, 300)
    r = resize_rect(start, "se", 50, 40, 200, 150)
    assert (r.x, r.y, r.w, r.h) == (100, 100, 450, 340)
    r = resize_rect(start, "nw", 1000, 1000, 200, 150)
    assert (r.w, r.h) == (200, 150) and r.right == start.right and r.bottom == start.bottom
    r = resize_rect(start, "e", 5000, 0, 200, 150, max_w=900)
    assert r.w == 900


def test_resize_grab_updates_window_and_app_size():
    wm = make_wm()
    w = wm.open("a", Rect(100, 100, 400, 300))
    g = WindowResizeGrab(wm, w, "se", ev(500, 400))
    g.move(ev(600, 480))
    g.release(ev(600, 480))
    assert (w.rect.w, w.rect.h) == (500, 380)
    assert w.app.size == (500, 380 - w.titlebar_h)
    g2 = WindowResizeGrab(wm, w, "se", ev(600, 480))
    g2.move(ev(0, 0))
    assert (w.rect.w, w.rect.h) == (w.min_w, w.min_h)


def test_two_hand_stretch_scales_about_centre():
    start = Rect(100, 100, 400, 300)
    a = TwoHandInfo(300, 250, 200, 0)
    b = TwoHandInfo(300, 250, 300, 0)
    r = stretch_rect(start, a, b, 100, 100)
    assert (r.w, r.h) == (600, 450) and (r.cx, r.cy) == (300, 250)


def test_two_hand_resize_through_drag_grab():
    wm = make_wm()
    w = wm.open("a", Rect(300, 300, 400, 300))
    g = WindowDragGrab(wm, w, ev(350, 320))
    assert g.two_hand_begin(TwoHandInfo(500, 450, 200, 0))
    g.two_hand_update(TwoHandInfo(500, 450, 200, 0), TwoHandInfo(500, 450, 260, 0))
    assert w.resizing and w.rect.w == pytest.approx(520)
    g.two_hand_end()
    g.move(ev(360, 330, 0.5))  # continuing to drag doesn't jump
    assert w.rect.w == pytest.approx(520)
    g.release(ev(360, 330, 0.6))
    assert not w.resizing and not w.dragging


def test_layout_persistence_roundtrip(tmp_path):
    wm = make_wm()
    a = wm.open("a", Rect(160, 200, 400, 300))
    a.app.state = {"k": 1}
    b = wm.open("b", Rect(800, 400, 500, 350))
    wm.minimize(b)
    saved_order = [w.app_type for w in sorted(wm.windows, key=lambda w: w.z)]
    store = WorkspaceStore(tmp_path / "ws.json")
    store.save(wm.to_layout())

    wm2 = make_wm()
    wm2.apply_layout(store.load())
    ra = wm2.find("a")
    rb = wm2.find("b")
    assert (ra.rect.x, ra.rect.y, ra.rect.w) == (160, 200, 400)
    assert ra.app.state == {"k": 1}
    assert rb.minimized
    assert [w.app_type for w in sorted(wm2.windows, key=lambda w: w.z)] == saved_order


def test_bounds_change_scales_positions():
    wm = make_wm()
    w = wm.open("a", Rect(600, 400, 400, 200))
    cx = w.rect.cx / 1600
    wm.set_bounds(Rect(0, 0, 3200, 2000))
    assert w.rect.cx / 3200 == pytest.approx(cx)
    assert w.rect.w == 400  # sizes stay in logical pixels


def test_unknown_types_are_skipped_on_restore(tmp_path):
    from airspace.persistence.workspace_storage import WindowRecord, WorkspaceLayout

    wm = make_wm()
    wm.apply_layout(WorkspaceLayout([WindowRecord("a", 0.1, 0.1, 300, 200), WindowRecord("ghost", 0.2, 0.2, 300, 200)]),
                    known_types={"a"})
    assert [w.app_type for w in wm.windows] == ["a"]
