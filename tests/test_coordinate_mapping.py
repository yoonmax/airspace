import pytest

from airspace.vision.coordinate_mapper import CoordinateMapper, mirror_u


def test_mirroring_flips_horizontal_axis():
    assert mirror_u(0.2) == pytest.approx(0.8)
    assert mirror_u(mirror_u(0.37)) == pytest.approx(0.37)


def test_identity_when_frame_matches_view():
    m = CoordinateMapper(1280, 720, 1280, 720)
    assert m.norm_to_view(0, 0) == (0, 0)
    assert m.norm_to_view(1, 1) == (1280, 720)
    assert m.norm_to_view(0.5, 0.5) == (640, 360)


@pytest.mark.parametrize("frame,view", [((640, 480), (1280, 720)), ((1920, 1080), (900, 900)),
                                        ((1280, 720), (2560, 1600)), ((1280, 720), (1000, 1400))])
def test_cover_preserves_aspect_and_fills_view(frame, view):
    m = CoordinateMapper(*frame, *view)
    r = m.image_rect
    assert r.w / r.h == pytest.approx(frame[0] / frame[1])
    assert r.x <= 1e-9 and r.y <= 1e-9
    assert r.x + r.w >= view[0] - 1e-9 and r.y + r.h >= view[1] - 1e-9
    # centred crop
    assert r.x == pytest.approx((view[0] - r.w) / 2)
    assert r.y == pytest.approx((view[1] - r.h) / 2)


@pytest.mark.parametrize("frame,view", [((640, 480), (1280, 720)), ((1920, 1080), (3024, 1964))])
def test_view_norm_roundtrip(frame, view):
    m = CoordinateMapper(*frame, *view)
    for u, v in [(0.1, 0.2), (0.5, 0.5), (0.93, 0.71)]:
        x, y = m.norm_to_view(u, v)
        assert m.view_to_norm(x, y) == pytest.approx((u, v))


def test_pointer_with_unit_reach_sits_on_fingertip():
    m = CoordinateMapper(1280, 720, 1600, 1000, reach=1.0)
    assert m.pointer_to_view(0.4, 0.6) == pytest.approx(m.norm_to_view(0.4, 0.6))


def test_reach_expands_about_centre_and_reaches_edges():
    m = CoordinateMapper(1280, 720, 1280, 720, reach=1.5)
    assert m.pointer_to_view(0.5, 0.5) == pytest.approx((640, 360))
    # A hand at 1/6 from the edge already reaches the screen edge (clamped).
    x, _ = m.pointer_to_view(1 / 6, 0.5)
    assert x == pytest.approx(m.edge_margin)


def test_screen_edge_clamping():
    m = CoordinateMapper(1280, 720, 1280, 720, reach=1.0, edge_margin=2.0)
    assert m.pointer_to_view(-0.3, 1.4) == (2.0, 718.0)
    assert m.pointer_to_view(1.2, -0.1) == (1278.0, 2.0)


def test_fullscreen_resize_keeps_relative_position():
    m = CoordinateMapper(1280, 720, 1280, 720)
    small = m.pointer_to_view(0.25, 0.75)
    m.configure(view_w=2560, view_h=1440)
    big = m.pointer_to_view(0.25, 0.75)
    assert big == pytest.approx((small[0] * 2, small[1] * 2))


def test_pointer_inverse():
    m = CoordinateMapper(1280, 720, 1500, 900, reach=1.3)
    u, v = m.view_to_pointer(*m.pointer_to_view(0.42, 0.58, clamp=False))
    assert (u, v) == pytest.approx((0.42, 0.58))
