"""Model-level tests for canvas, gallery, music and objects."""

import numpy as np
import pytest

from airspace.miniapps.canvas.brush_engine import CanvasDocument, smoothed_segments
from airspace.miniapps.gallery.gallery import GalleryModel
from airspace.miniapps.gallery.image_viewer import ViewTransform, fit_scale
from airspace.miniapps.music.queue import PlayQueue, format_time
from airspace.miniapps.objects.viewer import GLITCH_PX, MAX_STEP, TransformState


# ------------------------------------------------------------------ canvas
def draw(doc, pts, t0=0.0):
    doc.begin(*pts[0], (255, 255, 255, 255), 8.0, False, t0)
    for i, (x, y) in enumerate(pts[1:], 1):
        doc.add_point(x, y, t0 + i / 30)
    return doc.end()


def test_stroke_undo_redo():
    doc = CanvasDocument()
    draw(doc, [(10, 10), (50, 50), (90, 20)])
    draw(doc, [(100, 100), (150, 120)])
    assert len(doc.strokes) == 2
    assert doc.undo() and len(doc.strokes) == 1
    assert doc.redo() and len(doc.strokes) == 2
    doc.undo()
    draw(doc, [(1, 1), (5, 5)])  # new stroke clears redo
    assert not doc.can_redo


def test_clear_is_undoable():
    doc = CanvasDocument()
    draw(doc, [(10, 10), (50, 50)])
    assert doc.clear() and doc.is_empty
    assert doc.undo() and len(doc.strokes) == 1
    assert doc.redo() and doc.is_empty


def test_tap_makes_a_dot_and_sparse_points_are_filtered():
    doc = CanvasDocument()
    s = draw(doc, [(10, 10)])
    assert len(s.points) == 2
    s = draw(doc, [(10, 10), (10.2, 10.1), (10.3, 10.2), (40, 40)])
    assert len(s.points) == 2  # sub-pixel jitter is dropped


def test_velocity_sensitive_width_stays_bounded():
    doc = CanvasDocument()
    s = draw(doc, [(i * 60.0, 0) for i in range(20)])  # very fast stroke
    widths = [w for _, _, w in s.points]
    assert min(widths) >= 8 * 0.55 - 1e-6 and max(widths) <= 8 * 1.15 + 1e-6
    assert widths[-1] < widths[0]


def test_smoothed_segments_are_continuous():
    pts = [(0, 0, 4), (10, 0, 4), (20, 10, 4), (30, 10, 4)]
    segs = list(smoothed_segments(pts))
    for a, b in zip(segs, segs[1:]):
        assert a[2] == b[0]  # each segment starts where the previous ended
    assert segs[0][0] == (0, 0) and segs[-1][2] == (30, 10)


def test_canvas_save_load(tmp_path):
    doc = CanvasDocument()
    draw(doc, [(10, 10), (50, 50), (90, 20)])
    path = tmp_path / "d.json"
    doc.save(path)
    doc2 = CanvasDocument()
    doc2.load(path)
    assert len(doc2.strokes) == 1 and len(doc2.strokes[0].points) == len(doc.strokes[0].points)
    (tmp_path / "bad.json").write_text('{"format": "other"}')
    with pytest.raises(ValueError):
        doc2.load(tmp_path / "bad.json")


# ----------------------------------------------------------------- gallery
def test_gallery_navigation_wraps_and_removal():
    g = GalleryModel()
    assert g.add(["a.png", "b.png", "c.png", "a.png"]) == 3
    assert g.current == "a.png"
    g.prev()
    assert g.current == "c.png"
    g.next()
    g.next()
    assert g.current == "b.png"
    assert g.remove_current() == "b.png" and g.current == "c.png"
    g.select(0)
    assert g.current == "a.png"


def test_view_transform_zoom_about_point_and_limits():
    v = ViewTransform()
    v.zoom_by(2.0, 100, 0)
    assert v.zoom == 2.0 and v.pan_x == pytest.approx(-100)
    for _ in range(50):
        v.zoom_by(2.0)
    assert v.zoom == v.MAX_ZOOM
    v.rotate_90()
    v.rotate_90(-1)
    assert v.rotation == 0
    v.reset()
    assert (v.zoom, v.pan_x, v.angle) == (1.0, 0.0, 0.0)


def test_fit_scale_handles_rotation():
    assert fit_scale(400, 200, 400, 400, 0) == pytest.approx(1.0)
    assert fit_scale(400, 200, 400, 400, 90) == pytest.approx(1.0)
    assert fit_scale(400, 200, 200, 400, 90) == pytest.approx(1.0)


# ------------------------------------------------------------------- music
def test_queue_transport():
    q = PlayQueue()
    assert q.add(["a.mp3", "b.wav", "notes.txt", "a.mp3"]) == 2
    assert q.current.path == "a.mp3"
    assert q.next().path == "b.wav"
    assert q.next().path == "a.mp3"  # repeat wraps
    q.repeat = False
    q.select(1)
    assert q.next() is None
    assert q.prev().path == "a.mp3"
    q.remove(0)
    assert q.current.path == "b.wav"
    q.remove(0)
    assert q.current is None and q.index == -1


def test_format_time():
    assert format_time(0) == "0:00"
    assert format_time(61_000) == "1:01"
    assert format_time(3_725_000) == "1:02:05"


# ----------------------------------------------------------------- objects
def test_rotation_steps_are_clamped():
    tf = TransformState()
    before = tf.rot.copy()
    tf.rotate(100.0, 0.0)  # a huge glitchy step
    angle = np.arccos(np.clip((np.trace(before.T @ tf.rot) - 1) / 2, -1, 1))
    assert angle <= MAX_STEP + 1e-6
    assert np.allclose(tf.rot @ tf.rot.T, np.eye(3), atol=1e-9)


def test_inertia_decays_to_rest():
    tf = TransformState()
    tf.velocity = np.array([4.0, -2.0])
    steps = 0
    while tf.step_inertia(1 / 60) and steps < 1000:
        steps += 1
    assert steps < 300 and not np.any(tf.velocity)


def test_zoom_limits():
    tf = TransformState()
    for _ in range(100):
        tf.zoom(2.0)
    assert tf.distance == 1.8
    assert GLITCH_PX > 50
