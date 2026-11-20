"""Vision pipeline tests using the real MediaPipe models on synthetic frames.

Synthetic frames contain no hands, so these tests verify plumbing, threading,
latest-frame dropping and timing — not detection quality.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from airspace.camera.frame_buffer import Frame, LatestFrameBuffer
from airspace.vision.model_files import model_path
from airspace.vision.segmentation import MaskBuilder, build_hand_mask, person_bbox
from airspace.workspace.geometry import Rect
from airspace.workspace.layout_manager import GRID_H, GRID_W, PlacementModel
from tests.synthetic_hands import make_hand

HAVE_MODELS = model_path("hand_landmarker.task").exists()


def test_latest_frame_buffer_drops_stale_frames():
    buf = LatestFrameBuffer()
    for _ in range(5):
        buf.put(Frame(np.zeros((2, 2, 3), np.uint8), time.perf_counter(), buf.next_index()))
    f = buf.get_newer(-1, timeout=0.01)
    assert f is not None and f.index == 5
    assert buf.dropped == 4
    assert buf.get_newer(5, timeout=0.01) is None  # nothing newer


@pytest.mark.skipif(not HAVE_MODELS, reason="models not downloaded")
def test_vision_worker_processes_newest_frames(qapp):
    from airspace.vision.vision_worker import VisionConfig, VisionWorker

    buf = LatestFrameBuffer()
    worker = VisionWorker(buf, model_path("hand_landmarker.task"), model_path("selfie_segmenter.tflite"))
    worker.set_config(VisionConfig(segmentation_interval=1, occlusion="balanced", need_person_mask=True))
    packets = []
    worker.packetReady.connect(packets.append)
    worker.start()
    rng = np.random.default_rng(0)
    try:
        deadline = time.perf_counter() + 4.0
        while time.perf_counter() < deadline and len(packets) < 30:
            img = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
            buf.put(Frame(img, time.perf_counter(), buf.next_index()))
            qapp.processEvents()
            time.sleep(1 / 60)  # camera faster than 30 fps → some frames must be skipped
        for _ in range(20):
            qapp.processEvents()
            time.sleep(0.01)
    finally:
        worker.stop()
    assert len(packets) >= 10
    p = packets[-1]
    assert p.frame_rgb.shape == (720, 1280, 3)
    assert p.mirrored and p.hands == []
    assert p.person_mask is not None and p.person_mask.dtype == np.float32
    assert p.processed_ts >= p.capture_ts
    indices = [pk.frame_index for pk in packets]
    assert indices == sorted(indices)  # never processes an older frame after a newer one


def test_hand_mask_covers_hand_region():
    hand = make_hand(center=(0.5, 0.7), scale=0.2)
    m = build_hand_mask([hand], (320, 180))
    tip = hand.landmarks[8]
    assert m[int(tip[1] * 180), int(tip[0] * 320)] > 0.5
    assert m[5, 5] == 0.0


def test_mask_builder_modes():
    frame = np.full((180, 320, 3), 128, np.uint8)
    person = np.zeros((180, 320), np.float32)
    person[40:180, 100:220] = 1.0
    hand = make_hand(center=(0.5, 0.8), scale=0.2)
    mb = MaskBuilder()
    fg, alpha = mb.build("clear", frame, [hand], person)
    assert fg is None
    fg, alpha = mb.build("balanced", frame, [hand], person)
    assert fg is not None and fg.rgba.shape[2] == 4
    # Balanced keeps a clear "lens" at the fingertip so the target stays visible.
    tip = hand.landmarks[8]
    lens = alpha[int(tip[1] * 180), int(tip[0] * 320)]
    assert lens < alpha.max()
    fg_c, alpha_c = mb.build("cinematic", frame, [], person)
    assert fg_c is not None and alpha_c[100, 160] > 0.9 and alpha_c[100, 20] < 0.05


def test_person_bbox():
    m = np.zeros((90, 160), np.float32)
    m[20:80, 50:110] = 1
    u0, v0, u1, v1 = person_bbox(m)
    assert u0 == pytest.approx(50 / 160) and v1 == pytest.approx(80 / 90)


def test_placement_avoids_person_and_windows():
    pm = PlacementModel()
    grid = np.zeros((GRID_H, GRID_W))
    grid[6:, 20:44] = 1.0  # person in the centre
    pm.update_person(grid)
    b = Rect(0, 0, 1600, 900)
    r = pm.place(420, 320, b, 60, 20, [])
    assert r.right <= 20 / GRID_W * 1600 + 40 or r.x >= 44 / GRID_W * 1600 - 40
    r2 = pm.place(420, 320, b, 60, 20, [r])
    assert r2.intersection_area(r) < 0.1 * r.w * r.h
