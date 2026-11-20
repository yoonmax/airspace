"""Measure airspace's processing costs on this machine.

    python scripts/benchmark.py

Runs the real MediaPipe models and the real compositor on synthetic frames.
Synthetic frames contain no hands, so hand-landmark timing reflects the palm
detector path; with a hand in view MediaPipe also runs the landmark model and
tracks between frames. Use the in-app tracking inspector / monitor (or the
`perf` lines in the log) for live numbers with a real camera.
"""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def stats(samples: list[float]) -> str:
    s = sorted(samples)
    return f"median {statistics.median(s):6.2f} ms · p95 {s[int(len(s) * 0.95) - 1]:6.2f} ms"


def main() -> int:
    import cv2

    from airspace.vision.hand_tracker import HandTracker
    from airspace.vision.model_files import model_path
    from airspace.vision.segmentation import MaskBuilder, PersonSegmenter, mask_size
    from tests.synthetic_hands import make_hand

    print(f"airspace benchmark · {platform.platform()} · {platform.machine()} · Python {platform.python_version()}")
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(8)]
    tracker = HandTracker(model_path("hand_landmarker.task"), max_hands=2)
    seg = PersonSegmenter(model_path("selfie_segmenter.tflite"))
    hand_t, seg_t, pre_t, mask_t = [], [], [], []
    mb = MaskBuilder()
    hands = [make_hand(center=(0.45, 0.75), scale=0.18), make_hand(center=(0.65, 0.7), scale=0.18, handedness="Left")]
    for i in range(120):
        img = frames[i % len(frames)]
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        t1 = time.perf_counter()
        tracker.detect(rgb, i * 33)
        t2 = time.perf_counter()
        small = cv2.resize(rgb, mask_size(1280, 720), interpolation=cv2.INTER_AREA)
        person = seg.segment(small, i * 33)
        t3 = time.perf_counter()
        mb.build("balanced", rgb, hands, person)
        t4 = time.perf_counter()
        if i >= 10:
            pre_t.append((t1 - t0) * 1000)
            hand_t.append((t2 - t1) * 1000)
            seg_t.append((t3 - t2) * 1000)
            mask_t.append((t4 - t3) * 1000)
    print(f"  mirror + colour convert (1280×720)   {stats(pre_t)}")
    print(f"  hand landmarker (2 hands max)        {stats(hand_t)}")
    print(f"  selfie segmentation (320×180)        {stats(seg_t)}")
    print(f"  occlusion mask + foreground (2 hands){stats(mask_t)}")
    total = [a + b + c + d for a, b, c, d in zip(pre_t, hand_t, seg_t, mask_t)]
    print(f"  vision frame total                   {stats(total)}  → max ~{1000 / statistics.median(total):.0f} fps")

    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841 - must exist for painting
    from airspace.vision.tracking import HandTrackingProcessor
    from airspace.vision.vision_worker import VisionPacket
    from airspace.workspace.compositing import FrameLayers
    from airspace.workspace.rendering import Renderer, RenderInputs
    from tests.test_interactions import Rig

    proc = HandTrackingProcessor()
    track_t = []
    for i in range(300):
        t0 = time.perf_counter()
        proc.process(hands, i / 30, 16 / 9)
        track_t.append((time.perf_counter() - t0) * 1000)
    print(f"  tracking+gestures (UI thread, 2 hands){stats(track_t[20:])}")

    import tempfile

    rig = Rig(Path(tempfile.mkdtemp()))
    for key in ("objects", "music", "calculator"):
        rig.wm.open(key)
    for _ in range(20):
        rig.ws.tick(1 / 60)
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    fg, alpha = MaskBuilder().build("balanced", rgb, hands, None)
    pkt = VisionPacket(1, rgb, cv2.GaussianBlur(cv2.resize(rgb, (128, 72)), (0, 0), 2.2), hands, None, fg, alpha,
                       0.0, 0.0, 0.0, 0.0, 0.0, True)
    layers = FrameLayers()
    layers.update(pkt)
    img = QImage(2560, 1440, QImage.Format.Format_ARGB32_Premultiplied)
    paint_t = []
    renderer = Renderer()
    for i in range(60):
        rig.ws.tick(1 / 60)
        t0 = time.perf_counter()
        p = QPainter(img)
        p.scale(2, 2)
        ri = RenderInputs(rig.theme, rig.settings, rig.ws, rig.mgr, rig.shell.tracking_frame, rig.mapper, layers, True,
                          1280, 720)
        renderer.render(p, ri, 1 / 60)
        p.end()
        paint_t.append((time.perf_counter() - t0) * 1000)
    print(f"  compositor, 3 windows, 2560×1440 CPU raster {stats(paint_t[5:])}")
    obj = rig.wm.find("objects").app
    print(f"  3D viewer render + readback (last frame) {obj.render_ms:.2f} ms")
    tracker.close()
    seg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
