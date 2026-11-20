"""Cursor placement: on the fingertip, and stable through a pinch."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from airspace.interaction.cursor import PinchStablePointer
from airspace.vision.filters import SMOOTHING_PRESETS
from airspace.vision.gesture_classifier import compute_features
from tests.synthetic_hands import make_hand

BASE = compute_features(make_hand().landmarks, 16 / 9)
DT = 1 / 30


def feat(tip, knuckle, ratio=0.9):
    return replace(BASE, index_tip=np.array(tip, float), knuckle=np.array(knuckle, float), pinch_ratio=ratio)


def run(ptr, frames, t=0.0):
    """frames: list of (features, pinched). Returns (last value, t)."""
    for f, pinched in frames:
        v = ptr.update(f, pinched, t)
        t += DT
    return v, t


def test_cursor_sits_on_fingertip_with_relaxed_hand():
    # A relaxed hand whose thumb rests near the index (ratio 0.45, as measured live)
    # must still put the cursor on the fingertip, not on a knuckle-based estimate.
    ptr = PinchStablePointer(SMOOTHING_PRESETS["balanced"])
    v, _ = run(ptr, [(feat((0.60, 0.40), (0.58, 0.48), ratio=0.45), False)] * 30)
    assert ptr.mode == "tip"
    assert np.allclose(v, (0.60, 0.40), atol=1e-3)


def test_click_lands_where_finger_pointed_before_closing():
    ptr = PinchStablePointer(SMOOTHING_PRESETS["balanced"])
    knuckle = (0.50, 0.50)
    _, t = run(ptr, [(feat((0.52, 0.40), knuckle), False)] * 20)
    # Fingers close over ~100 ms: the tip slides 0.03 towards the thumb.
    closing = [(feat((0.52 - 0.01 * i, 0.40 + 0.01 * i), knuckle, ratio=0.6 - 0.1 * i), False) for i in range(1, 4)]
    _, t = run(ptr, closing, t)
    v, t = run(ptr, [(feat((0.49, 0.43), knuckle, ratio=0.2), True)], t)  # pinch starts
    assert ptr.mode == "knuckle"
    assert np.allclose(v, (0.52, 0.40), atol=2e-3)
    v, t = run(ptr, [(feat((0.49, 0.43), knuckle, ratio=0.2), True)] * 10, t)
    assert np.allclose(v, (0.52, 0.40), atol=2e-3)  # held still → cursor still


def test_pinched_cursor_follows_moving_hand_and_returns_to_tip():
    ptr = PinchStablePointer(SMOOTHING_PRESETS["balanced"])
    _, t = run(ptr, [(feat((0.52, 0.40), (0.50, 0.50)), False)] * 20)
    _, t = run(ptr, [(feat((0.49, 0.43), (0.50, 0.50), 0.2), True)], t)
    # Drag: the whole hand moves right by 0.2 while pinched.
    drag = [(feat((0.49 + 0.01 * i, 0.43), (0.50 + 0.01 * i, 0.50), 0.2), True) for i in range(1, 21)]
    v, t = run(ptr, drag + [(feat((0.69, 0.43), (0.70, 0.50), 0.2), True)] * 15, t)
    assert np.allclose(v, (0.72, 0.40), atol=3e-3)  # knuckle + pre-closure offset
    # Release: the cursor glides back onto the fingertip.
    v, t = run(ptr, [(feat((0.72, 0.40), (0.70, 0.50)), False)] * 20, t)
    assert ptr.mode == "tip" and np.allclose(v, (0.72, 0.40), atol=2e-3)
