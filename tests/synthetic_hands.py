"""Synthetic MediaPipe-style hand landmarks for deterministic tests.

These are geometric stand-ins for model output. They let us test the
gesture, identity, pointer and interaction logic without a webcam, but they
say nothing about real-world tracking quality.
"""

from __future__ import annotations

import numpy as np

from airspace.vision.hand_tracker import HandObservation

# Canonical right hand, palm facing camera, fingers up. Units: ~palm size,
# origin at the wrist, +y pointing down (image convention).
_OPEN = np.array([
    (0.00, 0.00),  # 0 wrist
    (-0.28, -0.15), (-0.52, -0.33), (-0.70, -0.50), (-0.85, -0.64),  # thumb
    (-0.25, -0.90), (-0.28, -1.25), (-0.30, -1.48), (-0.31, -1.68),  # index
    (0.00, -0.95), (0.00, -1.33), (0.00, -1.58), (0.00, -1.80),  # middle
    (0.22, -0.90), (0.25, -1.24), (0.27, -1.46), (0.28, -1.64),  # ring
    (0.42, -0.80), (0.48, -1.05), (0.52, -1.22), (0.55, -1.38),  # pinky
], dtype=np.float64)

_FINGERS = {1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 16), 4: (17, 18, 19, 20)}
_PINCH_MEET = np.array([-0.52, -1.12])


def _folded(mcp: np.ndarray) -> list[np.ndarray]:
    return [mcp + (0, -0.28), mcp + (0, -0.12) + (0, 0.05), mcp + (0, 0.28)]


def make_hand(center=(0.5, 0.55), scale=0.11, pinch=0.0, curl=(0.0, 0.0, 0.0, 0.0),
              thumb_curl=0.0, aspect=16 / 9, handedness="Right", score=0.97,
              jitter=0.0, rng: np.random.Generator | None = None) -> HandObservation:
    """Build a hand.

    ``scale`` is the palm size as a fraction of frame *height*; ``center`` is the
    wrist position in normalised (u, v). ``pinch`` in [0, 1] closes thumb and
    index; ``curl`` folds index/middle/ring/pinky into the palm.
    """
    pts = _OPEN.copy()
    for f, c in zip((1, 2, 3, 4), curl):
        if c <= 0:
            continue
        mcp = pts[_FINGERS[f][0]]
        for idx, folded in zip(_FINGERS[f][1:], _folded(mcp)):
            pts[idx] = (1 - c) * pts[idx] + c * folded
    if thumb_curl > 0:
        target = np.array([-0.05, -0.75])
        for k, idx in enumerate((2, 3, 4)):
            w = thumb_curl * (0.4 + 0.3 * k)
            pts[idx] = (1 - w) * pts[idx] + w * (target + (0, 0.1 * (2 - k)))
    if pinch > 0:
        pts[4] = (1 - pinch) * pts[4] + pinch * (_PINCH_MEET + (-0.025, 0.025))
        pts[3] = (1 - pinch * 0.5) * pts[3] + pinch * 0.5 * (_PINCH_MEET + (-0.12, 0.15))
        pts[8] = (1 - pinch) * pts[8] + pinch * (_PINCH_MEET + (0.025, -0.025))
        pts[7] = (1 - pinch * 0.6) * pts[7] + pinch * 0.6 * (_PINCH_MEET + (0.12, -0.12))
    if handedness == "Left":
        pts[:, 0] *= -1
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = center[0] + pts[:, 0] * scale / aspect
    lm[:, 1] = center[1] + pts[:, 1] * scale
    if jitter and rng is not None:
        lm[:, :2] += rng.normal(0, jitter, size=(21, 2))
    return HandObservation(lm, handedness, score)


def open_palm(**kw) -> HandObservation:
    return make_hand(**kw)


def fist(**kw) -> HandObservation:
    return make_hand(curl=(1, 1, 1, 1), thumb_curl=1.0, **kw)


def pointing(**kw) -> HandObservation:
    return make_hand(curl=(0, 1, 1, 1), thumb_curl=0.7, **kw)
