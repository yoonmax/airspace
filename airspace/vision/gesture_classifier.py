"""Per-frame geometric hand features and static pose classification.

All measurements are normalised by a palm-size estimate so thresholds work
across hand sizes and distances from the camera. Coordinates are first
converted to an isotropic space (x scaled by the frame aspect ratio) so that a
distance means the same thing horizontally and vertically.

Static poses are only *evidence*; temporal logic (hysteresis, minimum
durations, cooldowns) lives in :mod:`gesture_state_machine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from airspace.vision.hand_tracker import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    THUMB_TIP,
    WRIST,
)


class Pose(str, Enum):
    NONE = "none"
    POINT = "point"
    PINCH = "pinch"
    OPEN_PALM = "open palm"
    FIST = "fist"
    OTHER = "hand"


@dataclass(frozen=True)
class HandFeatures:
    palm_scale: float  # isotropic units (fraction of frame height)
    palm_center: np.ndarray  # (2,) normalised u, v
    pinch_ratio: float  # thumb-tip ↔ index-tip distance / palm scale
    pinch_point: np.ndarray  # (2,) midpoint of thumb and index tips
    index_tip: np.ndarray  # (2,)
    knuckle: np.ndarray  # (2,) index MCP — stable during pinch closure
    straightness: tuple[float, ...]  # per finger, 1.0 = fully straight
    extended: tuple[bool, ...]  # thumb, index, middle, ring, pinky
    curled: tuple[bool, ...]
    tip_to_palm: tuple[float, ...]  # fingertip distance to palm centre / palm scale
    pose: Pose
    spread: float = 0.0  # index tip ↔ pinky tip / palm scale (finger fan)

    @property
    def extended_count(self) -> int:
        return sum(self.extended[1:])


def _iso(lm: np.ndarray, aspect: float) -> np.ndarray:
    """Isotropic 3D coordinates: x scaled by aspect so x/y units match."""
    out = lm.astype(np.float64).copy()
    out[:, 0] *= aspect
    out[:, 2] *= aspect  # MediaPipe z uses roughly the same scale as x
    return out


def compute_features(landmarks: np.ndarray, aspect: float) -> HandFeatures:
    """Compute features from (21, 3) normalised landmarks; ``aspect`` = width / height."""
    p = _iso(landmarks, aspect)
    p2 = p[:, :2]
    palm_idx = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    palm_center_iso = p2[palm_idx].mean(axis=0)
    # Palm scale: average of three bone spans that barely change with finger pose.
    palm_scale = float(np.mean([
        np.linalg.norm(p[WRIST] - p[INDEX_MCP]),
        np.linalg.norm(p[WRIST] - p[PINKY_MCP]),
        np.linalg.norm(p[INDEX_MCP] - p[PINKY_MCP]) * 1.6,
    ]))
    palm_scale = max(palm_scale, 1e-4)

    pinch_ratio = float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_TIP]) / palm_scale)

    straight: list[float] = []
    extended: list[bool] = []
    curled: list[bool] = []
    tip_palm: list[float] = []
    for f, chain in enumerate(FINGER_CHAINS):
        pts = p[list(chain)]
        seg = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        direct = float(np.linalg.norm(pts[-1] - pts[0]))
        s = direct / seg if seg > 1e-6 else 0.0
        straight.append(s)
        d_palm = float(np.linalg.norm(p2[chain[-1]] - palm_center_iso) / palm_scale)
        tip_palm.append(d_palm)
        if f == 0:
            # Thumb: extended when its tip is away from the index knuckle.
            d = float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_MCP]) / palm_scale)
            extended.append(d > 0.5 and s > 0.8)
            curled.append(d < 0.4)
        else:
            # Extended: nearly straight and pointing away from the wrist.
            d_tip = float(np.linalg.norm(p[chain[-1]] - p[WRIST]))
            d_pip = float(np.linalg.norm(p[chain[1]] - p[WRIST]))
            extended.append(s > 0.8 and d_tip > d_pip * 1.08)
            # "Curled" means folded into the palm, not merely bent: a pinching
            # index finger bends a lot but its tip stays away from the palm.
            curled.append(s < 0.6 and d_palm < 0.8)

    spread = float(np.linalg.norm(p2[INDEX_TIP] - p2[PINKY_TIP]) / palm_scale)
    pose = classify_pose(pinch_ratio, extended, curled, spread=spread)
    pc = palm_center_iso.copy()
    pc[0] /= aspect
    return HandFeatures(
        palm_scale=palm_scale,
        palm_center=pc,
        pinch_ratio=pinch_ratio,
        pinch_point=((landmarks[THUMB_TIP, :2] + landmarks[INDEX_TIP, :2]) / 2.0).astype(np.float64),
        index_tip=landmarks[INDEX_TIP, :2].astype(np.float64),
        knuckle=landmarks[INDEX_MCP, :2].astype(np.float64),
        straightness=tuple(straight),
        extended=tuple(extended),
        curled=tuple(curled),
        tip_to_palm=tuple(tip_palm),
        pose=pose,
        spread=spread,
    )


OPEN_PALM_MIN_GAP = 0.55


def classify_pose(pinch_ratio: float, extended: list[bool] | tuple[bool, ...],
                  curled: list[bool] | tuple[bool, ...], pinch_threshold: float = 0.35,
                  spread: float = 1.0) -> Pose:
    fingers_ext = sum(extended[1:])
    fingers_curled = sum(curled[1:])
    if fingers_curled == 4 and not extended[0]:
        return Pose.FIST
    if pinch_ratio < pinch_threshold:
        return Pose.PINCH
    # A presented palm: all fingers extended, thumb out, fingers fanned (not held together).
    # Thumb-index gap threshold measured on a real webcam session: open hands
    # read ~0.45-0.65 palm widths there, so 0.8 (synthetic-only) almost never fired.
    if fingers_ext == 4 and not curled[0] and pinch_ratio > OPEN_PALM_MIN_GAP and spread > 0.8:
        return Pose.OPEN_PALM
    # Real pointing hands often hold the other fingers loosely bent rather than
    # folded into the palm, so "index is the only extended finger" also counts.
    if extended[1] and (sum(curled[2:]) >= 2 or fingers_ext == 1):
        return Pose.POINT
    return Pose.OTHER
