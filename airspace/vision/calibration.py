"""Personal pinch-threshold calibration.

During calibration the user holds a pinch, then relaxes their hand. We record
normalised pinch ratios for both and place the hysteresis band between them:
closer to the pinched value for activation, further out for release.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ENTER = 0.30
DEFAULT_EXIT = 0.45


@dataclass(frozen=True)
class PinchCalibration:
    closed: float
    open: float
    enter: float
    exit: float
    valid: bool
    message: str = ""


def robust_level(samples: list[float] | np.ndarray) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    lo, hi = np.percentile(arr, [15, 85])
    core = arr[(arr >= lo) & (arr <= hi)]
    return float(np.median(core if core.size else arr))


def compute_pinch_thresholds(closed_samples: list[float], open_samples: list[float]) -> PinchCalibration:
    closed = robust_level(closed_samples)
    opened = robust_level(open_samples)
    if not np.isfinite(closed) or not np.isfinite(opened) or len(closed_samples) < 5 or len(open_samples) < 5:
        return PinchCalibration(closed, opened, DEFAULT_ENTER, DEFAULT_EXIT, False, "Not enough samples")
    span = opened - closed
    if span < 0.25:
        return PinchCalibration(closed, opened, DEFAULT_ENTER, DEFAULT_EXIT, False,
                                "Pinch and open hand looked too similar — using defaults")
    enter = closed + 0.30 * span
    exit_ = closed + 0.50 * span
    enter = float(np.clip(enter, 0.12, 0.6))
    exit_ = float(np.clip(max(exit_, enter + 0.08), enter + 0.08, 0.95))
    return PinchCalibration(closed, opened, enter, exit_, True, "Calibrated")
