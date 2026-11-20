"""Real-time signal filters for noisy landmark data.

The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) adapts its cutoff to
speed: slow movements are smoothed heavily (removing jitter) while fast
movements pass through with little lag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass(frozen=True)
class OneEuroParams:
    min_cutoff: float = 1.2  # Hz; lower = smoother at rest
    beta: float = 12.0  # speed coefficient; higher = less lag when moving
    d_cutoff: float = 1.0


SMOOTHING_PRESETS: dict[str, OneEuroParams] = {
    # Values are in normalised image units per second. Tuned by simulating a
    # fingertip moving at 0.1 / 0.3 / 1.0 widths/s plus landmark noise: "balanced"
    # trails a moving fingertip by ~3-5 px on a 1512 px view (was 6-15 px) for
    # ~0.5 px more jitter at rest.
    "responsive": OneEuroParams(min_cutoff=1.8, beta=60.0),
    "balanced": OneEuroParams(min_cutoff=1.0, beta=40.0),
    "steady": OneEuroParams(min_cutoff=0.6, beta=18.0),
}


class OneEuroFilter:
    """Vector One Euro filter over numpy arrays of any shape."""

    def __init__(self, params: OneEuroParams = OneEuroParams()) -> None:
        self.params = params
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    @property
    def value(self) -> np.ndarray | None:
        return self._x

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 1e-6:
            return self._x.copy()
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.params.d_cutoff, dt)
        assert self._dx is not None
        self._dx = a_d * dx + (1 - a_d) * self._dx
        speed = float(np.linalg.norm(self._dx))
        cutoff = self.params.min_cutoff + self.params.beta * speed
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class ExponentialSmoother:
    """Simple EMA, used for slowly-varying derived values (scales, masks)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def __call__(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None
