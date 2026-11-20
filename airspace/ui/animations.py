"""Physically-based motion primitives.

Springs are integrated with a fixed small substep for stability regardless of
frame time. When reduced motion is enabled, springs settle instantly.
"""

from __future__ import annotations

import math

REDUCED_MOTION = False


def set_reduced_motion(value: bool) -> None:
    global REDUCED_MOTION
    REDUCED_MOTION = value


class Spring:
    def __init__(self, value: float = 0.0, stiffness: float = 260.0, damping: float | None = None,
                 mass: float = 1.0) -> None:
        self.value = value
        self.target = value
        self.velocity = 0.0
        self.stiffness = stiffness
        # Critically damped by default: fast, no overshoot. Lower for a touch of bounce.
        self.damping = damping if damping is not None else 2.0 * math.sqrt(stiffness * mass)
        self.mass = mass

    def set(self, value: float) -> None:
        self.value = self.target = value
        self.velocity = 0.0

    def step(self, dt: float) -> float:
        if REDUCED_MOTION:
            self.value, self.velocity = self.target, 0.0
            return self.value
        dt = min(dt, 0.1)
        steps = max(1, int(math.ceil(dt / 0.004)))
        h = dt / steps
        for _ in range(steps):
            force = -self.stiffness * (self.value - self.target) - self.damping * self.velocity
            self.velocity += force / self.mass * h
            self.value += self.velocity * h
        if abs(self.value - self.target) < 1e-4 and abs(self.velocity) < 1e-3:
            self.value, self.velocity = self.target, 0.0
        return self.value

    @property
    def settled(self) -> bool:
        return self.value == self.target and self.velocity == 0.0


class Spring2D:
    def __init__(self, x: float = 0.0, y: float = 0.0, stiffness: float = 260.0, damping: float | None = None) -> None:
        self.x = Spring(x, stiffness, damping)
        self.y = Spring(y, stiffness, damping)

    def set(self, x: float, y: float) -> None:
        self.x.set(x)
        self.y.set(y)

    def target(self, x: float, y: float) -> None:
        self.x.target, self.y.target = x, y

    def step(self, dt: float) -> tuple[float, float]:
        return self.x.step(dt), self.y.step(dt)

    @property
    def value(self) -> tuple[float, float]:
        return self.x.value, self.y.value

    @property
    def settled(self) -> bool:
        return self.x.settled and self.y.settled


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t * t * t


def approach(current: float, target: float, rate: float, dt: float) -> float:
    """Frame-rate independent exponential approach (rate in 1/s)."""
    if REDUCED_MOTION:
        return target
    return target + (current - target) * math.exp(-rate * dt)
