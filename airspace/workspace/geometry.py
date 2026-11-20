"""Small geometry types used by the window manager (Qt-free for testability)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def contains(self, px: float, py: float, slop: float = 0.0) -> bool:
        return self.x - slop <= px <= self.right + slop and self.y - slop <= py <= self.bottom + slop

    def intersection_area(self, other: "Rect") -> float:
        w = min(self.right, other.right) - max(self.x, other.x)
        h = min(self.bottom, other.bottom) - max(self.y, other.y)
        return max(0.0, w) * max(0.0, h)

    def copy(self) -> "Rect":
        return Rect(self.x, self.y, self.w, self.h)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h
