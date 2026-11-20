"""Stroke model for the canvas (Qt-free).

Points live in a fixed logical canvas space (``WIDTH`` × ``HEIGHT``) so a
drawing scales cleanly with its window. Stroke width responds to speed —
fast strokes thin out slightly, like a real brush — and is smoothed so it
never jumps between samples.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

WIDTH, HEIGHT = 1600.0, 1000.0
FORMAT = "airspace-canvas"


@dataclass
class Stroke:
    color: tuple[int, int, int, int]
    size: float
    eraser: bool = False
    points: list[tuple[float, float, float]] = field(default_factory=list)  # x, y, width

    def to_json(self) -> dict:
        return {"color": list(self.color), "size": self.size, "eraser": self.eraser,
                "points": [[round(x, 2), round(y, 2), round(w, 2)] for x, y, w in self.points]}

    @classmethod
    def from_json(cls, d: dict) -> "Stroke":
        color = tuple(int(c) for c in d.get("color", (255, 255, 255, 255)))[:4]
        if len(color) != 4:
            color = (255, 255, 255, 255)
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in d.get("points", []) if len(p) >= 3]
        return cls(color=color, size=float(d.get("size", 8)), eraser=bool(d.get("eraser", False)), points=pts)


class CanvasDocument:
    MIN_STEP = 1.2  # logical px between stored samples

    def __init__(self) -> None:
        self.strokes: list[Stroke] = []
        self._undo: list[tuple[str, object]] = []
        self._redo: list[tuple[str, object]] = []
        self.current: Stroke | None = None
        self._last: tuple[float, float, float] | None = None  # x, y, t
        self._width: float | None = None
        self.revision = 0  # bumps whenever committed content changes

    # ----------------------------------------------------------- drawing
    def begin(self, x: float, y: float, color: tuple[int, int, int, int], size: float, eraser: bool, t: float) -> None:
        self.end()
        self.current = Stroke(color=color, size=size, eraser=eraser)
        self._width = size
        self.current.points.append((x, y, size))
        self._last = (x, y, t)

    def add_point(self, x: float, y: float, t: float) -> None:
        if self.current is None or self._last is None:
            return
        lx, ly, lt = self._last
        d = math.hypot(x - lx, y - ly)
        if d < self.MIN_STEP:
            return
        dt = max(t - lt, 1e-3)
        speed = d / dt  # logical px per second
        size = self.current.size
        if self.current.eraser:
            target = size
        else:
            target = size * max(0.55, min(1.15, 1.18 - speed / 2600.0))
        assert self._width is not None
        self._width = 0.65 * self._width + 0.35 * target
        self.current.points.append((x, y, self._width))
        self._last = (x, y, t)

    def end(self) -> Stroke | None:
        stroke = self.current
        self.current = None
        self._last = None
        if stroke is None:
            return None
        if len(stroke.points) == 1:  # a tap draws a dot
            x, y, w = stroke.points[0]
            stroke.points.append((x + 0.01, y, w))
        self.strokes.append(stroke)
        self._undo.append(("stroke", stroke))
        self._redo.clear()
        self.revision += 1
        return stroke

    def cancel(self) -> None:
        self.current = None
        self._last = None

    # --------------------------------------------------------- history
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        kind, data = self._undo.pop()
        if kind == "stroke":
            self.strokes.remove(data)  # type: ignore[arg-type]
        elif kind == "clear":
            self.strokes = list(data)  # type: ignore[arg-type]
        self._redo.append((kind, data))
        self.revision += 1
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        kind, data = self._redo.pop()
        if kind == "stroke":
            self.strokes.append(data)  # type: ignore[arg-type]
        elif kind == "clear":
            self.strokes = []
        self._undo.append((kind, data))
        self.revision += 1
        return True

    def clear(self) -> bool:
        if not self.strokes:
            return False
        self._undo.append(("clear", list(self.strokes)))
        self._redo.clear()
        self.strokes = []
        self.revision += 1
        return True

    @property
    def is_empty(self) -> bool:
        return not self.strokes

    # ------------------------------------------------------------- io
    def to_json(self) -> dict:
        return {"format": FORMAT, "version": 1, "width": WIDTH, "height": HEIGHT,
                "strokes": [s.to_json() for s in self.strokes]}

    def load_json(self, data: dict) -> None:
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError("Not an airspace canvas project")
        sx = WIDTH / float(data.get("width", WIDTH))
        sy = HEIGHT / float(data.get("height", HEIGHT))
        strokes = []
        for raw in data.get("strokes", []):
            s = Stroke.from_json(raw)
            s.points = [(x * sx, y * sy, w) for x, y, w in s.points]
            strokes.append(s)
        self.strokes = strokes
        self._undo.clear()
        self._redo.clear()
        self.revision += 1

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json()), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: Path) -> None:
        self.load_json(json.loads(path.read_text(encoding="utf-8")))


def smoothed_segments(points: list[tuple[float, float, float]]):
    """Yield quadratic segments (start, control, end, width) through stroke midpoints.

    Drawing curves through midpoints with the samples as control points gives
    a smooth, continuous line even with sparse (camera-rate) input.
    """
    n = len(points)
    if n < 2:
        return
    if n == 2:
        (x0, y0, w0), (x1, y1, w1) = points
        yield (x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2), (x1, y1), (w0 + w1) / 2
        return
    prev = (points[0][0], points[0][1])
    for i in range(1, n - 1):
        x, y, w = points[i]
        nx, ny, _ = points[i + 1]
        mid = ((x + nx) / 2, (y + ny) / 2)
        yield prev, (x, y), mid, w
        prev = mid
    lx, ly, lw = points[-1]
    yield prev, ((prev[0] + lx) / 2, (prev[1] + ly) / 2), (lx, ly), lw
