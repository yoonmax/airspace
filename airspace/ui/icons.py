"""Geometric line icons drawn with QPainter on a 24-unit grid.

Icons are vector so they stay crisp at any window scale and in fullscreen.
"""

from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen

IconFn = Callable[[QPainter, QPainterPath], None]
_ICONS: dict[str, Callable[[QPainterPath], None]] = {}
_FILLED: set[str] = set()


def icon(name: str, filled: bool = False):
    def deco(fn: Callable[[QPainterPath], None]):
        _ICONS[name] = fn
        if filled:
            _FILLED.add(name)
        return fn

    return deco


def draw_icon(p: QPainter, name: str, rect: QRectF, color: QColor, stroke: float = 1.7) -> None:
    fn = _ICONS.get(name)
    if fn is None:
        return
    path = QPainterPath()
    fn(path)
    size = min(rect.width(), rect.height())
    s = size / 24.0
    p.save()
    p.translate(rect.center().x() - 12 * s, rect.center().y() - 12 * s)
    p.scale(s, s)
    if name in _FILLED:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
    else:
        pen = QPen(color, stroke / max(s, 1e-3))  # constant pixel width at any size
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.restore()


def has_icon(name: str) -> bool:
    return name in _ICONS


# --------------------------------------------------------------- helpers
def _line(path: QPainterPath, *pts: tuple[float, float]) -> None:
    path.moveTo(*pts[0])
    for pt in pts[1:]:
        path.lineTo(*pt)


def _circle(path: QPainterPath, cx: float, cy: float, r: float) -> None:
    path.addEllipse(QPointF(cx, cy), r, r)


def _rrect(path: QPainterPath, x: float, y: float, w: float, h: float, r: float) -> None:
    path.addRoundedRect(QRectF(x, y, w, h), r, r)


# ---------------------------------------------------------------- icons
@icon("close")
def _(p):
    _line(p, (6, 6), (18, 18)); _line(p, (18, 6), (6, 18))


@icon("minimize")
def _(p):
    _line(p, (6, 12), (18, 12))


@icon("plus")
def _(p):
    _line(p, (12, 5), (12, 19)); _line(p, (5, 12), (19, 12))


@icon("minus")
def _(p):
    _line(p, (5, 12), (19, 12))


@icon("check")
def _(p):
    _line(p, (5, 12.5), (10, 17.5), (19, 7))


@icon("chevron-left")
def _(p):
    _line(p, (15, 5), (8, 12), (15, 19))


@icon("chevron-right")
def _(p):
    _line(p, (9, 5), (16, 12), (9, 19))


@icon("play", filled=True)
def _(p):
    p.moveTo(8, 5); p.lineTo(19, 12); p.lineTo(8, 19); p.closeSubpath()


@icon("pause", filled=True)
def _(p):
    p.addRoundedRect(QRectF(6.5, 5, 4, 14), 1.2, 1.2)
    p.addRoundedRect(QRectF(13.5, 5, 4, 14), 1.2, 1.2)


@icon("next", filled=True)
def _(p):
    p.moveTo(5, 6); p.lineTo(14, 12); p.lineTo(5, 18); p.closeSubpath()
    p.addRoundedRect(QRectF(15.5, 6, 3, 12), 1, 1)


@icon("prev", filled=True)
def _(p):
    p.moveTo(19, 6); p.lineTo(10, 12); p.lineTo(19, 18); p.closeSubpath()
    p.addRoundedRect(QRectF(5.5, 6, 3, 12), 1, 1)


@icon("grid")
def _(p):
    for x in (4.5, 13.5):
        for y in (4.5, 13.5):
            _rrect(p, x, y, 6, 6, 1.8)


@icon("apps")
def _(p):
    for x in (5, 12, 19):
        for y in (5, 12, 19):
            _circle(p, x, y, 1.6)


@icon("settings")
def _(p):
    _circle(p, 12, 12, 3.2)
    for i in range(8):
        a = i * math.pi / 4
        _line(p, (12 + 6.0 * math.cos(a), 12 + 6.0 * math.sin(a)), (12 + 8.6 * math.cos(a), 12 + 8.6 * math.sin(a)))
    _circle(p, 12, 12, 6.0)


@icon("fullscreen")
def _(p):
    _line(p, (4, 9), (4, 4), (9, 4)); _line(p, (15, 4), (20, 4), (20, 9))
    _line(p, (20, 15), (20, 20), (15, 20)); _line(p, (9, 20), (4, 20), (4, 15))


@icon("exit-fullscreen")
def _(p):
    _line(p, (9, 4), (9, 9), (4, 9)); _line(p, (15, 4), (15, 9), (20, 9))
    _line(p, (20, 15), (15, 15), (15, 20)); _line(p, (4, 15), (9, 15), (9, 20))


@icon("music")
def _(p):
    _line(p, (9, 17), (9, 5), (19, 3), (19, 15))
    _circle(p, 6.5, 17.5, 2.5); _circle(p, 16.5, 15.5, 2.5)


@icon("brush")
def _(p):
    _line(p, (20, 4), (11, 13))
    p.moveTo(11, 13); p.cubicTo(7, 11, 5, 15, 6, 17); p.cubicTo(5, 19, 4, 20, 3.5, 20.5)
    p.cubicTo(8, 21, 12, 19, 11, 13)


@icon("image")
def _(p):
    _rrect(p, 3.5, 5, 17, 14, 2.5)
    _line(p, (4, 16.5), (9, 11.5), (13, 15.5), (15.5, 13), (20, 17.5))
    _circle(p, 15.5, 9, 1.6)


@icon("cube")
def _(p):
    _line(p, (12, 3), (20, 7.5), (20, 16.5), (12, 21), (4, 16.5), (4, 7.5), (12, 3))
    _line(p, (4, 7.5), (12, 12), (20, 7.5)); _line(p, (12, 12), (12, 21))


@icon("chart")
def _(p):
    _line(p, (3, 16), (8, 11), (12, 14), (17, 7), (21, 10))
    _line(p, (3, 20.5), (21, 20.5))


@icon("calculator")
def _(p):
    _rrect(p, 5, 3, 14, 18, 2.5)
    _rrect(p, 7.5, 5.5, 9, 4, 1)
    for x in (8.5, 12, 15.5):
        for y in (13, 17):
            _circle(p, x, y, 0.6)


@icon("hand")
def _(p):
    p.moveTo(7, 13); p.lineTo(7, 7.5); p.cubicTo(7, 6, 9.4, 6, 9.4, 7.5); p.lineTo(9.4, 11)
    p.moveTo(9.4, 11); p.lineTo(9.4, 5); p.cubicTo(9.4, 3.5, 11.8, 3.5, 11.8, 5); p.lineTo(11.8, 11)
    p.moveTo(11.8, 11); p.lineTo(11.8, 5.8); p.cubicTo(11.8, 4.3, 14.2, 4.3, 14.2, 5.8); p.lineTo(14.2, 11)
    p.moveTo(14.2, 11); p.lineTo(14.2, 8); p.cubicTo(14.2, 6.5, 16.6, 6.5, 16.6, 8); p.lineTo(16.6, 14)
    p.cubicTo(16.6, 18.5, 14, 21, 11, 21); p.cubicTo(8.5, 21, 7, 19.5, 5.5, 17); p.lineTo(4, 14.5)
    p.cubicTo(3.3, 13.3, 5.5, 12, 7, 13.5)


@icon("eraser")
def _(p):
    _line(p, (9, 20), (4.5, 15.5), (14.5, 5.5), (20, 11), (11, 20), (9, 20))
    _line(p, (7.5, 12.5), (13, 18)); _line(p, (11, 20), (20, 20))


@icon("pen")
def _(p):
    _line(p, (4, 20), (5, 15.5), (16, 4.5), (19.5, 8), (8.5, 19), (4, 20))
    _line(p, (13.5, 7), (17, 10.5))


@icon("undo")
def _(p):
    _line(p, (8, 5), (4, 9), (8, 13))
    p.moveTo(4, 9); p.lineTo(14, 9); p.cubicTo(18, 9, 20, 11.5, 20, 14.5); p.cubicTo(20, 17.5, 18, 20, 14, 20); p.lineTo(9, 20)


@icon("redo")
def _(p):
    _line(p, (16, 5), (20, 9), (16, 13))
    p.moveTo(20, 9); p.lineTo(10, 9); p.cubicTo(6, 9, 4, 11.5, 4, 14.5); p.cubicTo(4, 17.5, 6, 20, 10, 20); p.lineTo(15, 20)


@icon("save")
def _(p):
    _line(p, (12, 4), (12, 15)); _line(p, (7.5, 10.5), (12, 15), (16.5, 10.5))
    _line(p, (4, 15), (4, 20), (20, 20), (20, 15))


@icon("folder")
def _(p):
    _line(p, (3.5, 7), (3.5, 18.5), (20.5, 18.5), (20.5, 8.5), (11.5, 8.5), (9.5, 5.5), (3.5, 5.5), (3.5, 7))


@icon("trash")
def _(p):
    _line(p, (4, 7), (20, 7)); _line(p, (9, 7), (9.5, 4), (14.5, 4), (15, 7))
    _line(p, (6, 7), (7, 20), (17, 20), (18, 7)); _line(p, (10, 11), (10, 16)); _line(p, (14, 11), (14, 16))


@icon("zoom-in")
def _(p):
    _circle(p, 10.5, 10.5, 6.5); _line(p, (15.5, 15.5), (20, 20))
    _line(p, (10.5, 7.5), (10.5, 13.5)); _line(p, (7.5, 10.5), (13.5, 10.5))


@icon("zoom-out")
def _(p):
    _circle(p, 10.5, 10.5, 6.5); _line(p, (15.5, 15.5), (20, 20)); _line(p, (7.5, 10.5), (13.5, 10.5))


@icon("rotate")
def _(p):
    p.moveTo(19, 12); p.arcTo(QRectF(5, 5, 14, 14), 0, 270)
    _line(p, (12, 2.5), (15, 5), (12, 7.5))


@icon("fit")
def _(p):
    _rrect(p, 6.5, 6.5, 11, 11, 2)
    _line(p, (3, 7), (3, 3), (7, 3)); _line(p, (17, 3), (21, 3), (21, 7))
    _line(p, (21, 17), (21, 21), (17, 21)); _line(p, (7, 21), (3, 21), (3, 17))


@icon("reset")
def _(p):
    p.moveTo(5, 12); p.arcTo(QRectF(5, 5, 14, 14), 180, -300)
    _line(p, (4, 7), (5, 12), (9.5, 10.5))


@icon("camera")
def _(p):
    _rrect(p, 3, 7, 18, 12, 2.5); _circle(p, 12, 13, 3.5)
    _line(p, (8, 7), (9.5, 4.5), (14.5, 4.5), (16, 7))


@icon("light")
def _(p):
    _circle(p, 12, 12, 4)
    for i in range(8):
        a = i * math.pi / 4
        _line(p, (12 + 6.5 * math.cos(a), 12 + 6.5 * math.sin(a)), (12 + 8.8 * math.cos(a), 12 + 8.8 * math.sin(a)))


@icon("wireframe")
def _(p):
    _line(p, (12, 3), (20, 7.5), (20, 16.5), (12, 21), (4, 16.5), (4, 7.5), (12, 3))
    _line(p, (4, 7.5), (20, 16.5)); _line(p, (20, 7.5), (4, 16.5)); _line(p, (12, 3), (12, 21))


@icon("material")
def _(p):
    _circle(p, 12, 12, 8)
    p.moveTo(8, 9); p.cubicTo(9, 7, 11, 6.5, 12.5, 7)


@icon("layers")
def _(p):
    _line(p, (12, 4), (21, 8.5), (12, 13), (3, 8.5), (12, 4))
    _line(p, (3, 12.5), (12, 17), (21, 12.5)); _line(p, (3, 16.5), (12, 21), (21, 16.5))


@icon("volume")
def _(p):
    _line(p, (4, 9.5), (8, 9.5), (13, 5), (13, 19), (8, 14.5), (4, 14.5), (4, 9.5))
    p.moveTo(16, 9); p.cubicTo(17.6, 10.5, 17.6, 13.5, 16, 15)
    p.moveTo(18.5, 6.5); p.cubicTo(21.5, 9.5, 21.5, 14.5, 18.5, 17.5)


@icon("queue")
def _(p):
    _line(p, (4, 6), (16, 6)); _line(p, (4, 11), (16, 11)); _line(p, (4, 16), (11, 16))
    _line(p, (15, 14), (15, 20)); _line(p, (12, 17), (18, 17))


@icon("shuffle")
def _(p):
    _line(p, (4, 7), (8, 7), (16, 17), (20, 17)); _line(p, (4, 17), (8, 17), (16, 7), (20, 7))
    _line(p, (17.5, 4.5), (20, 7), (17.5, 9.5)); _line(p, (17.5, 14.5), (20, 17), (17.5, 19.5))


@icon("eye")
def _(p):
    p.moveTo(2.5, 12); p.cubicTo(6, 5.5, 18, 5.5, 21.5, 12); p.cubicTo(18, 18.5, 6, 18.5, 2.5, 12)
    _circle(p, 12, 12, 3)


@icon("target")
def _(p):
    _circle(p, 12, 12, 8); _circle(p, 12, 12, 3.5)
    _line(p, (12, 1.5), (12, 5)); _line(p, (12, 19), (12, 22.5)); _line(p, (1.5, 12), (5, 12)); _line(p, (19, 12), (22.5, 12))


@icon("arrange")
def _(p):
    _rrect(p, 3.5, 4, 8, 7, 1.5); _rrect(p, 12.5, 4, 8, 7, 1.5); _rrect(p, 3.5, 13, 17, 7, 1.5)


@icon("backspace")
def _(p):
    _line(p, (8.5, 5.5), (20.5, 5.5), (20.5, 18.5), (8.5, 18.5), (2.5, 12), (8.5, 5.5))
    _line(p, (11.5, 9), (17, 15)); _line(p, (17, 9), (11.5, 15))


@icon("info")
def _(p):
    _circle(p, 12, 12, 9); _line(p, (12, 11), (12, 17)); _circle(p, 12, 7.6, 0.4)


@icon("pan")
def _(p):
    _line(p, (12, 3), (12, 21)); _line(p, (3, 12), (21, 12))
    _line(p, (9.5, 5.5), (12, 3), (14.5, 5.5)); _line(p, (9.5, 18.5), (12, 21), (14.5, 18.5))
    _line(p, (5.5, 9.5), (3, 12), (5.5, 14.5)); _line(p, (18.5, 9.5), (21, 12), (18.5, 14.5))


@icon("sparkle", filled=True)
def _(p):
    p.moveTo(12, 3); p.cubicTo(12.8, 9, 15, 11.2, 21, 12); p.cubicTo(15, 12.8, 12.8, 15, 12, 21)
    p.cubicTo(11.2, 15, 9, 12.8, 3, 12); p.cubicTo(9, 11.2, 11.2, 9, 12, 3)


@icon("dot", filled=True)
def _(p):
    _circle(p, 12, 12, 4)


@icon("restore")
def _(p):
    _rrect(p, 4, 8, 12, 12, 2); _line(p, (8, 8), (8, 4), (20, 4), (20, 16), (16, 16))


@icon("power")
def _(p):
    _line(p, (12, 3), (12, 11))
    p.moveTo(7, 6.5); p.arcTo(QRectF(4, 4.5, 16, 16), 125, 290)
