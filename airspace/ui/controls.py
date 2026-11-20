"""Retained-mode controls shared by every mini-app.

Controls are painted with QPainter and receive device-independent pointer
events, so mouse and hand input behave identically. Hit testing is slightly
forgiving for hand input (``slop``) because a fingertip is less precise than a
mouse.
"""

from __future__ import annotations

from typing import Callable, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

from airspace.interaction.events import CallbackGrab, Grab, InteractionState, PointerEvent
from airspace.ui.animations import Spring
from airspace.ui.icons import draw_icon
from airspace.ui.styles import Theme, mix, with_alpha

HAND_SLOP = 8.0


class Control:
    dwell_clickable = False

    def __init__(self, tooltip: str = "") -> None:
        self.rect = QRectF()
        self.visible = True
        self.enabled = True
        self.hovered = False
        self.pressed = False
        self.tooltip = tooltip
        self._hover = Spring(0.0, 500)
        self._press = Spring(0.0, 900)

    def contains(self, x: float, y: float, slop: float = 0.0) -> bool:
        return self.visible and self.rect.adjusted(-slop, -slop, slop, slop).contains(QPointF(x, y))

    def distance(self, x: float, y: float) -> float:
        r = self.rect
        dx = max(r.left() - x, 0.0, x - r.right())
        dy = max(r.top() - y, 0.0, y - r.bottom())
        return (dx * dx + dy * dy) ** 0.5

    def tick(self, dt: float) -> None:
        self._hover.target = 1.0 if (self.hovered and self.enabled) else 0.0
        self._press.target = 1.0 if self.pressed else 0.0
        self._hover.step(dt)
        self._press.step(dt)

    @property
    def hover_t(self) -> float:
        return self._hover.value

    @property
    def press_t(self) -> float:
        return self._press.value

    def press(self, ev: PointerEvent) -> Grab | None:
        return None

    def wheel(self, delta: float) -> bool:
        return False

    def paint(self, p: QPainter, theme: Theme) -> None:
        raise NotImplementedError


def _rounded(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


class Button(Control):
    dwell_clickable = True

    def __init__(self, label: str = "", icon: str | None = None, on_click: Callable[[], None] | None = None,
                 variant: str = "default", checked: Callable[[], bool] | None = None, tooltip: str = "",
                 font_size: float = 11.0, radius: float | None = None) -> None:
        super().__init__(tooltip or label)
        self.label = label
        self.icon = icon
        self.on_click = on_click
        self.variant = variant  # default | primary | ghost | danger | icon | key
        self.checked = checked
        self.font_size = font_size
        self.radius = radius
        self.color: QColor | None = None  # optional override for icon/text colour

    def is_checked(self) -> bool:
        return bool(self.checked and self.checked())

    def press(self, ev: PointerEvent) -> Grab | None:
        if not self.enabled:
            return None
        self.pressed = True
        slop = HAND_SLOP * 1.5 if ev.is_hand else 0.0

        def move(e: PointerEvent) -> None:
            self.pressed = self.contains(e.x, e.y, slop)

        def release(e: PointerEvent) -> None:
            inside = self.contains(e.x, e.y, slop)
            self.pressed = False
            if inside and not e.cancelled and self.on_click:
                self.on_click()

        def cancel() -> None:
            self.pressed = False

        return CallbackGrab(InteractionState.CLICK_PENDING, move, release, cancel)

    def paint(self, p: QPainter, theme: Theme) -> None:
        if not self.visible:
            return
        r = self.rect
        radius = self.radius if self.radius is not None else min(theme.radius_small, r.height() / 2)
        checked = self.is_checked()
        h, pr = self.hover_t, self.press_t
        if self.variant == "primary":
            bg = mix(theme.accent, QColor(255, 255, 255), 0.15 * h)
            bg = with_alpha(bg, 0.92 - 0.15 * pr)
            fg = QColor(18, 20, 24)
        elif self.variant == "danger":
            bg = with_alpha(theme.danger, 0.20 + 0.18 * h + 0.1 * pr)
            fg = theme.text
        elif self.variant in ("ghost", "icon"):
            a = 0.0 + 0.10 * h + 0.08 * pr + (0.14 if checked else 0.0)
            bg = QColor(255, 255, 255)
            bg.setAlphaF(a)
            fg = theme.accent if checked else mix(theme.text_secondary, theme.text, h)
        else:  # default / key
            base = theme.surface_active if checked else theme.surface
            bg = mix(base, theme.surface_hover, h) if not checked else base
            if pr:
                bg = mix(bg, theme.surface_active, pr)
            fg = theme.accent if checked and self.variant != "key" else theme.text
        if self.color is not None:
            fg = self.color
        if not self.enabled:
            fg = with_alpha(fg, 0.35)
        # Press feedback: a subtle scale-down.
        s = 1.0 - 0.04 * pr
        rr = QRectF(r.center().x() - r.width() * s / 2, r.center().y() - r.height() * s / 2, r.width() * s, r.height() * s)
        p.setPen(Qt.PenStyle.NoPen)
        if bg.alphaF() > 0.001:
            p.setBrush(bg)
            p.drawPath(_rounded(rr, radius))
        if self.variant in ("default", "key") or checked:
            border = theme.accent if checked else theme.border
            p.setPen(QPen(with_alpha(border, border.alphaF() * (0.7 + 0.5 * h)), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(_rounded(rr.adjusted(0.5, 0.5, -0.5, -0.5), radius))
        if self.icon and self.label:
            isz = min(rr.height() * 0.5, theme.s(18))
            p.setFont(theme.font(self.font_size, QFont.Weight.Medium))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(self.label)
            total = isz + theme.s(7) + tw
            x0 = rr.center().x() - total / 2
            draw_icon(p, self.icon, QRectF(x0, rr.center().y() - isz / 2, isz, isz), fg, 1.7)
            p.setPen(fg)
            p.drawText(QRectF(x0 + isz + theme.s(7), rr.top(), tw + 4, rr.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.label)
        elif self.icon:
            isz = min(rr.height(), rr.width()) * 0.52
            draw_icon(p, self.icon, QRectF(rr.center().x() - isz / 2, rr.center().y() - isz / 2, isz, isz), fg,
                      1.8 if isz > 14 else 1.5)
        elif self.label:
            weight = QFont.Weight.Medium if self.variant != "key" else QFont.Weight.Normal
            p.setFont(theme.font(self.font_size, weight))
            p.setPen(fg)
            p.drawText(rr, Qt.AlignmentFlag.AlignCenter, self.label)


class Slider(Control):
    def __init__(self, get: Callable[[], float], on_change: Callable[[float], None],
                 on_commit: Callable[[float], None] | None = None, minimum: float = 0.0, maximum: float = 1.0,
                 step: float = 0.0, label: str = "", fmt: Callable[[float], str] | None = None,
                 thin: bool = False, accent: QColor | None = None) -> None:
        super().__init__(label)
        self.get, self.on_change, self.on_commit = get, on_change, on_commit
        self.minimum, self.maximum, self.step = minimum, maximum, step
        self.label, self.fmt, self.thin, self.accent = label, fmt, thin, accent
        self.dragging = False
        self.secondary: Callable[[], float] | None = None  # e.g. buffered range

    def _track(self) -> QRectF:
        r = self.rect
        return QRectF(r.left() + 2, r.center().y() - 2, r.width() - 4, 4)

    def _value_at(self, x: float) -> float:
        tr = self._track()
        frac = (x - tr.left()) / max(tr.width(), 1.0)
        v = self.minimum + max(0.0, min(1.0, frac)) * (self.maximum - self.minimum)
        if self.step > 0:
            v = round(v / self.step) * self.step
        return max(self.minimum, min(self.maximum, v))

    def fraction(self) -> float:
        span = self.maximum - self.minimum
        return 0.0 if span <= 0 else max(0.0, min(1.0, (self.get() - self.minimum) / span))

    def press(self, ev: PointerEvent) -> Grab | None:
        if not self.enabled:
            return None
        self.pressed = self.dragging = True
        self.on_change(self._value_at(ev.x))

        def move(e: PointerEvent) -> None:
            self.on_change(self._value_at(e.x))

        def release(e: PointerEvent) -> None:
            self.pressed = self.dragging = False
            if self.on_commit:
                self.on_commit(self.get())

        def cancel() -> None:
            self.pressed = self.dragging = False

        return CallbackGrab(InteractionState.DRAGGING_SLIDER, move, release, cancel)

    def wheel(self, delta: float) -> bool:
        span = self.maximum - self.minimum
        v = max(self.minimum, min(self.maximum, self.get() + span * 0.04 * (1 if delta > 0 else -1)))
        self.on_change(v)
        if self.on_commit:
            self.on_commit(v)
        return True

    def paint(self, p: QPainter, theme: Theme) -> None:
        if not self.visible:
            return
        tr = self._track()
        h = self.hover_t
        th = 3.0 + 2.0 * max(h, self.press_t) if not self.thin else 2.0 + 2.0 * max(h, self.press_t)
        track = QRectF(tr.left(), tr.center().y() - th / 2, tr.width(), th)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(theme.text, 0.14))
        p.drawRoundedRect(track, th / 2, th / 2)
        if self.secondary:
            sec = QRectF(track)
            sec.setWidth(track.width() * max(0.0, min(1.0, self.secondary())))
            p.setBrush(with_alpha(theme.text, 0.12))
            p.drawRoundedRect(sec, th / 2, th / 2)
        frac = self.fraction()
        fill = QRectF(track)
        fill.setWidth(max(th, track.width() * frac))
        accent = self.accent or theme.accent
        p.setBrush(mix(accent, theme.text, 0.25 * h))
        p.drawRoundedRect(fill, th / 2, th / 2)
        knob_r = (3.5 if self.thin else 6.0) + 2.5 * max(h, self.press_t)
        kx = track.left() + track.width() * frac
        p.setBrush(with_alpha(QColor(0, 0, 0), 0.25))
        p.drawEllipse(QPointF(kx, track.center().y() + 1.2), knob_r + 1, knob_r + 1)
        p.setBrush(theme.text)
        p.drawEllipse(QPointF(kx, track.center().y()), knob_r, knob_r)
        if self.dragging and self.fmt:
            txt = self.fmt(self.get())
            p.setFont(theme.font(9.5, QFont.Weight.Medium))
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(txt) + 14
            bubble = QRectF(kx - w / 2, track.top() - 30, w, 20)
            p.setBrush(theme.glass_strong)
            p.setPen(QPen(theme.border, 1))
            p.drawRoundedRect(bubble, 6, 6)
            p.setPen(theme.text)
            p.drawText(bubble, Qt.AlignmentFlag.AlignCenter, txt)


class Segmented(Control):
    dwell_clickable = True

    def __init__(self, options: Sequence[str], get_index: Callable[[], int], on_select: Callable[[int], None],
                 icons: Sequence[str | None] | None = None, font_size: float = 10.5) -> None:
        super().__init__()
        self.options = list(options)
        self.icons = list(icons) if icons else [None] * len(self.options)
        self.get_index, self.on_select = get_index, on_select
        self.font_size = font_size
        self._hover_index = -1
        self._knob = Spring(float(max(0, get_index())), 420)

    def _segment_at(self, x: float) -> int:
        n = max(1, len(self.options))
        frac = (x - self.rect.left()) / max(self.rect.width(), 1.0)
        return max(0, min(n - 1, int(frac * n)))

    def update_hover(self, x: float, y: float) -> None:
        self._hover_index = self._segment_at(x) if self.hovered else -1

    def press(self, ev: PointerEvent) -> Grab | None:
        idx = self._segment_at(ev.x)
        self.pressed = True
        slop = HAND_SLOP if ev.is_hand else 0.0

        def release(e: PointerEvent) -> None:
            self.pressed = False
            if self.contains(e.x, e.y, slop) and not e.cancelled:
                self.on_select(self._segment_at(e.x) if not e.is_hand else idx)

        return CallbackGrab(InteractionState.CLICK_PENDING, None, release, lambda: setattr(self, "pressed", False))

    def tick(self, dt: float) -> None:
        super().tick(dt)
        self._knob.target = float(max(0, self.get_index()))
        self._knob.step(dt)

    def paint(self, p: QPainter, theme: Theme) -> None:
        if not self.visible or not self.options:
            return
        r = self.rect
        n = len(self.options)
        seg_w = r.width() / n
        rad = min(theme.radius_small, r.height() / 2)
        p.setPen(QPen(theme.border, 1))
        p.setBrush(with_alpha(QColor(0, 0, 0), 0.18))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), rad, rad)
        knob = QRectF(r.left() + seg_w * self._knob.value + 3, r.top() + 3, seg_w - 6, r.height() - 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.surface_active)
        p.drawRoundedRect(knob, rad - 2, rad - 2)
        current = self.get_index()
        p.setFont(theme.font(self.font_size, QFont.Weight.Medium))
        for i, (label, ico) in enumerate(zip(self.options, self.icons)):
            cell = QRectF(r.left() + seg_w * i, r.top(), seg_w, r.height())
            if i == self._hover_index and i != current:
                p.setBrush(with_alpha(theme.text, 0.06))
                p.drawRoundedRect(cell.adjusted(3, 3, -3, -3), rad - 2, rad - 2)
            color = theme.text if i == current else theme.text_secondary
            if ico:
                isz = min(cell.height() * 0.5, 18)
                if label:
                    draw_icon(p, ico, QRectF(cell.left() + 10, cell.center().y() - isz / 2, isz, isz), color)
                    p.setPen(color)
                    p.drawText(cell.adjusted(isz + 14, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
                else:
                    draw_icon(p, ico, QRectF(cell.center().x() - isz / 2, cell.center().y() - isz / 2, isz, isz), color)
            else:
                p.setPen(color)
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter, label)


class Toggle(Control):
    dwell_clickable = True

    def __init__(self, label: str, get: Callable[[], bool], set_: Callable[[bool], None], description: str = "") -> None:
        super().__init__(label)
        self.label, self.get, self.set, self.description = label, get, set_, description
        self._knob = Spring(1.0 if get() else 0.0, 500)

    def press(self, ev: PointerEvent) -> Grab | None:
        self.pressed = True
        slop = HAND_SLOP * 1.5 if ev.is_hand else 0.0

        def release(e: PointerEvent) -> None:
            self.pressed = False
            if self.contains(e.x, e.y, slop) and not e.cancelled:
                self.set(not self.get())

        return CallbackGrab(InteractionState.CLICK_PENDING, None, release, lambda: setattr(self, "pressed", False))

    def tick(self, dt: float) -> None:
        super().tick(dt)
        self._knob.target = 1.0 if self.get() else 0.0
        self._knob.step(dt)

    def paint(self, p: QPainter, theme: Theme) -> None:
        if not self.visible:
            return
        r = self.rect
        if self.hover_t > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(with_alpha(theme.text, 0.05 * self.hover_t))
            p.drawRoundedRect(r, 8, 8)
        sw, sh = theme.s(38), theme.s(22)
        sx = r.right() - sw - 8
        sy = r.center().y() - sh / 2
        k = self._knob.value
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(mix(with_alpha(theme.text, 0.16), theme.accent, k))
        p.drawRoundedRect(QRectF(sx, sy, sw, sh), sh / 2, sh / 2)
        p.setBrush(QColor(250, 249, 246) if k > 0.5 else mix(theme.text, QColor(250, 249, 246), k))
        kr = sh / 2 - 3
        p.drawEllipse(QPointF(sx + sh / 2 + (sw - sh) * k, sy + sh / 2), kr, kr)
        p.setPen(theme.text)
        p.setFont(theme.font(11))
        text_r = QRectF(r.left() + 8, r.top(), r.width() - sw - 24, r.height())
        if self.description:
            p.drawText(text_r.adjusted(0, 0, 0, -r.height() * 0.42), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, self.label)
            p.setPen(theme.text_tertiary)
            p.setFont(theme.font(9.5))
            p.drawText(text_r.adjusted(0, r.height() * 0.55, 0, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self.description)
        else:
            p.drawText(text_r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.label)


class Swatch(Control):
    dwell_clickable = True

    def __init__(self, color: QColor, selected: Callable[[], bool], on_click: Callable[[], None]) -> None:
        super().__init__()
        self.color, self.selected, self.on_click = color, selected, on_click

    def press(self, ev: PointerEvent) -> Grab | None:
        self.pressed = True
        slop = HAND_SLOP if ev.is_hand else 0.0

        def release(e: PointerEvent) -> None:
            self.pressed = False
            if self.contains(e.x, e.y, slop) and not e.cancelled:
                self.on_click()

        return CallbackGrab(InteractionState.CLICK_PENDING, None, release, lambda: setattr(self, "pressed", False))

    def paint(self, p: QPainter, theme: Theme) -> None:
        r = self.rect
        c = r.center()
        rad = min(r.width(), r.height()) / 2 - 3
        rad *= 1.0 + 0.08 * self.hover_t - 0.06 * self.press_t
        if self.selected():
            p.setPen(QPen(theme.text, 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rad + 3, rad + 3)
        p.setPen(QPen(with_alpha(QColor(255, 255, 255), 0.25), 1))
        p.setBrush(self.color)
        p.drawEllipse(c, rad, rad)


class ControlSet:
    """Hover/press routing for a flat list of controls."""

    def __init__(self) -> None:
        self.controls: list[Control] = []
        self.hovered: Control | None = None

    def add(self, control: Control) -> Control:
        self.controls.append(control)
        return control

    def clear(self) -> None:
        self.controls.clear()
        self.hovered = None

    def find(self, x: float, y: float, hand: bool = False) -> Control | None:
        slop = HAND_SLOP if hand else 0.0
        best, best_d = None, float("inf")
        for c in reversed(self.controls):
            if not c.visible or not c.enabled:
                continue
            if c.contains(x, y):
                return c
            if slop and c.contains(x, y, slop):
                d = c.distance(x, y)
                if d < best_d:
                    best, best_d = c, d
        return best

    def hover(self, x: float, y: float, hand: bool = False) -> Control | None:
        c = self.find(x, y, hand)
        if c is not self.hovered:
            if self.hovered:
                self.hovered.hovered = False
            self.hovered = c
            if c:
                c.hovered = True
        if isinstance(c, Segmented):
            c.update_hover(x, y)
        return c

    def leave(self) -> None:
        if self.hovered:
            self.hovered.hovered = False
            if isinstance(self.hovered, Segmented):
                self.hovered.update_hover(-1, -1)
        self.hovered = None

    def press(self, ev: PointerEvent) -> Grab | None:
        c = self.find(ev.x, ev.y, ev.is_hand)
        return c.press(ev) if c else None

    def wheel(self, x: float, y: float, delta: float) -> bool:
        c = self.find(x, y)
        return bool(c and c.wheel(delta))

    def tick(self, dt: float) -> None:
        for c in self.controls:
            c.tick(dt)

    def paint(self, p: QPainter, theme: Theme) -> None:
        for c in self.controls:
            if c.visible:
                c.paint(p, theme)
