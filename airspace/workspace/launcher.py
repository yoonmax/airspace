"""The radial application launcher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient

from airspace.interaction.events import CallbackGrab, Grab, InteractionState, PointerEvent
from airspace.interaction.hit_testing import item_angle, radial_index
from airspace.ui.animations import Spring
from airspace.ui.icons import draw_icon
from airspace.ui.styles import Theme, mix, with_alpha


@dataclass(frozen=True)
class LauncherItem:
    key: str
    label: str
    icon: str


class Launcher:
    def __init__(self, items: list[LauncherItem], on_launch: Callable[[str], None]) -> None:
        self.items = items
        self.on_launch = on_launch
        self.cx = self.cy = 0.0
        self.open_t = Spring(0.0, 340, damping=30)
        self.is_open = False
        self.hover_index: int | None = None
        self.pressed_index: int | None = None
        self.hub_hover = False
        self._item_hover = [Spring(0.0, 520) for _ in items]
        self.bounds = QRectF(0, 0, 1280, 720)
        self.scale = 1.0

    # ------------------------------------------------------------- geometry
    @property
    def radius(self) -> float:
        return 150.0 * self.scale

    @property
    def item_radius(self) -> float:
        return 36.0 * self.scale

    @property
    def hub_radius(self) -> float:
        return 34.0 * self.scale

    def item_center(self, i: int) -> QPointF:
        a = item_angle(i, len(self.items))
        r = self.radius * (0.55 + 0.45 * self.open_t.value)
        return QPointF(self.cx + r * math.cos(a), self.cy + r * math.sin(a))

    def contains(self, x: float, y: float) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= self.radius + self.item_radius + 34 * self.scale

    # ---------------------------------------------------------------- state
    def open(self, x: float, y: float) -> None:
        margin = self.radius + self.item_radius + 30
        b = self.bounds
        self.cx = min(max(x, b.left() + margin), b.right() - margin)
        self.cy = min(max(y, b.top() + margin + 40), b.bottom() - margin)
        self.is_open = True
        self.open_t.target = 1.0
        self.hover_index = None

    def close(self) -> None:
        self.is_open = False
        self.open_t.target = 0.0
        self.hover_index = None
        self.pressed_index = None
        self.hub_hover = False

    def toggle(self, x: float, y: float) -> None:
        if self.is_open:
            self.close()
        else:
            self.open(x, y)

    @property
    def visible(self) -> bool:
        return self.is_open or self.open_t.value > 0.01

    def tick(self, dt: float) -> None:
        self.open_t.step(dt)
        for i, s in enumerate(self._item_hover):
            s.target = 1.0 if i == self.hover_index else 0.0
            s.step(dt)

    # ---------------------------------------------------------------- input
    def index_at(self, x: float, y: float) -> int | None:
        return radial_index(x, y, self.cx, self.cy, len(self.items), self.hub_radius + 6,
                            self.radius + self.item_radius + 34 * self.scale)

    def hover(self, x: float, y: float) -> object | None:
        self.hover_index = self.index_at(x, y)
        self.hub_hover = math.hypot(x - self.cx, y - self.cy) <= self.hub_radius
        return self if (self.hover_index is not None or self.hub_hover) else None

    @property
    def dwell_clickable(self) -> bool:
        return True

    def press(self, ev: PointerEvent) -> Grab:
        idx = self.index_at(ev.x, ev.y)
        self.pressed_index = idx

        def move(e: PointerEvent) -> None:
            self.hover(e.x, e.y)

        def release(e: PointerEvent) -> None:
            self.pressed_index = None
            if e.cancelled:
                return
            target = self.index_at(e.x, e.y)
            if target is not None and target == idx:
                self.close()
                self.on_launch(self.items[target].key)
            elif idx is None:
                self.close()  # hub or outside the ring closes the launcher

        return CallbackGrab(InteractionState.CLICK_PENDING, move, release, lambda: setattr(self, "pressed_index", None))

    # ---------------------------------------------------------------- paint
    def paint(self, p: QPainter, theme: Theme) -> None:
        t = self.open_t.value
        if t <= 0.01:
            return
        self.scale = theme.scale
        c = QPointF(self.cx, self.cy)
        # Soft backdrop disc that separates the menu from busy video.
        outer = self.radius + self.item_radius + 46 * self.scale
        grad = QRadialGradient(c, outer)
        grad.setColorAt(0.0, with_alpha(QColor(10, 11, 14), 0.62 * t))
        grad.setColorAt(0.72, with_alpha(QColor(10, 11, 14), 0.48 * t))
        grad.setColorAt(1.0, with_alpha(QColor(10, 11, 14), 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(c, outer, outer)
        # Orbit ring.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(theme.text, 0.08 * t), 1.0))
        p.drawEllipse(c, self.radius * (0.55 + 0.45 * t), self.radius * (0.55 + 0.45 * t))
        # Hub.
        hub_r = self.hub_radius * (0.7 + 0.3 * t)
        p.setPen(QPen(with_alpha(theme.border_strong, t), 1))
        p.setBrush(with_alpha(mix(theme.glass_strong, theme.surface_hover, 0.6 if self.hub_hover else 0.0), 0.9 * t))
        p.drawEllipse(c, hub_r, hub_r)
        isz = hub_r * 0.7
        draw_icon(p, "close", QRectF(c.x() - isz / 2, c.y() - isz / 2, isz, isz),
                  with_alpha(theme.text_secondary if not self.hub_hover else theme.text, t), 1.6)
        n = len(self.items)
        for i, item in enumerate(self.items):
            # Staggered entrance.
            local = max(0.0, min(1.0, (t - i * 0.035) / max(1e-3, 1 - n * 0.035)))
            if local <= 0.01:
                continue
            h = self._item_hover[i].value
            pressed = self.pressed_index == i
            pos = self.item_center(i)
            r = self.item_radius * (0.6 + 0.4 * local) * (1 + 0.14 * h - (0.06 if pressed else 0.0))
            if h > 0.01:
                glow = QRadialGradient(pos, r * 1.8)
                glow.setColorAt(0.0, with_alpha(theme.accent, 0.28 * h * local))
                glow.setColorAt(1.0, with_alpha(theme.accent, 0.0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(glow)
                p.drawEllipse(pos, r * 1.8, r * 1.8)
            p.setPen(QPen(with_alpha(mix(theme.border, theme.accent, h), local * (0.8 + 0.2 * h)), 1.0 + 0.6 * h))
            p.setBrush(with_alpha(mix(theme.glass_strong, theme.surface_active, 0.5 * h), 0.92 * local))
            p.drawEllipse(pos, r, r)
            isz = r * 0.82
            draw_icon(p, item.icon, QRectF(pos.x() - isz / 2, pos.y() - isz / 2, isz, isz),
                      with_alpha(mix(theme.text_secondary, theme.text, h), local), 1.7)
            # Label under each item; the hovered label is brighter.
            p.setFont(theme.font(10 + 0.6 * h, QFont.Weight.Medium if h > 0.5 else QFont.Weight.Normal))
            p.setPen(with_alpha(mix(theme.text_secondary, theme.text, h), local * (0.75 + 0.25 * h)))
            label_rect = QRectF(pos.x() - 60, pos.y() + r + 4, 120, 18 * theme.scale)
            p.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, item.label)
