"""Heads-up display: wordmark, status pills, global actions, dock and toasts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QPainter, QPainterPath, QPen

from airspace.interaction.events import Grab, PointerEvent
from airspace.ui.animations import Spring
from airspace.ui.controls import Button, ControlSet
from airspace.ui.icons import draw_icon
from airspace.ui.styles import Theme, with_alpha


@dataclass
class StatusInfo:
    camera_state: str = "off"  # ok | warn | error | off
    camera_label: str = "camera off"
    hands: int = 0
    tracking_label: str = "no hands"
    gesture: str = "—"
    fps: float = 0.0
    state: str = "idle"
    tracking_lost: bool = False


@dataclass
class Toast:
    text: str
    born: float
    duration: float = 2.2
    icon: str | None = None

    def alpha(self, now: float) -> float:
        age = now - self.born
        if age < 0.18:
            return age / 0.18
        if age > self.duration - 0.35:
            return max(0.0, (self.duration - age) / 0.35)
        return 1.0


@dataclass
class HudActions:
    open_launcher: Callable[[], None]
    arrange: Callable[[], None]
    open_settings: Callable[[], None]
    toggle_fullscreen: Callable[[], None]
    is_fullscreen: Callable[[], bool]
    restore_window: Callable[[int], None]
    toggle_camera: Callable[[], None]
    camera_enabled: Callable[[], bool]


class Hud:
    TOP = 14.0

    def __init__(self, actions: HudActions, status: Callable[[], StatusInfo],
                 minimized: Callable[[], list[tuple[int, str, str]]]) -> None:
        self.actions = actions
        self.status = status
        self.minimized = minimized
        self.controls = ControlSet()
        self.bounds = QRectF(0, 0, 1280, 720)
        self.toasts: list[Toast] = []
        self.visible = True
        self._dock_buttons: dict[int, Button] = {}
        self._lost = Spring(0.0, 200)
        s = self.controls
        self.btn_apps = s.add(Button(icon="apps", variant="icon", on_click=actions.open_launcher, tooltip="Apps  ⌘O"))
        self.btn_arrange = s.add(Button(icon="arrange", variant="icon", on_click=actions.arrange, tooltip="Arrange windows"))
        self.btn_camera = s.add(Button(icon="camera", variant="icon", on_click=actions.toggle_camera,
                                       checked=lambda: not actions.camera_enabled(), tooltip="Camera on/off"))
        self.btn_settings = s.add(Button(icon="settings", variant="icon", on_click=actions.open_settings, tooltip="Settings  ⌘,"))
        self.btn_full = s.add(Button(icon="fullscreen", variant="icon", on_click=actions.toggle_fullscreen, tooltip="Fullscreen  F11"))

    @property
    def top_bar_height(self) -> float:
        return 58.0

    def layout(self, bounds: QRectF, theme: Theme) -> None:
        self.bounds = QRectF(bounds)
        s = theme.scale
        size = 34 * s
        gap = 6 * s
        x = bounds.right() - 18 * s - size
        for b in (self.btn_full, self.btn_settings, self.btn_camera, self.btn_arrange, self.btn_apps):
            b.rect = QRectF(x, self.TOP + 4 * s, size, size)
            x -= size + gap
        self.btn_full.icon = "exit-fullscreen" if self.actions.is_fullscreen() else "fullscreen"
        self._layout_dock(theme)

    def _layout_dock(self, theme: Theme) -> None:
        items = self.minimized()
        ids = {i for i, _, _ in items}
        for wid in list(self._dock_buttons):
            if wid not in ids:
                self.controls.controls.remove(self._dock_buttons.pop(wid))
        s = theme.scale
        h = 36 * s
        widths = [max(110 * s, (46 + 7.4 * len(title)) * s) for _, title, _ in items]
        total = sum(widths) + 8 * s * max(0, len(widths) - 1)
        x = self.bounds.center().x() - total / 2
        y = self.bounds.bottom() - 20 * s - h
        for (wid, title, icon), w in zip(items, widths):
            btn = self._dock_buttons.get(wid)
            if btn is None:
                btn = Button(title, icon=icon, on_click=lambda w=wid: self.actions.restore_window(w), tooltip="Restore")
                self._dock_buttons[wid] = self.controls.add(btn)
            btn.rect = QRectF(x, y, w, h)
            x += w + 8 * s

    def dock_rect(self) -> QRectF | None:
        if not self._dock_buttons:
            return None
        rects = [b.rect for b in self._dock_buttons.values()]
        r = rects[0]
        for o in rects[1:]:
            r = r.united(o)
        return r.adjusted(-8, -8, 8, 8)

    def toast(self, text: str, icon: str | None = None, duration: float = 2.2) -> None:
        now = time.monotonic()
        self.toasts = [t for t in self.toasts if now - t.born < t.duration and t.text != text][-2:]
        self.toasts.append(Toast(text, now, duration, icon))

    # --------------------------------------------------------------- input
    def hover(self, ev: PointerEvent):
        return self.controls.hover(ev.x, ev.y, ev.is_hand)

    def leave(self) -> None:
        self.controls.leave()

    def press(self, ev: PointerEvent) -> Grab | None:
        return self.controls.press(ev)

    def hit(self, x: float, y: float, hand: bool = False) -> bool:
        return self.controls.find(x, y, hand) is not None

    def tick(self, dt: float) -> None:
        self.controls.tick(dt)
        self._lost.target = 1.0 if self.status().tracking_lost else 0.0
        self._lost.step(dt)

    # --------------------------------------------------------------- paint
    def paint(self, p: QPainter, theme: Theme) -> None:
        if not self.visible:
            return
        st = self.status()
        s = theme.scale
        self._paint_wordmark(p, theme, s)
        self._paint_pills(p, theme, st, s)
        # Action buttons sit on a single glass capsule.
        btns = [self.btn_apps, self.btn_arrange, self.btn_camera, self.btn_settings, self.btn_full]
        cap = btns[0].rect.united(btns[-1].rect).adjusted(-6 * s, -5 * s, 6 * s, 5 * s)
        self._capsule(p, cap, theme)
        self.controls.paint(p, theme)
        hovered = self.controls.hovered
        if hovered is not None and hovered.tooltip and hovered in btns:
            self._tooltip(p, theme, hovered.rect, hovered.tooltip)
        if self._dock_buttons:
            dr = self.dock_rect()
            if dr is not None:
                p.save()
                self._capsule(p, dr, theme)
                p.restore()
                for b in self._dock_buttons.values():
                    b.paint(p, theme)
        self._paint_toasts(p, theme, s)
        if self._lost.value > 0.01:
            self._paint_tracking_lost(p, theme, s)

    def _capsule(self, p: QPainter, rect: QRectF, theme: Theme) -> None:
        path = QPainterPath()
        path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        p.setPen(QPen(theme.border, 1))
        p.setBrush(theme.glass)
        p.drawPath(path)

    def _paint_wordmark(self, p: QPainter, theme: Theme, s: float) -> None:
        x, y = 22 * s, self.TOP + 4 * s
        h = 34 * s
        # A small orbit glyph: a ring with a satellite dot.
        c = QPointF(x + h / 2, y + h / 2)
        p.setPen(QPen(with_alpha(theme.text, 0.85), 1.6 * s))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, 7.5 * s, 7.5 * s)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.accent)
        p.drawEllipse(QPointF(c.x() + 6.5 * s, c.y() - 5.2 * s), 2.6 * s, 2.6 * s)
        p.setPen(theme.text)
        p.setFont(theme.font(15.5, QFont.Weight.Medium, letter_spacing=0.6))
        p.drawText(QRectF(x + h + 4 * s, y, 140 * s, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "airspace")

    def _paint_pills(self, p: QPainter, theme: Theme, st: StatusInfo, s: float) -> None:
        colors = {"ok": theme.success, "warn": theme.warning, "error": theme.danger, "off": theme.text_tertiary}
        pills = [
            (colors.get(st.camera_state, theme.text_tertiary), st.camera_label),
            (theme.success if st.hands else theme.text_tertiary, st.tracking_label),
            (None, st.gesture),
            (None, f"{st.fps:.0f} fps"),
        ]
        x = 178 * s
        y = self.TOP + 7 * s
        h = 28 * s
        p.setFont(theme.font(10, QFont.Weight.Medium))
        fm = p.fontMetrics()
        for dot, text in pills:
            w = fm.horizontalAdvance(text) + (34 if dot is not None else 22) * s
            rect = QRectF(x, y, w, h)
            self._capsule(p, rect, theme)
            tx = rect.left() + 11 * s
            if dot is not None:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(dot)
                p.drawEllipse(QPointF(tx + 3.5 * s, rect.center().y()), 3.5 * s, 3.5 * s)
                tx += 13 * s
            p.setPen(theme.text_secondary)
            p.drawText(QRectF(tx, rect.top(), w, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            x += w + 8 * s

    def _tooltip(self, p: QPainter, theme: Theme, anchor: QRectF, text: str) -> None:
        p.setFont(theme.font(9.5))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 18
        r = QRectF(anchor.center().x() - w / 2, anchor.bottom() + 10, w, 24)
        if r.right() > self.bounds.right() - 8:
            r.moveRight(self.bounds.right() - 8)
        p.setPen(QPen(theme.border, 1))
        p.setBrush(theme.glass_strong)
        p.drawRoundedRect(r, 7, 7)
        p.setPen(theme.text)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_toasts(self, p: QPainter, theme: Theme, s: float) -> None:
        now = time.monotonic()
        self.toasts = [t for t in self.toasts if now - t.born < t.duration]
        dock = self.dock_rect()
        base_y = (dock.top() if dock else self.bounds.bottom() - 16 * s) - 16 * s
        p.setFont(theme.font(10.5, QFont.Weight.Medium))
        fm = p.fontMetrics()
        for i, toast in enumerate(reversed(self.toasts)):
            a = toast.alpha(now)
            w = fm.horizontalAdvance(toast.text) + 36 * s
            h = 34 * s
            y = base_y - h - i * (h + 8 * s) + (1 - a) * 8
            r = QRectF(self.bounds.center().x() - w / 2, y, w, h)
            p.save()
            p.setOpacity(a)
            self._capsule(p, r, theme)
            p.setPen(theme.text)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, toast.text)
            p.restore()

    def _paint_tracking_lost(self, p: QPainter, theme: Theme, s: float) -> None:
        a = self._lost.value
        text = "hand tracking lost — raise your hand into view"
        p.setFont(theme.font(10, QFont.Weight.Medium))
        w = p.fontMetrics().horizontalAdvance(text) + 46 * s
        r = QRectF(self.bounds.center().x() - w / 2, self.TOP + 58 * s, w, 28 * s)
        p.save()
        p.setOpacity(a)
        self._capsule(p, r, theme)
        draw_icon(p, "hand", QRectF(r.left() + 10 * s, r.center().y() - 8 * s, 16 * s, 16 * s), theme.warning, 1.4)
        p.setPen(theme.text_secondary)
        p.drawText(r.adjusted(32 * s, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        p.restore()
