"""Modal overlays: confirmation dialogs and informational cards.

Overlays capture all pointer input while open, so a stray pinch can never
click something behind a dialog.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QKeyEvent, QLinearGradient, QPainter, QPainterPath, QPen

from airspace.interaction.events import CallbackGrab, Grab, InteractionState, PointerEvent
from airspace.ui.animations import Spring
from airspace.ui.controls import Button, ControlSet
from airspace.ui.icons import draw_icon
from airspace.ui.styles import Theme, with_alpha


def paint_glass_card(p: QPainter, rect: QRectF, theme: Theme, alpha: float = 1.0, radius: float | None = None,
                     strong: bool = True) -> None:
    radius = theme.radius if radius is None else radius
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    # Shadow: a few expanding translucent layers (cheap soft shadow).
    p.setPen(Qt.PenStyle.NoPen)
    for i in range(4):
        grow = 6 + i * 6
        sp = QPainterPath()
        sp.addRoundedRect(rect.adjusted(-grow / 2, -grow / 3, grow / 2, grow), radius + grow / 2, radius + grow / 2)
        p.setBrush(with_alpha(QColor(0, 0, 0), 0.07 * alpha))
        p.drawPath(sp)
    base = theme.glass_strong if strong else theme.glass
    p.setBrush(with_alpha(base, base.alphaF() * alpha))
    p.drawPath(path)
    grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    grad.setColorAt(0.0, with_alpha(QColor(255, 255, 255), 0.07 * alpha))
    grad.setColorAt(0.25, with_alpha(QColor(255, 255, 255), 0.0))
    p.setBrush(grad)
    p.drawPath(path)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(with_alpha(theme.border_strong, theme.border_strong.alphaF() * alpha), 1))
    p.drawPath(path)


class Overlay:
    modal = True
    dims_background = True

    def __init__(self) -> None:
        self.controls = ControlSet()
        self.appear = Spring(0.0, 320, damping=30)
        self.appear.target = 1.0
        self.closing = False
        self.bounds = QRectF(0, 0, 1280, 720)
        self.on_closed: Callable[[], None] | None = None

    @property
    def visible(self) -> bool:
        return not (self.closing and self.appear.value <= 0.01)

    def layout(self, bounds: QRectF) -> None:
        self.bounds = QRectF(bounds)

    def dismiss(self) -> None:
        if not self.closing:
            self.closing = True
            self.appear.target = 0.0
            if self.on_closed:
                self.on_closed()

    def tick(self, dt: float) -> None:
        self.appear.step(dt)
        self.controls.tick(dt)

    def hover(self, ev: PointerEvent):
        return self.controls.hover(ev.x, ev.y, ev.is_hand)

    def leave(self) -> None:
        self.controls.leave()

    def press(self, ev: PointerEvent) -> Grab:
        grab = self.controls.press(ev)
        return grab or CallbackGrab(InteractionState.CLICK_PENDING)  # swallow clicks

    def key(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key.Key_Escape:
            self.dismiss()
            return True
        return False

    def wheel(self, x: float, y: float, delta: float) -> bool:
        return self.controls.wheel(x, y, delta)

    def paint_dim(self, p: QPainter) -> None:
        if self.dims_background:
            p.fillRect(self.bounds, with_alpha(QColor(6, 7, 9), 0.45 * self.appear.value))

    def paint(self, p: QPainter, theme: Theme) -> None:
        self.paint_dim(p)


class MessageCard(Overlay):
    """A centred card with a title, explanatory text, bullet points and actions."""

    def __init__(self, title: str, body: str, bullets: list[str] | None = None,
                 actions: list[tuple[str, Callable[[], None], str]] | None = None, icon: str = "info",
                 dismissable: bool = True, width: float = 520) -> None:
        super().__init__()
        self.title, self.body, self.bullets, self.icon = title, body, bullets or [], icon
        self.dismissable = dismissable
        self.width = width
        self.card = QRectF()
        self.buttons: list[Button] = []
        for label, cb, variant in actions or []:
            b = Button(label, on_click=cb, variant=variant)
            self.buttons.append(self.controls.add(b))

    def key(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key.Key_Escape and self.dismissable:
            self.dismiss()
            return True
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.buttons:
            primary = next((b for b in self.buttons if b.variant == "primary"), self.buttons[0])
            if primary.on_click:
                primary.on_click()
            return True
        return False

    def _height(self, theme: Theme) -> float:
        s = theme.scale
        chars_per_line = max(30, int(self.width / 7.4))
        body_lines = max(1, len(self.body) // chars_per_line + 1 + self.body.count("\n"))
        bullet_lines = sum(max(1, len(b) // (chars_per_line - 4) + 1) for b in self.bullets)
        return (78 + body_lines * 19 + bullet_lines * 20 + len(self.bullets) * 6 + (76 if self.buttons else 24)) * s

    def layout(self, bounds: QRectF, theme: Theme | None = None) -> None:
        super().layout(bounds)
        s = theme.scale if theme else 1.0
        w = min(self.width * s, bounds.width() - 32)
        h = self._height(theme) if theme else 320
        self.card = QRectF(bounds.center().x() - w / 2, bounds.center().y() - h / 2, w, h)
        bh = 40 * s
        gap = 10 * s
        widths = [max(110 * s, 18 * s + 8.2 * s * len(b.label)) for b in self.buttons]
        x = self.card.right() - 24 * s - sum(widths) - gap * max(0, len(widths) - 1)
        for b, bw in zip(self.buttons, widths):
            b.rect = QRectF(x, self.card.bottom() - 24 * s - bh, bw, bh)
            x += bw + gap

    def paint(self, p: QPainter, theme: Theme) -> None:
        self.layout(self.bounds, theme)
        a = self.appear.value
        if a <= 0.01:
            return
        self.paint_dim(p)
        p.save()
        p.setOpacity(a)
        c = self.card
        off = (1 - a) * 14
        p.translate(0, off)
        paint_glass_card(p, c, theme)
        s = theme.scale
        isz = 22 * s
        draw_icon(p, self.icon, QRectF(c.left() + 26 * s, c.top() + 28 * s, isz, isz), theme.accent, 1.7)
        p.setPen(theme.text)
        p.setFont(theme.font(15, QFont.Weight.DemiBold))
        p.drawText(QRectF(c.left() + 60 * s, c.top() + 24 * s, c.width() - 84 * s, 30 * s),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.title)
        p.setFont(theme.font(11))
        p.setPen(theme.text_secondary)
        body_rect = QRectF(c.left() + 26 * s, c.top() + 66 * s, c.width() - 52 * s, c.height())
        br = p.boundingRect(body_rect, Qt.TextFlag.TextWordWrap, self.body)
        p.drawText(body_rect, Qt.TextFlag.TextWordWrap, self.body)
        y = br.bottom() + 12 * s
        for bullet in self.bullets:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.accent)
            p.drawEllipse(QPointF(c.left() + 32 * s, y + 9 * s), 2.2 * s, 2.2 * s)
            p.setPen(theme.text)
            r = QRectF(c.left() + 44 * s, y, c.width() - 70 * s, 40 * s)
            bb = p.boundingRect(r, Qt.TextFlag.TextWordWrap, bullet)
            p.drawText(r, Qt.TextFlag.TextWordWrap, bullet)
            y = bb.bottom() + 6 * s
        self.controls.paint(p, theme)
        p.restore()


class ConfirmDialog(MessageCard):
    def __init__(self, title: str, message: str, confirm_label: str, on_confirm: Callable[[], None],
                 danger: bool = True) -> None:
        def confirm() -> None:
            self.dismiss()
            on_confirm()

        super().__init__(title, message, actions=[
            ("Cancel", self.dismiss, "default"),
            (confirm_label, confirm, "danger" if danger else "primary"),
        ], icon="info", width=440)
