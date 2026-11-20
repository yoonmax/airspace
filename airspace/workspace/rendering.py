"""The compositor: paints every layer of the workspace with QPainter."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
    QTransform,
)

from airspace.interaction.events import InteractionState
from airspace.interaction.interaction_manager import InteractionManager
from airspace.persistence.settings import Settings
from airspace.ui.animations import Spring
from airspace.ui.icons import draw_icon
from airspace.ui.styles import Theme, mix, with_alpha
from airspace.vision.coordinate_mapper import CoordinateMapper
from airspace.vision.hand_tracker import HAND_CONNECTIONS
from airspace.vision.tracking import TrackingFrame
from airspace.workspace.compositing import FrameLayers
from airspace.workspace.spatial_window import SpatialWindow
from airspace.workspace.workspace import Workspace


@dataclass
class Ripple:
    x: float
    y: float
    born: float
    color: QColor


@dataclass
class CursorVisual:
    ring: Spring = field(default_factory=lambda: Spring(1.0, 520))
    alpha: Spring = field(default_factory=lambda: Spring(0.0, 160))
    hover: Spring = field(default_factory=lambda: Spring(0.0, 420))
    ripples: list[Ripple] = field(default_factory=list)
    trail: deque = field(default_factory=lambda: deque(maxlen=40))
    last_x: float = 0.0
    last_y: float = 0.0
    was_pressed: bool = False


@dataclass
class RenderInputs:
    theme: Theme
    settings: Settings
    workspace: Workspace
    manager: InteractionManager
    tracking: TrackingFrame | None
    mapper: CoordinateMapper
    layers: FrameLayers
    camera_live: bool
    width: float
    height: float
    calibration_active: bool = False


class Renderer:
    def __init__(self) -> None:
        self.cursor = CursorVisual()
        self._t = time.monotonic()

    # ================================================================ main
    def render(self, p: QPainter, ri: RenderInputs, dt: float) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, ri.settings.performance.render_quality == "high")
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self._background(p, ri)
        tr = ri.settings.tracking
        if tr.show_mask:
            self._mask_debug(p, ri)
        ws = ri.workspace
        below: list[QPainterPath] = []  # outlines of windows already painted (lower in z-order)
        for win in ws.wm.windows:
            if win.visible:
                self._window(p, ri, win, below)
        if ri.settings.interface.occlusion != "clear":
            self._foreground(p, ri)
        ws.launcher.paint(p, ri.theme)
        ws.hud.paint(p, ri.theme)
        for ov in ws.overlays:
            ov.paint(p, ri.theme)
        if tr.show_region:
            self._interaction_region(p, ri)
        if tr.show_skeleton or tr.show_labels or tr.show_zones:
            self._skeletons(p, ri)
        self._cursor(p, ri, dt)

    # ========================================================== background
    def _background(self, p: QPainter, ri: RenderInputs) -> None:
        full = QRectF(0, 0, ri.width, ri.height)
        mode = ri.settings.interface.background
        layers = ri.layers
        if not ri.camera_live or layers.frame is None or mode == "solid":
            grad = QLinearGradient(full.topLeft(), full.bottomRight())
            grad.setColorAt(0.0, QColor(24, 26, 31))
            grad.setColorAt(0.55, QColor(15, 16, 19))
            grad.setColorAt(1.0, QColor(11, 12, 14))
            p.fillRect(full, grad)
            glow = QRadialGradient(QPointF(ri.width * 0.72, ri.height * 0.18), ri.width * 0.6)
            glow.setColorAt(0.0, with_alpha(ri.theme.accent, 0.10))
            glow.setColorAt(1.0, with_alpha(ri.theme.accent, 0.0))
            p.fillRect(full, glow)
        else:
            img_rect = layers.image_rect(ri.mapper)
            if mode == "blurred" and layers.blur is not None:
                p.drawImage(img_rect, layers.blur)
                p.fillRect(full, QColor(0, 0, 0, 70))
            else:
                p.drawImage(img_rect, layers.frame)
                if mode == "dimmed":
                    p.fillRect(full, QColor(8, 9, 11, 150))
        # Top and bottom scrims keep HUD text readable over bright video.
        top = QLinearGradient(0, 0, 0, 120)
        top.setColorAt(0.0, QColor(0, 0, 0, 110))
        top.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(0, 0, ri.width, 120), top)
        bottom = QLinearGradient(0, ri.height - 110, 0, ri.height)
        bottom.setColorAt(0.0, QColor(0, 0, 0, 0))
        bottom.setColorAt(1.0, QColor(0, 0, 0, 90))
        p.fillRect(QRectF(0, ri.height - 110, ri.width, 110), bottom)

    def _mask_debug(self, p: QPainter, ri: RenderInputs) -> None:
        img = ri.layers.mask_image()
        if img is None:
            return
        p.save()
        p.setOpacity(0.35)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)
        p.drawImage(ri.layers.image_rect(ri.mapper), img)
        p.restore()

    # ============================================================= windows
    def _window(self, p: QPainter, ri: RenderInputs, win: SpatialWindow,
                below: list[QPainterPath] | None = None) -> None:
        th = ri.theme
        r = win.display_rect()
        rect = QRectF(r.x, r.y, r.w, r.h)
        a = max(0.0, min(1.0, win.appear.value))
        lift = win.lift.value
        scale = (0.94 + 0.06 * a) * (1.0 + 0.012 * win.highlight.value)
        radius = th.radius
        p.save()
        c = rect.center()
        p.translate(c)
        p.scale(scale, scale)
        p.translate(-c)
        p.setOpacity(a)

        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        # Soft shadow; deeper when the window is lifted (focused / dragged).
        p.setPen(Qt.PenStyle.NoPen)
        layers = 5
        for i in range(layers):
            grow = (4 + i * (5 + 4 * lift))
            dy = 4 + 10 * lift
            sp = QPainterPath()
            sp.addRoundedRect(rect.adjusted(-grow / 2, -grow / 2 + dy, grow / 2, grow / 2 + dy), radius + grow / 2,
                              radius + grow / 2)
            p.setBrush(QColor(0, 0, 0, int(14 + 8 * lift)))
            p.drawPath(sp)

        # Glass: the live room (and your hands) stay visible through the window
        # background; a light frost and tint keep the content readable on top.
        p.save()
        p.setClipPath(path)
        if ri.camera_live and ri.layers.blur is not None and ri.settings.interface.background != "solid":
            img_rect = ri.layers.image_rect(ri.mapper)
            p.setOpacity(a * 0.35)
            p.drawImage(img_rect, ri.layers.blur)
            p.setOpacity(a)
            # Where this window covers other windows, show the frosted room instead,
            # so their text never reads through this one.
            if below:
                # Outlines are kept in device pixels; set the clip under an identity
                # transform, then draw with this window's transform again.
                covered = QPainterPath()
                for other in below:
                    covered = covered.united(other)
                covered = covered.intersected(p.transform().map(path))
                if not covered.isEmpty():
                    p.save()
                    xf = p.worldTransform()
                    p.setWorldTransform(QTransform())
                    p.setClipPath(covered, Qt.ClipOperation.IntersectClip)
                    p.setWorldTransform(xf)
                    p.drawImage(img_rect, ri.layers.blur)
                    p.restore()
        else:
            # Nothing behind to see: an opaque base keeps stacked windows from bleeding through.
            p.fillRect(rect, QColor(14, 18, 24))
        p.fillRect(rect, th.glass)
        self._hologram_fill(p, th, win, rect, path)
        # Title bar
        tb = QRectF(rect.left(), rect.top(), rect.width(), win.titlebar_h)
        title_hover = win.hover_part == "titlebar"
        p.fillRect(tb, with_alpha(th.holo, 0.05 + (0.04 if title_hover else 0) + 0.05 * win.highlight.value))
        p.setPen(QPen(with_alpha(th.holo, 0.18), 1))
        p.drawLine(QPointF(tb.left(), tb.bottom()), QPointF(tb.right(), tb.bottom()))
        p.restore()

        self._hologram_edge(p, th, win, rect, path)

        self._titlebar(p, th, win, rect)

        # Content
        cr = QRectF(rect.left(), rect.top() + win.titlebar_h, rect.width(), rect.height() - win.titlebar_h)
        content_path = QPainterPath()
        content_path.addRoundedRect(cr, radius, radius)
        content_path = content_path.united(QPainterPath())  # normalise
        clip = QPainterPath()
        clip.addRect(QRectF(cr.left(), cr.top(), cr.width(), cr.height() - radius))
        clip = clip.united(content_path)
        p.save()
        p.setClipPath(clip)
        p.translate(cr.topLeft())
        try:
            win.app.paint(p, th)
        except Exception:  # noqa: BLE001 - a broken app must never break the compositor
            import logging

            logging.getLogger(__name__).exception("%s.paint failed", win.app_type)
        p.restore()

        self._resize_grip(p, th, win, rect)
        if win.resizing:
            self._size_label(p, th, rect)
        if below is not None and a > 0.5:
            below.append(p.transform().map(path))
        p.restore()

    # ------------------------------------------------------------ hologram
    # The static parts of the effect (sheen, scanlines, rim glow, colour fringe,
    # corner brackets) depend only on window size, focus and theme, so they are
    # rendered once into cached images; only the light sweep is drawn per frame.
    EDGE_MARGIN = 14.0

    def _holo_image(self, key: tuple, w: float, h: float, dpr: float, paint) -> QImage:
        cache: dict = self.__dict__.setdefault("_holo_cache", {})
        img = cache.get(key)
        if img is None:
            if len(cache) > 48:
                cache.clear()
            img = QImage(max(1, int(round(w * dpr))), max(1, int(round(h * dpr))),
                         QImage.Format.Format_ARGB32_Premultiplied)
            img.setDevicePixelRatio(dpr)
            img.fill(QColor(0, 0, 0, 0))
            q = QPainter(img)
            q.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            paint(q)
            q.end()
            cache[key] = img
        return img

    @staticmethod
    def _dpr(p: QPainter) -> float:
        dev = p.device()
        base = dev.devicePixelRatioF() if dev is not None else 1.0
        return max(1.0, round(base * abs(p.worldTransform().m11()) * 2) / 2)

    def _hologram_fill(self, p: QPainter, th: Theme, win: SpatialWindow, rect: QRectF, path: QPainterPath) -> None:
        """Inside the glass: top sheen, faint scanlines, inner rim glow (cached) and a slow light sweep."""
        w, h, r = rect.width(), rect.height(), th.radius
        dpr = self._dpr(p)
        focused = bool(win.focused)

        def paint(q: QPainter) -> None:
            local = QRectF(0, 0, w, h)
            lp = QPainterPath()
            lp.addRoundedRect(local, r, r)
            q.setClipPath(lp)
            sheen = QLinearGradient(local.topLeft(), local.bottomLeft())
            sheen.setColorAt(0.0, with_alpha(th.holo, 0.10))
            sheen.setColorAt(0.22, with_alpha(th.holo, 0.02))
            sheen.setColorAt(1.0, with_alpha(th.holo, 0.04))
            q.fillRect(local, sheen)
            q.setPen(QPen(QColor(255, 255, 255, 12), 1.0 / dpr))
            y = 0.5
            while y < h:
                q.drawLine(QPointF(0, y), QPointF(w, y))
                y += 2.0
            q.setBrush(Qt.BrushStyle.NoBrush)
            q.setPen(QPen(with_alpha(th.holo, 0.10 if focused else 0.06), 10))
            q.drawPath(lp)

        key = ("fill", round(w), round(h), focused, dpr, r, th.holo.rgba())
        p.drawImage(rect.topLeft(), self._holo_image(key, w, h, dpr, paint))
        # Light sweep: a soft diagonal band crosses each window every few seconds.
        if not th.reduced_motion:
            period, travel = 7.0, 1.6
            phase = ((time.monotonic() + win.id * 2.3) % period) / travel
            if phase < 1.0:
                x = rect.left() - w * 0.4 + phase * w * 1.8
                band = QLinearGradient(QPointF(x, rect.top()), QPointF(x + w * 0.35, rect.bottom()))
                band.setColorAt(0.0, with_alpha(th.holo, 0.0))
                band.setColorAt(0.5, with_alpha(th.holo, 0.07))
                band.setColorAt(1.0, with_alpha(th.holo, 0.0))
                p.fillRect(rect, band)

    def _hologram_edge(self, p: QPainter, th: Theme, win: SpatialWindow, rect: QRectF, path: QPainterPath) -> None:
        """Projected-light rim: gradient edge, colour fringe, outer glow and corner brackets (cached)."""
        hl = round(win.highlight.value * 6) / 6  # a few steps keep the cache small while dragging
        focused = bool(win.focused)
        w, h, r, m = rect.width(), rect.height(), th.radius, self.EDGE_MARGIN
        dpr = self._dpr(p)

        def paint(q: QPainter) -> None:
            local = QRectF(m, m, w, h)
            lp = QPainterPath()
            lp.addRoundedRect(local, r, r)
            strength = (1.0 if focused else 0.6) + 0.4 * hl
            q.setBrush(Qt.BrushStyle.NoBrush)
            q.setPen(QPen(with_alpha(th.holo, (0.07 + 0.12 * hl) * strength), 6 + 3 * hl))
            q.drawPath(lp)
            for dx, col in ((-0.9, QColor(90, 210, 255)), (0.9, QColor(255, 110, 210))):
                q.setPen(QPen(with_alpha(col, 0.10 * strength), 1.0))
                q.drawPath(lp.translated(dx, 0))
            g = QLinearGradient(local.topLeft(), local.bottomRight())
            rim = mix(th.holo, th.accent, hl)
            g.setColorAt(0.0, with_alpha(rim, 0.70 * strength))
            g.setColorAt(0.5, with_alpha(rim, 0.22 * strength))
            g.setColorAt(1.0, with_alpha(rim, 0.45 * strength))
            q.setPen(QPen(QBrush(g), 1.2 + 0.8 * hl))
            q.drawPath(lp)
            k = min(22.0 * th.scale, w * 0.08, h * 0.08)
            o = 5.0
            q.setPen(QPen(with_alpha(th.holo, 0.55 * strength), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            L, T, R, B = local.left() - o, local.top() - o, local.right() + o, local.bottom() + o
            for (cx, cy, sx, sy) in ((L, T, 1, 1), (R, T, -1, 1), (L, B, 1, -1), (R, B, -1, -1)):
                q.drawLine(QPointF(cx, cy), QPointF(cx + sx * k, cy))
                q.drawLine(QPointF(cx, cy), QPointF(cx, cy + sy * k))

        key = ("edge", round(w), round(h), focused, hl, dpr, r, th.scale, th.holo.rgba(), th.accent.rgba())
        p.drawImage(QPointF(rect.left() - m, rect.top() - m), self._holo_image(key, w + 2 * m, h + 2 * m, dpr, paint))

    def _titlebar(self, p: QPainter, th: Theme, win: SpatialWindow, rect: QRectF) -> None:
        h = win.titlebar_h
        isz = h * 0.40
        draw_icon(p, win.app.icon, QRectF(rect.left() + h * 0.38, rect.top() + (h - isz) / 2, isz, isz),
                  th.text_secondary if not win.focused else th.text, 1.6)
        p.setPen(th.text if win.focused else th.text_secondary)
        p.setFont(th.font(11.5, QFont.Weight.Medium))
        title_r = QRectF(rect.left() + h * 0.38 + isz + 10, rect.top(), rect.width() * 0.5, h)
        p.drawText(title_r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, win.title)
        status = ""
        try:
            status = win.app.status_text()
        except Exception:  # noqa: BLE001
            pass
        buttons = win.button_rects()
        if status:
            p.setPen(th.text_tertiary)
            p.setFont(th.font(10))
            sr = QRectF(rect.left() + rect.width() * 0.35, rect.top(), buttons["minimize"].x - rect.left() - rect.width() * 0.35 - 12, h)
            p.drawText(sr, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, status)
        dx, dy = rect.left() - win.rect.x, rect.top() - win.rect.y
        for name, br in buttons.items():
            b = QRectF(br.x + dx, br.y + dy, br.w, br.h)
            hovered = win.hover_part == name
            if hovered:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(with_alpha(th.danger, 0.85) if name == "close" else th.surface_hover)
                p.drawEllipse(b)
            icon_sz = b.width() * 0.5
            color = th.text if hovered else th.text_secondary
            draw_icon(p, name, QRectF(b.center().x() - icon_sz / 2, b.center().y() - icon_sz / 2, icon_sz, icon_sz),
                      color, 1.6)

    def _resize_grip(self, p: QPainter, th: Theme, win: SpatialWindow, rect: QRectF) -> None:
        hovered = win.hover_part == "resize:se" or win.resizing
        color = th.accent if hovered else with_alpha(th.text, 0.28)
        p.setPen(QPen(color, 1.6 if hovered else 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        x, y = rect.right() - 7, rect.bottom() - 7
        for k in (6, 11, 16):
            p.drawLine(QPointF(x - k, y), QPointF(x, y - k))

    def _size_label(self, p: QPainter, th: Theme, rect: QRectF) -> None:
        text = f"{rect.width():.0f} × {rect.height():.0f}"
        p.setFont(th.font(9.5, QFont.Weight.Medium))
        w = p.fontMetrics().horizontalAdvance(text) + 16
        lr = QRectF(rect.right() - w - 10, rect.bottom() + 10, w, 22)
        p.setPen(QPen(th.border, 1))
        p.setBrush(th.glass_strong)
        p.drawRoundedRect(lr, 7, 7)
        p.setPen(th.text)
        p.drawText(lr, Qt.AlignmentFlag.AlignCenter, text)
        p.setPen(QPen(with_alpha(th.accent, 0.5), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(-6, -6, 6, 6), th.radius + 6, th.radius + 6)

    # ========================================================== foreground
    def _foreground(self, p: QPainter, ri: RenderInputs) -> None:
        layers = ri.layers
        if not ri.camera_live or layers.foreground is None or ri.settings.interface.background in ("solid", "blurred"):
            return
        rect = layers.foreground_rect(ri.mapper)
        if rect is None:
            return
        p.drawImage(rect, layers.foreground)
        if ri.settings.interface.background == "dimmed":
            p.save()
            p.setClipRect(rect)
            p.restore()

    # ============================================================== debug
    def _interaction_region(self, p: QPainter, ri: RenderInputs) -> None:
        reg = ri.mapper.interaction_region()
        r = QRectF(reg.x, reg.y, reg.w, reg.h)
        p.setPen(QPen(with_alpha(ri.theme.accent, 0.55), 1.2, Qt.PenStyle.DashLine))
        p.setBrush(with_alpha(ri.theme.accent, 0.04))
        p.drawRoundedRect(r, 14, 14)
        p.setPen(ri.theme.text_secondary)
        p.setFont(ri.theme.font(9))
        p.drawText(QRectF(r.left() + 10, r.top() + 6, 260, 18), Qt.AlignmentFlag.AlignLeft,
                   f"pointer reach ×{ri.mapper.reach:.2f} — hand region mapped to the full workspace")

    def _skeletons(self, p: QPainter, ri: RenderInputs) -> None:
        if ri.tracking is None:
            return
        th = ri.theme
        tr = ri.settings.tracking
        for hand in ri.tracking.hands:
            if not hand.present or hand.observation is None:
                continue
            lm = hand.observation.landmarks
            pts = [QPointF(*ri.mapper.norm_to_view(float(x), float(y))) for x, y, _ in lm]
            col = th.accent if hand.is_primary else th.warm
            if tr.show_skeleton:
                p.setPen(QPen(with_alpha(col, 0.85), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                for a, b in HAND_CONNECTIONS:
                    p.drawLine(pts[a], pts[b])
                p.setPen(Qt.PenStyle.NoPen)
                for i, pt in enumerate(pts):
                    z = float(lm[i, 2])
                    rr = max(2.0, 3.6 - z * 18)  # relative depth cue (heuristic)
                    p.setBrush(QColor(250, 249, 246) if i in (4, 8) else with_alpha(col, 0.95))
                    p.drawEllipse(pt, rr, rr)
                if hand.features is not None:
                    pc = QPointF(*ri.mapper.norm_to_view(*map(float, hand.features.palm_center)))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(with_alpha(col, 0.6), 1.2))
                    p.drawEllipse(pc, 6, 6)
                    xs = [pt.x() for pt in pts]
                    ys = [pt.y() for pt in pts]
                    p.setPen(QPen(with_alpha(col, 0.35), 1, Qt.PenStyle.DashLine))
                    p.drawRoundedRect(QRectF(min(xs) - 12, min(ys) - 12, max(xs) - min(xs) + 24,
                                             max(ys) - min(ys) + 24), 10, 10)
                    p.setFont(th.font(9, QFont.Weight.Medium))
                    p.setPen(col)
                    p.drawText(QPointF(min(xs) - 12, min(ys) - 18),
                               f"#{hand.id} {hand.handedness.lower()} · {hand.pose.value}")
            if tr.show_labels:
                p.setFont(th.font(7.5))
                p.setPen(th.text)
                for i, pt in enumerate(pts):
                    p.drawText(pt + QPointF(5, -4), str(i))
            if tr.show_zones and hand.features is not None:
                # Pinch thumb↔index distance with the enter/exit thresholds drawn as rings.
                f = hand.features
                scale_px = ri.mapper.norm_length_to_view(f.palm_scale) / (ri.mapper.frame_w / ri.mapper.frame_h)
                mid = (pts[4] + pts[8]) / 2
                cfg = ri.settings.tracking
                for thr, color in ((cfg.pinch_enter, th.success), (cfg.pinch_exit, th.warning)):
                    rad = thr * scale_px / 2
                    p.setPen(QPen(with_alpha(color, 0.6), 1, Qt.PenStyle.DashLine))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(mid, rad, rad)
                p.setPen(QPen(th.success if hand.pinched else th.text, 1.6))
                p.drawLine(pts[4], pts[8])
                p.setFont(th.font(8.5, QFont.Weight.Medium))
                p.drawText(mid + QPointF(10, 14), f"pinch {f.pinch_ratio:.2f}")

    # ============================================================= cursor
    def _cursor(self, p: QPainter, ri: RenderInputs, dt: float) -> None:
        cv = self.cursor
        m = ri.manager
        hp = m.pointers["hand"]
        th = ri.theme
        primary = ri.tracking.primary if ri.tracking else None
        visible = hp.visible and primary is not None and primary.present
        cv.alpha.target = 1.0 if visible else 0.0
        cv.alpha.step(dt)
        now = time.monotonic()
        # Ripple on press.
        if hp.pressed and not cv.was_pressed:
            cv.ripples.append(Ripple(hp.x, hp.y, now, th.accent))
        cv.was_pressed = hp.pressed
        cv.ripples = [r for r in cv.ripples if now - r.born < 0.45]
        for r in cv.ripples:
            age = (now - r.born) / 0.45
            p.setPen(QPen(with_alpha(r.color, 0.55 * (1 - age)), 2.0 * (1 - age) + 0.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            rad = 12 + 26 * age
            p.drawEllipse(QPointF(r.x, r.y), rad, rad)

        # Secondary hand indicator (used for two-hand gestures).
        if ri.tracking:
            sec = ri.tracking.secondary
            if sec is not None and sec.pointer is not None:
                sx, sy = ri.mapper.pointer_to_view(float(sec.pointer[0]), float(sec.pointer[1]))
                color = th.warm
                p.setPen(QPen(with_alpha(color, 0.8), 1.6))
                p.setBrush(with_alpha(color, 0.35 if sec.pinched else 0.0))
                rr = 9 if sec.pinched else 11
                p.drawEllipse(QPointF(sx, sy), rr, rr)
                if m.two_hand_active:
                    p.setPen(QPen(with_alpha(th.accent, 0.5), 1.2, Qt.PenStyle.DashLine))
                    p.drawLine(QPointF(hp.x, hp.y), QPointF(sx, sy))

        a = cv.alpha.value
        if a <= 0.01:
            cv.trail.clear()
            return
        x, y = hp.x, hp.y
        if visible:
            cv.last_x, cv.last_y = x, y
            cv.trail.append((x, y))
        else:
            x, y = cv.last_x, cv.last_y
        if ri.settings.tracking.show_trail and len(cv.trail) > 2:
            pts = list(cv.trail)
            for i in range(1, len(pts)):
                f = i / len(pts)
                p.setPen(QPen(with_alpha(th.accent, 0.5 * f * a), 1 + 2.5 * f, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(*pts[i - 1]), QPointF(*pts[i]))

        size = ri.settings.interface.cursor_size * th.scale
        pinch_p = primary.pinch_progress if primary else 0.0
        palm_p = primary.palm_progress if primary else 0.0
        pressed = hp.pressed
        state = m.state
        hovering = m.hand_hover_target is not None
        cv.hover.target = 1.0 if hovering else 0.0
        cv.hover.step(dt)
        ring_target = 0.55 if pressed else (1.0 - 0.38 * pinch_p)
        cv.ring.target = ring_target
        cv.ring.step(dt)
        base_r = 15 * size * (1 + 0.15 * cv.hover.value)
        ring_r = base_r * cv.ring.value
        c = QPointF(x, y)
        # Soft glow
        glow = QRadialGradient(c, base_r * 2.2)
        gcol = th.accent if (pressed or hovering) else QColor(255, 255, 255)
        glow.setColorAt(0.0, with_alpha(gcol, 0.22 * a))
        glow.setColorAt(1.0, with_alpha(gcol, 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(c, base_r * 2.2, base_r * 2.2)
        # Dark contrast ring then the bright ring, readable on any background.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(QColor(0, 0, 0), 0.35 * a), 3.4))
        p.drawEllipse(c, ring_r, ring_r)
        ring_col = th.accent if pressed else mix(QColor(250, 249, 246), th.accent, cv.hover.value * 0.6)
        p.setPen(QPen(with_alpha(ring_col, 0.95 * a), 1.8))
        if pressed:
            p.setBrush(with_alpha(th.accent, 0.35 * a))
        p.drawEllipse(c, ring_r, ring_r)
        # Centre dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(QColor(250, 249, 246), a))
        p.drawEllipse(c, 2.6 * size, 2.6 * size)
        # Activation arcs: pinch approach (inner), palm hold (outer), dwell (outer).
        if 0.05 < pinch_p < 1.0 and not pressed:
            p.setPen(QPen(with_alpha(th.accent, 0.8 * a), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            rr = base_r + 5
            p.drawArc(QRectF(x - rr, y - rr, rr * 2, rr * 2), 90 * 16, int(-360 * 16 * pinch_p))
        progress = max(palm_p, m.dwell_progress)
        if progress > 0.02 and not pressed:
            col = th.warm if palm_p >= m.dwell_progress else th.success
            rr = base_r + 10
            p.setPen(QPen(with_alpha(col, 0.9 * a), 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(QRectF(x - rr, y - rr, rr * 2, rr * 2), 90 * 16, int(-360 * 16 * progress))
        # State glyph for drags.
        if state in (InteractionState.DRAGGING_WINDOW, InteractionState.RESIZING_WINDOW):
            gsz = 14 * size
            draw_icon(p, "pan" if state == InteractionState.DRAGGING_WINDOW else "fit",
                      QRectF(x + base_r + 4, y + base_r - 2, gsz, gsz), with_alpha(th.text, a), 1.4)
