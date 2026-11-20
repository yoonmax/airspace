"""tracking — a live window into airspace's computer vision.

Everything shown is either a direct model output (landmarks, handedness and
its score), a measured timing, or a clearly labelled derived heuristic
(pinch ratio, jitter). Nothing is invented.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from airspace.miniapps.base_app import MiniApp
from airspace.ui.controls import Button
from airspace.ui.styles import Theme, mix, with_alpha
from airspace.vision.filters import SMOOTHING_PRESETS
from airspace.vision.hand_tracker import FINGER_CHAINS, HAND_CONNECTIONS

TOGGLES = [
    ("show_skeleton", "skeleton"),
    ("show_labels", "labels"),
    ("show_trail", "trail"),
    ("show_zones", "pinch zones"),
    ("show_mask", "mask"),
    ("show_region", "region"),
]


class TrackingInspectorApp(MiniApp):
    app_type = "tracking"
    title = "tracking"
    icon = "hand"
    default_size = (600.0, 560.0)
    min_size = (480.0, 460.0)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.toggles: list[Button] = []
        for attr, label in TOGGLES:
            b = Button(label, on_click=lambda a=attr: self._toggle(a), checked=lambda a=attr: getattr(self.ctx.settings.tracking, a),
                       font_size=10)
            self.toggles.append(self.controls.add(b))
        self.diagram = QRectF()
        self.stats = QRectF()

    def _toggle(self, attr: str) -> None:
        t = self.ctx.settings.tracking
        setattr(t, attr, not getattr(t, attr))
        self.ctx.shell.apply_settings()

    def layout(self, w: float, h: float) -> None:
        s = self.theme.scale
        pad = 14 * s
        bh = 28 * s
        cols = 3
        gw = (w - 2 * pad - (cols - 1) * 6 * s) / cols
        for i, b in enumerate(self.toggles):
            r, c = divmod(i, cols)
            b.rect = QRectF(pad + c * (gw + 6 * s), h - pad - (2 - r) * (bh + 6 * s) + 6 * s, gw, bh)
        top = 10 * s
        bottom = h - pad - 2 * (bh + 6 * s) - 8 * s
        dw = min(w * 0.42, bottom - top)
        self.diagram = QRectF(pad, top, dw, bottom - top)
        self.stats = QRectF(pad + dw + 14 * s, top, w - dw - 2 * pad - 14 * s, bottom - top)

    def status_text(self) -> str:
        frame = self.ctx.tracking()
        n = frame.present_count if frame else 0
        return f"{n} hand{'s' if n != 1 else ''}"

    # -------------------------------------------------------------- paint
    def paint_content(self, p: QPainter, theme: Theme) -> None:
        frame = self.ctx.tracking()
        self._paint_diagram(p, theme, frame)
        self._paint_stats(p, theme, frame)

    def _paint_diagram(self, p: QPainter, theme: Theme, frame) -> None:
        r = self.diagram
        p.setPen(QPen(theme.border, 1))
        p.setBrush(with_alpha(QColor(8, 9, 11), 0.45))
        p.drawRoundedRect(r, theme.radius_small, theme.radius_small)
        hand = frame.primary if frame else None
        if hand is None or not hand.present or hand.observation is None:
            p.setPen(theme.text_tertiary)
            p.setFont(theme.font(10))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "raise a hand\nto see its landmarks")
            return
        lm = hand.observation.landmarks
        # Normalise the hand into the box (aspect-correct), so the diagram is size-invariant.
        aspect = self.ctx.perf.frame_size[0] / max(1, self.ctx.perf.frame_size[1]) if self.ctx.perf.frame_size[1] else 16 / 9
        xs, ys = lm[:, 0] * aspect, lm[:, 1]
        span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-4)
        inner = r.adjusted(24, 24, -24, -40)
        sc = min(inner.width(), inner.height()) / span
        cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
        pts = [QPointF(inner.center().x() + (x - cx) * sc, inner.center().y() + (y - cy) * sc) for x, y in zip(xs, ys)]
        z = lm[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        p.setPen(QPen(with_alpha(theme.text, 0.35), 1.6))
        for a, b in HAND_CONNECTIONS:
            p.drawLine(pts[a], pts[b])
        # Fingers coloured by extension; joints coloured by relative depth (nearer = warmer).
        f = hand.features
        for fi, chain in enumerate(FINGER_CHAINS):
            ext = f.extended[fi] if f else False
            col = theme.accent if ext else with_alpha(theme.text, 0.55)
            p.setPen(QPen(col, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for a, b in zip(chain, chain[1:]):
                p.drawLine(pts[a], pts[b])
        p.setPen(Qt.PenStyle.NoPen)
        for i, pt in enumerate(pts):
            t = 0.0 if zmax - zmin < 1e-6 else (z[i] - zmin) / (zmax - zmin)
            p.setBrush(mix(theme.warm, theme.accent, float(t)))
            p.drawEllipse(pt, 4.2 if i in (4, 8) else 3.2, 4.2 if i in (4, 8) else 3.2)
        p.setPen(QPen(theme.success if hand.pinched else theme.text_secondary, 1.4, Qt.PenStyle.DashLine))
        p.drawLine(pts[4], pts[8])
        p.setFont(theme.font(9))
        p.setPen(theme.text_tertiary)
        p.drawText(QRectF(r.left() + 10, r.bottom() - 26, r.width() - 20, 18), Qt.AlignmentFlag.AlignLeft,
                   "21 landmarks · colour = relative depth")

    def _row(self, p: QPainter, theme: Theme, y: float, label: str, value: str, color: QColor | None = None) -> float:
        r = self.stats
        p.setFont(theme.font(10))
        p.setPen(theme.text_tertiary)
        p.drawText(QRectF(r.left(), y, r.width() * 0.5, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        p.setFont(theme.font(10, QFont.Weight.Medium))
        p.setPen(color or theme.text)
        p.drawText(QRectF(r.left() + r.width() * 0.42, y, r.width() * 0.58, 18),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, value)
        return y + 19 * theme.scale

    def _header(self, p: QPainter, theme: Theme, y: float, text: str) -> float:
        p.setFont(theme.font(9, QFont.Weight.DemiBold, letter_spacing=0.8))
        p.setPen(theme.accent)
        p.drawText(QRectF(self.stats.left(), y, self.stats.width(), 16), Qt.AlignmentFlag.AlignLeft, text.upper())
        return y + 20 * theme.scale

    def _paint_stats(self, p: QPainter, theme: Theme, frame) -> None:
        perf = self.ctx.perf
        s = self.ctx.settings
        y = self.stats.top()
        y = self._header(p, theme, y, "hands")
        hands = [h for h in (frame.hands if frame else []) if h.present]
        y = self._row(p, theme, y, "detected", f"{len(hands)} / {s.tracking.max_hands}")
        y = self._row(p, theme, y, "interaction", self.ctx.shell.manager.state.value)
        for h in hands[:2]:
            f = h.features
            tag = "primary" if h.is_primary else "secondary"
            y = self._row(p, theme, y, f"#{h.id} {tag}", f"{h.handedness.lower()} · score {h.handedness_score:.2f}",
                          theme.accent if h.is_primary else theme.warm)
            y = self._row(p, theme, y, "  gesture", h.pose.value)
            if f is not None:
                y = self._pinch_bar(p, theme, y, f.pinch_ratio, h.pinched)
                tip = h.observation.landmarks[8] if h.observation is not None else None
                if tip is not None:
                    y = self._row(p, theme, y, "  index tip (u, v, z)", f"{tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:+.3f}")
                y = self._row(p, theme, y, "  pointer anchor", h.pointer_mode)
                y = self._row(p, theme, y, "  jitter (derived)", f"{h.jitter * 1000:.1f} ‰ of frame")
        if not hands:
            y = self._row(p, theme, y, "status", "no hand in view", theme.text_secondary)
        y += 6
        y = self._header(p, theme, y, "pipeline")
        y = self._row(p, theme, y, "camera", f"{perf.camera_fps:.1f} fps · {perf.frame_size[0]}×{perf.frame_size[1]}")
        y = self._row(p, theme, y, "tracking", f"{perf.tracking_fps:.1f} fps")
        y = self._row(p, theme, y, "hand inference", f"{perf.hand_ms:.1f} ms")
        y = self._row(p, theme, y, "segmentation", f"{perf.seg_ms:.1f} ms" if s.performance.segmentation_interval else "off")
        y = self._row(p, theme, y, "capture → display", f"{perf.capture_to_display_ms:.0f} ms")
        if perf.gesture_latency_ms is not None:
            y = self._row(p, theme, y, "last pinch → action", f"{perf.gesture_latency_ms:.0f} ms")
        y = self._row(p, theme, y, "frames skipped", f"{perf.dropped_frames}")
        y += 6
        y = self._header(p, theme, y, "filtering")
        preset = "steady (precision)" if s.tracking.precision_mode else s.tracking.smoothing
        params = SMOOTHING_PRESETS["steady" if s.tracking.precision_mode else s.tracking.smoothing]
        y = self._row(p, theme, y, "one euro", f"{preset} · fc {params.min_cutoff} Hz · β {params.beta}")
        self._row(p, theme, y, "pinch on / off", f"{s.tracking.pinch_enter:.2f} / {s.tracking.pinch_exit:.2f}"
                  + (" (calibrated)" if s.calibrated else ""))

    def _pinch_bar(self, p: QPainter, theme: Theme, y: float, ratio: float, pinched: bool) -> float:
        r = self.stats
        t = self.ctx.settings.tracking
        p.setFont(theme.font(10))
        p.setPen(theme.text_tertiary)
        p.drawText(QRectF(r.left(), y, r.width() * 0.5, 18), Qt.AlignmentFlag.AlignVCenter, "  pinch ratio")
        bar = QRectF(r.left() + r.width() * 0.45, y + 6, r.width() * 0.38, 6)
        mx = 1.2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(theme.text, 0.12))
        p.drawRoundedRect(bar, 3, 3)
        fill = QRectF(bar)
        fill.setWidth(bar.width() * min(1.0, ratio / mx))
        p.setBrush(theme.success if pinched else theme.accent)
        p.drawRoundedRect(fill, 3, 3)
        for thr, col in ((t.pinch_enter, theme.success), (t.pinch_exit, theme.warning)):
            x = bar.left() + bar.width() * min(1.0, thr / mx)
            p.setPen(QPen(col, 1.4))
            p.drawLine(QPointF(x, bar.top() - 3), QPointF(x, bar.bottom() + 3))
        p.setPen(theme.text)
        p.setFont(theme.font(10, QFont.Weight.Medium))
        p.drawText(QRectF(bar.right() + 6, y, r.right() - bar.right() - 6, 18),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{ratio:.2f}")
        return y + 19 * theme.scale
