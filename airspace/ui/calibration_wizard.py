"""First-run setup and gesture calibration wizard.

Steps: welcome → camera → mirror → framing → fingertip targets → pinch →
release → open palm → drag test → done. Every step can be completed with the
hand, advanced with the mouse/keyboard, or skipped entirely.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen

from airspace.interaction.events import CallbackGrab, Grab, InteractionState, PointerEvent
from airspace.ui.animations import Spring
from airspace.ui.controls import Button
from airspace.ui.dialogs import Overlay, paint_glass_card
from airspace.ui.icons import draw_icon
from airspace.ui.styles import Theme, with_alpha
from airspace.vision.calibration import PinchCalibration, compute_pinch_thresholds
from airspace.vision.gesture_classifier import Pose
from airspace.vision.tracking import TrackingFrame

STEPS = ["welcome", "camera", "mirror", "frame", "point", "pinch", "release", "palm", "drag", "done"]

TEXT = {
    "welcome": ("Welcome to airspace", "Let's tune hand tracking to you. It takes about a minute — "
                "you can skip at any time and use the defaults."),
    "camera": ("Choose your camera", "Pick the webcam airspace should use. You can change this later in settings."),
    "mirror": ("Check the mirror", "Wave your right hand. If it moves on the right side of the screen, "
               "the view is mirrored correctly."),
    "frame": ("Find your spot", "Sit so your face and both hands are comfortably in view, about an arm's "
              "length from the screen. Raise a hand to continue."),
    "point": ("Point at the targets", "Point with your index finger. Hold the cursor over each target "
              "until it fills."),
    "pinch": ("Pinch and hold", "Touch your thumb and index fingertips together and hold for a moment."),
    "release": ("Now relax", "Open your fingers to a relaxed, natural hand and hold still."),
    "palm": ("Open palm", "Show an open palm to the camera and hold it. This gesture opens the app launcher."),
    "drag": ("Drag the card", "Pinch the card, move it into the outlined zone, then release."),
    "done": ("You're set", "Preferences saved. Open your palm any time to launch an app — or press ⌘O."),
}


class CalibrationWizard(Overlay):
    dims_background = False

    def __init__(self, cameras: list[tuple[int, str]], current_camera: int, mirror: bool,
                 on_camera: Callable[[int], None], on_mirror: Callable[[bool], None],
                 on_calibrated: Callable[[PinchCalibration], None], on_finish: Callable[[bool], None],
                 pointer: Callable[[], Any]) -> None:
        super().__init__()
        self.cameras = cameras
        self.camera = current_camera
        self.mirror = mirror
        self.on_camera, self.on_mirror = on_camera, on_mirror
        self.on_calibrated, self.on_finish = on_calibrated, on_finish
        self.pointer = pointer
        self.step = 0
        self.card = QRectF()
        self._step_t = time.monotonic()
        self._progress = Spring(0.0, 300)
        self._reset_step_state()
        self.btn_back = self.controls.add(Button("Back", on_click=self.back))
        self.btn_skip = self.controls.add(Button("Skip setup", variant="ghost", on_click=self.skip))
        self.btn_next = self.controls.add(Button("Next", variant="primary", on_click=self.next))
        self.cam_buttons: list[Button] = []
        for idx, name in cameras:
            b = Button(f"{name}", icon="camera", on_click=lambda i=idx: self._pick_camera(i),
                       checked=lambda i=idx: self.camera == i)
            self.cam_buttons.append(self.controls.add(b))
        self.mirror_btn = self.controls.add(Button("Mirror view", icon="restore", on_click=self._toggle_mirror,
                                                   checked=lambda: self.mirror))
        self.calibration: PinchCalibration | None = None

    # ------------------------------------------------------------- state
    def _reset_step_state(self) -> None:
        self.complete = False
        self.samples: list[float] = []
        self.hold_since: float | None = None
        self.targets_done = [False, False, False]
        self.target_since: float | None = None
        self.card_pos: QPointF | None = None
        self.card_drag = False
        self._step_t = time.monotonic()
        self.__dict__.pop("_completed_at", None)
        self.__dict__.pop("_closed_done", None)

    @property
    def name(self) -> str:
        return STEPS[self.step]

    def next(self) -> None:
        if self.name == "release" and self.calibration is None:
            self._finish_pinch_calibration()
        if self.step >= len(STEPS) - 1:
            self.dismiss()
            self.on_finish(True)
            return
        self.step += 1
        self._reset_step_state()

    def back(self) -> None:
        if self.step > 0:
            self.step -= 1
            self._reset_step_state()

    def skip(self) -> None:
        self.dismiss()
        self.on_finish(False)

    def key(self, event: QKeyEvent) -> bool:
        k = event.key()
        if k == Qt.Key.Key_Escape:
            self.skip()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Right):
            self.next()
        elif k == Qt.Key.Key_Left:
            self.back()
        return True

    def _pick_camera(self, idx: int) -> None:
        self.camera = idx
        self.on_camera(idx)

    def _toggle_mirror(self) -> None:
        self.mirror = not self.mirror
        self.on_mirror(self.mirror)

    def _finish_pinch_calibration(self) -> None:
        closed = getattr(self, "_closed_samples", [])
        cal = compute_pinch_thresholds(closed, self.samples)
        self.calibration = cal
        self.on_calibrated(cal)

    # ------------------------------------------------------- per-frame
    def on_tracking(self, frame: TrackingFrame | None) -> None:
        if self.closing or frame is None:
            return
        now = time.monotonic()
        hand = frame.primary
        present = hand is not None and hand.present and hand.features is not None
        n = self.name
        if n == "frame":
            self._hold(present, now, 1.0)
        elif n == "point":
            self._update_targets(now)
        elif n == "pinch":
            if present and hand.features.pinch_ratio < 0.55:
                self.samples.append(hand.features.pinch_ratio)
                self._hold(True, now, 1.2)
            else:
                self._hold(False, now, 1.2)
            if self.complete and not hasattr(self, "_closed_done"):
                self._closed_samples = list(self.samples)
                self._closed_done = True
        elif n == "release":
            closed_level = min(getattr(self, "_closed_samples", [0.2]) or [0.2])
            if present and hand.features.pinch_ratio > closed_level + 0.35:
                self.samples.append(hand.features.pinch_ratio)
                self._hold(True, now, 1.0)
            else:
                self._hold(False, now, 1.0)
            if self.complete and self.calibration is None:
                self._finish_pinch_calibration()
        elif n == "palm":
            self._hold(present and hand.features.pose == Pose.OPEN_PALM, now, 0.6)
        elif n == "mirror":
            self.complete = True
        if self.complete and n in ("frame", "pinch", "release", "palm", "point", "drag"):
            # Auto-advance shortly after success so the flow feels fluid.
            done_at = self.__dict__.setdefault("_completed_at", now)
            if now - done_at > 0.8:
                self.next()

    def _hold(self, active: bool, now: float, seconds: float) -> None:
        if self.complete:
            return
        if active:
            if self.hold_since is None:
                self.hold_since = now
            self._progress.target = min(1.0, (now - self.hold_since) / seconds)
            if now - self.hold_since >= seconds:
                self.complete = True
                self._completed_at = now
        else:
            self.hold_since = None
            self._progress.target = 0.0

    def _target_positions(self) -> list[QPointF]:
        b = self.bounds
        return [QPointF(b.left() + b.width() * 0.2, b.top() + b.height() * 0.28),
                QPointF(b.left() + b.width() * 0.8, b.top() + b.height() * 0.28),
                QPointF(b.left() + b.width() * 0.5, b.top() + b.height() * 0.45)]

    def _update_targets(self, now: float) -> None:
        ptr = self.pointer()
        if ptr is None or not ptr.visible:
            self.target_since = None
            return
        for i, t in enumerate(self._target_positions()):
            if self.targets_done[i]:
                continue
            if math.hypot(ptr.x - t.x(), ptr.y - t.y()) < 42:
                if self.target_since is None:
                    self.target_since = now
                if now - self.target_since > 0.45:
                    self.targets_done[i] = True
                    self.target_since = None
            break  # targets are completed in order
        if all(self.targets_done):
            self.complete = True
            self.hold_since = now
            if not hasattr(self, "_completed_at"):
                self._completed_at = now

    # -------------------------------------------------------- pointer
    def _card_rect(self) -> QRectF:
        b = self.bounds
        if self.card_pos is None:
            self.card_pos = QPointF(b.left() + b.width() * 0.22, b.top() + b.height() * 0.3)
        return QRectF(self.card_pos.x(), self.card_pos.y(), 180, 110)

    def _zone_rect(self) -> QRectF:
        b = self.bounds
        return QRectF(b.left() + b.width() * 0.62, b.top() + b.height() * 0.26, 220, 150)

    def press(self, ev: PointerEvent) -> Grab:
        if self.name == "drag" and not self.complete and self._card_rect().contains(QPointF(ev.x, ev.y)):
            start = QPointF(self.card_pos)
            ox, oy = ev.x - start.x(), ev.y - start.y()
            self.card_drag = True

            def move(e: PointerEvent) -> None:
                self.card_pos = QPointF(e.x - ox, e.y - oy)

            def release(e: PointerEvent) -> None:
                self.card_drag = False
                if self._zone_rect().contains(self._card_rect().center()):
                    self.complete = True
                    self._completed_at = time.monotonic()

            return CallbackGrab(InteractionState.DRAGGING_WINDOW, move, release,
                                lambda: setattr(self, "card_drag", False))
        return super().press(ev)

    # ---------------------------------------------------------- layout
    def layout(self, bounds: QRectF, theme: Theme | None = None) -> None:
        super().layout(bounds)
        s = theme.scale if theme else 1.0
        w = min(600 * s, bounds.width() - 32)
        h = 236 * s + (54 * s if self.name in ("camera", "mirror") else 0)
        self.card = QRectF(bounds.center().x() - w / 2, bounds.bottom() - h - 40 * s, w, h)
        bh = 38 * s
        y = self.card.bottom() - 22 * s - bh
        self.btn_next.rect = QRectF(self.card.right() - 22 * s - 116 * s, y, 116 * s, bh)
        self.btn_back.rect = QRectF(self.btn_next.rect.left() - 10 * s - 96 * s, y, 96 * s, bh)
        self.btn_skip.rect = QRectF(self.card.left() + 16 * s, y, 120 * s, bh)
        self.btn_back.visible = 0 < self.step < len(STEPS) - 1
        self.btn_skip.visible = self.step < len(STEPS) - 1
        self.btn_next.label = {"welcome": "Start", "done": "Finish"}.get(self.name, "Next" if not self.complete else "Continue")
        cam_y = self.card.top() + 120 * s
        bw = min(220 * s, (self.card.width() - 44 * s) / max(1, len(self.cam_buttons)) - 8)
        for i, b in enumerate(self.cam_buttons):
            b.visible = self.name == "camera"
            b.rect = QRectF(self.card.left() + 22 * s + i * (bw + 8), cam_y, bw, 40 * s)
        self.mirror_btn.visible = self.name == "mirror"
        self.mirror_btn.rect = QRectF(self.card.left() + 22 * s, cam_y, 170 * s, 40 * s)

    def tick(self, dt: float) -> None:
        super().tick(dt)
        self._progress.step(dt)

    # ----------------------------------------------------------- paint
    def paint(self, p: QPainter, theme: Theme) -> None:
        self.layout(self.bounds, theme)
        a = self.appear.value
        if a <= 0.01:
            return
        s = theme.scale
        p.save()
        p.setOpacity(a)
        n = self.name
        if n == "point":
            self._paint_targets(p, theme)
        elif n == "drag":
            self._paint_drag(p, theme)
        c = self.card
        paint_glass_card(p, c, theme)
        # progress dots
        for i in range(len(STEPS)):
            x = c.left() + 24 * s + i * 14 * s
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.accent if i <= self.step else with_alpha(theme.text, 0.18))
            p.drawEllipse(QPointF(x, c.top() + 24 * s), 3.2 * s if i == self.step else 2.4 * s,
                          3.2 * s if i == self.step else 2.4 * s)
        p.setPen(theme.text_tertiary)
        p.setFont(theme.font(9.5))
        p.drawText(QRectF(c.right() - 160 * s, c.top() + 14 * s, 140 * s, 20 * s),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"step {self.step + 1} of {len(STEPS)}")
        title, body = TEXT[n]
        p.setPen(theme.text)
        p.setFont(theme.font(17, QFont.Weight.DemiBold))
        p.drawText(QRectF(c.left() + 22 * s, c.top() + 40 * s, c.width() - 140 * s, 32 * s),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, title)
        p.setFont(theme.font(11))
        p.setPen(theme.text_secondary)
        p.drawText(QRectF(c.left() + 22 * s, c.top() + 76 * s, c.width() - 150 * s, 44 * s),
                   Qt.TextFlag.TextWordWrap, body)
        self._paint_status(p, theme, c, s)
        self.controls.paint(p, theme)
        p.restore()

    def _paint_status(self, p: QPainter, theme: Theme, c: QRectF, s: float) -> None:
        n = self.name
        ring_c = QPointF(c.right() - 62 * s, c.top() + 92 * s)
        r = 30 * s
        if n in ("frame", "pinch", "release", "palm", "point", "drag"):
            prog = 1.0 if self.complete else (sum(self.targets_done) / 3 if n == "point" else self._progress.value)
            p.setPen(QPen(with_alpha(theme.text, 0.12), 4 * s))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(ring_c, r, r)
            col = theme.success if self.complete else theme.accent
            p.setPen(QPen(col, 4 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(QRectF(ring_c.x() - r, ring_c.y() - r, 2 * r, 2 * r), 90 * 16, int(-360 * 16 * prog))
            icon = "check" if self.complete else {"frame": "eye", "pinch": "target", "release": "hand",
                                                  "palm": "hand", "point": "target", "drag": "pan"}[n]
            isz = 24 * s
            draw_icon(p, icon, QRectF(ring_c.x() - isz / 2, ring_c.y() - isz / 2, isz, isz),
                      col if self.complete else theme.text_secondary, 1.8)
        if n == "release" and self.calibration is not None:
            cal = self.calibration
            p.setFont(theme.font(10))
            p.setPen(theme.success if cal.valid else theme.warning)
            msg = (f"pinch at {cal.enter:.2f} · release at {cal.exit:.2f}  (your pinch {cal.closed:.2f}, open {cal.open:.2f})"
                   if cal.valid else cal.message)
            p.drawText(QRectF(c.left() + 22 * s, c.top() + 128 * s, c.width() - 44 * s, 24 * s),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, msg)
        if n == "done" and self.calibration is not None and self.calibration.valid:
            p.setFont(theme.font(10))
            p.setPen(theme.text_secondary)
            p.drawText(QRectF(c.left() + 22 * s, c.top() + 128 * s, c.width() - 44 * s, 24 * s),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"Personal pinch thresholds: activate {self.calibration.enter:.2f}, release {self.calibration.exit:.2f}")

    def _paint_targets(self, p: QPainter, theme: Theme) -> None:
        now = time.monotonic()
        active_found = False
        for i, t in enumerate(self._target_positions()):
            done = self.targets_done[i]
            active = not done and not active_found
            active_found = active_found or active
            r = 30
            p.setPen(QPen(with_alpha(theme.success if done else theme.text, 0.9 if active or done else 0.3), 2))
            p.setBrush(with_alpha(QColor(10, 11, 14), 0.45))
            p.drawEllipse(t, r, r)
            if active and self.target_since is not None:
                prog = min(1.0, (now - self.target_since) / 0.45)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(with_alpha(theme.accent, 0.6))
                p.drawEllipse(t, r * prog, r * prog)
            if done:
                draw_icon(p, "check", QRectF(t.x() - 12, t.y() - 12, 24, 24), theme.success, 2)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(theme.text if active else with_alpha(theme.text, 0.3))
                p.drawEllipse(t, 3, 3)

    def _paint_drag(self, p: QPainter, theme: Theme) -> None:
        zone = self._zone_rect()
        p.setPen(QPen(with_alpha(theme.success if self.complete else theme.accent, 0.8), 1.6, Qt.PenStyle.DashLine))
        p.setBrush(with_alpha(theme.accent, 0.06))
        p.drawRoundedRect(zone, 16, 16)
        p.setPen(theme.text_tertiary)
        p.setFont(theme.font(10))
        p.drawText(zone, Qt.AlignmentFlag.AlignCenter, "drop here")
        card = self._card_rect()
        paint_glass_card(p, card, theme, radius=14)
        if self.card_drag:
            p.setPen(QPen(theme.accent, 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(card, 14, 14)
        draw_icon(p, "pan", QRectF(card.center().x() - 14, card.top() + 24, 28, 28), theme.text, 1.6)
        p.setPen(theme.text_secondary)
        p.drawText(QRectF(card.left(), card.top() + 62, card.width(), 24), Qt.AlignmentFlag.AlignCenter, "pinch to grab")
