"""settings — every preference, applied live and saved automatically."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QFont, QPainter, QPen

from airspace.app.configuration import APP_TAGLINE, APP_VERSION
from airspace.interaction.events import CallbackGrab, Grab, InteractionState, PointerEvent
from airspace.miniapps.base_app import MiniApp
from airspace.persistence.settings import BACKGROUND_MODES, CAPTURE_PROFILES, OCCLUSION_MODES, SMOOTHING_PRESETS
from airspace.ui.controls import Button, Control, Segmented, Slider, Toggle
from airspace.ui.styles import Theme, with_alpha

CATEGORIES = [
    ("camera", "camera"), ("tracking", "hand"), ("interface", "layers"), ("performance", "chart"),
    ("accessibility", "eye"), ("workspace", "arrange"), ("about", "info"),
]

SHORTCUTS = [
    ("⌘O", "open the launcher"), ("⌘,", "settings"), ("F11 / ⌃⌘F", "fullscreen"), ("Esc", "close / cancel / exit fullscreen"),
    ("⌘S", "save project or layout"), ("⌘R", "reset workspace"), ("⌘W", "close window"), ("⌘M / ⇧⌘M", "minimise / restore all"),
    ("⌘1–8", "open an app"), ("⌘D", "tracking overlay"), ("⌘K", "camera on/off"), ("⌘B / ⌘L", "background / occlusion mode"),
    ("⇧⌘C", "calibrate"), ("⇧⌘A", "arrange windows"), ("⇧⌘M", "mouse input on/off"), ("⌘Q", "quit (always works)"),
]


class _Row:
    def __init__(self, label: str, control: Control | None, description: str = "", height: float = 46.0,
                 full: bool = False) -> None:
        self.label, self.control, self.description, self.height, self.full = label, control, description, height, full
        self.y = 0.0


class SettingsApp(MiniApp):
    app_type = "settings"
    title = "settings"
    icon = "settings"
    default_size = (640.0, 540.0)
    min_size = (520.0, 420.0)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.category = 0
        self.scroll = 0.0
        self.content_h = 0.0
        self.panel = QRectF()
        self.rows: list[_Row] = []
        self.nav: list[Button] = []
        for i, (name, icon) in enumerate(CATEGORIES):
            b = Button(name, icon=icon, variant="ghost", on_click=lambda i=i: self._select(i),
                       checked=lambda i=i: self.category == i, font_size=10.5)
            self.nav.append(b)
        self._build()

    # ------------------------------------------------------------ helpers
    @property
    def s(self):
        return self.ctx.settings

    @property
    def shell(self):
        return self.ctx.shell

    def _apply(self, save: bool = True) -> None:
        self.shell.apply_settings(save=save)

    def _setter(self, section: str, attr: str, live: bool = True) -> Callable:
        def set_value(v) -> None:
            setattr(getattr(self.s, section), attr, v)
            self._apply(save=not live)
        return set_value

    def _commit(self, *_args) -> None:
        self._apply(save=True)

    def _toggle(self, label: str, section: str, attr: str, description: str = "") -> _Row:
        sec = getattr(self.s, section)

        def set_(v: bool) -> None:
            setattr(sec, attr, v)
            self._apply()

        return _Row("", Toggle(label, lambda: getattr(getattr(self.s, section), attr), set_, description),
                    height=52 if description else 42, full=True)

    def _segmented(self, label: str, options: list[str], get_index: Callable[[], int], on_select: Callable[[int], None],
                   description: str = "") -> _Row:
        return _Row(label, Segmented(options, get_index, on_select), description)

    def _slider(self, label: str, section: str, attr: str, lo: float, hi: float, fmt: Callable[[float], str],
                step: float = 0.0, cast=float, description: str = "") -> _Row:
        sec_name = section

        def change(v: float) -> None:
            setattr(getattr(self.s, sec_name), attr, cast(v))
            self._apply(save=False)

        sl = Slider(lambda: float(getattr(getattr(self.s, sec_name), attr)), change, self._commit, lo, hi, step, label, fmt)
        row = _Row(label, sl, description)
        row.value_fmt = lambda: fmt(float(getattr(getattr(self.s, sec_name), attr)))  # type: ignore[attr-defined]
        return row

    def _button_row(self, buttons: list[tuple[str, str, Callable[[], None], str]]) -> _Row:
        row = _Row("", None, height=44, full=True)
        row.buttons = [Button(label, icon=icon, on_click=cb, variant=variant)  # type: ignore[attr-defined]
                       for label, icon, cb, variant in buttons]
        return row

    def _select(self, i: int) -> None:
        self.category = i
        self.scroll = 0.0
        self._build()
        self.relayout()

    # -------------------------------------------------------------- build
    def _build(self) -> None:
        name = CATEGORIES[self.category][0]
        rows: list[_Row] = []
        later = lambda fn: (lambda: QTimer.singleShot(0, fn))  # noqa: E731
        if name == "camera":
            cams = self.shell.list_cameras()
            row = _Row("device", None, "Names come from the system; airspace opens cameras by index.",
                       height=max(64, 30 + 40 * ((len(cams) + 1) // 2)))
            row.buttons = [Button(f"{i} · {n}", icon="camera", on_click=lambda i=i: self.shell.set_camera(i),  # type: ignore[attr-defined]
                                  checked=lambda i=i: self.s.camera.device_index == i, font_size=10) for i, n in cams]
            row.grid = True  # type: ignore[attr-defined]
            rows.append(row)
            profiles = list(CAPTURE_PROFILES)
            rows.append(self._segmented("resolution", [f"{p} {CAPTURE_PROFILES[p][1]}p" for p in profiles],
                                        lambda: profiles.index(self.s.camera.profile),
                                        lambda i: self.shell.set_camera(self.s.camera.device_index, profiles[i]),
                                        "Requested capture size at 30 fps; the camera may choose the nearest mode."))
            rows.append(self._toggle("mirror view", "camera", "mirror", "Feels like a mirror — recommended"))
            rows.append(_Row("", Toggle("camera enabled", lambda: self.s.camera.enabled, lambda v: self.shell.toggle_camera()),
                             height=42, full=True))
            rows.append(self._button_row([("Calibrate gestures", "target", later(self.shell.start_calibration), "primary")]))
        elif name == "tracking":
            presets = list(SMOOTHING_PRESETS)
            rows.append(self._segmented("smoothing", presets, lambda: presets.index(self.s.tracking.smoothing),
                                        lambda i: (setattr(self.s.tracking, "smoothing", presets[i]), self._apply()),
                                        "One Euro filter: responsive has least lag, steady least jitter."))
            rows.append(self._slider("pointer reach", "tracking", "pointer_reach", 1.0, 1.8, lambda v: f"×{v:.2f}",
                                     description="Above ×1.0 small hand movements reach the screen edges."))
            rows.append(self._slider("pinch activates at", "tracking", "pinch_enter", 0.1, 0.9, lambda v: f"{v:.2f}",
                                     description="Thumb–index distance relative to palm size. Lower = firmer pinch."))
            rows.append(self._slider("pinch releases at", "tracking", "pinch_exit", 0.1, 0.9, lambda v: f"{v:.2f}",
                                     description="Kept above the activation point (hysteresis prevents flicker)."))
            rows.append(self._slider("pinch hold", "tracking", "pinch_hold_ms", 100, 900, lambda v: f"{v:.0f} ms", 10, int))
            rows.append(self._slider("open-palm hold", "tracking", "palm_hold_ms", 150, 1200, lambda v: f"{v:.0f} ms", 10, int,
                                     "How long to show a palm before the launcher opens."))
            rows.append(self._segmented("hands", ["one", "two"], lambda: self.s.tracking.max_hands - 1,
                                        lambda i: (setattr(self.s.tracking, "max_hands", i + 1), self._apply())))
            hands = ["auto", "left", "right"]
            rows.append(self._segmented("pointer hand", hands, lambda: hands.index(self.s.tracking.dominant_hand),
                                        lambda i: (setattr(self.s.tracking, "dominant_hand", hands[i]), self._apply())))
            rows.append(self._toggle("precision mode", "tracking", "precision_mode",
                                     "Steadier filtering and 1:1 reach for small targets"))
            rows.append(self._toggle("hand skeleton overlay", "tracking", "show_skeleton"))
            rows.append(self._toggle("pinch zones overlay", "tracking", "show_zones"))
        elif name == "interface":
            occ = list(OCCLUSION_MODES)
            rows.append(self._segmented("occlusion", occ, lambda: occ.index(self.s.interface.occlusion),
                                        lambda i: (setattr(self.s.interface, "occlusion", occ[i]), self._apply()),
                                        "balanced: hands pass in front · cinematic: whole body · clear: UI always on top"))
            bg = list(BACKGROUND_MODES)
            rows.append(self._segmented("background", bg, lambda: bg.index(self.s.interface.background),
                                        lambda i: (setattr(self.s.interface, "background", bg[i]), self._apply()),
                                        "Tracking keeps running with any background."))
            rows.append(self._slider("glass opacity", "interface", "transparency", 0.3, 0.95, lambda v: f"{v * 100:.0f}%"))
            rows.append(self._slider("interface scale", "interface", "window_scale", 0.75, 1.5, lambda v: f"{v * 100:.0f}%"))
            rows.append(self._slider("cursor size", "interface", "cursor_size", 0.6, 2.0, lambda v: f"{v * 100:.0f}%"))
            rows.append(self._toggle("animations", "interface", "animations"))
            rows.append(self._toggle("interface sounds", "interface", "sounds"))
        elif name == "performance":
            seg_opts = [0, 1, 2, 3]
            rows.append(self._segmented("segmentation", ["off", "every frame", "every 2nd", "every 3rd"],
                                        lambda: seg_opts.index(min(3, self.s.performance.segmentation_interval)),
                                        lambda i: (setattr(self.s.performance, "segmentation_interval", seg_opts[i]), self._apply()),
                                        "Person mask used for occlusion and window placement."))
            q = ["high", "low"]
            rows.append(self._segmented("render quality", q, lambda: q.index(self.s.performance.render_quality),
                                        lambda i: (setattr(self.s.performance, "render_quality", q[i]), self._apply())))
            fps = [30, 60, 120]
            rows.append(self._segmented("ui frame rate", [f"{f} fps" for f in fps],
                                        lambda: fps.index(self.s.performance.fps_target) if self.s.performance.fps_target in fps else 1,
                                        lambda i: (setattr(self.s.performance, "fps_target", fps[i]), self._apply())))
            r = ["opengl", "raster"]
            rows.append(self._segmented("renderer", r, lambda: r.index(self.s.performance.renderer),
                                        lambda i: (setattr(self.s.performance, "renderer", r[i]), self._apply(),
                                                   self.ctx.toast("Renderer change applies after restart")),
                                        "Takes effect the next time airspace starts."))
            profiles = list(CAPTURE_PROFILES)
            rows.append(self._segmented("capture profile", profiles, lambda: profiles.index(self.s.camera.profile),
                                        lambda i: self.shell.set_camera(self.s.camera.device_index, profiles[i])))
        elif name == "accessibility":
            rows.append(self._toggle("reduced motion", "accessibility", "reduced_motion", "Settle animations instantly"))
            rows.append(self._toggle("larger buttons", "accessibility", "large_buttons"))
            rows.append(self._toggle("high contrast", "accessibility", "high_contrast"))
            rows.append(self._toggle("mouse & trackpad", "accessibility", "mouse_fallback", "⇧⌘M toggles this from anywhere"))
            rows.append(self._toggle("dwell to click", "accessibility", "dwell_click",
                                     "Hold the hand cursor still over a button to press it — no pinch needed"))
            rows.append(self._slider("dwell time", "accessibility", "dwell_ms", 400, 2500, lambda v: f"{v:.0f} ms", 50, int))
        elif name == "workspace":
            sh = self.shell
            rows.append(self._button_row([("Save layout", "save", sh.save_layout, "default"),
                                          ("Restore layout", "restore", sh.restore_layout, "default")]))
            rows.append(self._button_row([("Arrange windows", "arrange", sh.arrange_windows, "default"),
                                          ("Minimise all", "minimize", sh.wm.minimize_all, "default")]))
            rows.append(self._button_row([("Reset layout", "reset", lambda: sh.reset_workspace(True), "danger"),
                                          ("Clear saved settings", "trash", sh.clear_saved_settings, "danger")]))
        else:  # about
            rows.append(_Row("about", None, height=200, full=True))
            rows.append(_Row("shortcuts", None, height=24 + 21 * ((len(SHORTCUTS) + 1) // 2), full=True))
        self.rows = rows
        keep = list(self.nav)
        self.controls.clear()
        for b in keep:
            self.controls.add(b)
        for r in rows:
            if r.control:
                self.controls.add(r.control)
            for b in getattr(r, "buttons", []):
                self.controls.add(b)

    # ------------------------------------------------------------- layout
    def layout(self, w: float, h: float) -> None:
        sc = self.theme.scale
        nav_w = 150 * sc
        y = 12 * sc
        for b in self.nav:
            b.rect = QRectF(10 * sc, y, nav_w - 16 * sc, 34 * sc)
            y += 38 * sc
        self.panel = QRectF(nav_w + 6 * sc, 0, w - nav_w - 6 * sc, h)
        x0 = self.panel.left() + 12 * sc
        cw = self.panel.width() - 28 * sc
        y = 14 * sc - self.scroll
        for r in self.rows:
            r.y = y
            hh = r.height * sc
            if r.control is not None:
                if r.full:
                    r.control.rect = QRectF(x0 - 8 * sc, y + 2, cw + 8 * sc, hh - 6)
                else:
                    top = y + 22 * sc
                    r.control.rect = QRectF(x0, top, cw, (hh - 26 * sc) if isinstance(r.control, Segmented) else 22 * sc)
                    if isinstance(r.control, Segmented):
                        r.control.rect.setHeight(30 * sc)
            btns = getattr(r, "buttons", [])
            if btns:
                if getattr(r, "grid", False):
                    bw = (cw - 8 * sc) / 2
                    for i, b in enumerate(btns):
                        row_i, col = divmod(i, 2)
                        b.rect = QRectF(x0 + col * (bw + 8 * sc), y + 24 * sc + row_i * 40 * sc, bw, 34 * sc)
                else:
                    bw = (cw - 8 * sc * (len(btns) - 1)) / len(btns)
                    for i, b in enumerate(btns):
                        b.rect = QRectF(x0 + i * (bw + 8 * sc), y + 4 * sc, bw, 36 * sc)
            extra = (r.description and not r.full) and 18 * sc or 0
            r.extra = extra  # type: ignore[attr-defined]
            y += hh + extra + 8 * sc
        self.content_h = y + self.scroll
        for c in self.controls.controls:
            if c in self.nav:
                continue
            c.visible = c.rect.bottom() > 4 and c.rect.top() < h - 4

    # -------------------------------------------------------------- input
    def wheel(self, x: float, y: float, delta: float, modifiers: int = 0) -> bool:
        if self.panel.contains(QPointF(x, y)):
            self._scroll_by(-delta * 0.6)
            return True
        return False

    def _scroll_by(self, d: float) -> None:
        max_scroll = max(0.0, self.content_h - self.height + 20)
        self.scroll = max(0.0, min(max_scroll, self.scroll + d))
        self.relayout()

    def content_press(self, ev: PointerEvent) -> Grab | None:
        if not self.panel.contains(QPointF(ev.x, ev.y)):
            return None
        start, sy = self.scroll, ev.y

        def move(e: PointerEvent) -> None:
            self._scroll_by(start - (e.y - sy) - self.scroll)

        return CallbackGrab(InteractionState.PANNING, move, None)

    # -------------------------------------------------------------- paint
    def paint_content(self, p: QPainter, theme: Theme) -> None:
        sc = theme.scale
        p.setPen(QPen(theme.border, 1))
        p.drawLine(QPointF(self.panel.left(), 10), QPointF(self.panel.left(), self.height - 10))
        p.save()
        p.setClipRect(self.panel)
        x0 = self.panel.left() + 12 * sc
        cw = self.panel.width() - 28 * sc
        name = CATEGORIES[self.category][0]
        for r in self.rows:
            if r.y > self.height or r.y + r.height * sc < 0:
                continue
            if r.label == "about" and name == "about":
                self._paint_about(p, theme, QRectF(x0, r.y, cw, r.height * sc))
                continue
            if r.label == "shortcuts":
                self._paint_shortcuts(p, theme, QRectF(x0, r.y, cw, r.height * sc))
                continue
            if r.label and not r.full:
                p.setFont(theme.font(10.5, QFont.Weight.Medium))
                p.setPen(theme.text)
                p.drawText(QRectF(x0, r.y, cw, 18 * sc), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, r.label)
                fmt = getattr(r, "value_fmt", None)
                if fmt:
                    p.setPen(theme.text_secondary)
                    p.drawText(QRectF(x0, r.y, cw, 18 * sc), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fmt())
            if r.description and not r.full:
                p.setFont(theme.font(9))
                p.setPen(theme.text_tertiary)
                bottom = r.control.rect.bottom() if r.control else r.y + r.height * sc
                if getattr(r, "grid", False):
                    btns = getattr(r, "buttons", [])
                    bottom = max(b.rect.bottom() for b in btns) if btns else bottom
                p.drawText(QRectF(x0, bottom + 3, cw, 16 * sc), Qt.AlignmentFlag.AlignLeft, r.description)
        p.restore()

    def paint(self, p: QPainter, theme: Theme) -> None:
        self.paint_content(p, theme)
        for b in self.nav:
            b.paint(p, theme)
        p.save()
        p.setClipRect(self.panel)
        for c in self.controls.controls:
            if c not in self.nav and c.visible:
                c.paint(p, theme)
        p.restore()
        # scroll indicator
        if self.content_h > self.height:
            frac = self.height / self.content_h
            bar_h = max(30.0, self.height * frac)
            y = (self.height - bar_h) * (self.scroll / max(1.0, self.content_h - self.height + 20))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(with_alpha(theme.text, 0.18))
            p.drawRoundedRect(QRectF(self.width - 6, y, 3, bar_h), 1.5, 1.5)

    def _paint_about(self, p: QPainter, theme: Theme, r: QRectF) -> None:
        sc = theme.scale
        p.setPen(theme.text)
        p.setFont(theme.font(24, QFont.Weight.Light, letter_spacing=0.8))
        p.drawText(QRectF(r.left(), r.top(), r.width(), 36 * sc), Qt.AlignmentFlag.AlignLeft, "airspace")
        p.setFont(theme.font(10.5))
        p.setPen(theme.text_secondary)
        p.drawText(QRectF(r.left(), r.top() + 38 * sc, r.width(), 18 * sc), Qt.AlignmentFlag.AlignLeft,
                   f"version {APP_VERSION} · {APP_TAGLINE}")
        text = ("Privacy: all video is processed on this computer. airspace never uploads or records camera "
                "frames, never uses the microphone, and keeps no remote analytics.\n\n"
                "Built on MediaPipe (Apache 2.0), Qt for Python / PySide6 (LGPLv3), OpenCV (Apache 2.0), NumPy, "
                "psutil, ModernGL and trimesh (MIT/BSD). See THIRD_PARTY_NOTICES.md for details.")
        p.setFont(theme.font(10))
        p.drawText(QRectF(r.left(), r.top() + 66 * sc, r.width(), r.height() - 66 * sc), Qt.TextFlag.TextWordWrap, text)

    def _paint_shortcuts(self, p: QPainter, theme: Theme, r: QRectF) -> None:
        sc = theme.scale
        p.setFont(theme.font(9, QFont.Weight.DemiBold, letter_spacing=0.8))
        p.setPen(theme.accent)
        p.drawText(QRectF(r.left(), r.top(), r.width(), 18 * sc), Qt.AlignmentFlag.AlignLeft, "KEYBOARD")
        col_w = r.width() / 2
        for i, (keys, desc) in enumerate(SHORTCUTS):
            col, row = divmod(i, (len(SHORTCUTS) + 1) // 2)
            y = r.top() + 22 * sc + row * 21 * sc
            x = r.left() + col * col_w
            p.setFont(theme.font(9.5, QFont.Weight.Medium))
            p.setPen(theme.text)
            p.drawText(QRectF(x, y, 86 * sc, 18 * sc), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, keys)
            p.setFont(theme.font(9.5))
            p.setPen(theme.text_secondary)
            p.drawText(QRectF(x + 88 * sc, y, col_w - 92 * sc, 18 * sc), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, desc)
