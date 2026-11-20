"""canvas — air drawing.

Pinch inside the drawing surface to put the brush down; move to draw; open
the pinch to lift it. Because the window manager hands the whole pinch to
whichever target it started on, pinching the title bar drags the window and
never draws, and simply pointing never draws.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPainterPath, QPen

from airspace.interaction.events import Grab, InteractionState, PointerEvent
from airspace.miniapps.base_app import MiniApp
from airspace.miniapps.canvas.brush_engine import HEIGHT, WIDTH, CanvasDocument, smoothed_segments
from airspace.ui.controls import Button, Swatch
from airspace.ui.styles import Theme, with_alpha

log = logging.getLogger(__name__)

PALETTE = [
    QColor(244, 241, 234), QColor(156, 182, 214), QColor(232, 196, 150),
    QColor(226, 122, 111), QColor(140, 199, 160), QColor(186, 160, 222),
]
SIZES = [4.0, 9.0, 18.0]
PAPER = QColor(19, 21, 25)


def paint_stroke(p: QPainter, stroke, scale: float, eraser_clear: bool = True) -> None:
    color = QColor(*stroke.color)
    if stroke.eraser and eraser_clear:
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        color = QColor(0, 0, 0)
    for (x0, y0), (cx, cy), (x1, y1), w in smoothed_segments(stroke.points):
        path = QPainterPath(QPointF(x0 * scale, y0 * scale))
        path.quadTo(QPointF(cx * scale, cy * scale), QPointF(x1 * scale, y1 * scale))
        pen = QPen(color, max(0.8, w * scale), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)


class _DrawGrab(Grab):
    kind = InteractionState.DRAWING

    def __init__(self, app: "CanvasApp", ev: PointerEvent) -> None:
        self.app = app
        x, y = app.to_doc(ev.x, ev.y)
        app.doc.begin(x, y, app.color_tuple(), app.brush_size(), app.eraser, ev.t)

    def move(self, ev: PointerEvent) -> None:
        x, y = self.app.to_doc(ev.x, ev.y)
        self.app.doc.add_point(x, y, ev.t)

    def release(self, ev: PointerEvent) -> None:
        if not ev.cancelled:
            x, y = self.app.to_doc(ev.x, ev.y)
            self.app.doc.add_point(x, y, ev.t)
        self.app.doc.end()

    def cancel(self) -> None:
        self.app.doc.cancel()


class CanvasApp(MiniApp):
    app_type = "canvas"
    title = "canvas"
    icon = "brush"
    default_size = (720.0, 520.0)
    min_size = (460.0, 340.0)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.doc = CanvasDocument()
        self.color_index = 0
        self.size_index = 1
        self.eraser = False
        self.surface = QRectF()
        self._layer: QImage | None = None
        self._layer_rev = -1
        self._layer_size = (0, 0)
        self.project_path: Path | None = None
        c = self.controls
        self.btn_pen = c.add(Button(icon="pen", variant="icon", on_click=lambda: self._set_eraser(False),
                                    checked=lambda: not self.eraser, tooltip="Brush (P)"))
        self.btn_eraser = c.add(Button(icon="eraser", variant="icon", on_click=lambda: self._set_eraser(True),
                                       checked=lambda: self.eraser, tooltip="Eraser (E)"))
        self.swatches = [c.add(Swatch(col, lambda i=i: self.color_index == i and not self.eraser,
                                      lambda i=i: self._set_color(i))) for i, col in enumerate(PALETTE)]
        self.size_btns = [c.add(_SizeButton(i, self)) for i in range(len(SIZES))]
        self.btn_undo = c.add(Button(icon="undo", variant="icon", on_click=self.doc.undo, tooltip="Undo ⌘Z"))
        self.btn_redo = c.add(Button(icon="redo", variant="icon", on_click=self.doc.redo, tooltip="Redo ⇧⌘Z"))
        self.btn_clear = c.add(Button(icon="trash", variant="icon", on_click=self._clear, tooltip="Clear"))
        self.btn_open = c.add(Button(icon="folder", variant="icon", on_click=lambda: QTimer.singleShot(0, self.open_project),
                                     tooltip="Open project"))
        self.btn_save = c.add(Button(icon="save", variant="icon", on_click=lambda: QTimer.singleShot(0, self.save_project),
                                     tooltip="Save project ⌘S"))
        self.btn_png = c.add(Button("PNG", icon="image", variant="ghost", on_click=lambda: QTimer.singleShot(0, self.export_png),
                                    tooltip="Export PNG"))
        self._autosave = ctx.data_dir / "canvas_autosave.json"

    # ---------------------------------------------------------- helpers
    def color_tuple(self) -> tuple[int, int, int, int]:
        c = PALETTE[self.color_index]
        return c.red(), c.green(), c.blue(), 255

    def brush_size(self) -> float:
        return SIZES[self.size_index] * (2.2 if self.eraser else 1.0)

    def _set_eraser(self, value: bool) -> None:
        self.eraser = value

    def _set_color(self, i: int) -> None:
        self.color_index = i
        self.eraser = False

    def _clear(self) -> None:
        if self.doc.clear():
            self.ctx.toast("Canvas cleared — undo to restore")

    @property
    def doc_scale(self) -> float:
        return self.surface.width() / WIDTH if self.surface.width() > 0 else 1.0

    def to_doc(self, x: float, y: float) -> tuple[float, float]:
        s = self.doc_scale
        dx = (x - self.surface.left()) / s
        dy = (y - self.surface.top()) / s
        return min(max(dx, 0.0), WIDTH), min(max(dy, 0.0), HEIGHT)

    # ----------------------------------------------------------- layout
    def layout(self, w: float, h: float) -> None:
        s = self.theme.scale
        bar = 46 * s
        btn = 32 * s
        y = (bar - btn) / 2 + 2
        x = 12 * s
        for b in (self.btn_pen, self.btn_eraser):
            b.rect = QRectF(x, y, btn, btn)
            x += btn + 2
        x += 8 * s
        for sw in self.swatches:
            sw.rect = QRectF(x, y, btn - 4, btn)
            x += btn - 2
        x += 8 * s
        for sb in self.size_btns:
            sb.rect = QRectF(x, y, btn - 4, btn)
            x += btn - 2
        right = w - 12 * s
        png_w = 70 * s
        self.btn_png.rect = QRectF(right - png_w, y, png_w, btn)
        right -= png_w + 4
        for b in (self.btn_save, self.btn_open, self.btn_clear, self.btn_redo, self.btn_undo):
            b.rect = QRectF(right - btn, y, btn, btn)
            right -= btn + 2
        compact = x > right - 6
        for b in (self.btn_open, self.btn_save):
            b.visible = not compact
        # Surface keeps the document's aspect ratio inside the remaining area.
        pad = 12 * s
        avail = QRectF(pad, bar + 4, w - 2 * pad, h - bar - 4 - pad)
        aspect = WIDTH / HEIGHT
        if avail.width() / max(avail.height(), 1) > aspect:
            sw = avail.height() * aspect
            self.surface = QRectF(avail.center().x() - sw / 2, avail.top(), sw, avail.height())
        else:
            sh = avail.width() / aspect
            self.surface = QRectF(avail.left(), avail.center().y() - sh / 2, avail.width(), sh)

    # ------------------------------------------------------------ input
    def content_press(self, ev: PointerEvent) -> Grab | None:
        if self.surface.contains(QPointF(ev.x, ev.y)):
            return _DrawGrab(self, ev)
        return None

    def key(self, event: QKeyEvent) -> bool:
        k, mods = event.key(), event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if ctrl and k == Qt.Key.Key_Z:
            self.doc.redo() if shift else self.doc.undo()
            return True
        if ctrl and k == Qt.Key.Key_S:
            QTimer.singleShot(0, self.save_project)
            return True
        if ctrl and k == Qt.Key.Key_E:
            QTimer.singleShot(0, self.export_png)
            return True
        if ctrl:
            return False
        if k == Qt.Key.Key_E:
            self.eraser = True
        elif k in (Qt.Key.Key_P, Qt.Key.Key_B):
            self.eraser = False
        elif k == Qt.Key.Key_BracketLeft:
            self.size_index = max(0, self.size_index - 1)
        elif k == Qt.Key.Key_BracketRight:
            self.size_index = min(len(SIZES) - 1, self.size_index + 1)
        elif Qt.Key.Key_1 <= k <= Qt.Key.Key_6:
            self._set_color(k - Qt.Key.Key_1)
        else:
            return False
        return True

    # -------------------------------------------------------------- io
    def save_project(self) -> None:
        default = (self.project_path.name if self.project_path else "drawing.airspace-canvas.json")
        path = self.ctx.save_file("Save canvas project", default, "airspace canvas (*.json)")
        if not path:
            return
        try:
            self.doc.save(Path(path))
            self.project_path = Path(path)
            self.ctx.toast(f"Saved {Path(path).name}")
        except OSError as exc:
            self.ctx.toast(f"Couldn't save: {exc}")

    def open_project(self) -> None:
        files = self.ctx.open_files("Open canvas project", "airspace canvas (*.json)", multiple=False)
        if not files:
            return
        try:
            self.doc.load(Path(files[0]))
            self.project_path = Path(files[0])
            self.ctx.toast(f"Opened {Path(files[0]).name}")
        except (OSError, ValueError) as exc:
            self.ctx.toast(f"Couldn't open: {exc}")

    def render_image(self, scale: float = 1.0, background: QColor | None = PAPER) -> QImage:
        img = QImage(int(WIDTH * scale), int(HEIGHT * scale), QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for stroke in self.doc.strokes:
            paint_stroke(p, stroke, scale)
        p.end()
        if background is None:
            return img
        out = QImage(img.size(), QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(background)
        p = QPainter(out)
        p.drawImage(0, 0, img)
        p.end()
        return out

    def export_png(self) -> None:
        path = self.ctx.save_file("Export drawing as PNG", "drawing.png", "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if self.render_image().save(path, "PNG"):
            self.ctx.toast(f"Exported {Path(path).name}")
        else:
            self.ctx.toast("Couldn't write the PNG")

    def save_state(self) -> dict:
        try:
            self.doc.save(self._autosave)
        except OSError:
            log.warning("Canvas autosave failed")
        return {"color": self.color_index, "size": self.size_index, "autosave": True}

    def restore_state(self, state: dict) -> None:
        self.color_index = int(state.get("color", 0)) % len(PALETTE)
        self.size_index = int(state.get("size", 1)) % len(SIZES)
        if state.get("autosave") and self._autosave.exists():
            try:
                self.doc.load(self._autosave)
            except (OSError, ValueError):
                log.warning("Canvas autosave unreadable")

    def status_text(self) -> str:
        n = len(self.doc.strokes)
        return f"{n} stroke{'s' if n != 1 else ''}" if n else "pinch to draw"

    # ----------------------------------------------------------- paint
    def _ensure_layer(self, dpr: float = 2.0) -> None:
        w, h = int(self.surface.width() * dpr), int(self.surface.height() * dpr)
        if w <= 0 or h <= 0:
            return
        if self._layer is None or self._layer_size != (w, h) or self._layer_rev != self.doc.revision:
            img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(QColor(0, 0, 0, 0))
            p = QPainter(img)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            scale = w / WIDTH
            for stroke in self.doc.strokes:
                paint_stroke(p, stroke, scale)
            p.end()
            self._layer, self._layer_size, self._layer_rev = img, (w, h), self.doc.revision

    def paint_content(self, p: QPainter, theme: Theme) -> None:
        s = self.surface
        path = QPainterPath()
        path.addRoundedRect(s, theme.radius_small, theme.radius_small)
        p.setPen(QPen(theme.border, 1))
        p.setBrush(with_alpha(PAPER, 0.82))
        p.drawPath(path)
        p.save()
        p.setClipPath(path)
        self._ensure_layer()
        if self._layer is not None:
            p.drawImage(s, self._layer)
        cur = self.doc.current
        if cur is not None:
            p.translate(s.topLeft())
            if cur.eraser:
                # Preview the eraser path as a soft outline (actual erasing happens on commit).
                ghost = type(cur)(color=(255, 255, 255, 60), size=cur.size, eraser=False, points=cur.points)
                paint_stroke(p, ghost, self.doc_scale)
            else:
                paint_stroke(p, cur, self.doc_scale)
        p.restore()
        if self.doc.is_empty and cur is None:
            p.setPen(theme.text_tertiary)
            p.setFont(theme.font(11))
            p.drawText(s, Qt.AlignmentFlag.AlignCenter, "pinch and move to draw · drag with the mouse")
        # Undo/redo availability
        self.btn_undo.enabled = self.doc.can_undo
        self.btn_redo.enabled = self.doc.can_redo
        self.btn_clear.enabled = not self.doc.is_empty


class _SizeButton(Button):
    def __init__(self, index: int, app: CanvasApp) -> None:
        super().__init__(variant="icon", on_click=lambda: setattr(app, "size_index", index),
                         checked=lambda: app.size_index == index, tooltip="Brush size")
        self.index = index
        self.app = app

    def paint(self, p: QPainter, theme: Theme) -> None:
        super().paint(p, theme)
        r = 2.0 + self.index * 2.6
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.text if self.is_checked() else theme.text_secondary)
        p.drawEllipse(self.rect.center(), r, r)
