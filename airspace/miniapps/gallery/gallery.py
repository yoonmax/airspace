"""gallery — browse imported photos with gestures.

Pinch-drag the image to pan; add the second hand to zoom and rotate; swipe to
change images; pinch a thumbnail to open it. Images are only ever read; view
rotation and zoom never touch the original files.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPainterPath, QPen

from airspace.interaction.events import Grab, InteractionState, PointerEvent, TwoHandInfo
from airspace.miniapps.base_app import MiniApp
from airspace.miniapps.gallery.image_viewer import ImageCache, ViewTransform, fit_scale
from airspace.ui.controls import Button
from airspace.ui.styles import Theme, with_alpha

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.heic)"
THUMB = 192
FULL = 2400


class GalleryModel:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.index = 0

    @property
    def current(self) -> str | None:
        return self.paths[self.index] if self.paths else None

    def add(self, paths: list[str]) -> int:
        added = 0
        for p in paths:
            if p not in self.paths:
                self.paths.append(p)
                added += 1
        if added and len(self.paths) == added:
            self.index = 0
        return added

    def next(self) -> bool:
        if len(self.paths) < 2:
            return False
        self.index = (self.index + 1) % len(self.paths)
        return True

    def prev(self) -> bool:
        if len(self.paths) < 2:
            return False
        self.index = (self.index - 1) % len(self.paths)
        return True

    def select(self, i: int) -> None:
        if 0 <= i < len(self.paths):
            self.index = i

    def remove_current(self) -> str | None:
        if not self.paths:
            return None
        removed = self.paths.pop(self.index)
        self.index = min(self.index, max(0, len(self.paths) - 1))
        return removed


class _PanGrab(Grab):
    kind = InteractionState.PANNING
    supports_two_hand = True

    def __init__(self, app: "GalleryApp", ev: PointerEvent) -> None:
        self.app = app
        self.sx, self.sy = ev.x, ev.y
        v = app.view
        self.start = (v.pan_x, v.pan_y)
        self._two: tuple[float, float, float, float] | None = None
        self._rebase = False

    def move(self, ev: PointerEvent) -> None:
        if self._two is not None:
            return
        v = self.app.view
        if self._rebase:  # continue panning after a two-hand gesture without a jump
            self.sx, self.sy, self.start, self._rebase = ev.x, ev.y, (v.pan_x, v.pan_y), False
        v.pan_x = self.start[0] + ev.x - self.sx
        v.pan_y = self.start[1] + ev.y - self.sy

    def release(self, ev: PointerEvent) -> None:
        v = self.app.view
        nearest = round(v.free_angle / 90.0) * 90.0
        if abs(v.free_angle - nearest) < 8:  # snap near-square rotations
            v.free_angle = nearest

    def two_hand_begin(self, info: TwoHandInfo) -> bool:
        v = self.app.view
        self._two = (v.zoom, v.pan_x, v.pan_y, v.free_angle)
        self.kind = InteractionState.MANIPULATING_OBJECT
        return True

    def two_hand_update(self, start: TwoHandInfo, now: TwoHandInfo) -> None:
        if self._two is None:
            return
        zoom0, px0, py0, ang0 = self._two
        v = self.app.view
        factor = now.distance / max(start.distance, 1.0)
        v.zoom = max(v.MIN_ZOOM, min(v.MAX_ZOOM, zoom0 * factor))
        v.pan_x = px0 + (now.cx - start.cx)
        v.pan_y = py0 + (now.cy - start.cy)
        d = math.degrees(now.angle - start.angle)
        v.free_angle = ang0 + ((d + 180) % 360 - 180)

    def two_hand_end(self) -> None:
        self._two = None
        self.kind = InteractionState.PANNING
        self._rebase = True


class GalleryApp(MiniApp):
    app_type = "gallery"
    title = "gallery"
    icon = "image"
    default_size = (680.0, 520.0)
    min_size = (420.0, 360.0)
    accepts_swipe = True

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.model = GalleryModel()
        self.view = ViewTransform()
        self.cache = ImageCache()
        self.viewer = QRectF()
        self.strip = QRectF()
        self.strip_offset = 0.0
        self.thumb_rects: list[QRectF] = []
        self._slide = 0.0  # transition offset after swipes
        c = self.controls
        defer = lambda fn: (lambda: QTimer.singleShot(0, fn))  # noqa: E731
        self.btn_import = c.add(Button("Import", icon="folder", on_click=defer(self.import_images), variant="ghost"))
        self.btn_prev = c.add(Button(icon="chevron-left", variant="icon", on_click=self.prev, tooltip="Previous"))
        self.btn_next = c.add(Button(icon="chevron-right", variant="icon", on_click=self.next, tooltip="Next"))
        self.btn_zoom_out = c.add(Button(icon="zoom-out", variant="icon", on_click=lambda: self.view.zoom_by(1 / 1.25)))
        self.btn_zoom_in = c.add(Button(icon="zoom-in", variant="icon", on_click=lambda: self.view.zoom_by(1.25)))
        self.btn_rotate = c.add(Button(icon="rotate", variant="icon", on_click=lambda: self.view.rotate_90(1), tooltip="Rotate view"))
        self.btn_fit = c.add(Button(icon="fit", variant="icon", on_click=self.view.fit, tooltip="Fit"))
        self.btn_reset = c.add(Button(icon="reset", variant="icon", on_click=self.view.reset, tooltip="Reset view"))
        self.btn_remove = c.add(Button(icon="trash", variant="icon", on_click=self.remove_current, tooltip="Remove from gallery"))
        self.btn_empty_import = c.add(Button("Import photos", icon="folder", variant="primary", on_click=defer(self.import_images)))
        self.btn_samples = c.add(Button("Add sample images", icon="sparkle", on_click=defer(self.add_samples)))

    # ---------------------------------------------------------- actions
    def import_images(self) -> None:
        files = self.ctx.open_files("Import images into gallery", IMAGE_FILTER)
        if files:
            n = self.model.add(files)
            self.ctx.toast(f"Added {n} image{'s' if n != 1 else ''}")
            self.relayout()

    def add_samples(self) -> None:
        paths = generate_sample_images(self.ctx.data_dir / "sample_images")
        self.model.add([str(p) for p in paths])
        self.relayout()

    def next(self) -> None:
        if self.model.next():
            self.view.reset()
            self._slide = 1.0

    def prev(self) -> None:
        if self.model.prev():
            self.view.reset()
            self._slide = -1.0

    def remove_current(self) -> None:
        removed = self.model.remove_current()
        if removed:
            self.cache.forget(removed)
            self.view.reset()
            self.ctx.toast("Removed from gallery (the file is untouched)")
            self.relayout()

    def swipe(self, direction: str) -> bool:
        if direction == "left":
            self.next()
            return bool(self.model.paths)
        if direction == "right":
            self.prev()
            return bool(self.model.paths)
        return False

    # ----------------------------------------------------------- layout
    def layout(self, w: float, h: float) -> None:
        s = self.theme.scale
        bar = 44 * s
        btn = 32 * s
        y = (bar - btn) / 2 + 2
        self.btn_import.rect = QRectF(10 * s, y, 96 * s, btn)
        x = w - 10 * s
        for b in (self.btn_remove, self.btn_reset, self.btn_fit, self.btn_rotate, self.btn_zoom_in,
                  self.btn_zoom_out, self.btn_next, self.btn_prev):
            x -= btn
            b.rect = QRectF(x, y, btn, btn)
            x -= 2
        has = bool(self.model.paths)
        for b in (self.btn_prev, self.btn_next, self.btn_zoom_in, self.btn_zoom_out, self.btn_rotate, self.btn_fit,
                  self.btn_reset, self.btn_remove):
            b.visible = has
        strip_h = (78 * s) if has else 0
        self.viewer = QRectF(10 * s, bar + 2, w - 20 * s, h - bar - strip_h - 14 * s)
        self.strip = QRectF(10 * s, h - strip_h - 6 * s, w - 20 * s, strip_h)
        self.btn_empty_import.visible = self.btn_samples.visible = not has
        cx, cy = self.viewer.center().x(), self.viewer.center().y()
        self.btn_empty_import.rect = QRectF(cx - 170 * s, cy + 20 * s, 160 * s, 40 * s)
        self.btn_samples.rect = QRectF(cx + 10 * s, cy + 20 * s, 170 * s, 40 * s)
        self._layout_thumbs()

    def _layout_thumbs(self) -> None:
        s = self.theme.scale
        size = self.strip.height() - 12 * s
        gap = 8 * s
        self.thumb_rects = []
        if not self.model.paths or size <= 0:
            return
        total = len(self.model.paths) * (size + gap) - gap
        # Keep the current thumbnail visible.
        cur_x = self.model.index * (size + gap)
        if cur_x + self.strip_offset < 0:
            self.strip_offset = -cur_x
        elif cur_x + size + self.strip_offset > self.strip.width():
            self.strip_offset = self.strip.width() - cur_x - size
        self.strip_offset = min(0.0, max(self.strip_offset, min(0.0, self.strip.width() - total)))
        for i in range(len(self.model.paths)):
            self.thumb_rects.append(QRectF(self.strip.left() + self.strip_offset + i * (size + gap),
                                           self.strip.top() + 6 * s, size, size))

    # ------------------------------------------------------------ input
    def content_press(self, ev: PointerEvent) -> Grab | None:
        pt = QPointF(ev.x, ev.y)
        for i, r in enumerate(self.thumb_rects):
            if r.contains(pt) and self.strip.contains(pt):
                from airspace.interaction.events import CallbackGrab

                def release(e: PointerEvent, i=i) -> None:
                    if not e.cancelled and self.thumb_rects and self.thumb_rects[i].adjusted(-8, -8, 8, 8).contains(QPointF(e.x, e.y)):
                        if i != self.model.index:
                            self.model.select(i)
                            self.view.reset()

                return CallbackGrab(InteractionState.CLICK_PENDING, None, release)
        if self.viewer.contains(pt) and self.model.current:
            return _PanGrab(self, ev)
        return None

    def wheel(self, x: float, y: float, delta: float, modifiers: int = 0) -> bool:
        if self.controls.wheel(x, y, delta):
            return True
        if self.strip.contains(QPointF(x, y)):
            self.strip_offset += delta * 0.5
            self._layout_thumbs()
            return True
        if self.viewer.contains(QPointF(x, y)) and self.model.current:
            c = self.viewer.center()
            self.view.zoom_by(1.0015 ** delta, x - c.x(), y - c.y())
            return True
        return False

    def key(self, event: QKeyEvent) -> bool:
        k = event.key()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if k == Qt.Key.Key_I:
                QTimer.singleShot(0, self.import_images)
                return True
            return False
        actions = {
            Qt.Key.Key_Right: self.next, Qt.Key.Key_Left: self.prev,
            Qt.Key.Key_Plus: lambda: self.view.zoom_by(1.25), Qt.Key.Key_Equal: lambda: self.view.zoom_by(1.25),
            Qt.Key.Key_Minus: lambda: self.view.zoom_by(0.8), Qt.Key.Key_R: lambda: self.view.rotate_90(1),
            Qt.Key.Key_F: self.view.fit, Qt.Key.Key_0: self.view.reset, Qt.Key.Key_Delete: self.remove_current,
            Qt.Key.Key_Backspace: self.remove_current,
        }
        fn = actions.get(k)
        if fn:
            fn()
            return True
        return False

    def tick(self, dt: float) -> None:
        super().tick(dt)
        self._slide *= math.exp(-dt * 14)
        if abs(self._slide) < 0.001:
            self._slide = 0.0

    # ------------------------------------------------------------ state
    def save_state(self) -> dict:
        return {"paths": self.model.paths[:500], "index": self.model.index}

    def restore_state(self, state: dict) -> None:
        paths = [p for p in state.get("paths", []) if isinstance(p, str) and Path(p).exists()]
        self.model.paths = paths
        self.model.index = min(int(state.get("index", 0)), max(0, len(paths) - 1))

    def status_text(self) -> str:
        if not self.model.paths:
            return ""
        z = f" · {self.view.zoom * 100:.0f}%" if abs(self.view.zoom - 1) > 0.01 else ""
        return f"{self.model.index + 1} / {len(self.model.paths)}{z}"

    # ------------------------------------------------------------ paint
    def paint_content(self, p: QPainter, theme: Theme) -> None:
        v = self.viewer
        path = QPainterPath()
        path.addRoundedRect(v, theme.radius_small, theme.radius_small)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(QColor(8, 9, 11), 0.55))
        p.drawPath(path)
        cur = self.model.current
        if not cur:
            p.setPen(theme.text_secondary)
            p.setFont(theme.font(13, QFont.Weight.Medium))
            p.drawText(QRectF(v.left(), v.center().y() - 60, v.width(), 28), Qt.AlignmentFlag.AlignCenter, "No photos yet")
            p.setFont(theme.font(10))
            p.setPen(theme.text_tertiary)
            p.drawText(QRectF(v.left(), v.center().y() - 30, v.width(), 22), Qt.AlignmentFlag.AlignCenter,
                       "Import from your computer — files are opened read-only")
            return
        img = self.cache.get(cur, FULL) or self.cache.get(cur, THUMB)
        p.save()
        p.setClipPath(path)
        if img is None:
            p.setPen(theme.text_tertiary)
            p.setFont(theme.font(10))
            msg = "Couldn't read this image" if cur in self.cache.failed else "loading…"
            p.drawText(v, Qt.AlignmentFlag.AlignCenter, msg)
        else:
            iw, ih = img.width(), img.height()
            base = fit_scale(iw, ih, v.width() - 16, v.height() - 16, self.view.angle)
            scale = base * self.view.zoom
            p.translate(v.center().x() + self.view.pan_x + self._slide * v.width() * 0.25,
                        v.center().y() + self.view.pan_y)
            p.rotate(self.view.angle)
            p.scale(scale, scale)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.setOpacity(1.0 - min(0.6, abs(self._slide)))
            p.drawImage(QRectF(-iw / 2, -ih / 2, iw, ih), img)
        p.restore()
        # Thumbnail strip
        p.save()
        p.setClipRect(self.strip)
        for i, r in enumerate(self.thumb_rects):
            if r.right() < self.strip.left() or r.left() > self.strip.right():
                continue
            t = self.cache.get(self.model.paths[i], THUMB)
            tp = QPainterPath()
            tp.addRoundedRect(r, 8, 8)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.surface)
            p.drawPath(tp)
            if t is not None:
                p.save()
                p.setClipPath(tp)
                s = max(r.width() / t.width(), r.height() / t.height())
                tw, th = t.width() * s, t.height() * s
                p.drawImage(QRectF(r.center().x() - tw / 2, r.center().y() - th / 2, tw, th), t)
                p.restore()
            selected = i == self.model.index
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(theme.accent if selected else theme.border, 2 if selected else 1))
            p.drawPath(tp)
        p.restore()


# ------------------------------------------------------------- samples
def generate_sample_images(directory: Path, count: int = 6) -> list[Path]:
    """Create original procedural artwork so the gallery can be tried instantly."""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    out = []
    palettes = [((24, 32, 54), (196, 150, 120)), ((18, 44, 52), (150, 210, 200)), ((40, 22, 48), (230, 160, 190)),
                ((28, 30, 34), (220, 210, 190)), ((12, 26, 40), (120, 170, 240)), ((44, 30, 20), (240, 190, 110))]
    h, w = 900, 1400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(count):
        path = directory / f"airspace-sample-{i + 1}.png"
        out.append(path)
        if path.exists():
            continue
        rng = np.random.default_rng(100 + i)
        a, b = palettes[i % len(palettes)]
        field = np.zeros((h, w), np.float32)
        for _ in range(5):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            fx, fy = rng.uniform(0.002, 0.009, 2)
            field += np.sin((xx - cx) * fx + np.cos((yy - cy) * fy) * 2.0) * rng.uniform(0.4, 1.0)
        field = (field - field.min()) / (np.ptp(field) + 1e-6)
        bands = 0.5 + 0.5 * np.sin(field * np.pi * rng.integers(6, 14))
        t = (0.65 * field + 0.35 * bands)[..., None]
        img = (np.array(a, np.float32) * (1 - t) + np.array(b, np.float32) * t)
        vign = 1 - 0.45 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)[..., None] * 2
        img = np.clip(img * vign, 0, 255).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out
