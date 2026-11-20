"""objects — an interactive 3D viewport.

Pinch and move to rotate (with gentle inertia), add the second hand to zoom
and roll, right-drag/shift-drag to pan with the mouse, scroll to zoom.
Transient tracking glitches are filtered: per-event rotation is clamped and
implausible jumps are ignored, so an object never spins out of control.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPainterPath, QPen

from airspace.interaction.events import Grab, InteractionState, PointerEvent, TwoHandInfo
from airspace.miniapps.base_app import MiniApp
from airspace.miniapps.objects.models import BUILTIN, Mesh, load_model_file
from airspace.miniapps.objects.renderer import LIGHTING, MATERIALS, orthonormalize, rotation
from airspace.ui.controls import Button, Segmented
from airspace.ui.styles import Theme, with_alpha

log = logging.getLogger(__name__)

MODEL_KEYS = list(BUILTIN)
MODEL_LABELS = {"knot": "knot", "cube": "cube", "sphere": "sphere", "torus": "torus", "sculpture": "stone", "globe": "globe"}
LIGHT_KEYS = list(LIGHTING)
MODEL_FILTER = "3D models (*.glb *.gltf *.obj *.stl *.ply *.off)"
MAX_STEP = 0.35  # radians of rotation allowed per pointer event
GLITCH_PX = 170.0  # pointer jumps larger than this are treated as tracking glitches
ROT_GAIN = 0.0095  # radians per pixel


class TransformState:
    """Object transform with inertia (Qt-free; unit tested)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rot = rotation(np.array([0.0, 1.0, 0.0]), 0.55) @ rotation(np.array([1.0, 0.0, 0.0]), 0.28)
        self.distance = 3.7
        self.pan = [0.0, 0.0]
        self.velocity = np.zeros(2)  # yaw, pitch rad/s

    def rotate(self, yaw: float, pitch: float, roll: float = 0.0) -> None:
        yaw = max(-MAX_STEP, min(MAX_STEP, yaw))
        pitch = max(-MAX_STEP, min(MAX_STEP, pitch))
        r = rotation(np.array([0.0, 1.0, 0.0]), yaw) @ rotation(np.array([1.0, 0.0, 0.0]), pitch)
        if roll:
            r = rotation(np.array([0.0, 0.0, 1.0]), roll) @ r
        self.rot = orthonormalize(r @ self.rot)

    def zoom(self, factor: float) -> None:
        self.distance = max(1.8, min(10.0, self.distance / factor))

    def step_inertia(self, dt: float) -> bool:
        if not np.any(np.abs(self.velocity) > 0.02):
            self.velocity[:] = 0
            return False
        self.rotate(self.velocity[0] * dt, self.velocity[1] * dt)
        self.velocity *= math.exp(-2.6 * dt)
        return True


class _RotateGrab(Grab):
    kind = InteractionState.MANIPULATING_OBJECT
    supports_two_hand = True

    def __init__(self, app: "ObjectsApp", ev: PointerEvent, pan: bool = False) -> None:
        self.app = app
        self.pan = pan
        self.last = (ev.x, ev.y, ev.t)
        self.vel = np.zeros(2)
        self._two: tuple[float, np.ndarray] | None = None
        app.tf.velocity[:] = 0
        app.dragging = True

    def move(self, ev: PointerEvent) -> None:
        if self._two is not None:
            return
        lx, ly, lt = self.last
        dx, dy = ev.x - lx, ev.y - ly
        self.last = (ev.x, ev.y, ev.t)
        if math.hypot(dx, dy) > GLITCH_PX:  # tracking glitch: ignore and re-anchor
            self.vel[:] = 0
            return
        tf = self.app.tf
        if self.pan:
            k = tf.distance / max(self.app.viewport.height(), 1.0) * 0.55
            tf.pan[0] += dx * k
            tf.pan[1] -= dy * k
        else:
            yaw, pitch = dx * ROT_GAIN, dy * ROT_GAIN
            tf.rotate(yaw, pitch)
            dt = max(ev.t - lt, 1e-3)
            inst = np.array([yaw, pitch]) / dt
            self.vel = 0.6 * self.vel + 0.4 * inst
        self.app.dirty = True

    def release(self, ev: PointerEvent) -> None:
        self.app.dragging = False
        if not ev.cancelled and not self.pan and ev.t - self.last[2] < 0.12:
            self.app.tf.velocity = np.clip(self.vel, -5.0, 5.0)

    def cancel(self) -> None:
        self.app.dragging = False

    def two_hand_begin(self, info: TwoHandInfo) -> bool:
        tf = self.app.tf
        self._two = (tf.distance, tf.rot.copy())
        return True

    def two_hand_update(self, start: TwoHandInfo, now: TwoHandInfo) -> None:
        if self._two is None:
            return
        d0, rot0 = self._two
        tf = self.app.tf
        factor = now.distance / max(start.distance, 1.0)
        tf.distance = max(1.8, min(10.0, d0 / factor))
        roll = (math.degrees(now.angle - start.angle) + 180) % 360 - 180
        tf.rot = orthonormalize(rotation(np.array([0.0, 0.0, 1.0]), -math.radians(roll)) @ rot0)
        self.app.dirty = True

    def two_hand_end(self) -> None:
        self._two = None
        self.vel[:] = 0


class ObjectsApp(MiniApp):
    app_type = "objects"
    title = "objects"
    icon = "cube"
    default_size = (560.0, 520.0)
    min_size = (440.0, 400.0)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.tf = TransformState()
        self.model_key = "knot"
        self.material_index = 0
        self.light_index = 0
        self.wireframe = False
        self.auto_rotate = True
        self.dragging = False
        self.dirty = True
        self.viewport = QRectF()
        self.image: QImage | None = None
        self._image_data: np.ndarray | None = None
        self.renderer = None
        self.render_error = ""
        self.mesh: Mesh | None = None
        self.imported_name = ""
        self._pending: tuple[str, Mesh | None, str] | None = None
        self._loading = False
        self._idle_since = time.monotonic()
        self.render_ms = 0.0
        c = self.controls
        self.model_ctl = c.add(Segmented([MODEL_LABELS[k] for k in MODEL_KEYS], self._model_index, self._select_model,
                                         font_size=10))
        self.material_ctl = c.add(Segmented([m.name for m in MATERIALS], lambda: self.material_index,
                                            self._select_material, font_size=10))
        self.btn_light = c.add(Button(icon="light", variant="icon", on_click=self._cycle_light, tooltip="Lighting"))
        self.btn_wire = c.add(Button(icon="wireframe", variant="icon", on_click=self._toggle_wire,
                                     checked=lambda: self.wireframe, tooltip="Wireframe"))
        self.btn_spin = c.add(Button(icon="rotate", variant="icon", on_click=self._toggle_spin,
                                     checked=lambda: self.auto_rotate, tooltip="Auto-rotate"))
        self.btn_fit = c.add(Button(icon="fit", variant="icon", on_click=self._fit, tooltip="Fit model"))
        self.btn_reset = c.add(Button(icon="reset", variant="icon", on_click=self._reset, tooltip="Reset orientation"))
        self.btn_import = c.add(Button(icon="folder", variant="icon", on_click=lambda: QTimer.singleShot(0, self.import_model),
                                       tooltip="Import GLB / GLTF / OBJ / STL"))
        self._init_renderer()
        self._load_builtin(self.model_key)

    # -------------------------------------------------------------- setup
    def _init_renderer(self) -> None:
        try:
            from airspace.miniapps.objects.renderer import ObjectRenderer

            self.renderer = ObjectRenderer()
        except Exception as exc:  # noqa: BLE001 - GL may be unavailable (e.g. remote sessions)
            log.exception("3D renderer unavailable")
            self.render_error = f"3D rendering is unavailable on this system: {exc}"

    def _load_builtin(self, key: str) -> None:
        self.model_key = key
        self.imported_name = ""
        self._set_mesh(BUILTIN[key]().normalized())
        if key == "globe":
            self.material_index = len(MATERIALS) - 1  # natural colours
        elif MATERIALS[self.material_index].name == "natural":
            self.material_index = 0

    def _set_mesh(self, mesh: Mesh) -> None:
        self.mesh = mesh
        if self.renderer is not None:
            try:
                self.renderer.set_mesh(mesh)
            except Exception as exc:  # noqa: BLE001
                log.exception("Mesh upload failed")
                self.render_error = str(exc)
        self.dirty = True

    # ------------------------------------------------------------ actions
    def _model_index(self) -> int:
        return MODEL_KEYS.index(self.model_key) if not self.imported_name else -1

    def _select_model(self, i: int) -> None:
        self._load_builtin(MODEL_KEYS[i])

    def _select_material(self, i: int) -> None:
        self.material_index = i
        self.dirty = True

    def _cycle_light(self) -> None:
        self.light_index = (self.light_index + 1) % len(LIGHT_KEYS)
        self.ctx.toast(f"Lighting: {LIGHT_KEYS[self.light_index]}")
        self.dirty = True

    def _toggle_wire(self) -> None:
        self.wireframe = not self.wireframe
        self.dirty = True

    def _toggle_spin(self) -> None:
        self.auto_rotate = not self.auto_rotate

    def _fit(self) -> None:
        self.tf.distance = 3.7
        self.tf.pan = [0.0, 0.0]
        self.dirty = True

    def _reset(self) -> None:
        self.tf.reset()
        self.dirty = True

    def import_model(self) -> None:
        files = self.ctx.open_files("Import 3D model", MODEL_FILTER, multiple=False)
        if not files or self._loading:
            return
        path = files[0]
        self._loading = True
        self.ctx.toast("Loading model…")

        def work() -> None:
            try:
                self._pending = (path, load_model_file(path), "")
            except Exception as exc:  # noqa: BLE001
                self._pending = (path, None, str(exc))

        threading.Thread(target=work, daemon=True, name="airspace-model-import").start()

    # --------------------------------------------------------------- frame
    def tick(self, dt: float) -> None:
        super().tick(dt)
        if self._pending is not None:
            path, mesh, err = self._pending
            self._pending = None
            self._loading = False
            if mesh is None:
                self.ctx.toast(f"Couldn't load model: {err}")
            else:
                self._set_mesh(mesh)
                self.imported_name = mesh.name
                if MATERIALS[self.material_index].name == "natural" and not mesh.has_colors:
                    self.material_index = 0
                self.tf.reset()
                self.ctx.toast(f"Loaded {mesh.name} · {mesh.triangle_count:,} triangles")
        if self.tf.step_inertia(dt):
            self.dirty = True
            self._idle_since = time.monotonic()
        elif self.dragging:
            self._idle_since = time.monotonic()
        elif self.auto_rotate and time.monotonic() - self._idle_since > 1.5:
            self.tf.rotate(0.22 * dt, 0.0)
            self.dirty = True
        if self.dirty:
            self._render()

    def _render(self) -> None:
        if self.renderer is None or self.viewport.width() < 8 or self.viewport.height() < 8:
            return
        dpr = 2.0
        w = min(int(self.viewport.width() * dpr), 1600)
        h = int(w * self.viewport.height() / self.viewport.width())
        acc = self.theme.accent
        t0 = time.perf_counter()
        try:
            data = self.renderer.render(w, h, self.tf.rot, self.tf.distance, tuple(self.tf.pan), 1.0,
                                        MATERIALS[self.material_index], LIGHT_KEYS[self.light_index], self.wireframe,
                                        (acc.redF(), acc.greenF(), acc.blueF()))
        except Exception as exc:  # noqa: BLE001
            log.exception("3D render failed")
            self.render_error = f"Rendering failed: {exc}"
            self.renderer = None
            return
        self.render_ms = (time.perf_counter() - t0) * 1000
        self._image_data = data
        self.image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0],
                            QImage.Format.Format_RGBA8888_Premultiplied)
        self.dirty = False

    # -------------------------------------------------------------- layout
    def layout(self, w: float, h: float) -> None:
        s = self.theme.scale
        pad = 12 * s
        rowh = 30 * s
        self.model_ctl.rect = QRectF(pad, 10 * s, w - 2 * pad, rowh)
        y2 = 10 * s + rowh + 8 * s
        btn = 30 * s
        right = w - pad
        for b in (self.btn_import, self.btn_reset, self.btn_fit, self.btn_spin, self.btn_wire, self.btn_light):
            right -= btn
            b.rect = QRectF(right, y2, btn, btn)
            right -= 2
        self.material_ctl.rect = QRectF(pad, y2, right - pad - 8 * s, rowh)
        top = y2 + rowh + 8 * s
        self.viewport = QRectF(pad, top, w - 2 * pad, h - top - pad)
        self.dirty = True

    # --------------------------------------------------------------- input
    def content_press(self, ev: PointerEvent) -> Grab | None:
        if not self.viewport.contains(QPointF(ev.x, ev.y)):
            return None
        pan = ev.button == 2 or bool(ev.modifiers & Qt.KeyboardModifier.ShiftModifier.value)
        return _RotateGrab(self, ev, pan=pan)

    def wheel(self, x: float, y: float, delta: float, modifiers: int = 0) -> bool:
        if self.controls.wheel(x, y, delta):
            return True
        if self.viewport.contains(QPointF(x, y)):
            self.tf.zoom(1.0012 ** delta)
            self.dirty = True
            return True
        return False

    def key(self, event: QKeyEvent) -> bool:
        k = event.key()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        step = 0.12
        if k == Qt.Key.Key_Left:
            self.tf.rotate(-step, 0)
        elif k == Qt.Key.Key_Right:
            self.tf.rotate(step, 0)
        elif k == Qt.Key.Key_Up:
            self.tf.rotate(0, -step)
        elif k == Qt.Key.Key_Down:
            self.tf.rotate(0, step)
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.tf.zoom(1.15)
        elif k == Qt.Key.Key_Minus:
            self.tf.zoom(1 / 1.15)
        elif k == Qt.Key.Key_W:
            self._toggle_wire()
        elif k == Qt.Key.Key_L:
            self._cycle_light()
        elif k == Qt.Key.Key_R:
            self._reset()
        elif k == Qt.Key.Key_Space:
            self._toggle_spin()
        else:
            return False
        self.dirty = True
        return True

    # --------------------------------------------------------------- state
    def save_state(self) -> dict:
        return {"model": self.model_key, "material": self.material_index, "light": self.light_index,
                "wire": self.wireframe, "spin": self.auto_rotate}

    def restore_state(self, state: dict) -> None:
        key = state.get("model", "knot")
        if key in BUILTIN:
            self._load_builtin(key)
        self.material_index = int(state.get("material", self.material_index)) % len(MATERIALS)
        self.light_index = int(state.get("light", 0)) % len(LIGHT_KEYS)
        self.wireframe = bool(state.get("wire", False))
        self.auto_rotate = bool(state.get("spin", True))
        self.dirty = True

    def on_close(self) -> None:
        if self.renderer is not None:
            self.renderer.release()
            self.renderer = None

    def status_text(self) -> str:
        if self.mesh is None:
            return ""
        name = self.imported_name or MODEL_LABELS.get(self.model_key, self.model_key)
        return f"{name} · {self.mesh.triangle_count:,} tris"

    # --------------------------------------------------------------- paint
    def paint_content(self, p: QPainter, theme: Theme) -> None:
        v = self.viewport
        path = QPainterPath()
        path.addRoundedRect(v, theme.radius_small, theme.radius_small)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(QColor(10, 11, 14), 0.38))
        p.drawPath(path)
        if self.render_error:
            p.setPen(theme.text_secondary)
            p.setFont(theme.font(10.5))
            p.drawText(v.adjusted(20, 20, -20, -20), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.render_error)
            return
        if self.image is not None:
            p.save()
            p.setClipPath(path)
            p.drawImage(v, self.image)
            p.restore()
        if self.dragging:
            p.setPen(QPen(with_alpha(theme.accent, 0.5), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        p.setFont(theme.font(9))
        p.setPen(theme.text_tertiary)
        hint = "pinch + move to rotate · two hands to zoom & roll" if not self.dragging else \
            f"{self.render_ms:.1f} ms / frame"
        p.drawText(QRectF(v.left() + 12, v.bottom() - 24, v.width() - 24, 18),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, hint)
        if self._loading:
            p.setFont(theme.font(11, QFont.Weight.Medium))
            p.setPen(theme.text)
            p.drawText(v, Qt.AlignmentFlag.AlignCenter, "loading model…")
