"""Paint every mini-app and overlay through the real compositor into an image.

This catches painting errors; visual quality was reviewed with real screenshots.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QImage, QPainter

from airspace.miniapps.registry import APPS
from airspace.ui.calibration_wizard import CalibrationWizard
from airspace.ui.dialogs import ConfirmDialog
from airspace.workspace.compositing import FrameLayers
from airspace.workspace.rendering import Renderer, RenderInputs
from tests.test_interactions import Rig


def _render(rig, layers=None) -> QImage:
    img = QImage(1280, 720, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0))
    p = QPainter(img)
    ri = RenderInputs(rig.theme, rig.settings, rig.ws, rig.mgr, rig.shell.tracking_frame, rig.mapper,
                      layers or FrameLayers(), layers is not None, 1280, 720)
    Renderer().render(p, ri, 1 / 60)
    p.end()
    return img


def test_every_app_paints(qapp, data_dir):
    rig = Rig(data_dir)
    for spec in APPS:
        rig.wm.open(spec.key)
    for _ in range(30):
        rig.ws.tick(1 / 30)
    img = _render(rig)
    assert not img.isNull()
    for w in rig.wm.windows:
        assert w.app.width > 0 and w.app.height > 0


def test_overlays_and_debug_layers_paint(qapp, data_dir):
    rig = Rig(data_dir)
    t = rig.settings.tracking
    t.show_skeleton = t.show_labels = t.show_trail = t.show_zones = t.show_region = t.show_mask = True
    rig.ws.push_overlay(ConfirmDialog("Reset?", "Body", "Reset", lambda: None))
    rig.ws.push_overlay(CalibrationWizard([(0, "Camera")], 0, True, lambda i: None, lambda m: None,
                                          lambda c: None, lambda ok: None, lambda: rig.mgr.pointers["hand"]))
    rig.ws.open_launcher(640, 360)
    rig.move_to(500, 400)
    for _ in range(20):
        rig.ws.tick(1 / 30)
    from airspace.vision.vision_worker import VisionPacket

    frame = np.full((720, 1280, 3), 90, np.uint8)
    pkt = VisionPacket(1, frame, frame[::10, ::10].copy(), [], np.zeros((180, 320), np.float32), None, None,
                       0.0, 0.0, 1.0, 1.0, 2.0, True)
    layers = FrameLayers()
    layers.update(pkt)
    for bg in ("camera", "blurred", "dimmed", "solid"):
        rig.settings.interface.background = bg
        assert not _render(rig, layers).isNull()


def test_camera_denied_card_offers_privacy_settings(qapp, data_dir, monkeypatch):
    import sys

    from airspace.app.main_window import MainWindow
    from airspace.camera.capture import CameraState, CameraStatus
    from airspace.persistence.settings import SettingsStore

    monkeypatch.setattr(sys, "platform", "darwin")
    win = MainWindow(SettingsStore(), start_camera=False)
    try:
        win._permission_denied = True
        win._show_camera_error(CameraStatus(CameraState.FAILED, "denied", device_index=0))
        labels = [b.label for b in win._camera_card.buttons]
        assert "Open Privacy Settings" in labels and "Retry" in labels
        win._permission_denied = False
        win._show_camera_error(CameraStatus(CameraState.FAILED, "busy", device_index=0))
        assert "Next camera" in [b.label for b in win._camera_card.buttons]
    finally:
        win.close()


def _camera_layers(color=(240, 40, 40)):
    from airspace.vision.vision_worker import VisionPacket

    small = np.zeros((72, 128, 3), np.uint8)
    small[:] = color
    big = np.zeros((720, 1280, 3), np.uint8)
    big[:] = color
    layers = FrameLayers()
    layers.update(VisionPacket(1, big, small, [], None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, True))
    return layers


def test_window_background_lets_the_room_through(qapp, data_dir):
    rig = Rig(data_dir)
    win = rig.wm.open("calculator")
    for _ in range(40):
        rig.ws.tick(1 / 30)
    r = win.rect
    x, y = int(r.x + r.w / 2), int(r.y + 6)  # glass in the title bar, clear of text and buttons
    red = _render(rig, _camera_layers((240, 40, 40))).pixelColor(x, y)
    blue = _render(rig, _camera_layers((40, 40, 240))).pixelColor(x, y)
    # What is behind the window must visibly change the glass colour.
    assert red.red() - blue.red() > 40 and blue.blue() - red.blue() > 40


def test_hologram_layers_are_cached(qapp, data_dir):
    rig = Rig(data_dir)
    rig.wm.open("calculator")
    rig.wm.open("music")
    for _ in range(40):
        rig.ws.tick(1 / 30)
    layers = _camera_layers()
    renderer = Renderer()
    for _ in range(3):
        img = QImage(1280, 720, QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        renderer.render(p, RenderInputs(rig.theme, rig.settings, rig.ws, rig.mgr, None, rig.mapper, layers, True,
                                        1280, 720), 1 / 60)
        p.end()
        size = len(renderer._holo_cache)
    assert size == 4  # fill + edge per window, reused across frames
