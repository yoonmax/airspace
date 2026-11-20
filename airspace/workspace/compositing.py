"""Conversion of vision outputs into paintable layers.

Layers (bottom → top):
  1. background  — mirrored camera frame, or a blurred/dimmed/solid variant
  2. windows     — frosted glass sampling the blurred frame behind them
  3. foreground  — camera pixels of the hands/person, alpha-masked, drawn
                   *over* the windows so the user appears in front of them
  4. launcher, HUD, overlays, cursor — always on top for readability
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage

from airspace.vision.coordinate_mapper import CoordinateMapper
from airspace.vision.vision_worker import VisionPacket


def rgb_to_qimage(arr: np.ndarray) -> QImage:
    """Wrap an (H, W, 3) uint8 RGB array. The returned image shares memory with
    ``arr`` — callers must keep the array alive while the image is in use."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGB888)


def rgba_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGBA8888)


def gray_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_Grayscale8)


class FrameLayers:
    """Caches QImage wrappers for the current packet (one conversion per frame)."""

    def __init__(self) -> None:
        self.packet: VisionPacket | None = None
        self.frame: QImage | None = None
        self.blur: QImage | None = None
        self.foreground: QImage | None = None
        self.mask_debug: QImage | None = None
        self._keep: list[np.ndarray] = []

    def update(self, packet: VisionPacket | None) -> None:
        if packet is self.packet:
            return
        self.packet = packet
        self._keep = []
        self.frame = self.blur = self.foreground = self.mask_debug = None
        if packet is None:
            return
        self._keep.append(packet.frame_rgb)
        self.frame = rgb_to_qimage(packet.frame_rgb)
        self._keep.append(packet.blur_small)
        self.blur = rgb_to_qimage(packet.blur_small)
        if packet.foreground is not None:
            self._keep.append(packet.foreground.rgba)
            self.foreground = rgba_to_qimage(packet.foreground.rgba)

    def mask_image(self) -> QImage | None:
        """Lazily build a debug visualisation of the person/occlusion mask."""
        if self.mask_debug is not None or self.packet is None:
            return self.mask_debug
        m = self.packet.person_mask if self.packet.person_mask is not None else self.packet.occlusion_alpha
        if m is None:
            return None
        g = np.ascontiguousarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
        self._keep.append(g)
        self.mask_debug = gray_to_qimage(g)
        return self.mask_debug

    def foreground_rect(self, mapper: CoordinateMapper) -> QRectF | None:
        fg = self.packet.foreground if self.packet else None
        if fg is None:
            return None
        r, s = mapper.image_rect, mapper.scale
        h, w = fg.rgba.shape[:2]
        return QRectF(r.x + fg.x * s, r.y + fg.y * s, w * s, h * s)

    def image_rect(self, mapper: CoordinateMapper) -> QRectF:
        r = mapper.image_rect
        return QRectF(r.x, r.y, r.w, r.h)
