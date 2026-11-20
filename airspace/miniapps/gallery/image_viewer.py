"""Image loading (off the UI thread) and the viewer's view transform."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader


@dataclass
class ViewTransform:
    """Zoom / pan / rotation applied on top of 'fit to view'. Never alters the file."""

    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    rotation: int = 0  # degrees, multiples of 90 from buttons; free angle from gestures
    free_angle: float = 0.0

    MIN_ZOOM = 0.2
    MAX_ZOOM = 12.0

    def reset(self) -> None:
        self.zoom, self.pan_x, self.pan_y, self.rotation, self.free_angle = 1.0, 0.0, 0.0, 0, 0.0

    def fit(self) -> None:
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0

    def zoom_by(self, factor: float, cx: float = 0.0, cy: float = 0.0) -> None:
        """Zoom about a point given relative to the viewer centre."""
        new = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom * factor))
        f = new / self.zoom
        self.pan_x = cx - (cx - self.pan_x) * f
        self.pan_y = cy - (cy - self.pan_y) * f
        self.zoom = new

    def rotate_90(self, direction: int = 1) -> None:
        self.rotation = (self.rotation + 90 * direction) % 360

    @property
    def angle(self) -> float:
        return self.rotation + self.free_angle


def fit_scale(img_w: float, img_h: float, view_w: float, view_h: float, angle_deg: float) -> float:
    a = math.radians(angle_deg)
    bw = abs(img_w * math.cos(a)) + abs(img_h * math.sin(a))
    bh = abs(img_w * math.sin(a)) + abs(img_h * math.cos(a))
    if bw <= 0 or bh <= 0:
        return 1.0
    return min(view_w / bw, view_h / bh)


class _Signals(QObject):
    loaded = Signal(str, int, QImage)  # path, max_dim, image


class _LoadTask(QRunnable):
    def __init__(self, path: str, max_dim: int, signals: _Signals) -> None:
        super().__init__()
        self.path, self.max_dim, self.signals = path, max_dim, signals

    def run(self) -> None:
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)  # honour EXIF orientation
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > self.max_dim:
            scaled = size.scaled(QSize(self.max_dim, self.max_dim), Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(scaled)
        img = reader.read()
        if img.isNull():
            img = QImage()
        self.signals.loaded.emit(self.path, self.max_dim, img)


class ImageCache(QObject):
    """LRU cache of decoded images; decoding happens on a thread pool."""

    changed = Signal()

    def __init__(self, capacity: int = 24) -> None:
        super().__init__()
        self.capacity = capacity
        self._cache: OrderedDict[tuple[str, int], QImage] = OrderedDict()
        self._pending: set[tuple[str, int]] = set()
        self.failed: set[str] = set()
        self._signals = _Signals()
        self._signals.loaded.connect(self._on_loaded)
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(2)

    def get(self, path: str, max_dim: int) -> QImage | None:
        key = (path, max_dim)
        img = self._cache.get(key)
        if img is not None:
            self._cache.move_to_end(key)
            return img
        if key not in self._pending and path not in self.failed:
            self._pending.add(key)
            self.pool.start(_LoadTask(path, max_dim, self._signals))
        return None

    def _on_loaded(self, path: str, max_dim: int, img: QImage) -> None:
        self._pending.discard((path, max_dim))
        if img.isNull():
            self.failed.add(path)
        else:
            self._cache[(path, max_dim)] = img
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        self.changed.emit()

    def forget(self, path: str) -> None:
        for key in [k for k in self._cache if k[0] == path]:
            del self._cache[key]
        self.failed.discard(path)
