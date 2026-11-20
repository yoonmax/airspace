"""monitor — live system and pipeline statistics.

CPU, memory and battery come from psutil; frame rates and latencies are
measured inside airspace. GPU utilisation is not shown because there is no
reliable cross-platform way to read it without elevated privileges.
"""

from __future__ import annotations

import os
import time
from collections import deque

import psutil
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen

from airspace.miniapps.base_app import MiniApp
from airspace.ui.controls import Segmented
from airspace.ui.styles import Theme, with_alpha

INTERVALS = [0.5, 1.0, 2.0]


class MonitorApp(MiniApp):
    app_type = "monitor"
    title = "monitor"
    icon = "chart"
    default_size = (560.0, 430.0)
    min_size = (420.0, 360.0)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.proc = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)  # prime the counter
        self.proc.cpu_percent(interval=None)
        self.interval_index = 1
        self._last = 0.0
        self.hist = {k: deque(maxlen=90) for k in ("cpu", "ram", "proc_cpu", "ui", "track", "latency")}
        self.values: dict[str, float | str | None] = {}
        self.interval_ctl = self.controls.add(Segmented(["0.5 s", "1 s", "2 s"], lambda: self.interval_index,
                                                        self._set_interval))
        self.tiles: list[QRectF] = []
        self._sample()

    def _set_interval(self, i: int) -> None:
        self.interval_index = i

    def _sample(self) -> None:
        vm = psutil.virtual_memory()
        perf = self.ctx.perf
        cpu = psutil.cpu_percent(interval=None)
        proc_cpu = self.proc.cpu_percent(interval=None) / max(1, psutil.cpu_count() or 1)
        rss = self.proc.memory_info().rss
        self.values = {
            "cpu": cpu, "ram": vm.percent, "ram_used": vm.used, "ram_total": vm.total, "rss": rss,
            "proc_cpu": proc_cpu, "ui": perf.ui_fps, "camera": perf.camera_fps, "track": perf.tracking_fps,
            "latency": perf.capture_to_display_ms, "hand_ms": perf.hand_ms, "paint_ms": perf.paint_ms,
            "cores": psutil.cpu_count(logical=True),
        }
        try:
            bat = psutil.sensors_battery()
        except Exception:  # noqa: BLE001
            bat = None
        self.values["battery"] = None if bat is None else (bat.percent, bat.power_plugged)
        for k, v in (("cpu", cpu), ("ram", vm.percent), ("proc_cpu", proc_cpu), ("ui", perf.ui_fps),
                     ("track", perf.tracking_fps), ("latency", perf.capture_to_display_ms)):
            self.hist[k].append(float(v))

    def tick(self, dt: float) -> None:
        super().tick(dt)
        now = time.monotonic()
        if now - self._last >= INTERVALS[self.interval_index]:
            self._last = now
            self._sample()

    def layout(self, w: float, h: float) -> None:
        s = self.theme.scale
        pad = 14 * s
        self.interval_ctl.rect = QRectF(w - pad - 170 * s, 10 * s, 170 * s, 28 * s)
        top = 48 * s
        cols, rows = 3, 2
        gap = 10 * s
        tw = (w - 2 * pad - gap * (cols - 1)) / cols
        th = (h - top - pad - gap * (rows - 1)) / rows
        self.tiles = [QRectF(pad + c * (tw + gap), top + r * (th + gap), tw, th) for r in range(rows) for c in range(cols)]

    def status_text(self) -> str:
        return f"every {INTERVALS[self.interval_index]:g} s"

    def paint_content(self, p: QPainter, theme: Theme) -> None:
        s = theme.scale
        v = self.values
        p.setFont(theme.font(10))
        p.setPen(theme.text_tertiary)
        bat = v.get("battery")
        bat_text = "" if bat is None else f" · battery {bat[0]:.0f}%{' charging' if bat[1] else ''}"
        p.drawText(QRectF(14 * s, 10 * s, self.width * 0.6, 28 * s), Qt.AlignmentFlag.AlignVCenter,
                   f"{v.get('cores', 0)} logical cores{bat_text}")
        gb = 1024 ** 3
        tiles = [
            ("cpu", f"{v.get('cpu', 0):.0f}%", "system cpu", self.hist["cpu"], 100.0, theme.accent),
            ("ram", f"{v.get('ram', 0):.0f}%", f"memory · {v.get('ram_used', 0) / gb:.1f} / {v.get('ram_total', 0) / gb:.0f} GB",
             self.hist["ram"], 100.0, theme.warm),
            ("proc", f"{v.get('rss', 0) / 1024 ** 2:.0f} MB", f"airspace · {v.get('proc_cpu', 0):.0f}% cpu",
             self.hist["proc_cpu"], 100.0, theme.success),
            ("ui", f"{v.get('ui', 0):.0f}", f"ui fps · paint {v.get('paint_ms', 0):.1f} ms", self.hist["ui"], 120.0, theme.accent),
            ("track", f"{v.get('track', 0):.0f}", f"tracking fps · camera {v.get('camera', 0):.0f}",
             self.hist["track"], 60.0, theme.success),
            ("lat", f"{v.get('latency', 0):.0f} ms", f"capture → display · infer {v.get('hand_ms', 0):.1f} ms",
             self.hist["latency"], 150.0, theme.warm),
        ]
        for rect, (key, big, label, hist, mx, color) in zip(self.tiles, tiles):
            self._tile(p, theme, rect, big, label, list(hist), mx, color)

    def _tile(self, p: QPainter, theme: Theme, r: QRectF, big: str, label: str, hist: list[float], mx: float,
              color: QColor) -> None:
        s = theme.scale
        path = QPainterPath()
        path.addRoundedRect(r, theme.radius_small, theme.radius_small)
        p.setPen(QPen(theme.border, 1))
        p.setBrush(theme.surface)
        p.drawPath(path)
        p.setPen(theme.text)
        p.setFont(theme.font(19, QFont.Weight.Light))
        p.drawText(QRectF(r.left() + 12 * s, r.top() + 8 * s, r.width() - 24 * s, 30 * s),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, big)
        p.setFont(theme.font(9))
        p.setPen(theme.text_tertiary)
        fm = p.fontMetrics()
        p.drawText(QRectF(r.left() + 12 * s, r.top() + 38 * s, r.width() - 24 * s, 16 * s),
                   Qt.AlignmentFlag.AlignLeft, fm.elidedText(label, Qt.TextElideMode.ElideRight, int(r.width() - 24 * s)))
        g = QRectF(r.left() + 8 * s, r.top() + 60 * s, r.width() - 16 * s, r.height() - 68 * s)
        if len(hist) < 2 or g.height() < 8:
            return
        top = max(mx, max(hist) * 1.1)
        n = self.hist["cpu"].maxlen or 90
        step = g.width() / (n - 1)
        x0 = g.right() - step * (len(hist) - 1)
        line = QPainterPath()
        for i, val in enumerate(hist):
            pt = QPointF(x0 + i * step, g.bottom() - g.height() * min(1.0, val / top))
            line.moveTo(pt) if i == 0 else line.lineTo(pt)
        area = QPainterPath(line)
        area.lineTo(QPointF(g.right(), g.bottom()))
        area.lineTo(QPointF(x0, g.bottom()))
        area.closeSubpath()
        grad = QLinearGradient(g.topLeft(), g.bottomLeft())
        grad.setColorAt(0.0, with_alpha(color, 0.28))
        grad.setColorAt(1.0, with_alpha(color, 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawPath(area)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, 1.6))
        p.drawPath(line)
