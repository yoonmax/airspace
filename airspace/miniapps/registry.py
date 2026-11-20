"""Registry of mini-apps available in the launcher."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSpec:
    key: str
    label: str
    icon: str
    module: str
    cls: str
    in_launcher: bool = True


APPS: list[AppSpec] = [
    AppSpec("music", "music", "music", "airspace.miniapps.music.player", "MusicApp"),
    AppSpec("canvas", "canvas", "brush", "airspace.miniapps.canvas.drawing_canvas", "CanvasApp"),
    AppSpec("gallery", "gallery", "image", "airspace.miniapps.gallery.gallery", "GalleryApp"),
    AppSpec("objects", "objects", "cube", "airspace.miniapps.objects.viewer", "ObjectsApp"),
    AppSpec("monitor", "monitor", "chart", "airspace.miniapps.monitor.system_monitor", "MonitorApp"),
    AppSpec("calculator", "calculator", "calculator", "airspace.miniapps.calculator.calculator", "CalculatorApp"),
    AppSpec("tracking", "tracking", "hand", "airspace.miniapps.tracking.inspector", "TrackingInspectorApp"),
    AppSpec("settings", "settings", "settings", "airspace.miniapps.settings.settings_app", "SettingsApp"),
]

BY_KEY = {a.key: a for a in APPS}


def load_class(key: str):
    spec = BY_KEY[key]
    module = importlib.import_module(spec.module)
    return getattr(module, spec.cls)
