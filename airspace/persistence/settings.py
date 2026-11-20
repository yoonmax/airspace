"""Typed, validated, persistent application settings.

Settings are grouped into dataclass sections. Loading is tolerant: unknown
keys are ignored and invalid values fall back to defaults, so a corrupted or
older settings file never prevents airspace from starting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from airspace.persistence.paths import atomic_write_text, data_dir

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

CAPTURE_PROFILES: dict[str, tuple[int, int, int]] = {
    # name: (width, height, fps)
    "performance": (640, 480, 30),
    "balanced": (1280, 720, 30),
    "quality": (1920, 1080, 30),
}

OCCLUSION_MODES = ("balanced", "cinematic", "clear")
BACKGROUND_MODES = ("camera", "blurred", "dimmed", "solid")
SMOOTHING_PRESETS = ("responsive", "balanced", "steady")


@dataclass
class CameraSettings:
    device_index: int = 0
    profile: str = "balanced"
    mirror: bool = True
    enabled: bool = True


@dataclass
class TrackingSettings:
    max_hands: int = 2
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.5
    smoothing: str = "balanced"
    pointer_reach: float = 1.0  # 1.0 = cursor sits exactly on the fingertip; >1 amplifies motion
    pinch_enter: float = 0.30  # normalised thumb-index distance / palm scale
    pinch_exit: float = 0.45
    pinch_hold_ms: int = 260
    palm_hold_ms: int = 380
    dominant_hand: str = "auto"  # auto | left | right
    precision_mode: bool = False
    show_skeleton: bool = False
    show_labels: bool = False
    show_trail: bool = False
    show_zones: bool = False
    show_mask: bool = False
    show_region: bool = False


@dataclass
class InterfaceSettings:
    occlusion: str = "balanced"
    background: str = "camera"
    transparency: float = 0.55  # glass tint opacity; lower = more of the room and your hands show through
    window_scale: float = 1.0
    animations: bool = True
    cursor_size: float = 1.0
    sounds: bool = True


@dataclass
class PerformanceSettings:
    segmentation_interval: int = 2  # run segmentation every N frames; 0 = off
    render_quality: str = "high"  # high | low
    fps_target: int = 60
    renderer: str = "opengl"  # opengl | raster


@dataclass
class AccessibilitySettings:
    reduced_motion: bool = False
    large_buttons: bool = False
    high_contrast: bool = False
    mouse_fallback: bool = True
    dwell_click: bool = False  # hover to click alternative to pinching
    dwell_ms: int = 900


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    interface: InterfaceSettings = field(default_factory=InterfaceSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    accessibility: AccessibilitySettings = field(default_factory=AccessibilitySettings)
    calibrated: bool = False
    first_run_complete: bool = False

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        settings = cls()
        if not isinstance(data, dict):
            return settings
        for section_field in fields(cls):
            raw = data.get(section_field.name)
            current = getattr(settings, section_field.name)
            if hasattr(current, "__dataclass_fields__"):
                if isinstance(raw, dict):
                    _merge_section(current, raw)
            elif raw is not None and isinstance(raw, type(current)):
                setattr(settings, section_field.name, raw)
        schema = data.get("schema", 1)
        if not isinstance(schema, int) or schema < 2:
            # Schema 1 defaulted reach to 1.15, which put the cursor visibly off the
            # fingertip away from the screen centre. Start everyone back on the tip.
            settings.tracking.pointer_reach = 1.0
            if abs(settings.interface.transparency - 0.80) < 1e-6:  # old default → new see-through default
                settings.interface.transparency = 0.55
        settings.validate()
        return settings

    def validate(self) -> None:
        """Clamp and correct values so the rest of the app can trust them."""
        c, t, i, p = self.camera, self.tracking, self.interface, self.performance
        if c.profile not in CAPTURE_PROFILES:
            c.profile = "balanced"
        c.device_index = max(0, min(int(c.device_index), 16))
        t.max_hands = max(1, min(int(t.max_hands), 2))
        t.detection_confidence = _clamp(t.detection_confidence, 0.1, 0.95)
        t.tracking_confidence = _clamp(t.tracking_confidence, 0.1, 0.95)
        if t.smoothing not in SMOOTHING_PRESETS:
            t.smoothing = "balanced"
        t.pointer_reach = _clamp(t.pointer_reach, 1.0, 1.8)
        t.pinch_enter = _clamp(t.pinch_enter, 0.08, 0.8)
        t.pinch_exit = _clamp(t.pinch_exit, t.pinch_enter + 0.04, 1.2)
        t.pinch_hold_ms = int(_clamp(t.pinch_hold_ms, 80, 1200))
        t.palm_hold_ms = int(_clamp(t.palm_hold_ms, 150, 1500))
        if t.dominant_hand not in ("auto", "left", "right"):
            t.dominant_hand = "auto"
        if i.occlusion not in OCCLUSION_MODES:
            i.occlusion = "balanced"
        if i.background not in BACKGROUND_MODES:
            i.background = "camera"
        i.transparency = _clamp(i.transparency, 0.3, 0.95)
        i.window_scale = _clamp(i.window_scale, 0.75, 1.5)
        i.cursor_size = _clamp(i.cursor_size, 0.6, 2.0)
        p.segmentation_interval = int(_clamp(p.segmentation_interval, 0, 6))
        if p.render_quality not in ("high", "low"):
            p.render_quality = "high"
        p.fps_target = int(_clamp(p.fps_target, 24, 120))
        if p.renderer not in ("opengl", "raster"):
            p.renderer = "opengl"
        a = self.accessibility
        a.dwell_ms = int(_clamp(a.dwell_ms, 400, 2500))


def _clamp(value: Any, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _merge_section(section: Any, raw: dict[str, Any]) -> None:
    for f in fields(section):
        if f.name not in raw:
            continue
        value = raw[f.name]
        default = getattr(section, f.name)
        # bool is a subclass of int, so check it explicitly first.
        if isinstance(default, bool):
            if isinstance(value, bool):
                setattr(section, f.name, value)
        elif isinstance(default, (int, float)) and not isinstance(value, bool):
            if isinstance(value, (int, float)):
                setattr(section, f.name, type(default)(value))
        elif isinstance(default, str) and isinstance(value, str):
            setattr(section, f.name, value)


class SettingsStore:
    """Loads and saves :class:`Settings` as JSON in the user data directory."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "settings.json")

    def load(self) -> Settings:
        if not self.path.exists():
            return Settings()
        try:
            return Settings.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read settings (%s); using defaults", exc)
            return Settings()

    def save(self, settings: Settings) -> None:
        settings.validate()
        atomic_write_text(self.path, json.dumps(settings.to_dict(), indent=2))

    def clear(self) -> Settings:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete settings: %s", exc)
        return Settings()
