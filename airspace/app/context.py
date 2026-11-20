"""Services exposed to mini-apps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from airspace.persistence.settings import Settings
from airspace.ui.styles import Theme

if TYPE_CHECKING:
    from airspace.vision.tracking import TrackingFrame
    from airspace.vision.vision_worker import VisionPacket


@dataclass
class PerfStats:
    """Measured (never estimated) runtime statistics."""

    ui_fps: float = 0.0
    camera_fps: float = 0.0
    tracking_fps: float = 0.0
    hand_ms: float = 0.0
    seg_ms: float = 0.0
    vision_ms: float = 0.0
    capture_to_display_ms: float = 0.0  # capture timestamp → frame painted
    paint_ms: float = 0.0
    gesture_latency_ms: float | None = None  # capture of the pinch frame → interaction dispatched
    dropped_frames: int = 0
    frame_size: tuple[int, int] = (0, 0)
    history: dict[str, deque] = field(default_factory=lambda: {
        k: deque(maxlen=120) for k in ("ui_fps", "camera_fps", "tracking_fps", "latency", "cpu", "ram")
    })


class AppContext:
    def __init__(self, settings: Settings, theme_fn: Callable[[], Theme], shell: Any, data_dir: Path) -> None:
        self.settings = settings
        self._theme_fn = theme_fn
        self.shell = shell  # the main window controller: exposes workspace-level actions
        self.data_dir = data_dir
        self.perf = PerfStats()

    @property
    def theme(self) -> Theme:
        return self._theme_fn()

    def toast(self, text: str) -> None:
        self.shell.toast(text)

    def tracking(self) -> "TrackingFrame | None":
        return self.shell.tracking_frame

    def packet(self) -> "VisionPacket | None":
        return self.shell.latest_packet

    def open_files(self, title: str, name_filter: str, multiple: bool = True) -> list[str]:
        return self.shell.open_files(title, name_filter, multiple)

    def save_file(self, title: str, default_name: str, name_filter: str) -> str | None:
        return self.shell.save_file(title, default_name, name_filter)

    def play(self, sound: str) -> None:
        self.shell.play_sound(sound)
