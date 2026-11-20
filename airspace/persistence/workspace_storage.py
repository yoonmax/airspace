"""Persistence of the window layout and per-app state.

Window positions are stored *normalised* to the workspace size so a layout
saved in a small window restores sensibly in fullscreen, while window sizes are
stored in logical pixels (content should not grow just because the screen did).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from airspace.persistence.paths import atomic_write_text, data_dir

log = logging.getLogger(__name__)


@dataclass
class WindowRecord:
    app_type: str
    nx: float  # normalised x of the window's left edge
    ny: float
    width: float
    height: float
    z: int = 0
    minimized: bool = False
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkspaceLayout:
    windows: list[WindowRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "windows": [asdict(w) for w in self.windows]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceLayout":
        layout = cls()
        for raw in data.get("windows", []) if isinstance(data, dict) else []:
            try:
                layout.windows.append(
                    WindowRecord(
                        app_type=str(raw["app_type"]),
                        nx=min(max(float(raw["nx"]), -0.5), 1.0),
                        ny=min(max(float(raw["ny"]), -0.5), 1.0),
                        width=max(float(raw["width"]), 50.0),
                        height=max(float(raw["height"]), 50.0),
                        z=int(raw.get("z", 0)),
                        minimized=bool(raw.get("minimized", False)),
                        state=dict(raw.get("state", {}) or {}),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping invalid window record %r: %s", raw, exc)
        return layout


class WorkspaceStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "workspace.json")

    def load(self) -> WorkspaceLayout | None:
        if not self.path.exists():
            return None
        try:
            return WorkspaceLayout.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read workspace layout: %s", exc)
            return None

    def save(self, layout: WorkspaceLayout) -> None:
        atomic_write_text(self.path, json.dumps(layout.to_dict(), indent=2))

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
