"""Filesystem locations used by airspace.

All user data stays on the local machine. The data directory can be
overridden with the ``AIRSPACE_DATA_DIR`` environment variable, which the test
suite uses to avoid touching real user settings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "airspace"


def package_root() -> Path:
    """Directory containing the ``airspace`` package (works when frozen too)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "airspace"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def assets_dir() -> Path:
    return package_root() / "assets"


def models_dir() -> Path:
    return assets_dir() / "models"


def data_dir() -> Path:
    override = os.environ.get("AIRSPACE_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    elif sys.platform.startswith("win"):
        path = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    else:
        path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    """Write text so a crash never leaves a half-written settings file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
