"""Guards for the PyInstaller build and platform permission helpers."""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parent.parent / "packaging" / "airspace.spec"


def _spec_excludes() -> list[str]:
    match = re.search(r"excludes=(\[[^\]]*\])", SPEC.read_text())
    assert match, "excludes list not found in spec"
    return ast.literal_eval(match.group(1))


def test_spec_does_not_exclude_modules_mediapipe_imports():
    # A packaged build once excluded matplotlib, which mediapipe's vision
    # package imports unconditionally, so hand tracking failed to start.
    vision = importlib.import_module("mediapipe.tasks.python.vision")
    assert vision.HandLandmarker

    loaded = set(sys.modules)
    clashes = [name for name in _spec_excludes() if name in loaded]
    assert not clashes, f"spec excludes modules the app imports: {clashes}"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS camera authorisation only")
def test_camera_authorization_status_is_queryable():
    from airspace.camera.permissions import AUTHORIZED, NOT_DETERMINED, authorization_status

    status = authorization_status()
    assert status is not None and NOT_DETERMINED <= status <= AUTHORIZED


@pytest.mark.skipif(not hasattr(sys, "set_lazy_imports_filter"), reason="Python without lazy imports (PEP 810)")
def test_stdlib_lazy_imports_resolve_with_pyside_loaded():
    # Python 3.15 + PySide6 6.12: lazy stdlib imports stayed unresolved placeholders,
    # and MediaPipe's ThreadPoolExecutor call failed. airspace/__init__ prevents that.
    import concurrent.futures

    importlib.import_module("airspace")
    importlib.import_module("PySide6.QtCore")

    assert isinstance(concurrent.futures.ThreadPoolExecutor, type)
    assert isinstance(concurrent.futures.ProcessPoolExecutor, type)
