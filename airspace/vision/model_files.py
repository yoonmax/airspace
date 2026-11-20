"""Model file locations and download helpers.

The MediaPipe models are published by Google under the Apache 2.0 license.
They are downloaded on request (never silently) from Google's official model
storage.
"""

from __future__ import annotations

import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from airspace.persistence.paths import models_dir


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    approx_size: str
    required: bool


MODELS = [
    ModelFile("hand_landmarker.task",
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
              "7.8 MB", True),
    ModelFile("selfie_segmenter.tflite",
              "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
              "0.25 MB", False),
]


def model_path(name: str) -> Path:
    return models_dir() / name


def missing_models(required_only: bool = False) -> list[ModelFile]:
    return [m for m in MODELS if (m.required or not required_only)
            and not (model_path(m.name).exists() and model_path(m.name).stat().st_size > 0)]


def _ssl_context() -> ssl.SSLContext:
    """python.org macOS builds ship without CA certs; prefer certifi when present."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_model(model: ModelFile, progress: Callable[[str], None] | None = None) -> Path:
    target = model_path(model.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if progress:
        progress(f"Downloading {model.name} ({model.approx_size})…")
    with urllib.request.urlopen(model.url, timeout=60, context=_ssl_context()) as response, open(tmp, "wb") as fh:
        while chunk := response.read(1 << 16):
            fh.write(chunk)
    tmp.replace(target)
    return target


def download_missing(progress: Callable[[str], None] | None = None) -> list[Path]:
    return [download_model(m, progress) for m in missing_models()]
