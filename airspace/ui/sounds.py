"""Subtle interface sounds, synthesised locally at startup (no bundled audio)."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RATE = 44100


def _envelope(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    env = np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)
    return env


def _tone(freqs: list[float], dur: float, attack: float, decay: float, gain: float) -> np.ndarray:
    n = int(RATE * dur)
    t = np.arange(n) / RATE
    sig = sum(np.sin(2 * np.pi * f * t) * (0.6 ** i) for i, f in enumerate(freqs))
    return sig * _envelope(n, attack, decay) * gain


SOUNDS = {
    "click": lambda: _tone([1320, 2640], 0.06, 0.001, 0.012, 0.18),
    "release": lambda: _tone([990], 0.05, 0.001, 0.010, 0.10),
    "open": lambda: np.concatenate([_tone([660, 1320], 0.07, 0.003, 0.03, 0.14), _tone([990, 1980], 0.12, 0.003, 0.05, 0.14)]),
    "close": lambda: np.concatenate([_tone([990], 0.06, 0.003, 0.025, 0.12), _tone([660], 0.1, 0.003, 0.04, 0.12)]),
}


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


class SoundBank:
    def __init__(self, directory: Path) -> None:
        self.enabled = True
        self._effects: dict = {}
        try:
            from PySide6.QtCore import QCoreApplication, QUrl
            from PySide6.QtMultimedia import QSoundEffect

            directory.mkdir(parents=True, exist_ok=True)
            for name, fn in SOUNDS.items():
                path = directory / f"{name}.wav"
                if not path.exists():
                    write_wav(path, fn())
                # Parented to the application: Qt loads the sample asynchronously, and an
                # unparented effect freed by Python mid-load crashes inside Qt Multimedia.
                fx = QSoundEffect(QCoreApplication.instance())
                fx.setSource(QUrl.fromLocalFile(str(path)))
                fx.setVolume(0.35)
                self._effects[name] = fx
        except Exception as exc:  # noqa: BLE001 - sounds are optional
            log.info("Interface sounds unavailable: %s", exc)

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        fx = self._effects.get(name)
        if fx is not None:
            fx.play()
