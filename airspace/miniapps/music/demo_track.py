"""Synthesise an original demo track so the player can be tried without
bundling any copyrighted audio. Generated locally on request."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

RATE = 44100


def _note(freq: float, dur: float, kind: str = "pad") -> np.ndarray:
    n = int(RATE * dur)
    t = np.arange(n) / RATE
    if kind == "pad":
        sig = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2.002 * t)
               + 0.18 * np.sin(2 * np.pi * freq * 0.998 * 3 * t))
        env = np.minimum(1, t / 0.6) * np.minimum(1, (dur - t) / 0.8)
    else:  # pluck
        sig = np.sin(2 * np.pi * freq * t) * np.exp(-t * 5) + 0.3 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t * 9)
        env = np.minimum(1, t / 0.005)
    return sig * np.clip(env, 0, 1)


def generate_demo_track(path: Path, seconds: float = 32.0) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(RATE * seconds)
    left = np.zeros(total)
    right = np.zeros(total)
    # A slow I–vi–IV–V style progression (A major), entirely synthesised.
    chords = [(220.0, 277.18, 329.63), (185.0, 220.0, 277.18), (146.83, 220.0, 293.66), (164.81, 207.65, 246.94)]
    bar = 4.0
    for b in range(int(seconds // bar)):
        root = chords[b % len(chords)]
        start = int(b * bar * RATE)
        for i, f in enumerate(root):
            pad = _note(f, bar + 0.6) * 0.09
            end = min(total, start + len(pad))
            pan = 0.35 + 0.15 * i
            left[start:end] += pad[: end - start] * (1 - pan)
            right[start:end] += pad[: end - start] * pan
        for k in range(8):  # gentle arpeggio
            f = root[k % 3] * (2 if k % 4 < 2 else 4)
            s = start + int(k * bar / 8 * RATE)
            pl = _note(f, 0.9, "pluck") * 0.05
            end = min(total, s + len(pl))
            pan = 0.25 if k % 2 else 0.75
            left[s:end] += pl[: end - s] * (1 - pan)
            right[s:end] += pl[: end - s] * pan
    # Soft room: a couple of short delays.
    for delay, gain in ((0.19, 0.25), (0.37, 0.15)):
        d = int(delay * RATE)
        left[d:] += right[:-d] * gain
        right[d:] += left[:-d] * gain
    fade = np.minimum(1, np.minimum(np.arange(total) / (RATE * 1.5), (total - np.arange(total)) / (RATE * 3)))
    stereo = np.stack([left * fade, right * fade], axis=1)
    stereo /= max(1e-6, np.abs(stereo).max()) / 0.8
    pcm = (stereo * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return path
