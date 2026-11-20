"""Play queue model (Qt-free, unit tested)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aiff", ".aif"}


@dataclass
class Track:
    path: str
    title: str = ""
    artist: str = ""
    duration_ms: int = 0

    @property
    def display_title(self) -> str:
        return self.title or Path(self.path).stem.replace("_", " ")


class PlayQueue:
    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.index = -1
        self.repeat = True

    @property
    def current(self) -> Track | None:
        return self.tracks[self.index] if 0 <= self.index < len(self.tracks) else None

    def add(self, paths: list[str]) -> int:
        known = {t.path for t in self.tracks}
        added = 0
        for p in paths:
            if p in known or Path(p).suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            self.tracks.append(Track(p))
            known.add(p)
            added += 1
        if self.index < 0 and self.tracks:
            self.index = 0
        return added

    def select(self, i: int) -> Track | None:
        if 0 <= i < len(self.tracks):
            self.index = i
        return self.current

    def next(self) -> Track | None:
        if not self.tracks:
            return None
        if self.index + 1 >= len(self.tracks):
            if not self.repeat:
                return None
            self.index = 0
        else:
            self.index += 1
        return self.current

    def prev(self) -> Track | None:
        if not self.tracks:
            return None
        self.index = (self.index - 1) % len(self.tracks)
        return self.current

    def remove(self, i: int) -> None:
        if not 0 <= i < len(self.tracks):
            return
        self.tracks.pop(i)
        if not self.tracks:
            self.index = -1
        elif i < self.index or self.index >= len(self.tracks):
            self.index = max(0, self.index - 1)


def format_time(ms: int) -> str:
    s = max(0, int(ms // 1000))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
