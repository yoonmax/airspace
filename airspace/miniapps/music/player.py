"""music — a local audio player with gesture-friendly transport.

Playback uses Qt Multimedia (QMediaPlayer + QAudioOutput). The visualiser is
driven by real decoded audio from QAudioBufferOutput when the backend
supports it; otherwise it is simply hidden — never faked.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaMetaData, QMediaPlayer

from airspace.interaction.events import CallbackGrab, InteractionState, PointerEvent
from airspace.miniapps.base_app import MiniApp
from airspace.miniapps.music.demo_track import generate_demo_track
from airspace.miniapps.music.queue import PlayQueue, format_time
from airspace.ui.controls import Button, Slider
from airspace.ui.icons import draw_icon
from airspace.ui.styles import Theme, with_alpha

log = logging.getLogger(__name__)

AUDIO_FILTER = "Audio (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.aiff *.aif)"
BANDS = 28


class MusicApp(MiniApp):
    app_type = "music"
    title = "music"
    icon = "music"
    default_size = (470.0, 330.0)
    min_size = (400.0, 290.0)
    accepts_swipe = True

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.queue = PlayQueue()
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.audio.setVolume(0.7)
        self.player.setAudioOutput(self.audio)
        self.duration = 0
        self.position = 0
        self.seek_preview: float | None = None
        self.art: QImage | None = None
        self.error = ""
        self.show_queue = False
        self.levels = np.zeros(BANDS)
        self._target_levels = np.zeros(BANDS)
        self._last_buffer = 0.0
        self.player.durationChanged.connect(self._on_duration)
        self.player.positionChanged.connect(self._on_position)
        self.player.mediaStatusChanged.connect(self._on_status)
        self.player.metaDataChanged.connect(self._on_meta)
        self.player.errorOccurred.connect(self._on_error)
        self._setup_visualiser()

        defer = lambda fn: (lambda: QTimer.singleShot(0, fn))  # noqa: E731
        c = self.controls
        self.btn_import = c.add(Button(icon="plus", variant="icon", on_click=defer(self.import_files), tooltip="Import audio"))
        self.btn_queue = c.add(Button(icon="queue", variant="icon", on_click=self._toggle_queue,
                                      checked=lambda: self.show_queue, tooltip="Queue"))
        self.btn_prev = c.add(Button(icon="prev", variant="icon", on_click=self.prev))
        self.btn_play = c.add(_PlayButton(self))
        self.btn_next = c.add(Button(icon="next", variant="icon", on_click=self.next))
        self.progress = c.add(Slider(self._progress_value, self._seek_preview, self._seek_commit, fmt=self._fmt_pos))
        self.progress.secondary = None
        self.volume = c.add(Slider(lambda: self.audio.volume(), self._set_volume, minimum=0.0, maximum=1.0, thin=True,
                                   fmt=lambda v: f"{v * 100:.0f}%"))
        self.btn_demo = c.add(Button("Add demo track", icon="sparkle", on_click=defer(self.add_demo)))
        self.btn_empty_import = c.add(Button("Import audio", icon="folder", variant="primary", on_click=defer(self.import_files)))
        self.queue_rows: list[tuple[QRectF, int]] = []

    # ------------------------------------------------------- visualiser
    def _setup_visualiser(self) -> None:
        self.has_visualiser = False
        try:
            from PySide6.QtMultimedia import QAudioBufferOutput, QAudioFormat

            fmt = QAudioFormat()
            fmt.setSampleRate(44100)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
            self._buffer_out = QAudioBufferOutput(fmt)
            self._buffer_out.audioBufferReceived.connect(self._on_buffer)
            self.player.setAudioBufferOutput(self._buffer_out)
            self.has_visualiser = True
        except Exception as exc:  # noqa: BLE001 - optional feature
            log.info("Audio visualiser unavailable: %s", exc)

    def _on_buffer(self, buf) -> None:
        try:
            n = buf.sampleCount()
            if n <= 0:
                return
            raw = buf.constData()
            data = np.frombuffer(bytes(raw)[: buf.byteCount()], dtype=np.float32)
            if data.size < 64:
                return
            win = data[-1024:] * np.hanning(min(1024, data.size))
            spec = np.abs(np.fft.rfft(win))
            # Log-spaced bands, 40 Hz → 16 kHz.
            freqs = np.fft.rfftfreq(win.size, 1 / 44100)
            edges = np.geomspace(40, 16000, BANDS + 1)
            bands = np.array([spec[(freqs >= lo) & (freqs < hi)].mean() if np.any((freqs >= lo) & (freqs < hi)) else 0
                              for lo, hi in zip(edges[:-1], edges[1:])])
            db = 20 * np.log10(bands + 1e-6)
            self._target_levels = np.clip((db + 20) / 50, 0, 1)
            self._last_buffer = time.monotonic()
        except Exception:  # noqa: BLE001
            log.debug("visualiser buffer failed", exc_info=True)

    # ---------------------------------------------------------- transport
    def _fmt_pos(self, v: float) -> str:
        return format_time(int(v * self.duration))

    def _progress_value(self) -> float:
        if self.seek_preview is not None:
            return self.seek_preview
        return self.position / self.duration if self.duration else 0.0

    def _seek_preview(self, v: float) -> None:
        self.seek_preview = v

    def _seek_commit(self, v: float) -> None:
        if self.duration:
            self.player.setPosition(int(v * self.duration))
            self.position = int(v * self.duration)
        self.seek_preview = None

    def _set_volume(self, v: float) -> None:
        self.audio.setVolume(max(0.0, min(1.0, v)))

    @property
    def playing(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def toggle_play(self) -> None:
        if self.queue.current is None:
            return
        if self.player.source().isEmpty():
            self._load(self.queue.current.path, autoplay=True)
        elif self.playing:
            self.player.pause()
        else:
            self.player.play()

    def next(self) -> None:
        t = self.queue.next()
        if t:
            self._load(t.path, autoplay=True)

    def prev(self) -> None:
        if self.position > 3000:
            self.player.setPosition(0)
            return
        t = self.queue.prev()
        if t:
            self._load(t.path, autoplay=True)

    def play_index(self, i: int) -> None:
        t = self.queue.select(i)
        if t:
            self._load(t.path, autoplay=True)

    def _load(self, path: str, autoplay: bool) -> None:
        self.error = ""
        self.art = None
        self.duration = self.position = 0
        self.player.setSource(QUrl.fromLocalFile(path))
        if autoplay:
            self.player.play()

    def _on_duration(self, d: int) -> None:
        self.duration = int(d)
        if self.queue.current:
            self.queue.current.duration_ms = int(d)

    def _on_position(self, pos: int) -> None:
        self.position = int(pos)

    def _on_status(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.queue.index + 1 < len(self.queue.tracks) or self.queue.repeat:
                self.next()
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            self.error = "This file can't be played by the audio backend"

    def _on_meta(self) -> None:
        md = self.player.metaData()
        t = self.queue.current
        if t is None:
            return
        title = md.stringValue(QMediaMetaData.Key.Title)
        artist = md.stringValue(QMediaMetaData.Key.ContributingArtist) or md.stringValue(QMediaMetaData.Key.AlbumArtist)
        if title:
            t.title = title
        if artist:
            t.artist = artist
        for key in (QMediaMetaData.Key.CoverArtImage, QMediaMetaData.Key.ThumbnailImage):
            img = md.value(key)
            if isinstance(img, QImage) and not img.isNull():
                self.art = img
                break

    def _on_error(self, error, message: str = "") -> None:
        self.error = message or "Playback error"
        log.warning("Media error: %s", message)

    # ------------------------------------------------------------- files
    def import_files(self) -> None:
        files = self.ctx.open_files("Import audio", AUDIO_FILTER)
        if not files:
            return
        was_empty = self.queue.current is None
        n = self.queue.add(files)
        self.ctx.toast(f"Added {n} track{'s' if n != 1 else ''} to the queue")
        if was_empty and self.queue.current:
            self._load(self.queue.current.path, autoplay=False)
        self.relayout()

    def add_demo(self) -> None:
        path = generate_demo_track(self.ctx.data_dir / "demo" / "airspace-demo-ambient.wav")
        was_empty = self.queue.current is None
        self.queue.add([str(path)])
        if self.queue.current and self.queue.current.path == str(path):
            self.queue.current.title = "airspace ambient (demo)"
            self.queue.current.artist = "synthesised locally"
        if was_empty:
            self._load(str(path), autoplay=True)
        self.relayout()

    def _toggle_queue(self) -> None:
        self.show_queue = not self.show_queue
        self.relayout()

    def swipe(self, direction: str) -> bool:
        if direction == "left":
            self.next()
        elif direction == "right":
            self.prev()
        else:
            return False
        return self.queue.current is not None

    def key(self, event: QKeyEvent) -> bool:
        k = event.key()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        if k == Qt.Key.Key_Space:
            self.toggle_play()
        elif k == Qt.Key.Key_Right:
            self.player.setPosition(min(self.duration, self.position + 5000))
        elif k == Qt.Key.Key_Left:
            self.player.setPosition(max(0, self.position - 5000))
        elif k == Qt.Key.Key_Up:
            self._set_volume(self.audio.volume() + 0.05)
        elif k == Qt.Key.Key_Down:
            self._set_volume(self.audio.volume() - 0.05)
        elif k == Qt.Key.Key_N:
            self.next()
        elif k == Qt.Key.Key_P:
            self.prev()
        else:
            return False
        return True

    # ------------------------------------------------------------ layout
    def layout(self, w: float, h: float) -> None:
        s = self.theme.scale
        pad = 16 * s
        art = min(h * 0.46, w * 0.34)
        self.art_rect = QRectF(pad, pad, art, art)
        tx = pad + art + 16 * s
        self.info_rect = QRectF(tx, pad, w - tx - pad, art)
        self.btn_import.rect = QRectF(w - pad - 30 * s, pad - 4, 30 * s, 30 * s)
        self.btn_queue.rect = QRectF(w - pad - 64 * s, pad - 4, 30 * s, 30 * s)
        prog_y = pad + art + 18 * s
        self.progress.rect = QRectF(pad, prog_y, w - 2 * pad, 18 * s)
        ctl_y = prog_y + 40 * s
        big = 52 * s
        cx = w / 2
        self.btn_play.rect = QRectF(cx - big / 2, ctl_y, big, big)
        self.btn_prev.rect = QRectF(cx - big / 2 - 14 * s - 42 * s, ctl_y + 5 * s, 42 * s, 42 * s)
        self.btn_next.rect = QRectF(cx + big / 2 + 14 * s, ctl_y + 5 * s, 42 * s, 42 * s)
        vol_w = min(130 * s, w * 0.25)
        self.volume.rect = QRectF(w - pad - vol_w, ctl_y + big / 2 - 9 * s, vol_w, 18 * s)
        self.vis_rect = QRectF(pad, ctl_y + big + 10 * s, w - 2 * pad, max(0.0, h - (ctl_y + big + 10 * s) - pad))
        empty = not self.queue.tracks
        for b in (self.btn_demo, self.btn_empty_import):
            b.visible = empty
        self.btn_empty_import.rect = QRectF(tx, pad + art - 84 * s, 150 * s, 36 * s)
        self.btn_demo.rect = QRectF(tx, pad + art - 42 * s, 170 * s, 36 * s)
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.enabled = not empty
        self.progress.enabled = not empty and self.duration > 0

    def tick(self, dt: float) -> None:
        super().tick(dt)
        if not self.playing or time.monotonic() - self._last_buffer > 0.3:
            self._target_levels = self._target_levels * 0.0
        rate = 18 if np.any(self._target_levels > self.levels) else 6
        self.levels += (self._target_levels - self.levels) * min(1.0, dt * rate)
        self.progress.enabled = bool(self.queue.tracks) and self.duration > 0
        empty = not self.queue.tracks
        if self.btn_demo.visible != empty:
            self.relayout()

    # ------------------------------------------------------------- input
    def content_press(self, ev: PointerEvent):
        if self.show_queue:
            for rect, i in self.queue_rows:
                if rect.contains(QPointF(ev.x, ev.y)):
                    remove_hit = ev.x > rect.right() - 30

                    def release(e: PointerEvent, i=i, remove_hit=remove_hit, rect=rect) -> None:
                        if e.cancelled or not rect.adjusted(-6, -6, 6, 6).contains(QPointF(e.x, e.y)):
                            return
                        if remove_hit:
                            playing_removed = i == self.queue.index
                            self.queue.remove(i)
                            if playing_removed:
                                self.player.stop()
                                self.player.setSource(QUrl())
                        else:
                            self.play_index(i)

                    return CallbackGrab(InteractionState.CLICK_PENDING, None, release)
        return None

    # ------------------------------------------------------------- state
    def save_state(self) -> dict:
        return {"paths": [t.path for t in self.queue.tracks][:300], "index": self.queue.index,
                "volume": round(self.audio.volume(), 3)}

    def restore_state(self, state: dict) -> None:
        paths = [p for p in state.get("paths", []) if isinstance(p, str) and Path(p).exists()]
        self.queue.add(paths)
        idx = int(state.get("index", 0))
        if self.queue.tracks:
            self.queue.select(min(max(0, idx), len(self.queue.tracks) - 1))
            self._load(self.queue.current.path, autoplay=False)
        self._set_volume(float(state.get("volume", 0.7)))
        for t in self.queue.tracks:
            if t.path.endswith("airspace-demo-ambient.wav"):
                t.title, t.artist = "airspace ambient (demo)", "synthesised locally"

    def on_close(self) -> None:
        self.player.stop()

    def status_text(self) -> str:
        if not self.queue.tracks:
            return ""
        return f"{self.queue.index + 1} of {len(self.queue.tracks)}"

    # ------------------------------------------------------------- paint
    def paint_content(self, p: QPainter, theme: Theme) -> None:
        s = theme.scale
        if self.show_queue:
            self._paint_queue(p, theme)
        else:
            self._paint_art(p, theme)
        info = self.info_rect
        t = self.queue.current
        p.setPen(theme.text)
        p.setFont(theme.font(15, QFont.Weight.DemiBold))
        title = t.display_title if t else "Nothing playing"
        fm = p.fontMetrics()
        title_w = info.width() - 74 * s
        p.drawText(QRectF(info.left(), info.top() + 4 * s, title_w, 26 * s), Qt.AlignmentFlag.AlignLeft,
                   fm.elidedText(title, Qt.TextElideMode.ElideRight, int(title_w)))
        p.setFont(theme.font(10.5))
        p.setPen(theme.text_secondary)
        sub = (t.artist or "local file") if t else "Import audio files or try the demo track"
        p.drawText(QRectF(info.left(), info.top() + 32 * s, info.width(), 20 * s), Qt.AlignmentFlag.AlignLeft, sub)
        if self.error:
            p.setPen(theme.danger)
            p.setFont(theme.font(9.5))
            p.drawText(QRectF(info.left(), info.top() + 54 * s, info.width(), 18 * s), Qt.AlignmentFlag.AlignLeft, self.error)
        # times under the progress bar
        pr = self.progress.rect
        p.setFont(theme.font(9))
        p.setPen(theme.text_tertiary)
        pos = int(self._progress_value() * self.duration) if self.duration else self.position
        p.drawText(QRectF(pr.left(), pr.bottom() + 2, 80, 14), Qt.AlignmentFlag.AlignLeft, format_time(pos))
        p.drawText(QRectF(pr.right() - 80, pr.bottom() + 2, 80, 14), Qt.AlignmentFlag.AlignRight,
                   format_time(self.duration) if self.duration else "--:--")
        vr = self.volume.rect
        draw_icon(p, "volume", QRectF(vr.left() - 24 * s, vr.center().y() - 8 * s, 16 * s, 16 * s), theme.text_secondary, 1.4)
        self._paint_visualiser(p, theme)

    def _paint_art(self, p: QPainter, theme: Theme) -> None:
        r = self.art_rect
        path = QPainterPath()
        path.addRoundedRect(r, theme.radius_small, theme.radius_small)
        p.save()
        p.setClipPath(path)
        if self.art is not None:
            sc = max(r.width() / self.art.width(), r.height() / self.art.height())
            w, h = self.art.width() * sc, self.art.height() * sc
            p.drawImage(QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h), self.art)
        else:
            grad = QLinearGradient(r.topLeft(), r.bottomRight())
            grad.setColorAt(0.0, QColor(58, 70, 92))
            grad.setColorAt(1.0, QColor(26, 28, 36))
            p.fillRect(r, grad)
            isz = r.width() * 0.34
            draw_icon(p, "music", QRectF(r.center().x() - isz / 2, r.center().y() - isz / 2, isz, isz),
                      with_alpha(theme.text, 0.75), 1.8)
        p.restore()
        p.setPen(QPen(theme.border, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _paint_queue(self, p: QPainter, theme: Theme) -> None:
        r = self.art_rect.united(QRectF(self.art_rect.left(), self.art_rect.top(), self.width - 2 * self.art_rect.left(),
                                        self.art_rect.height()))
        self.info_rect = QRectF(-9999, 0, 0, 0)  # queue view replaces the info block
        s = theme.scale
        p.setPen(QPen(theme.border, 1))
        p.setBrush(with_alpha(QColor(8, 9, 11), 0.35))
        p.drawRoundedRect(r, theme.radius_small, theme.radius_small)
        self.queue_rows = []
        row_h = 30 * s
        y = r.top() + 6 * s
        if not self.queue.tracks:
            p.setPen(theme.text_tertiary)
            p.setFont(theme.font(10))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "queue is empty")
        for i, t in enumerate(self.queue.tracks):
            if y + row_h > r.bottom() - 4:
                break
            rr = QRectF(r.left() + 6 * s, y, r.width() - 12 * s - 64 * s, row_h)
            current = i == self.queue.index
            if current:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(theme.surface_active)
                p.drawRoundedRect(rr, 8, 8)
            p.setPen(theme.text if current else theme.text_secondary)
            p.setFont(theme.font(10.5, QFont.Weight.Medium if current else QFont.Weight.Normal))
            fm = p.fontMetrics()
            label = fm.elidedText(f"{i + 1}.  {t.display_title}", Qt.TextElideMode.ElideRight, int(rr.width() - 80 * s))
            p.drawText(rr.adjusted(10 * s, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            p.setPen(theme.text_tertiary)
            p.setFont(theme.font(9))
            p.drawText(rr.adjusted(0, 0, -34 * s, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       format_time(t.duration_ms) if t.duration_ms else "")
            draw_icon(p, "close", QRectF(rr.right() - 24 * s, rr.center().y() - 7 * s, 14 * s, 14 * s), theme.text_tertiary, 1.3)
            self.queue_rows.append((rr, i))
            y += row_h + 2

    def _paint_visualiser(self, p: QPainter, theme: Theme) -> None:
        r = self.vis_rect
        if r.height() < 14 or not self.has_visualiser:
            return
        n = BANDS
        gap = 3.0
        bw = (r.width() - gap * (n - 1)) / n
        p.setPen(Qt.PenStyle.NoPen)
        for i, lv in enumerate(self.levels):
            h = max(2.0, r.height() * float(lv))
            bar = QRectF(r.left() + i * (bw + gap), r.bottom() - h, bw, h)
            p.setBrush(with_alpha(theme.accent, 0.25 + 0.55 * float(lv)))
            p.drawRoundedRect(bar, min(2.5, bw / 2), min(2.5, bw / 2))


class _PlayButton(Button):
    def __init__(self, app: MusicApp) -> None:
        super().__init__(variant="primary", on_click=app.toggle_play, radius=999)
        self.app = app

    def paint(self, p: QPainter, theme: Theme) -> None:
        self.icon = "pause" if self.app.playing else "play"
        self.radius = self.rect.height() / 2
        super().paint(p, theme)
