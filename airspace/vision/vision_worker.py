"""The vision thread: frame → mirrored RGB → hands + segmentation + masks.

Runs MediaPipe inference off the UI thread. Each processed frame becomes an
immutable :class:`VisionPacket` delivered to the UI through a queued Qt signal.
The worker always processes the *newest* frame, so if inference is slower
than the camera, stale frames are skipped rather than queued.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from airspace.camera.frame_buffer import LatestFrameBuffer
from airspace.vision.hand_tracker import HandObservation, HandTracker
from airspace.vision.segmentation import ForegroundLayer, MaskBuilder, PersonSegmenter, mask_size

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionConfig:
    mirror: bool = True
    max_hands: int = 2
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.5
    segmentation_interval: int = 2  # 0 disables segmentation
    occlusion: str = "balanced"
    need_person_mask: bool = False  # for placement / inspector even in "clear" mode


@dataclass(frozen=True)
class VisionPacket:
    frame_index: int
    frame_rgb: np.ndarray  # (H, W, 3) mirrored display frame
    blur_small: np.ndarray  # heavily blurred thumbnail for frosted glass
    hands: list[HandObservation]
    person_mask: np.ndarray | None  # small float32 confidence mask
    foreground: ForegroundLayer | None
    occlusion_alpha: np.ndarray | None  # small mask used for the foreground (debug view)
    capture_ts: float
    processed_ts: float
    hand_ms: float
    seg_ms: float
    total_ms: float
    mirrored: bool
    stats: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.frame_rgb.shape[1], self.frame_rgb.shape[0]


class VisionWorker(QObject):
    packetReady = Signal(object)  # VisionPacket
    errorOccurred = Signal(str)

    def __init__(self, buffer: LatestFrameBuffer, hand_model: Path, seg_model: Path) -> None:
        super().__init__()
        self.buffer = buffer
        self.hand_model = hand_model
        self.seg_model = seg_model
        self._config = VisionConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._times: deque[float] = deque(maxlen=60)
        self.processed = 0

    # Assigning a new frozen config object is atomic; the thread reads it once per frame.
    def set_config(self, config: VisionConfig) -> None:
        self._config = config

    @property
    def config(self) -> VisionConfig:
        return self._config

    @property
    def fps(self) -> float:
        times = list(self._times)
        if len(times) < 2 or time.perf_counter() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / max(times[-1] - times[0], 1e-6)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="airspace-vision", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self.buffer.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        self._thread = None

    # -------------------------------------------------------------- thread
    def _run(self) -> None:
        tracker: HandTracker | None = None
        segmenter: PersonSegmenter | None = None
        tracker_key = None
        masks = MaskBuilder()
        person: np.ndarray | None = None
        last_index = -1
        n = 0
        try:
            while not self._stop.is_set():
                frame = self.buffer.get_newer(last_index, timeout=0.1)
                if frame is None:
                    continue
                last_index = frame.index
                cfg = self._config
                key = (cfg.max_hands, round(cfg.detection_confidence, 2), round(cfg.tracking_confidence, 2))
                if tracker is None or key != tracker_key:
                    if tracker:
                        tracker.close()
                    try:
                        tracker = HandTracker(self.hand_model, cfg.max_hands, cfg.detection_confidence,
                                              cfg.tracking_confidence)
                        tracker_key = key
                    except Exception as exc:  # noqa: BLE001
                        log.exception("Hand tracker failed to start")
                        self.errorOccurred.emit(f"Hand tracking model could not be loaded: {exc}")
                        return
                want_seg = cfg.segmentation_interval > 0 and (cfg.occlusion != "clear" or cfg.need_person_mask)
                if want_seg and segmenter is None and self.seg_model.exists():
                    try:
                        segmenter = PersonSegmenter(self.seg_model)
                    except Exception:  # noqa: BLE001
                        log.exception("Segmenter failed to start; continuing without segmentation")
                        self.seg_model = Path("/nonexistent")

                t0 = time.perf_counter()
                image = cv2.flip(frame.image, 1) if cfg.mirror else frame.image
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                fh, fw = rgb.shape[:2]

                t_h = time.perf_counter()
                hands = tracker.detect(rgb, int(frame.timestamp * 1000), mirrored=cfg.mirror)
                hand_ms = (time.perf_counter() - t_h) * 1000

                seg_ms = 0.0
                if want_seg and segmenter is not None:
                    if n % max(1, cfg.segmentation_interval) == 0:
                        t_s = time.perf_counter()
                        small = cv2.resize(rgb, mask_size(fw, fh), interpolation=cv2.INTER_AREA)
                        person = masks.smooth_person(segmenter.segment(small, int(frame.timestamp * 1000)))
                        seg_ms = (time.perf_counter() - t_s) * 1000
                else:
                    person = None
                    masks.reset()

                foreground, alpha = masks.build(cfg.occlusion, rgb, hands, person)
                blur = cv2.resize(rgb, (max(8, fw // 10), max(8, fh // 10)), interpolation=cv2.INTER_AREA)
                blur = cv2.GaussianBlur(blur, (0, 0), 2.2)

                done = time.perf_counter()
                self._times.append(done)
                n += 1
                self.processed = n
                self.packetReady.emit(VisionPacket(
                    frame_index=frame.index, frame_rgb=rgb, blur_small=blur, hands=hands,
                    person_mask=person, foreground=foreground, occlusion_alpha=alpha,
                    capture_ts=frame.timestamp, processed_ts=done, hand_ms=hand_ms, seg_ms=seg_ms,
                    total_ms=(done - t0) * 1000, mirrored=cfg.mirror,
                    stats={"dropped": self.buffer.dropped},
                ))
        except Exception as exc:  # noqa: BLE001
            log.exception("Vision thread crashed")
            self.errorOccurred.emit(f"Vision processing stopped: {exc}")
        finally:
            if tracker:
                tracker.close()
            if segmenter:
                segmenter.close()
