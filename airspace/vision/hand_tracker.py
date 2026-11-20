"""MediaPipe Hand Landmarker wrapper.

Runs in VIDEO mode (synchronous, temporally tracked) on the vision thread and
converts results into plain numpy-backed :class:`HandObservation` objects so
nothing downstream depends on MediaPipe types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Landmark indices (MediaPipe hand topology).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_CHAINS = (
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
PALM_POLYGON = (0, 1, 5, 9, 13, 17)


@dataclass(frozen=True)
class HandObservation:
    """One detected hand in display (possibly mirrored) normalised coordinates."""

    landmarks: np.ndarray  # (21, 3): x, y in [0, 1] of the frame, z relative depth
    handedness: str  # "Left" / "Right" from the user's perspective
    handedness_score: float  # model-reported classification score
    world: np.ndarray | None = None  # (21, 3) metric-ish world landmarks if available


class HandTracker:
    """Thin stateful wrapper around ``mediapipe.tasks.vision.HandLandmarker``."""

    def __init__(self, model_path: Path, max_hands: int = 2,
                 detection_confidence: float = 0.6, tracking_confidence: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            # Keeping an already-tracked hand needs less certainty than finding a new
            # one; live sessions showed brief dropouts at 0.6. 0.5 is MediaPipe's default.
            min_hand_presence_confidence=min(detection_confidence, 0.5),
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int, mirrored: bool = True) -> list[HandObservation]:
        # MediaPipe requires strictly increasing timestamps in VIDEO mode.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        hands: list[HandObservation] = []
        for i, lms in enumerate(result.hand_landmarks):
            arr = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
            label, score = "Unknown", 0.0
            if i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label = cat.category_name or cat.display_name or "Unknown"
                score = float(cat.score)
                # The model assumes a mirrored (selfie) image; if the user turned
                # mirroring off, the label refers to the opposite hand.
                if not mirrored and label in ("Left", "Right"):
                    label = "Right" if label == "Left" else "Left"
            world = None
            if i < len(result.hand_world_landmarks):
                world = np.array([(p.x, p.y, p.z) for p in result.hand_world_landmarks[i]], dtype=np.float32)
            hands.append(HandObservation(arr, label, score, world))
        return hands

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass
