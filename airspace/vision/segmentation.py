"""Person segmentation and occlusion masks.

``PersonSegmenter`` wraps MediaPipe's selfie segmenter. ``build_hand_mask``
rasterises a hand silhouette from landmarks; intersecting it with the person
mask snaps the silhouette to real image edges. ``MaskBuilder`` turns these into
a feathered, temporally stable foreground layer for compositing.

A normal RGB webcam gives no metric depth: these masks express "this pixel is
the user" — not how far away it is — and occlusion is a compositing choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from airspace.vision.hand_tracker import (
    HAND_CONNECTIONS,
    INDEX_TIP,
    PALM_POLYGON,
    HandObservation,
)

MASK_WIDTH = 320  # internal mask resolution (width); height follows the frame aspect


class PersonSegmenter:
    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._last_ts = -1

    def segment(self, rgb_small: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Return a float32 person-confidence mask in [0, 1] at the input size."""
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_small))
        result = self._segmenter.segment_for_video(image, ts)
        masks = result.confidence_masks
        conf = masks[-1].numpy_view()  # selfie segmenter: single "person" channel
        return np.ascontiguousarray(conf.reshape(conf.shape[0], conf.shape[1])).astype(np.float32)

    def close(self) -> None:
        try:
            self._segmenter.close()
        except Exception:  # noqa: BLE001
            pass


def mask_size(frame_w: int, frame_h: int) -> tuple[int, int]:
    return MASK_WIDTH, max(1, int(round(MASK_WIDTH * frame_h / frame_w)))


def build_hand_mask(hands: list[HandObservation], size: tuple[int, int], dilation: float = 1.0) -> np.ndarray:
    """Rasterise hand silhouettes from landmarks. Returns float32 in [0, 1]."""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    for hand in hands:
        pts = np.column_stack([hand.landmarks[:, 0] * w, hand.landmarks[:, 1] * h]).astype(np.float32)
        palm = float(np.linalg.norm(pts[0] - pts[9]))
        thick = max(2, int(palm * 0.32 * dilation))
        ipts = np.round(pts).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(ipts[list(PALM_POLYGON)]), 255, lineType=cv2.LINE_AA)
        for a, b in HAND_CONNECTIONS:
            cv2.line(mask, tuple(ipts[a]), tuple(ipts[b]), 255, thick, lineType=cv2.LINE_AA)
        for p in ipts:
            cv2.circle(mask, tuple(p), max(1, thick // 2), 255, -1, lineType=cv2.LINE_AA)
        # Extend towards the wrist/forearm a little so the cut-off isn't abrupt.
        wrist, mid = pts[0], pts[9]
        forearm = wrist + (wrist - mid) * 0.6
        cv2.line(mask, tuple(ipts[0]), tuple(np.round(forearm).astype(np.int32)), 255, int(thick * 1.6),
                 lineType=cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


@dataclass(frozen=True)
class ForegroundLayer:
    """Camera pixels with alpha, cropped to the region that actually occludes."""

    x: int
    y: int
    rgba: np.ndarray  # (h, w, 4) uint8, straight alpha, frame resolution


class MaskBuilder:
    """Builds the occlusion foreground for the selected occlusion mode."""

    def __init__(self) -> None:
        self._person_ema: np.ndarray | None = None

    def reset(self) -> None:
        self._person_ema = None

    def smooth_person(self, person: np.ndarray | None) -> np.ndarray | None:
        if person is None:
            return self._person_ema
        if self._person_ema is None or self._person_ema.shape != person.shape:
            self._person_ema = person
        else:
            self._person_ema = 0.55 * person + 0.45 * self._person_ema
        return self._person_ema

    def build(self, mode: str, frame_rgb: np.ndarray, hands: list[HandObservation],
              person: np.ndarray | None) -> tuple[ForegroundLayer | None, np.ndarray | None]:
        """Return ``(foreground, alpha_small)``; foreground is None for 'clear' or empty masks."""
        if mode == "clear":
            return None, None
        fh, fw = frame_rgb.shape[:2]
        size = mask_size(fw, fh)
        if mode == "cinematic":
            if person is None:
                alpha = build_hand_mask(hands, size) if hands else None
            else:
                alpha = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                alpha = np.clip((alpha - 0.35) / 0.4, 0.0, 1.0)  # confidence threshold with soft ramp
        else:  # balanced: hands only, with a clear "lens" at the fingertip
            if not hands:
                return None, None
            alpha = build_hand_mask(hands, size, dilation=1.35 if person is not None else 1.0)
            if person is not None:
                if person.shape != alpha.shape:
                    person = cv2.resize(person, size, interpolation=cv2.INTER_LINEAR)
                alpha = alpha * np.clip((person - 0.25) / 0.35, 0.0, 1.0)
            alpha = self._fingertip_lens(alpha, hands, size)
            alpha *= 0.94
        if alpha is None:
            return None, None
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
        return self._compose(frame_rgb, alpha), alpha

    @staticmethod
    def _fingertip_lens(alpha: np.ndarray, hands: list[HandObservation], size: tuple[int, int]) -> np.ndarray:
        """Keep the UI under the index fingertip visible so targets stay readable."""
        w, h = size
        yy, xx = np.ogrid[:h, :w]
        out = alpha
        for hand in hands:
            tip = hand.landmarks[INDEX_TIP]
            cx, cy = tip[0] * w, tip[1] * h
            palm = float(np.hypot((hand.landmarks[0, 0] - hand.landmarks[9, 0]) * w,
                                  (hand.landmarks[0, 1] - hand.landmarks[9, 1]) * h))
            r = max(4.0, palm * 0.45)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            out = out * np.clip((d - r * 0.55) / (r * 0.6), 0.25, 1.0)
        return out

    @staticmethod
    def _compose(frame_rgb: np.ndarray, alpha_small: np.ndarray) -> ForegroundLayer | None:
        ys, xs = np.nonzero(alpha_small > 0.02)
        if len(xs) == 0:
            return None
        fh, fw = frame_rgb.shape[:2]
        sh, sw = alpha_small.shape
        sx, sy = fw / sw, fh / sh
        x0 = max(0, int((xs.min() - 1) * sx))
        x1 = min(fw, int((xs.max() + 2) * sx))
        y0 = max(0, int((ys.min() - 1) * sy))
        y1 = min(fh, int((ys.max() + 2) * sy))
        if x1 <= x0 or y1 <= y0:
            return None
        alpha_full = cv2.resize(alpha_small, (fw, fh), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        rgba = np.empty((y1 - y0, x1 - x0, 4), np.uint8)
        rgba[..., :3] = frame_rgb[y0:y1, x0:x1]
        rgba[..., 3] = np.clip(alpha_full * 255.0, 0, 255).astype(np.uint8)
        return ForegroundLayer(x0, y0, rgba)


def person_bbox(person: np.ndarray | None, threshold: float = 0.5) -> tuple[float, float, float, float] | None:
    """Normalised bounding box (u0, v0, u1, v1) of the person region, if any."""
    if person is None:
        return None
    ys, xs = np.nonzero(person > threshold)
    if len(xs) < 20:
        return None
    h, w = person.shape
    return xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h
