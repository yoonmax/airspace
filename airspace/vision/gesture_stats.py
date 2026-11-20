"""Rolling gesture statistics for the local diagnostics log.

Collected on the UI thread from each tracking frame and summarised every few
seconds, so gesture thresholds can be tuned from real sessions. Nothing here
leaves the machine; only aggregate numbers are kept, never images.
"""

from __future__ import annotations

import statistics
from collections import Counter

from airspace.vision.gesture_state_machine import GestureType
from airspace.vision.tracking import TrackingFrame

SHORT_PINCH_S = 0.12  # pinches shorter than this are probably flicker, not intent
DROPOUT_S = 0.5  # hand gaps shorter than this are tracker losses, not leaving view


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


class GestureStats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen_once = getattr(self, "_seen_once", False)
        self.frames = 0
        self.hand_frames = 0
        self.events: Counter[str] = Counter()
        self.poses: Counter[str] = Counter()
        self.pinch_durations: list[float] = []
        self.cancelled_pinches = 0
        self.open_ratios: list[float] = []  # primary hand pinch ratio while not pinched
        self.closed_ratios: list[float] = []  # while pinched
        self.jitter: list[float] = []
        self._pinch_start: dict[int, float] = {}
        self.extended_hist: Counter[int] = Counter()
        self.spreads: list[float] = []
        self.dropouts = 0  # hand vanished and returned within DROPOUT_S: tracker loss, not leaving view
        self._absent_since: float | None = getattr(self, "_absent_since", None)

    def add(self, frame: TrackingFrame) -> None:
        self.frames += 1
        hand = frame.primary
        present = hand is not None and hand.present and hand.features is not None
        if present and self._absent_since is not None and frame.t - self._absent_since < DROPOUT_S:
            self.dropouts += 1
        if present:
            self._absent_since = None
        elif self._absent_since is None and self._seen_once:
            self._absent_since = frame.t
        if present:
            self._seen_once = True
            self.hand_frames += 1
            self.extended_hist[hand.features.extended_count] += 1
            self.spreads.append(hand.features.spread)
            self.poses[hand.pose.value] += 1
            (self.closed_ratios if hand.pinched else self.open_ratios).append(hand.features.pinch_ratio)
            if hand.jitter > 0:
                self.jitter.append(hand.jitter)
        for ev in frame.events:
            if ev.type == GestureType.PINCH_HOLD:
                continue
            self.events[ev.type.value] += 1
            if ev.type == GestureType.PINCH_START:
                self._pinch_start[ev.hand_id] = ev.t
            elif ev.type == GestureType.PINCH_END:
                start = self._pinch_start.pop(ev.hand_id, None)
                if start is not None:
                    self.pinch_durations.append(ev.t - start)
                if ev.cancelled:
                    self.cancelled_pinches += 1

    def summary(self) -> str | None:
        """One log line, or None when no hand was seen in this window."""
        if not self.hand_frames:
            return None
        parts = [f"hand_present={100 * self.hand_frames / max(1, self.frames):.0f}%"]
        parts.append("poses=" + ",".join(f"{k}:{v}" for k, v in self.poses.most_common()))
        if self.events:
            parts.append("events=" + ",".join(f"{k}:{v}" for k, v in sorted(self.events.items())))
        if self.pinch_durations:
            short = sum(1 for d in self.pinch_durations if d < SHORT_PINCH_S)
            parts.append(f"pinch_ms_median={1000 * statistics.median(self.pinch_durations):.0f} short_pinches={short}")
        if self.cancelled_pinches:
            parts.append(f"cancelled_pinches={self.cancelled_pinches}")
        if self.open_ratios:
            parts.append(f"open_ratio_p10={_pct(self.open_ratios, 0.10):.2f} open_ratio_median="
                         f"{statistics.median(self.open_ratios):.2f}")
        if self.closed_ratios:
            parts.append(f"pinched_ratio_median={statistics.median(self.closed_ratios):.2f} "
                         f"pinched_ratio_p90={_pct(self.closed_ratios, 0.90):.2f}")
        parts.append("fingers_extended=" + ",".join(f"{k}:{v}" for k, v in sorted(self.extended_hist.items())))
        parts.append(f"spread_median={statistics.median(self.spreads):.2f} spread_p90={_pct(self.spreads, 0.90):.2f}")
        if self.dropouts:
            parts.append(f"dropouts={self.dropouts}")
        if self.jitter:
            parts.append(f"jitter_median={statistics.median(self.jitter):.4f}")
        return " ".join(parts)
