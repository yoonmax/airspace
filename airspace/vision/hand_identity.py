"""Stable hand identities across frames.

MediaPipe does not guarantee a consistent ordering of detected hands. Each
observation is matched to an existing track by palm position (with handedness
as a soft cue); unmatched observations start new tracks. A track survives a
short grace period without observations so momentary dropouts don't reset
filters or gesture state, but a hand that re-enters after the grace period is
a *new* track — it never silently re-attaches to an old drag.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from airspace.vision.hand_tracker import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST, HandObservation


def palm_center(obs: HandObservation) -> np.ndarray:
    lm = obs.landmarks
    return lm[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP], :2].mean(axis=0).astype(np.float64)


@dataclass
class HandTrack:
    id: int
    handedness: str
    center: np.ndarray
    first_seen: float
    last_seen: float
    observation: HandObservation | None = None
    present: bool = True
    hits: int = 1
    handedness_votes: dict[str, float] = field(default_factory=dict)

    def vote(self, label: str, score: float) -> None:
        self.handedness_votes[label] = self.handedness_votes.get(label, 0.0) * 0.9 + score
        self.handedness = max(self.handedness_votes, key=self.handedness_votes.get)  # type: ignore[arg-type]


class HandIdentityTracker:
    def __init__(self, max_match_distance: float = 0.22, grace_s: float = 0.25) -> None:
        self.max_match_distance = max_match_distance
        self.grace_s = grace_s
        self.tracks: dict[int, HandTrack] = {}
        self._ids = itertools.count(1)

    def update(self, observations: list[HandObservation], t: float) -> list[HandTrack]:
        centers = [palm_center(o) for o in observations]
        live = [tr for tr in self.tracks.values() if t - tr.last_seen <= self.grace_s]
        assignment = self._assign(live, observations, centers)
        matched_tracks: set[int] = set()
        for obs_i, track in assignment.items():
            obs = observations[obs_i]
            track.center = centers[obs_i]
            track.last_seen = t
            track.observation = obs
            track.present = True
            track.hits += 1
            track.vote(obs.handedness, obs.handedness_score)
            matched_tracks.add(track.id)
        for i, obs in enumerate(observations):
            if i in assignment:
                continue
            track = HandTrack(next(self._ids), obs.handedness, centers[i], t, t, obs)
            track.vote(obs.handedness, obs.handedness_score)
            self.tracks[track.id] = track
            matched_tracks.add(track.id)
        for track in list(self.tracks.values()):
            if track.id not in matched_tracks:
                track.present = False
                track.observation = None
                if t - track.last_seen > self.grace_s:
                    del self.tracks[track.id]
        return list(self.tracks.values())

    def _assign(self, tracks: list[HandTrack], observations: list[HandObservation],
                centers: list[np.ndarray]) -> dict[int, HandTrack]:
        """Optimal partial assignment: maximise matched pairs, then minimise cost.

        With at most two hands the search space is tiny, so exhaustive search is
        exact and costs microseconds.
        """
        if not tracks or not observations:
            return {}
        cost: dict[tuple[int, int], float] = {}
        for oi, c in enumerate(centers):
            for ti, tr in enumerate(tracks):
                d = float(np.linalg.norm(c - tr.center))
                if d <= self.max_match_distance:
                    # Handedness disagreement is a soft penalty (labels flicker when hands cross).
                    cost[(oi, ti)] = d + (0.05 if observations[oi].handedness != tr.handedness else 0.0)
        best: list[tuple[int, int]] = []
        best_key = (0, 0.0)
        pairs = list(cost)
        for r in range(1, min(len(tracks), len(observations)) + 1):
            for combo in itertools.combinations(pairs, r):
                obs_used = {o for o, _ in combo}
                trk_used = {t for _, t in combo}
                if len(obs_used) != r or len(trk_used) != r:
                    continue
                total = sum(cost[p] for p in combo)
                key = (r, -total)
                if key > best_key:
                    best_key, best = key, list(combo)
        return {oi: tracks[ti] for oi, ti in best}
