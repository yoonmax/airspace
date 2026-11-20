"""UI-thread hand processing: identity → features → gestures → pointer.

This is pure computation (no Qt), fed by :class:`VisionPacket` hand lists or
by synthetic landmark sequences in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from airspace.interaction.cursor import PinchStablePointer
from airspace.vision.filters import SMOOTHING_PRESETS, OneEuroParams
from airspace.vision.gesture_classifier import HandFeatures, Pose, compute_features
from airspace.vision.gesture_state_machine import GestureConfig, GestureEvent, HandGestureMachine
from airspace.vision.hand_identity import HandIdentityTracker, HandTrack
from airspace.vision.hand_tracker import HandObservation


@dataclass
class TrackedHand:
    id: int
    handedness: str
    handedness_score: float
    observation: HandObservation | None
    features: HandFeatures | None
    pointer: np.ndarray | None  # filtered normalised pointer position (u, v)
    pinched: bool
    pinch_progress: float
    palm_progress: float
    present: bool
    is_primary: bool = False
    pointer_mode: str = "tip"
    jitter: float = 0.0  # derived heuristic: RMS frame-to-frame fingertip motion at rest

    @property
    def pose(self) -> Pose:
        if not self.present or self.features is None:
            return Pose.NONE
        return Pose.PINCH if self.pinched else self.features.pose


@dataclass
class TrackingFrame:
    t: float
    hands: list[TrackedHand] = field(default_factory=list)
    events: list[GestureEvent] = field(default_factory=list)

    @property
    def primary(self) -> TrackedHand | None:
        return next((h for h in self.hands if h.is_primary), None)

    @property
    def secondary(self) -> TrackedHand | None:
        return next((h for h in self.hands if not h.is_primary and h.present), None)

    @property
    def present_count(self) -> int:
        return sum(1 for h in self.hands if h.present)


class _HandState:
    def __init__(self, track_id: int, gesture_config: GestureConfig, params: OneEuroParams) -> None:
        self.machine = HandGestureMachine(track_id, gesture_config)
        self.pointer = PinchStablePointer(params)
        self.jitter_hist: list[float] = []
        self.last_tip: np.ndarray | None = None


class HandTrackingProcessor:
    def __init__(self, gesture_config: GestureConfig | None = None, smoothing: str = "balanced",
                 dominant: str = "auto") -> None:
        self.gesture_config = gesture_config or GestureConfig()
        self.params = SMOOTHING_PRESETS.get(smoothing, SMOOTHING_PRESETS["balanced"])
        self.smoothing = smoothing
        self.dominant = dominant
        self.identity = HandIdentityTracker()
        self._states: dict[int, _HandState] = {}
        self._primary_id: int | None = None
        self.last_frame = TrackingFrame(0.0)

    def configure(self, gesture_config: GestureConfig | None = None, smoothing: str | None = None,
                  dominant: str | None = None) -> None:
        if gesture_config is not None:
            self.gesture_config = gesture_config
            for st in self._states.values():
                st.machine.config = gesture_config
        if smoothing is not None:
            self.smoothing = smoothing
            self.params = SMOOTHING_PRESETS.get(smoothing, SMOOTHING_PRESETS["balanced"])
            for st in self._states.values():
                st.pointer.set_params(self.params)
        if dominant is not None:
            self.dominant = dominant

    def process(self, observations: list[HandObservation], t: float, aspect: float) -> TrackingFrame:
        tracks = self.identity.update(observations, t)
        live_ids = {tr.id for tr in tracks}
        events: list[GestureEvent] = []
        # Tracks that disappeared entirely: make sure their pinches end.
        for tid in list(self._states):
            if tid not in live_ids:
                events += self._states[tid].machine.force_release(t)
                del self._states[tid]

        hands: list[TrackedHand] = []
        for tr in tracks:
            st = self._states.get(tr.id)
            if st is None:
                st = self._states[tr.id] = _HandState(tr.id, self.gesture_config, self.params)
            if tr.present and tr.observation is not None:
                f = compute_features(tr.observation.landmarks, aspect)
                events += st.machine.update(f, t)
                ptr = st.pointer.update(f, st.machine.pinched, t)
                hands.append(self._tracked(tr, st, f, ptr, present=True))
            else:
                events += st.machine.lost(t)
                hands.append(self._tracked(tr, st, st.machine.last_features, st.pointer.value, present=False))

        self._choose_primary(tracks, hands)
        frame = TrackingFrame(t, hands, events)
        self.last_frame = frame
        return frame

    def _tracked(self, tr: HandTrack, st: _HandState, f: HandFeatures | None, ptr: np.ndarray | None,
                 present: bool) -> TrackedHand:
        jitter = 0.0
        if present and f is not None:
            if st.last_tip is not None:
                d = float(np.linalg.norm(f.index_tip - st.last_tip))
                st.jitter_hist.append(d)
                del st.jitter_hist[:-15]
                jitter = float(np.sqrt(np.mean(np.square(st.jitter_hist))))
            st.last_tip = f.index_tip.copy()
        score = tr.observation.handedness_score if tr.observation else 0.0
        return TrackedHand(
            id=tr.id, handedness=tr.handedness, handedness_score=score, observation=tr.observation,
            features=f, pointer=None if ptr is None else ptr.copy(), pinched=st.machine.pinched,
            pinch_progress=st.machine.pinch_progress, palm_progress=st.machine.palm_progress,
            present=present, pointer_mode=st.pointer.mode, jitter=jitter,
        )

    def _choose_primary(self, tracks: list[HandTrack], hands: list[TrackedHand]) -> None:
        by_id = {t.id: t for t in tracks}
        current = self._primary_id if self._primary_id in by_id else None
        # Never switch the primary hand while it is pinching (would teleport a drag).
        if current is not None:
            cur_hand = next(h for h in hands if h.id == current)
            if cur_hand.pinched:
                cur_hand.is_primary = True
                return
        present = [t for t in tracks if t.present]
        candidates = present or tracks
        chosen: int | None = None
        if self.dominant in ("left", "right"):
            want = self.dominant.capitalize()
            pref = [t for t in candidates if t.handedness == want]
            if pref:
                chosen = min(pref, key=lambda t: t.first_seen).id
        if chosen is None:
            if current is not None and by_id[current].present:
                chosen = current
            elif candidates:
                chosen = min(candidates, key=lambda t: t.first_seen).id
        self._primary_id = chosen
        for h in hands:
            h.is_primary = h.id == chosen

    def reset(self) -> None:
        self.identity = HandIdentityTracker()
        self._states.clear()
        self._primary_id = None
