"""Gesture engine tests on synthetic landmark sequences."""

from __future__ import annotations

import numpy as np
import pytest

from airspace.vision.calibration import compute_pinch_thresholds
from airspace.vision.gesture_classifier import Pose, compute_features
from airspace.vision.gesture_state_machine import GestureConfig, GestureType, HandGestureMachine
from airspace.vision.hand_identity import HandIdentityTracker
from airspace.vision.tracking import HandTrackingProcessor
from tests.synthetic_hands import fist, make_hand, open_palm, pointing

ASPECT = 16 / 9
DT = 1 / 30


def feats(obs):
    return compute_features(obs.landmarks, ASPECT)


def run(machine, sequence, t0=0.0):
    """Feed (observation|None) frames; returns list of (frame_index, event)."""
    out = []
    t = t0
    for i, obs in enumerate(sequence):
        evs = machine.lost(t) if obs is None else machine.update(feats(obs), t)
        out += [(i, e) for e in evs]
        t += DT
    return out


def types(events):
    return [e.type for _, e in events]


# ------------------------------------------------------------- features
def test_features_are_scale_invariant():
    small = feats(make_hand(scale=0.07, pinch=0.5))
    large = feats(make_hand(scale=0.2, pinch=0.5))
    assert small.pinch_ratio == pytest.approx(large.pinch_ratio, rel=1e-3)


def test_pose_classification():
    assert feats(open_palm()).pose == Pose.OPEN_PALM
    assert feats(fist()).pose == Pose.FIST
    assert feats(pointing()).pose == Pose.POINT
    assert feats(make_hand(pinch=1.0)).pose == Pose.PINCH


def test_open_hand_pinch_ratio_far_above_threshold():
    cfg = GestureConfig()
    assert feats(open_palm()).pinch_ratio > cfg.pinch_exit * 1.8
    assert feats(make_hand(pinch=1.0)).pinch_ratio < cfg.pinch_enter * 0.5


# ---------------------------------------------------------------- pinch
def test_single_pinch_produces_exactly_one_start_and_end():
    m = HandGestureMachine(1)
    seq = [open_palm()] * 5 + [make_hand(pinch=1.0)] * 20 + [open_palm()] * 10
    ev = types(run(m, seq))
    assert ev.count(GestureType.PINCH_START) == 1
    assert ev.count(GestureType.PINCH_END) == 1
    assert ev.index(GestureType.PINCH_START) < ev.index(GestureType.PINCH_END)


def test_single_frame_pinch_glitch_is_ignored():
    m = HandGestureMachine(1)
    seq = [open_palm()] * 5 + [make_hand(pinch=1.0)] + [open_palm()] * 5
    assert GestureType.PINCH_START not in types(run(m, seq))


def test_hysteresis_band_does_not_toggle():
    cfg = GestureConfig(pinch_enter=0.30, pinch_exit=0.45)
    m = HandGestureMachine(1, cfg)
    # Find a pinch amount whose ratio sits inside the hysteresis band.
    band = next(p for p in np.linspace(0.3, 1.0, 200) if 0.33 < feats(make_hand(pinch=p)).pinch_ratio < 0.42)
    seq = [make_hand(pinch=1.0)] * 5 + [make_hand(pinch=band), make_hand(pinch=1.0)] * 10
    ev = types(run(m, seq))
    assert ev.count(GestureType.PINCH_START) == 1
    assert GestureType.PINCH_END not in ev


def test_release_bounce_does_not_double_click():
    m = HandGestureMachine(1)
    seq = ([open_palm()] * 3 + [make_hand(pinch=1.0)] * 6 + [open_palm()]  # 1-frame open bounce
           + [make_hand(pinch=1.0)] * 6 + [open_palm()] * 6)
    ev = types(run(m, seq))
    assert ev.count(GestureType.PINCH_START) == 1
    assert ev.count(GestureType.PINCH_END) == 1


def test_debounce_blocks_immediate_repinch():
    cfg = GestureConfig(pinch_debounce_ms=200)
    m = HandGestureMachine(1, cfg)
    seq = [make_hand(pinch=1.0)] * 4 + [open_palm()] * 3 + [make_hand(pinch=1.0)] * 2 + [open_palm()] * 3
    assert types(run(m, seq)).count(GestureType.PINCH_START) == 1


def test_pinch_hold_emitted_once_after_duration():
    cfg = GestureConfig(pinch_hold_ms=200)
    m = HandGestureMachine(1, cfg)
    ev = run(m, [make_hand(pinch=1.0)] * 30)
    holds = [i for i, e in ev if e.type == GestureType.PINCH_HOLD]
    assert len(holds) == 1
    assert holds[0] * DT >= 0.2


def test_tracking_loss_releases_pinch_after_grace():
    cfg = GestureConfig(loss_grace_ms=150)
    m = HandGestureMachine(1, cfg)
    ev = run(m, [make_hand(pinch=1.0)] * 5 + [None] * 10)
    ends = [(i, e) for i, e in ev if e.type == GestureType.PINCH_END]
    assert len(ends) == 1 and ends[0][1].cancelled
    assert (ends[0][0] - 5) * DT >= 0.15 - 1e-9


def test_brief_tracking_dropout_keeps_pinch():
    m = HandGestureMachine(1, GestureConfig(loss_grace_ms=180))
    ev = types(run(m, [make_hand(pinch=1.0)] * 5 + [None] * 2 + [make_hand(pinch=1.0)] * 5))
    assert ev.count(GestureType.PINCH_START) == 1
    assert GestureType.PINCH_END not in ev


def test_fist_is_not_a_pinch():
    m = HandGestureMachine(1)
    ev = types(run(m, [open_palm()] * 3 + [fist()] * 20))
    assert GestureType.PINCH_START not in ev
    assert ev.count(GestureType.FIST) == 1


# ------------------------------------------------------------ open palm
def test_open_palm_fires_once_while_held_and_rearms():
    cfg = GestureConfig(palm_hold_ms=300, palm_cooldown_ms=0, palm_rearm_ms=100)
    m = HandGestureMachine(1, cfg)
    seq = [open_palm()] * 40 + [pointing()] * 6 + [open_palm()] * 20
    ev = types(run(m, seq))
    assert ev.count(GestureType.PALM_OPEN) == 2


def test_open_palm_requires_minimum_duration():
    m = HandGestureMachine(1, GestureConfig(palm_hold_ms=400))
    assert GestureType.PALM_OPEN not in types(run(m, [open_palm()] * 8 + [pointing()] * 5))


def test_palm_cooldown():
    cfg = GestureConfig(palm_hold_ms=100, palm_cooldown_ms=2000, palm_rearm_ms=30)
    m = HandGestureMachine(1, cfg)
    seq = [open_palm()] * 8 + [pointing()] * 3 + [open_palm()] * 8
    assert types(run(m, seq)).count(GestureType.PALM_OPEN) == 1


def test_moving_palm_does_not_open_launcher():
    m = HandGestureMachine(1, GestureConfig(palm_hold_ms=200))
    seq = [open_palm(center=(0.2 + 0.03 * i, 0.6)) for i in range(20)]
    assert GestureType.PALM_OPEN not in types(run(m, seq))


# ---------------------------------------------------------------- swipe
def test_horizontal_swipe_detected_once():
    m = HandGestureMachine(1)
    seq = [open_palm(center=(0.3, 0.6))] * 3 + [open_palm(center=(0.3 + 0.06 * i, 0.6)) for i in range(1, 8)]
    ev = types(run(m, seq))
    assert ev.count(GestureType.SWIPE_RIGHT) == 1


def test_slow_drift_is_not_a_swipe():
    m = HandGestureMachine(1)
    seq = [open_palm(center=(0.3 + 0.004 * i, 0.6)) for i in range(60)]
    ev = types(run(m, seq))
    assert not any(e.value.startswith("swipe") for e in ev)


def test_no_swipe_while_pinched():
    m = HandGestureMachine(1)
    seq = [make_hand(pinch=1.0, center=(0.3 + 0.06 * i, 0.6)) for i in range(10)]
    ev = types(run(m, seq))
    assert not any(e.value.startswith("swipe") for e in ev)


# ------------------------------------------------------- rapid sequences
def test_rapid_transitions_stay_consistent():
    """Every PINCH_START is followed by exactly one PINCH_END, in order."""
    rng = np.random.default_rng(3)
    m = HandGestureMachine(1)
    choices = [open_palm, fist, pointing, lambda: make_hand(pinch=1.0), lambda: make_hand(pinch=0.5)]
    seq = [choices[rng.integers(len(choices))]() for _ in range(400)]
    seq = [s if rng.random() > 0.05 else None for s in seq]
    depth = 0
    for _, e in run(m, seq):
        if e.type == GestureType.PINCH_START:
            depth += 1
        elif e.type == GestureType.PINCH_END:
            depth -= 1
        assert depth in (0, 1)


# ------------------------------------------------------------- identity
def test_identity_stable_across_order_swaps():
    tracker = HandIdentityTracker()
    left = make_hand(center=(0.3, 0.6), handedness="Left")
    right = make_hand(center=(0.7, 0.6), handedness="Right")
    a = {t.handedness: t.id for t in tracker.update([left, right], 0.0)}
    b = {t.handedness: t.id for t in tracker.update([right, left], 0.033)}
    assert a == b


def test_reentry_after_grace_creates_new_identity():
    tracker = HandIdentityTracker(grace_s=0.2)
    first = tracker.update([make_hand()], 0.0)[0].id
    tracker.update([], 0.1)
    tracker.update([], 0.5)
    again = tracker.update([make_hand()], 0.6)[0].id
    assert again != first


def test_processor_cancels_pinch_when_hand_vanishes():
    proc = HandTrackingProcessor(GestureConfig(loss_grace_ms=100))
    t = 0.0
    for _ in range(5):
        frame = proc.process([make_hand(pinch=1.0)], t, ASPECT)
        t += DT
    assert frame.primary is not None and frame.primary.pinched
    ends = []
    for _ in range(15):
        frame = proc.process([], t, ASPECT)
        ends += [e for e in frame.events if e.type == GestureType.PINCH_END]
        t += DT
    assert len(ends) == 1 and ends[0].cancelled


def test_primary_hand_does_not_switch_while_pinching():
    proc = HandTrackingProcessor()
    t = 0.0
    for _ in range(5):
        proc.process([make_hand(center=(0.6, 0.6), pinch=1.0)], t, ASPECT)
        t += DT
    primary = proc.last_frame.primary.id
    for _ in range(5):
        proc.process([make_hand(center=(0.6, 0.6), pinch=1.0),
                      make_hand(center=(0.25, 0.6), handedness="Left")], t, ASPECT)
        t += DT
    assert proc.last_frame.primary.id == primary


# ----------------------------------------------------------- calibration
def test_calibration_places_band_between_levels():
    cal = compute_pinch_thresholds([0.12, 0.1, 0.11, 0.13, 0.12, 0.1], [1.0, 1.05, 0.95, 1.1, 1.0, 0.98])
    assert cal.valid
    assert 0.12 < cal.enter < cal.exit < 1.0
    assert cal.exit - cal.enter >= 0.08


def test_calibration_rejects_indistinct_samples():
    cal = compute_pinch_thresholds([0.5] * 10, [0.55] * 10)
    assert not cal.valid


def test_gesture_stats_summarises_pinches():
    from airspace.vision.gesture_stats import GestureStats
    from airspace.vision.tracking import HandTrackingProcessor
    from tests.synthetic_hands import make_hand

    proc = HandTrackingProcessor()
    stats = GestureStats()
    assert stats.summary() is None
    t = 0.0
    for i in range(60):
        pinch = 20 <= i < 40
        hand = make_hand(center=(0.5, 0.6), scale=0.2, pinch=1.0 if pinch else 0.0)
        stats.add(proc.process([hand], t, 16 / 9))
        t += 1 / 30
    line = stats.summary()
    assert line is not None
    assert "hand_present=100%" in line
    assert "pinch_start:1" in line and "pinch_end:1" in line
    assert "short_pinches=0" in line and "pinched_ratio_median=" in line


def test_loosely_bent_pointing_hand_is_point():
    # From a live session: the other fingers are bent but not folded into the palm.
    from airspace.vision.gesture_classifier import Pose, compute_features
    from tests.synthetic_hands import make_hand

    f = compute_features(make_hand(curl=(0, 0.42, 0.42, 0.42), thumb_curl=0.7).landmarks, 16 / 9)
    assert not any(f.curled[2:]) and f.extended_count == 1
    assert f.pose == Pose.POINT


def test_pinch_survives_brief_tracking_dropout():
    from airspace.vision.gesture_state_machine import GestureConfig, GestureType, HandGestureMachine
    from airspace.vision.gesture_classifier import compute_features
    from tests.synthetic_hands import make_hand

    m = HandGestureMachine(1, GestureConfig())
    pinched = compute_features(make_hand(pinch=1.0).landmarks, 16 / 9)
    t, events = 0.0, []
    for _ in range(10):
        events += m.update(pinched, t)
        t += 1 / 30
    assert m.pinched
    for _ in range(12):  # 0.4 s without a hand, shorter than the grace period
        events += m.lost(t)
        t += 1 / 30
    assert m.pinched and not any(e.type == GestureType.PINCH_END for e in events)
    events += m.update(pinched, t)
    assert m.pinched
