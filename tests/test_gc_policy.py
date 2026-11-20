"""The UI-thread garbage collector still reclaims cycles and freezes start-up objects."""

from __future__ import annotations

import gc
import weakref

from airspace.app import gc_policy
from airspace.app.gc_policy import UiThreadGarbageCollector


class _Node:
    pass


def test_disables_automatic_gc_but_still_collects_cycles(qapp):
    collector = UiThreadGarbageCollector()
    try:
        collector.start()
        assert not gc.isenabled()
        a, b = _Node(), _Node()
        a.other, b.other = b, a  # a reference cycle: only the cyclic collector frees it
        ref = weakref.ref(a)
        del a, b
        assert ref() is not None
        collector.tick()
        assert ref() is None
    finally:
        collector.stop()
    assert gc.isenabled()


def test_freezes_startup_objects_then_runs_full_collections(qapp, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(gc_policy.time, "monotonic", lambda: clock[0])
    calls = []
    monkeypatch.setattr(gc_policy.gc, "collect", lambda *a: calls.append(a) or 0)
    froze = []
    monkeypatch.setattr(gc_policy.gc, "freeze", lambda: froze.append(True))
    collector = UiThreadGarbageCollector()
    collector.tick()
    assert calls[-1] == (0,) and not froze  # young collection only
    clock[0] += collector.FREEZE_AFTER_S
    collector.tick()
    assert froze and calls[-1] == ()  # full collection, then freeze
    clock[0] += collector.FULL_EVERY_S
    collector.tick()
    assert calls[-1] == () and len(froze) == 1
