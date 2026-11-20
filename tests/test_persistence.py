import json

from airspace.persistence.settings import Settings, SettingsStore
from airspace.persistence.workspace_storage import WorkspaceLayout, WorkspaceStore


def test_settings_roundtrip(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    s = Settings()
    s.camera.device_index = 2
    s.tracking.pinch_enter = 0.22
    s.tracking.pinch_exit = 0.4
    s.interface.occlusion = "cinematic"
    s.accessibility.dwell_click = True
    store.save(s)
    loaded = store.load()
    assert loaded.camera.device_index == 2
    assert loaded.tracking.pinch_enter == 0.22
    assert loaded.interface.occlusion == "cinematic"
    assert loaded.accessibility.dwell_click is True


def test_invalid_values_fall_back_or_clamp():
    s = Settings.from_dict({
        "camera": {"profile": "ultra", "device_index": -4, "mirror": "yes"},
        "tracking": {"pinch_enter": 5.0, "pinch_exit": 0.1, "smoothing": "wobbly", "max_hands": 7},
        "interface": {"occlusion": "x-ray", "transparency": 2},
        "performance": {"fps_target": 1000},
    })
    assert s.camera.profile == "balanced"
    assert s.camera.device_index == 0
    assert s.camera.mirror is True  # wrong type ignored
    assert s.tracking.pinch_enter <= 0.8 and s.tracking.pinch_exit > s.tracking.pinch_enter
    assert s.tracking.smoothing == "balanced"
    assert s.tracking.max_hands == 2
    assert s.interface.occlusion == "balanced"
    assert s.interface.transparency <= 0.95
    assert s.performance.fps_target == 120


def test_corrupt_settings_file_uses_defaults(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert SettingsStore(p).load().camera.profile == "balanced"


def test_unknown_keys_ignored_and_clear(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"future_section": {"x": 1}, "camera": {"surprise": True}}))
    store = SettingsStore(p)
    assert store.load().camera.enabled is True
    store.clear()
    assert not p.exists()


def test_workspace_store_handles_bad_records(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"windows": [{"app_type": "calculator", "nx": 0.1, "ny": 0.2, "width": 300, "height": 400},
                                         {"nx": "bad"}, {"app_type": "x", "nx": 9, "ny": -9, "width": 1, "height": 1}]}))
    layout = WorkspaceStore(p).load()
    assert isinstance(layout, WorkspaceLayout)
    assert len(layout.windows) == 2
    clamped = layout.windows[1]
    assert clamped.nx <= 1.0 and clamped.ny >= -0.5 and clamped.width >= 50


def test_schema1_reach_migrates_to_fingertip():
    from airspace.persistence.settings import Settings

    old = {"tracking": {"pointer_reach": 1.15, "pinch_enter": 0.43, "pinch_exit": 0.55}, "schema": 1}
    s = Settings.from_dict(old)
    assert s.tracking.pointer_reach == 1.0
    assert (s.tracking.pinch_enter, s.tracking.pinch_exit) == (0.43, 0.55)  # calibration kept
    new = Settings.from_dict({"tracking": {"pointer_reach": 1.3}, "schema": 2})
    assert new.tracking.pointer_reach == 1.3  # a deliberate choice is respected
