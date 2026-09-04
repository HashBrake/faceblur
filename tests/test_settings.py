"""Settings defaults and validation."""
from __future__ import annotations

import pytest

from faceblur.settings import Settings, SettingsError, parse_det_sizes


def test_defaults_match_the_build_plan():
    s = Settings()
    assert s.engine == "yunet"
    assert s.conf == 0.25
    assert s.det_sizes == (640, 1280)
    assert s.stride == 1
    assert s.persist == 6
    assert s.grow == 0.06
    assert s.pad == 0.30
    assert s.mode == "blur"
    assert s.strength == 28
    assert s.crf == 20
    assert s.suffix == "_blurred"
    assert s.nms_detect == 0.35
    assert s.nms_propagate == 0.60


@pytest.mark.parametrize("changes", [
    {"engine": "yolo"},
    {"mode": "smudge"},
    {"conf": 0.0},
    {"conf": 1.5},
    {"det_sizes": ()},
    {"det_sizes": (32,)},
    {"stride": 0},
    {"persist": -1},
    {"grow": -0.1},
    {"pad": -0.1},
    {"strength": 0},
    {"crf": 60},
])
def test_a_setting_outside_its_range_is_refused(changes):
    with pytest.raises(SettingsError):
        Settings(**changes)


def test_det_sizes_become_a_tuple_of_ints():
    assert Settings(det_sizes=[640.0, 1280]).det_sizes == (640, 1280)


def test_settings_do_not_change_after_they_are_made():
    s = Settings()
    with pytest.raises(Exception):
        s.conf = 0.9


def test_with_changes_returns_a_new_settings():
    s = Settings()
    t = s.with_changes(mode="solid")
    assert s.mode == "blur"
    assert t.mode == "solid"


def test_to_dict_holds_every_setting_the_audit_record_needs():
    d = Settings().to_dict()
    for key in ("engine", "conf", "det_sizes", "stride", "persist", "grow",
                "pad", "mode", "strength", "crf", "preset"):
        assert key in d


def test_parse_det_sizes_reads_a_list():
    assert parse_det_sizes("640,1280") == (640, 1280)
    assert parse_det_sizes(" 640 , 1280 , 1920 ") == (640, 1280, 1920)


def test_parse_det_sizes_refuses_an_empty_list():
    with pytest.raises(SettingsError):
        parse_det_sizes("  ")
