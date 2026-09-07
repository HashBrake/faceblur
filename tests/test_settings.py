"""Settings defaults and validation."""
from __future__ import annotations

import pytest

from faceblur.settings import Settings, SettingsError, parse_det_sizes


def test_defaults_are_the_precision_first_ones():
    s = Settings()
    assert s.engine == "yunet"
    assert s.conf == 0.6
    assert s.det_sizes == (1280, 1920)
    assert s.verify is True
    assert s.max_face_frac == 0.15
    assert s.min_track == 2
    assert s.max_gap == 5
    assert s.max_gap_fast == 10
    assert s.tail == 2
    assert s.tail_before_max == 12
    assert s.confirm_tta == 2 and s.verify_conf_view == 0.4
    assert s.third_opinion is True and s.verify_conf_low == 0.25
    assert s.camera_comp is True and s.link_dist == 0.75
    assert s.ellipse_w == 1.10
    assert s.ellipse_h == 1.15
    assert s.mode == "blur"
    assert s.crf == 12
    assert s.suffix == "_blurred"
    assert s.mask_budget == 0.05


@pytest.mark.parametrize("changes", [
    {"engine": "yolo"},
    {"mode": "smudge"},
    {"conf": 0.0},
    {"conf": 1.5},
    {"det_sizes": ()},
    {"det_sizes": (32,)},
    {"max_face_frac": 0.0},
    {"max_face_frac": 1.5},
    {"stride": 0},
    {"min_track": 0},
    {"max_gap": -1},
    {"tail": -1},
    {"ellipse_w": 0},
    {"ellipse_h": -1},
    {"pad": -0.1},
    {"feather": -1},
    {"strength": 0},
    {"crf": 60},
])
def test_a_setting_outside_its_range_is_refused(changes):
    with pytest.raises(SettingsError):
        Settings(**changes)


def test_det_sizes_become_a_tuple_of_ints():
    assert Settings(det_sizes=[1280.0, 1920]).det_sizes == (1280, 1920)


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
    for key in ("engine", "conf", "det_sizes", "verify", "max_face_frac",
                "min_track", "max_gap", "tail", "ellipse_w", "ellipse_h",
                "pad", "feather", "mode", "strength", "crf", "preset", "mask_budget"):
        assert key in d


def test_parse_det_sizes_reads_a_list():
    assert parse_det_sizes("1280,1920") == (1280, 1920)
    assert parse_det_sizes(" 640 , 1280 , 1920 ") == (640, 1280, 1920)


def test_parse_det_sizes_refuses_an_empty_list():
    with pytest.raises(SettingsError):
        parse_det_sizes("  ")
