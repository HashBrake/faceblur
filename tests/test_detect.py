"""Detector backends, raw candidates and the filter that trusts a face."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.detect import (
    CenterFaceBackend, Detection, DetectorBank, YuNetBackend, filter_candidates,
    iou, nms_detections, resize_long_side,
)
from faceblur.settings import Settings

LOOSE = Settings(conf=0.25, conf_weak=0.25, det_sizes=(640, 1280), verify=False, max_face_frac=1.0)


def test_finds_at_least_ten_faces_in_the_crowd_photo(messi):
    """The plan's Phase 1 check, at the plan's loose settings."""
    image = cv2.imread(str(messi))
    assert len(DetectorBank(LOOSE).detect(image)) >= 10


def test_boxes_land_inside_the_frame_and_carry_landmarks(messi):
    image = cv2.imread(str(messi))
    h, w = image.shape[:2]
    for d in DetectorBank(LOOSE).detect(image):
        assert d.w > 0 and d.h > 0
        assert 0.0 < d.score <= 1.0
        assert d.x < w and d.y < h and d.x + d.w > 0 and d.y + d.h > 0
        assert d.landmarks is not None and len(d.landmarks) == 5


def test_verification_keeps_only_faces_both_detectors_saw(lena):
    image = cv2.imread(str(lena))
    trusted = DetectorBank(Settings(max_face_frac=1.0)).detect(image)
    assert len(trusted) >= 1
    assert all(d.verified for d in trusted)


def test_the_size_cap_drops_implausibly_large_boxes():
    raw = {"yunet": [Detection(0, 0, 900, 900, 0.9, None), Detection(10, 10, 60, 60, 0.9, None)],
           "centerface": []}
    s = Settings(verify=False, max_face_frac=0.2)
    kept = filter_candidates(raw, s, (1300, 1600))
    assert [d.w for d in kept] == [60]


def test_the_aspect_check_drops_strips():
    raw = {"yunet": [Detection(0, 0, 200, 40, 0.9, None), Detection(0, 0, 40, 200, 0.9, None),
                     Detection(0, 0, 50, 60, 0.9, None)], "centerface": []}
    kept = filter_candidates(raw, Settings(verify=False), (1300, 1600))
    assert len(kept) == 1


def test_the_threshold_applies_to_the_primary_detector():
    raw = {"yunet": [Detection(0, 0, 50, 50, 0.45, None), Detection(100, 100, 50, 50, 0.55, None)],
           "centerface": []}
    kept = filter_candidates(raw, Settings(conf=0.5, verify=False), (1300, 1600))
    assert [d.score for d in kept] == [0.55]


def test_verification_needs_an_overlapping_centerface_box():
    yn = Detection(100, 100, 50, 50, 0.9, None)
    far = Detection(600, 600, 50, 50, 0.9, None)
    raw = {"yunet": [yn, far], "centerface": [Detection(105, 102, 48, 50, 0.6, None)]}
    kept = filter_candidates(raw, Settings(), (1300, 1600))
    assert len(kept) == 1 and kept[0].x == 100 and kept[0].verified


def test_engine_both_unions_without_verification():
    raw = {"yunet": [Detection(100, 100, 50, 50, 0.9, None)],
           "centerface": [Detection(600, 600, 50, 50, 0.9, None)]}
    kept = filter_candidates(raw, Settings(engine="both"), (1300, 1600))
    assert len(kept) == 2


def test_nms_hands_landmarks_to_the_winner():
    a = Detection(100, 100, 50, 50, 0.9, None, "centerface")
    b = Detection(101, 101, 50, 50, 0.8, ((1, 1),) * 5, "yunet")
    kept = nms_detections([a, b], 0.35)
    assert len(kept) == 1 and kept[0].score == 0.9 and kept[0].landmarks is not None


def test_iou_of_identical_boxes_is_one():
    d = Detection(0, 0, 10, 10, 1.0, None)
    assert iou(d, d) == pytest.approx(1.0)
    assert iou(d, [20, 20, 10, 10]) == 0.0


@pytest.mark.parametrize("det_size", [320, 640, 1280])
def test_resize_long_side_hits_the_target(det_size):
    frame = np.zeros((342, 548, 3), np.uint8)
    resized, factor = resize_long_side(frame, det_size)
    assert max(resized.shape[:2]) == det_size
    assert factor == pytest.approx(det_size / 548, rel=1e-3)


def test_detection_scales_back_with_landmarks():
    d = Detection(10, 20, 30, 40, 0.5, ((2.0, 4.0),) * 5).scaled(2.0)
    assert (d.x, d.y, d.w, d.h) == (5, 10, 15, 20)
    assert d.landmarks[0] == (1.0, 2.0)


def test_each_backend_finds_the_face_in_lena(lena):
    image = cv2.imread(str(lena))
    for backend in (YuNetBackend(), CenterFaceBackend()):
        resized, _ = resize_long_side(image, 640)
        found = [d for d in backend.detect_resized(resized) if d.score >= 0.5]
        assert len(found) >= 1, f"{backend.name} found no face"
        assert found[0].landmarks is not None


def test_blank_frame_finds_nothing():
    assert DetectorBank(Settings()).detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_raw_candidates_are_cheap_to_refilter(lena):
    image = cv2.imread(str(lena))
    bank = DetectorBank(Settings())
    raw = bank.detect_raw(image)
    strict = filter_candidates(raw, Settings(conf=0.9, max_face_frac=1.0), image.shape[:2])
    loose = filter_candidates(raw, Settings(conf=0.3, conf_weak=0.3, max_face_frac=1.0), image.shape[:2])
    assert len(loose) >= 1
    assert len(loose) >= len(strict)
