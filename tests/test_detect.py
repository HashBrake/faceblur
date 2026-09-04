"""Detector backends."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.detect import (
    CenterFaceBackend,
    DetectorBank,
    YuNetBackend,
    resize_long_side,
)
from faceblur.settings import Settings


def test_finds_at_least_ten_faces_in_the_crowd_photo(messi):
    """Phase 1 verify: at least 10 faces at conf 0.25 with det_sizes 640,1280."""
    image = cv2.imread(str(messi))
    bank = DetectorBank(Settings(conf=0.25, det_sizes=(640, 1280)))
    boxes = bank.detect(image)
    assert len(boxes) >= 10, f"found only {len(boxes)} faces"


def test_boxes_land_inside_the_frame(messi):
    image = cv2.imread(str(messi))
    h, w = image.shape[:2]
    bank = DetectorBank(Settings(conf=0.25, det_sizes=(640, 1280)))
    for x, y, bw, bh, score in bank.detect(image):
        assert bw > 0 and bh > 0
        assert 0.0 < score <= 1.0
        # A box may hang over an edge. It may not sit outside the frame.
        assert x < w and y < h
        assert x + bw > 0 and y + bh > 0


@pytest.mark.parametrize("det_size", [320, 640, 1280])
def test_resize_long_side_hits_the_target(det_size):
    frame = np.zeros((342, 548, 3), np.uint8)
    resized, factor = resize_long_side(frame, det_size)
    assert max(resized.shape[:2]) == det_size
    assert factor == pytest.approx(det_size / 548, rel=1e-3)


def test_resize_long_side_leaves_a_matching_frame_alone():
    frame = np.zeros((342, 548, 3), np.uint8)
    resized, factor = resize_long_side(frame, 548)
    assert factor == 1.0
    assert resized is frame


def test_detection_size_does_not_depend_on_source_resolution(lena):
    """A big source and a small source cost the same to scan.

    Both sources hold the same picture at different sizes, so both should yield
    about the same faces.
    """
    small = cv2.imread(str(lena))
    big = cv2.resize(small, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    bank = DetectorBank(Settings(det_sizes=(640,)))
    assert len(bank.detect(small)) == len(bank.detect(big))


def test_both_engine_runs_two_backends():
    bank = DetectorBank(Settings(engine="both"))
    assert [b.name for b in bank.backends] == ["yunet", "centerface"]
    assert set(bank.model_hashes) == {"yunet", "centerface"}


def test_each_backend_finds_the_face_in_lena(lena):
    """Both detectors fire on one clear front facing face."""
    image = cv2.imread(str(lena))
    for backend in (YuNetBackend(0.25), CenterFaceBackend(0.25)):
        resized, _ = resize_long_side(image, 640)
        boxes = backend.detect_resized(resized)
        assert len(boxes) >= 1, f"{backend.name} found no face"


def test_ensemble_never_finds_fewer_than_one_detector_alone(lena):
    image = cv2.imread(str(lena))
    alone = DetectorBank(Settings(engine="yunet")).detect(image)
    both = DetectorBank(Settings(engine="both")).detect(image)
    assert len(both) >= len(alone)


def test_blank_frame_finds_nothing():
    bank = DetectorBank(Settings())
    assert bank.detect(np.zeros((480, 640, 3), np.uint8)) == []
