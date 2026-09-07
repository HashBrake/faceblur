"""Confirmation with more than one view, the third opinion, and the growth cap."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.detect import (Detection, DetectorBank, UltraFaceBackend, _crop_reflect, _unflip,
                             confirmation, confirmed, filter_candidates, weak_candidates)
from faceblur.settings import Settings

SHAPE = (1300, 1600)


def box(x, y, w, h, score, source="yunet"):
    return Detection(float(x), float(y), float(w), float(h), score, None, source)


def raw_with(cf=0.0, flip=0.0, tight=0.0, third=0.0, at=(100, 100, 50, 50)):
    x, y, w, h = at
    r = {"yunet": [box(x, y, w, h, 0.8)], "centerface": [], "centerface_flip": [],
         "centerface_tight": [], "ultraface": []}
    if cf:
        r["centerface"].append(box(x, y, w, h, cf, "centerface"))
    if flip:
        r["centerface_flip"].append(box(x, y, w, h, flip, "centerface"))
    if tight:
        r["centerface_tight"].append(box(x, y, w, h, tight, "centerface"))
    if third:
        r["ultraface"].append(box(x, y, w, h, third, "ultraface"))
    return r


def test_the_best_view_confirms():
    s = Settings(confirm_tta=2)
    d = raw_with(cf=0.1, flip=0.5)["yunet"][0]
    assert confirmed(d, raw_with(cf=0.1, flip=0.5), s)
    assert confirmed(d, raw_with(cf=0.1, tight=0.5), s)
    assert not confirmed(d, raw_with(cf=0.1, tight=0.5), Settings(confirm_tta=1))
    assert not confirmed(d, raw_with(cf=0.1, flip=0.5), Settings(confirm_tta=0))


def test_the_third_opinion_only_counts_when_centerface_is_unsure():
    s = Settings(third_opinion=True, verify_conf=0.3, verify_conf_low=0.1, third_conf=0.5)
    d = raw_with()["yunet"][0]
    assert confirmed(d, raw_with(cf=0.15, third=0.9), s)         # unsure, third agrees
    assert not confirmed(d, raw_with(cf=0.0, third=0.9), s)      # nothing, third alone is no
    assert not confirmed(d, raw_with(cf=0.15, third=0.4), s)     # third not sure either
    assert not confirmed(d, raw_with(cf=0.15, third=0.9), Settings(third_opinion=False))


def test_confirmation_reports_score_and_third():
    d = raw_with()["yunet"][0]
    assert confirmation(d, raw_with(cf=0.2, flip=0.4, third=0.7), Settings()) == (0.4, True)


def test_a_large_confirmed_face_may_continue_but_not_start():
    """300 px on a 1600 px frame is over the 15 percent start cap and under
    the 35 percent growth cap: weak (continuation only), never strong."""
    s = Settings()
    r = raw_with(cf=0.8, at=(100, 100, 300, 300))
    assert filter_candidates(r, s, SHAPE) == []
    weak = weak_candidates(r, s, SHAPE)
    assert len(weak) == 1 and weak[0].verified


def test_a_large_unconfirmed_box_is_nothing():
    r = raw_with(cf=0.0, at=(100, 100, 300, 300))
    assert filter_candidates(r, Settings(), SHAPE) == []
    assert weak_candidates(r, Settings(), SHAPE) == []


def test_a_box_over_the_growth_cap_is_nothing_even_when_confirmed():
    r = raw_with(cf=0.9, at=(100, 100, 700, 700))
    assert weak_candidates(r, Settings(), SHAPE) == []


def test_reflect_crop_fills_the_part_past_the_edge():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :, 0] = np.arange(100, dtype=np.uint8)[None, :]      # ramp along x
    crop = _crop_reflect(frame, -20, 10, 60)
    assert crop.shape == (60, 60, 3)
    # Reflected: column 0 of the crop mirrors column 20 of the picture.
    assert int(crop[0, 0, 0]) == int(frame[10, 20, 0])
    assert int(crop[0, 20, 0]) == int(frame[10, 0, 0])
    inside = _crop_reflect(frame, 10, 10, 60)
    assert np.array_equal(inside, frame[10:70, 10:70])


def test_reflect_crop_far_past_the_edge_does_not_fail():
    frame = np.full((100, 100, 3), 7, np.uint8)
    crop = _crop_reflect(frame, -90, -90, 120)
    assert crop.shape == (120, 120, 3)


def test_unflip_mirrors_the_box_back():
    d = Detection(10, 20, 30, 40, 0.9, ((12.0, 25.0), (35.0, 25.0), (20.0, 30.0),
                                        (15.0, 50.0), (30.0, 50.0)))
    u = _unflip(d, 100)
    assert u.x == 60 and u.w == 30 and u.y == 20
    assert u.landmarks[0] == (88.0, 25.0)


def test_ultraface_finds_the_face_in_lena(lena):
    frame = cv2.imread(str(lena))
    dets = UltraFaceBackend(device="cpu").detect_resized(frame)
    assert dets, "no face"
    best = max(dets, key=lambda d: d.score)
    H, W = frame.shape[:2]
    assert best.score >= 0.7
    assert 0.2 * W < best.cx < 0.8 * W and 0.2 * H < best.cy < 0.8 * H


def test_raw_candidates_carry_every_view(lena):
    frame = cv2.imread(str(lena))
    raw = DetectorBank(Settings(device="cpu", max_face_frac=1.0)).detect_raw(frame)
    for key in ("yunet", "centerface", "centerface_flip", "centerface_tight", "ultraface"):
        assert key in raw
    assert raw["centerface"] and raw["centerface_flip"] and raw["ultraface"]


def test_a_face_cut_by_the_frame_edge_is_still_confirmed(lena):
    """Lena with the left 40 percent of her face outside the picture."""
    frame = cv2.imread(str(lena))
    bank = DetectorBank(Settings(device="cpu", max_face_frac=1.0))     # her face fills the picture
    full = bank.detect(frame)
    assert full, "no face in the whole picture"
    f = max(full, key=lambda d: d.score)
    cut = frame[:, int(f.cx - 0.1 * f.w):]
    dets = bank.detect(cut)
    assert dets and max(d.score for d in dets) >= 0.5


def test_a_bank_without_the_third_opinion_loads_two_models():
    bank = DetectorBank(Settings(device="cpu", third_opinion=False))
    assert bank.ultraface is None
    assert set(bank.model_hashes) == {"yunet", "centerface"}


def test_crops_are_cut_only_from_the_threshold_up(lena):
    """A box below conf can never be confirmed, so it gets no crop; the
    floor follows conf down to CROP_FLOOR so a cache serves a sweep."""
    frame = cv2.imread(str(lena))
    weak_box = Detection(100.0, 100.0, 60.0, 60.0, 0.4, None)
    bank = DetectorBank(Settings(device="cpu", conf=0.6))
    assert all(v == [] for v in bank.confirm_on_crops(frame, [weak_box]).values())
    bank = DetectorBank(Settings(device="cpu", conf=0.4))
    out = bank.confirm_on_crops(frame, [weak_box])
    assert "centerface" in out       # a crop was cut; whether it holds a face is not the point
