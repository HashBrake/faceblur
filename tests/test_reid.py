"""The re-identification harness: the recogniser works, and the mask defeats it."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from eval.reid import COSINE_THRESHOLD, Recogniser, calibrate, cosine, by_size, verdict
from faceblur.detect import DetectorBank
from faceblur.redact import redact
from faceblur.settings import Settings


@pytest.fixture(scope="module")
def rec():
    return Recogniser()


@pytest.fixture(scope="module")
def face(lena):
    frame = cv2.imread(str(lena))
    bank = DetectorBank(Settings(device="cpu", max_face_frac=1.0))
    dets = [d for d in bank.detect(frame) if d.landmarks is not None]
    assert dets, "no face with landmarks in the test picture"
    return frame, max(dets, key=lambda d: d.score)


def test_the_recogniser_matches_a_face_to_a_smaller_copy_of_itself(rec, face):
    frame, det = face
    base = rec.embed(frame, det)
    small = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3),
                       interpolation=cv2.INTER_AREA)
    bank = DetectorBank(Settings(device="cpu", max_face_frac=1.0))
    dets = [d for d in bank.detect(small) if d.landmarks is not None]
    assert dets
    other = rec.embed(small, max(dets, key=lambda d: d.score))
    assert cosine(base, other) >= COSINE_THRESHOLD


def test_the_mask_leaves_a_face_no_more_like_itself_than_a_stranger(rec, face):
    """The premise of the whole tool, in one assertion."""
    frame, det = face
    before = rec.embed(frame, det)
    for mode in ("blur", "pixelate", "solid"):
        masked, _ = redact(frame, [det], Settings(mode=mode))
        after = rec.embed(masked, det)
        assert cosine(before, after) < COSINE_THRESHOLD, mode


def test_a_face_the_pipeline_did_not_touch_still_matches_itself(rec, face):
    """The leak test's positive control: no mask, near perfect score."""
    frame, det = face
    assert cosine(rec.embed(frame, det), rec.embed(frame.copy(), det)) > 0.99


def test_embeddings_are_unit_length(rec, face):
    frame, det = face
    assert float(np.linalg.norm(rec.embed(frame, det))) == pytest.approx(1.0, abs=1e-6)


def test_a_face_without_landmarks_cannot_be_embedded(rec, face):
    from dataclasses import replace
    frame, det = face
    assert rec.embed(frame, replace(det, landmarks=None)) is None


def test_calibration_reads_the_threshold_off_the_impostors():
    genuine = [(0.8, 60, 60), (0.7, 40, 90)]
    impostor = [(0.1, 30, 30), (0.2, 30, 40), (0.5, 50, 50)] * 40
    c = calibrate(genuine, impostor)
    assert c["threshold_fmr1"] == pytest.approx(0.5, abs=0.01)
    assert c["true_match_at_fmr1"] == 1.0
    assert c["impostor_pairs"] == 120


def test_by_size_buckets_on_the_smaller_face():
    genuine = [(0.9, 20, 100), (0.2, 50, 60)]
    impostor = [(0.1, 20, 20), (0.9, 50, 50)]
    out = by_size(genuine, impostor, 0.5, edges=(24, 10 ** 6))
    assert out["0-24 px"]["same_person_pairs"] == 1 and out["0-24 px"]["recognised"] == 1.0
    assert out["24+ px"]["recognised"] == 0.0
    assert out["24+ px"]["false_match"] == 1.0


def test_a_probe_whose_own_mask_would_not_defeat_the_recogniser_decides_nothing():
    """The crop is mostly unchanged room: neither a leak nor a covered face."""
    assert verdict(score=0.99, floor=0.98, self_match=0.9, threshold=0.6) == "blind"
    assert verdict(score=0.10, floor=0.98, self_match=0.9, threshold=0.6) == "blind"


def test_a_face_the_recogniser_cannot_place_in_the_source_decides_nothing_either():
    """Nobody is identified in it before the mask, so nothing survives it."""
    assert verdict(score=0.99, floor=0.2, self_match=0.3, threshold=0.6) == "unidentifiable"
    assert verdict(score=0.10, floor=0.2, self_match=0.3, threshold=0.6) == "unidentifiable"


def test_a_probe_the_mask_would_have_defeated_reports_what_the_copy_did():
    assert verdict(score=0.99, floor=0.20, self_match=0.9, threshold=0.6) == "leak"
    assert verdict(score=0.20, floor=0.20, self_match=0.9, threshold=0.6) == "covered"


def test_the_floor_is_the_mask_this_pipeline_applies(rec, face):
    """The floor of a face large enough to fill its own crop is a stranger's score."""
    from faceblur.settings import Settings
    frame, det = face
    masked, _ = redact(frame, [det], Settings())
    floor = cosine(rec.embed(frame, det), rec.embed(masked, det))
    assert verdict(score=0.99, floor=floor, self_match=0.99,
                   threshold=COSINE_THRESHOLD) == "leak"
