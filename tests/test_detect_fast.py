"""T05: YuNet's input is built in place, and nothing it detects changes.

The old input path, `_pad32` then `cv2.dnn.blobFromImage`, is kept here as the
reference. A scale of 1 and a mean of 0 make it a pure pad, cast and
transpose, so the new path has to match it bit for bit, not to a tolerance.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.detect import DetectorBank, YuNetOrtBackend, _pad32
from faceblur.settings import Settings

CPU = Settings(device="cpu", max_face_frac=1.0)


def _old_blob(image: np.ndarray) -> np.ndarray:
    padded, pw, ph = _pad32(image)
    return cv2.dnn.blobFromImage(padded, 1.0, (pw, ph), (0, 0, 0), False, False)


@pytest.fixture(scope="module")
def yunet() -> YuNetOrtBackend:
    return YuNetOrtBackend(device="cpu")


@pytest.mark.parametrize("h,w", [(1080, 1920), (720, 1280), (640, 640), (360, 640),
                                 (101, 77), (31, 33), (32, 32)])
def test_the_input_tensor_is_the_old_one_exactly(yunet, h, w):
    rng = np.random.default_rng(h * 10000 + w)
    for _ in range(2):      # the second frame reuses the buffer
        image = rng.integers(0, 256, (h, w, 3), np.uint8)
        new, ref = yunet.blob(image), _old_blob(image)
        assert new.dtype == ref.dtype and new.shape == ref.shape
        assert np.array_equal(new, ref)


def test_a_smaller_frame_does_not_inherit_a_larger_ones_pixels(yunet):
    """Two sizes that pad to the same tensor keep separate buffers, so the
    pad of the smaller one is still zero."""
    big = np.full((64, 64, 3), 255, np.uint8)
    small = np.full((40, 40, 3), 7, np.uint8)
    yunet.blob(big)
    assert np.array_equal(yunet.blob(small), _old_blob(small))


def _raw_with(bank: DetectorBank, frame: np.ndarray, blob_fn) -> dict:
    saved = bank.yunet.blob
    bank.yunet.blob = blob_fn
    try:
        return bank.detect_raw(frame)
    finally:
        bank.yunet.blob = saved


def _face_frame(lena) -> np.ndarray:
    """The face clip's frame: lena at 160 wide, centred on a grey 640x640."""
    face = cv2.imread(str(lena))
    face = cv2.resize(face, (160, int(round(face.shape[0] * 160 / face.shape[1] / 2)) * 2))
    frame = np.full((640, 640, 3), 128, np.uint8)
    frame[240:240 + face.shape[0], 240:400] = face
    return frame


@pytest.mark.parametrize("which", ["lena", "messi"])
def test_detect_raw_is_unchanged(which, lena, messi):
    frame = _face_frame(lena) if which == "lena" else cv2.imread(str(messi))
    bank = DetectorBank(CPU)
    old = _raw_with(bank, frame, _old_blob)
    new = bank.detect_raw(frame)
    assert old.get("yunet"), "the reference frame has to carry a face"
    assert list(new) == list(old)
    for name in old:
        # Every field, boxes, scores, landmarks, kind and order, exactly.
        assert new[name] == old[name]
