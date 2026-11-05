"""T14: the face backends report the provider their pool actually ran on.

onnxruntime may fall back to the CPU when the first GPU session is built,
which happens after the backend was made. A backend that copied the pool's
provider at construction then told the audit a GPU ran a model the CPU ran.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from faceblur.detect import CenterFaceBackend, DetectorBank, UltraFaceBackend, YuNetOrtBackend
from faceblur.settings import Settings

BACKENDS = (YuNetOrtBackend, CenterFaceBackend, UltraFaceBackend)
# Every face backend loaded: YuNet, CenterFace as the verifier, UltraFace as the third opinion.
ALL_FACES = dict(engine="yunet", verify=True, confirm_on_crops=True, third_opinion=True)


@pytest.mark.parametrize("cls", BACKENDS, ids=lambda c: c.__name__)
def test_provider_follows_the_pool_after_construction(cls):
    backend = cls(device="cpu")
    assert backend.provider == "CPUExecutionProvider"
    backend.pool = SimpleNamespace(provider="SomethingElseExecutionProvider")
    assert backend.provider == "SomethingElseExecutionProvider"


def test_a_cpu_bank_reports_the_cpu_for_every_backend():
    bank = DetectorBank(Settings(device="cpu", **ALL_FACES))
    assert {b.name for b in bank.backends} == {"yunet", "centerface", "ultraface"}
    bank.detect_raw(np.full((240, 320, 3), 128, np.uint8))
    assert bank.compute == {"yunet": "CPUExecutionProvider",
                            "centerface": "CPUExecutionProvider",
                            "ultraface": "CPUExecutionProvider"}


@pytest.mark.skipif("DmlExecutionProvider" not in ort.get_available_providers(),
                    reason="DirectML is not available to onnxruntime here")
def test_an_auto_bank_on_this_pc_reports_directml_after_a_frame():
    bank = DetectorBank(Settings(device="auto", **ALL_FACES))
    bank.detect_raw(np.full((240, 320, 3), 128, np.uint8))
    assert bank.compute
    assert set(bank.compute.values()) == {"DmlExecutionProvider"}, bank.compute
