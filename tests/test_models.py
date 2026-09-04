"""Phase 0 verify: the committed models are the exact files the build plan names."""
import hashlib
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"

EXPECTED = {
    "yunet.onnx": (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        232589,
    ),
    "centerface.onnx": (
        "09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc",
        7304518,
    ),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_model_present(name):
    assert (MODELS / name).is_file(), f"missing model file: {MODELS / name}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_model_sha256(name):
    expected_hash, expected_size = EXPECTED[name]
    data = (MODELS / name).read_bytes()
    assert len(data) == expected_size
    assert hashlib.sha256(data).hexdigest() == expected_hash
