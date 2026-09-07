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
    "yunet_dynamic.onnx": (
        "c0aa2a665abc3daba84ab666ee6b15352852128c84c5267897ad18e590ac466f",
        232622,
    ),
    "centerface.onnx": (
        "09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc",
        7304518,
    ),
    "centerface_dynamic.onnx": (
        "e50c58b64599f94a343ac1dc58fa4902635c2e1939b316afb39142c39c99b22e",
        7304533,
    ),
    "ultraface.onnx": (
        "34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017",
        1270727,
    ),
    "ultraface_dynamic.onnx": (
        "043540a05268e7b6392a949ec708a707da66151a32bcb5dbb6f9e685bc393051",
        1259728,
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
