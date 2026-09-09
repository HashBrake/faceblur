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
    "sface.onnx": (
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        38696353,
    ),
    "ultraface_dynamic.onnx": (
        "043540a05268e7b6392a949ec708a707da66151a32bcb5dbb6f9e685bc393051",
        1259728,
    ),
    "palm_detection.onnx": (
        "03440370cb546a4fdcff45524300a5a8293afc84cfaf44c0b112d209386ffa30",
        4605970,
    ),
    "hand_landmark.onnx": (
        "00e6f22f25156220974589e012562ccce84b3d4b043690ac6085e8701261a9df",
        10914627,
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


SPEC = Path(__file__).resolve().parents[1] / "build" / "faceblur.spec"


def test_every_model_in_the_build_carries_its_licence():
    """A model whose licence asks to travel with it has to be in the build too."""
    spec = SPEC.read_text(encoding="utf-8")
    for licence in sorted(MODELS.glob("*.LICENSE.txt")):
        family = licence.name.split(".")[0]
        shipped = [n for n in EXPECTED if n.startswith(family) and n in spec]
        if shipped:
            assert licence.name in spec, f"{shipped} ships without {licence.name}"


def test_the_face_recogniser_stays_out_of_the_build():
    """A tool that redacts faces has no business shipping a face recogniser."""
    assert "sface" not in SPEC.read_text(encoding="utf-8")


def test_the_hand_models_are_in_the_build():
    """The gate needs them: without the hand rule it holds back every copy that
    shows the wearer's own hand."""
    spec = SPEC.read_text(encoding="utf-8")
    for name in ("palm_detection.onnx", "hand_landmark.onnx"):
        assert name in spec, f"{name} is not in the build"
