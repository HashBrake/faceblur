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
    "yolox_tiny.onnx": (
        "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        20219662,
    ),
    "text_detection_ppocrv3.onnx": (
        "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
        2423490,
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


def spec_datas() -> str:
    """The spec with its comments taken out.

    A file named in a comment is not a file the build carries, and these
    tests ask what the build carries.
    """
    lines = SPEC.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def test_every_model_in_the_build_carries_its_licence():
    """A model whose licence asks to travel with it has to be in the build too."""
    spec = spec_datas()
    for licence in sorted(MODELS.glob("*.LICENSE.txt")):
        family = licence.name.split(".")[0]
        shipped = [n for n in EXPECTED if n.startswith(family) and n in spec]
        if shipped:
            assert licence.name in spec, f"{shipped} ships without {licence.name}"


def test_the_face_recogniser_stays_out_of_the_build():
    """A tool that redacts faces has no business shipping a face recogniser."""
    assert "sface" not in spec_datas()


def test_the_evaluation_only_models_stay_out_of_the_build():
    """The oracle and everything it needs are evaluation only.

    OWLv2 is 1.4 GB and torch is larger still, they live in `.venv-oracle`
    which the packaged app knows nothing about, and neither has ever produced
    a mask. A build that carried them would be shipping a general purpose
    object detector to hide faces with.
    """
    spec = spec_datas().lower()
    for name in ("owl", "torch", "transformers", "mediapipe", "sface"):
        assert name not in spec, f"{name} is evaluation only and the build carries it"


def test_the_oracle_never_runs_in_the_pipeline():
    """Nothing the packaged app imports may reach the oracle or torch.

    Read by `ast` rather than by searching the source for a word. A grep here
    used to fail whenever a comment mentioned torch, which made the rule
    impossible to write about, and it would have passed a dynamic import
    assembled from pieces.
    """
    from test_classes import imported_names

    package = Path(__file__).resolve().parents[1] / "faceblur"
    for module in sorted(package.glob("*.py")):
        names = imported_names(module)
        for forbidden in ("oracle_owl", "eval.oracle_owl", "transformers", "torch",
                          "huggingface_hub", "mediapipe"):
            assert forbidden not in names, (
                f"faceblur/{module.name} imports {forbidden}, which is evaluation "
                f"only and is not in the packaged build")


def test_the_screen_model_is_in_the_build():
    """A checkbox for screens with no model behind it in the packaged app
    would mask nothing and say nothing."""
    assert "yolox_tiny.onnx" in spec_datas()


# A model in this folder that the build does not carry has to say why here.
# Everything else must be in the spec, so that the next kind's model cannot
# be committed, wired up and then left out of the packaged app, which is how
# the screen and hand models spent a week missing from dist\FaceBlur.
NOT_SHIPPED = {
    "sface.onnx":
        "the face recogniser: evaluation only, and a tool that redacts faces "
        "has no business shipping one",
    "centerface.onnx":
        "the static graph models/make_dynamic.py derives centerface_dynamic "
        "from. Nothing loads it at run time",
    "ultraface.onnx":
        "the static graph models/make_dynamic.py derives ultraface_dynamic "
        "from. Nothing loads it at run time",
    "text_detection_ppocrv3.onnx":
        "package T1 measures text and nothing masks it: classes.TEXT.ready is "
        "False, so the app cannot select the kind and the model would be 2.4 MB "
        "of dead weight. Package T3 turns text on and moves this line into the "
        "spec, and test_a_ready_kind_has_its_model_in_the_build fails until it "
        "does",
}


def test_every_model_is_either_in_the_build_or_says_why_not():
    """A model nobody decided about is a model the packaged app is missing."""
    spec = spec_datas()
    for model in sorted(MODELS.glob("*.onnx")):
        if model.name in NOT_SHIPPED:
            assert model.name not in spec, (
                f"{model.name} is listed as not shipped, with the reason "
                f"{NOT_SHIPPED[model.name]!r}, and the spec ships it anyway")
            continue
        assert model.name in spec, (
            f"{model.name} is in models/ and not in build/faceblur.spec. Add it "
            f"to the spec's datas, or add it to NOT_SHIPPED here with the "
            f"reason the build does not need it")


def test_a_ready_kind_has_its_model_in_the_build():
    """The rule that NOT_SHIPPED must not be allowed to outlive its reason.

    A kind the window offers is a kind the packaged app has to be able to
    mask. The screen and hand models spent a week missing from the packaged
    app and the only symptom was a checkbox that masked nothing in somebody
    else's copy. So the moment `classes.TEXT.ready` becomes True, the text
    model has to be in the spec.
    """
    from faceblur import classes

    if classes.TEXT.ready:
        assert "text_detection_ppocrv3.onnx" in spec_datas(), (
            "text is ready and its model is not in build/faceblur.spec. Add it "
            "to the spec's datas and take it out of NOT_SHIPPED here")


def test_every_committed_model_has_a_pinned_hash():
    """A model with no hash in this file is a model that can be swapped."""
    for model in sorted(MODELS.glob("*.onnx")):
        assert model.name in EXPECTED, (
            f"{model.name} is committed with no sha256 in EXPECTED")


def test_the_hand_models_are_in_the_build():
    """The gate needs them: without the hand rule it holds back every copy that
    shows the wearer's own hand."""
    spec = spec_datas()
    for name in ("palm_detection.onnx", "hand_landmark.onnx"):
        assert name in spec, f"{name} is not in the build"
