"""Text masking end to end: found, held, filled solid, and checked in the copy.

Package T05 turned text on. These are its acceptance criteria. A clip of
pasted text, three black lines on a grey panel, goes through the real
pipeline and the copy is read back: the ink has to be gone under a solid
fill, and the page around it has to be the page. Decision D-C says text is
filled solid whatever `settings.mode` says, because a blurred line keeps its
length and its word breaks and a short string is easy to guess, so that is
tested on `redact` directly as well. The handled zone still wins over text,
and the output-side check has to see text left in a copy and hold it back.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import run_ffmpeg
from faceblur import classes
from faceblur.detect import Detection
from faceblur.redact import fill_mode, redact
from faceblur.settings import Settings, SettingsError
from faceblur.zone import ZoneFrame

SIZE = (480, 640)
LINES = ("INVOICE 4471", "JANE DOE", "PAID 12.50")


def text_frame() -> np.ndarray:
    """A grey wall, a lighter panel, three lines of black text on it."""
    frame = np.full((*SIZE, 3), 150, np.uint8)
    cv2.rectangle(frame, (60, 100), (580, 380), (215, 215, 215), -1)
    for k, line in enumerate(LINES):
        cv2.putText(frame, line, (90, 170 + 80 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 3, cv2.LINE_AA)
    return frame


def decode_all(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


@pytest.fixture(scope="module")
def text_clip(tmp_path_factory) -> Path:
    """Six frames of the pasted text at 10 fps, encoded nearly lossless."""
    folder = tmp_path_factory.mktemp("text")
    still = folder / "text.png"
    cv2.imwrite(str(still), text_frame())
    out = folder / "text.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-loop", "1", "-i", str(still),
                         "-frames:v", "6", "-r", "10", "-c:v", "libx264", "-preset",
                         "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the text clip: {result.stderr[-500:]}")
    return out


def run(src: Path, dst: Path, mask: tuple[str, ...]):
    from faceblur.batch import run_video, serial_submit

    settings = Settings(mask=mask, zone=False, copy_clean=False, device="cpu")
    return run_video(src, dst, settings, serial_submit), settings


@pytest.fixture(scope="module")
def masked(text_clip, tmp_path_factory):
    out = tmp_path_factory.mktemp("masked") / "text_masked.mp4"
    record, settings = run(text_clip, out, ("text",))
    return out, record, settings


@pytest.fixture(scope="module")
def unmasked(text_clip, tmp_path_factory):
    """The same clip through a run that only wants faces: nothing is masked."""
    out = tmp_path_factory.mktemp("unmasked") / "text_faces_only.mp4"
    record, settings = run(text_clip, out, ("face",))
    return out, record, settings


# ---------------------------------------------------------------- end to end

def test_pasted_text_is_filled_solid_and_the_page_around_it_is_left(text_clip, masked):
    from faceblur.pipeline import STATUS_DONE

    out, record, _ = masked
    assert record.status == STATUS_DONE, record.error
    assert record.text_lines > 0, "the detector found the lines"
    assert record.text_frames >= 4, "and the hold carried them over the clip"
    assert "text" in record.masked_mean_by_kind

    source = text_frame()
    copy = decode_all(out)
    assert len(copy) == 6
    frame = copy[3].astype(np.int16)
    ink = (source.min(axis=2) < 80).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8)).astype(bool)   # 2 px each way
    filled = (np.abs(frame - 0) <= 8).all(axis=2)
    share = filled[ink].mean()
    assert share >= 0.99, f"only {share:.3%} of the ink is under the solid fill"

    # More than 40 px from any line: the page and the wall are the page and
    # the wall, give or take the encoder.
    far = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5) > 40
    assert far.any()
    diff = np.abs(frame - source.astype(np.int16)).mean(axis=2)
    assert diff[far].mean() < 3, f"pixels far from the text moved by {diff[far].mean():.2f}"


def test_the_check_finds_no_text_left_in_a_masked_copy(text_clip, masked):
    from faceblur.verify import check, held_for

    out, _, settings = masked
    report = check(text_clip, out, settings)
    assert report["residual_text"] == 0
    assert report["residual_text_runs"] == []
    assert held_for(report, settings) == ()


def test_the_check_holds_back_a_copy_that_still_shows_text(text_clip, unmasked):
    from faceblur.verify import check, held_for

    out, record, settings = unmasked
    assert record.text_lines == 0, "a run that did not want text never looked for it"
    wants_text = settings.with_changes(mask=("text",))
    report = check(text_clip, out, wants_text)
    assert report["residual_text"] > 0
    assert report["residual_text_runs"], "the lines last the whole clip"
    assert all(row["label"] == "ppocr" for row in report["residual_list"]
               if row["kind"] == "text")
    assert held_for(report, wants_text) == ("text",)


# ----------------------------------------------------- solid whatever the mode

QUAD = ((100.0, 80.0), (300.0, 90.0), (296.0, 130.0), (96.0, 120.0))


def line_of_text() -> Detection:
    xs, ys = [p[0] for p in QUAD], [p[1] for p in QUAD]
    return Detection(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), 0.9, None,
                     "ppocr", True, 0.9, "text", QUAD)


def inside_quad(shape, erode: int = 2) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.array(QUAD)).astype(np.int32), 1)
    return cv2.erode(mask, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)).astype(bool)


@pytest.mark.parametrize("mode", ["blur", "pixelate"])
def test_text_is_filled_solid_whatever_the_mode(mode):
    frame = np.random.default_rng(5).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    out, _ = redact(frame, [line_of_text()], Settings(mode=mode))
    inside = inside_quad(frame.shape[:2])
    assert inside.sum() > 1000
    # The feathered edge is kept as it is for every kind: two pixels inside
    # the quad the alpha is 0.988, so at most 1.2 percent of the source, three
    # levels, shows through there. From four pixels in it is the fill exactly.
    assert out[inside].max() <= 3, f"text in {mode} mode is not a solid fill"
    deep = inside_quad(frame.shape[:2], erode=4)
    assert (out[deep] == 0).all(), f"text in {mode} mode is not a solid fill"


def test_only_text_and_plates_are_forced_solid():
    assert fill_mode("face", Settings(mode="blur")) == "blur"
    assert fill_mode("screen", Settings(mode="pixelate")) == "pixelate"
    assert fill_mode("text", Settings(mode="blur")) == "solid"
    assert fill_mode("plate", Settings(mode="pixelate")) == "solid"


def test_text_the_wearer_is_handling_is_left_alone():
    frame = np.random.default_rng(6).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    around = ((60.0, 40.0), (340.0, 40.0), (340.0, 170.0), (60.0, 170.0))
    out, _ = redact(frame, [line_of_text()], Settings(), zone=ZoneFrame(live=(around,)))
    inside = inside_quad(frame.shape[:2], erode=0)
    assert (out[inside] == frame[inside]).all(), "the zone wins over text"


# ------------------------------------------------------------------ the switch

def test_text_can_be_asked_for_on_every_surface():
    import cli

    args = cli.build_parser().parse_args(["x", "--mask", "text"])
    assert classes.parse(args.mask) == ("text",)
    assert classes.parse("text") == ("text",)
    assert Settings(mask=("text",)).wants("text")
    with pytest.raises(SettingsError):
        Settings(text_gate_min_run=0)
    with pytest.raises(SettingsError):
        Settings(text_gate_min_px=-1)
    recorded = Settings().to_dict()
    assert recorded["text_gate_min_px"] == 24 and recorded["text_gate_min_run"] == 2
