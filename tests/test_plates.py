"""Licence plates end to end: found, held, filled solid, and checked in the copy.

Package T06b turned plates on, on the pattern T05 set for text. The detector
is asked about the one CC0 plate photo and a grey frame; then the plate is
cut out of that photo, pasted twice onto a grey frame, made into a clip and
put through the real pipeline, and the copy is read back: the plate has to be
under a solid fill and the grey around it has to be the grey. The handled
zone still wins over a plate, and the output-side check has to see a plate
left in a copy and hold it back.

"The plate" in the clip is the plate's own quadrilateral, the corners
`models/README.md` records for this photo, mapped into the paste. The
hand-measured box is the plate's upright bounds, and the photo's plate is
tilted, so the box's corners are bumper rather than plate.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from conftest import DATA, run_ffmpeg
from faceblur import classes
from faceblur.detect import Detection, iou
from faceblur.plates import PLATE_MODEL, PlateDetector, canvas_for, preprocess, prior_count
from faceblur.redact import redact
from faceblur.settings import Settings, SettingsError
from faceblur.zone import ZoneFrame

PHOTO = DATA / "plate_cc0.jpg"
PLATE_XYWH = (136, 411, 182, 64)          # hand-measured, tests/data/PLATE_IMAGES.md
# The plate's corners in the photo, from models/README.md (the CPU decode at
# the photo's own size): top left, top right, bottom right, bottom left.
PLATE_QUAD = ((144, 410), (309, 424), (314, 475), (149, 460))

SIZE = (720, 960)
PASTE_W, PASTE_H = 160, 56                # 182x64 scaled to 160 wide
POSITIONS = ((120, 150), (620, 480))


def pasted() -> tuple[np.ndarray, np.ndarray]:
    """The grey frame with the plate pasted twice, and where the plates are."""
    x, y, w, h = PLATE_XYWH
    crop = cv2.resize(cv2.imread(str(PHOTO))[y:y + h, x:x + w], (PASTE_W, PASTE_H),
                      interpolation=cv2.INTER_AREA)
    # 150, the grey of the text clip. The pipeline's own encode brings flat
    # grey back two to five levels off whatever is masked (128 comes back as
    # 123, 126, 124), and at 150 that drift stays under the 3 level bar.
    frame = np.full((*SIZE, 3), 150, np.uint8)
    plate = np.zeros(SIZE, np.uint8)
    for px, py in POSITIONS:
        frame[py:py + PASTE_H, px:px + PASTE_W] = crop
        quad = (np.array(PLATE_QUAD, float) - [x, y]) * [PASTE_W / w, PASTE_H / h] + [px, py]
        cv2.fillConvexPoly(plate, np.round(quad).astype(np.int32), 1)
    return frame, plate.astype(bool)


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


# ------------------------------------------------------------------ detector

@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    return cv2.imread(str(PHOTO))


def test_the_plate_in_the_photo_is_found_once(photo):
    found = PlateDetector(Settings(mask=("plate",))).detect(photo)
    assert len(found) == 1, "NMS leaves one box per plate"
    plate = found[0]
    assert plate.kind == "plate" and plate.source == "lpd_yunet"
    assert iou(plate, PLATE_XYWH) >= 0.5
    assert len(plate.quad) == 4


def test_a_grey_frame_has_no_plate():
    assert PlateDetector(Settings(mask=("plate",))).detect(
        np.full((480, 640, 3), 128, np.uint8)) == []


def test_a_plate_size_smaller_than_the_frame_still_finds_it(photo):
    found = PlateDetector(Settings(mask=("plate",), plate_size=800)).detect(photo)
    assert len(found) == 1
    assert iou(found[0], PLATE_XYWH) >= 0.5


@pytest.mark.parametrize("shape", [(1300, 1600), (676, 1024), (1080, 1920), (480, 640)])
def test_the_padded_canvas_gives_the_rows_the_graph_returns(shape):
    """The DirectML graph is built with `prior_count` rows, so it has to be
    what the dynamic graph returns at the padded size."""
    canvas, _ = canvas_for(np.zeros((*shape, 3), np.uint8), 1280)
    assert canvas.shape[0] % 32 == 0 and canvas.shape[1] % 32 == 0
    assert max(canvas.shape[:2]) == 1280
    cpu = ort.InferenceSession(str(PLATE_MODEL), providers=["CPUExecutionProvider"])
    (loc,) = cpu.run(["loc"], {"input": preprocess(canvas)})
    assert len(loc) == prior_count(canvas.shape[1], canvas.shape[0])


# ---------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def plate_clip(tmp_path_factory) -> Path:
    """Four frames of the pasted plates at 10 fps, encoded nearly lossless."""
    folder = tmp_path_factory.mktemp("plate")
    still = folder / "plate.png"
    cv2.imwrite(str(still), pasted()[0])
    out = folder / "plate.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-loop", "1", "-i", str(still),
                         "-frames:v", "4", "-r", "10", "-c:v", "libx264", "-preset",
                         "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the plate clip: {result.stderr[-500:]}")
    return out


def run(src: Path, dst: Path, mask: tuple[str, ...]):
    from faceblur.batch import run_video, serial_submit

    settings = Settings(mask=mask, zone=False, copy_clean=False)
    return run_video(src, dst, settings, serial_submit), settings


@pytest.fixture(scope="module")
def masked(plate_clip, tmp_path_factory):
    out = tmp_path_factory.mktemp("masked") / "plate_masked.mp4"
    record, settings = run(plate_clip, out, ("plate",))
    return out, record, settings


@pytest.fixture(scope="module")
def unmasked(plate_clip, tmp_path_factory):
    """The same clip through a run that only wants faces: nothing is masked."""
    out = tmp_path_factory.mktemp("unmasked") / "plate_faces_only.mp4"
    record, settings = run(plate_clip, out, ("face",))
    return out, record, settings


def test_pasted_plates_are_filled_solid_and_the_grey_around_them_is_left(plate_clip, masked):
    from faceblur.pipeline import STATUS_DONE

    out, record, _ = masked
    assert record.status == STATUS_DONE, record.error
    assert record.plates > 0, "the detector found the plates"
    assert record.plate_frames >= 2, "and the hold kept them"
    assert "plate" in record.masked_mean_by_kind

    _, plate = pasted()
    # The source clip as decoded, which is what the pipeline read.
    source = decode_all(plate_clip)[2]
    copy = decode_all(out)
    assert len(copy) == 4
    frame = copy[2].astype(np.int16)
    filled = (np.abs(frame) <= 8).all(axis=2)
    share = filled[plate].mean()
    assert share >= 0.95, f"only {share:.3%} of the plates is under the solid fill"

    pasted_box = np.zeros(SIZE, np.uint8)
    for px, py in POSITIONS:
        pasted_box[py:py + PASTE_H, px:px + PASTE_W] = 1
    far = cv2.distanceTransform(1 - pasted_box, cv2.DIST_L2, 5) > 40
    diff = np.abs(frame - source.astype(np.int16)).mean(axis=2)
    assert diff[far].mean() < 3, f"pixels far from the plates moved by {diff[far].mean():.2f}"


def test_the_check_finds_no_plate_left_in_a_masked_copy(plate_clip, masked):
    from faceblur.verify import check, held_for

    out, _, settings = masked
    report = check(plate_clip, out, settings)
    assert report["residual_plates"] == 0
    assert report["residual_plate_runs"] == []
    assert held_for(report, settings) == ()


def test_the_check_holds_back_a_copy_that_still_shows_a_plate(plate_clip, unmasked):
    from faceblur.verify import check, describe, held_for

    out, record, settings = unmasked
    assert record.plates == 0, "a run that did not want plates never looked for them"
    wants_plates = settings.with_changes(mask=("plate",))
    report = check(plate_clip, out, wants_plates)
    assert report["residual_plates"] > 0
    assert report["residual_plate_max_px"] >= 24
    assert all(row["label"] == "lpd_yunet" for row in report["residual_list"]
               if row["kind"] == "plate")
    assert held_for(report, wants_plates) == ("plate",)
    assert "plates still visible" in describe(report, wants_plates)


# ------------------------------------------------------- solid, and the zone

QUAD = ((100.0, 80.0), (260.0, 92.0), (256.0, 140.0), (96.0, 128.0))


def a_plate() -> Detection:
    xs, ys = [p[0] for p in QUAD], [p[1] for p in QUAD]
    return Detection(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), 0.99, None,
                     "lpd_yunet", True, 0.99, "plate", QUAD)


def inside_quad(shape, erode: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.array(QUAD)).astype(np.int32), 1)
    if erode:
        mask = cv2.erode(mask, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return mask.astype(bool)


@pytest.mark.parametrize("mode", ["blur", "pixelate"])
def test_a_plate_is_filled_solid_whatever_the_mode(mode):
    frame = np.random.default_rng(7).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    out, _ = redact(frame, [a_plate()], Settings(mode=mode))
    deep = inside_quad(frame.shape[:2], erode=4)
    assert deep.sum() > 1000
    assert (out[deep] == 0).all(), f"a plate in {mode} mode is not a solid fill"


def test_a_plate_the_wearer_is_handling_is_left_alone():
    frame = np.random.default_rng(8).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    around = ((60.0, 40.0), (320.0, 40.0), (320.0, 180.0), (60.0, 180.0))
    out, _ = redact(frame, [a_plate()], Settings(), zone=ZoneFrame(live=(around,)))
    inside = inside_quad(frame.shape[:2], erode=0)
    assert (out[inside] == frame[inside]).all(), "the zone wins over a plate"


# ------------------------------------------------------------------ the switch

PLATE_SETTINGS = ("plate_size", "plate_conf", "plate_nms", "plate_min_px", "plate_max_frac",
                  "plate_pad", "plate_min_run", "plate_gap", "plate_tail", "plate_link_iou",
                  "plate_gate_min_px", "plate_gate_min_run", "plate_device")


def test_plates_can_be_asked_for_on_every_surface():
    import cli

    recorded = Settings().to_dict()
    for name in PLATE_SETTINGS:
        assert name in recorded, name
    assert recorded["plate_size"] == 1280 and recorded["plate_conf"] == 0.8
    assert recorded["plate_max_frac"] == 0.25
    with pytest.raises(SettingsError):
        Settings(plate_conf=1.5)
    with pytest.raises(SettingsError):
        Settings(plate_device="tpu")
    assert classes.parse("plate,face") == ("face", "plate")
    args = cli.build_parser().parse_args(["x", "--mask", "plate"])
    assert classes.parse(args.mask) == ("plate",)
    assert Settings(mask=("plate",)).wants("plate")


# ------------------------------------------------------------- the aspect rule

class _Pool:
    outputs = ("loc", "conf", "iou")

    def run(self, blob):
        return (None, None, None)


def _found_for(monkeypatch, quad_wh: tuple[float, float]) -> list[Detection]:
    """What `detect` keeps of one quad of width x height at (300, 300) in a 1600x1300 frame."""
    import faceblur.plates as plates

    settings = Settings(mask=("plate",))
    detector = object.__new__(PlateDetector)
    detector.settings, detector.pool = settings, _Pool()
    scale = settings.plate_size / 1600
    w, h = quad_wh
    corners = np.array([[[300, 300], [300 + w, 300], [300 + w, 300 + h], [300, 300 + h]]],
                       np.float32) * scale
    monkeypatch.setattr(plates, "decode",
                        lambda *a: (np.array([0.95], np.float32), corners))
    return detector.detect(np.zeros((1300, 1600, 3), np.uint8))


def test_a_wide_plate_is_kept(monkeypatch):
    assert len(_found_for(monkeypatch, (200, 50))) == 1


def test_a_portrait_box_is_not_a_plate(monkeypatch):
    # The pilot's false plates: 166x282 boxes on bins.
    assert _found_for(monkeypatch, (166, 282)) == []


def test_plate_aspect_max_is_a_setting():
    assert Settings().to_dict()["plate_aspect_max"] == 0.8
    with pytest.raises(SettingsError):
        Settings(plate_aspect_max=0)
