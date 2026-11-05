"""T06a: the licence plate detector is the pinned file and finds a plate.

`models/lpd_yunet_dynamic.onnx` is opencv_zoo's LPD-YuNet with its input
freed by `models/make_plates.py`. The preprocessing and decode here are the
ones `models/README.md` documents, written out in full so that
`faceblur/plates.py` can be checked against them: the frame at its own size,
BGR, 0 to 255 as float32, NCHW, no mean and no standard deviation; then a
score of sqrt(conf[:, 1] * clip(iou, 0, 1)) per prior and four corners from
loc[:, 4:8] and loc[:, 10:14] against the prior boxes the zoo's
`lpd_yunet.py` builds.
"""
import hashlib
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "lpd_yunet_dynamic.onnx"
DATA = ROOT / "tests" / "data"
PHOTO = DATA / "plate_cc0.jpg"

SHA256 = "ec61f09928cb61510b7f23ce686cf30969f857817fcb779bec03da608c3121d8"
SIZE = 4143560

# Hand-measured in tests/data/PLATE_IMAGES.md, x, y, w, h.
PLATE_XYWH = (136, 411, 182, 64)

THRESHOLD = 0.8     # the zoo's own default

# From the zoo's lpd_yunet.py: anchor sizes per level, the level strides, and
# the SSD variance the offsets are scaled by.
MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
STEPS = [8, 16, 32, 64]
VARIANCE = 0.1


def priors(width: int, height: int) -> np.ndarray:
    """[N, 4] prior boxes, cx, cy, w, h normalised, in the order the graph's rows are."""
    h, w = (height + 1) // 2 // 2, (width + 1) // 2 // 2
    maps = []
    for _ in STEPS:
        h, w = h // 2, w // 2
        maps.append((h, w))
    rows = []
    for (fh, fw), sizes, step in zip(maps, MIN_SIZES, STEPS):
        for i, j in product(range(fh), range(fw)):
            for size in sizes:
                rows.append([(j + 0.5) * step / width, (i + 0.5) * step / height,
                             size / width, size / height])
    return np.array(rows, np.float32)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    return bgr.astype(np.float32).transpose(2, 0, 1)[None]


def decode(loc, conf, iou, width: int, height: int):
    """Scores [N] and corners [N, 4, 2] in frame pixels.

    The corners come in the order top left, top right, bottom right, bottom
    left of the plate as written.
    """
    scores = np.sqrt(conf[:, 1] * np.clip(iou[:, 0], 0.0, 1.0))
    p = priors(width, height)
    corners = np.stack([p[:, 0:2] + loc[:, k:k + 2] * VARIANCE * p[:, 2:4]
                        for k in (4, 6, 10, 12)], axis=1)
    return scores, corners * [width, height]


def detect(session: ort.InferenceSession, bgr: np.ndarray):
    loc, conf, iou = session.run(["loc", "conf", "iou"], {"input": preprocess(bgr)})
    return decode(loc, conf, iou, bgr.shape[1], bgr.shape[0])


def iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


@pytest.fixture(scope="module")
def cpu():
    return ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


def test_the_model_is_the_pinned_file():
    data = MODEL.read_bytes()
    assert len(data) == SIZE
    assert hashlib.sha256(data).hexdigest() == SHA256


def test_the_model_declares_the_documented_layout(cpu):
    (inp,) = cpu.get_inputs()
    assert (inp.name, inp.shape, inp.type) == ("input", [1, 3, "H", "W"], "tensor(float)")
    assert [(o.name, o.shape) for o in cpu.get_outputs()] == [
        ("loc", ["N", 14]), ("conf", ["N", 2]), ("iou", ["N", 1])]


def test_the_rows_match_the_priors(cpu):
    """The decode is only right if row i of the graph is prior i."""
    for width, height in [(320, 240), (1024, 676), (1920, 1080)]:
        loc, _, _ = cpu.run(["loc", "conf", "iou"],
                            {"input": np.zeros((1, 3, height, width), np.float32)})
        assert len(loc) == len(priors(width, height))


def test_the_top_box_is_the_plate(cpu):
    scores, corners = detect(cpu, cv2.imread(str(PHOTO)))
    top = int(np.argmax(scores))
    assert scores[top] >= THRESHOLD
    quad = corners[top]
    assert quad.shape == (4, 2)
    x, y, w, h = PLATE_XYWH
    rect = (*quad.min(axis=0), *quad.max(axis=0))
    assert iou_xyxy(rect, (x, y, x + w, y + h)) >= 0.5


def test_the_corners_go_round_the_plate(cpu):
    """Four distinct points, top left, top right, bottom right, bottom left."""
    scores, corners = detect(cpu, cv2.imread(str(PHOTO)))
    tl, tr, br, bl = corners[int(np.argmax(scores))]
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]
    # The plate is tilted: its right end sits lower than its left.
    assert tr[1] > tl[1]


def negatives():
    rng = np.random.default_rng(0)
    yield "grey", np.full((480, 640, 3), 128, np.uint8)
    yield "noise", rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    yield "lena", cv2.imread(str(DATA / "lena.jpg"))
    yield "messi5", cv2.imread(str(DATA / "messi5.jpg"))
    for level in [*range(0, 256, 32), 255]:
        yield f"flat {level}", np.full((480, 640, 3), level, np.uint8)


@pytest.mark.parametrize("name,frame", list(negatives()), ids=lambda v: v if isinstance(v, str) else "")
def test_no_plate_where_there_is_none(cpu, name, frame):
    scores, _ = detect(cpu, frame)
    assert scores.max() < THRESHOLD, name


def test_directml_agrees_with_the_cpu(cpu):
    if "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("DirectML is not available here")
    try:
        dml = ort.InferenceSession(str(MODEL), providers=["DmlExecutionProvider"])
    except Exception as error:  # a driver that refuses the graph is a skip, not a failure
        pytest.skip(f"DirectML refused the model: {error}")
    frame = cv2.imread(str(PHOTO))
    cpu_scores, cpu_corners = detect(cpu, frame)
    dml_scores, dml_corners = detect(dml, frame)
    assert np.abs(dml_corners[np.argmax(dml_scores)] - cpu_corners[np.argmax(cpu_scores)]).max() <= 2
