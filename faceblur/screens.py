"""Screens as objects: a television, a monitor, a laptop, a phone in a hand.

The face side of this project asks "who is this", and answers it with three
detectors that have to agree. A screen needs none of that. It is a large,
rigid, high-contrast rectangle, one general-purpose object detector finds it,
and there is no identity to weigh: whatever is on it is hidden, so the only
question is where the glass is.

The model is YOLOX-tiny, Apache 2.0, from Megvii's own release. It was chosen
over the alternatives on licence first: most YOLO derivatives in common use
are AGPL, which this project cannot ship. Of what is left it is the smallest
that decodes the way YuNet already does in `detect.py`, which means the
decode here could be checked against a frame by eye rather than trusted.

It is a COCO detector, so it knows 80 things and this module wants three of
them: a television, a laptop and a phone. `SCREEN_LABELS` is that list and
nothing else in the file hard-codes it.

What this does not do is decide whether a screen is worth hiding. A monitor
showing a spreadsheet of names and a monitor showing a scoreboard are the
same object to a detector. The scope decision of 2026-09-10 is that a screen
is masked as an object, whatever is on it, so that judgement never has to be
made; the cost is that a screen showing nothing private is masked too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .detect import MODEL_DIR, Detection, _require, nms_indices, providers_for
from .settings import Settings

SCREEN_MODEL = MODEL_DIR / "yolox_tiny.onnx"

# The graph takes one 416x416 BGR image, 0 to 255, NCHW, with no mean or
# standard deviation applied. Normalising it the ImageNet way drops the best
# score on a frame with a television from 0.85 to 0.004, which is the check
# that this is right.
INPUT_SIZE = 416
# Anchor free, three feature maps. 52*52 + 26*26 + 13*13 = 3549, which is the
# length of the model's output and so the check that these are the strides.
STRIDES = (8, 16, 32)
# YOLOX pads the short side of the letterbox with this, not with black.
PAD_VALUE = 114

# The COCO classes that are a screen, by index. This model knows 80 things.
SCREEN_LABELS: dict[int, str] = {62: "tv", 63: "laptop", 67: "phone"}


class ScreenModelMissing(FileNotFoundError):
    """The screen model is not on disk, so screens cannot be masked."""


@dataclass(frozen=True)
class _Grid:
    """Cell centres and strides, one row per output box."""

    gx: np.ndarray
    gy: np.ndarray
    stride: np.ndarray


def _build_grid(size: int = INPUT_SIZE) -> _Grid:
    gx, gy, st = [], [], []
    for s in STRIDES:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        gx.append(xv.reshape(-1))
        gy.append(yv.reshape(-1))
        st.append(np.full(n * n, s))
    return _Grid(np.concatenate(gx).astype(np.float32),
                 np.concatenate(gy).astype(np.float32),
                 np.concatenate(st).astype(np.float32))


GRID = _build_grid()


def letterbox(frame: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """Fit the frame into the square input at the top left, on grey.

    Top left, not centred, because that is what YOLOX's own preprocessing
    does and the offset has to match or every box lands short.
    """
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r


def decode(raw: np.ndarray, ratio: float, conf: float, nms: float,
           labels: dict[int, str]) -> list[Detection]:
    """Model output to screens in source frame pixels.

    Each row is four box numbers, an objectness, then one score per class.
    The box numbers are an offset from the cell and a log of the size in
    strides, the same shape of decode as YuNet's.
    """
    if raw.ndim == 3:
        raw = raw[0]
    scores = raw[:, 4:5] * raw[:, 5:]
    wanted = sorted(labels)
    best = scores[:, wanted]
    which = best.argmax(axis=1)
    score = best.max(axis=1)
    keep = np.where(score >= conf)[0]
    if keep.size == 0:
        return []
    cx = (raw[keep, 0] + GRID.gx[keep]) * GRID.stride[keep]
    cy = (raw[keep, 1] + GRID.gy[keep]) * GRID.stride[keep]
    bw = np.exp(raw[keep, 2]) * GRID.stride[keep]
    bh = np.exp(raw[keep, 3]) * GRID.stride[keep]
    boxes = np.stack([(cx - bw / 2) / ratio, (cy - bh / 2) / ratio,
                      bw / ratio, bh / ratio], axis=1).astype(np.float64)
    picked = score[keep].astype(np.float64)
    out = []
    # One suppression per class: a laptop and the phone lying on it are two
    # things, and a single pass over both would drop the smaller.
    for index, name in ((i, labels[c]) for i, c in enumerate(wanted)):
        rows = np.where(which[keep] == index)[0]
        if rows.size == 0:
            continue
        for j in nms_indices(boxes[rows], picked[rows], nms):
            row = rows[j]
            out.append(Detection(float(boxes[row, 0]), float(boxes[row, 1]),
                                 float(boxes[row, 2]), float(boxes[row, 3]),
                                 float(picked[row]), None, name, True,
                                 float(picked[row]), "screen"))
    return out


def plausible(det: Detection, shape: tuple[int, int], settings: Settings) -> bool:
    """Is a box this large a screen, or a table the detector mistook for one.

    A COCO detector calls any large flat rectangle a television, and this
    footage is made of them: a blue table tennis table reads as a laptop at
    0.88 covering a third of the frame, a washroom mirror as a television, a
    glass wall as both. Nothing in the score separates those from a real
    screen, because a real television 1 percent of the frame across the hall
    also scores 0.85. What separates them here is size, and only size.

    The cap is measured, not guessed, and it is not free: section 15.3 of
    docs/report.md has it losing two real screens of fifty four, both of them
    large close monitors. That is the wrong place to lose one, and it is the
    honest state of a COCO detector on this footage.
    """
    h, w = shape
    return det.w * det.h <= settings.screen_max_area * w * h


class ScreenDetector:
    """YOLOX-tiny through onnxruntime, one frame at a time.

    Fixed shape, batch of one: the frame is already decoded for the face
    pass, this is one more inference on it at a sixteenth of the area, and a
    fixed graph is what DirectML wants.
    """

    name = "yolox"

    def __init__(self, settings: Settings, threads: Optional[int] = None):
        import onnxruntime as ort

        if not SCREEN_MODEL.is_file():
            raise ScreenModelMissing(
                f"Could not find the screen model {SCREEN_MODEL}. Reinstall FaceBlur, "
                f"or run without screens in --mask.")
        ort.set_default_logger_severity(3)
        self.settings = settings
        self.model_path = _require(SCREEN_MODEL)
        self.labels = {i: n for i, n in SCREEN_LABELS.items()
                       if n in settings.screen_labels}
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), options,
                                            providers=providers_for(settings.device))
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    @property
    def compute(self) -> dict[str, str]:
        return {self.name: self.provider}

    @property
    def model_hashes(self) -> dict[str, str]:
        from .detect import model_sha256
        return {self.name: model_sha256(self.model_path)}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        canvas, ratio = letterbox(frame)
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]
        raw = self.session.run(None, {self.input_name: blob})[0]
        found = decode(raw, ratio, self.settings.screen_conf,
                       self.settings.screen_nms, self.labels)
        h, w = frame.shape[:2]
        found = [d for d in found if plausible(d, (h, w), self.settings)]
        return [self._pad(d, (h, w)) for d in found]

    def _pad(self, det: Detection, shape: tuple[int, int]) -> Detection:
        """Grow the box by `screen_pad` of its own size, clipped to the frame.

        A detector's box stops at the glass. The bezel is not the thing being
        hidden, but a box that stops exactly at the glass leaves a rim of
        picture when the camera moves between the frame it was found in and
        the frame it is masked in.
        """
        from dataclasses import replace

        pad = self.settings.screen_pad
        if pad <= 0:
            return det
        h, w = shape
        x = max(0.0, det.x - det.w * pad / 2)
        y = max(0.0, det.y - det.h * pad / 2)
        x1 = min(float(w), det.x + det.w * (1 + pad / 2))
        y1 = min(float(h), det.y + det.h * (1 + pad / 2))
        return replace(det, x=x, y=y, w=max(1.0, x1 - x), h=max(1.0, y1 - y))


def _iou(a: Detection, b: Detection) -> float:
    from .detect import iou
    return iou(a, b)


@dataclass
class Run:
    """One screen, seen over a stretch of frames."""

    label: str
    hits: list          # (frame, Detection), in frame order

    @property
    def last(self) -> int:
        return self.hits[-1][0]

    @property
    def last_box(self) -> Detection:
        return self.hits[-1][1]


def runs_in(per_frame: dict[int, list[Detection]], settings: Settings) -> list[Run]:
    """Group per frame boxes into runs: same label, overlapping, close in time.

    Deliberately simple beside `track.py`. There is no identity to confirm,
    no second detector to agree with and no hand to keep out, so a run is
    just the same rectangle seen again nearby: link each box to the open run
    of its own label whose last box it overlaps most.
    """
    open_runs: list[Run] = []
    done: list[Run] = []
    for i in sorted(per_frame):
        for det in per_frame[i]:
            best, best_iou = None, 0.0
            for run in open_runs:
                if run.label != det.source or i - run.last - 1 > settings.screen_gap:
                    continue
                overlap = _iou(run.last_box, det)
                if overlap > best_iou:
                    best, best_iou = run, overlap
            if best is not None and best_iou >= settings.track_iou:
                best.hits.append((i, det))
            else:
                open_runs.append(Run(det.source, [(i, det)]))
        still = [r for r in open_runs if i - r.last - 1 <= settings.screen_gap]
        done += [r for r in open_runs if i - r.last - 1 > settings.screen_gap]
        open_runs = still
    return done + open_runs


def hold(per_frame: dict[int, list[Detection]], settings: Settings,
         frames: int) -> dict[int, list[Detection]]:
    """Drop the flickers, close short gaps, and hold a screen past its last
    sighting.

    Three things, in this order, and the first is the one that matters.

    **A screen has to be seen `screen_min_run` times before any of it is
    masked.** A table only looks like a laptop from some angles, so the
    detector calls it one for a frame and then stops; a monitor on a wall
    stays a monitor. On the sample footage 14 of the 29 false runs last a
    single frame and not one real screen does. This is the same rule as
    `established_after` on the face side, and it is what lets the size cap
    sit at 12 percent instead of 4 and so keep the large close monitors that
    matter most.

    **Then the gaps inside a surviving run are filled**, because a screen
    does not leave the room between one frame and the next, and a detector
    that drops it for two frames has not been told that. The filled box is
    the later of the two: a screen that moved across a gap moved because the
    camera did, and where it ended up is the better guess.

    **Then the last sighting runs on for `screen_tail` frames**, for the same
    reason the face masks have a tail.
    """
    kept: dict[int, list[Detection]] = {}

    def add(i: int, det: Detection) -> None:
        if 0 <= i < frames:
            kept.setdefault(i, []).append(det)

    for run in runs_in(per_frame, settings):
        if len(run.hits) < settings.screen_min_run:
            continue
        for (a, _), (b, det_b) in zip(run.hits, run.hits[1:]):
            for i in range(a, b):
                add(i, run.hits[0][1] if i == a and a == run.hits[0][0] else det_b)
        first_frame, first_det = run.hits[0]
        add(first_frame, first_det)
        last_frame, last_det = run.hits[-1]
        for i in range(last_frame, last_frame + settings.screen_tail + 1):
            add(i, last_det)
    # One box per place per frame: the fill and the tail can both reach the
    # same frame from either side of a gap.
    out: dict[int, list[Detection]] = {}
    for i, dets in kept.items():
        unique: list[Detection] = []
        for det in dets:
            if not any(d.source == det.source and _iou(d, det) >= settings.track_iou
                       for d in unique):
                unique.append(det)
        out[i] = unique
    return out
