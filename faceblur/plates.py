"""Licence plates: a vehicle registration plate, found and filled solid.

A plate is the shortest string this project hides and the easiest to read
back. It is not a face, there is nothing to confirm, and like a line of text
it is hidden by the four corners it sits in rather than by its upright box,
because a plate photographed from the side is a parallelogram and its box
takes in the bumper. `redact.fill_mode` fills it solid whatever the run's mode
(decision D-C): a blurred plate keeps its length and its character breaks.

**The model** is LPD-YuNet from OpenCV's model zoo, Apache 2.0, with its input
freed by `models/make_plates.py` so it runs at any size. `models/README.md`
documents the layout and `tests/test_plate_model.py` pins it; the priors and
the decode below are the ones those two describe, and live here so that there
is one copy of them.

**Size.** The zoo runs the graph at the image's own size. The camera's frames
are 1600x1300 and this resizes them so the long side is `plate_size` (1280 by
default), then pads them below and to the right to a multiple of 32. The
model finds a plate once it is about 120 px wide at its input, so this misses
plates under about 150 px wide in the frame; a bigger size finds smaller
plates and more false ones, and costs more. `docs/privacy_module_report.md`
has the measurement and the reason for 1280.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Optional

import cv2
import numpy as np

from .detect import MODEL_DIR, Detection, SessionPool, _require, nms_indices
from .settings import Settings

PLATE_MODEL = MODEL_DIR / "lpd_yunet_dynamic.onnx"

# From the zoo's lpd_yunet.py: anchor sizes per level, the level strides, and
# the SSD variance the offsets are scaled by.
MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
STEPS = [8, 16, 32, 64]
VARIANCE = 0.1
# The padded input is a multiple of this, so every level's map is whole down
# to stride 32 and a frame of one size always makes one graph on DirectML.
STRIDE = 32


class PlateModelMissing(FileNotFoundError):
    """The plate model is not on disk. Raised where it can be reported."""


def _maps(width: int, height: int) -> list[tuple[int, int]]:
    """The feature map sizes per level, the way the zoo's `_priorGen` computes them."""
    h, w = (height + 1) // 2 // 2, (width + 1) // 2 // 2
    maps = []
    for _ in STEPS:
        h, w = h // 2, w // 2
        maps.append((h, w))
    return maps


def prior_count(width: int, height: int) -> int:
    """How many rows the graph returns for this input size."""
    return sum(fh * fw * len(sizes) for (fh, fw), sizes in zip(_maps(width, height), MIN_SIZES))


def priors(width: int, height: int) -> np.ndarray:
    """[N, 4] prior boxes, cx, cy, w, h normalised, in the order the graph's rows are."""
    rows = []
    for (fh, fw), sizes, step in zip(_maps(width, height), MIN_SIZES, STEPS):
        for i, j in product(range(fh), range(fw)):
            for size in sizes:
                rows.append([(j + 0.5) * step / width, (i + 0.5) * step / height,
                             size / width, size / height])
    return np.array(rows, np.float32).reshape(-1, 4)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR, 0 to 255 as float32, NCHW, no mean and no standard deviation."""
    return bgr.astype(np.float32).transpose(2, 0, 1)[None]


def decode(loc, conf, iou, width: int, height: int):
    """Scores [N] and corners [N, 4, 2] in input pixels.

    The corners come in the order top left, top right, bottom right, bottom
    left of the plate as written.
    """
    scores = np.sqrt(conf[:, 1] * np.clip(iou[:, 0], 0.0, 1.0))
    p = priors(width, height)
    corners = np.stack([p[:, 0:2] + loc[:, k:k + 2] * VARIANCE * p[:, 2:4]
                        for k in (4, 6, 10, 12)], axis=1)
    return scores, corners * [width, height]


def _output_dims(n: int, h: int, w: int) -> dict:
    """The row count is the number of priors for the input size."""
    rows = prior_count(w, h)
    return {"loc": [rows, 14], "conf": [rows, 2], "iou": [rows, 1]}


def canvas_for(frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """The frame scaled to `size` on its long side, padded to a stride.

    Black padding below and to the right, so a coordinate in the canvas
    divided by the scale is a coordinate in the frame.
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    canvas = np.zeros(((nh + STRIDE - 1) // STRIDE * STRIDE,
                       (nw + STRIDE - 1) // STRIDE * STRIDE, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


class PlateDetector:
    """LPD-YuNet through onnxruntime, one frame at a time."""

    name = "lpd_yunet"

    def __init__(self, settings: Settings, threads: Optional[int] = None):
        if not PLATE_MODEL.is_file():
            raise PlateModelMissing(
                f"Could not find the plate model {PLATE_MODEL}. Reinstall FaceBlur, "
                f"or run without plate in --mask.")
        self.settings = settings
        self.model_path = _require(PLATE_MODEL)
        # A static graph per input shape on DirectML, as for the text model.
        # A video has one frame size, so the pool holds one session.
        self.pool = SessionPool(self.model_path, settings.plate_device, threads, "input",
                                _output_dims)

    @property
    def provider(self) -> str:
        return self.pool.provider

    @property
    def compute(self) -> dict[str, str]:
        return {self.name: self.pool.provider}

    @property
    def model_hashes(self) -> dict[str, str]:
        from .detect import model_sha256
        return {self.name: model_sha256(self.model_path)}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        s = self.settings
        h, w = frame.shape[:2]
        canvas, scale = canvas_for(frame, s.plate_size)
        ch, cw = canvas.shape[:2]
        outputs = dict(zip(self.pool.outputs, self.pool.run(preprocess(canvas))))
        scores, corners = decode(outputs["loc"], outputs["conf"], outputs["iou"], cw, ch)
        keep = scores >= s.plate_conf
        scores, corners = scores[keep], corners[keep] / scale
        if not len(scores):
            return []
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        boxes = np.concatenate([lo, hi - lo], axis=1).astype(np.float32)
        found = []
        for i in nms_indices(boxes, scores.astype(np.float32), s.plate_nms):
            x, y, bw, bh = (float(v) for v in boxes[i])
            # Too small to read, or wider than a plate in this frame could be:
            # a band across a wall or a mirror, not a registration number.
            if max(bw, bh) < s.plate_min_px or bw > s.plate_max_frac * w:
                continue
            score = round(float(scores[i]), 4)
            quad = tuple((float(px), float(py)) for px, py in corners[i])
            found.append(self._pad(Detection(x, y, bw, bh, score, None, self.name, True,
                                             score, "plate", quad), (h, w)))
        return found

    def _pad(self, det: Detection, shape: tuple[int, int]) -> Detection:
        """Grow the quad by `plate_pad` of its own size, clipped to the frame.

        The corners stop at the plate's edge, and a plate masked on a frame
        next to the one it was found on has moved by a few pixels; the screws
        and the frame of the plate are not the number, but a rim of it is.
        """
        pad = self.settings.plate_pad
        if pad <= 0:
            return det
        h, w = shape
        q = np.asarray(det.quad, np.float64)
        c = q.mean(axis=0)
        q = c + (q - c) * (1 + pad)
        q[:, 0] = q[:, 0].clip(0, w - 1)
        q[:, 1] = q[:, 1].clip(0, h - 1)
        x, y = float(q[:, 0].min()), float(q[:, 1].min())
        return replace(det, x=x, y=y, w=max(1.0, float(q[:, 0].max()) - x),
                       h=max(1.0, float(q[:, 1].max()) - y),
                       quad=tuple((float(px), float(py)) for px, py in q))


def hold(per_frame: dict[int, list[Detection]], settings: Settings,
         frames: int) -> dict[int, list[Detection]]:
    """Plates over time, on the pattern of `text.hold`.

    A parked car does not blink and the detector does, so a plate is linked
    frame to frame by the overlap of its box, short gaps are closed, and a
    run carries a tail. `plate_min_run` drops what was seen once.
    """
    from .screens import hold as hold_screens

    return hold_screens(per_frame, settings.with_changes(
        track_iou=settings.plate_link_iou, screen_gap=settings.plate_gap,
        screen_tail=settings.plate_tail, screen_min_run=settings.plate_min_run), frames)
