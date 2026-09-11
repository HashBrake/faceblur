"""Text as a shape on a surface: a sign, a label, a screen's caption, a badge.

The face side of this project asks "who is this" and answers it with three
detectors that have to agree, because a false face is a person's head wiped
out of a frame. Text needs none of that agreement and it is not the same
question. Under the rule of 2026-09-11 a line of text outside the handled
zone carries information the collector is not handling, so it goes, and what
it says never has to be read to decide that. This module finds where the
words are. Nothing here reads them, and package T4 is the only thing that
ever would.

**The model** is PP-OCRv3 detection, Apache 2.0, from PaddlePaddle's own
release, converted to ONNX by OpenCV's model zoo. That is the same zoo
`yunet.onnx` comes from, which matters: the build plan asked for the file to
be converted here with `paddle2onnx` rather than taken pre converted from an
unknown repository, and a zoo this project already depends on is not that. A
conversion done here would also have a hash only this machine could produce,
where the zoo's file has a URL and a sha256 anybody can check. The deviation
and the reason are in `models/README.md` and in `STATE.md`.

It is a DB model: it returns one probability map at the size of the input,
and the boxes are contours of that map. Three numbers turn a map into text:
`text_thresh` is where the map counts as ink, `text_box_thresh` is the mean
probability a contour needs to survive, and `text_unclip` grows the box back
out, because the map is trained to shrink each text region away from its
neighbours and the raw contour cuts the letters off.

**Quads, not boxes.** A line of text on a sign photographed from the side is
a rotated, sheared rectangle, and its upright bounding box takes in most of
the wall. `redact.build_alpha` masks a POLYGON kind by the four corners it is
given, so this returns `cv2.minAreaRect`'s corners and not the box around
them.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detect import MODEL_DIR, Detection, SessionPool, _require, nms_indices
from .settings import Settings

TEXT_MODEL = MODEL_DIR / "text_detection_ppocrv3.onnx"

# The graph takes one NCHW image of floats, plain 0 to 1. The zoo's own
# wrapper applies the ImageNet mean and standard deviation instead, and on a
# drawn card of black words the two are level. On real frames they are not:
# plain finds more of the map above threshold at every scale, and at one
# scale on one frame the ImageNet path returns an empty map, which a
# preprocessing that matched the export would not do. Report 21.1 has the
# table and `eval.text --normalisation` prints it.
SCALE = 1.0 / 255.0
# DB's map is the size of its input and the model strides by 32.
STRIDE = 32


class TextModelMissing(FileNotFoundError):
    """The text model is not on disk. Raised where it can be reported."""


def _output_dims(n: int, h: int, w: int) -> dict:
    """The map is the size of the input, one channel and no batch."""
    return {"4": [h, w]}


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """The frame scaled to `size` on its long side, padded to a stride.

    Aspect preserved, because a text line squeezed into a square stops being
    the shape the model was trained on. The padding is black and sits below
    and to the right, so a coordinate in the canvas divided by the scale is a
    coordinate in the frame.
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    canvas = np.zeros(((nh + STRIDE - 1) // STRIDE * STRIDE,
                       (nw + STRIDE - 1) // STRIDE * STRIDE, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


def unclipped(rect, ratio: float):
    """`cv2.minAreaRect` grown outward the way DB's own postprocessing does.

    DB is trained on regions shrunk away from their neighbours, so a contour
    of the map stops inside the letters. The published rule offsets the
    polygon by `area * ratio / perimeter`. For a rectangle that offset is
    exactly a rectangle with both sides longer by twice the distance, so this
    needs no polygon offsetting library to be right.
    """
    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return rect
    distance = (w * h * ratio) / (2 * (w + h))
    return ((cx, cy), (w + 2 * distance, h + 2 * distance), angle)


def box_score(prob: np.ndarray, quad: np.ndarray) -> float:
    """The mean probability inside a quad, which is what decides it.

    The peak would call a whole wall text on one bright pixel. The mean over
    the quad is what DB's own postprocessing reads and it is the number
    `text_box_thresh` compares against.
    """
    h, w = prob.shape[:2]
    x0 = max(0, min(w - 1, int(np.floor(quad[:, 0].min()))))
    x1 = max(0, min(w - 1, int(np.ceil(quad[:, 0].max()))))
    y0 = max(0, min(h - 1, int(np.floor(quad[:, 1].min()))))
    y1 = max(0, min(h - 1, int(np.ceil(quad[:, 1].max()))))
    if x1 < x0 or y1 < y0:
        return 0.0
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    shifted = quad.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    cv2.fillPoly(mask, [shifted.round().astype(np.int32)], 1)
    if not mask.any():
        return 0.0
    return float(cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], mask)[0])


def quads_from(prob: np.ndarray, scale: float, shape: tuple[int, int],
               settings: Settings) -> list[Detection]:
    """Every text quad in one probability map, in frame coordinates."""
    h, w = shape
    ink = (prob >= settings.text_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Detection] = []
    for contour in contours:
        if len(contour) < 4:
            continue
        rect = cv2.minAreaRect(contour)
        score = box_score(prob, cv2.boxPoints(rect))
        if score < settings.text_box_thresh:
            continue
        quad = cv2.boxPoints(unclipped(rect, settings.text_unclip)) / scale
        quad[:, 0] = quad[:, 0].clip(0, w - 1)
        quad[:, 1] = quad[:, 1].clip(0, h - 1)
        side = min(cv2.minAreaRect(quad.astype(np.float32))[1])
        if side < settings.text_min_px:
            continue
        xs, ys = quad[:, 0], quad[:, 1]
        x, y = float(xs.min()), float(ys.min())
        bw, bh = float(xs.max() - x), float(ys.max() - y)
        if bw < 1 or bh < 1:
            continue
        found.append(Detection(x, y, bw, bh, round(score, 4), None, "ppocr", True,
                               round(score, 4), "text",
                               tuple((float(px), float(py)) for px, py in quad)))
    return found


def is_page(det: Detection, shape: tuple[int, int], settings: Settings) -> bool:
    """A quad over `text_max_frac` of the frame: a page, a poster, a menu.

    Kept rather than dropped. The rule is about information and a page of it
    is more, not less, but it is counted apart because one of these covers a
    third of a frame on its own and would otherwise read as the detector
    failing.
    """
    h, w = shape
    return (det.w * det.h) >= settings.text_max_frac * h * w


def merged(found: list[Detection], settings: Settings) -> list[Detection]:
    """One list from several scales: union, then NMS on the bounding boxes.

    The quads are not compared directly. Two scales find the same line as two
    slightly different parallelograms and any overlap measure between them is
    a judgement about which is better; the boxes around them answer the only
    question being asked here, which is whether these are the same line.
    """
    if not found:
        return []
    boxes = np.array([[d.x, d.y, d.w, d.h] for d in found], np.float32)
    scores = np.array([d.score for d in found], np.float32)
    return [found[i] for i in nms_indices(boxes, scores, settings.text_nms)]


class TextDetector:
    """PP-OCRv3 detection through onnxruntime, one frame at a time.

    Two scales by default. A sign across a room and a label in the wearer's
    hand are two orders of magnitude apart in pixels and one input size
    cannot have both.
    """

    name = "ppocr"

    def __init__(self, settings: Settings, threads: Optional[int] = None):
        if not TEXT_MODEL.is_file():
            raise TextModelMissing(
                f"Could not find the text model {TEXT_MODEL}. Reinstall FaceBlur, "
                f"or run without text in --mask.")
        self.settings = settings
        self.model_path = _require(TEXT_MODEL)
        self.pool = SessionPool(self.model_path, settings.device, threads, "x",
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

    def map_for(self, frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
        """The probability map at one scale, and the scale that made it."""
        canvas, scale = letterbox(frame, size)
        blob = np.transpose(canvas.astype(np.float32) * SCALE, (2, 0, 1))[None]
        raw = self.pool.run(blob)[0]
        return raw.reshape(raw.shape[-2], raw.shape[-1]), scale

    def detect(self, frame: np.ndarray) -> list[Detection]:
        shape = frame.shape[:2]
        found: list[Detection] = []
        for size in self.settings.text_sizes:
            prob, scale = self.map_for(frame, size)
            found.extend(quads_from(prob, scale, shape, self.settings))
        return merged(found, self.settings)


def hold(per_frame: dict[int, list[Detection]], settings: Settings,
         frames: int) -> dict[int, list[Detection]]:
    """Text over time, on the pattern of `screens.hold`.

    A line of text does not blink. The detector does: a sign at the edge of
    the picture is found on one frame in three as the camera moves, and a
    mask that follows the detector strobes. So a line is linked frame to
    frame by the overlap of its box, short gaps are closed, and a run carries
    a tail after the last sighting.

    `text_min_run` is the one rule that drops anything: a line seen on fewer
    frames than that is a pattern on a tile or a fold in a shirt, which is
    what `screens.hold` learned about tables in report 15.2.
    """
    from .screens import hold as hold_screens

    # The same function, told the text numbers. Grouping a line of text over
    # time and grouping a monitor over time is one problem, and a second copy
    # of it would be a second place for the gap fill and the tail to drift.
    # `runs_in` links on `track_iou`, which is why that is set here too.
    return hold_screens(per_frame, settings.with_changes(
        track_iou=settings.text_link_iou, screen_gap=settings.text_gap,
        screen_tail=settings.text_tail, screen_min_run=settings.text_min_run), frames)
