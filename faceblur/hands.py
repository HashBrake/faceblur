"""Where the hands are, so that a check reading the output can tell one from a face.

Every face detector in this project calls the wearer's own hand a face.
Section 10 measured YuNet scoring a hand on a mop at 0.74 to 0.82, and the
pipeline keeps those out of the mask with track-level rules: a hand is
confirmed two or three frames at a time and never earns a sure track.
`faceblur/verify.py` looks at one frame at a time and has nothing of the sort,
so on the table tennis footage it reports the hand holding the bat as a face
the run missed.

This is the answer that does not depend on how long something was seen: ask
models trained on hands. Both are MediaPipe's, Apache 2.0, converted from the
tflite in its wheel by `models/make_hands.py`. They are asked only about a box
the check has already flagged, so they cost the pipeline nothing.

Two stages, for the same reason the face side has two. The palm detector on
its own is loose: of the 138 boxes it put on the crops taken around the 108
things the check found in the four sample files, the landmark model rejects
97, and two crops in five carry one. Since the region a palm box hands on is
2.6 times its size, a spurious one swallows whatever face is nearby — on
`004100` frame 450 it covered a real missed face with a hand it had invented
on the man's shoulder. So every palm box goes to the landmark model, which
returns how sure it is that the rectangle holds a hand, and only a confirmed
one counts. Against those 108 boxes, labelled by eye: over 1050 combinations
of window set, palm floor, region and coverage, the palm detector alone loses
no face in 256 of them and the best of those reaches 11 of the 20 hands; with
the confirmation all 1050 lose no face and the best reaches 13. What the
second model buys is not mainly the two hands, it is that the geometry stops
mattering.

    from faceblur.hands import HandRule
    rule = HandRule(settings)
    cover = rule.hand_cover(frame, det)     # 0.0 to 1.0, above hand_cover it is a hand
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np

from .detect import MODEL_DIR, Detection, _require, providers_for
from .settings import Settings

PALM_MODEL = MODEL_DIR / "palm_detection.onnx"
LANDMARK_MODEL = MODEL_DIR / "hand_landmark.onnx"

# The graph is fixed at one 192x192 RGB image in [0, 1], NHWC.
PALM_SIZE = 192
# MediaPipe's own SsdAnchorsCalculator settings for this model, from
# palm_detection_cpu.pbtxt. The anchor count they produce, 2016, is the
# model's output length, which is the check that they are the right ones.
ANCHOR_LAYERS = 4
ANCHOR_STRIDES = (8, 16, 16, 16)
# The calculator's min_scale and max_scale do not appear here on purpose: this
# model sets `fixed_anchor_size`, so every anchor is one input wide whatever
# its layer's scale, and only the centres are ever read.
# TensorsToDetectionsCalculator: each row is four box numbers then 7 keypoints
# as x, y pairs, all divided by the input size, and a score through a sigmoid.
# DetectionsToRectsCalculator: the hand's axis runs from the wrist, keypoint 0,
# to the middle finger's knuckle, keypoint 2, and is turned to point up.
ROTATION_FROM, ROTATION_TO = 0, 2
TARGET_ANGLE = math.pi / 2
# RectTransformationCalculator, from hand_detection: square off the long side,
# grow 2.6 times and shift half a height towards the fingers.
RECT_SCALE = 2.6
RECT_SHIFT_Y = -0.5
# The landmark model takes that rectangle, warped square, at this size.
LANDMARK_SIZE = 224


class PalmModelMissing(FileNotFoundError):
    """The palm model is not on disk, so the hand rule cannot run."""


@dataclass(frozen=True)
class Hand:
    """One hand, as the four corners of a rotated rectangle, in frame pixels."""

    quad: tuple           # ((x, y), ...) clockwise, from the rect transformation
    score: float
    palm: tuple           # (x, y, w, h) the detector's own upright palm box
    presence: float = 1.0  # what the landmark model makes of it, 1.0 if not asked

    def bounds(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return (int(math.floor(min(xs))), int(math.floor(min(ys))),
                int(math.ceil(max(xs))), int(math.ceil(max(ys))))

    def to_dict(self) -> dict:
        x, y, w, h = self.palm
        return {"score": round(self.score, 3), "presence": round(self.presence, 3),
                "palm": [round(v, 1) for v in (x, y, w, h)],
                "quad": [[round(px, 1), round(py, 1)] for px, py in self.quad]}

    def grown(self, factor: float) -> "Hand":
        """The same quad scaled about its centre. `factor` is a multiple of the
        palm box, so 2.6 is the quad as MediaPipe hands it to its landmark
        model and 1.0 is the palm itself."""
        q = np.asarray(self.quad, np.float64)
        c = q.mean(axis=0)
        return Hand(tuple(map(tuple, c + (q - c) * (factor / RECT_SCALE))),
                    self.score, self.palm, self.presence)


@lru_cache(maxsize=1)
def anchors() -> np.ndarray:
    """Anchor centres, normalised, one row per output box.

    A transcription of ssd_anchors_calculator.cc for this model's options.
    Every anchor is square and one input wide (`fixed_anchor_size`), so only
    the centres are ever used and the array holds nothing else. Layers that
    share a stride share a feature map and stack their anchors on it: stride
    8 carries two, the three stride 16 layers carry six between them, which
    is 24*24*2 + 12*12*6 = 2016.
    """
    out = []
    layer = 0
    while layer < ANCHOR_LAYERS:
        last = layer
        per_cell = 0
        while last < ANCHOR_LAYERS and ANCHOR_STRIDES[last] == ANCHOR_STRIDES[layer]:
            per_cell += 2       # the layer's own scale, and the interpolated one
            last += 1
        stride = ANCHOR_STRIDES[layer]
        rows = cols = int(math.ceil(PALM_SIZE / stride))
        ys = (np.arange(rows) + 0.5) / rows
        xs = (np.arange(cols) + 0.5) / cols
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        out.append(np.repeat(grid, per_cell, axis=0))
        layer = last
    return np.concatenate(out).astype(np.float32)


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Fit an image into the square input, centred, on black.

    MediaPipe's ImageToTensorCalculator keeps the aspect ratio and pads with
    zeros, and it centres what it pads. Getting that wrong puts every box half
    a frame out, which is why the offsets come back with the scale.
    """
    h, w = image.shape[:2]
    f = min(PALM_SIZE / float(w), PALM_SIZE / float(h))
    nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
    small = cv2.resize(image, (nw, nh),
                       interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((PALM_SIZE, PALM_SIZE, 3), np.uint8)
    ox, oy = (PALM_SIZE - nw) // 2, (PALM_SIZE - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas, f, ox, oy


def rect_for(cx: float, cy: float, w: float, h: float,
             wrist: tuple[float, float], knuckle: tuple[float, float]) -> tuple:
    """The whole hand's rotated rectangle, from the palm box and two keypoints.

    DetectionsToRectsCalculator then RectTransformationCalculator, in the
    order MediaPipe runs them: find the angle that puts the wrist-to-knuckle
    line upright, shift the centre along that angle, square the box off on its
    long side, and grow it. Returns the four corners.
    """
    angle = TARGET_ANGLE - math.atan2(-(knuckle[1] - wrist[1]), knuckle[0] - wrist[0])
    angle = angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))
    ca, sa = math.cos(angle), math.sin(angle)
    # shift_x is zero for hands, so only the height term survives.
    cx += -h * RECT_SHIFT_Y * sa
    cy += h * RECT_SHIFT_Y * ca
    side = max(w, h) * RECT_SCALE
    hx, hy = side / 2, side / 2
    corners = []
    for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        corners.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return tuple(corners)


def decode(boxes: np.ndarray, scores: np.ndarray, conf: float, nms: float,
           f: float, ox: int, oy: int) -> list[Hand]:
    """Raw model output to hands in frame pixels.

    Everything the model says is relative to its own 192x192 input and to the
    anchor the row belongs to; `f`, `ox` and `oy` undo the letterbox.
    """
    from .detect import nms_indices

    p = 1.0 / (1.0 + np.exp(-np.clip(scores[:, 0], -100.0, 100.0)))
    sel = np.where(p >= conf)[0]
    if sel.size == 0:
        return []
    a = anchors()[sel]
    raw = boxes[sel]
    cx = raw[:, 0] / PALM_SIZE + a[:, 0]
    cy = raw[:, 1] / PALM_SIZE + a[:, 1]
    bw = raw[:, 2] / PALM_SIZE
    bh = raw[:, 3] / PALM_SIZE
    xywh = np.stack([(cx - bw / 2) * PALM_SIZE, (cy - bh / 2) * PALM_SIZE,
                     bw * PALM_SIZE, bh * PALM_SIZE], axis=1).astype(np.float64)
    keep = nms_indices(xywh, p[sel].astype(np.float64), nms)

    def to_frame(px: float, py: float) -> tuple[float, float]:
        return ((px - ox) / f, (py - oy) / f)

    out = []
    for i in keep:
        kp = []
        for k in (ROTATION_FROM, ROTATION_TO):
            j = 4 + 2 * k
            kp.append(to_frame((raw[i, j] / PALM_SIZE + a[i, 0]) * PALM_SIZE,
                               (raw[i, j + 1] / PALM_SIZE + a[i, 1]) * PALM_SIZE))
        x, y = to_frame(xywh[i, 0], xywh[i, 1])
        w, h = xywh[i, 2] / f, xywh[i, 3] / f
        out.append(Hand(rect_for(x + w / 2, y + h / 2, w, h, kp[0], kp[1]),
                        float(p[sel][i]), (x, y, w, h)))
    return out


class HandLandmarks:
    """MediaPipe's hand-landmark model, used for the one number it also returns.

    The palm detector alone is not enough. It is loose on this footage: of
    138 boxes it put on the crops around what the check had found, this model
    rejects 97, and two crops in five carry one. Since the region a palm box
    hands on is 2.6 times its size, a spurious one swallows whatever face is
    nearby. That is the same problem the face side of this project has, and it
    gets the same answer: a second model of a different kind has to agree.

    MediaPipe already runs one. Its landmark stage takes the rotated square
    the palm detector produced, warps it to 224x224, and returns 21 landmarks
    together with a hand-presence score, and MediaPipe drops the hand when
    that score is low. Only the score is used here; the landmarks are what the
    score is a statement about.
    """

    name = "hand_landmark"

    def __init__(self, device: str = "auto", threads: Optional[int] = None):
        import onnxruntime as ort

        if not LANDMARK_MODEL.is_file():
            raise PalmModelMissing(
                f"Could not find the hand model {LANDMARK_MODEL}. Reinstall FaceBlur, or "
                f"run the check with the hand rule off.")
        ort.set_default_logger_severity(3)
        self.model_path = _require(LANDMARK_MODEL)
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), options,
                                            providers=providers_for(device))
        self.provider = self.session.get_providers()[0]
        self.outputs = [o.name for o in self.session.get_outputs()]

    def presence(self, frame: np.ndarray, hand: Hand) -> float:
        """How sure the landmark model is that this rectangle holds a hand.

        The rectangle is rotated, so the crop is an affine warp rather than a
        slice: three of its corners fix the map onto the square input.

        The output needs no sigmoid, whatever MediaPipe's graph does with it
        downstream: this model already returns a probability. Black reads
        0.007, flat grey 0.004, random noise 0.009 and a hand 0.89, and a
        sigmoid over that would compress the lot into 0.50 to 0.71 and leave
        nothing to threshold.
        """
        src = np.array(hand.quad[:3], np.float32)
        n = float(LANDMARK_SIZE)
        dst = np.array([[0, 0], [n, 0], [n, n]], np.float32)
        crop = cv2.warpAffine(frame, cv2.getAffineTransform(src, dst),
                              (LANDMARK_SIZE, LANDMARK_SIZE), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        blob = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)[None]
        out = dict(zip(self.outputs, self.session.run(None, {"input": blob})))
        return float(np.clip(out["presence"].ravel()[0], 0.0, 1.0))


class PalmDetector:
    """MediaPipe's palm detector through onnxruntime, one frame at a time.

    Fixed shape, batch of one: it is asked about the handful of frames where
    something was already found, so there is nothing to batch and a fixed
    graph is what DirectML wants. The model is converted from the tflite in
    the MediaPipe wheel by `models/make_hands.py`; the conversion is faithful
    to 1.5e-4 on the raw output.
    """

    name = "palm"

    def __init__(self, conf: float = 0.5, nms: float = 0.3, device: str = "auto",
                 threads: Optional[int] = None):
        """`conf` defaults to MediaPipe's own min_score_thresh. The check
        passes `settings.hand_conf`, which is lower, because the landmark
        model decides afterwards and this only has to propose."""
        import onnxruntime as ort

        if not PALM_MODEL.is_file():
            raise PalmModelMissing(
                f"Could not find the hand model {PALM_MODEL}. Reinstall FaceBlur, or "
                f"run the check with the hand rule off.")
        ort.set_default_logger_severity(3)
        self.model_path = _require(PALM_MODEL)
        self.conf = conf
        self.nms = nms
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), options,
                                            providers=providers_for(device))
        self.provider = self.session.get_providers()[0]

    def detect(self, frame: np.ndarray) -> list[Hand]:
        canvas, f, ox, oy = letterbox(frame)
        blob = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)[None]
        boxes, scores = self.session.run(None, {"input": blob})
        return decode(boxes[0], scores[0], self.conf, self.nms, f, ox, oy)


class HandRule:
    """Is this box a hand? The palm detector, the landmark model, and a window.

    A window rather than a multiple of the box. Scaling the crop with the box
    was the first thing tried and it is unstable: the same face read 0.00 hand
    at one multiple and 1.00 at the next, because what the palm detector makes
    of a crop depends on how large a hand is inside it, and a multiple of the
    box holds that constant only if the box is the hand. A fixed pixel window
    downscaled to the model's own 192 puts every hand at the size the model
    was trained for, whatever the box around it. Two windows are used, 384 and
    512: with one 512 window the rule sets aside 12 of 20 hands, with both 13,
    and adding a third at 768 adds none.
    """

    def __init__(self, settings: Settings, threads: Optional[int] = None):
        self.settings = settings
        device = settings.hand_device
        self.palm = PalmDetector(settings.hand_conf, device=device, threads=threads)
        self.landmarks = HandLandmarks(device=device, threads=threads)

    @property
    def compute(self) -> dict[str, str]:
        return {self.palm.name: self.palm.provider,
                self.landmarks.name: self.landmarks.provider}

    @property
    def model_hashes(self) -> dict[str, str]:
        from .detect import model_sha256
        return {self.palm.name: model_sha256(self.palm.model_path),
                self.landmarks.name: model_sha256(self.landmarks.model_path)}

    def hands_near(self, frame: np.ndarray, det: Detection) -> list[Hand]:
        """Confirmed hands around a box, in frame coordinates.

        One pass per window, and the results are pooled rather than made to
        agree: a hand seen in either window is a hand. Requiring both takes
        the hands set aside from 13 to 8 and removes no face, because the
        confirmation has already taken out what agreement was there to take.
        """
        h, w = frame.shape[:2]
        found: list[Hand] = []
        for side in self.settings.hand_windows:
            x0 = int(max(0, round(det.cx - side / 2)))
            y0 = int(max(0, round(det.cy - side / 2)))
            x1 = int(min(w, round(det.cx + side / 2)))
            y1 = int(min(h, round(det.cy + side / 2)))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            for hand in self.palm.detect(crop):
                presence = self.landmarks.presence(crop, hand)
                if presence < self.settings.hand_presence:
                    continue
                found.append(Hand(tuple((px + x0, py + y0) for px, py in hand.quad),
                                  hand.score, (hand.palm[0] + x0, hand.palm[1] + y0,
                                               hand.palm[2], hand.palm[3]), presence))
        return found

    def hand_cover(self, frame: np.ndarray, det: Detection) -> float:
        """Share of the box that confirmed hands cover, 0 to 1.

        The caller decides what to do with it. `settings.hand_cover` is where
        the check draws the line.
        """
        return covered_by(det, self.hands_near(frame, det))


def covered_by(det: Detection, hands: Sequence[Hand]) -> float:
    """Share of a face box that lies inside some hand, 0 to 1.

    The hands are rotated rectangles, so the honest answer needs the polygons
    rather than their bounding boxes: an outstretched hand's rectangle is 2.6
    times the palm on a diagonal, and its upright bounds take in a lot of room
    it does not occupy. Drawing them into a mask the size of the face box
    costs nothing at these sizes and is exact.
    """
    if not hands:
        return 0.0
    x0, y0 = int(math.floor(det.x)), int(math.floor(det.y))
    x1, y1 = int(math.ceil(det.x + det.w)), int(math.ceil(det.y + det.h))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    for hand in hands:
        hx0, hy0, hx1, hy1 = hand.bounds()
        if hx1 <= x0 or hy1 <= y0 or hx0 >= x1 or hy0 >= y1:
            continue
        quad = np.array([[p[0] - x0, p[1] - y0] for p in hand.quad], np.int32)
        cv2.fillConvexPoly(mask, quad, 1)
    return float(mask.sum()) / (w * h)
