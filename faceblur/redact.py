"""Mask geometry and pixel destruction.

Two shapes, chosen by what the box is (`faceblur/classes.py`), because the
smallest region that hides a thing depends on the thing.

A **face** is hidden by an ellipse fitted to the box and rotated to the eye
line. It covers eyebrows to chin and leaves the corners of the box, which
hold hair, background, and in this footage the wearer's hands.

**Text and screens** are hidden by the quadrilateral the detector gave. An
ellipse over a line of text would spill onto the page at the middle of the
long edges and miss the first and last characters at the ends, which is the
worst of both; and unlike a face, every pixel inside a text quad is the thing
being hidden, so there is nothing to spare.

Either way the pixels inside are replaced from a copy of that region shrunk
to a few blocks across, so the original cannot be recovered. A plain Gaussian
blur is a linear filter and can be partly undone, so no mode ships that only
blurs. Both shapes get the same soft edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .classes import ELLIPSE, POLYGON, kind as kind_of
from .detect import Detection
from .settings import Settings


@dataclass(frozen=True)
class FaceEllipse:
    cx: float
    cy: float
    ax: float      # semi axis along the eye line
    ay: float      # semi axis across it
    angle: float   # degrees, eye line from horizontal

    def bounds(self, margin: float = 0.0) -> tuple[int, int, int, int]:
        """Axis aligned x0, y0, x1, y1 that holds the rotated ellipse."""
        r = max(self.ax, self.ay) + margin
        return (int(math.floor(self.cx - r)), int(math.floor(self.cy - r)),
                int(math.ceil(self.cx + r)), int(math.ceil(self.cy + r)))

    def block_scale(self) -> float:
        """Half the size the block count is measured across. A face is round
        enough that its long side is the right one."""
        return max(self.ax, self.ay)


@dataclass(frozen=True)
class QuadRegion:
    """Four corners, for a kind that is not oval: text, a screen.

    Held as corners rather than a rectangle because a line of text on a page
    held at an angle is a rotated quad, and its upright bounding box takes in
    a lot of page it has no business destroying.
    """

    corners: tuple

    def bounds(self, margin: float = 0.0) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.corners]
        ys = [p[1] for p in self.corners]
        return (int(math.floor(min(xs) - margin)), int(math.floor(min(ys) - margin)),
                int(math.ceil(max(xs) + margin)), int(math.ceil(max(ys) + margin)))

    def block_scale(self) -> float:
        """Half the *short* side. A line of text is wide and low, and sizing
        the blocks by its length would leave every character legible: at
        strength 6 a 400 px line would get 66 px blocks against a 20 px cap
        height, which is one block over the whole line and no destruction at
        all across it."""
        q = self.corners
        sides = [math.dist(q[i], q[(i + 1) % len(q)]) for i in range(len(q))]
        return max(1.0, min(sides) / 2)


def region_for(det: Detection, settings: Settings):
    """The shape that hides this box, chosen by what the box is."""
    if shape_of(det) == ELLIPSE:
        return ellipse_for(det, settings)
    return QuadRegion(tuple(tuple(p) for p in det.corners()))


def ellipse_for(det: Detection, settings: Settings) -> FaceEllipse:
    """The smallest region that hides identity, from the box and its landmarks."""
    if det.landmarks is not None:
        e1, e2, nose, m1, m2 = det.landmarks
        lm_cx = sum(p[0] for p in det.landmarks) / 5
        lm_cy = sum(p[1] for p in det.landmarks) / 5
        # Landmarks sit low in the box (no forehead point). Blend with the box
        # centre so the ellipse still reaches the eyebrows.
        cx, cy = 0.5 * (det.cx + lm_cx), 0.5 * (det.cy + lm_cy)
        angle = math.degrees(math.atan2(e2[1] - e1[1], e2[0] - e1[0]))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return FaceEllipse(cx, cy, settings.ellipse_w * det.w / 2,
                           settings.ellipse_h * det.h / 2, angle)
    # No landmarks: a padded, upright ellipse in the box.
    return FaceEllipse(det.cx, det.cy, (1 + settings.pad) * det.w / 2,
                       (1 + settings.pad) * det.h / 2, 0.0)


def grow_quad(corners: Sequence[Sequence[float]], margin: float) -> np.ndarray:
    """The quad pushed out by `margin` pixels on every side, about its centre.

    The same trick the ellipse uses: draw it enlarged by the feather width and
    blur, so the alpha is still 1 at the true edge instead of half way down
    the ramp. Scaling about the centre rather than offsetting each edge is
    close enough at these margins and cannot fold a thin quad inside out.
    """
    q = np.asarray(corners, np.float64)
    c = q.mean(axis=0)
    spread = np.abs(q - c).max(axis=0)
    if margin <= 0 or spread.min() <= 1e-6:
        return q
    return c + (q - c) * (1.0 + margin / spread)


def shape_of(det: Detection) -> str:
    """Which mask shape a box gets. An unknown kind is masked as a polygon:
    the box itself, which hides everything the detector pointed at."""
    try:
        return kind_of(det.kind).shape
    except ValueError:
        return POLYGON


def build_alpha(shape: tuple[int, int], dets: Sequence[Detection],
                settings: Settings) -> np.ndarray:
    """Soft mask, float32 in [0, 1], 1 inside every region.

    Each region is drawn enlarged by the feather width and then blurred, so
    the alpha stays at 1 up to the true edge and fades outside it.
    """
    h, w = shape
    canvas = np.zeros((h, w), np.uint8)
    f = settings.feather
    for det in dets:
        if shape_of(det) == ELLIPSE:
            e = ellipse_for(det, settings)
            cv2.ellipse(canvas, (int(round(e.cx)), int(round(e.cy))),
                        (int(round(e.ax + f)), int(round(e.ay + f))),
                        e.angle, 0, 360, 255, -1, cv2.LINE_AA)
        else:
            quad = grow_quad(det.corners(), f)
            cv2.fillConvexPoly(canvas, np.round(quad).astype(np.int32), 255, cv2.LINE_AA)
    if f > 0 and canvas.any():
        canvas = cv2.GaussianBlur(canvas, (0, 0), f / 2)
    return canvas.astype(np.float32) / 255.0


def build_mask(shape: tuple[int, int], dets: Sequence[Detection],
               settings: Settings) -> np.ndarray:
    """Hard uint8 mask, 255 where alpha is at least one half."""
    return (build_alpha(shape, dets, settings) >= 0.5).astype(np.uint8) * 255


def _cover_region(frame: np.ndarray, cover: np.ndarray, e: "Region",
                  mode: str, strength: int, feather: float) -> None:
    """Fill `cover` inside the region's bounding box with destroyed pixels."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = e.bounds(feather * 2)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    if mode == "solid":
        cover[y0:y1, x0:x1] = 0
        return
    rh, rw = roi.shape[:2]
    # Block size from the region's own size, so a small one is destroyed as
    # thoroughly as a large one. For a line of text that is its height, not
    # its length: `short_side` is what a reader needs, and sizing the blocks
    # by a long line's length would leave every character legible.
    k = max(2, int(round(2 * e.block_scale() / strength)))
    small = cv2.resize(roi, (max(1, rw // k), max(1, rh // k)),
                       interpolation=cv2.INTER_AREA)
    if mode == "pixelate":
        up = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        up = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (0, 0), max(1.0, k / 2))
    cover[y0:y1, x0:x1] = up


def redact(frame: np.ndarray, dets: Sequence[Detection],
           settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame with faces destroyed, alpha). Pixels at alpha 0 stay exact."""
    if not dets:
        return frame, np.zeros(frame.shape[:2], np.float32)
    alpha = build_alpha(frame.shape[:2], dets, settings)
    H, W = frame.shape[:2]
    # Everything happens inside the box around the ellipses. Blending the whole
    # frame in float was most of the cost of the write pass.
    regions = [region_for(det, settings) for det in dets]
    margin = settings.feather * 2 + 2
    x0 = max(0, min(e.bounds(margin)[0] for e in regions))
    y0 = max(0, min(e.bounds(margin)[1] for e in regions))
    x1 = min(W, max(e.bounds(margin)[2] for e in regions))
    y1 = min(H, max(e.bounds(margin)[3] for e in regions))
    if x1 <= x0 or y1 <= y0:
        return frame, alpha
    out = frame.copy()
    roi = frame[y0:y1, x0:x1]
    cover = roi.copy()
    for e in regions:
        shifted = (FaceEllipse(e.cx - x0, e.cy - y0, e.ax, e.ay, e.angle)
                   if isinstance(e, FaceEllipse)
                   else QuadRegion(tuple((px - x0, py - y0) for px, py in e.corners)))
        _cover_region(roi, cover, shifted, settings.mode, settings.strength, settings.feather)
    a = alpha[y0:y1, x0:x1, None]
    blended = roi.astype(np.float32) * (1.0 - a) + cover.astype(np.float32) * a
    out[y0:y1, x0:x1] = np.where(a > 0, np.clip(blended + 0.5, 0, 255).astype(np.uint8), roi)
    return out, alpha
