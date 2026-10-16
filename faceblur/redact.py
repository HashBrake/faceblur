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
    return _alpha_in((0, 0, shape[1], shape[0]), dets, settings)


def _blur_reach(feather: float) -> int:
    """Pixels the feather blur reaches on each side, with room to spare.

    `cv2.GaussianBlur` on uint8 at sigma `feather / 2` takes a kernel of
    `round(6 sigma + 1) | 1`, so it reaches `1.5 * feather + 1` at most.
    """
    return int(math.ceil(2 * feather)) + 2


def _crop(shape: tuple[int, int], dets: Sequence[Detection],
          settings: Settings) -> "tuple[int, int, int, int] | None":
    """The part of the frame one frame's mask work needs, as x0, y0, x1, y1.

    Every region drawn grown by the feather (plus a pixel of rounding and one
    of anti aliasing), then the blur's reach, then that reach again: the
    alpha is zero past the first reach, and a canvas edge one more reach
    away blurs to the same numbers the full frame does, because what the blur
    reflects in from that edge is zeros either way. Clipped to the frame,
    where the canvas edge is the frame's own and reflects the same.

    The profile of 2026-10-16 had the write pass building, blurring and
    averaging full 1920x1080 float alphas several times a frame for masks
    that cover under one percent of it.
    """
    h, w = shape
    margin = settings.feather + 3 + 2 * _blur_reach(settings.feather)
    bounds = [region_for(det, settings).bounds(margin) for det in dets]
    x0 = max(0, min(b[0] for b in bounds))
    y0 = max(0, min(b[1] for b in bounds))
    x1 = min(w, max(b[2] for b in bounds))
    y1 = min(h, max(b[3] for b in bounds))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _alpha_in(box: tuple[int, int, int, int], dets: Sequence[Detection],
              settings: Settings) -> np.ndarray:
    """`build_alpha` over `box` of the frame only.

    Points are rounded in frame coordinates and then moved by whole pixels,
    so the rasteriser sees the sub pixel positions it would see on the full
    frame. Moving the float points first and rounding after would not: numpy
    rounds halves to even, and an odd offset flips which way a half goes.
    """
    x0, y0, x1, y1 = box
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    f = settings.feather
    for det in dets:
        if shape_of(det) == ELLIPSE:
            e = ellipse_for(det, settings)
            cv2.ellipse(canvas, (int(round(e.cx)) - x0, int(round(e.cy)) - y0),
                        (int(round(e.ax + f)), int(round(e.ay + f))),
                        e.angle, 0, 360, 255, -1, cv2.LINE_AA)
        else:
            quad = np.round(grow_quad(det.corners(), f)).astype(np.int32)
            cv2.fillConvexPoly(canvas, quad - np.array([x0, y0], np.int32),
                               255, cv2.LINE_AA)
    if f > 0 and canvas.any():
        canvas = cv2.GaussianBlur(canvas, (0, 0), f / 2)
    return canvas.astype(np.float32) / 255.0


def _zone_in(shape: tuple[int, int], box: tuple[int, int, int, int],
             polygons: Sequence, feather: float) -> np.ndarray:
    """`zone.alpha_for` over `box` of the frame only, with the same numbers.

    The polygons are grown and rounded exactly as `alpha_for` does it, in
    frame coordinates. A hand's polygon is often larger than the box, and it
    is drawn whole (inside the frame) on a canvas that holds it rather than
    cut at the box: OpenCV clips an anti aliased edge to the canvas before
    stepping along it, and a clipped edge can land a level away from the same
    edge drawn whole. A polygon too far away to reach the box through the
    blur is not drawn at all.
    """
    from .zone import polygon_area

    h, w = shape
    reach = _blur_reach(feather) + 2
    bx0, by0, bx1, by1 = box
    x0, y0, x1, y1 = box
    drawn = []
    for poly in polygons:
        q = np.asarray(poly, np.float64)
        if len(q) < 3 or polygon_area(q) <= 0:
            continue
        if feather > 0:
            c = q.mean(axis=0)
            spread = np.abs(q - c).max(axis=0)
            if spread.min() > 1e-6:
                q = c + (q - c) * (1.0 + feather / spread)
        pts = np.round(q).astype(np.int32)
        px0, py0 = (int(v) - reach for v in pts.min(axis=0))
        px1, py1 = (int(v) + reach + 1 for v in pts.max(axis=0))
        if px1 <= bx0 or px0 >= bx1 or py1 <= by0 or py0 >= by1:
            continue
        drawn.append(pts)
        x0, y0, x1, y1 = min(x0, px0), min(y0, py0), max(x1, px1), max(y1, py1)
    if not drawn:
        return np.zeros((by1 - by0, bx1 - bx0), np.float32)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for pts in drawn:
        cv2.fillConvexPoly(canvas, pts - np.array([x0, y0], np.int32), 255, cv2.LINE_AA)
    if feather > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), feather / 2)
    return canvas[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0].astype(np.float32) / 255.0


class _Masks:
    """One frame's alphas over one box of it, each drawn at most once.

    The write pass asks for the plain alpha (to say what the zone took back),
    the zone cut alpha (what is destroyed) and, on a frame with two kinds,
    each kind's alpha. These share drawings, so each is kept by the
    detections it was drawn from.
    """

    def __init__(self, shape: tuple[int, int], box: tuple[int, int, int, int],
                 settings: Settings, zone=None):
        self.shape, self.box, self.settings, self.zone = shape, box, settings, zone
        self._plain: dict = {}
        self._zones: dict = {}

    def plain(self, dets: Sequence[Detection]) -> np.ndarray:
        key = tuple(map(id, dets))
        if key not in self._plain:
            self._plain[key] = _alpha_in(self.box, dets, self.settings)
        return self._plain[key]

    def _cut(self, mask: np.ndarray, name: str, polygons: tuple) -> np.ndarray:
        if not polygons:
            return mask
        if name not in self._zones:
            self._zones[name] = _zone_in(self.shape, self.box, polygons,
                                         self.settings.feather)
        return mask * (1.0 - self._zones[name])

    def zoned(self, dets: Sequence[Detection]) -> np.ndarray:
        """`alpha_with_zone` over the box; the reasons are written there."""
        from .zone import REMEMBERED_PROTECTS

        zone, settings = self.zone, self.settings
        if not zone or not dets:
            return self.plain(dets)
        reach = tuple(zone.live) + tuple(zone.remembered)
        on_faces = tuple(zone.hands) if settings.zone_face_needs_hand else tuple(zone.live)
        if not reach and not on_faces:
            return self.plain(dets)
        by_reach = [d for d in dets if d.kind in REMEMBERED_PROTECTS]
        by_hand = [d for d in dets if d.kind not in REMEMBERED_PROTECTS]
        if not by_reach:
            return self._cut(self.plain(dets), "faces", on_faces)
        if not by_hand:
            return self._cut(self.plain(dets), "reach", reach)
        return np.maximum(self._cut(self.plain(by_hand), "faces", on_faces),
                          self._cut(self.plain(by_reach), "reach", reach))


def alpha_with_zone(shape: tuple[int, int], dets: Sequence[Detection],
                    settings: Settings, zone=None) -> np.ndarray:
    """`build_alpha`, with the handled zone subtracted from it.

    The zone is applied here, at the last step, rather than by dropping
    detections earlier. Two reasons. The audit can then report both what would
    have been masked and what was, which is what tells a reader the zone is
    doing something rather than the detector finding nothing. And a partly
    covered thing is partly masked: a face half behind the wearer's hand loses
    the half that is behind the hand and keeps the rest, where dropping the
    detection would have left the whole face visible.

    Live polygons protect every kind. Remembered ones protect text and screens
    only, so when both exist the two groups of detections need their own
    canvas. With no zone, or with nothing remembered, there is one canvas and
    the result is `build_alpha` exactly, which is what makes `zone=False`
    reproduce the old behaviour byte for byte.

    Two protections, because they are not the same promise. What the wearer
    is reaching over protects text and screens: that is the thing in their
    hand. A face is protected only where a hand is actually on it, because a
    face inside somebody's reach is usually their own face. See
    `zone_face_needs_hand` and report section 17.
    """
    return _Masks(shape, (0, 0, shape[1], shape[0]), settings, zone).zoned(dets)


def build_mask(shape: tuple[int, int], dets: Sequence[Detection],
               settings: Settings) -> np.ndarray:
    """Hard uint8 mask, 255 where alpha is at least one half."""
    return (build_alpha(shape, dets, settings) >= 0.5).astype(np.uint8) * 255


def masked_shares(shape: tuple[int, int], dets: Sequence[Detection],
                  settings: Settings,
                  alpha: "np.ndarray | None" = None, zone=None) -> dict[str, float]:
    """Share of the frame each kind's regions cover, at alpha one half or more.

    The union alpha cannot answer this and the difference is not small. A
    screen mask is large by design: on the table tennis file the screen class
    destroys up to 13 percent of a frame while the face class averages under
    one. A frame whose union share is 3 percent may be 0.4 percent of face and
    the rest glass, and every budget and gate written about faces has to keep
    reading the face number now that a second kind can mask beside it.

    The union alpha is passed in when the caller already has it, because when
    a frame carries only one kind, which is most frames, the union *is* that
    kind's mask and there is nothing more to compute.
    """
    kinds = {det.kind for det in dets}
    if not kinds:
        return {}
    if len(kinds) == 1 and alpha is not None:
        return {kinds.pop(): float((alpha >= 0.5).mean())}
    # Measured after the zone is applied, because the question the record
    # answers is how much was destroyed, not how much was proposed.
    return {name: float((alpha_with_zone(shape, [d for d in dets if d.kind == name],
                                         settings, zone) >= 0.5).mean())
            for name in sorted(kinds)}


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
           settings: Settings, zone=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame with the sensitive things destroyed, alpha).

    Pixels at alpha 0 stay exact, which is what makes the handled zone a
    promise rather than a preference: inside it the alpha is zero, so the
    source pixel is copied through untouched.
    """
    alpha = np.zeros(frame.shape[:2], np.float32)
    if not dets:
        return frame, alpha
    box = _crop(frame.shape[:2], dets, settings)
    if box is None:
        return frame, alpha
    out, cropped = _redact_in(frame, dets, settings, _Masks(frame.shape[:2], box, settings, zone))
    # The alpha is zero outside the box by construction, see `_crop`.
    alpha[box[1]:box[3], box[0]:box[2]] = cropped
    return out, alpha


def redact_measured(frame: np.ndarray, dets: Sequence[Detection], settings: Settings,
                    zone=None) -> tuple[np.ndarray, float, dict[str, float], float]:
    """`redact` and the write pass's numbers for the frame, from one set of alphas.

    Returns (frame out, masked share, `masked_shares`, share the zone took
    back). The same numbers `redact`, `build_alpha` and `masked_shares` give
    on the full frame, worked out on `_crop`'s box: a share is a count over
    the box divided by the frame's pixel count, which is exact because the
    alpha is zero outside the box. A frame with no regions does no mask work.
    """
    if not dets:
        return frame, 0.0, {}, 0.0
    h, w = frame.shape[:2]
    box = _crop((h, w), dets, settings)
    if box is None:
        return frame, 0.0, {det.kind: 0.0 for det in dets}, 0.0
    masks = _Masks((h, w), box, settings, zone)
    out, alpha = _redact_in(frame, dets, settings, masks)
    n = h * w
    hard = alpha >= 0.5
    masked = np.count_nonzero(hard) / n
    prevented = (np.count_nonzero((masks.plain(dets) >= 0.5) & ~hard) / n
                 if zone else 0.0)
    kinds = sorted({det.kind for det in dets})
    if len(kinds) == 1:
        shares = {kinds[0]: masked}
    else:
        # Measured after the zone, as `masked_shares` does.
        shares = {name: np.count_nonzero(
                      masks.zoned([d for d in dets if d.kind == name]) >= 0.5) / n
                  for name in kinds}
    return out, masked, shares, prevented


def _redact_in(frame: np.ndarray, dets: Sequence[Detection], settings: Settings,
               masks: _Masks) -> tuple[np.ndarray, np.ndarray]:
    """`redact` with its alpha over `masks.box` only. Returns (frame, that alpha)."""
    alpha = masks.zoned(dets)
    if not alpha.any():
        return frame, alpha
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
    cx0, cy0 = masks.box[:2]
    out = frame.copy()
    roi = frame[y0:y1, x0:x1]
    cover = roi.copy()
    for e in regions:
        shifted = (FaceEllipse(e.cx - x0, e.cy - y0, e.ax, e.ay, e.angle)
                   if isinstance(e, FaceEllipse)
                   else QuadRegion(tuple((px - x0, py - y0) for px, py in e.corners)))
        _cover_region(roi, cover, shifted, settings.mode, settings.strength, settings.feather)
    # The blend box sits inside the mask box: `_crop`'s margin is the wider.
    a = alpha[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]
    # Only the pixels the alpha touches are blended; the rest of the box is
    # already the source in `out`. The box around two faces at opposite sides
    # of the frame is most of the frame, and blending all of it in float was
    # 33 ms of a 57 ms frame (sample frames, 2 to 4 boxes spread over the
    # frame, T14). Same arithmetic per pixel.
    on = a > 0
    w = a[on][:, None]
    blended = roi[on].astype(np.float32) * (1.0 - w) + cover[on].astype(np.float32) * w
    out[y0:y1, x0:x1][on] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return out, alpha
