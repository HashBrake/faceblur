"""The handled zone: the region nothing is ever masked in.

The owner's rule, 2026-09-11: *anything the collector is interacting with or
handling is not blurred at all; everything else that carries information is
blurred.* This module is the first half of that sentence. Every other part of
the pipeline finds things to destroy; this finds the one region that is never
destroyed, and `redact.redact` subtracts it from every mask at the last step.

That ordering matters and is deliberate. Detectors, tracker and holds all run
exactly as they did, so the audit record can say both what would have been
masked and what was, and turning the zone off reproduces the old behaviour
byte for byte.

**What counts as handled.** The wearer's hands, what is in or touching them,
and, for a few seconds, anything they put down and has not moved. A card, a
phone or a bat in the hand is inside the zone because the hand's own rectangle
is grown to cover it. A wall sign the collector reads is not handled and is
masked.

**Whose hands.** Bystanders have hands too, and nothing here can read
intention. In egocentric footage the wearer's hands are the largest, the
lowest and the most persistent, so a hand qualifies when its palm box is at
least `zone_min_hand_px` across **or** its rectangle reaches the bottom
`zone_edge_frac` of the frame. Both numbers are first guesses that report
section 17 measures, and the audit under `docs/audits/` is where the cases
this gets wrong are written down.

**Two kinds of protection, because they are not the same promise.** A live
polygon, around a hand seen in this frame, protects every kind: whatever is in
the hand is part of the task. A remembered polygon, where a handled thing was
in the last `zone_memory` seconds, protects text and screens only. A person
who walks into the space where a card was put down is still a person, and
their face is still masked. Remembered polygons move with the camera each
frame so they stay on the object rather than on the pixels.

    from faceblur.zone import HandFinder, build
    hands = {i: HandFinder(settings).detect(frame) for i, frame in ...}
    zones = build(hands, shifts, settings, frames, fps)
    out, alpha = redact(frame, dets, settings, zone=zones[i])
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .hands import Hand, HandLandmarks, PalmDetector
from .settings import Settings

# A remembered polygon protects these kinds and no others. A face is never
# protected by memory: the thing that was handled has been put down, and the
# person now standing where it was is a person.
REMEMBERED_PROTECTS: tuple[str, ...] = ("text", "screen")


@dataclass(frozen=True)
class ZoneFrame:
    """The protected polygons of one frame, split by what they protect."""

    live: tuple = ()          # the grown reach around a hand seen this frame
    remembered: tuple = ()    # REMEMBERED_PROTECTS only
    # The hands themselves, at `zone_face_scale` times the palm box. A face is
    # protected only where one of these covers it: the grown reach is for what
    # a hand is holding, and a face inside somebody's reach is usually their
    # face rather than a picture of one. See `zone_face_needs_hand` and report
    # section 17.
    hands: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.live or self.remembered or self.hands)

    def for_kind(self, kind: str, face_needs_hand: bool = True) -> tuple:
        """The polygons that protect this kind on this frame."""
        if kind in REMEMBERED_PROTECTS:
            return tuple(self.live) + tuple(self.remembered)
        if face_needs_hand:
            return tuple(self.hands)
        return tuple(self.live)

    def kinds_differ(self) -> bool:
        """Do two kinds see different zones here? Only then is the split needed."""
        return bool(self.remembered) and bool(self.live or self.remembered)

    def to_dict(self) -> dict:
        return {"live": [[[round(x, 1), round(y, 1)] for x, y in poly]
                         for poly in self.live],
                "remembered": [[[round(x, 1), round(y, 1)] for x, y in poly]
                               for poly in self.remembered],
                "hands": [[[round(x, 1), round(y, 1)] for x, y in poly]
                          for poly in self.hands]}


def windows(shape: tuple[int, int], side: int, overlap: float = 0.5
            ) -> list[tuple[int, int, int, int]]:
    """Sliding windows covering the frame, as (x0, y0, x1, y1).

    The palm model takes 192 px and the wearer's hands on this camera run 150
    to 500 px across, so the whole frame downscaled to 192 puts a hand at 20
    px and the model sees nothing. A window of `side` pixels downscaled to 192
    puts it at the size the model was trained for, which is the same reason
    `hands.HandRule` uses fixed windows rather than a multiple of a box.

    The last window in each direction is pulled back against the far edge
    rather than hanging past it, so the frame is covered exactly and no window
    is part black.
    """
    h, w = shape
    step = max(1, int(round(side * (1.0 - overlap))))

    def starts(extent: int) -> list[int]:
        if extent <= side:
            return [0]
        out = list(range(0, extent - side + 1, step))
        if out[-1] != extent - side:
            out.append(extent - side)
        return out

    return [(x, y, min(w, x + side), min(h, y + side))
            for y in starts(h) for x in starts(w)]


def qualifies(hand: Hand, shape: tuple[int, int], settings: Settings) -> bool:
    """Is this the wearer's hand rather than a bystander's?

    Size or position, either one. In egocentric footage the wearer's hands are
    the largest thing of their kind in the frame because they are the closest,
    and they enter from below because the camera is on the chest. A bystander
    across a table fails both; a bystander leaning in over the wearer's own
    hands passes, and report section 17 says how often that happens.
    """
    h, w = shape
    x, y, bw, bh = hand.palm
    if max(bw, bh) >= settings.zone_min_hand_px:
        return True
    lowest = max(py for _, py in hand.quad)
    return lowest >= h * (1.0 - settings.zone_edge_frac)


def polygon_for(hand: Hand, settings: Settings, shape: tuple[int, int]) -> tuple:
    """The protected polygon around one hand, clipped to the frame.

    `Hand.quad` is already MediaPipe's own rectangle, 2.6 times the palm box,
    which covers the hand itself and a little around it. `zone_scale` grows it
    further to reach what the hand is holding: a card held at arm's length
    extends about one hand's width past the fingers, and a phone about the
    same.

    Clipping is a straight clamp of each corner rather than a polygon
    intersection. A clamped quad stays convex, which `cv2.fillConvexPoly`
    needs, and the difference against a true intersection is pixels outside
    the frame that nothing reads.
    """
    h, w = shape
    grown = hand.grown(hand_scale(settings))
    return tuple((min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h)))
                 for x, y in grown.quad)


def hand_scale(settings: Settings) -> float:
    """`zone_scale` as a multiple of the palm box, which is what `Hand.grown`
    takes. `Hand.quad` is 2.6 palm boxes, so a `zone_scale` of 1.5 asks for
    1.5 times that."""
    from .hands import RECT_SCALE
    return RECT_SCALE * settings.zone_scale


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    q = np.asarray(poly, np.float64)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def overlaps(a: Sequence, b: Sequence) -> bool:
    """Do two convex quads share any area? Bounding boxes first, then exact."""
    ax = [p[0] for p in a]
    ay = [p[1] for p in a]
    bx = [p[0] for p in b]
    by = [p[1] for p in b]
    if max(ax) < min(bx) or max(bx) < min(ax) or max(ay) < min(by) or max(by) < min(ay):
        return False
    inter, _ = cv2.intersectConvexConvex(
        np.asarray(a, np.float32), np.asarray(b, np.float32))
    return inter > 0.0


def moved(poly: Sequence, dx: float, dy: float) -> tuple:
    return tuple((x + dx, y + dy) for x, y in poly)


def inside_frame(poly: Sequence, shape: tuple[int, int]) -> bool:
    """Is any of this polygon still in the picture?"""
    h, w = shape
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return max(xs) > 0 and min(xs) < w and max(ys) > 0 and min(ys) < h


class HandFinder:
    """Every hand in a frame, over sliding windows, confirmed by the second model.

    `hands.HandRule` answers "is this box a hand" about a box something else
    already flagged. This answers "where are the hands" about a whole frame,
    which is a different question and needs the frame covered rather than one
    place looked at.

    Palm boxes are pooled across windows and suppressed **before** the landmark
    model is asked, so a hand that lands in four overlapping windows costs four
    cheap inferences and one expensive one rather than four of each. The
    landmark model then runs on the full frame, which is the same geometry as
    running it on the window, because the quad carries frame coordinates by
    then.
    """

    name = "zone"

    def __init__(self, settings: Settings, threads: Optional[int] = None):
        self.settings = settings
        device = settings.zone_device
        self.palm = PalmDetector(settings.hand_conf, device=device, threads=threads)
        self.landmarks = HandLandmarks(device=device, threads=threads)

    @property
    def compute(self) -> dict[str, str]:
        return {"zone_palm": self.palm.provider, "zone_landmark": self.landmarks.provider}

    @property
    def model_hashes(self) -> dict[str, str]:
        from .detect import model_sha256
        return {"zone_palm": model_sha256(self.palm.model_path),
                "zone_landmark": model_sha256(self.landmarks.model_path)}

    def propose(self, frame: np.ndarray) -> list[Hand]:
        """Palm boxes over every window, in frame coordinates, suppressed."""
        from .detect import nms_indices

        h, w = frame.shape[:2]
        found: list[Hand] = []
        for x0, y0, x1, y1 in windows((h, w), self.settings.zone_window):
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            for hand in self.palm.detect(crop):
                found.append(Hand(tuple((px + x0, py + y0) for px, py in hand.quad),
                                  hand.score,
                                  (hand.palm[0] + x0, hand.palm[1] + y0,
                                   hand.palm[2], hand.palm[3])))
        if not found:
            return []
        boxes = np.array([h.palm for h in found], dtype=np.float64)
        scores = np.array([h.score for h in found], dtype=np.float64)
        return [found[i] for i in nms_indices(boxes, scores, self.settings.zone_nms)]

    def detect(self, frame: np.ndarray, qualify: bool = True) -> list[Hand]:
        """Confirmed hands in this frame, in frame coordinates.

        Confirmation is the same plateau `hands.py` measured: the palm
        detector proposes at `hand_conf` and the landmark model has to agree at
        `hand_presence`. Section 14.2 measured that the geometry stops
        mattering once the second model is there, which is what makes it safe
        to grow a polygon around the result.

        Qualification runs **before** confirmation, because it is free and the
        landmark model is not: on this footage the palm pass proposes about ten
        boxes a frame at 2 ms each and each confirmation costs about 7 ms, so
        asking only about the hands that could be the wearer's is most of the
        cost of the zone. `qualify=False` asks about all of them, which is what
        the audit needs to count the ones the size and edge rules throw away.
        """
        shape = frame.shape[:2]
        out = []
        for hand in self.propose(frame):
            if qualify and not qualifies(hand, shape, self.settings):
                continue
            presence = self.landmarks.presence(frame, hand)
            if presence >= self.settings.hand_presence:
                out.append(Hand(hand.quad, hand.score, hand.palm, presence))
        return out


@dataclass
class _Remembered:
    """A polygon left behind by a hand, and when it stops being protected."""

    poly: tuple
    expires: int


def build(hands: dict[int, list[Hand]], shifts: Optional[Sequence[tuple[float, float]]],
          settings: Settings, frames: int, fps: float,
          shape: Optional[tuple[int, int]] = None) -> list[ZoneFrame]:
    """One `ZoneFrame` per frame, from the hands found and the camera shift.

    `hands` may be sparse: with `zone_stride` above 1 the detector runs on
    some frames only, and the frames between them are covered by memory, which
    is why memory is what makes a stride affordable at all.

    A remembered polygon is moved by each frame's camera shift as it goes, so
    it follows the object rather than staying on the pixels. That is a
    translation only, because `motion.estimate_shift` measures a translation
    only; under rotation or zoom it drifts, which is why the memory is short
    and why report section 17 measures the drift.
    """
    if not settings.zone:
        return [ZoneFrame() for _ in range(frames)]
    shape = shape or (0, 0)
    memory = max(0, int(round(settings.zone_memory * max(1e-6, fps))))
    out: list[ZoneFrame] = []
    kept: list[_Remembered] = []
    for i in range(frames):
        dx, dy = (shifts[i] if shifts is not None and i < len(shifts) else (0.0, 0.0))
        if dx or dy:
            for r in kept:
                r.poly = moved(r.poly, dx, dy)
        kept = [r for r in kept
                if r.expires > i and (not shape[0] or inside_frame(r.poly, shape))]

        live, bare = [], []
        for hand in hands.get(i, ()):
            if shape[0] and not qualifies(hand, shape, settings):
                continue
            live.append(polygon_for(hand, settings, shape) if shape[0]
                        else tuple(hand.grown(hand_scale(settings)).quad))
            bare.append(tuple(hand.grown(settings.zone_face_scale).quad))
        # A hand back on a thing it put down refreshes that memory rather than
        # stacking a second polygon on the same object.
        for poly in live:
            refreshed = False
            for r in kept:
                if overlaps(poly, r.poly):
                    r.poly, r.expires, refreshed = poly, i + memory, True
                    break
            if not refreshed and memory:
                kept.append(_Remembered(poly, i + memory))
        # What is live this frame is live, not remembered: a polygon is in one
        # list or the other, never both, so the alphas do not double count.
        remembered = tuple(r.poly for r in kept
                           if not any(overlaps(r.poly, poly) for poly in live))
        out.append(ZoneFrame(tuple(live), remembered, tuple(bare)))
    return out


def alpha_for(shape: tuple[int, int], polygons: Iterable[Sequence],
              feather: float) -> np.ndarray:
    """Soft mask, float32 in [0, 1], 1 everywhere inside every polygon.

    Drawn **grown** by the feather and then blurred, exactly as
    `redact.build_alpha` draws a mask, and for the same reason: a blur moves
    the half way point of the ramp to where the shape's edge was, so a shape
    drawn at its true size and blurred is only half covered at its own
    boundary.

    Getting this backwards is not a rounding error. Drawn shrunk and blurred,
    the alpha reads 0.137 two pixels inside the edge of a 200 px square, so 86
    percent of a mask survived in a band about twelve pixels wide all the way
    around every hand, and the zone's promise held only in its middle. The
    output side check found it on three of the four sample files before anyone
    looked; report section 17 has the numbers.

    Erring outwards protects a little more than was asked, which is the right
    direction for a promise that is supposed to be absolute.
    """
    h, w = shape
    canvas = np.zeros((h, w), np.uint8)
    drawn = False
    for poly in polygons:
        q = np.asarray(poly, np.float64)
        if len(q) < 3 or polygon_area(q) <= 0:
            continue
        if feather > 0:
            c = q.mean(axis=0)
            spread = np.abs(q - c).max(axis=0)
            if spread.min() > 1e-6:
                q = c + (q - c) * (1.0 + feather / spread)
        cv2.fillConvexPoly(canvas, np.round(q).astype(np.int32), 255, cv2.LINE_AA)
        drawn = True
    if feather > 0 and drawn:
        canvas = cv2.GaussianBlur(canvas, (0, 0), feather / 2)
    return canvas.astype(np.float32) / 255.0


def share(shape: tuple[int, int], zone: Optional[ZoneFrame]) -> float:
    """Share of the frame the zone protects, live and remembered together."""
    if not zone:
        return 0.0
    polys = tuple(zone.live) + tuple(zone.remembered)
    if not polys:
        return 0.0
    return float((alpha_for(shape, polys, 0.0) >= 0.5).mean())


def describe(zones: Sequence[ZoneFrame], shape: tuple[int, int]) -> dict:
    """Summary numbers for the audit record."""
    shares = [share(shape, z) for z in zones]
    live = [len(z.live) for z in zones]
    return {"zone_frames": sum(1 for z in zones if z),
            "zone_share_mean": round(float(np.mean(shares)) if shares else 0.0, 5),
            "zone_share_max": round(float(np.max(shares)) if shares else 0.0, 5),
            "hands_per_frame_mean": round(float(np.mean(live)) if live else 0.0, 3)}
