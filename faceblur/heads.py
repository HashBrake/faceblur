"""The head backstop: a face mask for people the face detectors cannot see.

YuNet and CenterFace are trained on upright faces. At the fisheye rim of the
camera a face is turned 60 to 90 degrees, or cut in half by the frame edge,
and neither finds it (docs/privacy_module_fieldtest.md, defect 1). The YOLOX
model that already runs for screens also knows the COCO class `person`, and a
person is found at the rim when the face is not, because a body is large and
does not depend on the face's angle.

So each person box gives a head region: the top `head_frac` of the box, the
middle `head_width_frac` of its width. The region is a rectangle and it is
masked as one, a polygon of kind "face" (its `quad`), not as the upright
ellipse a face box without landmarks gets: a head turned by the lens is not
an upright oval, and the rectangle is what the person box measured.

A box cut off by the left or right edge of the frame is a person leaning in
from the rim, and the lens tips him so that his head is at the top corner on
the cut side (the pilot's AM 032259 frame 2653: box [0, 20, 706, 921], face
in its top-left corner). There the region sits against the cut edge, and is
as tall as it is wide, because the box's height is no longer the body's.

By default (`head_edge_only`) only such edge-cut boxes give a head region
at all. Measured on the pilot clips and 004100, a box away from the edge
puts its top strip on a back, a shoulder or the wall above a seated person
about half the time, and the face consensus already finds most of those
faces.

Three things are not masked:

- the wearer's own body: a person box in the bottom band of the frame that
  is wide or starts in the lower half (`is_wearer`) has no head in view;
- a region a tracked face already covers, on that frame: the face mask hides
  it, and a second mask over it only adds area;
- a region inside the handled zone, which `redact` leaves clear as it does
  every face.

Regions are held over time the way plates are (`screens.hold` under the
`head_*` settings), so a person seen once is not masked.
"""
from __future__ import annotations

from .detect import Detection, iou
from .settings import Settings

SOURCE = "person_head"
# Within this share of the frame width of a side edge counts as cut by it:
# YOLOX boxes at the edge stop a few pixels short or run past it.
EDGE_SLACK = 0.01


def head_region(person: Detection, settings: Settings,
                shape: tuple[int, int] | None = None) -> Detection:
    """The head rectangle of a person box, as a face polygon.

    With `shape` (height, width) the region is clipped to the frame, and a
    box cut by the left or right edge puts it against that edge.
    """
    w = person.w * settings.head_width_frac
    x0 = person.x + (person.w - w) / 2
    h = person.h * settings.head_frac
    if shape is not None:
        H, W = shape
        left = person.x <= W * EDGE_SLACK
        right = person.x + person.w >= W * (1 - EDGE_SLACK)
        if left != right:
            x0 = person.x if left else person.x + person.w - w
            h = min(person.h, max(h, w))
    y0 = person.y
    x1, y1 = x0 + w, y0 + h
    if shape is not None:
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(W), x1), min(float(H), y1)
    x1, y1 = max(x1, x0 + 1.0), max(y1, y0 + 1.0)
    quad = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return Detection(x0, y0, x1 - x0, y1 - y0, person.score, None, SOURCE, False,
                     person.score, "face", quad)


def is_wearer(person: Detection, shape: tuple[int, int], settings: Settings) -> bool:
    """Is this box the wearer's own body: in the bottom band, and wide or
    starting in the lower part of the frame."""
    H, W = shape
    if person.y + person.h < H * settings.head_wearer_bottom:
        return False
    return person.w > settings.head_wearer_frac * W or person.y >= H * settings.head_wearer_top


def at_side_edge(person: Detection, shape: tuple[int, int]) -> bool:
    """Is this box cut by the left or right edge of the frame."""
    W = shape[1]
    return person.x <= W * EDGE_SLACK or person.x + person.w >= W * (1 - EDGE_SLACK)


def regions(people: list[Detection], shape: tuple[int, int],
            settings: Settings) -> list[Detection]:
    """Head regions of one frame's people, the wearer left out, and with
    `head_edge_only` only from boxes cut by a side edge."""
    return [head_region(p, settings, shape) for p in people
            if not is_wearer(p, shape, settings)
            and (not settings.head_edge_only or at_side_edge(p, shape))]


def covered(head: Detection, faces: list[Detection]) -> bool:
    """Does a face mask already hide this head: overlapping at IoU 0.3, or a
    face whose centre is inside the region."""
    for face in faces:
        if iou(head, face) >= 0.3:
            return True
        if head.x <= face.cx <= head.x + head.w and head.y <= face.cy <= head.y + head.h:
            return True
    return False


def hold(found: dict[int, list[Detection]], settings: Settings,
         frames: int) -> dict[int, list[Detection]]:
    """Head regions over time, on the pattern of `plates.hold`."""
    from .screens import hold as hold_screens

    return hold_screens(found, settings.with_changes(
        track_iou=settings.head_link_iou, screen_gap=settings.head_gap,
        screen_tail=settings.head_tail, screen_min_run=settings.head_min_run), frames)


def uncovered(held: dict[int, list[Detection]],
              per_frame: list[list[Detection]]) -> dict[int, list[Detection]]:
    """`held` less every region a face mask in `per_frame` already covers.

    `per_frame` is the tracker's final masks: a face is any box of kind
    "face" that is not itself a head region.
    """
    out: dict[int, list[Detection]] = {}
    for i, heads in held.items():
        if not 0 <= i < len(per_frame):
            continue
        faces = [d for d in per_frame[i] if d.kind == "face" and d.source != SOURCE]
        kept = [h for h in heads if not covered(h, faces)]
        if kept:
            out[i] = kept
    return out
