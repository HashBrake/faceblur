"""Measure a settings choice on real footage, against consensus pseudo-labels.

Numbers per settings:

- recall: share of pseudo-faces whose identity ellipse the mask covers
- off_face: share of the frame masked where no detector sees a face. The zone
  a mask may sit in is every pseudo-face, plus every box any of the three
  detectors scored at 0.5 or more. The gate uses this number.
- off_face_strict: the same, counting only pseudo-faces as face. A real face
  the pseudo-label set missed counts against the pipeline here, so this is an
  upper bound. It stays in the report.
- off_face_detections: the part of off_face that comes from boxes a detector
  produced, as against tails and gap fills the tracker drew. A tail before a
  face enters the picture is off-face by construction; a detection box off a
  face is the thing the gate exists to catch.
- hand_damage_wearer: share of hand pixels the mask touches
- exposed_40, exposed_24_40: frames in which a face a confirmed track knew
  about is still visible (YuNet at 0.3 or more within a box size of where the
  track's motion and the camera put it, up to 30 frames from a sighting) and
  no mask covers it, for faces of 40 px and more, and of 24 to 40 px. The
  proxy for the hard gate: every such frame is a possible leak. Some are the
  detector firing on the wearer's hand next to a face; the count can be
  compared between settings, not read as a truth.
- masked: share of the frame masked, mean, p95, max

    .venv\\Scripts\\python.exe -m eval.measure VIDEO
"""
from __future__ import annotations

import argparse
import math
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.common import (CACHE_DIR, REFERENCE, build_cache, det_from_dict,
                         ellipse_mask, load_json, polygon_mask, shifts_list, size_bucket)
from faceblur.detect import confirmation, filter_candidates, iou, weak_candidates
from faceblur.pipeline import tracker_for
from faceblur.redact import alpha_with_zone
from faceblur.settings import Settings


def tracked(cache: dict, settings: Settings):
    n = max(cache["frames"]) + 1
    shape = tuple(cache["shape"])
    have = [i in cache["frames"] and i % settings.stride == 0 for i in range(n)]
    per = [filter_candidates(cache["frames"][i], settings, shape) if have[i] else []
           for i in range(n)]
    weak = [weak_candidates(cache["frames"][i], settings, shape) if have[i] else []
            for i in range(n)]
    shifts = shifts_list(cache, n) if settings.camera_comp else None
    return tracker_for(settings).run(per, weak, shifts, shape)


def dilated_box(d, factor=0.5):
    return (d.x - factor * d.w, d.y - factor * d.h, d.w * (1 + 2 * factor), d.h * (1 + 2 * factor))


def _paint(zone, box, W, H):
    x, y, w, h = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 > x0 and y1 > y0:
        zone[y0:y1, x0:x1] = True


def continuity(cache: dict, settings: Settings, per_frame, tracks, reach: int = 15) -> float:
    """Share of frames with a confirmed face still visible that carry a mask.

    For every confirmed track, look up to `reach` frames past each end. A frame
    counts as "face still visible" when YuNet fired at 0.3 or more on a box
    that overlaps the track's last box. The consensus recall cannot see these
    frames, because they are where the detectors disagree.
    """
    n = len(per_frame)
    seen = masked = 0
    for t in tracks:
        first, last = t.first, t.last
        head, foot = t.dets[first], t.dets[last]
        for i in range(first, last + 1):
            seen += 1
            masked += any(iou(d, t.dets.get(i) or head) >= 0.3 for d in per_frame[i]) \
                if i in t.dets else bool(per_frame[i])
        for anchor, rng in ((head, range(first - 1, max(-1, first - reach - 1), -1)),
                            (foot, range(last + 1, min(n, last + reach + 1)))):
            for i in rng:
                raw = cache["frames"].get(i)
                if raw is None:
                    break
                near = [d for d in raw.get("yunet", []) if d.score >= 0.3 and iou(d, anchor) >= 0.3]
                if not near:
                    break
                seen += 1
                masked += any(iou(d, near[0]) >= 0.3 for d in per_frame[i])
    return masked / seen if seen else 1.0


def exposure(cache: dict, settings: Settings, per_frame, tracks, gap: int = 45,
             slack: float = 8.0) -> dict:
    """Frames between two sightings of one face that carry no mask near it.

    A sighting is a detection that passed every check with CenterFace at 0.5
    or more (hands almost never reach that). Two sightings up to `gap` frames
    apart, of similar size, within a box size plus `slack` pixels a frame of
    each other, bracket a stretch where the face was there all along, at
    about the straight line between them. A frame in that stretch is exposed
    when no mask centre lies within one and a half box sizes of that line.
    Counted by the size of the face: 40 px and more, and 24 to 40 px. The
    reference is the detections, so the count means the same thing whatever
    the tracker does.
    """
    from faceblur.detect import filter_candidates
    n = len(per_frame)
    shape = tuple(cache["shape"])
    strong = {}
    for i in cache["frames"]:
        if i % settings.stride:
            continue
        sure = []
        for d in filter_candidates(cache["frames"][i], settings, shape):
            cf, _ = confirmation(d, cache["frames"][i], settings)
            if cf >= 0.5 and d.long_side >= 24:
                sure.append(d)
        if sure:
            strong[i] = sure
    keys = sorted(strong)
    big, mid = set(), set()
    for ai, a in enumerate(keys):
        for da in strong[a]:
            # The next sighting of this face: the nearest later frame with a
            # matching box, so a run of sightings gives short stretches.
            for b in keys[ai + 1:]:
                if b - a > gap:
                    break
                match = None
                for db in strong[b]:
                    size = max(da.long_side, db.long_side)
                    if min(da.long_side, db.long_side) / size < 0.5:
                        continue
                    if math.hypot(da.cx - db.cx, da.cy - db.cy) <= size + slack * (b - a):
                        match = db
                        break
                if match is None:
                    continue
                size = max(da.long_side, match.long_side)
                for k in range(a + 1, b):
                    t = (k - a) / (b - a)
                    cx, cy = da.cx + t * (match.cx - da.cx), da.cy + t * (match.cy - da.cy)
                    if not any(math.hypot(m.cx - cx, m.cy - cy) <= 1.5 * size for m in per_frame[k]):
                        (big if size >= 40 else mid).add(k)
                break
    return {"exposed_40": len(big), "exposed_24_40": len(mid)}


def zones_for(video, settings: Settings, cache: dict, frames: int):
    """The handled zone per frame, from the hands cache, or none at all.

    The harness builds its masks from a cache rather than from the pipeline,
    so without this it scores masks the pipeline would never apply: the zone
    is subtracted from every mask that ships. Report 17.4 found that, and
    package F2's numbers would have described a build that does not exist.

    A missing hands cache is not silently a zone of nothing. Every gate here
    is about what reaches the frame, and scoring with no zone at all reads as
    more off face masking than the build produces, so it says so and the
    caller decides.
    """
    from eval.zone import load_hands, zone_from_cache
    from faceblur.zone import ZoneFrame

    if not settings.zone:
        return [ZoneFrame() for _ in range(frames)], "the zone is off"
    data = load_hands(Path(video), settings) if video else None
    if data is None:
        return ([ZoneFrame() for _ in range(frames)],
                "no hands cache; run eval.zone --cache-hands")
    shifts = shifts_list(cache, frames) if settings.camera_comp else None
    return zone_from_cache(data, settings, frames, 30.0, shifts), ""


def evaluate(settings: Settings, cache: dict, consensus: dict, oracle: dict,
             video=None) -> dict:
    shape = tuple(cache["shape"])
    H, W = shape
    per_frame, tracks = tracked(cache, settings)
    zones, zone_note = zones_for(video, settings, cache, len(per_frame))

    covered = total = 0
    by_size = defaultdict(lambda: [0, 0])
    off, off_loose, off_det, masked = [], [], [], []
    # Split, because the rule of 2026-09-11 protects the wearer's hands and
    # not everybody's. MediaPipe's hulls are every hand in the frame, so a
    # mask landing on a bystander's hand is permitted now and counting it
    # against the build measures the scope rather than a fault. Report 17.4.
    hand_hit = hand_area = 0.0
    other_hit = other_area = 0.0
    hulls_mine = hulls_other = 0
    for key, faces in consensus["frames"].items():
        i = int(key)
        zone = zones[i] if i < len(zones) else None
        alpha = (alpha_with_zone(shape, per_frame[i], settings, zone)
                 if per_frame[i] else None)
        hard = (alpha >= 0.5) if alpha is not None else np.zeros(shape, bool)
        masked.append(float(hard.mean()))

        face_zone = np.zeros(shape, bool)
        for f in faces:
            d = det_from_dict(f)
            total += 1
            bucket = size_bucket(d.long_side)
            by_size[bucket][1] += 1
            ident = ellipse_mask(shape, d, REFERENCE)
            hit = alpha is not None and ident.any() and float(alpha[ident].mean()) >= 0.9
            covered += hit
            by_size[bucket][0] += hit
            _paint(face_zone, dilated_box(d), W, H)
        off.append(float((hard & ~face_zone).mean()))

        # Loose zone: anything any detector scored at 0.5 or more, any size,
        # or a box YuNet and CenterFace both scored at 0.35 or more. Masked
        # pixels outside this are on something no detector calls a face.
        loose_zone = face_zone.copy()
        raw = cache["frames"].get(i, {})
        for name in ("yunet", "centerface"):
            for d in raw.get(name, []):
                if d.score >= 0.5:
                    _paint(loose_zone, dilated_box(d, 0.25), W, H)
        for d in raw.get("yunet", []):
            if 0.35 <= d.score < 0.5 and confirmation(d, raw, settings)[0] >= 0.35:
                _paint(loose_zone, dilated_box(d, 0.25), W, H)
        for b in oracle["frames"].get(key, {}).get("faces", []):
            if b[4] >= 0.5:
                _paint(loose_zone, (b[0] - 0.25 * b[2], b[1] - 0.25 * b[3],
                                    1.5 * b[2], 1.5 * b[3]), W, H)
        off_loose.append(float((hard & ~loose_zone).mean()))
        dets_only = [d for d in per_frame[i] if d.source != "track"]
        hard_det = (alpha_with_zone(shape, dets_only, settings, zone) >= 0.5) \
            if dets_only else np.zeros(shape, bool)
        off_det.append(float((hard_det & ~loose_zone).mean()))

        hands = oracle["frames"].get(key, {}).get("hands", [])
        if hands:
            # One hull at a time, so each can be asked whether it is the
            # wearer's. A hull the zone touches is theirs; the rest are other
            # people's and are reported without a gate.
            # Live polygons and the strict quads, not the remembered ones:
            # memory protects text and screens only, so a face mask may land
            # in a remembered polygon without breaking any promise, and
            # charging it to the wearer would gate on a rule that does not
            # exist.
            in_zone = None
            if zone is not None and (zone.live or zone.hands):
                from faceblur.zone import alpha_for
                in_zone = alpha_for(shape, tuple(zone.live) + tuple(zone.hands),
                                    0.0) >= 0.5
            for hull in hands:
                hm = polygon_mask(shape, [hull])
                area = float(hm.sum())
                if not area:
                    continue
                mine = in_zone is not None and bool((hm & in_zone).any())
                if mine:
                    hulls_mine += 1
                    hand_area += area
                    hand_hit += float((hm & hard).sum())
                else:
                    hulls_other += 1
                    other_area += area
                    other_hit += float((hm & hard).sum())

    off_a, off_l, m_a = np.asarray(off), np.asarray(off_loose), np.asarray(masked)
    off_d = np.asarray(off_det)
    return {
        "settings": settings.to_dict(),
        "pseudo_faces": total,
        "recall": covered / total if total else None,
        "recall_by_size": {k: [v[0], v[1]] for k, v in sorted(by_size.items())},
        "off_face_strict_mean": float(off_a.mean()),
        "off_face_strict_max": float(off_a.max()),
        "off_face_mean": float(off_l.mean()),
        "off_face_max": float(off_l.max()),
        "off_face_detections_mean": float(off_d.mean()),
        # The wearer's hands, which the zone protects and the gate reads.
        "hand_damage_wearer": hand_hit / hand_area if hand_area else None,
        # Everybody else's, reported and not gated: the rule does not protect
        # them, and masking a bystander's hand is over masking rather than a
        # broken promise.
        "hand_damage_other": other_hit / other_area if other_area else None,
        "hand_hulls_wearer": hulls_mine,
        "hand_hulls_other": hulls_other,
        "zone_note": zone_note,
        "masked_mean": float(m_a.mean()),
        "masked_p95": float(np.percentile(m_a, 95)),
        "masked_max": float(m_a.max()),
        "tracks": len(tracks),
        "track_len_mean": float(np.mean([len(t) for t in tracks])) if tracks else 0.0,
        "continuity": continuity(cache, settings, per_frame, tracks),
        "frames_evaluated": len(masked),
        **exposure(cache, settings, per_frame, tracks),
    }


def load_inputs(video: Path, settings: Settings | None = None):
    """The raw cache, the pseudo-labels and the oracle.

    `settings` chooses which raw cache, because package F2 sweeps settings
    that change what is detected: the cache for `engine both` holds boxes the
    cache for `engine yunet` was never asked for.
    """
    cache = build_cache(video, settings)
    consensus = load_json(CACHE_DIR / f"{video.stem}.consensus.json")
    oracle = load_json(CACHE_DIR / f"{video.stem}.oracle.json")
    return cache, consensus, oracle


def describe(r: dict) -> str:
    rb = "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in r["recall_by_size"].items())
    hd = "n/a" if r["hand_damage_wearer"] is None else f"{100*r['hand_damage_wearer']:.2f}%"
    return (f"recall {100*(r['recall'] or 0):5.1f}%  off-face mean {100*r['off_face_mean']:.2f}% "
            f"max {100*r['off_face_max']:.2f}% (strict {100*r['off_face_strict_mean']:.2f}%, "
            f"detections {100*r['off_face_detections_mean']:.2f}%)  "
            f"exposed 40+ {r['exposed_40']} fr, 24-40 {r['exposed_24_40']} fr  "
            f"hands {hd}  continuity {100*r['continuity']:.1f}%  masked mean {100*r['masked_mean']:.2f}% "
            f"p95 {100*r['masked_p95']:.2f}% max {100*r['masked_max']:.2f}%  "
            f"tracks {r['tracks']}  [{rb}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--json", default=None, help="settings overrides as JSON")
    args = ap.parse_args()
    video = Path(args.video)
    settings = Settings(**(json.loads(args.json) if args.json else {}))
    # The settings choose the cache as well as what is done with it: a
    # `--json` that names `engine` or `det_sizes` needs the cache built for
    # them, not the default one.
    cache, consensus, oracle = load_inputs(video, settings)
    r = evaluate(settings, cache, consensus, oracle, video)
    print(describe(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
