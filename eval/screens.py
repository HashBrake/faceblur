"""The shipped screen class, measured against a witness that is not it.

Section 15.4 of `docs/report.md` can give screens a precision figure and no
recall figure at all, and says so: with one detector and nobody to check it, a
screen the detector never finds is not counted anywhere. That is the same hole
the face side had before MediaPipe, and this closes it the same way.

`eval/oracle_owl.py` writes what OWLv2 sees. This runs the shipped screen
class over the same video, applies the rule the pipeline actually applies, and
asks two questions on the frames where the oracle has an opinion:

- **recall.** Of the screens the oracle finds, how many does the shipped rule
  mask? This is the number section 15.4 could not give.
- **precision.** Of the screens the shipped rule masks, how many does the
  oracle also call a screen?

Neither is truth. The oracle is one model's opinion at a threshold somebody
chose, so both numbers are reported across a range of thresholds rather than
at one, and the reader can see how much the answer depends on that choice.
Where the two disagree, neither is right by definition.

Two things are done to the oracle's boxes before they are counted, and both
matter to the number:

- **They are suppressed.** OWLv2 returns overlapping proposals for one object
  and does no suppression of its own. Counting them raw would make recall a
  measure of how many proposals the shipped rule happened to cover.
- **They are floored by size.** A screen at 0.1 percent of the frame shows
  nothing anybody could read, and the shipped class is not trying to find one.
  The default floor is 0.4 percent, which section 15.2 records as the size the
  real wall television reads at.

    .venv\\Scripts\\python.exe -m eval.screens VIDEO [--floor 0.2] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from eval.oracle_owl import SCREEN_PROMPTS, cache_path as owl_path  # noqa: E402
from faceblur.detect import Detection, iou, nms_indices  # noqa: E402
from faceblur.screens import hold  # noqa: E402
from faceblur.settings import Settings  # noqa: E402

# How much of a frame a screen has to cover before it counts for either
# number. Section 15.2: the real wall television reads at 0.4 to 1.6 percent
# and the false runs at 2.4 to 35, so this is below everything real that was
# looked at and above nothing at all.
MIN_SHARE = 0.004
# Overlap at which two boxes are the same screen. Loose on purpose: the
# shipped box is grown by screen_pad and the oracle's is tight to the glass,
# so the same monitor lands as two boxes that differ by a bezel.
MATCH_IOU = 0.3
# Suppression over the oracle's own proposals, which it does not do itself.
ORACLE_NMS = 0.5
# The thresholds recall and precision are reported at. The answer moves with
# this, which is the point of showing several.
FLOORS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def oracle_screens(data: dict, frame: int, floor: float,
                   min_share: float = MIN_SHARE) -> list[Detection]:
    """The oracle's screens on one frame: suppressed, floored, as Detections."""
    h, w = data["shape"]
    rows = [r for r in data["frames"].get(str(frame), [])
            if r["label"] in SCREEN_PROMPTS and r["score"] >= floor]
    keep = []
    for r in rows:
        x0, y0, x1, y1 = r["box"]
        bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
        if bw * bh >= min_share * w * h:
            keep.append((Detection(x0, y0, bw, bh, r["score"], None,
                                   r["label"].removeprefix("a "), True, r["score"],
                                   "screen"), r["score"]))
    if not keep:
        return []
    boxes = np.array([[d.x, d.y, d.w, d.h] for d, _ in keep], dtype=np.float64)
    scores = np.array([s for _, s in keep], dtype=np.float64)
    return [keep[i][0] for i in nms_indices(boxes, scores, ORACLE_NMS)]


def shipped_masks(video: Path, settings: Settings, stride: int = 1
                  ) -> tuple[dict[int, list[Detection]], dict[int, list[Detection]], int]:
    """(what would be masked, what the detector reported, frames in the video).

    The whole rule, not just the detector: the size cap and the pad inside
    `ScreenDetector.detect`, then `screens.hold` for the persistence rule, the
    gap filling and the tail. Persistence counts consecutive source frames, so
    the detector runs on every frame even when only some are compared.
    """
    import cv2

    from faceblur.batch import _screens

    detector = _screens(settings)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found: dict[int, list[Detection]] = {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            hits = detector.detect(frame)
            if hits:
                found[index] = hits
        index += 1
    cap.release()
    return hold(found, settings, total), found, total


# Package S2, 2026-09-12. The shipped floor is `screen_conf` 0.5 and section
# 16.2 measured 0 to 23 percent recall against the oracle. Under the rule of
# 2026-09-11 a missed screen outside the handled zone is a leak and a masked
# table is a cost to the environment rather than to privacy, so the floor is
# asked again, with the zone on.
S2_CONFS = (0.5, 0.4, 0.3, 0.25)


def zones_for(video: Path, settings: Settings, frames: int):
    """The handled zone per frame, or None when there is no hands cache.

    A screen the wearer is holding is protected whatever the floor, so it is
    neither a mask nor a miss, and it has to be taken out of both counts
    before either means anything.
    """
    from eval.zone import load_hands, zone_from_cache

    held = load_hands(video, settings)
    if held is None:
        return None
    return zone_from_cache(held, settings, frames, 30.0, None)


def protected_by(zone, det: Detection, shape) -> bool:
    """Is this box mostly inside what the zone protects for a screen?

    Live polygons and remembered ones both, because `REMEMBERED_PROTECTS`
    holds screens: a phone the wearer put down a second ago is still the
    thing they were handling.
    """
    from faceblur.zone import alpha_for

    if zone is None:
        return False
    polys = tuple(zone.live) + tuple(zone.remembered)
    if not polys:
        return False
    import numpy as np

    protected = alpha_for(shape, polys, 0.0) >= 0.5
    x0, y0 = max(0, int(det.x)), max(0, int(det.y))
    x1 = min(shape[1], int(det.x + det.w))
    y1 = min(shape[0], int(det.y + det.h))
    if x1 <= x0 or y1 <= y0:
        return False
    inside = protected[y0:y1, x0:x1]
    return float(inside.mean()) >= 0.5


def masked_area(video: Path, settings: Settings, per_frame: dict, zones,
                shape, frames: int, sample: int = 120) -> dict:
    """How much of the frame the screen masks take, and how often that is over
    the budget the pipeline ships.

    Sampled rather than exhaustive: the masks are polygons and the answer is
    a share, so a spread of frames over the file gives the same number for a
    fraction of the work.
    """
    import numpy as np

    from faceblur.redact import alpha_with_zone

    if frames <= 0:
        return {"masked_mean": 0.0, "masked_max": 0.0, "over_budget": 0, "frames": 0}
    picks = ([int(round(i * (frames - 1) / max(1, sample - 1))) for i in range(sample)]
             if sample < frames else list(range(frames)))
    shares = []
    for i in sorted(set(picks)):
        dets = per_frame.get(i) or []
        if not dets:
            shares.append(0.0)
            continue
        zone = zones[i] if zones is not None and i < len(zones) else None
        alpha = alpha_with_zone(shape, dets, settings, zone)
        shares.append(float((alpha >= 0.5).mean()))
    a = np.asarray(shares) if shares else np.zeros(1)
    return {"masked_mean": float(a.mean()), "masked_max": float(a.max()),
            "over_budget": int((a > settings.mask_budget).sum()),
            "frames": len(shares)}


def sweep(video: Path, confs=S2_CONFS, floors=FLOORS,
          min_share: float = MIN_SHARE) -> dict:
    """Package S2: recall, precision, masked share and budget per `screen_conf`.

    The detector runs once, at the lowest floor asked for, and the higher
    floors are taken by filtering its boxes. That is the same answer: NMS
    keeps the highest scoring box of a cluster, so a box above 0.5 can never
    have been suppressed by one below it.
    """
    import cv2

    base = Settings(mask=("face", "screen"), screen_conf=min(confs))
    path = owl_path(video)
    if not path.is_file():
        raise SystemExit(
            f"no oracle cache at {path}. Run\n"
            f"  .venv-oracle\\Scripts\\python.exe eval\\oracle_owl.py {video} --stride 5")
    data = json.loads(path.read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(video))
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    _, raw, total = shipped_masks(video, base)
    zones = zones_for(video, base, total)
    frames = sorted(int(k) for k in data["frames"])

    rows = []
    for conf in confs:
        settings = base.with_changes(screen_conf=conf)
        kept = {i: [d for d in dets if d.score >= conf] for i, dets in raw.items()}
        kept = {i: dets for i, dets in kept.items() if dets}
        masked = hold(kept, settings, total)
        # Once per floor of the detector, not once per floor of the oracle:
        # what is masked depends on `screen_conf` alone, and the oracle's
        # floor only decides which of its boxes are worth comparing against.
        area = {f"area_{k}": v for k, v in masked_area(
            video, settings, masked, zones, shape, total).items()}
        for floor in floors:
            hit = missed = set_aside = 0
            matched = unmatched = in_zone = 0
            for i in frames:
                zone = zones[i] if zones is not None and i < len(zones) else None
                theirs = oracle_screens(data, i, floor, min_share)
                ours = masked.get(i, [])
                for o in theirs:
                    if any(iou(o, m) >= MATCH_IOU for m in ours):
                        hit += 1
                    elif protected_by(zone, o, shape):
                        set_aside += 1
                    else:
                        missed += 1
                for m in ours:
                    if protected_by(zone, m, shape):
                        in_zone += 1
                    elif any(iou(o, m) >= MATCH_IOU for o in theirs):
                        matched += 1
                    else:
                        unmatched += 1
            rows.append({
                "screen_conf": conf, "floor": floor,
                "oracle_screens": hit + missed, "recalled": hit,
                "recall": hit / (hit + missed) if hit + missed else None,
                "handled_set_aside": set_aside,
                "masked_boxes": matched + unmatched, "confirmed": matched,
                "precision": matched / (matched + unmatched) if matched + unmatched else None,
                "masked_in_zone": in_zone,
                **area,
            })
    return {"video": video.name, "frames_compared": len(frames), "frames_total": total,
            "stride": data["stride"], "min_share": min_share,
            "zone": zones is not None, "rows": rows}


def describe_sweep(result: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{100 * v:.0f}%"

    out = [f"{result['video']}: {result['frames_compared']} frames compared of "
           f"{result['frames_total']}"
           + ("" if result["zone"] else ", NO HANDS CACHE so the zone is empty"),
           "| `screen_conf` | Oracle floor | Oracle screens | Recall | Masked boxes | "
           "Precision | Handled, set aside | Masked frame, mean | Max | Over budget |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        out.append(
            f"| {r['screen_conf']} | {r['floor']} | {r['oracle_screens']} | "
            f"{pct(r['recall'])} | {r['masked_boxes']} | {pct(r['precision'])} | "
            f"{r['handled_set_aside'] + r['masked_in_zone']} | "
            f"{100 * r['area_masked_mean']:.2f}% | {100 * r['area_masked_max']:.2f}% | "
            f"{r['area_over_budget']} of {r['area_frames']} |")
    return "\n".join(out)


def score(video: Path, settings: Settings | None = None,
          floors=FLOORS, min_share: float = MIN_SHARE) -> dict:
    """Recall and precision of the shipped rule against the oracle."""
    settings = settings or Settings(mask=("face", "screen"))
    path = owl_path(video)
    if not path.is_file():
        raise SystemExit(
            f"no oracle cache at {path}. Run\n"
            f"  .venv-oracle\\Scripts\\python.exe eval\\oracle_owl.py {video} --stride 5")
    data = json.loads(path.read_text(encoding="utf-8"))
    masked, raw, total = shipped_masks(video, settings)
    frames = sorted(int(k) for k in data["frames"])

    rows = []
    for floor in floors:
        hit = missed = 0
        matched = unmatched = 0
        for i in frames:
            theirs = oracle_screens(data, i, floor, min_share)
            ours = masked.get(i, [])
            for o in theirs:
                if any(iou(o, m) >= MATCH_IOU for m in ours):
                    hit += 1
                else:
                    missed += 1
            for m in ours:
                if any(iou(o, m) >= MATCH_IOU for o in theirs):
                    matched += 1
                else:
                    unmatched += 1
        rows.append({"floor": floor,
                     "oracle_screens": hit + missed,
                     "recalled": hit,
                     "recall": hit / (hit + missed) if hit + missed else None,
                     "masked_boxes": matched + unmatched,
                     "confirmed": matched,
                     "precision": matched / (matched + unmatched)
                     if matched + unmatched else None})
    return {"video": video.name,
            "frames_compared": len(frames),
            "frames_total": total,
            "stride": data["stride"],
            "oracle": {"model": data["model"], "revision": data["revision"],
                       "threshold": data["threshold"]},
            "min_share": min_share,
            "match_iou": MATCH_IOU,
            "raw_detections": sum(len(v) for v in raw.values()),
            "masked_frames": len(masked),
            "rows": rows}


def describe(result: dict) -> str:
    out = [f"{result['video']}: {result['frames_compared']} frames compared of "
           f"{result['frames_total']}, oracle at stride {result['stride']}",
           "| Oracle floor | Oracle screens | Masked by the rule | Recall | "
           "Boxes the rule masks | Oracle agrees | Precision |",
           "|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        def pct(v):
            return "n/a" if v is None else f"{100 * v:.0f} %"
        out.append(f"| {r['floor']} | {r['oracle_screens']} | {r['recalled']} | "
                   f"{pct(r['recall'])} | {r['masked_boxes']} | {r['confirmed']} | "
                   f"{pct(r['precision'])} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="+")
    ap.add_argument("--min-share", type=float, default=MIN_SHARE,
                    help=f"smallest oracle screen that counts, as a share of the "
                         f"frame (default: {MIN_SHARE})")
    ap.add_argument("--s2", action="store_true",
                    help="package S2: sweep screen_conf 0.5 to 0.25 with the "
                         "handled zone on, and report the masked share and the "
                         "frames over budget as well")
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()
    results = []
    for name in args.video:
        if args.s2:
            result = sweep(Path(name), min_share=args.min_share)
            results.append(result)
            print(describe_sweep(result), flush=True)
            print()
            continue
        result = score(Path(name), min_share=args.min_share)
        results.append(result)
        print(describe(result), flush=True)
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
