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
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()
    results = []
    for name in args.video:
        result = score(Path(name), min_share=args.min_share)
        results.append(result)
        print(describe(result), flush=True)
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
