"""Unconfirmed stretches: the yardstick for the missed-face work. No labels.

A stretch is three frames or more in which YuNet saw a box at full threshold
that nothing confirmed and no mask covers. Most are hands and objects, some
are faces the confirmers did not recognise. The count cannot say which is
which, but a change that finds more faces makes it fall while the gates
(off-face masking, hands) stay put, and a change that admits hands makes the
gates fail. Both are visible without a person looking.

    .venv\\Scripts\\python.exe -m eval.misses VIDEO [--json SETTINGS] [--images DIR]

`--images` writes one annotated frame per stretch, green for masked boxes,
red for the ones nothing confirmed, with the YuNet and CenterFace scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from eval import measure
from eval.common import CACHE_DIR, build_cache, read_frames, save_json
from faceblur.batch import unconfirmed_runs
from faceblur.detect import confirmation, iou, weak_candidates
from faceblur.settings import Settings

EDGE = 0.04     # a box this close to the border, as a share of the frame, is "at the edge"


def find(cache: dict, settings: Settings) -> list[list]:
    """Every stretch as [start, end, frames, best YuNet score, best CenterFace
    score, third detector fired, at the frame edge]."""
    per_frame, _ = measure.tracked(cache, settings)
    n = len(per_frame)
    shape = tuple(cache["shape"])
    H, W = shape
    have = [i in cache["frames"] and i % settings.stride == 0 for i in range(n)]
    weak = [weak_candidates(cache["frames"][i], settings, shape) if have[i] else []
            for i in range(n)]
    runs, _ = unconfirmed_runs(weak, per_frame, settings.conf)
    out = []
    for a, b in runs:
        best_y = best_cf = 0.0
        third = edge = False
        for i in range(a, b + 1):
            for d in weak[i]:
                if d.score < settings.conf or any(iou(d, m) >= 0.3 for m in per_frame[i]):
                    continue
                best_y = max(best_y, d.score)
                cf, t = confirmation(d, cache["frames"][i], settings)
                best_cf = max(best_cf, cf)
                third = third or t
                if d.x < EDGE * W or d.y < EDGE * H or d.x + d.w > (1 - EDGE) * W \
                        or d.y + d.h > (1 - EDGE) * H:
                    edge = True
        out.append([a, b, b - a + 1, round(best_y, 2), round(best_cf, 2), third, edge])
    return out


def summary(runs: list[list]) -> dict:
    return {
        "runs": len(runs),
        "frames": sum(r[2] for r in runs),
        "edge_runs": sum(1 for r in runs if r[6]),
        "interior_runs": sum(1 for r in runs if not r[6]),
        "near_confirmed_runs": sum(1 for r in runs if r[4] >= 0.1),
    }


def draw(video: Path, cache: dict, settings: Settings, runs: list[list], out_dir: Path) -> None:
    per_frame, _ = measure.tracked(cache, settings)
    shape = tuple(cache["shape"])
    wanted = {}
    for a, b, *_ in runs:
        # The frame in the stretch with the strongest unmasked box.
        best, best_i = -1.0, a
        for i in range(a, b + 1):
            for d in weak_candidates(cache["frames"][i], settings, shape):
                if d.score >= settings.conf and d.score > best \
                        and not any(iou(d, m) >= 0.3 for m in per_frame[i]):
                    best, best_i = d.score, i
        wanted[best_i] = (a, b)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in read_frames(video, set(wanted)):
        a, b = wanted[i]
        img = cv2.resize(frame, (800, int(round(800 * shape[0] / shape[1]))))
        f = 800 / shape[1]
        for m in per_frame[i]:
            cv2.rectangle(img, (int(m.x * f), int(m.y * f)), (int((m.x + m.w) * f), int((m.y + m.h) * f)),
                          (255, 200, 0), 1)
        for d in cache["frames"][i].get("yunet", []):
            if d.score < settings.conf_weak:
                continue
            covered = any(iou(d, m) >= 0.3 for m in per_frame[i])
            col = (0, 255, 0) if covered else (0, 0, 255)
            cf, third = confirmation(d, cache["frames"][i], settings)
            cv2.rectangle(img, (int(d.x * f), int(d.y * f)), (int((d.x + d.w) * f), int((d.y + d.h) * f)),
                          col, 2)
            cv2.putText(img, f"y{d.score:.2f} cf{cf:.2f}{' u' if third else ''}",
                        (int(d.x * f), int(d.y * f) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(img, f"frame {i}  stretch {a}-{b}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"miss_{a:05d}_{b:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--json", default=None, help="settings overrides as JSON")
    ap.add_argument("--images", default=None, help="folder for one annotated frame per stretch")
    args = ap.parse_args()
    video = Path(args.video)
    settings = Settings(**(json.loads(args.json) if args.json else {}))
    cache = build_cache(video)
    runs = find(cache, settings)
    s = summary(runs)
    print(f"{s['runs']} unconfirmed stretches, {s['frames']} frames: {s['edge_runs']} at the edge, "
          f"{s['interior_runs']} interior, {s['near_confirmed_runs']} where CenterFace scored 0.1 or more")
    for a, b, n, y, cf, third, edge in runs:
        print(f"  {a:5d}-{b:5d}  {n:3d} fr  yunet {y:.2f}  centerface {cf:.2f}"
              f"{'  ultraface' if third else ''}{'  edge' if edge else ''}")
    save_json(CACHE_DIR / f"{video.stem}.misses.json",
              {"settings": settings.to_dict(), "summary": s, "runs": runs})
    if args.images:
        draw(video, cache, settings, runs, Path(args.images))
    return 0


if __name__ == "__main__":
    sys.exit(main())
