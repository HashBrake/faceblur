"""Pseudo-labels: faces that independent detectors agree on. No human involved.

A pseudo-face is a YuNet or CenterFace box, at a high threshold, that at least
one other detector (the other of those two, or BlazeFace) also fired on. Boxes
keep their landmarks so the identity ellipse can be measured.

    .venv\\Scripts\\python.exe -m eval.consensus VIDEO --stride 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.common import CACHE_DIR, build_cache, iou, load_json, save_json


def consensus(cache: dict, oracle: dict, stride: int, yunet_conf: float = 0.6,
              cf_conf: float = 0.5, blaze_conf: float = 0.5, agree_iou: float = 0.4) -> dict:
    frames = {}
    for idx, raw in cache["frames"].items():
        if idx % stride != 0:
            continue
        yn = [d for d in raw.get("yunet", []) if d.score >= yunet_conf]
        cf = [d for d in raw.get("centerface", []) if d.score >= cf_conf]
        bz = [b for b in oracle["frames"].get(str(idx), {}).get("faces", [])
              if b[4] >= blaze_conf]
        found = []
        for d in yn:
            support = sum(1 for c in cf if iou(d, c) >= agree_iou) + \
                      sum(1 for b in bz if iou(d, b) >= agree_iou)
            if support >= 1:
                found.append(d.to_dict() | {"support": support + 1})
        for c in cf:
            if any(iou(c, d) >= agree_iou for d in yn):
                continue  # already represented by the YuNet box
            support = sum(1 for b in bz if iou(c, b) >= agree_iou)
            if support >= 1:
                found.append(c.to_dict() | {"support": support + 1})
        frames[str(idx)] = found
    return {"video": cache["video"], "stride": stride, "shape": list(cache["shape"]),
            "frames": frames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    video = Path(args.video)
    cache = build_cache(video)
    oracle = load_json(CACHE_DIR / f"{video.stem}.oracle.json")
    result = consensus(cache, oracle, args.stride)
    out = CACHE_DIR / f"{video.stem}.consensus.json"
    save_json(out, result)
    n = sum(len(v) for v in result["frames"].values())
    print(f"{video.name}: {len(result['frames'])} frames, {n} pseudo-faces -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
