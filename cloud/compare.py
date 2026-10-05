"""Compare two runs of the same videos, record by record.

The cloud run uses CUDA where this PC uses DirectML. The models are the same
files, but two runtimes do not round the same way, and a box that scores 0.501
on one can score 0.499 on the other. This says how far apart the two runs are
on what matters, before the cloud run is trusted with the whole set.

    python cloud/compare.py LOCAL_OUT CLOUD_OUT

Each folder is searched for `*.mp4.json` records, delivered or in quarantine,
and records are matched by source name.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIELDS = ["status", "frames", "detections", "tracks", "frames_with_mask", "masked_mean",
          "masked_p95", "zone_share_mean", "hands_per_frame_mean", "residual_faces",
          "residual_max_px", "quarantined", "held_back_for", "detect_seconds",
          "encode_seconds", "check_seconds", "wall_seconds"]


def records(folder: Path) -> dict[str, dict]:
    found = {}
    for p in folder.rglob("*.mp4.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        found[r["source"]] = r
    return found


def main(argv: list[str]) -> int:
    a, b = records(Path(argv[0])), records(Path(argv[1]))
    names = sorted(set(a) | set(b))
    for name in names:
        print(name)
        if name not in a or name not in b:
            print(f"  only in {'the first' if name in a else 'the second'} run")
            continue
        for f in FIELDS:
            x, y = a[name].get(f), b[name].get(f)
            flag = "" if x == y else "   <- differs"
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x and x == x:
                flag += f" ({(y - x) / abs(x):+.1%})" if x != y else ""
            print(f"  {f:22s} {x!s:>14} {y!s:>14}{flag}")
        print(f"  {'compute':22s} {a[name].get('compute', {}).get('yunet', '')!s:>14} "
              f"{b[name].get('compute', {}).get('yunet', '')!s:>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
