"""Choose one default from the per video sweeps.

`eval/sweep.py` picks a winner for one video. A default has to hold on every
video, and on 2026-09-12 the first two files package F2 measured disagreed:
`004100` chose `verify` on and `004310` chose it off. Off masks 0.39 percent
of the wearer's hand pixels on `004100` and 0.07 percent on `004310`, so one
file refuses at the gate what the other allows. The promise inside the zone
is absolute, so a setting that breaks it anywhere cannot be the default
however well it scores where it does not.

So the rule here: a candidate has to pass every gate on every file, and among
those the fewest frames with a known face left visible, summed over the
files, wins. The same order as `eval/sweep.py`, applied to all of them at
once rather than to each in turn.

    .venv\\Scripts\\python.exe -m eval.combine VIDEO VIDEO VIDEO
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.common import CACHE_DIR, REPORT_DIR  # noqa: E402
from eval.sweep import GATES  # noqa: E402
from faceblur.settings import Settings  # noqa: E402


def key_of(changes: dict) -> str:
    """One settings choice, as a string that survives a JSON round trip.

    `det_sizes` comes back from a sweep file as a list and goes in as a
    tuple, and two rows that differ only in that are the same setting.
    """
    return json.dumps({k: list(v) if isinstance(v, (tuple, list)) else v
                       for k, v in sorted(changes.items())})


def load_sweeps(videos: list[Path]) -> dict[str, dict]:
    """Every sweep result, by video stem. A missing file is an error, not an
    empty result: a default chosen on two files out of three would look the
    same as one chosen on three."""
    out = {}
    for video in videos:
        path = CACHE_DIR / f"{Path(video).stem}.sweep.json"
        if not path.is_file():
            raise SystemExit(f"no sweep for {Path(video).name}; run eval.sweep --f2 on it")
        out[Path(video).stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def candidates(sweeps: dict[str, dict]) -> list[dict]:
    """Settings measured on every file, with their per file rows."""
    by_key: dict[str, dict[str, dict]] = {}
    for stem, sweep in sweeps.items():
        for r in sweep["results"]:
            by_key.setdefault(key_of(r["changes"]), {})[stem] = r
    whole = {k: v for k, v in by_key.items() if len(v) == len(sweeps)}
    rows = []
    for k, per_file in whole.items():
        rows.append({
            "key": k,
            "changes": json.loads(k),
            "per_file": per_file,
            "passes_everywhere": all(r["passes"] for r in per_file.values()),
            "exposed_40": sum(r["exposed_40"] for r in per_file.values()),
            "exposed_24_40": sum(r["exposed_24_40"] for r in per_file.values()),
            "score": sum(r["score"] for r in per_file.values()) / len(per_file),
            "synthetic": sum(r["synthetic_recall"] for r in per_file.values()) / len(per_file),
            "off_face_mean": max(r["off_face_mean"] for r in per_file.values()),
            "hand_worst": max((r["hand_damage_wearer"] or 0.0) for r in per_file.values()),
            "hand_other_worst": max((r["hand_damage_other"] or 0.0) for r in per_file.values()),
            "masked_mean": sum(r["masked_mean"] for r in per_file.values()) / len(per_file),
        })
    rows.sort(key=lambda r: (not r["passes_everywhere"], r["exposed_40"],
                             r["exposed_24_40"], -round(r["score"], 4),
                             r["off_face_mean"]))
    return rows


def shipped_key() -> str:
    """The build as it ships, in the same shape as a sweep row's changes."""
    s = Settings()
    return key_of({"engine": s.engine, "det_sizes": list(s.det_sizes), "conf": s.conf,
                   "verify": s.verify, "third_conf": s.third_conf,
                   "confirm_tta": s.confirm_tta, "verify_conf_view": s.verify_conf_view,
                   "verify_conf_sure": s.verify_conf_sure})


def table(rows: list[dict], stems: list[str]) -> str:
    def pct(v):
        return "n/a" if v is None else f"{100 * v:.2f}%"

    head = ("| Passes everywhere | Exposed 40+ | Exposed 24-40 | Score | Synthetic | "
            "Worst off-face mean | Worst hand damage | Masked mean | Settings |")
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        lines.append(
            f"| {'yes' if r['passes_everywhere'] else 'no'} | {r['exposed_40']} | "
            f"{r['exposed_24_40']} | {pct(r['score'])} | {pct(r['synthetic'])} | "
            f"{pct(r['off_face_mean'])} | {pct(r['hand_worst'])} | "
            f"{pct(r['masked_mean'])} | `{r['key']}` |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="+")
    ap.add_argument("--report", default="f2_combined.md",
                    help="file name under docs/ for the table")
    args = ap.parse_args()
    videos = [Path(v) for v in args.video]
    sweeps = load_sweeps(videos)
    stems = sorted(sweeps)
    rows = candidates(sweeps)
    if not rows:
        raise SystemExit("no setting was measured on every file")
    ok = [r for r in rows if r["passes_everywhere"]]
    shipped = next((r for r in rows if r["key"] == shipped_key()), None)

    print(f"{len(rows)} settings measured on all {len(stems)} files, "
          f"{len(ok)} pass every gate on every file")
    for label, r in (("shipped", shipped), ("chosen", ok[0] if ok else None)):
        if r is None:
            print(f"{label}: not among them")
            continue
        print(f"{label}: {r['key']}")
        print(f"  exposed 40+ {r['exposed_40']} fr, 24-40 {r['exposed_24_40']} fr, "
              f"synthetic {100 * r['synthetic']:.1f}%, worst off-face mean "
              f"{100 * r['off_face_mean']:.2f}%, worst hand damage "
              f"{100 * r['hand_worst']:.2f}%, passes everywhere "
              f"{'yes' if r['passes_everywhere'] else 'no'}")
        for stem in stems:
            f = r["per_file"][stem]
            print(f"    {stem[-30:]}: exposed {f['exposed_40']}/{f['exposed_24_40']}, "
                  f"synthetic {100 * f['synthetic_recall']:.1f}%, hands "
                  f"{100 * (f['hand_damage_wearer'] or 0):.2f}%, "
                  f"{'passes' if f['passes'] else 'FAILS'}")

    text = ["# F2: one default across every file", "",
            f"Settings measured on all of {', '.join(stems)}. A candidate has to pass "
            "every gate on every file; among those the fewest frames with a known face "
            "left visible wins, summed over the files.", "",
            "Gates: " + ", ".join(f"`{k}` at most {100 * v:.2f}%" for k, v in GATES.items()),
            "", table(rows, stems), ""]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / args.report).write_text("\n".join(text), encoding="utf-8")
    print(f"wrote docs/{args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
