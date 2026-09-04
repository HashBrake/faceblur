"""Choose the settings automatically.

Sweeps the detection and tracking knobs on real footage against the consensus
pseudo-labels and the hand regions, then the mask ellipse on the synthetic set.
Picks the highest recall that stays inside the gates, and writes the report.

    .venv\\Scripts\\python.exe -m eval.sweep VIDEO
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import pickle
import sys
from pathlib import Path

from eval import measure, synthetic
from eval.common import CACHE_DIR, REPORT_DIR, build_cache, load_json, save_json
from faceblur.settings import Settings

GATES = {"off_face_mean": 0.003, "off_face_max": 0.03, "hand_damage": 0.001}

_INPUTS = None
_SYNTH = None


def _init(video):
    global _INPUTS, _SYNTH
    video = Path(video)
    _INPUTS = measure.load_inputs(video)
    _SYNTH = pickle.loads((CACHE_DIR / f"{video.stem}.synthetic.pkl").read_bytes())


def _run(changes):
    s = Settings(**changes)
    r = measure.evaluate(s, *_INPUTS)
    y = synthetic.evaluate(s, _SYNTH)["recall"]
    r["synthetic"] = {k: v[2] for k, v in y.items()}
    r["synthetic_recall"] = y["all"][2]
    # The number the choice is made on: both kinds of recall, equal weight.
    r["score"] = 0.5 * (r["recall"] or 0.0) + 0.5 * r["synthetic_recall"]
    r["changes"] = changes
    return r


def passes(r: dict) -> bool:
    if r["off_face_mean"] > GATES["off_face_mean"] or r["off_face_max"] > GATES["off_face_max"]:
        return False
    if r["hand_damage"] is not None and r["hand_damage"] > GATES["hand_damage"]:
        return False
    return True


def grid() -> list[dict]:
    out = []
    for conf, cap, mt, gap, weak in itertools.product([0.5, 0.6, 0.7], [0.15, 0.20],
                                                      [2, 3], [2, 3, 5], [0.2, 0.3, 0.4]):
        out.append({"conf": conf, "max_face_frac": cap, "min_track": mt, "max_gap": gap,
                    "conf_weak": weak})
    for conf in (0.5, 0.6, 0.7):
        out.append({"conf": conf, "conf_weak": conf, "max_face_frac": 0.15,
                    "min_track": 2, "max_gap": 3})   # no continuation
    # Reference points: the first build's behaviour, and no verification.
    out.append({"conf": 0.25, "conf_weak": 0.25, "det_sizes": (640, 1280), "verify": False,
                "max_face_frac": 1.0, "min_track": 1, "max_gap": 6, "tail": 6,
                "pad": 0.30, "ellipse_w": 1.6, "ellipse_h": 1.6, "feather": 0.0})
    out.append({"conf": 0.5, "verify": False})
    out.append({"engine": "both", "conf": 0.5})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    video = Path(args.video)

    build_cache(video)
    synthetic.generate(video, load_json(CACHE_DIR / f"{video.stem}.consensus.json"))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(str(video),)) as pool:
        results = list(pool.imap_unordered(_run, grid()))
    for r in results:
        r["passes"] = passes(r)
    ok = [r for r in results if r["passes"] and r["recall"] is not None]
    ok.sort(key=lambda r: (-round(r["score"], 4), r["off_face_mean"], r["masked_mean"]))
    # Tie break: among settings within one point of the best score, prefer the
    # one that keeps confirmed faces covered longest. The two recalls cannot
    # see that, because it lives where the detectors disagree.
    if ok:
        best = ok[0]["score"]
        near = [r for r in ok if r["score"] >= best - 0.01]
        near.sort(key=lambda r: (-round(r["continuity"], 3), r["off_face_mean"], r["masked_mean"]))
        ok = near + [r for r in ok if r not in near]
    chosen = ok[0] if ok else None

    # Mask ellipse, on the synthetic set, at the chosen detection settings.
    consensus = load_json(CACHE_DIR / f"{video.stem}.consensus.json")
    data = synthetic.generate(video, consensus)
    base = dict(chosen["changes"]) if chosen else {}
    ellipse_rows = []
    for ew, eh in itertools.product([1.0, 1.1, 1.2], [1.05, 1.15, 1.25]):
        s = Settings(**base, ellipse_w=ew, ellipse_h=eh)
        r = synthetic.evaluate(s, data)
        ellipse_rows.append({"ellipse_w": ew, "ellipse_h": eh,
                             "recall": r["recall"]["all"][2],
                             "by": {k: v[2] for k, v in r["recall"].items()},
                             "area": r["mask_over_target_area"]})
    ellipse_rows.sort(key=lambda r: (-round(r["recall"], 3), r["area"] or 9))
    best_ellipse = ellipse_rows[0]
    synth_at_default = synthetic.evaluate(Settings(**base), data)

    out = {"video": video.name, "gates": GATES, "results": results, "chosen": chosen,
           "ellipse": ellipse_rows, "synthetic_at_chosen": synth_at_default}
    save_json(CACHE_DIR / f"{video.stem}.sweep.json", out)
    write_report(video, results, chosen, ellipse_rows, synth_at_default)
    if chosen:
        print("CHOSEN:", json.dumps(chosen["changes"]), measure.describe(chosen),
              f"synthetic {100*chosen['synthetic_recall']:.1f}%")
        print("ELLIPSE:", best_ellipse)
    else:
        print("no setting passed the gates; best by off-face:",
              measure.describe(min(results, key=lambda r: r["off_face_mean"])))
    return 0


def write_report(video, results, chosen, ellipse_rows, synth) -> None:
    def pct(v):
        return "n/a" if v is None else f"{100*v:.2f}%"

    lines = ["# Precision report", "",
             f"Automatic sweep on `{video.name}`. No human labels. See `eval/` for the method.",
             "", "## Gates", "",
             f"- off-face masked area, where no detector sees a face: mean at most "
             f"{pct(GATES['off_face_mean'])}, any frame at most {pct(GATES['off_face_max'])}",
             f"- hand pixels touched: at most {pct(GATES['hand_damage'])}",
             "- among settings inside the gates, the highest mean of consensus recall "
             "and synthetic recall wins; within one point of the best, the setting "
             "that keeps confirmed faces covered longest (continuity) wins", "",
             "## Chosen", ""]
    if chosen:
        lines += ["```", json.dumps(chosen["changes"]), "```", "",
                  f"Recall on pseudo-faces {pct(chosen['recall'])}, synthetic recall "
                  f"{pct(chosen['synthetic_recall'])}, continuity {pct(chosen['continuity'])}, off-face mean "
                  f"{pct(chosen['off_face_mean'])}, max {pct(chosen['off_face_max'])} "
                  f"(strict {pct(chosen['off_face_strict_mean'])}), "
                  f"hands {pct(chosen['hand_damage'])}, frame masked mean "
                  f"{pct(chosen['masked_mean'])}, max {pct(chosen['masked_max'])}.", ""]
    else:
        lines += ["No setting passed every gate.", ""]
    lines += ["## Synthetic recall at the chosen detection settings", "",
              "| Faces | Covered | Recall |", "|---|---|---|"]
    for k, (h, t, r) in synth["recall"].items():
        lines.append(f"| {k} | {h}/{t} | {pct(r)} |")
    lines += ["", f"Mask area over identity ellipse area: {synth['mask_over_target_area']:.2f}", "",
              "## Mask ellipse", "", "| ellipse_w | ellipse_h | Recall | Mask / target |",
              "|---|---|---|---|"]
    for r in ellipse_rows:
        lines.append(f"| {r['ellipse_w']} | {r['ellipse_h']} | {pct(r['recall'])} | "
                     f"{r['area']:.2f} |" if r["area"] else
                     f"| {r['ellipse_w']} | {r['ellipse_h']} | {pct(r['recall'])} | n/a |")
    lines += ["", "## Every setting tried", "",
              "Off-face is masked area where no detector sees a face. Strict counts only "
              "consensus faces as face, so it is an upper bound.", "",
              "| Passes | Score | Recall | Synthetic | Continuity | Off-face mean | "
              "Off-face max | Strict mean | Hands | Masked mean | Masked max | Tracks | Changes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (-r["score"], r["off_face_mean"])):
        lines.append(f"| {'yes' if r['passes'] else 'no'} | {pct(r['score'])} | "
                     f"{pct(r['recall'])} | {pct(r['synthetic_recall'])} | {pct(r['continuity'])} | "
                     f"{pct(r['off_face_mean'])} | {pct(r['off_face_max'])} | "
                     f"{pct(r['off_face_strict_mean'])} | "
                     f"{pct(r['hand_damage'])} | {pct(r['masked_mean'])} | "
                     f"{pct(r['masked_max'])} | {r['tracks']} | `{json.dumps(r['changes'])}` |")
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "precision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
