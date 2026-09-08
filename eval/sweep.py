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

from eval import measure, misses, synthetic
from eval.common import CACHE_DIR, REPORT_DIR, build_cache, load_json, save_json
from faceblur.settings import Settings

# off_face_mean was 0.3 percent until the entry tails grew to follow a face
# in from the frame edge; a tail before the face is visible is off-face by
# construction. The part from detector boxes has its own, tighter gate.
# 2026-09-08: the total went 0.5 to 0.7 percent and the detector-box part 0.2
# to 0.3 for the gap filling and stitching that keep a face covered through a
# fast pan; the hand gate did not move.
GATES = {"off_face_mean": 0.007, "off_face_max": 0.06, "off_face_detections_mean": 0.003,
         "hand_damage": 0.001}

_INPUTS = None
_SYNTH = None


def _init(video):
    global _INPUTS, _SYNTH
    video = Path(video)
    _INPUTS = measure.load_inputs(video)
    _SYNTH = synthetic.load(video)


def _run(changes):
    s = Settings(**changes)
    r = measure.evaluate(s, *_INPUTS)
    y = synthetic.evaluate(s, _SYNTH)["recall"]
    r["synthetic"] = {k: v[2] for k, v in y.items()}
    r["synthetic_recall"] = y["all"][2]
    # Stretches where YuNet saw a box at full threshold that nothing
    # confirmed and no mask covers: the yardstick for the misses work.
    runs = misses.find(_INPUTS[0], s)
    r["unconfirmed_runs"] = len(runs)
    r["unconfirmed_frames"] = sum(b - a + 1 for a, b, *_ in runs)
    # The number the choice is made on: both kinds of recall, equal weight.
    r["score"] = 0.5 * (r["recall"] or 0.0) + 0.5 * r["synthetic_recall"]
    r["changes"] = changes
    return r


def passes(r: dict) -> bool:
    if r["off_face_mean"] > GATES["off_face_mean"] or r["off_face_max"] > GATES["off_face_max"]:
        return False
    if r["off_face_detections_mean"] > GATES["off_face_detections_mean"]:
        return False
    if r["hand_damage"] is not None and r["hand_damage"] > GATES["hand_damage"]:
        return False
    return True


OLD_TRACKER = {"camera_comp": False, "link_dist": 0.0, "tail_before_max": 6, "tail_grow": 0.05,
               "conf_weak_long": 0.4, "stitch_gap": 0, "tail_long": 2, "blur_shift": 0.0,
               "continue_after": 2, "weak_run": 0, "gap_grow": 0.0, "track_sure_conf": 0.0}
PUSHED = {"camera_comp": True, "link_dist": 0.75, "tail_before_max": 12, "tail_grow": 0.03,
          "conf_weak_long": 0.4, "stitch_gap": 0, "tail_long": 2, "blur_shift": 0.0,
          "continue_after": 2, "weak_run": 0, "gap_grow": 0.0, "track_sure_conf": 0.0}
NEW_TRACKER = {"camera_comp": True, "link_dist": 0.75, "tail_before_max": 12, "tail_grow": 0.03,
               "conf_weak_long": 0.3, "tail_long": 4, "continue_after": 3, "weak_run": 30}
VIEWS = ({"confirm_tta": 0, "verify_conf_sure": 0.3},
         {"confirm_tta": 2, "verify_conf_view": 0.4, "verify_conf_sure": 0.5})
AGREE = ({"conf_agree": 0.6}, {"conf_agree": 0.35, "verify_conf_agree": 0.35, "agree_max_px": 48})


def grid() -> list[dict]:
    base = {"conf": 0.6, "max_face_frac": 0.15, "min_track": 2, "max_gap": 5, "conf_weak": 0.4,
            "third_opinion": True, "verify_conf_low": 0.25}
    out = []
    for views, agree, sure, stitch, grow, blur in itertools.product(
            VIEWS, AGREE, [0.4, 0.5, 0.6], [45, 90], [0.0, 0.04], [0.0, 8.0]):
        out.append({**base, **views, **agree, **NEW_TRACKER, "track_sure_conf": sure,
                    "stitch_gap": stitch, "gap_grow": grow, "blur_shift": blur})
    full = {**base, **VIEWS[1], **AGREE[1], **NEW_TRACKER, "track_sure_conf": 0.5,
            "stitch_gap": 45, "gap_grow": 0.04, "blur_shift": 8.0}
    # No sure rule at all, every second frame, the two earlier builds.
    out.append({**full, "track_sure_conf": 0.0})
    out.append({**full, "stride": 2})
    out.append({**base, **VIEWS[1], **AGREE[0], **PUSHED})
    out.append({**base, **VIEWS[0], **AGREE[0], **OLD_TRACKER})
    for conf in (0.5, 0.6, 0.7):
        out.append({"conf": conf, "conf_weak": conf, "max_face_frac": 0.15,
                    "min_track": 2, "max_gap": 3, "confirm_tta": 0, "third_opinion": False,
                    **OLD_TRACKER})   # no continuation, the old confirmation
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--report", default="precision_report.md",
                    help="file name under docs/ for the report")
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
    # one that keeps confirmed faces covered longest, then the one that leaves
    # the fewest unconfirmed frames. The two recalls cannot see either: the
    # pseudo-faces are built from the confirmers' agreement and the synthetic
    # faces are sharp and frontal, so a face only the mirror view or the third
    # detector confirms shows up nowhere else. The gates have already held.
    # The hard gate comes first: among settings inside the gates, the fewest
    # frames with a known face of 40 px or more left visible, then the
    # fewest of 24 to 40 px, then the score.
    if ok:
        ok.sort(key=lambda r: (r["exposed_40"], r["exposed_24_40"], -round(r["score"], 4),
                               -round(r["continuity"], 3), r["unconfirmed_frames"],
                               r["off_face_mean"], r["masked_mean"]))
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
    write_report(video, results, chosen, ellipse_rows, synth_at_default, args.report)
    if chosen:
        print("CHOSEN:", json.dumps(chosen["changes"]), measure.describe(chosen),
              f"synthetic {100*chosen['synthetic_recall']:.1f}%")
        print("ELLIPSE:", best_ellipse)
    else:
        print("no setting passed the gates; best by off-face:",
              measure.describe(min(results, key=lambda r: r["off_face_mean"])))
    return 0


def write_report(video, results, chosen, ellipse_rows, synth,
                 name: str = "precision_report.md") -> None:
    def pct(v):
        return "n/a" if v is None else f"{100*v:.2f}%"

    lines = ["# Precision report", "",
             f"Automatic sweep on `{video.name}`. No human labels. See `eval/` for the method.",
             "", "## Gates", "",
             f"- off-face masked area, where no detector sees a face: mean at most "
             f"{pct(GATES['off_face_mean'])}, any frame at most {pct(GATES['off_face_max'])}; "
             f"the part from detector boxes rather than tails at most "
             f"{pct(GATES['off_face_detections_mean'])}",
             f"- hand pixels touched: at most {pct(GATES['hand_damage'])}",
             "- among settings inside the gates, the fewest frames with a known face "
             "of 40 px or more left visible wins (the hard gate), then the fewest of "
             "24 to 40 px, then the highest mean of consensus recall and synthetic "
             "recall, then continuity and the fewest unconfirmed frames", "",
             "## Chosen", ""]
    if chosen:
        lines += ["```", json.dumps(chosen["changes"]), "```", "",
                  f"Exposed frames, known face 40 px and more: {chosen['exposed_40']}; "
                  f"24 to 40 px: {chosen['exposed_24_40']}. "
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
              "consensus faces as face, so it is an upper bound. Edge and pan are synthetic "
              "recall on faces entering at the frame edge and on faces during a camera pan. "
              "Unconfirmed runs are stretches of three frames or more where YuNet saw a box "
              "at full threshold that nothing confirmed and no mask covers; most are hands "
              "and objects, fewer is better only if the gates hold.", "",
              "| Passes | Exposed 40+ | Exposed 24-40 | Score | Recall | Synthetic | Edge | Pan | Continuity | Unconfirmed runs | "
              "Off-face mean | Off-face max | Off-face detections | Strict mean | Hands | "
              "Masked mean | Masked max | Tracks | Changes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["exposed_40"], r["exposed_24_40"], -r["score"])):
        lines.append(f"| {'yes' if r['passes'] else 'no'} | {r['exposed_40']} | {r['exposed_24_40']} | "
                     f"{pct(r['score'])} | "
                     f"{pct(r['recall'])} | {pct(r['synthetic_recall'])} | "
                     f"{pct(r['synthetic'].get('edge'))} | {pct(r['synthetic'].get('pan'))} | "
                     f"{pct(r['continuity'])} | {r['unconfirmed_runs']} ({r['unconfirmed_frames']} fr) | "
                     f"{pct(r['off_face_mean'])} | {pct(r['off_face_max'])} | "
                     f"{pct(r['off_face_detections_mean'])} | {pct(r['off_face_strict_mean'])} | "
                     f"{pct(r['hand_damage'])} | {pct(r['masked_mean'])} | "
                     f"{pct(r['masked_max'])} | {r['tracks']} | `{json.dumps(r['changes'])}` |")
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
