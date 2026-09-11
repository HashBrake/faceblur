"""What the text detector finds, before anything decides to mask it.

Package T1 measures and does not mask. `classes.TEXT.ready` is still False,
nothing in the pipeline calls `faceblur/text.py`, and this is the only thing
that runs it.

Four questions, and the third is the one that decides whether text is worth
shipping at all:

- **how much text is there**, in lines per frame and share of the frame;
- **what does it cost**, at one scale and at two;
- **what is each line sitting on**, against the OWLv2 oracle: a screen, a
  document, a badge, a sign, or nothing the oracle can name;
- **how much of it is inside the handled zone**, which under the rule of
  2026-09-11 is text the collector is handling and is never masked.

    .venv\\Scripts\\python.exe -m eval.text VIDEO [VIDEO ...]
    .venv\\Scripts\\python.exe -m eval.text VIDEO --sizes 960,1280,1920,2560
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from eval.common import CACHE_DIR, REPORT_DIR, read_frames  # noqa: E402
from eval.oracle_owl import SCREEN_PROMPTS, SURFACE_PROMPTS, boxes_for  # noqa: E402
from faceblur.settings import Settings  # noqa: E402
from faceblur.text import TextDetector, is_page, quads_from  # noqa: E402

# What a line can be sitting on. The oracle's own prompts, grouped, plus the
# answer that matters most on this footage: nothing it can name.
SURFACES = {prompt: "screen" for prompt in SCREEN_PROMPTS}
SURFACES.update({"a name badge": "badge", "a paper document": "document",
                 "a handwritten note": "document", "a sign": "sign"})
# Every surface the oracle was asked about has to have an answer here, so a
# prompt added there cannot quietly become "none" in this table.
assert set(SURFACE_PROMPTS) <= set(SURFACES), sorted(set(SURFACE_PROMPTS) - set(SURFACES))
NONE = "none"
# How much of a quad has to sit inside an oracle box to be called its surface.
ON_SURFACE = 0.5


def sample_frames(video: Path, count: int) -> list[int]:
    """Frame numbers spread evenly over the file."""
    from faceblur.segments import frame_times

    n = len(frame_times(video))
    if count >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / max(1, count - 1))) for i in range(count)]


def quad_mask(shape, quad) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(quad, np.float32).round().astype(np.int32), 1)
    return mask.astype(bool)


def box_mask(shape, box) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = box[:4]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
    mask = np.zeros(shape, bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def surface_of(quad, shape, oracle_rows) -> str:
    """What the oracle says this line is sitting on, or `none`.

    The quad has to sit mostly inside the oracle's box, not merely touch it:
    a sign on a wall and the wall beside it overlap at their edges, and the
    question here is what the text is written on.
    """
    quad_area = quad_mask(shape, quad)
    total = float(quad_area.sum())
    if not total:
        return NONE
    best, best_share = NONE, 0.0
    for row in oracle_rows:
        name = SURFACES.get(row["label"])
        if name is None:
            continue
        share = float((quad_area & box_mask(shape, row["box"])).sum()) / total
        if share > best_share:
            best, best_share = name, share
    return best if best_share >= ON_SURFACE else NONE


def zones_for(video: Path, settings: Settings, frames: int):
    """The handled zone per frame, or None when there is no hands cache."""
    from eval.zone import load_hands, zone_from_cache

    held = load_hands(video, settings)
    if held is None:
        return None
    return zone_from_cache(held, settings, frames, 30.0, None)


def measure(video: Path, settings: Settings, count: int, sizes=None) -> dict:
    """Lines, area, cost and surface over sampled frames of one video."""
    from faceblur.segments import frame_times

    detector = TextDetector(settings)
    picks = sample_frames(video, count)
    total_frames = len(frame_times(video))
    zones = zones_for(video, settings, total_frames)
    owl_path = CACHE_DIR / f"{video.stem}.owl.json"
    owl = json.loads(owl_path.read_text(encoding="utf-8")) if owl_path.is_file() else None

    per_size = {int(s): {"quads": 0, "area": [], "seconds": []} for s in (sizes or ())}
    lines, area, pages, seconds = [], [], 0, []
    by_surface: dict[str, int] = {}
    in_zone = 0
    shape = None
    for i, frame in read_frames(video, set(picks)):
        shape = frame.shape[:2]
        for size in per_size:
            t0 = time.time()
            prob, scale = detector.map_for(frame, size)
            found = quads_from(prob, scale, shape, settings)
            per_size[size]["seconds"].append(time.time() - t0)
            per_size[size]["quads"] += len(found)
            per_size[size]["area"].append(
                sum(d.w * d.h for d in found) / (shape[0] * shape[1]))
        t0 = time.time()
        found = detector.detect(frame)
        seconds.append(time.time() - t0)
        lines.append(len(found))
        area.append(sum(d.w * d.h for d in found) / (shape[0] * shape[1]))
        pages += sum(1 for d in found if is_page(d, shape, settings))
        rows = boxes_for(owl, i) if owl else []
        for det in found:
            where = surface_of(det.quad, shape, rows)
            by_surface[where] = 1 + by_surface.get(where, 0)
            if zones is not None and i < len(zones):
                zone = zones[i]
                polys = tuple(zone.live) + tuple(zone.remembered)
                if polys:
                    from faceblur.zone import alpha_for
                    protected = alpha_for(shape, polys, 0.0) >= 0.5
                    mine = quad_mask(shape, det.quad)
                    if mine.any() and float((mine & protected).sum()) / mine.sum() >= 0.5:
                        in_zone += 1
    out = {
        "video": video.name,
        "frames_sampled": len(picks),
        "shape": list(shape) if shape else None,
        "lines_per_frame_mean": float(np.mean(lines)) if lines else 0.0,
        "lines_per_frame_max": int(np.max(lines)) if lines else 0,
        "frames_with_text": int(sum(1 for n in lines if n)),
        "share_of_frame_mean": float(np.mean(area)) if area else 0.0,
        "share_of_frame_max": float(np.max(area)) if area else 0.0,
        "pages": pages,
        "seconds_per_frame": float(np.mean(seconds)) if seconds else 0.0,
        "lines_total": int(sum(lines)),
        "by_surface": dict(sorted(by_surface.items())),
        "lines_in_zone": in_zone,
        "zone": zones is not None,
        "oracle": owl is not None,
        "sizes": {str(s): {"quads": v["quads"],
                           "share_mean": float(np.mean(v["area"])) if v["area"] else 0.0,
                           "seconds": float(np.mean(v["seconds"])) if v["seconds"] else 0.0}
                  for s, v in per_size.items()},
        "settings": {k: getattr(settings, k) for k in
                     ("text_sizes", "text_thresh", "text_box_thresh", "text_unclip",
                      "text_min_px", "text_max_frac", "text_nms")},
    }
    return out


# The ImageNet mean and standard deviation, which the zoo's own wrapper
# applies through `cv.dnn_TextDetectionModel_DB`. `faceblur/text.py` does not,
# and `--normalisation` is the measurement behind that.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def word_card(w: int = 1600, h: int = 1300) -> np.ndarray:
    """A picture with four known lines of text in it.

    Drawn rather than photographed so that the answer is known: if the model
    does not fire on black words on a pale card the preprocessing is wrong,
    and if it does then the footage simply has less text than it looks like.
    """
    import cv2

    img = np.full((h, w, 3), 235, np.uint8)
    for text, y, scale, weight in (("CANTEEN MENU", 300, 3.0, 7),
                                   ("Fried rice  45000", 520, 2.0, 5),
                                   ("Exit  ->", 760, 2.4, 6),
                                   ("small print here", 980, 0.9, 2)):
        cv2.putText(img, text, (140, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (20, 20, 20), weight)
    return img


def normalisation(settings: Settings, image=None, sizes=(960, 1920, 2560)) -> list[dict]:
    """Plain 0 to 1 against the ImageNet normalisation, on one picture.

    The model ships with no note saying which it wants, and the zoo's own
    wrapper applies the ImageNet one. `faceblur/screens.py` asked the same
    question of YOLOX and the answer there was decisive; this is the same
    check, kept so it can be re-run rather than remembered.
    """
    import cv2

    from faceblur.text import letterbox

    detector = TextDetector(settings)
    card = word_card() if image is None else image
    rows = []
    for how in ("plain", "imagenet"):
        for size in sizes:
            canvas, scale = letterbox(card, size)
            x = canvas.astype(np.float32) / 255.0
            if how == "imagenet":
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
            blob = np.transpose(x, (2, 0, 1))[None]
            raw = detector.pool.run(blob)[0]
            prob = raw.reshape(raw.shape[-2], raw.shape[-1])
            found = quads_from(prob, scale, card.shape[:2], settings)
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                (prob >= settings.text_thresh).astype(np.uint8), 4)
            rows.append({"normalisation": how, "size": size,
                         "map_max": round(float(prob.max()), 3),
                         "blobs": int(n - 1), "quads": len(found),
                         "share": round(float(sum(d.w * d.h for d in found)
                                              / (card.shape[0] * card.shape[1])), 4)})
    return rows


AUDIT_CSV = REPO / "docs" / "audits" / "text_quads_2026-09-12.csv"
# The three verdicts the crops were labelled with. `has text` is the middle
# case and it matters: the quad holds writing and is the object it is printed
# on rather than the writing itself, so masking it hides the words and a good
# deal else with them.
VERDICTS = ("text", "has text", "none")


def audit_tables(path: Path = AUDIT_CSV) -> str:
    """Report 21.3's tables, from the committed labels.

    The labels are by eye and cannot be re-derived by a command. Everything
    computed from them can be, and is, which is the rule report 17.7 was
    fixed to keep.
    """
    import csv
    from collections import Counter

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = [f"{len(rows)} quads from {len(set(r['file'] for r in rows))} files, "
           f"labelled by eye on 2026-09-12", ""]
    counts = Counter(r["verdict"] for r in rows)
    out += ["| Verdict | Quads | Share |", "|---|---|---|"]
    for verdict in VERDICTS:
        out.append(f"| {verdict} | {counts[verdict]} | "
                   f"{100 * counts[verdict] / len(rows):.0f}% |")
    out += ["", "| | n | Score, lowest | Median | Highest | Short side, median |",
            "|---|---|---|---|---|---|"]
    for verdict in VERDICTS:
        here = [r for r in rows if r["verdict"] == verdict]
        if not here:
            continue
        scores = sorted(float(r["score"]) for r in here)
        px = sorted(int(r["px"]) for r in here)
        out.append(f"| {verdict} | {len(here)} | {scores[0]:.3f} | "
                   f"{scores[len(scores) // 2]:.3f} | {scores[-1]:.3f} | "
                   f"{px[len(px) // 2]} px |")
    out += ["", "What raising the score threshold would keep:", "",
            "| `text_box_thresh` | text | has text | none |", "|---|---|---|---|"]
    for thresh in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        kept = {v: sum(1 for r in rows if r["verdict"] == v
                       and float(r["score"]) >= thresh) for v in VERDICTS}
        out.append(f"| {thresh:.2f} | {kept['text']} | {kept['has text']} | "
                   f"{kept['none']} |")
    out += ["", "| File | Quads | text | has text | none |", "|---|---|---|---|---|"]
    for stem in sorted({r["file"] for r in rows}):
        here = [r for r in rows if r["file"] == stem]
        c = Counter(r["verdict"] for r in here)
        out.append(f"| `{stem}` | {len(here)} | {c['text']} | {c['has text']} | "
                   f"{c['none']} |")
    surfaces = Counter(r["oracle_surface"] for r in rows if r["verdict"] != "none")
    out += ["", "What the oracle called the quads that do carry writing: "
            + ", ".join(f"{k} {v}" for k, v in sorted(surfaces.items())) + ".", ""]
    return "\n".join(out)


def render_audit(out_dir: Path, settings: Settings, per_file: int = 16) -> int:
    """Write one crop per quad, so the committed labels can be checked.

    The crops are pixels of real footage and never go into the repository.
    The labels do, which is the only half of this a third party needs from
    here: they can regenerate the crops with this and disagree.
    """
    import csv

    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    detector = TextDetector(settings)
    rows = []
    index = 0
    for video in sorted((REPO / "footage").glob("*.mp4")):
        stem = video.stem.split("_")[3]
        owl_path = CACHE_DIR / f"{video.stem}.owl.json"
        owl = (json.loads(owl_path.read_text(encoding="utf-8"))
               if owl_path.is_file() else None)
        picks = sample_frames(video, per_file)
        for i, frame in read_frames(video, set(picks)):
            shape = frame.shape[:2]
            oracle_rows = boxes_for(owl, i) if owl else []
            for det in detector.detect(frame):
                quad = np.asarray(det.quad, np.float32)
                pad = 24
                x0, y0 = max(0, int(quad[:, 0].min()) - pad), max(0, int(quad[:, 1].min()) - pad)
                x1 = min(shape[1], int(quad[:, 0].max()) + pad)
                y1 = min(shape[0], int(quad[:, 1].max()) + pad)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                crop = frame[y0:y1, x0:x1].copy()
                shifted = quad.copy()
                shifted[:, 0] -= x0
                shifted[:, 1] -= y0
                cv2.polylines(crop, [shifted.round().astype(np.int32)], True,
                              (0, 0, 255), 2)
                cv2.imwrite(str(out_dir / f"{index:03d}_{stem}_f{i}.png"), crop)
                rect = cv2.minAreaRect(quad)
                rows.append({"index": index, "file": stem, "frame": i,
                             "px": int(round(min(rect[1]))),
                             "long_px": int(round(max(rect[1]))),
                             "score": round(det.score, 3),
                             "share_of_frame": round(
                                 det.w * det.h / (shape[0] * shape[1]), 5),
                             "oracle_surface": surface_of(det.quad, shape, oracle_rows),
                             "corners": " ".join(f"{px:.0f},{py:.0f}"
                                                 for px, py in det.quad),
                             "verdict": ""})
                index += 1
    with (out_dir / "quads.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} crops and an unlabelled quads.csv in {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="*")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames to sample, spread over the file (default: 40)")
    ap.add_argument("--sizes", default=None,
                    help="also measure each of these input sizes on its own, "
                         "comma separated, to choose text_sizes by evidence")
    ap.add_argument("--text-sizes", default=None,
                    help="the scales the detector itself runs at, comma "
                         "separated, overriding the shipped text_sizes")
    ap.add_argument("--audit", action="store_true",
                    help="the tables of report 21.3, from the committed labels")
    ap.add_argument("--render-audit", default=None,
                    help="write one crop per quad to this folder, with an "
                         "unlabelled CSV, so the committed labels can be checked")
    ap.add_argument("--normalisation", action="store_true",
                    help="plain 0 to 1 against the ImageNet normalisation, on a "
                         "drawn card of known words")
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
    settings = Settings()
    if args.text_sizes:
        settings = settings.with_changes(
            text_sizes=tuple(int(v) for v in args.text_sizes.split(",")))
    if args.audit:
        print(audit_tables())
        return 0
    if args.render_audit:
        return render_audit(Path(args.render_audit), settings)
    if args.normalisation:
        def table(label, image):
            print(label)
            print(f"{'how':10} {'size':>5} {'map max':>8} {'blobs':>6} {'quads':>6} {'share':>7}")
            for row in normalisation(settings, image):
                print(f"{row['normalisation']:10} {row['size']:5} {row['map_max']:8.3f} "
                      f"{row['blobs']:6} {row['quads']:6} {100 * row['share']:6.2f}%")

        table("four lines of known text, drawn not photographed", None)
        for name in args.video:
            video = Path(name)
            picks = sample_frames(video, 1)
            for _, frame in read_frames(video, set(picks)):
                print()
                table(f"{video.name} frame {picks[0]}", frame)
                break
        return 0
    results = []
    for name in args.video:
        video = Path(name)
        r = measure(video, settings, args.frames, sizes)
        results.append(r)
        print(f"{video.name}")
        print(f"  {r['lines_per_frame_mean']:.2f} lines a frame, most {r['lines_per_frame_max']}, "
              f"on {r['frames_with_text']} of {r['frames_sampled']} frames; "
              f"{100 * r['share_of_frame_mean']:.2f}% of the frame, most "
              f"{100 * r['share_of_frame_max']:.2f}%; {r['pages']} pages; "
              f"{1000 * r['seconds_per_frame']:.0f} ms a frame")
        if r["by_surface"]:
            print("  surfaces: " + ", ".join(f"{k} {v}" for k, v in r["by_surface"].items())
                  + (f"; {r['lines_in_zone']} inside the handled zone" if r["zone"]
                     else "; no hands cache, so the zone is unknown"))
        for size, row in sorted(r["sizes"].items(), key=lambda kv: int(kv[0])):
            print(f"    size {size:>5}: {row['quads']:4} quads, "
                  f"{100 * row['share_mean']:5.2f}% of the frame, "
                  f"{1000 * row['seconds']:4.0f} ms a frame")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    else:
        REPORT_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / "text_measure.json").write_text(json.dumps(results, indent=1),
                                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
