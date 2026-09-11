"""What the handled zone covers, what it costs, and what it gets wrong.

The zone is the one promise in this project that is absolute: nothing is ever
masked inside it. A promise like that needs three numbers and a pile of
pictures.

- **How much of the frame it takes.** A zone that covered half the picture
  would keep the rule and destroy the point of the tool.
- **Whether it is on the right thing.** The output side check already sorts
  what it finds into the wearer's hands and real faces, using two models that
  know nothing about the zone. Every hand it finds should be inside the zone
  and every face outside it, so that is a smoke test the zone cannot mark its
  own homework on.
- **What it costs**, at `zone_stride` 1 and 2.

And the pictures: `--audit` renders sampled frames with the zone drawn on them
and writes the CSV skeleton the by eye pass fills in. The labels go to
`docs/audits/`, the frames go to a gitignored folder and are deleted after.
The build plan's standing rule is that an audit without committed labels does
not count, because the two earlier audits in this project cannot be re-scored
for exactly that reason.

    .venv\\Scripts\\python.exe -m eval.zone VIDEO --frames 40
    .venv\\Scripts\\python.exe -m eval.zone VIDEO --report check.json
    .venv\\Scripts\\python.exe -m eval.zone VIDEO --audit OUTDIR --frames 25
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from faceblur.hands import Hand  # noqa: E402
from faceblur.settings import Settings  # noqa: E402
from faceblur.verify import detection_from_row, of_kind  # noqa: E402
from faceblur.zone import (HandFinder, alpha_for, build, polygon_for,  # noqa: E402
                           qualifies, share)

AUDIT_DIR = REPO / "docs" / "audits"
CACHE_DIR = REPO / "eval" / "cache"

# Bump when what the hands cache holds changes.
HANDS_CACHE_VERSION = 1
# What the hands cache depends on, beyond the module's own source.
HANDS_SETTINGS = ("zone_window", "zone_stride", "zone_nms", "zone_top_frac",
                  "hand_conf", "hand_presence", "zone_min_hand_px",
                  "zone_edge_frac", "zone_device")


def hands_fingerprint(settings: Settings) -> str:
    """What the hands cache depends on. A mismatch rebuilds rather than misleads.

    `faceblur/zone.py` is hashed into it, so an edit to the geometry or the
    qualification rule invalidates every cache built from the old one. That is
    the rule `eval/common.py` learned on 2026-09-08, when a stale cache made a
    week of numbers describe a build that no longer existed.
    """
    src = (REPO / "faceblur" / "zone.py").read_bytes()
    key = json.dumps({"version": HANDS_CACHE_VERSION,
                      "source": hashlib.sha256(src).hexdigest(),
                      "settings": {k: getattr(settings, k) for k in HANDS_SETTINGS}},
                     sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def hands_cache_path(video: Path) -> Path:
    return CACHE_DIR / f"{Path(video).stem}.hands.json"


def load_hands(video: Path, settings: Settings) -> dict | None:
    """The hands cache for this video, or None when there is none that fits."""
    path = hands_cache_path(video)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if data.get("fingerprint") == hands_fingerprint(settings) else None


def build_hands_cache(video: Path, settings: Settings | None = None,
                      quiet: bool = False) -> dict:
    """Every qualifying hand of every frame, cached as quads.

    The evaluation harness builds its masks from a cache rather than from the
    pipeline, so without this it scores masks the pipeline would never apply:
    it has no zone, and the zone is subtracted from every mask that ships.
    Report 17.4 found that and package F2 could not start until it was fixed.

    Boxes and quads only, as with every cache here. No crops, no frames.
    """
    from faceblur.segments import decode_range, frame_times
    from faceblur.video import probe

    settings = settings or Settings(mask=("face", "screen"))
    found = load_hands(video, settings)
    if found is not None:
        return found
    finder = HandFinder(settings)
    info, times = probe(video), frame_times(video)
    n = len(times)
    frames: dict[str, list] = {}
    started = time.time()
    for i, frame in enumerate(decode_range(video, times, 0, n, info, settings.hwaccel)):
        if i % max(1, settings.zone_stride):
            continue
        hands = [h for h in finder.detect(frame)
                 if qualifies(h, frame.shape[:2], settings)]
        if hands:
            frames[str(i)] = [h.to_dict() for h in hands]
        if not quiet and i and i % 500 == 0:
            rate = (time.time() - started) / i
            print(f"  {i}/{n} frames, {1000 * rate:.0f} ms each, "
                  f"{rate * (n - i) / 60:.0f} min left", flush=True)
    data = {"version": HANDS_CACHE_VERSION,
            "fingerprint": hands_fingerprint(settings),
            "video": video.name, "frames_total": n,
            "shape": [info.height, info.width],
            "seconds": round(time.time() - started, 1), "frames": frames}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    part = hands_cache_path(video).with_suffix(".part")
    part.write_text(json.dumps(data), encoding="utf-8")
    os.replace(part, hands_cache_path(video))
    return data


def zone_from_cache(data: dict, settings: Settings, frames: int,
                    fps: float, shifts=None) -> list:
    """`zone.build`'s answer, from a cache rather than from the models."""
    hands = {}
    for key, rows in data.get("frames", {}).items():
        hands[int(key)] = [
            Hand(tuple(tuple(p) for p in row["quad"]), row["score"],
                 tuple(row["palm"]), row.get("presence", 1.0))
            for row in rows]
    return build(hands, shifts, settings, frames, fps, tuple(data["shape"]))


# The angles the orientation table is printed at. Wide, because the question
# it answers is whether any threshold separates the two at all.
ORIENTATION_ANGLES = (60, 80, 100, 120, 140, 160, 180)


def pointing(corners: str) -> float | None:
    """Degrees from straight up the frame, from a committed quad.

    `hands.rect_for` builds the quad already rotated by the angle that puts
    the wrist to middle knuckle line upright, so the midpoint of the top edge
    minus the midpoint of the bottom edge is the direction the hand points.
    Nothing has to be re-detected: the number comes out of the four corners
    the audit committed.
    """
    pts = [tuple(map(float, c.split(","))) for c in corners.split()]
    if len(pts) != 4:
        return None
    top = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    bottom = ((pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2)
    return math.degrees(math.atan2(top[0] - bottom[0], -(top[1] - bottom[1])))


def orientation_tables(audit_dir: Path = AUDIT_DIR) -> str:
    """The two tables of report section 17.7, from the committed labels.

    The idea this scores: the wearer's hands point up the frame, away from a
    camera on their chest, while somebody facing them points down or across.
    It is refuted on this footage and the tables are how. They are printed by
    a command rather than kept in a note, because every number in the report
    has to come from something a third party can re-run, and the pass that
    first produced these broke that rule in the section that restates it.
    """
    by_verdict: dict[str, list[float]] = {}
    rows = files = 0
    for path in sorted(audit_dir.glob("zone_hands_*.csv")):
        files += 1
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                angle = pointing(row.get("corners", ""))
                if angle is not None and row.get("verdict"):
                    by_verdict.setdefault(row["verdict"], []).append(abs(angle))
    if not by_verdict:
        return (f"no labelled rows under {audit_dir}. The audit CSVs carry a "
                f"verdict column; an unfilled one measures nothing.")

    def pick(xs, f):
        return xs[min(len(xs) - 1, int(f * len(xs)))]

    out = [f"{rows} rows from {files} files under {audit_dir}", "",
           "| | n | min | median | p90 | max |", "|---|---|---|---|---|---|"]
    for name in ("wearer", "bystander", "cannot tell"):
        xs = sorted(by_verdict.get(name, []))
        if not xs:
            continue
        label = {"wearer": "The wearer's hands", "bystander": "A bystander's hands",
                 "cannot tell": "Could not be called"}[name]
        out.append(f"| {label} | {len(xs)} | {xs[0]:.0f} deg | {pick(xs, .5):.0f} deg | "
                   f"{pick(xs, .9):.0f} deg | {xs[-1]:.0f} deg |")
    wearer = by_verdict.get("wearer", [])
    other = by_verdict.get("bystander", [])
    out += ["", "| `zone_orientation` | The wearer's hands kept | A bystander's dropped |",
            "|---|---|---|"]
    for angle in ORIENTATION_ANGLES:
        kept = sum(1 for a in wearer if a <= angle)
        dropped = sum(1 for a in other if a > angle)
        out.append(f"| {angle} deg | {kept} of {len(wearer)} | {dropped} of {len(other)} |")
    return "\n".join(out)


def sample_frames(video: Path, count: int) -> list[int]:
    """Frame numbers spread evenly over the file, never the very first or last."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise SystemExit(f"could not read {video}")
    return [int(n * (k + 0.5) / count) for k in range(count)]


def read(video: Path, wanted):
    """Yield (index, frame) for the wanted frame numbers, in order."""
    cap = cv2.VideoCapture(str(video))
    for i in sorted(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            yield i, frame
    cap.release()


def summarise(video: Path, settings: Settings, count: int) -> dict:
    """Zone share, hands per frame and cost, over sampled frames."""
    finder = HandFinder(settings)
    picks = sample_frames(video, count)
    shares, hands_n, proposals, seconds, qualified, rejected = [], [], [], [], 0, 0
    shape = None
    for i, frame in read(video, picks):
        shape = frame.shape[:2]
        t0 = time.time()
        hands = finder.detect(frame)
        seconds.append(time.time() - t0)
        every = finder.propose(frame)
        proposals.append(len(every))
        qualified += sum(1 for h in every if qualifies(h, shape, settings))
        rejected += sum(1 for h in every if not qualifies(h, shape, settings))
        hands_n.append(len(hands))
        zone = build({i: hands}, None, settings, i + 1, 30.0, shape)[i]
        shares.append(share(shape, zone))
    return {"video": video.name,
            "frames_sampled": len(shares),
            "palm_proposals_per_frame": round(float(np.mean(proposals)), 2),
            "proposals_that_qualify": qualified,
            "proposals_the_rules_drop": rejected,
            "hands_per_frame": round(float(np.mean(hands_n)), 2),
            "zone_share_mean": round(float(np.mean(shares)), 5),
            "zone_share_p95": round(float(np.percentile(shares, 95)), 5),
            "zone_share_max": round(float(np.max(shares)), 5),
            "frames_with_no_zone": int(sum(1 for s in shares if s == 0)),
            "ms_per_frame": round(1000 * float(np.mean(seconds)), 1)}


def against_check(video: Path, report: dict, settings: Settings) -> dict:
    """Do the check's hands fall inside the zone, and its faces outside it?

    The output side check sorts what it finds into the wearer's hands and
    faces it missed, with MediaPipe's two hand models and three face
    detectors, none of which know the zone exists. So this is the zone marked
    by somebody else's homework.

    A hand counts as covered when most of its box is inside the zone, and a
    face counts as protected, which is the failure, on the same test. The
    build plan wanted the 20 hands and 52 faces of report section 14.1; that
    run's record was never committed and only its counts survive, so this
    rebuilds the same comparison from a check anybody can re-run.
    """
    rows = report.get("residual_list") or []
    hands = [r for r in rows if "hand" in r]
    faces = [r for r in of_kind(rows, "face") if "hand" not in r]
    wanted = {r["frame"] for r in hands + faces}
    if not wanted:
        return {"hand_rows": 0, "face_rows": 0}
    finder = HandFinder(settings)
    inside_hand, inside_face = 0, 0
    covered_hand, covered_face = [], []
    for i, frame in read(video, wanted):
        shape = frame.shape[:2]
        polys = [polygon_for(h, settings, shape) for h in finder.detect(frame)]
        mask = (alpha_for(shape, polys, 0.0) >= 0.5) if polys else None
        for rows_of, out, shares in ((hands, "hand", covered_hand),
                                     (faces, "face", covered_face)):
            for row in [r for r in rows_of if r["frame"] == i]:
                det = detection_from_row(row)
                x0, y0 = max(0, int(det.x)), max(0, int(det.y))
                x1 = min(shape[1], int(det.x + det.w))
                y1 = min(shape[0], int(det.y + det.h))
                if x1 <= x0 or y1 <= y0:
                    continue
                got = 0.0 if mask is None else float(mask[y0:y1, x0:x1].mean())
                shares.append(got)
                if got >= 0.5:
                    if out == "hand":
                        inside_hand += 1
                    else:
                        inside_face += 1
    return {"hand_rows": len(covered_hand),
            "hands_inside_the_zone": inside_hand,
            "hand_cover_mean": round(float(np.mean(covered_hand)), 3)
            if covered_hand else None,
            "face_rows": len(covered_face),
            "faces_inside_the_zone": inside_face,
            "face_cover_mean": round(float(np.mean(covered_face)), 3)
            if covered_face else None}


def hand_damage(video: Path, copy: Path, settings: Settings) -> dict:
    """Masked pixels inside MediaPipe's own hand regions, on the finished copy.

    The witness that the zone works, and it is not the zone: MediaPipe's hand
    hulls come from `eval/oracle_mediapipe.py` in a different environment, and
    nothing in the pipeline has seen them. `eval/measure.py` computes the same
    idea against simulated masks during a sweep; this reads the copy that
    actually shipped, which is the stronger statement and the one a third
    party can repeat.

    A pixel counts as destroyed when it moved further than the encoder's own
    noise plus `FLAT_DIFF`, the floor the output side check uses.
    """
    from faceblur.segments import decode_range, frame_times
    from faceblur.verify import FLAT_DIFF, frame_noise
    from faceblur.video import probe

    cache = REPO / "eval" / "cache" / f"{video.stem}.oracle.json"
    if not cache.is_file():
        return {"hand_damage": None, "reason": f"no MediaPipe hand regions at {cache.name}"}
    data = json.loads(cache.read_text(encoding="utf-8"))
    frames = {int(k): v.get("hands") or [] for k, v in data.get("frames", {}).items()}
    frames = {k: v for k, v in frames.items() if v}
    if not frames:
        return {"hand_damage": None, "reason": "the oracle found no hands"}
    info, out_info = probe(video), probe(copy)
    times, out_times = frame_times(video), frame_times(copy)
    n = min(len(times), len(out_times))
    hit = area = 0.0
    worst, worst_frame = 0.0, None
    # Both files walked once, in lockstep, rather than seeking per frame. On
    # the 3971 frame file a seek near the end fails outright, and this is the
    # pattern `verify.residual_job` already uses for the same reason.
    source = decode_range(video, times, 0, n, info, settings.hwaccel)
    made = decode_range(copy, out_times, 0, n, out_info, settings.hwaccel)
    for i, (before, after) in enumerate(zip(source, made)):
        if i not in frames:
            continue
        mask = np.zeros(before.shape[:2], np.uint8)
        for hull in frames[i]:
            pts = np.round(np.asarray(hull, np.float64)).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
        pixels = int(mask.sum())
        if not pixels:
            continue
        diff = np.abs(before.astype(np.float32) - after.astype(np.float32)).mean(axis=2)
        moved = int(((diff > frame_noise(before, after) + FLAT_DIFF) & (mask > 0)).sum())
        hit += moved
        area += pixels
        if moved / pixels > worst:
            worst, worst_frame = moved / pixels, i
    return {"hand_frames": len(frames),
            "hand_damage": round(hit / area, 5) if area else None,
            "worst_frame_share": round(worst, 5),
            "worst_frame": worst_frame}


def render_audit(video: Path, settings: Settings, count: int, out_dir: Path,
                 csv_path: Path) -> dict:
    """Draw the zone on sampled frames and write the CSV the eye pass fills in.

    The frames go to `out_dir`, which belongs outside the repository or under a
    gitignored folder and is deleted once the pass is done. The CSV carries
    frame numbers and polygon corners and no pixels, and it is committed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    finder = HandFinder(settings)
    picks = sample_frames(video, count)
    written = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh)
        out.writerow(["video", "frame", "polygon", "kind", "palm_px", "score",
                      "presence", "corners", "verdict", "note"])
        for i, frame in read(video, picks):
            shape = frame.shape[:2]
            hands = finder.detect(frame)
            zone = build({i: hands}, None, settings, i + 1, 30.0, shape)[i]
            canvas = frame.copy()
            for n, poly in enumerate(zone.live):
                pts = np.round(np.asarray(poly)).astype(np.int32)
                cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)
                cv2.putText(canvas, f"L{n}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)
            for hand, n in zip(hands, range(len(hands))):
                x, y, w, h = hand.palm
                cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)),
                              (255, 200, 0), 2)
                out.writerow([video.stem.split("_")[3], i, n, "live",
                              round(max(w, h)), round(hand.score, 3),
                              round(hand.presence, 3),
                              " ".join(f"{px:.0f},{py:.0f}"
                                       for px, py in zone.live[n]) if n < len(zone.live)
                              else "", "", ""])
            name = out_dir / f"{video.stem.split('_')[3]}_{i:06d}.jpg"
            cv2.imwrite(str(name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            written += 1
    return {"frames_rendered": written, "frames_dir": str(out_dir),
            "labels": str(csv_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="*")
    ap.add_argument("--cache-hands", action="store_true",
                    help="find every qualifying hand of every frame and cache the "
                         "quads, so the evaluation harness can subtract the zone "
                         "the way the pipeline does")
    ap.add_argument("--orientation", action="store_true",
                    help="print the wrist to knuckle tables of report 17.7 from the "
                         "committed audit labels, and exit. Needs no video and no "
                         "model: the angle comes out of the quads in the CSV")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames to sample, spread over the file (default: 40)")
    ap.add_argument("--report", default=None,
                    help="a JSON report from faceblur.verify --json, to check the "
                         "zone against the hands and faces it found")
    ap.add_argument("--audit", default=None,
                    help="render sampled frames with the zone drawn, into this "
                         "folder, and write the label CSV under docs/audits")
    ap.add_argument("--stride", type=int, default=None, help="override zone_stride")
    ap.add_argument("--copy", default=None,
                    help="a finished copy, to measure masked pixels inside "
                         "MediaPipe's own hand regions on it")
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()
    if args.orientation:
        print(orientation_tables())
        return 0
    if not args.video:
        ap.error("name a video, or pass --orientation")

    settings = Settings(mask=("face", "screen"))
    if args.cache_hands:
        for name in args.video:
            data = build_hands_cache(Path(name), settings)
            hands = sum(len(v) for v in data["frames"].values())
            print(f"{data['video']}: {hands} hands over {len(data['frames'])} of "
                  f"{data['frames_total']} frames, {data['seconds']}s "
                  f"-> {hands_cache_path(Path(name)).name}", flush=True)
        return 0
    if args.stride:
        settings = settings.with_changes(zone_stride=args.stride)
    results = []
    for name in args.video:
        video = Path(name)
        result = summarise(video, settings, args.frames)
        if args.report:
            report = json.loads(Path(args.report).read_text(encoding="utf-8"))
            result["against_the_check"] = against_check(video, report, settings)
        if args.copy:
            result["hand_damage"] = hand_damage(video, Path(args.copy), settings)
        if args.audit:
            stem = video.stem.split("_")[3]
            result["audit"] = render_audit(
                video, settings, args.frames, Path(args.audit),
                AUDIT_DIR / f"zone_hands_{stem}.csv")
        results.append(result)
        print(json.dumps(result, indent=1), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
