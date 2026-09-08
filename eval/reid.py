"""Re-identification: can a face recogniser still name anyone in the blurred copy?

Every other number in this harness asks whether a mask covered a box. This one
asks the question the law asks: is a person identifiable. A face recogniser
(SFace, Apache 2.0, the model that pairs with YuNet) embeds a face into 128
numbers; two embeddings of one person are close, of two people far apart. So:

- **identity**: embed the same face in two frames of the source, and two
  different faces in one frame. The two distributions say where the threshold
  sits on this footage, and how large a face has to be before the recogniser
  can tell anyone apart at all. Below that size a face is not identifiable by
  machine, whatever the mask does.
- **mask**: embed a face, then embed the same face after the pipeline's mask.
  If the two still match, the mask does not hide identity. Nothing had
  measured this before: six blocks across a face was an assumption.
- **leaks**: embed every face of the source, then the same rectangle of the
  finished copy. A face the pipeline masked well cannot match; a face it
  missed matches itself. Every probe is also scored against the same frame
  masked here and now, the floor, because the aligned crop around a small
  face is mostly shoulders and room, and those pixels are identical in both
  frames: where the floor alone clears the threshold the probe decides
  nothing and says so. Nor does a face the recogniser cannot match to the
  same person elsewhere in the source: it identifies nobody before the mask.
  What is left is the number to report, not a proxy.

The recogniser only sees faces the source-side detectors find, so a leak count
of zero is not proof of anonymity: it is proof that no face this pipeline can
find survives its own mask. Read it with the size curve, which bounds what the
faces it cannot find could give away.

    .venv\\Scripts\\python.exe -m eval.reid SOURCE [OUTPUT] [--stride N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from eval.common import CACHE_DIR, REPO, build_cache, save_json, size_bucket
from eval.measure import tracked
from faceblur.detect import Detection, iou
from faceblur.redact import redact
from faceblur.settings import Settings

SFACE_MODEL = REPO / "models" / "sface.onnx"

# opencv_zoo's recommendation for SFace: two embeddings of one person score
# above this. `identity` re-measures it on the footage in hand.
COSINE_THRESHOLD = 0.363

# At most this many sightings of one track make same-person pairs. See `pairs`.
PAIR_SAMPLE = 40

# A face smaller than this is not worth embedding: the aligned crop would be
# mostly interpolation. The size curve measures where the real limit is.
MIN_EMBED_PX = 12


class Recogniser:
    """SFace through OpenCV, which also does the 5 landmark alignment."""

    def __init__(self, path: Path = SFACE_MODEL):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"missing the recognition model {self.path}")
        self.net = cv2.FaceRecognizerSF.create(str(self.path), "")

    @staticmethod
    def _row(det: Detection) -> Optional[np.ndarray]:
        """A detection in the [x, y, w, h, 5 landmarks] layout alignCrop wants."""
        if det.landmarks is None:
            return None
        return np.array([[det.x, det.y, det.w, det.h]
                         + [c for p in det.landmarks for c in p]], np.float32)

    def align(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        row = self._row(det)
        if row is None or det.long_side < MIN_EMBED_PX:
            return None
        try:
            return self.net.alignCrop(frame, row)
        except cv2.error:
            return None

    def embed(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """Unit length embedding of one face, or None when it cannot be cut."""
        crop = self.align(frame, det)
        return None if crop is None else self.embed_crop(crop)

    def embed_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        feature = self.net.feature(crop)
        vector = np.asarray(feature, np.float64).ravel()
        norm = float(np.linalg.norm(vector))
        return None if norm < 1e-9 else vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def best_match(embedding: np.ndarray, gallery: list[np.ndarray]) -> float:
    return max((cosine(embedding, g) for g in gallery), default=-1.0)


# ---------------------------------------------------------------- frames

def walk(video: Path, wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) for the wanted frames, decoding in order."""
    cap = cv2.VideoCapture(str(video))
    last = max(wanted) if wanted else -1
    i = 0
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            yield i, frame
        i += 1
    cap.release()


def walk_pair(source: Path, output: Path, wanted: set[int]):
    """Yield (index, source frame, output frame) for the wanted frames."""
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(output))
    last = max(wanted) if wanted else -1
    i = 0
    while i <= last:
        ok_a, fa = a.read()
        ok_b, fb = b.read()
        if not (ok_a and ok_b):
            break
        if i in wanted:
            yield i, fa, fb
        i += 1
    a.release()
    b.release()


def sightings(cache: dict, settings: Settings, min_px: int = MIN_EMBED_PX):
    """Per track, the frames and boxes the detectors actually saw, largest first."""
    per_frame, tracks = tracked(cache, settings)
    out = {}
    for t in tracks:
        seen = [(i, d) for i, d in sorted(t.dets.items())
                if d.source != "track" and d.landmarks is not None and d.long_side >= min_px]
        if seen:
            out[t.id] = seen
    return out, per_frame, tracks


# ---------------------------------------------------------------- calibration

def pairs(video: Path, cache: dict, settings: Settings, rec: Recogniser,
          frame_stride: int = 2) -> tuple[list, list]:
    """Genuine and impostor pairs, each as (score, smaller face, larger face).

    Genuine: two sightings of one track in different frames, which is one
    person by construction. Impostor: two sightings in one frame that do not
    overlap and sit more than a box apart, which cannot be one person, since
    nobody is in two places at once.

    The separation matters. Suppression leaves more than one box on a busy
    face, and two boxes of one face land in two tracks; counted as strangers
    they score 1.0 and drag the threshold to nonsense. A mirror can still put
    one person in the frame twice, which makes the impostor distribution
    pessimistic, and that is the safe direction for a threshold.
    """
    seen, _, _ = sightings(cache, settings)
    per_frame: dict[int, list] = defaultdict(list)
    for tid, rows in seen.items():
        for i, d in rows:
            if i % max(1, frame_stride) == 0:
                per_frame[i].append((tid, d))
    embedded: dict[int, list] = defaultdict(list)      # frame -> (tid, det, embedding)
    by_track: dict[int, list] = defaultdict(list)      # track -> (frame, det, embedding)
    # Embed as the video decodes. Holding the wanted frames instead would ask
    # for 6 MB a frame, and the longest sample file has four thousand of them.
    for i, frame in walk(video, set(per_frame)):
        for tid, d in per_frame[i]:
            e = rec.embed(frame, d)
            if e is None:
                continue
            embedded[i].append((tid, d, e))
            by_track[tid].append((i, d, e))

    genuine = []
    # Every pair of sightings of one track is a same-person pair, which grows
    # with the square of a long track: one 4000 frame file would spend an hour
    # on one person. A track is sampled evenly down to PAIR_SAMPLE sightings,
    # which keeps its whole range of sizes and angles and bounds the work.
    for rows in by_track.values():
        if len(rows) > PAIR_SAMPLE:
            step = len(rows) / PAIR_SAMPLE
            rows = [rows[int(k * step)] for k in range(PAIR_SAMPLE)]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                if rows[a][0] == rows[b][0]:
                    continue                     # the same frame is not a second look
                genuine.append((cosine(rows[a][2], rows[b][2]),
                                min(rows[a][1].long_side, rows[b][1].long_side),
                                max(rows[a][1].long_side, rows[b][1].long_side)))
    impostor = []
    for rows in embedded.values():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                ta, da, ea = rows[a]
                tb, db, eb = rows[b]
                if ta == tb or iou(da, db) > 0.0:
                    continue
                apart = np.hypot(da.cx - db.cx, da.cy - db.cy)
                if apart < 1.5 * max(da.long_side, db.long_side):
                    continue                     # too close to be sure they are two people
                impostor.append((cosine(ea, eb), min(da.long_side, db.long_side),
                                 max(da.long_side, db.long_side)))
    return genuine, impostor


def calibrate(genuine: list, impostor: list) -> dict:
    """The score a stranger reaches on this footage, and what it costs to be sure.

    The published threshold comes from portrait photographs. A wide angle
    camera at four metres is not that, so the threshold is re-read here: the
    score below which 99 percent of strangers fall is the one a claim of "this
    is the same person" has to clear.
    """
    imp = np.asarray([s for s, _, _ in impostor]) if impostor else np.zeros(0)
    gen = np.asarray([s for s, _, _ in genuine]) if genuine else np.zeros(0)
    out = {"genuine_pairs": int(gen.size), "impostor_pairs": int(imp.size),
           "published_threshold": COSINE_THRESHOLD}
    if imp.size:
        out["impostor"] = {"median": round(float(np.median(imp)), 3),
                           "p95": round(float(np.percentile(imp, 95)), 3),
                           "p99": round(float(np.percentile(imp, 99)), 3),
                           "max": round(float(imp.max()), 3),
                           "over_published": round(float((imp >= COSINE_THRESHOLD).mean()), 3)}
        out["threshold_fmr1"] = round(float(np.percentile(imp, 99)), 3)
        out["threshold_fmr01"] = round(float(imp.max()), 3)
    if gen.size:
        out["genuine"] = {"median": round(float(np.median(gen)), 3),
                          "p05": round(float(np.percentile(gen, 5)), 3),
                          "over_published": round(float((gen >= COSINE_THRESHOLD).mean()), 3)}
    if imp.size and gen.size:
        t = out["threshold_fmr1"]
        out["true_match_at_fmr1"] = round(float((gen >= t).mean()), 3)
    return out


def by_size(genuine: list, impostor: list, threshold: float,
            edges=(16, 24, 32, 40, 64, 96, 10 ** 6)) -> dict:
    """How often one person's two sightings are recognised as one, by face size.

    The smaller of the two sightings sets the bucket: a face is only as
    identifiable as its worst look. Where this rate falls to the false match
    rate the threshold was set at, the recogniser is guessing, and a face of
    that size does not identify anybody.
    """
    out = {}
    for lo, hi in zip((0,) + edges[:-1], edges):
        gen = [s for s, small, _ in genuine if lo <= small < hi]
        imp = [s for s, small, _ in impostor if lo <= small < hi]
        label = f"{lo}-{hi} px" if hi < 10 ** 6 else f"{lo}+ px"
        out[label] = {
            "same_person_pairs": len(gen),
            "recognised": round(float(np.mean([s >= threshold for s in gen])), 3) if gen else None,
            "median_same": round(float(np.median(gen)), 3) if gen else None,
            "stranger_pairs": len(imp),
            "false_match": round(float(np.mean([s >= threshold for s in imp])), 3) if imp else None,
        }
    return out


def mask_curve(video: Path, cache: dict, settings: Settings, rec: Recogniser,
               threshold: float, modes=("blur", "pixelate", "solid"),
               strengths=(4, 6, 8, 12), gallery_px: int = 48, limit: int = 60) -> dict:
    """Does the pipeline's own mask defeat the recogniser?

    Each face is embedded from the source, then again after `redact` has run
    on that frame with that face as its only detection. Same frame, same
    alignment, so the only difference is the mask. A mask that hides identity
    leaves a score a stranger could have reached.
    """
    seen, _, _ = sightings(cache, settings, min_px=gallery_px)
    rows = sorted(((i, d) for rows in seen.values() for i, d in rows),
                  key=lambda r: -r[1].long_side)[:limit]
    frames = {i: f for i, f in walk(video, {i for i, _ in rows})}
    out = {}
    for mode in modes:
        for strength in (strengths if mode != "solid" else (settings.strength,)):
            s = settings.with_changes(mode=mode, strength=strength)
            matched = total = 0
            scores = []
            for i, d in rows:
                frame = frames.get(i)
                if frame is None:
                    continue
                base = rec.embed(frame, d)
                if base is None:
                    continue
                masked, _ = redact(frame, [d], s)
                after = rec.embed(masked, d)
                if after is None:
                    continue
                score = cosine(base, after)
                scores.append(score)
                total += 1
                matched += score >= threshold
            key = mode if mode == "solid" else f"{mode} {strength}"
            out[key] = {"matched": matched, "of": total,
                        "rate": round(matched / total, 3) if total else None,
                        "median": round(float(np.median(scores)), 3) if scores else None,
                        "worst": round(float(np.max(scores)), 3) if scores else None}
    return out


def ellipse_curve(video: Path, cache: dict, settings: Settings, rec: Recogniser,
                  threshold: float, sizes=((1.10, 1.15), (1.25, 1.30), (1.4, 1.45)),
                  gallery_px: int = 48, limit: int = 60) -> dict:
    """The same test over mask sizes: how much of the head has to go."""
    out = {}
    for w, h in sizes:
        s = settings.with_changes(ellipse_w=w, ellipse_h=h)
        r = mask_curve(video, cache, s, rec, threshold, modes=("blur",),
                       strengths=(s.strength,), gallery_px=gallery_px, limit=limit)
        out[f"{w}x{h}"] = r[f"blur {s.strength}"]
    return out


def verdict(score: float, floor: float, self_match: float, threshold: float) -> str:
    """What one probe proves: covered, leaked, or nothing at all.

    Three numbers, all against the same source face. `score` is the finished
    copy of that box. `floor` is the same box masked here and now, which is
    what a mask this pipeline applied correctly leaves. `self_match` is the
    same person in another frame of the source, untouched.

    Both of the last two decide whether the first means anything.

    A crop small enough to be all face turns the floor to a stranger's score,
    and then the copy's own score means what it says. A crop of a 16 px face
    is mostly hair, shoulders and the room behind, and those pixels are the
    same in both frames whatever the mask did: the floor clears the threshold
    by itself, and no answer about that face can be read off the copy.

    And a face the recogniser cannot match to the same person in the very
    next second of the source is a face it cannot identify at all. On this
    footage that is most of them, small and turned away and smeared. Whether
    the copy of such a face scores 0.2 or 0.7 says nothing about whether
    anybody is identifiable in it; the picture of the five worst on `004100`
    shows a blurred blob either way. Reporting those as leaks would put a
    number on noise, so they are set aside and counted.
    """
    if floor >= threshold:
        return "blind"
    if self_match < threshold:
        return "unidentifiable"
    return "leak" if score >= threshold else "covered"


def _box_diff(a: np.ndarray, b: np.ndarray, det: Detection) -> float:
    """Mean absolute difference over a detection's box."""
    h, w = a.shape[:2]
    x0, y0 = max(0, int(det.x)), max(0, int(det.y))
    x1, y1 = min(w, int(det.x + det.w)), min(h, int(det.y + det.h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.mean(np.abs(a[y0:y1, x0:x1].astype(np.float32)
                                - b[y0:y1, x0:x1].astype(np.float32))))


def leaks(source: Path, output: Path, cache: dict, settings: Settings,
          rec: Recogniser, threshold: float, stride: int = 1) -> dict:
    """Sightings of the finished copy that still look like the source face.

    Both crops come from the same frame, the same box and the same alignment,
    so the only difference between them is what the pipeline did. A face it
    masked scores what a stranger scores; a face it missed scores near one,
    because those pixels never changed.

    Every probe is scored twice: against the finished copy, and against the
    same frame masked here and now. The second is the floor, and `verdict`
    explains why a number without it is not worth reading. Faces on this
    footage are small, and on the first run 1160 of 8060 sightings looked
    like leaks while the picture showed the face plainly blurred: the crop
    around a 16 px face carries enough unchanged room to score 0.99 by
    itself. The rate below is over the probes that can decide.

    A probe also has to be a face the recogniser can identify in the source
    at all, which `verdict` measures against the same person in the source's
    other frames. Most sightings on this footage are not.

    Each leak also carries `applied`: how much the copy changed the box
    against how much the mask applied here changed it. Near zero means no
    mask reached that frame; near one means the mask ran and identity
    survived it. The two are different failures and want different fixes.
    """
    seen, per_frame, tracks = sightings(cache, settings)
    probes = [(tid, i, d) for tid, rows in seen.items()
              for k, (i, d) in enumerate(rows) if k % max(1, stride) == 0]
    if not probes:
        return {"checked": 0, "leaked": 0, "frames": [], "note": "no sighting to check"}
    by_frame = defaultdict(list)
    for tid, i, d in probes:
        by_frame[i].append((tid, d))

    # First pass: three embeddings of every probe, and the box it sat on.
    rows = []
    for i, fa, fb in walk_pair(source, output, set(by_frame)):
        for tid, d in by_frame[i]:
            before = rec.embed(fa, d)
            after = rec.embed(fb, d)
            control, _ = redact(fa, [d], settings)
            floor_vec = rec.embed(control, d)
            if before is None or after is None or floor_vec is None:
                continue
            rows.append({"track": tid, "frame": i, "det": d, "before": before,
                         "score": cosine(before, after), "floor": cosine(before, floor_vec),
                         "changed": _box_diff(fa, fb, d), "would": _box_diff(fa, control, d)})

    # Second pass: each probe against the same person in the source's other
    # frames, which says whether the recogniser can identify them at all.
    by_track = defaultdict(list)
    for k, r in enumerate(rows):
        by_track[r["track"]].append(k)
    for tid, idx in by_track.items():
        for k in idx:
            others = [rows[j]["before"] for j in idx if rows[j]["frame"] != rows[k]["frame"]]
            rows[k]["self_match"] = best_match(rows[k]["before"], others)

    counts = defaultdict(int)
    found, scores, floors = [], [], []
    buckets = defaultdict(lambda: defaultdict(int))
    for r in rows:
        d = r["det"]
        bucket = ("40+ px" if d.long_side >= 40 else
                  "24-40 px" if d.long_side >= 24 else "under 24 px")
        call = verdict(r["score"], r["floor"], r["self_match"], threshold)
        counts[call] += 1
        buckets[bucket][call] += 1
        if call in ("blind", "unidentifiable"):
            continue
        scores.append(r["score"])
        floors.append(r["floor"])
        if call == "leak":
            found.append({"frame": r["frame"], "score": round(r["score"], 3),
                          "floor": round(r["floor"], 3),
                          "self_match": round(r["self_match"], 3),
                          "px": round(d.long_side), "x": round(d.cx), "y": round(d.cy),
                          "applied": round(r["changed"] / r["would"], 2)
                          if r["would"] > 1e-6 else None})
    found.sort(key=lambda r: -r["score"])
    decidable = counts["leak"] + counts["covered"]
    return {"checked": len(rows), "decidable": decidable, "blind": counts["blind"],
            "unidentifiable": counts["unidentifiable"], "leaked": counts["leak"],
            "rate": round(counts["leak"] / decidable, 5) if decidable else None,
            "median": round(float(np.median(scores)), 3) if scores else None,
            "worst": round(float(np.max(scores)), 3) if scores else None,
            "median_floor": round(float(np.median(floors)), 3) if floors else None,
            "by_size": {k: dict(v) for k, v in sorted(buckets.items())},
            "frames": found[:200]}



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output", nargs="?", default=None,
                    help="the blurred copy; without it only the calibration and the mask are measured")
    ap.add_argument("--json", default=None, help="settings overrides as JSON")
    ap.add_argument("--stride", type=int, default=1, help="check every Nth sighting")
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine threshold; the default is calibrated on this video")
    args = ap.parse_args()
    source = Path(args.source)
    settings = Settings(**(json.loads(args.json) if args.json else {}))
    cache = build_cache(source)
    rec = Recogniser()

    report = {"source": source.name}
    genuine, impostor = pairs(source, cache, settings, rec)
    report["calibration"] = calibrate(genuine, impostor)
    c = report["calibration"]
    print(f"calibration on {source.name}: {c['genuine_pairs']} same-person pairs, "
          f"{c['impostor_pairs']} stranger pairs")
    if "impostor" in c:
        print(f"  strangers    median {c['impostor']['median']:+.3f}  p99 {c['impostor']['p99']:+.3f}  "
              f"max {c['impostor']['max']:+.3f}  over the published {COSINE_THRESHOLD}: "
              f"{100*c['impostor']['over_published']:.0f}%")
        print(f"  same person  median {c['genuine']['median']:+.3f}  "
              f"over the published threshold: {100*c['genuine']['over_published']:.0f}%")
        print(f"  threshold at 1 stranger in 100: {c['threshold_fmr1']:+.3f}  "
              f"(same people recognised there: {100*c['true_match_at_fmr1']:.0f}%)")
    threshold = args.threshold if args.threshold is not None else c.get("threshold_fmr1", COSINE_THRESHOLD)
    report["threshold"] = threshold

    report["by_size"] = by_size(genuine, impostor, threshold)
    print(f"\nidentifiable at what size, at threshold {threshold:+.3f}")
    print("  size          same person recognised   strangers matched")
    for label, row in report["by_size"].items():
        if row["same_person_pairs"]:
            fm = "n/a" if row["false_match"] is None else f"{100*row['false_match']:5.1f}% of {row['stranger_pairs']}"
            print(f"  {label:12s} {100*row['recognised']:5.1f}% of {row['same_person_pairs']:<5d}      {fm}")

    report["mask_curve"] = mask_curve(source, cache, settings, rec, threshold)
    print("\ndoes the mask hide identity (same frame, before and after)")
    for key, row in report["mask_curve"].items():
        if row["of"]:
            print(f"  {key:12s} still matched {row['matched']:3d}/{row['of']:<3d} = {100*row['rate']:5.1f}%  "
                  f"median {row['median']:+.3f}  worst {row['worst']:+.3f}")

    report["ellipse_curve"] = ellipse_curve(source, cache, settings, rec, threshold)
    print("\nmask size (ellipse_w x ellipse_h)")
    for key, row in report["ellipse_curve"].items():
        if row["of"]:
            print(f"  {key:12s} still matched {row['matched']:3d}/{row['of']:<3d} = {100*row['rate']:5.1f}%  "
                  f"worst {row['worst']:+.3f}")

    if args.output:
        output = Path(args.output)
        report["leaks"] = leaks(source, output, cache, settings, rec, threshold, args.stride)
        save_json(CACHE_DIR / f"{source.stem}.reid.json", report)      # before any printing
        r = report["leaks"]
        print(f"\nleaks in {output.name}: {r['leaked']} of {r['decidable']} sightings that can "
              f"decide still look like the source face "
              f"(median {r['median']:+.3f}, worst {r['worst']:+.3f})")
        print(f"  of {r['checked']} checked, {r['blind']} carry too much unchanged room to say "
              f"and {r['unidentifiable']} are faces the recogniser cannot identify in the source")
        for bucket, row in r.get("by_size", {}).items():
            decidable = row.get("leak", 0) + row.get("covered", 0)
            print(f"  {bucket:>12s}: {row.get('leak', 0)}/{decidable} leaked, "
                  f"{row.get('blind', 0)} blind, {row.get('unidentifiable', 0)} unidentifiable")
        for row in r["frames"][:15]:
            applied = "no mask" if (row["applied"] or 0) < 0.2 else f"mask {row['applied']:.2f}"
            print(f"    frame {row['frame']:5d} score {row['score']:+.3f} floor {row['floor']:+.3f} "
                  f"self {row['self_match']:+.3f} "
                  f"{row['px']}px at ({row['x']},{row['y']})  {applied}")
    save_json(CACHE_DIR / f"{source.stem}.reid.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
