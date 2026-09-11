"""An object oracle for evaluation: OWLv2, open vocabulary, never shipped.

The face side has MediaPipe. It is a different family from anything the
pipeline runs, it never touches an output, and it exists so that a face the
shipped detectors miss is still counted by something. Screens, cards and text
surfaces have had no such witness, which is why section 15.4 of
`docs/report.md` can give precision for screens and no recall at all: with one
detector and no oracle, a screen nothing finds is not counted anywhere.

This is that witness. OWLv2 takes a list of phrases and finds the things they
name, so one model answers for televisions, monitors, phones, cards, slabs,
badges, documents, handwriting and signs, which is what work packages F0, T1
and T2 need as well.

What it is not: a truth. It is one model's opinion, taken at a low threshold
so that a real screen it is unsure about still lands, and its recall figure is
a bound rather than a fact, the same caveat section 12.5 gives the recogniser.
Where the oracle and the shipped rule disagree, neither is right by
definition.

**It never ships and it never runs in the pipeline.** It lives in
`.venv-oracle`, which the packaged app knows nothing about, its weights sit in
a gitignored cache, and it is pinned to one model revision so that a rerun a
month from now measures the same thing. `tests/test_models.py` keeps torch and
the oracle out of `build/faceblur.spec`.

    .venv-oracle\\Scripts\\python.exe eval\\oracle_owl.py VIDEO [--stride 5]

Writes `eval/cache/<stem>.owl.json`: per frame, a label, a score and a box.
Boxes only. No crops, no frames, no pixels of the footage are written
anywhere, here or by anything that reads this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The weights live here and nowhere else. `eval/cache/` is gitignored, so a
# 1.4 GB model cannot reach the repository by accident.
HF_CACHE = REPO / "eval" / "cache" / "hf"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CACHE_DIR = REPO / "eval" / "cache"

# Pinned. `main` moving under a measurement is the same failure the raw face
# caches had on 2026-09-08, when every number since the previous pass turned
# out to describe a build that no longer existed.
MODEL = "google/owlv2-base-patch16-ensemble"
REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
# The weights, checked before they are used. A revision pin says which commit
# to fetch; it does not say that what is on this disk is still that commit. A
# cache can be corrupted, replaced, or fetched once through something that
# rewrote it, and a detector with altered weights does not fail, it answers
# differently. `models/README.md` carries the same two numbers and
# `tests/test_oracle.py` fails if the two ever disagree.
WEIGHTS = "model.safetensors"
WEIGHTS_SHA256 = "e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7"
WEIGHTS_BYTES = 619918824

# Bump when what the cache holds changes.
CACHE_VERSION = 1

# One phrase per thing the later work packages have to ask about. Screens for
# S1's recall number, cards and slabs for F0 and T2, the rest for T1's surface
# column. Phrased as "a thing", which is how OWLv2's own examples read.
PROMPTS: tuple[str, ...] = (
    "a television",
    "a computer monitor",
    "a laptop",
    "a mobile phone",
    "a trading card",
    "a graded card slab",
    "a binder page of trading cards",
    "a name badge",
    "a paper document",
    "a handwritten note",
    "a sign",
)

# Which prompts answer which question. The shipped screen class covers glass;
# the card group is what vetoes a face or a line of text in F0 and T2.
SCREEN_PROMPTS = ("a television", "a computer monitor", "a laptop", "a mobile phone")
CARD_PROMPTS = ("a trading card", "a graded card slab", "a binder page of trading cards")
SURFACE_PROMPTS = ("a name badge", "a paper document", "a handwritten note", "a sign")

# Low on purpose. A witness that only reports what it is sure of gives a
# flattering recall number, because the screens it was unsure about quietly
# stop existing. Everything down to here is written out with its score, and
# the reader of the cache picks a floor.
THRESHOLD = 0.10

# How often the partial cache is flushed to disk. A run over the four sample
# files is about two hours and a crash at ninety minutes should not cost the
# ninety minutes.
FLUSH_EVERY = 40


class WeightsWrong(RuntimeError):
    """The oracle's weights on this disk are not the ones that were pinned."""


def weights_path() -> Path | None:
    """Where the pinned weights sit in the cache, or None if not fetched yet."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    found = try_to_load_from_cache(MODEL, WEIGHTS, revision=REVISION)
    return Path(found) if isinstance(found, str) else None


def verify_weights() -> str:
    """Hash the weights and compare with the pin. Raises, or returns the hash.

    About a second on 620 MB, once per run, against a measurement that takes
    two hours and would otherwise be attributed to a model nobody could
    identify afterwards.
    """
    path = weights_path()
    if path is None or not path.is_file():
        raise WeightsWrong(
            f"the oracle weights are not in the cache. Fetch revision {REVISION} of "
            f"{MODEL} into {HF_CACHE} first, with the network on and "
            f"HF_HUB_OFFLINE unset.")
    size = path.stat().st_size
    if size != WEIGHTS_BYTES:
        raise WeightsWrong(
            f"{path} is {size} bytes and the pin says {WEIGHTS_BYTES}. These are "
            f"not the weights every oracle number in docs/report.md was measured "
            f"with.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise WeightsWrong(
            f"{path} hashes to {digest} and the pin says {WEIGHTS_SHA256}. A "
            f"detector with altered weights does not fail, it answers "
            f"differently, so this stops here.")
    return digest


def fingerprint(stride: int, threshold: float) -> str:
    """What this cache depends on. A mismatch rebuilds rather than misleads.

    The module's own source is in it, so that changing a prompt or the way a
    box is written down invalidates every cache built from the old one. That
    is the rule `eval/common.py` learned the hard way and it costs a rerun.
    """
    src = Path(__file__).read_bytes()
    key = json.dumps({"version": CACHE_VERSION, "model": MODEL, "revision": REVISION,
                      "prompts": list(PROMPTS), "stride": stride,
                      "threshold": round(threshold, 4),
                      "source": hashlib.sha256(src).hexdigest()},
                     sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def cache_path(video: Path) -> Path:
    return CACHE_DIR / f"{Path(video).stem}.owl.json"


def load(video: Path, stride: int = 5, threshold: float = THRESHOLD) -> dict | None:
    """The cache for this video, or None when there is none that still fits."""
    path = cache_path(video)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("fingerprint") != fingerprint(stride, threshold):
        return None
    return data


def boxes_for(data: dict, frame: int, prompts=None, floor: float = 0.0) -> list[dict]:
    """The rows of one frame, optionally narrowed to some prompts and a floor."""
    rows = data.get("frames", {}).get(str(frame), [])
    return [r for r in rows
            if r["score"] >= floor and (prompts is None or r["label"] in prompts)]


def _empty(video: Path, stride: int, threshold: float, shape, frames_total: int) -> dict:
    return {"version": CACHE_VERSION,
            "fingerprint": fingerprint(stride, threshold),
            "video": Path(video).name,
            "model": MODEL,
            "revision": REVISION,
            "prompts": list(PROMPTS),
            "stride": stride,
            "threshold": threshold,
            "shape": list(shape),
            "frames_total": frames_total,
            "seconds": 0.0,
            "frames": {}}


def run(video: Path, stride: int = 5, threshold: float = THRESHOLD,
        limit: int | None = None, quiet: bool = False) -> dict:
    """Detect on every `stride`th frame and write the cache. Resumable."""
    import cv2
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    video = Path(video)
    verify_weights()
    torch.set_grad_enabled(False)
    processor = Owlv2Processor.from_pretrained(MODEL, revision=REVISION)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL, revision=REVISION).eval()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = list(range(0, total, max(1, stride)))
    if limit:
        wanted = wanted[:limit]

    data = load(video, stride, threshold)
    if data is None:
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        data = _empty(video, stride, threshold, shape, total)
    todo = [i for i in wanted if str(i) not in data["frames"]]
    if not quiet:
        print(f"{video.name}: {len(wanted)} frames at stride {stride}, "
              f"{len(todo)} still to do", flush=True)
    if not todo:
        return data

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = 0
    # Sequential decode with a seek only when the gap is large: seeking every
    # frame on a long file costs more than reading through it.
    index = -1
    for target in todo:
        if target - index > 60 or target < index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            index = target - 1
        ok = True
        while ok and index < target:
            ok, frame = cap.read()
            index += 1
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=[list(PROMPTS)], images=image, return_tensors="pt")
        out = model(**inputs)
        found = processor.post_process_grounded_object_detection(
            out, threshold=threshold,
            target_sizes=torch.tensor([[frame.shape[0], frame.shape[1]]]))[0]
        rows = []
        for score, label, box in zip(found["scores"].tolist(), found["labels"].tolist(),
                                     found["boxes"].tolist()):
            x0, y0, x1, y1 = (round(float(v), 1) for v in box)
            rows.append({"label": PROMPTS[label], "score": round(float(score), 4),
                         "box": [x0, y0, x1, y1]})
        rows.sort(key=lambda r: -r["score"])
        data["frames"][str(target)] = rows
        done += 1
        if done % FLUSH_EVERY == 0:
            data["seconds"] = round(time.time() - started, 1)
            _write(video, data)
            if not quiet:
                rate = (time.time() - started) / done
                left = rate * (len(todo) - done)
                print(f"  {done}/{len(todo)} frames, {rate:.1f}s each, "
                      f"{left / 60:.0f} min left", flush=True)
    cap.release()
    data["seconds"] = round(time.time() - started, 1)
    _write(video, data)
    return data


def _write(video: Path, data: dict) -> None:
    """Write the cache through a temporary file, so a kill cannot truncate it."""
    path = cache_path(video)
    part = path.with_suffix(".part")
    part.write_text(json.dumps(data), encoding="utf-8")
    os.replace(part, path)


def describe(data: dict) -> str:
    counts: dict[str, int] = {}
    frames = 0
    for rows in data["frames"].values():
        frames += 1
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    parts = ", ".join(f"{name.removeprefix('a ')} {n}" for name, n in
                      sorted(counts.items(), key=lambda kv: -kv[1]))
    return (f"{data['video']}: {frames} frames at stride {data['stride']}, "
            f"{sum(counts.values())} boxes over {data['threshold']}, {data['seconds']}s\n"
            f"  {parts or 'nothing found'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="+", help="one or more source videos")
    ap.add_argument("--stride", type=int, default=5,
                    help="run on every Nth frame (default: 5)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"lowest score written to the cache (default: {THRESHOLD})")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many frames, for a trial run")
    args = ap.parse_args()
    for name in args.video:
        data = run(Path(name), args.stride, args.threshold, args.limit)
        print(describe(data), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
