"""Shared pieces of the evaluation harness."""
from __future__ import annotations

import json
import multiprocessing
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from faceblur.detect import Detection, DetectorBank, RawCandidates, iou  # noqa: E402
from faceblur.motion import downscale, estimate_shift  # noqa: E402
from faceblur.redact import ellipse_for  # noqa: E402
from faceblur.settings import Settings  # noqa: E402

# Bump when what the cache holds changes: 2 added the mirror and tight
# confirmation views, the third detector, and the camera shift per frame.
CACHE_VERSION = 2

FOOTAGE = REPO / "footage"
CACHE_DIR = REPO / "eval" / "cache"          # derived from footage, git ignores it
REPORT_DIR = REPO / "docs"

# Reference identity region, fixed so that sweeps over the mask ellipse are
# measured against a constant target.
REFERENCE = Settings(ellipse_w=1.0, ellipse_h=1.05, feather=0.0)


def videos() -> list[Path]:
    return sorted(FOOTAGE.glob("*.mp4"))


def shortest_video() -> Path:
    best, count = None, None
    for v in videos():
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count is None or n < count:
            best, count = v, n
    return best


def frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frames(video: Path, wanted: set[int] | None = None):
    """Yield (index, frame). Sequential decode, which is exact for h264."""
    cap = cv2.VideoCapture(str(video))
    i = 0
    last = max(wanted) if wanted else None
    while True:
        if last is not None and i > last:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if wanted is None or i in wanted:
            yield i, frame
        i += 1
    cap.release()


# ---------------------------------------------------------------- raw cache

_BANK = None


def _init(settings: Settings) -> None:
    global _BANK
    _BANK = DetectorBank(settings, threads=max(1, multiprocessing.cpu_count() // 10))


def _detect_share(args) -> tuple[dict[int, RawCandidates], dict[int, tuple[float, float]]]:
    video, share, workers = args
    out, shifts = {}, {}
    prev = None
    long_side = None
    for i, frame in read_frames(Path(video)):
        # Every worker decodes every frame, so each has the frame before its
        # own and can measure the camera shift into it.
        if long_side is None:
            long_side = max(frame.shape[:2])
        small = downscale(frame)
        if i % workers == share:
            out[i] = _BANK.detect_raw(frame)
            shifts[i] = estimate_shift(prev, small, long_side)
        prev = small
    return out, shifts


def shifts_list(cache: dict, n: int) -> list[tuple[float, float]]:
    """Per frame camera shift from the cache, zero where missing."""
    have = cache.get("shifts", {})
    return [tuple(have.get(i, (0.0, 0.0))) for i in range(n)]


def cache_path(video: Path) -> Path:
    return CACHE_DIR / f"{video.stem}.raw.pkl"


def build_cache(video: Path, settings: Settings | None = None,
                workers: int | None = None) -> dict:
    """Raw candidates for every frame, from every backend at every size.

    Runs across worker processes. Each decodes the whole video, which is cheap,
    and detects on its share of the frames.
    """
    path = cache_path(video)
    if path.exists():
        cache = pickle.loads(path.read_bytes())
        if cache.get("version") == CACHE_VERSION:
            return cache
        path.unlink()          # an older layout: rebuild
    settings = settings or Settings(verify=True)
    from faceblur.detect import default_workers
    workers = workers or default_workers()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()

    ctx = multiprocessing.get_context("spawn")
    frames: dict[int, RawCandidates] = {}
    shifts: dict[int, tuple[float, float]] = {}
    with ctx.Pool(workers, initializer=_init, initargs=(settings,)) as pool:
        for part, part_shifts in pool.imap_unordered(
                _detect_share, [(str(video), k, workers) for k in range(workers)]):
            frames.update(part)
            shifts.update(part_shifts)
    cache = {"version": CACHE_VERSION, "video": video.name, "shape": shape,
             "det_sizes": list(settings.det_sizes), "frames": frames, "shifts": shifts}
    path.write_bytes(pickle.dumps(cache))
    return cache


# ---------------------------------------------------------------- geometry

def ellipse_mask(shape, det: Detection, settings: Settings = REFERENCE) -> np.ndarray:
    """Boolean mask of a face's identity ellipse."""
    e = ellipse_for(det, settings)
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(round(e.cx)), int(round(e.cy))),
                (max(1, int(round(e.ax))), max(1, int(round(e.ay)))),
                e.angle, 0, 360, 255, -1)
    return m > 0


def polygon_mask(shape, polygons) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 255)
    return m > 0


def det_from_dict(d: dict) -> Detection:
    lm = d.get("landmarks")
    return Detection(d["x"], d["y"], d["w"], d["h"], d.get("score", 1.0),
                     None if lm is None else tuple(tuple(p) for p in lm),
                     d.get("source", "label"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def size_bucket(long_side: float) -> str:
    if long_side < 32:
        return "small <32"
    if long_side < 64:
        return "medium 32-63"
    return "large 64+"
