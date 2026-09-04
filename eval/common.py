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
from faceblur.redact import ellipse_for  # noqa: E402
from faceblur.settings import Settings  # noqa: E402

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


def _detect_share(args) -> dict[int, RawCandidates]:
    video, share, workers = args
    out = {}
    for i, frame in read_frames(Path(video)):
        if i % workers == share:
            out[i] = _BANK.detect_raw(frame)
    return out


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
        return pickle.loads(path.read_bytes())
    settings = settings or Settings(verify=True)
    workers = workers or max(1, min(10, multiprocessing.cpu_count() // 2))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()

    ctx = multiprocessing.get_context("spawn")
    frames: dict[int, RawCandidates] = {}
    with ctx.Pool(workers, initializer=_init, initargs=(settings,)) as pool:
        for part in pool.imap_unordered(_detect_share,
                                        [(str(video), k, workers) for k in range(workers)]):
            frames.update(part)
    cache = {"video": video.name, "shape": shape, "det_sizes": list(settings.det_sizes),
             "frames": frames}
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
