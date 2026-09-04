"""Face detector backends and the filter that turns raw candidates into faces.

Two stages. `DetectorBank.detect_raw` runs every backend at every detection size
with a very low threshold and returns everything, with landmarks. That result is
cheap to cache. `filter_candidates` then applies the settings: threshold, size
cap, aspect check, confirmation by the second detector, and NMS. Sweeps re-run
only the second stage.

Every box is in source frame coordinates.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .settings import Settings

Point = tuple[float, float]
Landmarks = tuple[Point, Point, Point, Point, Point]  # eye, eye, nose, mouth, mouth

# Everything below this score is discarded at the detector and never cached.
RAW_FLOOR = 0.05


def _base_dir() -> Path:
    """The folder that holds `models/`. A frozen build unpacks beside the exe."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


MODEL_DIR = _base_dir() / "models"
YUNET_MODEL = MODEL_DIR / "yunet.onnx"
CENTERFACE_MODEL = MODEL_DIR / "centerface_dynamic.onnx"


class ModelMissing(FileNotFoundError):
    """A model file the chosen engine needs is not on disk."""


@lru_cache(maxsize=8)
def model_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ModelMissing(f"Could not find the model file {path}. Reinstall FaceBlur.")
    return path


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: Optional[Landmarks] = None
    source: str = "yunet"      # yunet, centerface, track
    verified: bool = False

    @property
    def long_side(self) -> float:
        return max(self.w, self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def box(self) -> list[float]:
        return [self.x, self.y, self.w, self.h, self.score]

    def scaled(self, factor: float) -> "Detection":
        """Map from a resized image back to the source frame."""
        if factor == 1.0:
            return self
        lm = None
        if self.landmarks is not None:
            lm = tuple((px / factor, py / factor) for px, py in self.landmarks)
        return replace(self, x=self.x / factor, y=self.y / factor,
                       w=self.w / factor, h=self.h / factor, landmarks=lm)

    def to_dict(self) -> dict:
        return {"x": round(self.x, 1), "y": round(self.y, 1), "w": round(self.w, 1),
                "h": round(self.h, 1), "score": round(self.score, 3),
                "source": self.source, "verified": self.verified,
                "landmarks": None if self.landmarks is None
                else [[round(a, 1), round(b, 1)] for a, b in self.landmarks]}


def iou(a, b) -> float:
    """Intersection over union of two boxes, given as Detection or [x,y,w,h,...]."""
    ax, ay, aw, ah = (a.x, a.y, a.w, a.h) if isinstance(a, Detection) else a[:4]
    bx, by, bw, bh = (b.x, b.y, b.w, b.h) if isinstance(b, Detection) else b[:4]
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def nms_detections(dets: list[Detection], thr: float) -> list[Detection]:
    """Greedy NMS on Detection objects. Keeps the highest score of a cluster.

    A kept box without landmarks adopts the landmarks of a suppressed box that
    had them, so a CenterFace winner still carries YuNet's landmark geometry.
    """
    if not dets:
        return []
    order = sorted(dets, key=lambda d: -d.score)
    kept: list[Detection] = []
    while order:
        best = order.pop(0)
        rest = []
        for d in order:
            if iou(best, d) > thr:
                if best.landmarks is None and d.landmarks is not None:
                    best = replace(best, landmarks=d.landmarks)
                if d.verified and not best.verified:
                    best = replace(best, verified=True)
            else:
                rest.append(d)
        kept.append(best)
        order = rest
    return kept


def resize_long_side(frame: np.ndarray, det_size: int) -> tuple[np.ndarray, float]:
    """Scale the frame so its long side is `det_size`. Returns image and factor."""
    h, w = frame.shape[:2]
    factor = det_size / float(max(h, w))
    if abs(factor - 1.0) < 1e-6:
        return frame, 1.0
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (nw, nh), interpolation=interp), factor


class YuNetBackend:
    """YuNet through cv2.FaceDetectorYN. Apache 2.0, 232 KB, CPU. Gives landmarks."""

    name = "yunet"

    def __init__(self, conf: float = RAW_FLOOR, nms: float = 0.30, top_k: int = 5000):
        self.model_path = _require(YUNET_MODEL)
        self.net = cv2.FaceDetectorYN.create(str(self.model_path), "", (320, 320),
                                             conf, nms, top_k)

    def detect_resized(self, image: np.ndarray) -> list[Detection]:
        ih, iw = image.shape[:2]
        self.net.setInputSize((iw, ih))
        _, faces = self.net.detect(image)
        if faces is None:
            return []
        out = []
        for f in faces:
            lm = tuple((float(f[4 + 2 * k]), float(f[5 + 2 * k])) for k in range(5))
            out.append(Detection(float(f[0]), float(f[1]), float(f[2]), float(f[3]),
                                 float(f[-1]), lm, "yunet"))
        return out


class CenterFaceBackend:
    """CenterFace through onnxruntime. Decode ported from deface 1.5.0 (MIT)."""

    name = "centerface"
    input_name = "input.1"
    output_names = ("537", "538", "539", "540")  # heatmap, scale, offset, landmarks

    def __init__(self, conf: float = RAW_FLOOR, nms: float = 0.30,
                 threads: Optional[int] = None):
        import onnxruntime as ort

        self.model_path = _require(CENTERFACE_MODEL)
        self.conf = conf
        self.nms = nms
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), options,
                                            providers=["CPUExecutionProvider"])

    def detect_resized(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        net_w, net_h = int(np.ceil(w / 32) * 32), int(np.ceil(h / 32) * 32)
        sw, sh = net_w / float(w), net_h / float(h)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(rgb, 1.0, (net_w, net_h), (0, 0, 0), False, False)
        heatmap, scale, offset, lms = self.session.run(list(self.output_names),
                                                       {self.input_name: blob})
        hm = np.squeeze(heatmap)
        rows, cols = np.where(hm > self.conf)
        if rows.size == 0:
            return []
        scores = hm[rows, cols]
        bh = np.exp(scale[0, 0][rows, cols]) * 4
        bw = np.exp(scale[0, 1][rows, cols]) * 4
        x1 = np.clip((cols + offset[0, 1][rows, cols] + 0.5) * 4 - bw / 2, 0, net_w)
        y1 = np.clip((rows + offset[0, 0][rows, cols] + 0.5) * 4 - bh / 2, 0, net_h)
        x2 = np.minimum(x1 + bw, net_w)
        y2 = np.minimum(y1 + bh, net_h)
        dets = []
        for i in range(rows.size):
            ww, hh = (x2[i] - x1[i]) / sw, (y2[i] - y1[i]) / sh
            if ww <= 1 or hh <= 1:
                continue
            lm = tuple((float(lms[0, 2 * j + 1, rows[i], cols[i]] * bw[i] + x1[i]) / sw,
                        float(lms[0, 2 * j, rows[i], cols[i]] * bh[i] + y1[i]) / sh)
                       for j in range(5))
            dets.append(Detection(float(x1[i] / sw), float(y1[i] / sh), float(ww), float(hh),
                                  float(scores[i]), lm, "centerface"))
        return nms_detections(dets, self.nms)


RawCandidates = dict[str, list[Detection]]  # backend name -> detections, floor score


def _plausible(d: Detection, shape, settings: Settings) -> bool:
    cap = settings.max_face_frac * max(shape[0], shape[1])
    if d.long_side > cap or d.w < 2 or d.h < 2:
        return False
    aspect = d.w / d.h
    return settings.min_aspect <= aspect <= settings.max_aspect


def filter_candidates(raw: RawCandidates, settings: Settings, shape) -> list[Detection]:
    """Turn raw candidates into faces the pipeline trusts. Pure, cheap, cacheable."""
    yun = [d for d in raw.get("yunet", []) if d.score >= settings.conf]
    cf_all = raw.get("centerface", [])
    if settings.engine == "yunet":
        primary = yun
    elif settings.engine == "centerface":
        primary = [d for d in cf_all if d.score >= settings.conf]
    else:
        primary = yun + [d for d in cf_all if d.score >= settings.conf]
    primary = [d for d in primary if _plausible(d, shape, settings)]

    if settings.engine == "yunet" and settings.verify:
        confirmers = [d for d in cf_all if d.score >= settings.verify_conf]
        primary = [replace(d, verified=True) for d in primary
                   if any(iou(d, c) >= settings.verify_iou for c in confirmers)]
    return nms_detections(primary, settings.nms_detect)


def weak_candidates(raw: RawCandidates, settings: Settings, shape) -> list[Detection]:
    """Plausible primary boxes that did not pass every check.

    Only the tracker uses these, and only to continue a track that confirmed
    detections already started. Two kinds: boxes between conf_weak and conf,
    and boxes at full threshold that the second detector did not confirm.
    """
    source = "centerface" if settings.engine == "centerface" else "yunet"
    weak = [d for d in raw.get(source, [])
            if settings.conf_weak <= d.score < settings.conf and _plausible(d, shape, settings)]
    if settings.engine == "yunet" and settings.verify:
        confirmers = [d for d in raw.get("centerface", []) if d.score >= settings.verify_conf]
        weak += [d for d in raw.get("yunet", [])
                 if d.score >= settings.conf and _plausible(d, shape, settings)
                 and not any(iou(d, c) >= settings.verify_iou for c in confirmers)]
    return nms_detections(weak, settings.nms_detect)


class DetectorBank:
    """Runs every needed backend at every det_size. Holds the models."""

    def __init__(self, settings: Settings, threads: Optional[int] = None):
        """`threads` caps the CPU threads each model uses. A pool of workers
        should pass cpu_count // workers, or every worker spins up every core
        and they all crawl."""
        self.settings = settings
        self.yunet = None
        self.centerface = None
        if threads:
            cv2.setNumThreads(threads)
        needs_yunet = settings.engine in ("yunet", "both")
        needs_cf = settings.engine in ("centerface", "both") or (
            settings.engine == "yunet" and settings.verify)
        if needs_yunet:
            self.yunet = YuNetBackend(RAW_FLOOR, settings.nms_yunet)
        if needs_cf:
            self.centerface = CenterFaceBackend(RAW_FLOOR, settings.nms_yunet, threads)

    @property
    def backends(self) -> list:
        return [b for b in (self.yunet, self.centerface) if b is not None]

    @property
    def model_hashes(self) -> dict[str, str]:
        return {b.name: model_sha256(b.model_path) for b in self.backends}

    def detect_raw(self, frame: np.ndarray) -> RawCandidates:
        """Every candidate from every backend at every size, floor threshold."""
        raw: RawCandidates = {}
        for backend in self.backends:
            found: list[Detection] = []
            for det_size in self.settings.det_sizes:
                image, factor = resize_long_side(frame, det_size)
                found.extend(d.scaled(factor) for d in backend.detect_resized(image))
            raw[backend.name] = nms_detections(found, self.settings.nms_detect)
        return raw

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Faces in a BGR frame, after every check in the settings."""
        return filter_candidates(self.detect_raw(frame), self.settings, frame.shape[:2])
