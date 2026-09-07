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
# Primary candidates get a confirmation crop from min(conf, CROP_FLOOR) up.
# Confirmation only matters for a box at conf or above, so nothing is lost
# below that; the floor keeps a cache built at one conf usable for a sweep
# down to CROP_FLOOR. Cutting crops for every box from 0.2 up cost twice the
# confirmation time on busy footage.
CROP_FLOOR = 0.5
# Crops go to the second detector in batches of this size.
CROP_BATCH = 8
# UltraFace has no suppression of its own and scores thousands of anchors
# above the raw floor on every crop. It is only ever consulted at third_conf,
# so it keeps its own floor and the top anchors before suppression.
ULTRAFACE_FLOOR = 0.3
ULTRAFACE_TOP_K = 64


def _base_dir() -> Path:
    """The folder that holds `models/`. A frozen build unpacks beside the exe."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


MODEL_DIR = _base_dir() / "models"
YUNET_MODEL = MODEL_DIR / "yunet.onnx"
YUNET_DYNAMIC_MODEL = MODEL_DIR / "yunet_dynamic.onnx"
CENTERFACE_MODEL = MODEL_DIR / "centerface_dynamic.onnx"
ULTRAFACE_MODEL = MODEL_DIR / "ultraface_dynamic.onnx"


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


def nms_indices(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Greedy NMS on arrays. boxes are x, y, w, h rows. Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ov = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= thr]
    return np.asarray(keep, dtype=np.int64)


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


# Worker processes that share one consumer GPU. Each holds every model's
# graphs at every size, about 850 MB on an RTX 3070, and the desktop keeps
# about 2 GB of the 8 for itself. Six workers reached 7.3 GB and DirectML
# crawled; four leave room. The GPU is the bottleneck, not the worker count.
GPU_WORKERS = 4


def default_workers(cpu_count: Optional[int] = None, device: str = "auto") -> int:
    """Half the cores, capped when the models would share a GPU."""
    import os
    n = max(1, (cpu_count or os.cpu_count() or 2) // 2)
    if providers_for(device)[0] != "CPUExecutionProvider":
        n = min(n, GPU_WORKERS)
    return n


def providers_for(device: str) -> list[str]:
    """onnxruntime providers for a device choice, best first."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    wanted = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
    if device == "gpu" and not wanted:
        raise RuntimeError("No GPU provider is available to onnxruntime on this machine.")
    return wanted + ["CPUExecutionProvider"]


class SessionPool:
    """onnxruntime sessions for one model.

    The CPU provider takes any input shape, so it gets one session over the
    dynamic model. DirectML needs a static graph: it trusts the shape entries
    the export left inside the graph, which describe the original 640x640 or
    32x32 input, and either fails or computes at that size. So on the GPU each
    distinct input shape gets its own session, built from the dynamic model
    with those stale entries stripped and every dimension set to the real
    size. A video uses two detection sizes, so the pool stays tiny.
    """

    def __init__(self, path: Path, device: str, threads: Optional[int],
                 input_name: str, output_dims):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.path = path
        self.threads = threads
        self.input_name = input_name
        self.output_dims = output_dims        # (height, width) -> {output: dims}
        self.providers = providers_for(device)
        self.static = self.providers[0] != "CPUExecutionProvider"
        self._sessions: dict = {}
        self._proto = None
        first = self._make(None)
        self.provider = first.get_providers()[0] if not self.static else self.providers[0]
        self.outputs = [o.name for o in first.get_outputs()]
        if not self.static:
            self._sessions[None] = first

    def _options(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if self.threads:
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
        return options

    def _make(self, shape):
        import onnxruntime as ort

        if shape is None:
            # The dynamic model on the CPU. Also used once to read the output
            # names when the GPU path is active.
            return ort.InferenceSession(str(self.path), self._options(),
                                        providers=["CPUExecutionProvider"])
        import onnx
        from onnx.tools.update_model_dims import update_inputs_outputs_dims

        if self._proto is None:
            self._proto = onnx.load(str(self.path))
            del self._proto.graph.value_info[:]
        n, h, w = shape
        proto = onnx.ModelProto()
        proto.CopyFrom(self._proto)
        ins = {i.name: [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
               for i in proto.graph.input}
        ins[self.input_name] = [n, 3, h, w]
        outs = self.output_dims(n, h, w)
        proto = onnx.shape_inference.infer_shapes(update_inputs_outputs_dims(proto, ins, outs))
        session = ort.InferenceSession(proto.SerializeToString(), self._options(),
                                       providers=self.providers)
        self.provider = session.get_providers()[0]
        return session

    def run(self, blob: np.ndarray) -> list[np.ndarray]:
        key = (blob.shape[0], blob.shape[2], blob.shape[3]) if self.static else None
        session = self._sessions.get(key)
        if session is None:
            session = self._make(key)
            self._sessions[key] = session
        return session.run(None, {self.input_name: blob})


def _yunet_output_dims(n: int, h: int, w: int) -> dict:
    last = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}
    return {f"{k}_{s}": [n, (h // s) * (w // s), last[k]] for k in last for s in (8, 16, 32)}


def _centerface_output_dims(n: int, h: int, w: int) -> dict:
    return {"537": [n, 1, h // 4, w // 4], "538": [n, 2, h // 4, w // 4],
            "539": [n, 2, h // 4, w // 4], "540": [n, 10, h // 4, w // 4]}


def _pad32(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pad bottom and right so both sides divide by 32. OpenCV does the same."""
    h, w = image.shape[:2]
    pw, ph = int(np.ceil(w / 32) * 32), int(np.ceil(h / 32) * 32)
    if (pw, ph) == (w, h):
        return image, w, h
    return cv2.copyMakeBorder(image, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0), pw, ph


class YuNetOrtBackend:
    """YuNet through onnxruntime, so it can run on the GPU.

    The decode reproduces cv2.FaceDetectorYN (OpenCV 4.x, face_detect.cpp):
    three strides, score = sqrt(cls * obj), box centre and size from the
    grid cell, landmarks likewise. Validated box for box against OpenCV.
    """

    name = "yunet"
    strides = (8, 16, 32)

    def __init__(self, conf: float = RAW_FLOOR, nms: float = 0.30, top_k: int = 5000,
                 device: str = "auto", threads: Optional[int] = None):
        self.model_path = _require(YUNET_DYNAMIC_MODEL)
        self.conf = conf
        self.nms = nms
        self.top_k = top_k
        self.pool = SessionPool(self.model_path, device, threads, "input", _yunet_output_dims)
        self.provider = self.pool.provider
        self.outputs = self.pool.outputs

    def detect_resized(self, image: np.ndarray) -> list[Detection]:
        padded, pw, ph = _pad32(image)
        # BGR, no scaling. blobFromImage does the transpose and cast in C.
        blob = cv2.dnn.blobFromImage(padded, 1.0, (pw, ph), (0, 0, 0), False, False)
        outs = dict(zip(self.outputs, self.pool.run(blob)))
        boxes, scores, lms = [], [], []
        for s in self.strides:
            cls = np.clip(outs[f"cls_{s}"][0, :, 0], 0, 1)
            obj = np.clip(outs[f"obj_{s}"][0, :, 0], 0, 1)
            score = np.sqrt(cls * obj)
            keep = np.where(score >= self.conf)[0]
            if keep.size == 0:
                continue
            cols = pw // s
            row, col = keep // cols, keep % cols
            bbox = outs[f"bbox_{s}"][0, keep]
            kps = outs[f"kps_{s}"][0, keep]
            bw = np.exp(bbox[:, 2]) * s
            bh = np.exp(bbox[:, 3]) * s
            x1 = (col + bbox[:, 0]) * s - bw / 2
            y1 = (row + bbox[:, 1]) * s - bh / 2
            boxes.append(np.stack([x1, y1, bw, bh], axis=1))
            scores.append(score[keep])
            lm = np.empty((keep.size, 5, 2), np.float64)
            lm[:, :, 0] = (col[:, None] + kps[:, 0::2]) * s
            lm[:, :, 1] = (row[:, None] + kps[:, 1::2]) * s
            lms.append(lm)
        if not boxes:
            return []
        b = np.concatenate(boxes).astype(np.float64)
        sc = np.concatenate(scores).astype(np.float64)
        lm = np.concatenate(lms)
        keep = nms_indices(b, sc, self.nms)[: self.top_k]
        return [Detection(float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3]),
                          float(sc[i]), tuple((float(x), float(y)) for x, y in lm[i]), "yunet")
                for i in keep]


class CenterFaceBackend:
    """CenterFace through onnxruntime. Decode ported from deface 1.5.0 (MIT)."""

    name = "centerface"
    input_name = "input.1"
    output_names = ("537", "538", "539", "540")  # heatmap, scale, offset, landmarks

    def __init__(self, conf: float = RAW_FLOOR, nms: float = 0.30,
                 threads: Optional[int] = None, device: str = "auto"):
        self.model_path = _require(CENTERFACE_MODEL)
        self.conf = conf
        self.nms = nms
        self.pool = SessionPool(self.model_path, device, threads, self.input_name,
                                _centerface_output_dims)
        self.provider = self.pool.provider

    def detect_resized(self, image: np.ndarray) -> list[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run several images of one size in a single call. Boxes per image."""
        if not images:
            return []
        h, w = images[0].shape[:2]
        net_w, net_h = int(np.ceil(w / 32) * 32), int(np.ceil(h / 32) * 32)
        sw, sh = net_w / float(w), net_h / float(h)
        blobs = [cv2.dnn.blobFromImage(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 1.0, (net_w, net_h),
                                       (0, 0, 0), False, False) for im in images]
        blob = np.concatenate(blobs, axis=0)
        outs = dict(zip(self.pool.outputs, self.pool.run(blob)))
        heatmap, scale, offset, lms = (outs[n] for n in self.output_names)
        return [self._decode(heatmap[k], scale[k], offset[k], lms[k], net_w, net_h, sw, sh)
                for k in range(len(images))]

    def _decode(self, hm, scale, offset, lms, net_w, net_h, sw, sh) -> list[Detection]:
        hm = np.squeeze(hm)
        rows, cols = np.where(hm > self.conf)
        if rows.size == 0:
            return []
        scores = hm[rows, cols]
        bh = np.exp(scale[0][rows, cols]) * 4
        bw = np.exp(scale[1][rows, cols]) * 4
        x1 = np.clip((cols + offset[1][rows, cols] + 0.5) * 4 - bw / 2, 0, net_w)
        y1 = np.clip((rows + offset[0][rows, cols] + 0.5) * 4 - bh / 2, 0, net_h)
        x2 = np.minimum(x1 + bw, net_w)
        y2 = np.minimum(y1 + bh, net_h)
        bw_s, bh_s = (x2 - x1) / sw, (y2 - y1) / sh
        ok = (bw_s > 1) & (bh_s > 1)
        if not ok.any():
            return []
        b = np.stack([x1 / sw, y1 / sh, bw_s, bh_s], axis=1)[ok].astype(np.float64)
        sc = scores[ok].astype(np.float64)
        rows_k, cols_k = rows[ok], cols[ok]
        x1k, y1k, bwk, bhk = x1[ok], y1[ok], bw[ok], bh[ok]
        keep = nms_indices(b, sc, self.nms)
        dets = []
        for i in keep:
            r, c = rows_k[i], cols_k[i]
            lm = tuple((float(lms[2 * j + 1, r, c] * bwk[i] + x1k[i]) / sw,
                        float(lms[2 * j, r, c] * bhk[i] + y1k[i]) / sh) for j in range(5))
            dets.append(Detection(float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3]),
                                  float(sc[i]), lm, "centerface"))
        return dets


def _ultraface_output_dims(n: int, h: int, w: int) -> dict:
    # The anchor layout is built for 320x240; that is the only size used.
    return {"scores": [n, 4420, 2], "boxes": [n, 4420, 4]}


class UltraFaceBackend:
    """UltraFace RFB-320 through onnxruntime. MIT, from Linzaer's
    Ultra-Light-Fast-Generic-Face-Detector-1MB. A third family, asked only
    when CenterFace is unsure about a YuNet candidate."""

    name = "ultraface"
    input_name = "input"
    width, height = 320, 240

    def __init__(self, conf: float = ULTRAFACE_FLOOR, nms: float = 0.30,
                 threads: Optional[int] = None, device: str = "auto"):
        self.model_path = _require(ULTRAFACE_MODEL)
        self.conf = conf
        self.nms = nms
        self.pool = SessionPool(self.model_path, device, threads, self.input_name,
                                _ultraface_output_dims)
        self.provider = self.pool.provider

    def canvas(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Fit an image into the 320x240 input, top left, and the scale used."""
        h, w = image.shape[:2]
        f = min(self.width / float(w), self.height / float(h))
        small = cv2.resize(image, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                           interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        out = np.zeros((self.height, self.width, 3), np.uint8)
        out[:small.shape[0], :small.shape[1]] = small
        return out, f

    def detect_resized(self, image: np.ndarray) -> list[Detection]:
        canvas, f = self.canvas(image)
        return [d.scaled(f) for d in self.detect_batch([canvas])[0]]

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run several 320x240 canvases in one call. Boxes per image, in
        canvas pixels."""
        if not images:
            return []
        blob = np.stack([((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.0)
                          / 128.0).transpose(2, 0, 1) for im in images])
        outs = dict(zip(self.pool.outputs, self.pool.run(blob)))
        scores, boxes = outs["scores"], outs["boxes"]
        result = []
        for k in range(len(images)):
            p = scores[k, :, 1]
            sel = np.where(p >= self.conf)[0]
            if sel.size == 0:
                result.append([])
                continue
            if sel.size > ULTRAFACE_TOP_K:
                sel = sel[np.argsort(-p[sel])[:ULTRAFACE_TOP_K]]
            b = boxes[k, sel].astype(np.float64)
            x1 = np.clip(b[:, 0] * self.width, 0, self.width)
            y1 = np.clip(b[:, 1] * self.height, 0, self.height)
            x2 = np.clip(b[:, 2] * self.width, 0, self.width)
            y2 = np.clip(b[:, 3] * self.height, 0, self.height)
            xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
            ok = (xywh[:, 2] > 1) & (xywh[:, 3] > 1)
            xywh, sc = xywh[ok], p[sel][ok].astype(np.float64)
            keep = nms_indices(xywh, sc, self.nms)
            result.append([Detection(float(xywh[i, 0]), float(xywh[i, 1]), float(xywh[i, 2]),
                                     float(xywh[i, 3]), float(sc[i]), None, "ultraface")
                           for i in keep])
        return result


RawCandidates = dict[str, list[Detection]]  # backend name -> detections, floor score


def _plausible(d: Detection, shape, settings: Settings, cap_frac: Optional[float] = None) -> bool:
    cap = (settings.max_face_frac if cap_frac is None else cap_frac) * max(shape[0], shape[1])
    if d.long_side > cap or d.w < 2 or d.h < 2:
        return False
    aspect = d.w / d.h
    return settings.min_aspect <= aspect <= settings.max_aspect


def _best_over(view: list[Detection], d: Detection, thr: float) -> float:
    best = 0.0
    for c in view:
        if c.score > best and iou(d, c) >= thr:
            best = c.score
    return best


def confirmation(d: Detection, raw: RawCandidates, settings: Settings) -> tuple[float, bool]:
    """What the confirming detectors say about a primary box: CenterFace's
    score, taking a mirror or tight view into account only where that view
    clears the higher bar those views must meet, and whether the third
    detector fired on it."""
    best = _best_over(raw.get("centerface", []), d, settings.verify_iou)
    extra = []
    if settings.confirm_tta >= 1:
        extra.append(raw.get("centerface_flip", []))
    if settings.confirm_tta >= 2:
        extra.append(raw.get("centerface_tight", []))
    for view in extra:
        v = _best_over(view, d, settings.verify_iou)
        if v >= settings.verify_conf_view:
            best = max(best, v)
    third = any(u.score >= settings.third_conf and iou(d, u) >= settings.verify_iou
                for u in raw.get("ultraface", []))
    return best, third


def confirmed(d: Detection, raw: RawCandidates, settings: Settings) -> bool:
    """CenterFace agrees, or it is unsure and UltraFace agrees."""
    score, third = confirmation(d, raw, settings)
    if score >= settings.verify_conf:
        return True
    return settings.third_opinion and score >= settings.verify_conf_low and third


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
        primary = [replace(d, verified=True) for d in primary if confirmed(d, raw, settings)]
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
        for d in raw.get("yunet", []):
            if d.score < settings.conf:
                continue
            if _plausible(d, shape, settings):
                if not confirmed(d, raw, settings):
                    weak.append(d)
            elif _plausible(d, shape, settings, max(settings.grow_face_frac,
                                                    settings.max_face_frac)) \
                    and confirmed(d, raw, settings):
                # Larger than a face may be to start a track, confirmed all
                # the same: a person walking up to the camera. A track that
                # exists may follow it.
                weak.append(replace(d, verified=True))
    return nms_detections(weak, settings.nms_detect)


def _crop_reflect(frame: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """A square crop that may run past the frame edge; the missing part is
    the picture reflected, so a face cut by the edge keeps its context."""
    H, W = frame.shape[:2]
    x1, y1 = x0 + side, y0 + side
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - W), max(0, y1 - H)
    inner = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if inner.size == 0:
        return np.zeros((side, side, 3), np.uint8)
    if left or top or right or bottom:
        # Reflection needs at least as much picture as it mirrors; a crop far
        # past the edge gets the edge pixels repeated instead.
        ih, iw = inner.shape[:2]
        mode = cv2.BORDER_REFLECT_101 if max(left, right) < iw and max(top, bottom) < ih \
            else cv2.BORDER_REPLICATE
        inner = cv2.copyMakeBorder(inner, top, bottom, left, right, mode)
    return inner


def _offset(d: Detection, x0: int, y0: int) -> Detection:
    lm = None if d.landmarks is None else tuple((px + x0, py + y0) for px, py in d.landmarks)
    return replace(d, x=d.x + x0, y=d.y + y0, landmarks=lm)


def _unflip(d: Detection, side: int) -> Detection:
    """Map a box found in the mirror image back, in crop coordinates."""
    lm = None if d.landmarks is None else tuple((side - px, py) for px, py in d.landmarks)
    return replace(d, x=side - (d.x + d.w), landmarks=lm)


class DetectorBank:
    """Runs every needed backend at every det_size. Holds the models."""

    def __init__(self, settings: Settings, threads: Optional[int] = None):
        """`threads` caps the CPU threads each model uses. A pool of workers
        should pass cpu_count // workers, or every worker spins up every core
        and they all crawl."""
        self.settings = settings
        self.yunet = None
        self.centerface = None
        self.ultraface = None
        if threads:
            cv2.setNumThreads(threads)
        needs_yunet = settings.engine in ("yunet", "both")
        needs_cf = settings.engine in ("centerface", "both") or (
            settings.engine == "yunet" and settings.verify)
        if needs_yunet:
            self.yunet = YuNetOrtBackend(RAW_FLOOR, settings.nms_yunet, device=settings.device,
                                         threads=threads)
        if needs_cf:
            self.centerface = CenterFaceBackend(RAW_FLOOR, settings.nms_yunet, threads,
                                                device=settings.device)
        if needs_cf and settings.engine == "yunet" and settings.confirm_on_crops \
                and settings.third_opinion:
            self.ultraface = UltraFaceBackend(ULTRAFACE_FLOOR, settings.nms_yunet, threads,
                                              device=settings.device)

    @property
    def backends(self) -> list:
        return [b for b in (self.yunet, self.centerface, self.ultraface) if b is not None]

    @property
    def model_hashes(self) -> dict[str, str]:
        return {b.name: model_sha256(b.model_path) for b in self.backends}

    @property
    def compute(self) -> dict[str, str]:
        """Which onnxruntime provider each model actually ran on."""
        return {b.name: getattr(b, "provider", "cpu") for b in self.backends}

    def detect_raw(self, frame: np.ndarray) -> RawCandidates:
        """Every candidate from every backend, floor threshold.

        The primary detector scans the whole frame at every size. When the
        second detector only confirms, it looks at a crop around each primary
        candidate instead of the whole frame.
        """
        s = self.settings
        raw: RawCandidates = {}
        crops_only = (s.engine == "yunet" and s.verify and s.confirm_on_crops
                      and self.centerface is not None)
        for backend in self.backends:
            if backend is self.ultraface or (backend is self.centerface and crops_only):
                continue
            found: list[Detection] = []
            for det_size in s.det_sizes:
                image, factor = resize_long_side(frame, det_size)
                found.extend(d.scaled(factor) for d in backend.detect_resized(image))
            raw[backend.name] = nms_detections(found, s.nms_detect)
        if crops_only:
            raw.update(self.confirm_on_crops(frame, raw.get("yunet", [])))
        return raw

    def confirm_on_crops(self, frame: np.ndarray, candidates: list[Detection]) -> RawCandidates:
        """The confirming detectors on a crop around each candidate.

        Returns CenterFace's boxes for the plain crop ("centerface"), its
        mirror image ("centerface_flip") and a tighter crop ("centerface_tight"),
        and UltraFace's boxes on the plain crop ("ultraface"), all in frame
        coordinates. A crop that runs past the frame edge is filled with the
        picture reflected, so a face cut by the edge still sits in context.
        """
        s = self.settings
        H, W = frame.shape[:2]
        size = s.crop_size
        # The crop path keeps its own floor and cap, wider than the settings, so
        # that a sweep over conf or max_face_frac still has raw candidates to
        # work with. Nothing below conf is ever confirmed anyway.
        floor = min(s.conf, CROP_FLOOR)
        cap = min(1.0, max(2.0 * s.max_face_frac, s.grow_face_frac)) * max(H, W)
        windows: list[tuple[int, int, int]] = []       # x0, y0, side of the plain crop
        for d in candidates:
            if d.score < floor or d.long_side > cap or d.w < 2 or d.h < 2:
                continue
            side = int(round(max(48.0, d.long_side * s.crop_scale)))
            x0 = int(round(d.cx - side / 2))
            y0 = int(round(d.cy - side / 2))
            if any(abs(x0 - a) < 8 and abs(y0 - b) < 8 and abs(side - c) < 8
                   for a, b, c in windows):
                continue                        # two candidates on one face share a crop
            windows.append((x0, y0, side))
        out: RawCandidates = {"centerface": [], "centerface_flip": [], "centerface_tight": [],
                              "ultraface": []}
        if not windows:
            return out
        # Every view of every crop shares one shape, so the whole lot goes to
        # the GPU in a few calls.
        views: list[tuple[str, np.ndarray, int, int, float, int]] = []
        third: list[tuple[np.ndarray, int, int, float]] = []
        for x0, y0, side in windows:
            plain = _crop_reflect(frame, x0, y0, side)
            factor = size / float(side)
            image = cv2.resize(plain, (size, size), interpolation=cv2.INTER_AREA if factor < 1
                               else cv2.INTER_LINEAR)
            views.append(("centerface", image, x0, y0, factor, side))
            views.append(("centerface_flip", cv2.flip(image, 1), x0, y0, factor, side))
            tight_side = max(16, int(round(side * 0.6)))
            tx0, ty0 = x0 + (side - tight_side) // 2, y0 + (side - tight_side) // 2
            tight = _crop_reflect(frame, tx0, ty0, tight_side)
            tf = size / float(tight_side)
            views.append(("centerface_tight",
                          cv2.resize(tight, (size, size), interpolation=cv2.INTER_AREA if tf < 1
                                     else cv2.INTER_LINEAR), tx0, ty0, tf, tight_side))
            if self.ultraface is not None:
                canvas, uf = self.ultraface.canvas(plain)
                third.append((canvas, x0, y0, uf))
        for start in range(0, len(views), CROP_BATCH):
            chunk = views[start:start + CROP_BATCH]
            padded = [v[1] for v in chunk] + [np.zeros_like(chunk[0][1])] * (CROP_BATCH - len(chunk))
            for (name, _, x0, y0, factor, side), dets in zip(
                    chunk, self.centerface.detect_batch(padded)[: len(chunk)]):
                for c in dets:
                    c = c.scaled(factor)
                    if name == "centerface_flip":
                        c = _unflip(c, side)
                    out[name].append(_offset(c, x0, y0))
        for start in range(0, len(third), CROP_BATCH):
            chunk = third[start:start + CROP_BATCH]
            padded = [v[0] for v in chunk] + [np.zeros_like(chunk[0][0])] * (CROP_BATCH - len(chunk))
            for (_, x0, y0, uf), dets in zip(chunk, self.ultraface.detect_batch(padded)[: len(chunk)]):
                out["ultraface"].extend(_offset(u.scaled(uf), x0, y0) for u in dets)
        return {k: nms_detections(v, s.nms_detect) for k, v in out.items()}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Faces in a BGR frame, after every check in the settings."""
        return filter_candidates(self.detect_raw(frame), self.settings, frame.shape[:2])
