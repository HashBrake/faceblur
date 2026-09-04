"""Face detector backends: YuNet, CenterFace, and the union of both.

Every backend returns boxes as [x, y, w, h, score] in source frame coordinates.

The pipeline scans each frame at fixed absolute sizes, not at the source
resolution. Cost grows with the square of the detection size, so a 4K source and
a 540p source cost the same to scan. Small sizes find small faces and large sizes
find large faces, so the default runs two sizes and merges the result.
"""
from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from .settings import Settings
from .track import nms_merge


def _base_dir() -> Path:
    """The folder that holds `models/`.

    A PyInstaller build unpacks its data files next to the executable, so the
    frozen app looks there rather than beside this source file.
    """
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
        raise ModelMissing(
            f"Could not find the model file {path}. Reinstall FaceBlur."
        )
    return path


def resize_long_side(frame: np.ndarray, det_size: int) -> tuple[np.ndarray, float]:
    """Scale the frame so its long side is `det_size`. Returns the image and the factor.

    The factor can be above 1. Upscaling a small source is how the pipeline finds
    faces that are only a few pixels wide in the original.
    """
    h, w = frame.shape[:2]
    factor = det_size / float(max(h, w))
    if abs(factor - 1.0) < 1e-6:
        return frame, 1.0
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (nw, nh), interpolation=interp), factor


class Backend(Protocol):
    name: str
    model_path: Path

    def detect_resized(self, image: np.ndarray) -> list[list[float]]:
        """Detect on an already resized BGR image. Boxes in that image's coordinates."""


class YuNetBackend:
    """YuNet through cv2.FaceDetectorYN. Apache 2.0, 232 KB, CPU."""

    name = "yunet"

    def __init__(self, conf: float, nms: float = 0.30, top_k: int = 5000):
        self.model_path = _require(YUNET_MODEL)
        self.conf = conf
        self.net = cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320), conf, nms, top_k
        )

    def detect_resized(self, image: np.ndarray) -> list[list[float]]:
        ih, iw = image.shape[:2]
        self.net.setInputSize((iw, ih))
        _, faces = self.net.detect(image)
        if faces is None:
            return []
        # YuNet returns x, y, w, h, five landmarks, then the score last.
        return [[float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])]
                for f in faces]


class CenterFaceBackend:
    """CenterFace through onnxruntime.

    The model and the decode step below come from deface 1.5.0 (MIT,
    Copyright (c) 2020 Martin Drawitsch), file deface/centerface.py. The decode
    is rewritten with array operations instead of a Python loop over every
    heatmap peak, which matters at 1280 px on long recordings. FaceBlur does not
    import deface at runtime.
    """

    name = "centerface"
    input_name = "input.1"
    # heatmap, scale, offset, landmarks
    output_names = ("537", "538", "539", "540")

    def __init__(self, conf: float, nms: float = 0.30):
        import onnxruntime as ort

        self.model_path = _require(CENTERFACE_MODEL)
        self.conf = conf
        self.nms = nms
        ort.set_default_logger_severity(3)  # hide the unused initializer warnings
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

    def detect_resized(self, image: np.ndarray) -> list[list[float]]:
        h, w = image.shape[:2]
        # The network needs both spatial sizes to divide by 32.
        net_w, net_h = int(np.ceil(w / 32) * 32), int(np.ceil(h / 32) * 32)
        scale_w, scale_h = net_w / float(w), net_h / float(h)

        # The model was trained on RGB. OpenCV hands over BGR.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(
            rgb, scalefactor=1.0, size=(net_w, net_h), mean=(0, 0, 0),
            swapRB=False, crop=False,
        )
        heatmap, scale, offset, _lms = self.session.run(
            list(self.output_names), {self.input_name: blob}
        )
        boxes = self._decode(heatmap, scale, offset, (net_h, net_w))
        if not len(boxes):
            return []
        boxes[:, 0] /= scale_w
        boxes[:, 2] /= scale_w
        boxes[:, 1] /= scale_h
        boxes[:, 3] /= scale_h
        # Decode gives x1, y1, x2, y2, score. The rest of FaceBlur wants x, y, w, h.
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0]
        out[:, 1] = boxes[:, 1]
        out[:, 2] = boxes[:, 2] - boxes[:, 0]
        out[:, 3] = boxes[:, 3] - boxes[:, 1]
        out[:, 4] = boxes[:, 4]
        keep = out[:, 2] > 1
        keep &= out[:, 3] > 1
        return out[keep].tolist()

    def _decode(self, heatmap, scale, offset, size) -> np.ndarray:
        """Turn the four output tensors into x1, y1, x2, y2, score rows."""
        hm = np.squeeze(heatmap)
        s0, s1 = scale[0, 0], scale[0, 1]
        o0, o1 = offset[0, 0], offset[0, 1]
        rows, cols = np.where(hm > self.conf)
        if rows.size == 0:
            return np.empty((0, 5), np.float32)

        scores = hm[rows, cols]
        # The heatmap is a quarter of the network input in each direction.
        bh = np.exp(s0[rows, cols]) * 4
        bw = np.exp(s1[rows, cols]) * 4
        x1 = np.clip((cols + o1[rows, cols] + 0.5) * 4 - bw / 2, 0, size[1])
        y1 = np.clip((rows + o0[rows, cols] + 0.5) * 4 - bh / 2, 0, size[0])
        x2 = np.minimum(x1 + bw, size[1])
        y2 = np.minimum(y1 + bh, size[0])

        dets = np.stack([x1, y1, x2, y2, scores], axis=1).astype(np.float32)
        # Reuse the shared merger, which wants x, y, w, h.
        as_xywh = dets.copy()
        as_xywh[:, 2] = dets[:, 2] - dets[:, 0]
        as_xywh[:, 3] = dets[:, 3] - dets[:, 1]
        merged = nms_merge(as_xywh, self.nms)
        if not merged:
            return np.empty((0, 5), np.float32)
        m = np.asarray(merged, np.float32)
        back = m.copy()
        back[:, 2] = m[:, 0] + m[:, 2]
        back[:, 3] = m[:, 1] + m[:, 3]
        return back


class DetectorBank:
    """Runs every enabled backend at every det_size and merges the result."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backends: list[Backend] = []
        if settings.engine in ("yunet", "both"):
            self.backends.append(YuNetBackend(settings.conf, settings.nms_yunet))
        if settings.engine in ("centerface", "both"):
            self.backends.append(CenterFaceBackend(settings.conf, settings.nms_yunet))
        if not self.backends:
            raise ValueError(f"engine {settings.engine!r} enabled no detector")

    @property
    def model_hashes(self) -> dict[str, str]:
        return {b.name: model_sha256(b.model_path) for b in self.backends}

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """Detect faces in a BGR frame. Boxes come back in frame coordinates."""
        found: list[list[float]] = []
        for det_size in self.settings.det_sizes:
            image, factor = resize_long_side(frame, det_size)
            for backend in self.backends:
                for x, y, w, h, score in backend.detect_resized(image):
                    found.append([x / factor, y / factor, w / factor, h / factor, score])
        return nms_merge(found, self.settings.nms_detect)
