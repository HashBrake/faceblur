"""Every knob the pipeline has.

The defaults changed after the precision audit (docs/precision_report.md). The
first build was tuned to never miss a face and accepted over blurring. This
build destroys pixels only where several independent checks agree that a face
is there, and it masks the smallest region that hides identity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

ENGINES = ("yunet", "centerface", "both")
DEVICES = ("auto", "gpu", "cpu")
ENCODERS = ("auto", "nvenc", "x264")
MODES = ("blur", "pixelate", "solid")

# Video extensions the batch walker treats as input.
VIDEO_EXT = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
)


class SettingsError(ValueError):
    """A setting is outside the range the pipeline accepts."""


@dataclass(frozen=True)
class Settings:
    # --- detection -----------------------------------------------------------
    # yunet: YuNet finds faces, CenterFace confirms them (see verify).
    # both: union of the two, no confirmation. centerface: CenterFace alone.
    engine: str = "yunet"
    # YuNet score threshold. Chosen by eval/sweep.py: the highest recall that
    # keeps off-face masking under 0.3 percent and hands untouched.
    conf: float = 0.6
    # Absolute long side lengths to scan at. 640 was dropped: on a 1600 px frame
    # it shrinks a 35 px face to 14 px while a sink becomes a face.
    det_sizes: tuple[int, ...] = (1280, 1920)
    # A YuNet box survives only if CenterFace also fired on it.
    verify: bool = True
    verify_conf: float = 0.3
    verify_iou: float = 0.3
    # Confirmation runs on a crop around each candidate rather than on the
    # whole frame at every size. The crop is crop_scale times the box, scaled
    # so its long side is crop_size pixels. Same confirmations, a fraction of
    # the cost, and one fixed shape for the GPU.
    confirm_on_crops: bool = True
    crop_size: int = 224
    crop_scale: float = 4.0
    # Once a track is confirmed, YuNet alone may keep it alive at this lower
    # threshold, if the box overlaps where the track predicts the face to be.
    # A hand can never start a track, so this costs no precision.
    conf_weak: float = 0.4
    # Largest plausible face, as a share of the frame's long side. The largest
    # real face in the Ego footage was 150 px of 1600. False positives ran to 800.
    # 0.15 of 1600 is 240 px.
    max_face_frac: float = 0.15
    # Faces are roughly square. Anything wider or taller than this is not one.
    min_aspect: float = 0.4
    max_aspect: float = 2.5
    stride: int = 1
    # auto uses the GPU when onnxruntime can see one (DirectML on Windows,
    # CUDA elsewhere) and falls back to the CPU.
    device: str = "auto"

    # --- tracking -----------------------------------------------------------
    # A detection counts only when its track holds at least min_track detections.
    min_track: int = 2
    # Frames a track may go undetected before it closes. Gaps are interpolated.
    max_gap: int = 5
    # Frames the mask extends past the first and last detection, without growth.
    tail: int = 2
    track_iou: float = 0.3

    # --- mask geometry -------------------------------------------------------
    # Ellipse axes as a share of the detector box, rotated to the eye line.
    ellipse_w: float = 1.10
    ellipse_h: float = 1.15
    # Fallback pad for a box that carries no landmarks.
    pad: float = 0.10
    # Soft edge width in pixels.
    feather: float = 6.0

    # --- pixel destruction ---------------------------------------------------
    mode: str = "blur"
    # Blocks across a face after downsampling. 6 leaves nothing to recognise.
    strength: int = 6

    # --- output --------------------------------------------------------------
    # Near lossless. Untouched pixels are training data.
    crf: int = 12
    preset: str = "fast"
    # auto uses NVENC when the GPU and the bundled ffmpeg both support it.
    encoder: str = "auto"
    # NVENC constant quality, the counterpart of crf. 16 is near lossless.
    nvenc_cq: int = 16
    suffix: str = "_blurred"
    replace_existing: bool = False
    # A frame masked beyond this share is flagged in the audit record.
    mask_budget: float = 0.05

    # --- segments ------------------------------------------------------------
    # Detection runs in frame ranges of this many seconds, in parallel. 0 means
    # one range for the whole video.
    chunk_seconds: float = 15.0
    # Stretches with no mask are copied from the source byte for byte. If the
    # joined file does not verify, the video is encoded end to end instead.
    copy_clean: bool = True
    # A clean stretch shorter than this is encoded with its neighbours.
    min_copy_seconds: float = 2.0

    nms_detect: float = 0.35
    nms_yunet: float = 0.30

    def __post_init__(self) -> None:
        object.__setattr__(self, "det_sizes", tuple(int(s) for s in self.det_sizes))
        self.validate()

    def validate(self) -> None:
        if self.engine not in ENGINES:
            raise SettingsError(f"engine must be one of {ENGINES}, got {self.engine!r}")
        if self.mode not in MODES:
            raise SettingsError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.device not in DEVICES:
            raise SettingsError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.encoder not in ENCODERS:
            raise SettingsError(f"encoder must be one of {ENCODERS}, got {self.encoder!r}")
        if not 0.0 < self.conf <= 1.0:
            raise SettingsError(f"conf must be above 0 and at most 1, got {self.conf}")
        if self.chunk_seconds < 0 or self.min_copy_seconds < 0:
            raise SettingsError("chunk_seconds and min_copy_seconds must be 0 or more")
        if self.crop_size < 64 or self.crop_scale < 1.0:
            raise SettingsError("crop_size must be at least 64 and crop_scale at least 1")
        if not 0.0 < self.conf_weak <= self.conf:
            raise SettingsError(f"conf_weak must be above 0 and at most conf, got {self.conf_weak}")
        if not self.det_sizes:
            raise SettingsError("det_sizes must name at least one size")
        if any(s < 64 for s in self.det_sizes):
            raise SettingsError(f"every det_size must be at least 64, got {self.det_sizes}")
        if not 0.0 < self.max_face_frac <= 1.0:
            raise SettingsError(f"max_face_frac must be in (0, 1], got {self.max_face_frac}")
        if self.stride < 1:
            raise SettingsError(f"stride must be at least 1, got {self.stride}")
        if self.min_track < 1:
            raise SettingsError(f"min_track must be at least 1, got {self.min_track}")
        if self.max_gap < 0 or self.tail < 0:
            raise SettingsError("max_gap and tail must be 0 or more")
        if self.ellipse_w <= 0 or self.ellipse_h <= 0:
            raise SettingsError("ellipse_w and ellipse_h must be above 0")
        if self.pad < 0:
            raise SettingsError(f"pad must be 0 or more, got {self.pad}")
        if self.feather < 0:
            raise SettingsError(f"feather must be 0 or more, got {self.feather}")
        if self.strength < 1:
            raise SettingsError(f"strength must be at least 1, got {self.strength}")
        if not 0 <= self.crf <= 51:
            raise SettingsError(f"crf must be between 0 and 51, got {self.crf}")

    def with_changes(self, **changes) -> "Settings":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "engine", "conf", "conf_weak", "det_sizes", "verify", "verify_conf", "verify_iou",
            "confirm_on_crops", "crop_size", "crop_scale",
            "max_face_frac", "min_aspect", "max_aspect", "stride", "device",
            "min_track", "max_gap", "tail", "track_iou",
            "ellipse_w", "ellipse_h", "pad", "feather",
            "mode", "strength", "crf", "preset", "encoder", "nvenc_cq", "device", "mask_budget",
            "chunk_seconds", "copy_clean", "min_copy_seconds",
            "nms_detect", "nms_yunet")}
        d["det_sizes"] = list(self.det_sizes)
        return d


def parse_det_sizes(text: str) -> tuple[int, ...]:
    """Read a det_sizes string such as '1280,1920' into a tuple."""
    sizes = tuple(int(part) for part in text.split(",") if part.strip())
    if not sizes:
        raise SettingsError(f"could not read any det_size from {text!r}")
    return sizes
