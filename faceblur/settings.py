"""Every knob the pipeline has, with the defaults from the build plan."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

ENGINES = ("yunet", "centerface", "both")
MODES = ("blur", "pixelate", "solid")

# Video extensions the batch walker treats as input.
VIDEO_EXT = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
)


class SettingsError(ValueError):
    """A setting is outside the range the pipeline accepts."""


@dataclass(frozen=True)
class Settings:
    """Defaults come from Section 5 of the build plan.

    det_sizes are absolute long side lengths in pixels, not multipliers. A 4K
    source and a 540p source therefore cost the same to scan.
    """

    engine: str = "yunet"
    conf: float = 0.25
    det_sizes: tuple[int, ...] = (640, 1280)
    stride: int = 1
    persist: int = 6
    grow: float = 0.06
    pad: float = 0.30
    mode: str = "blur"
    strength: int = 28
    crf: int = 20
    preset: str = "medium"
    suffix: str = "_blurred"
    replace_existing: bool = False

    # NMS thresholds from Section 5. Merging detections uses 0.35. Collapsing
    # duplicates after propagation uses 0.6, which is looser on purpose: boxes
    # copied from the same face across frames should survive as one box.
    nms_detect: float = 0.35
    nms_propagate: float = 0.60

    # YuNet's own NMS, applied inside cv2.FaceDetectorYN before this code sees
    # the boxes. The prototype used 0.3.
    nms_yunet: float = 0.30

    def __post_init__(self) -> None:
        object.__setattr__(self, "det_sizes", tuple(int(s) for s in self.det_sizes))
        self.validate()

    def validate(self) -> None:
        if self.engine not in ENGINES:
            raise SettingsError(f"engine must be one of {ENGINES}, got {self.engine!r}")
        if self.mode not in MODES:
            raise SettingsError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 < self.conf <= 1.0:
            raise SettingsError(f"conf must be above 0 and at most 1, got {self.conf}")
        if not self.det_sizes:
            raise SettingsError("det_sizes must name at least one size")
        if any(s < 64 for s in self.det_sizes):
            raise SettingsError(f"every det_size must be at least 64, got {self.det_sizes}")
        if self.stride < 1:
            raise SettingsError(f"stride must be at least 1, got {self.stride}")
        if self.persist < 0:
            raise SettingsError(f"persist must be 0 or more, got {self.persist}")
        if self.grow < 0:
            raise SettingsError(f"grow must be 0 or more, got {self.grow}")
        if self.pad < 0:
            raise SettingsError(f"pad must be 0 or more, got {self.pad}")
        if self.strength < 1:
            raise SettingsError(f"strength must be at least 1, got {self.strength}")
        if not 0 <= self.crf <= 51:
            raise SettingsError(f"crf must be between 0 and 51, got {self.crf}")

    @property
    def window(self) -> int:
        """Propagation reach in frames. Never shorter than the detect stride."""
        return max(self.persist, self.stride)

    def with_changes(self, **changes) -> "Settings":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "conf": self.conf,
            "det_sizes": list(self.det_sizes),
            "stride": self.stride,
            "persist": self.persist,
            "grow": self.grow,
            "pad": self.pad,
            "mode": self.mode,
            "strength": self.strength,
            "crf": self.crf,
            "preset": self.preset,
            "nms_detect": self.nms_detect,
            "nms_propagate": self.nms_propagate,
            "nms_yunet": self.nms_yunet,
        }


def parse_det_sizes(text: str) -> tuple[int, ...]:
    """Read a det_sizes string such as '640,1280' into a tuple."""
    sizes = tuple(int(part) for part in text.split(",") if part.strip())
    if not sizes:
        raise SettingsError(f"could not read any det_size from {text!r}")
    return sizes
