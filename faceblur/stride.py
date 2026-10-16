"""When face detection may skip a frame: the adaptive stride.

A pure policy, no models and no I/O, so that `batch.detect_job` and the
evaluation harness's replay (`eval.measure.tracked`) make the same choice on
the same sequence. Detection skips a frame only when the last few detected
frames saw nothing at all and the camera is nearly still: a face cannot have
been missed by a detector that did not look, and where nothing was seen and
nothing moved, the frame after is very likely empty too. On the sample, 438
of 984 frames had no detection, and each frame's face detection costs about
71 ms of GPU-bound work at 4 workers (docs/bench/profile_2026-10-16.md).

One policy per chunk: chunks are detected by different workers, so the state
never crosses a chunk boundary and the first frame of every chunk is detected.
Frames off the plain `stride` never reach the policy.
"""
from __future__ import annotations

import math
from typing import Optional

from .settings import Settings


def skip_next(quiet: int, skipped: int, shift_px: Optional[float],
              settings: Settings) -> bool:
    """May the frame about to be detected be skipped?

    `quiet` counts the consecutive detected frames that had no strong and no
    weak candidate, `skipped` the frames skipped since the last detected one,
    `shift_px` the camera shift into this frame (None when not measured,
    which counts as moving).
    """
    return (settings.adaptive_stride
            and quiet >= settings.adaptive_stride_quiet
            and skipped < settings.adaptive_stride_max - 1
            and shift_px is not None
            and shift_px < settings.adaptive_stride_shift)


class AdaptiveStride:
    """The policy's state over one chunk, frame by frame in order.

    For each frame on the plain stride call `skip(shift_px)`; when it says
    False, detect and pass what was found to `saw(strong, weak)`. A skipped
    frame is neither quiet nor busy evidence: it leaves `quiet` alone.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.quiet = 0          # 0 at the start, so a chunk's first frame is detected
        self.skipped = 0
        self.count = 0          # frames this policy skipped

    def skip(self, shift_px: Optional[float]) -> bool:
        if skip_next(self.quiet, self.skipped, shift_px, self.settings):
            self.skipped += 1
            self.count += 1
            return True
        return False

    def saw(self, strong, weak) -> None:
        self.skipped = 0
        self.quiet = 0 if (strong or weak) else self.quiet + 1


def shift_px(shift: Optional[tuple[float, float]]) -> Optional[float]:
    """The magnitude of a (dx, dy) camera shift, None when there is none."""
    return None if shift is None else math.hypot(*shift)
