"""Camera motion between consecutive frames.

A wearable camera pans and shakes. The tracker predicts where a face will be
from its last two sightings; during a fast pan that prediction is off by the
whole camera movement and the link breaks, so the track has to earn its
confirmations again and the face shows for a frame or two. Knowing the global
shift per frame lets the tracker take the camera out of the prediction.

The shift is the peak of the phase correlation between two small greyscale
copies of consecutive frames. One millisecond per frame, no features, no
training. It is a single translation: rotation and zoom are ignored, which is
right for the frame to frame movement of a camera worn on the chest.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Long side of the greyscale copy the shift is measured on.
SMALL = 400


def downscale(frame: np.ndarray) -> np.ndarray:
    """Greyscale float copy with the long side at SMALL pixels."""
    h, w = frame.shape[:2]
    f = SMALL / float(max(h, w))
    small = cv2.resize(frame, (max(8, int(round(w * f))), max(8, int(round(h * f)))),
                       interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.float32)


def estimate_shift(prev: Optional[np.ndarray], cur: np.ndarray,
                   full_long_side: int) -> tuple[float, float]:
    """Camera shift from `prev` to `cur`, in source pixels, as (dx, dy).

    Both inputs come from `downscale`. Positive dx means the picture moved to
    the right, so a static object sits dx pixels further right in `cur`.
    """
    if prev is None or prev.shape != cur.shape:
        return 0.0, 0.0
    (dx, dy), response = cv2.phaseCorrelate(prev, cur)
    if not np.isfinite(dx) or not np.isfinite(dy) or response < 0.02:
        # No clear peak: a cut, a black frame, or pure texture. Assume still.
        return 0.0, 0.0
    f = full_long_side / float(max(cur.shape))
    return float(dx * f), float(dy * f)
