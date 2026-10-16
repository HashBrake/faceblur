"""Synchronised reading of the left/right mp4s of one part.

Seeking by ``CAP_PROP_POS_FRAMES`` is wrong on this footage: OpenCV turns the
frame index into a time with the container's nominal rate (30.51 fps) while
frames really come at ~30 fps, so frame 18000 of AZER76400EV part 1 lands
~10 s early (on frame ~17700). Seeking by time is consistent instead: the
reader seeks to 1 s before the first wanted frame's container timestamp
(``CAP_PROP_POS_MSEC``), identifies each grabbed frame by its own
``CAP_PROP_POS_MSEC`` (identical to the ffmpeg packet pts used by
``hands.timing.container_timestamps``) and grabs forward to the wanted frame.
After that frames are read sequentially (no dropped or duplicated container
frames, see ``hands.timing``), with ``grab()`` skipping frames not in the pairs.
"""
from __future__ import annotations

import cv2
import numpy as np

SEEK_LEAD_MS = 1000.0
MATCH_TOL_US = 5000.0


class PartReader:
    def __init__(self, left_mp4, right_mp4, image_size, container_ts):
        """``image_size`` = calibration (w, h); ``container_ts`` = (left, right) frame times in us."""
        self.caps = (cv2.VideoCapture(str(left_mp4)), cv2.VideoCapture(str(right_mp4)))
        self.image_size = tuple(image_size)
        self.ts = tuple(np.asarray(t, np.float64) for t in container_ts)
        self.pos = [-1, -1]  # container index of the last grabbed frame per side

    def _seek(self, side: int, target: int) -> None:
        cap, ts = self.caps[side], self.ts[side]
        cap.set(cv2.CAP_PROP_POS_MSEC, max(ts[target] / 1000.0 - SEEK_LEAD_MS, 0.0))
        while True:
            if not cap.grab():
                raise ValueError(f"video ended before frame {target}")
            us = cap.get(cv2.CAP_PROP_POS_MSEC) * 1000.0
            i = int(np.argmin(np.abs(ts - us)))
            if abs(ts[i] - us) > MATCH_TOL_US:
                raise ValueError(f"decoded frame at {us / 1000:.1f} ms matches no container timestamp")
            if i >= target:
                break
        if i != target:
            raise ValueError(f"seek overshot: wanted frame {target}, got {i}")
        self.pos[side] = target

    def _read(self, side: int, target: int) -> np.ndarray:
        cap = self.caps[side]
        if self.pos[side] < 0:
            self._seek(side, target)
        while self.pos[side] < target:
            if not cap.grab():
                raise ValueError(f"video ended before frame {target}")
            self.pos[side] += 1
        ok, img = cap.retrieve()
        if not ok:
            raise ValueError(f"cannot decode frame {target}")
        h, w = img.shape[:2]
        if (w, h) != self.image_size:
            raise ValueError(f"frame size {w}x{h} differs from the calibration's "
                             f"{self.image_size[0]}x{self.image_size[1]}")
        return img

    def frames(self, pairs):
        """Yield (pair, left_bgr, right_bgr); pairs strictly increasing in both frame indices."""
        for pair in pairs:
            yield pair, self._read(0, pair.frame_left), self._read(1, pair.frame_right)

    def close(self) -> None:
        for cap in self.caps:
            cap.release()
