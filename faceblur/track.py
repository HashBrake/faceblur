"""Box merging and temporal propagation.

A box is [x, y, w, h, score] in source frame coordinates. The math here is the
math the prototype used. Do not change it without a test that shows a reason.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Box = list  # [x, y, w, h, score]


def nms_merge(boxes: Sequence[Sequence[float]], thr: float = 0.35) -> list[list[float]]:
    """Greedy non-maximum suppression. Keeps the highest scoring box of a cluster."""
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    area = b[:, 2] * b[:, 3]
    order = b[:, 4].argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return b[keep].tolist()


def propagate(
    per_frame: Sequence[Sequence[Sequence[float]]],
    window: int,
    grow: float,
    nms_thr: float = 0.60,
) -> list[list[list[float]]]:
    """Spread each detection forward and backward `window` frames.

    Each copy grows by `grow` per frame of distance, so it still covers a face
    that moved. This is the step that turns a detector which fires on 3 frames
    out of 4 into a mask on all 4.
    """
    n = len(per_frame)
    out: list[list[list[float]]] = [list(f) for f in per_frame]
    for i, boxes in enumerate(per_frame):
        for x, y, w, h, s in boxes:
            for d in range(1, window + 1):
                g = 1.0 + grow * d
                nw, nh = w * g, h * g
                nx, ny = x - (nw - w) / 2, y - (nh - h) / 2
                for j in (i - d, i + d):
                    if 0 <= j < n:
                        out[j].append([nx, ny, nw, nh, s])
    return [nms_merge(f, nms_thr) for f in out]
