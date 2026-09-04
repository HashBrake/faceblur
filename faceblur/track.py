"""Temporal reasoning: link detections into tracks, fill gaps, drop flickers.

The first build copied every detection six frames each way and grew it as it
went. That turned every false positive into thirteen frames of expanding mask.
This tracker does the opposite. A face has to be seen at least `min_track`
times, close together, before any pixel is touched. Gaps inside a track are
filled by interpolation, not growth. The mask reaches `tail` frames past the
ends of a track, at the same size.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .detect import Detection, Landmarks, iou


def nms_merge(boxes: Sequence[Sequence[float]], thr: float = 0.35) -> list[list[float]]:
    """Greedy NMS on plain [x, y, w, h, score] rows. Kept for the old tests."""
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
        ov = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= thr]
    return b[keep].tolist()


def propagate(per_frame, window: int, grow: float, nms_thr: float = 0.60):
    """The first build's box spreading. No longer used by the pipeline."""
    n = len(per_frame)
    out = [list(f) for f in per_frame]
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


@dataclass
class Track:
    id: int
    dets: dict[int, Detection] = field(default_factory=dict)  # frame -> detection
    confirmed: int = 1   # detections that passed every check, not continuations

    @property
    def first(self) -> int:
        return min(self.dets)

    @property
    def last(self) -> int:
        return max(self.dets)

    def __len__(self) -> int:
        return len(self.dets)

    def predict(self, frame: int) -> Detection:
        """Where the face should be at `frame`, from the last two detections."""
        frames = sorted(self.dets)
        last = self.dets[frames[-1]]
        if len(frames) < 2:
            return last
        prev = self.dets[frames[-2]]
        span = frames[-1] - frames[-2]
        ahead = frame - frames[-1]
        vx = (last.cx - prev.cx) / span
        vy = (last.cy - prev.cy) / span
        return replace(last, x=last.x + vx * ahead, y=last.y + vy * ahead)


def _lerp_landmarks(a: Optional[Landmarks], b: Optional[Landmarks], t: float):
    if a is None or b is None:
        return a if b is None else b if a is None else None
    return tuple((pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))
                 for pa, pb in zip(a, b))


def _lerp(a: Detection, b: Detection, t: float) -> Detection:
    return Detection(
        a.x + t * (b.x - a.x), a.y + t * (b.y - a.y),
        a.w + t * (b.w - a.w), a.h + t * (b.h - a.h),
        min(a.score, b.score), _lerp_landmarks(a.landmarks, b.landmarks, t),
        "track", a.verified and b.verified,
    )


class Tracker:
    """Offline tracker over a whole video's detections."""

    def __init__(self, min_track: int = 3, max_gap: int = 3, tail: int = 2,
                 iou_thr: float = 0.3, tail_before: Optional[int] = None,
                 tail_grow: float = 0.0):
        self.min_track = min_track
        self.max_gap = max_gap
        self.tail = tail
        self.tail_before = tail if tail_before is None else tail_before
        self.tail_grow = tail_grow
        self.iou_thr = iou_thr

    def link(self, per_frame: Sequence[Sequence[Detection]],
             weak: Optional[Sequence[Sequence[Detection]]] = None) -> list[Track]:
        """Link confirmed detections into tracks.

        `weak` holds, per frame, boxes the detector saw but nothing confirmed.
        A track that already holds `min_track` confirmed detections may continue
        on a weak box that overlaps its predicted position. Weak boxes never
        start a track and never count toward `min_track`.
        """
        tracks: list[Track] = []
        active: list[Track] = []
        next_id = 0
        for i, dets in enumerate(per_frame):
            active = [t for t in active if i - t.last - 1 <= self.max_gap]
            used = self._match(active, i, list(dets), set(), weak_boxes=False)
            if weak is not None and weak[i]:
                rest = [t for ti, t in enumerate(active)
                        if ti not in used and t.confirmed >= self.min_track]
                self._match(rest, i, list(weak[i]), set(), weak_boxes=True)
            for d in dets:
                if not any(t.dets.get(i) is d for t in active):
                    t = Track(next_id, {i: d})
                    next_id += 1
                    tracks.append(t)
                    active.append(t)
        kept = [t for t in tracks if t.confirmed >= self.min_track]
        self._extend_backwards(kept, per_frame, weak, tracks)
        return kept

    def _extend_backwards(self, kept, per_frame, weak, all_tracks) -> None:
        """Walk each confirmed track back in time over boxes nothing else owns.

        A face entering the picture is often seen weakly, or once, before it is
        confirmed twice. Those boxes were never allowed to start a track. Here
        they extend a track that did get confirmed, the mirror of the forward
        continuation, and the frames they cover get masked.
        """
        claimed = {(i, id(d)) for t in kept for i, d in t.dets.items()}
        for t in sorted(kept, key=lambda t: t.first):
            anchor = t.dets[t.first]
            frame = t.first - 1
            gap = 0
            while frame >= 0 and gap <= self.max_gap:
                pool = list(per_frame[frame]) + (list(weak[frame]) if weak is not None else [])
                pool = [d for d in pool if (frame, id(d)) not in claimed]
                best = max(pool, key=lambda d: iou(anchor, d), default=None)
                if best is not None and iou(anchor, best) >= self.iou_thr:
                    t.dets[frame] = replace(best, source="continued")
                    claimed.add((frame, id(best)))
                    anchor = best
                    gap = 0
                else:
                    gap += 1
                frame -= 1

    def _match(self, tracks: list[Track], frame: int, dets: list[Detection],
               used_t: set, weak_boxes: bool) -> set:
        pairs = []
        for ti, t in enumerate(tracks):
            pred = t.predict(frame)
            for di, d in enumerate(dets):
                score = iou(pred, d)
                if score >= self.iou_thr:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)
        used_d = set()
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            d = dets[di]
            if weak_boxes:
                d = replace(d, source="continued")
            else:
                tracks[ti].confirmed += 1
            tracks[ti].dets[frame] = d
            used_t.add(ti)
            used_d.add(di)
        return used_t

    def render(self, tracks: Sequence[Track], n_frames: int) -> list[list[Detection]]:
        """Per frame detections: real, interpolated inside gaps, short tails."""
        out: list[list[Detection]] = [[] for _ in range(n_frames)]
        for t in tracks:
            frames = sorted(t.dets)
            for a, b in zip(frames, frames[1:]):
                out[a].append(t.dets[a])
                da, db = t.dets[a], t.dets[b]
                for k in range(a + 1, b):
                    out[k].append(_lerp(da, db, (k - a) / (b - a)))
            out[frames[-1]].append(t.dets[frames[-1]])
            head, foot = t.dets[frames[0]], t.dets[frames[-1]]
            # Tails follow the motion at each end of the track and grow a little
            # per frame, so a face moving into or out of the picture stays
            # covered while the detector has not seen it yet.
            hv = self._velocity(t, frames[:2])
            fv = self._velocity(t, frames[-2:])
            for k in range(1, self.tail_before + 1):
                if frames[0] - k >= 0:
                    out[frames[0] - k].append(self._tail_box(head, hv, -k))
            for k in range(1, self.tail + 1):
                if frames[-1] + k < n_frames:
                    out[frames[-1] + k].append(self._tail_box(foot, fv, k))
        return out

    @staticmethod
    def _velocity(t: Track, pair) -> tuple[float, float]:
        if len(pair) < 2 or pair[1] == pair[0]:
            return 0.0, 0.0
        a, b = t.dets[pair[0]], t.dets[pair[1]]
        span = pair[1] - pair[0]
        return (b.cx - a.cx) / span, (b.cy - a.cy) / span

    def _tail_box(self, d: Detection, v: tuple[float, float], k: int) -> Detection:
        g = 1.0 + self.tail_grow * abs(k)
        w, h = d.w * g, d.h * g
        cx, cy = d.cx + v[0] * k, d.cy + v[1] * k
        lm = None
        if d.landmarks is not None:
            lm = tuple((px + v[0] * k, py + v[1] * k) for px, py in d.landmarks)
        return Detection(cx - w / 2, cy - h / 2, w, h, d.score, lm, "track", d.verified)

    def run(self, per_frame: Sequence[Sequence[Detection]],
            weak: Optional[Sequence[Sequence[Detection]]] = None
            ) -> tuple[list[list[Detection]], list[Track]]:
        tracks = self.link(per_frame, weak)
        return self.render(tracks, len(per_frame)), tracks
