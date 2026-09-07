"""Temporal reasoning: link detections into tracks, fill gaps, drop flickers.

The first build copied every detection six frames each way and grew it as it
went. That turned every false positive into thirteen frames of expanding mask.
This tracker does the opposite. A face has to be seen at least `min_track`
times, close together, before any pixel is touched. Gaps inside a track are
filled by interpolation, not growth. The mask reaches `tail` frames past the
ends of a track, at the same size.

The camera moves. Given the per frame camera shift (see motion.py) the tracker
predicts where a face will be from its own motion plus the camera's, so a pan
does not break the track, and it allows a longer gap while the camera moves
fast, which is when the detectors miss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .detect import Detection, Landmarks, iou

Shifts = Sequence[tuple[float, float]]     # per frame: camera shift from the frame before


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
    last_confirmed: int = -1   # frame of the latest one; set by the tracker
    best_conf: float = 0.0     # the strongest confirmation the track holds
    # Boxes the detectors saw near the face inside a gap, masked as well as
    # the interpolated box: a smeared face is wherever the evidence is.
    extras: dict[int, list] = field(default_factory=dict)

    @property
    def first(self) -> int:
        return min(self.dets)

    @property
    def last(self) -> int:
        return max(self.dets)

    def __len__(self) -> int:
        return len(self.dets)

    def predict(self, frame: int) -> Detection:
        """Where the face should be at `frame`, from the last two detections,
        with no knowledge of the camera. The tracker has a camera aware one."""
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


def _inflate(d: Detection, px: float) -> Detection:
    """The same box, `px` wider and taller about its centre."""
    return Detection(d.x - px / 2, d.y - px / 2, d.w + px, d.h + px, d.score, d.landmarks,
                     d.source, d.verified)


def _moved(d: Detection, dx: float, dy: float, grow: float = 1.0) -> Detection:
    """The same box shifted by (dx, dy) and scaled about its centre."""
    w, h = d.w * grow, d.h * grow
    cx, cy = d.cx + dx, d.cy + dy
    lm = None
    if d.landmarks is not None:
        lm = tuple((px + dx, py + dy) for px, py in d.landmarks)
    return Detection(cx - w / 2, cy - h / 2, w, h, d.score, lm, "track", d.verified)


class Tracker:
    """Offline tracker over a whole video's detections."""

    def __init__(self, min_track: int = 3, max_gap: int = 3, tail: int = 2,
                 iou_thr: float = 0.3, tail_before: Optional[int] = None,
                 tail_grow: float = 0.0, link_dist: float = 0.0,
                 fast_shift: float = 0.0, max_gap_fast: Optional[int] = None,
                 tail_before_max: Optional[int] = None, tail_trend: bool = True,
                 conf_weak: float = 0.0, conf_weak_long: float = 0.0,
                 established_after: int = 3, tail_long: Optional[int] = None,
                 stitch_gap: int = 0, blur_shift: float = 0.0,
                 continue_after: Optional[int] = None, weak_run: int = 0,
                 stitch_slack: float = 0.0, gap_grow: float = 0.0,
                 sure_conf: float = 0.0):
        self.min_track = min_track
        self.max_gap = max_gap
        self.tail = tail
        self.tail_before = tail if tail_before is None else tail_before
        self.tail_grow = tail_grow
        self.iou_thr = iou_thr
        self.link_dist = link_dist
        self.fast_shift = fast_shift
        self.max_gap_fast = max_gap if max_gap_fast is None else max(max_gap, max_gap_fast)
        self.tail_before_max = max(self.tail_before, tail_before_max or 0)
        self.tail_trend = tail_trend
        # Boxes between conf_weak_long and conf_weak continue only a track that
        # holds established_after confirmed detections.
        self.conf_weak = conf_weak
        self.conf_weak_long = min(conf_weak_long, conf_weak) if conf_weak_long else conf_weak
        self.established_after = max(1, established_after)
        self.tail_long = tail if tail_long is None else max(tail, tail_long)
        self.stitch_gap = max(0, stitch_gap)
        self.stitch_slack = max(0.0, stitch_slack)
        self.gap_grow = max(0.0, gap_grow)
        self.blur_shift = blur_shift
        # Continuation needs this many confirmed detections and stops weak_run
        # frames after the last confirmed one (0: no limit).
        self.continue_after = min_track if continue_after is None else max(1, continue_after)
        self.weak_run = max(0, weak_run)
        # A track is sure once a confirmation scored this much; only sure
        # tracks continue, stitch, fill gaps and get the long tails.
        self.sure_conf = sure_conf
        self._cum = None        # cumulative camera shift per frame, or None
        self._mag = None        # camera shift magnitude per frame

    # ------------------------------------------------------------ camera

    def _set_shifts(self, shifts: Optional[Shifts], n: int) -> None:
        if shifts is None or not any(dx or dy for dx, dy in shifts):
            self._cum = None
            self._mag = None
            return
        arr = np.zeros((n + 1, 2), np.float64)
        for i, (dx, dy) in enumerate(shifts[:n]):
            arr[i + 1] = arr[i] + (dx, dy)
        self._cum = arr[1:]
        self._mag = np.hypot(*np.asarray(list(shifts[:n]) + [(0.0, 0.0)] * (n - len(shifts[:n])),
                                        dtype=np.float64).T)

    def _camera(self, a: int, b: int) -> tuple[float, float]:
        """How far the picture moved between frames a and b, in pixels."""
        if self._cum is None:
            return 0.0, 0.0
        n = len(self._cum)
        a, b = min(max(a, 0), n - 1), min(max(b, 0), n - 1)
        d = self._cum[b] - self._cum[a]
        return float(d[0]), float(d[1])

    def _fast(self, a: int, b: int) -> bool:
        """Did the camera move faster than fast_shift, on average, from a to b?"""
        if self.fast_shift <= 0 or self._cum is None or b <= a:
            return False
        dx, dy = self._camera(a, b)
        return math.hypot(dx, dy) / (b - a) >= self.fast_shift

    def _gap_ok(self, last: int, frame: int) -> bool:
        gap = abs(frame - last) - 1
        if gap <= self.max_gap:
            return True
        return gap <= self.max_gap_fast and self._fast(min(last, frame), max(last, frame))

    def _own_velocity(self, t: Track, pair) -> tuple[float, float]:
        """The face's motion per frame between two of its detections, with
        the camera's movement taken out."""
        if len(pair) < 2 or pair[1] == pair[0]:
            return 0.0, 0.0
        a, b = t.dets[pair[0]], t.dets[pair[1]]
        span = pair[1] - pair[0]
        cx, cy = self._camera(pair[0], pair[1])
        return (b.cx - a.cx - cx) / span, (b.cy - a.cy - cy) / span

    def _predict(self, t: Track, frame: int) -> Detection:
        """Where the face should be at `frame`: its own motion plus the camera's."""
        frames = sorted(t.dets)
        last = t.dets[frames[-1]]
        cx, cy = self._camera(frames[-1], frame)
        if len(frames) < 2:
            return _moved(last, cx, cy)
        vx, vy = self._own_velocity(t, frames[-2:])
        ahead = frame - frames[-1]
        return _moved(last, vx * ahead + cx, vy * ahead + cy)

    # ------------------------------------------------------------ linking

    def _affinity(self, pred: Detection, d: Detection) -> float:
        """How well `d` fits the prediction. Overlap first; failing that, a
        centre close to the prediction and a similar size. 0 means no."""
        v = iou(pred, d)
        if v >= self.iou_thr:
            return v
        if self.link_dist > 0:
            size = max(pred.long_side, d.long_side)
            if size > 0 and min(pred.long_side, d.long_side) / size >= 0.5:
                gate = self.link_dist * size
                dist = math.hypot(pred.cx - d.cx, pred.cy - d.cy)
                if dist <= gate:
                    return max(1e-6, self.iou_thr * (1.0 - dist / gate))
        return 0.0

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
            active = [t for t in active if self._gap_ok(t.last, i)]
            used = self._match(active, i, list(dets), set(), weak_boxes=False)
            if weak is not None and weak[i]:
                rest = [t for ti, t in enumerate(active)
                        if ti not in used and t.confirmed >= self.continue_after
                        and self._may_continue(t, i)]
                # Young tracks see only the stronger weak boxes; an
                # established track may take a box down to conf_weak_long.
                strong_weak = [d for d in weak[i] if d.score >= self.conf_weak]
                low_weak = [d for d in weak[i] if self.conf_weak_long <= d.score < self.conf_weak]
                taken = self._match(rest, i, strong_weak, set(), weak_boxes=True)
                if low_weak:
                    long_lived = [t for ti, t in enumerate(rest)
                                  if ti not in taken and t.confirmed >= self.established_after]
                    self._match(long_lived, i, low_weak, set(), weak_boxes=True)
            for d in dets:
                if not any(t.dets.get(i) is d for t in active):
                    t = Track(next_id, {i: d}, last_confirmed=i, best_conf=d.confidence)
                    next_id += 1
                    tracks.append(t)
                    active.append(t)
        tracks = self._stitch(tracks)
        kept = [t for t in tracks if t.confirmed >= self.min_track]
        self._fill_gaps(kept, per_frame, weak)
        self._extend_backwards(kept, per_frame, weak, tracks)
        return kept

    def _fill_gaps(self, kept, per_frame, weak) -> None:
        """Walk every gap inside a track over the boxes the detectors did see.

        Between two sightings the face is somewhere near the straight line
        between them, but during a pan it travels an arc and a smeared face
        is a wider box than a sharp one. YuNet usually still fires on the
        smear at 0.3 or so. Walking the gap from both ends, each frame takes
        the nearest box to where the walk is, within the stitching slack, so
        the mask follows the real path and the real size. Only an
        established track does this, on boxes down to conf_weak_long.
        """
        if weak is None:
            return
        claimed = {(i, id(d)) for t in kept for i, d in t.dets.items()}
        for t in kept:
            if t.confirmed < self.established_after or not self._sure(t):
                continue
            frames = sorted(t.dets)
            for a, b in zip(frames, frames[1:]):
                if b - a < 2:
                    continue
                da, db = t.dets[a], t.dets[b]
                size = max(da.long_side, db.long_side)
                # Forward from a, then backward from b, each half of the gap.
                mid = (a + b) // 2
                for start, stop, step, anchor in ((a, mid, 1, da), (b, mid, -1, db)):
                    cur, cur_k = anchor, start
                    for k in range(start + step, stop + (1 if step > 0 else 0), step):
                        pool = [d for d in list(per_frame[k]) + list(weak[k])
                                if (k, id(d)) not in claimed and d.score >= self.conf_weak_long]
                        if not pool:
                            continue
                        # The slack accrues over every frame since the last box
                        # the walk stood on, and the camera's move with it.
                        cx, cy = self._camera(cur_k, k)
                        gate = 3.0 * size + self.stitch_slack * abs(k - cur_k)
                        best, best_d = None, gate
                        for d in pool:
                            # A smeared face is a box up to four times the
                            # sharp one, but only while the camera moves fast
                            # enough to smear it; otherwise sizes must agree.
                            fast = self._mag is not None and k < len(self._mag)                                 and self.blur_shift > 0 and self._mag[k] >= self.blur_shift
                            ratio = d.long_side / max(1.0, cur.long_side)
                            if not 0.5 <= ratio <= (4.0 if fast else 1.5):
                                continue
                            dist = math.hypot(d.cx - (cur.cx + cx), d.cy - (cur.cy + cy))
                            if dist < gate:
                                # Every box near the path is masked; the
                                # nearest one is where the walk goes on from.
                                t.extras.setdefault(k, []).append(replace(d, source="continued"))
                                claimed.add((k, id(d)))
                            if dist < best_d:
                                best, best_d = d, dist
                        if best is None:
                            continue
                        cur, cur_k = best, k

    def _sure(self, t: Track) -> bool:
        return self.sure_conf <= 0 or t.best_conf >= self.sure_conf

    def _may_continue(self, t: Track, frame: int) -> bool:
        if not self._sure(t):
            return False
        if self.weak_run <= 0:
            return True
        last = t.last_confirmed if t.last_confirmed >= 0 else t.first
        return abs(frame - last) <= self.weak_run

    def _stitch(self, tracks: list[Track]) -> list[Track]:
        """Join tracks of one face that the detectors lost for a while.

        A track that ends, and another that starts up to stitch_gap frames
        later where the first one's motion and the camera's put the face, are
        one track. The frames between are interpolated by `render`. With
        hindsight this is safe: the face came back where it was expected.
        Only detections that passed every check start a track, so the later
        piece is a face too.
        """
        if self.stitch_gap <= 0 or len(tracks) < 2:
            return tracks
        tracks = sorted(tracks, key=lambda t: t.first)
        alive = list(tracks)
        merged = True
        while merged:
            merged = False
            alive.sort(key=lambda t: t.last)
            for a in alive:
                if a.confirmed < self.continue_after or not self._sure(a):
                    continue
                best, best_score = None, 0.0
                for b in alive:
                    if b is a or b.first <= a.last or b.first - a.last - 1 > self.stitch_gap:
                        continue
                    pred = self._predict(a, b.first)
                    first = b.dets[b.first]
                    size = max(pred.long_side, first.long_side)
                    if size <= 0 or min(pred.long_side, first.long_side) / size < 0.5:
                        continue
                    dist = math.hypot(pred.cx - first.cx, pred.cy - first.cy)
                    gate = size + self.stitch_slack * (b.first - a.last)
                    if dist > gate:
                        continue
                    score = 1.0 - dist / gate
                    if score > best_score:
                        best, best_score = b, score
                if best is not None:
                    a.dets.update(best.dets)
                    a.confirmed += best.confirmed
                    a.last_confirmed = max(a.last_confirmed, best.last_confirmed)
                    a.best_conf = max(a.best_conf, best.best_conf)
                    alive.remove(best)
                    merged = True
                    break
        return alive

    def _extend_backwards(self, kept, per_frame, weak, all_tracks) -> None:
        """Walk each confirmed track back in time over boxes nothing else owns.

        A face entering the picture is often seen weakly, or once, before it is
        confirmed twice. Those boxes were never allowed to start a track. Here
        they extend a track that did get confirmed, the mirror of the forward
        continuation, and the frames they cover get masked.
        """
        claimed = {(i, id(d)) for t in kept for i, d in t.dets.items()}
        for t in sorted(kept, key=lambda t: t.first):
            if t.confirmed < self.established_after or not self._sure(t):
                continue             # too little behind it to reach back from
            anchor_frame = t.first
            anchor = t.dets[anchor_frame]
            frame = t.first - 1
            floor = self.conf_weak_long
            while frame >= 0 and self._gap_ok(anchor_frame, frame):
                pool = list(per_frame[frame]) + (
                    [d for d in weak[frame] if d.score >= floor] if weak is not None else [])
                pool = [d for d in pool if (frame, id(d)) not in claimed]
                cx, cy = self._camera(anchor_frame, frame)
                pred = _moved(anchor, cx, cy)
                best = max(pool, key=lambda d: self._affinity(pred, d), default=None)
                if best is not None and self._affinity(pred, best) > 0:
                    t.dets[frame] = replace(best, source="continued")
                    claimed.add((frame, id(best)))
                    anchor, anchor_frame = best, frame
                frame -= 1

    def _match(self, tracks: list[Track], frame: int, dets: list[Detection],
               used_t: set, weak_boxes: bool) -> set:
        pairs = []
        for ti, t in enumerate(tracks):
            pred = self._predict(t, frame)
            for di, d in enumerate(dets):
                score = self._affinity(pred, d)
                if score > 0:
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
                tracks[ti].last_confirmed = max(tracks[ti].last_confirmed, frame)
                tracks[ti].best_conf = max(tracks[ti].best_conf, d.confidence)
            tracks[ti].dets[frame] = d
            used_t.add(ti)
            used_d.add(di)
        return used_t

    # ------------------------------------------------------------ rendering

    def render(self, tracks: Sequence[Track], n_frames: int,
               shape: Optional[tuple[int, int]] = None) -> list[list[Detection]]:
        """Per frame detections: real, interpolated inside gaps, short tails.

        `shape` (height, width) lets the entry tail of a moving face reach
        further back, until the extrapolated box has left the picture.
        """
        out: list[list[Detection]] = [[] for _ in range(n_frames)]
        for t in tracks:
            frames = sorted(t.dets)
            for a, b in zip(frames, frames[1:]):
                out[a].append(t.dets[a])
                da, db = t.dets[a], t.dets[b]
                for k in range(a + 1, b):
                    box = _lerp(da, db, (k - a) / (b - a))
                    if self.gap_grow > 0:
                        # Least certain in the middle of the gap.
                        away = min(k - a, b - k)
                        box = _moved(box, 0.0, 0.0, 1.0 + self.gap_grow * away)
                    out[k].append(box)
            out[frames[-1]].append(t.dets[frames[-1]])
            for k, boxes in t.extras.items():
                if 0 <= k < n_frames:
                    out[k].extend(boxes)
            head, foot = t.dets[frames[0]], t.dets[frames[-1]]
            # Tails follow the motion at each end of the track and grow a little
            # per frame, so a face moving into or out of the picture stays
            # covered while the detector has not seen it yet. The size follows
            # its trend too: a face that was shrinking was larger before.
            hv = self._own_velocity(t, frames[:2])
            fv = self._own_velocity(t, frames[-2:])
            head_grow = self.tail_grow + self._size_trend(t, frames[:2], backwards=True)
            foot_grow = self.tail_grow + self._size_trend(t, frames[-2:], backwards=False)
            reach = self._entry_reach(t, frames, hv, shape)                 if t.confirmed >= self.established_after and self._sure(t) else 0
            for k in range(1, reach + 1):
                if frames[0] - k >= 0:
                    out[frames[0] - k].append(self._tail_box(head, frames[0], hv, -k, head_grow))
            # A track that is not established gets no tail at all: its
            # confirmed frames and the gaps between them, nothing more.
            established = t.confirmed >= self.established_after and self._sure(t)
            tail = (self.tail_long if established else 0) \
                if self.established_after > self.min_track else self.tail
            for k in range(1, tail + 1):
                if frames[-1] + k < n_frames:
                    out[frames[-1] + k].append(self._tail_box(foot, frames[-1], fv, k, foot_grow))
        if self.blur_shift > 0 and self._mag is not None:
            # A fast camera move smears a face by about the shift. Every mask
            # in such a frame grows by that much, so the smear stays covered.
            for i in range(min(n_frames, len(self._mag))):
                m = float(self._mag[i])
                if m > self.blur_shift and out[i]:
                    out[i] = [_inflate(d, m) for d in out[i]]
        return out

    def _size_trend(self, t: Track, pair, backwards: bool) -> float:
        """Growth per frame to apply along a tail, from how the box size changed
        between two detections. Only growth, never shrinking, capped."""
        if not self.tail_trend or len(pair) < 2 or pair[1] == pair[0]:
            return 0.0
        a, b = t.dets[pair[0]], t.dets[pair[1]]
        if a.long_side <= 0 or b.long_side <= 0:
            return 0.0
        rate = (b.long_side - a.long_side) / a.long_side / (pair[1] - pair[0])
        rate = -rate if backwards else rate
        return min(0.2, max(0.0, rate))

    def _entry_reach(self, t: Track, frames, hv, shape) -> int:
        """Frames the entry tail reaches back. tail_before for a face that
        stands still; up to tail_before_max for one that moves, until the
        extrapolated box has left the picture."""
        reach = self.tail_before
        if shape is None or self.tail_before_max <= self.tail_before:
            return reach
        head = t.dets[frames[0]]
        speed = math.hypot(hv[0], hv[1])
        if speed < 2.0 and self._cum is None:
            return reach
        H, W = shape
        for k in range(self.tail_before + 1, self.tail_before_max + 1):
            box = self._tail_box(head, frames[0], hv, -k, self.tail_grow)
            if box.cx < 0 or box.cy < 0 or box.cx > W or box.cy > H:
                break
            reach = k
        return reach

    def _tail_box(self, d: Detection, at: int, v: tuple[float, float], k: int,
                  grow: float) -> Detection:
        """The box `k` frames from `at` (k negative for earlier), following the
        face's own motion and the camera's, enlarged by `grow` per frame."""
        cx, cy = self._camera(at, at + k)
        return _moved(d, v[0] * k + cx, v[1] * k + cy, 1.0 + grow * abs(k))

    @staticmethod
    def _velocity(t: Track, pair) -> tuple[float, float]:
        """Plain image velocity between two detections. Kept for callers that
        do not know the camera."""
        if len(pair) < 2 or pair[1] == pair[0]:
            return 0.0, 0.0
        a, b = t.dets[pair[0]], t.dets[pair[1]]
        span = pair[1] - pair[0]
        return (b.cx - a.cx) / span, (b.cy - a.cy) / span

    def run(self, per_frame: Sequence[Sequence[Detection]],
            weak: Optional[Sequence[Sequence[Detection]]] = None,
            shifts: Optional[Shifts] = None,
            shape: Optional[tuple[int, int]] = None,
            ) -> tuple[list[list[Detection]], list[Track]]:
        self._set_shifts(shifts, len(per_frame))
        tracks = self.link(per_frame, weak)
        return self.render(tracks, len(per_frame), shape), tracks
