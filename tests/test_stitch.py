"""`Tracker._stitch` must join exactly what the plain version joined.

The plain version rescans every track after every merge and took the
min and max of each track's detections at every comparison; on footage with
thousands of short tracks it ran for over 12 hours (2026-10-15, slices 2/8
and 5/8). The fast version must give the same tracks, merged in the same
order, with the same detections.
"""
import math
import random
import time

from faceblur.detect import Detection
from faceblur.track import Track, Tracker


def plain_stitch(self, tracks):
    """The version before 2026-10-15, kept as the reference."""
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


def pieces(seed, faces=12, frames=3000, piece=(3, 25), gap=(1, 40)):
    """Short tracks cut from a few moving faces, with gaps between them,
    plus some unconfirmed and some unsure ones."""
    rnd = random.Random(seed)
    tracks, next_id = [], 0
    for _ in range(faces):
        x, y = rnd.uniform(100, 1500), rnd.uniform(100, 1200)
        vx, vy = rnd.uniform(-3, 3), rnd.uniform(-3, 3)
        w = rnd.uniform(20, 120)
        f = rnd.randrange(0, 200)
        while f < frames:
            n = rnd.randint(*piece)
            dets = {}
            for i in range(f, min(frames, f + n)):
                dets[i] = Detection(x + vx * i + rnd.gauss(0, 2), y + vy * i + rnd.gauss(0, 2),
                                    w * rnd.uniform(0.9, 1.1), w * rnd.uniform(0.9, 1.1), 0.9,
                                    verified=True, confidence=rnd.uniform(0.5, 0.99))
            if dets:
                conf = rnd.choice([1, 2, len(dets)])
                tracks.append(Track(next_id, dets, confirmed=conf, last_confirmed=max(dets),
                                    best_conf=rnd.uniform(0.3, 0.99)))
                next_id += 1
            f += n + rnd.randint(*gap)
    return tracks


def tracker(**kw):
    t = Tracker(min_track=2, stitch_gap=kw.get("stitch_gap", 30),
                stitch_slack=kw.get("stitch_slack", 4.0), sure_conf=kw.get("sure_conf", 0.5))
    if kw.get("camera"):
        rnd = random.Random(7)
        t._set_shifts([(rnd.uniform(-5, 5), rnd.uniform(-5, 5)) for _ in range(3000)], 3000)
    return t


def summary(tracks):
    return [(t.id, sorted(t.dets), t.confirmed, t.last_confirmed, t.best_conf) for t in tracks]


def test_same_merges_as_the_plain_version():
    for seed in range(12):
        for kw in ({}, {"camera": True}, {"stitch_gap": 5, "stitch_slack": 0.5},
                   {"sure_conf": 0.0, "stitch_gap": 60}):
            t = tracker(**kw)
            a = plain_stitch(t, pieces(seed, faces=6, frames=900))
            b = t._stitch(pieces(seed, faces=6, frames=900))
            assert summary(a) == summary(b), (seed, kw)


def test_thousands_of_tracks_in_seconds():
    tracks = pieces(1, faces=60, frames=6000)
    assert len(tracks) > 3000
    start = time.time()
    tracker()._stitch(tracks)
    assert time.time() - start < 60


def twins(seed):
    """Every track twice, so equal scores decide which one is taken."""
    out = []
    for t in pieces(seed, faces=4, frames=600):
        out.append(t)
        out.append(Track(t.id + 10_000, dict(t.dets), confirmed=t.confirmed,
                         last_confirmed=t.last_confirmed, best_conf=t.best_conf))
    return out


def test_equal_scores_go_where_the_plain_version_sent_them():
    for seed in range(20):
        for kw in ({}, {"camera": True}, {"sure_conf": 0.0, "stitch_gap": 60}):
            t = tracker(**kw)
            assert summary(plain_stitch(t, twins(seed))) == summary(t._stitch(twins(seed))), (seed, kw)


def test_more_seeds_and_settings():
    for seed in range(12, 60):
        kw = [{}, {"camera": True}, {"stitch_gap": 5, "stitch_slack": 0.5},
              {"sure_conf": 0.0, "stitch_gap": 60}, {"stitch_gap": 200, "sure_conf": 0.0}][seed % 5]
        t = tracker(**kw)
        a = plain_stitch(t, pieces(seed, faces=8, frames=1500, gap=(1, 80)))
        b = t._stitch(pieces(seed, faces=8, frames=1500, gap=(1, 80)))
        assert summary(a) == summary(b), (seed, kw)


def test_tens_of_thousands_of_tracks_in_seconds():
    # 2026-10-16: the version that re-sorted every track after every merge
    # was still stitching one video of FF_000137 after 1.7 hours.
    tracks = pieces(2, faces=60, frames=40000, gap=(1, 20))
    assert len(tracks) > 30000
    start = time.time()
    tracker(sure_conf=0.0)._stitch(tracks)
    assert time.time() - start < 30
