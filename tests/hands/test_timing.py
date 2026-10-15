"""Frame timing: container frame -> csv row, stereo pairing by timestamp."""
from __future__ import annotations

import numpy as np
import pytest

from hands.timing import (
    container_timestamps,
    find_part_files,
    frame_to_row,
    load_pts,
    pair_stereo,
    stereo_frames,
)

PERIOD = 1e6 / 30


def _csv(n, t0=347_470_640_146, seed=0):
    rng = np.random.default_rng(seed)
    return (t0 + np.arange(n) * PERIOD + rng.uniform(-500, 500, n)).astype(np.int64)


def _container(pts, skew_ppm=-169.0, offset_us=0.0):
    return (pts - pts[0]) * (1 + skew_ppm * 1e-6) + offset_us


def _write_csv(path, pts):
    path.write_text("timestamp_us\n" + "\n".join(map(str, pts)) + "\n", encoding="utf-8")


def test_frame_to_row_identity_with_clock_skew():
    pts = _csv(20_000)  # ~11 min: 169 ppm skew is 113 ms by the end
    rows = frame_to_row(_container(pts), pts)
    assert np.array_equal(rows, np.arange(len(pts)))


def test_frame_to_row_duplicate_at_index_1_maps_to_minus_one():
    pts = _csv(3000)
    c = _container(pts)
    c = np.concatenate([c[:1], [c[0] + PERIOD / 2], c[1:]])  # 16.7 ms step
    rows = frame_to_row(c, pts)
    assert rows[0] == 0 and rows[1] == -1
    assert np.array_equal(rows[2:], np.arange(1, len(pts)))


def test_frame_to_row_dropped_frame_skips_the_row():
    pts = _csv(3000)
    c = np.delete(_container(pts), 1500)
    rows = frame_to_row(c, pts)
    assert np.array_equal(rows, np.delete(np.arange(len(pts)), 1500))


def test_frame_to_row_half_frame_offset_first_frame_clamped():
    # Real layout: frame 0 at 0, the rest shifted by -16.9 ms.
    pts = _csv(3000)
    c = _container(pts, offset_us=-16_900.0)
    c[0] = 0.0
    rows = frame_to_row(c, pts)
    assert rows[0] == -1
    assert np.array_equal(rows[1:], np.arange(1, len(pts)))


def test_pair_stereo_offset_and_missing():
    left = _csv(500)
    right = np.concatenate([left[:10] - 10 * 33_333, left]) + np.random.default_rng(1).integers(-5, 6, 510)
    right = np.delete(right, 10 + 200)  # right misses the twin of left row 200
    pairs = pair_stereo(left, right)
    assert len(pairs) == 499
    assert 200 not in pairs[:, 0]
    assert np.all(np.abs(left[pairs[:, 0]] - right[pairs[:, 1]]) <= 2000)
    assert np.all(np.diff(pairs[:, 0]) > 0) and np.all(np.diff(pairs[:, 1]) > 0)
    assert pairs[0].tolist() == [0, 10]


def test_stereo_frames_window(tmp_path):
    stem = "Orbbec_Ego_X_19700101_000000_camera"
    left = _csv(900)
    right = np.concatenate([left[:10] - 10 * 33_333, left[:-10]])
    _write_csv(tmp_path / f"{stem}_left_part0001_pts.csv", left)
    _write_csv(tmp_path / f"{stem}_right_part0001_pts.csv", right)
    (tmp_path / f"{stem}_left_part0001.mp4").touch()
    (tmp_path / f"{stem}_right_part0001.mp4").touch()
    cl = _container(left)
    cr = np.concatenate([_container(right)[:5], [_container(right)[4] + PERIOD / 2], _container(right)[5:]])

    allp = stereo_frames(tmp_path, 1, container_ts=(cl, cr))
    assert len(allp) == 890
    assert (allp[0].row_left, allp[0].row_right) == (0, 10)
    assert (allp[0].frame_left, allp[0].frame_right) == (0, 11)  # one extra right frame before
    assert allp[0].timestamp_us == left[0]

    win = stereo_frames(tmp_path, 1, start_s=10.0, duration_s=5.0, container_ts=(cl, cr))
    ts = np.array([p.timestamp_us for p in win]) - left[0]
    assert ts.min() >= 10e6 and ts.max() < 15e6
    assert len(win) == 150
    assert win[0].row_left == 300 and win[0].frame_left == 300


def test_find_part_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_part_files(tmp_path, 1)


@pytest.mark.real_footage
def test_real_part1_first_3000_frames(recording_ev):
    left_mp4, right_mp4, left_csv, right_csv = find_part_files(recording_ev, 1)
    cl = container_timestamps(left_mp4, max_frames=3000)
    cr = container_timestamps(right_mp4, max_frames=3000)
    assert len(cl) == len(cr) == 3000
    pairs = stereo_frames(recording_ev, 1, container_ts=(cl, cr))
    pts_l, pts_r = load_pts(left_csv), load_pts(right_csv)

    rl = np.array([p.row_left for p in pairs])
    rr = np.array([p.row_right for p in pairs])
    fl = np.array([p.frame_left for p in pairs])
    fr = np.array([p.frame_right for p in pairs])
    assert np.all(np.abs(pts_l[rl] - pts_r[rr]) <= 2000)
    assert np.all(np.diff(fl) > 0) and np.all(np.diff(fr) > 0)
    assert np.all(np.abs((rr - rl)[:100] + 10) <= 1)
    # Left rows 10..2999 have a right twin inside the first 3000 right frames.
    assert len(pairs) >= 0.95 * (3000 - 10)
