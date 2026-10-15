"""Frame timing: csv pts, container frame -> csv row, left/right pairing.

Each camera part is an mp4 plus a ``*_pts.csv`` (header ``timestamp_us``, one
sensor timestamp per captured frame). Findings on AZER76400EV
``..._003108`` part 1 (left and right, one hour each):

* ``cv2.CAP_PROP_FRAME_COUNT`` says 109,858 / 109,863 frames, but that is only
  duration x the container's average rate (30.51 fps). Counting packets, or
  ``grab()`` to the end, gives exactly 108,030 frames per camera, the same as
  the csv row count. There are **no extra container frames** (0 per hour) and
  no duplicates: frames 0..3 differ by 6.2-6.7 grey levels mean abs, and the
  container has no step shorter than 32.8 ms except one.
* That one is the first step: frame 0 has pts 0 and frame 1 pts 1500/90000 s
  (16.7 ms), then ~33.3 ms steps that copy the csv jitter (32.8-33.8 ms). So
  the container timeline is the sensor timeline shifted by about -16.9 ms,
  with frame 0 clamped to 0. Frame 0 is a distinct image, so container frame i
  is csv row i; frame 0 is left unmatched by the tolerance rule below (one
  row per part is lost), which is the documented, accepted cost.
* Clock skew: the container clock runs 169 ppm slow against the sensor clock
  (left -169.2 ppm, right -168.8 ppm): 627 ms over the hour, far beyond any
  matching tolerance. After a linear fit ``container = k * rel + b`` the
  residual over the whole hour is 0.21 ms median, 0.63 ms p99, 0.98 ms max on
  both cameras, so a linear drift model is enough for a 4 ms tolerance.
* Left and right csv are hardware-synchronised (99.99 % of left timestamps
  have a right one within 5 us) but the right csv starts 10 rows (333 ms)
  later, so the cameras are paired by timestamp, never by row index.

Container timestamps come from ffmpeg packet pts (``-c copy -f framecrc``,
the ffmpeg binary of ``imageio_ffmpeg``): identical to
``grab()`` + ``CAP_PROP_POS_MSEC`` (max difference 0.5 ns over the hour) but
about 3x faster because nothing is decoded (100-120 s vs ~300 s per hour).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FramePair:
    frame_left: int
    frame_right: int
    row_left: int
    row_right: int
    timestamp_us: int


def load_pts(csv_path) -> np.ndarray:
    return np.loadtxt(csv_path, skiprows=1, dtype=np.int64, ndmin=1)


def container_timestamps(video_path, max_frames=None) -> np.ndarray:
    """Presentation time of every container frame in us, in presentation order."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video_path),
           "-map", "0:v:0", "-c", "copy"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    out = subprocess.run(cmd + ["-f", "framecrc", "-"], capture_output=True, text=True, check=True).stdout
    tb = 1.0
    pts = []
    for line in out.splitlines():
        if line.startswith("#tb 0:"):
            num, den = line.split(":", 1)[1].strip().split("/")
            tb = int(num) / int(den)
        elif line and not line.startswith("#"):
            pts.append(int(line.split(",")[2]))
    return np.sort(np.array(pts, dtype=np.float64)) * tb * 1e6


def _nearest(sorted_vals, x):
    j = np.clip(np.searchsorted(sorted_vals, x), 1, len(sorted_vals) - 1)
    left_closer = np.abs(x - sorted_vals[j - 1]) <= np.abs(sorted_vals[j] - x)
    j = np.where(left_closer, j - 1, j)
    return j, np.abs(sorted_vals[j] - x)


def frame_to_row(container_ts_us, pts_us, tolerance_us=4000) -> np.ndarray:
    """csv row of each container frame, or -1 when no row is within tolerance.

    Fits ``container = k * (pts - pts[0]) + b`` (clock skew and offset) on
    matched frames: ``b`` from the first 300 frames, then the fit is refined
    while the matched span doubles, then every frame is matched once. A
    duplicated container frame (half-frame step) lands between two rows and
    gets -1; a row with no container frame (dropped frame) is simply skipped.
    """
    c = np.asarray(container_ts_us, dtype=np.float64)
    rel = (np.asarray(pts_us, dtype=np.int64) - pts_us[0]).astype(np.float64)

    def match(n, k, b):
        j, d = _nearest(rel, (c[:n] - b) / k)
        return np.where(d <= tolerance_us, j, -1)

    n = min(len(c), 300)
    cands = np.unique((c[:20, None] - rel[None, :20]).ravel())
    counts = np.array([(match(n, 1.0, b0) >= 0).sum() for b0 in cands])
    # Whole-frame shifts of b match almost equally well, so among the offsets
    # within 2 matches of the best, take the one closest to 0 that does not
    # put the container after the csv (part 1: -16.9 ms beats +16.4 ms).
    b = max(zip(counts, cands), key=lambda cb: (cb[0] >= counts.max() - 2, cb[1] <= tolerance_us, -abs(cb[1])))[1]
    k = 1.0
    while True:
        rows = match(n, k, b)
        ok = rows >= 0
        if ok.sum() >= 2:
            k, b = np.polyfit(rel[rows[ok]], c[:n][ok], 1)
        if n == len(c):
            return match(n, k, b).astype(np.int64)
        n = min(len(c), 2 * n)


def pair_stereo(rows_L_ts, rows_R_ts, tolerance_us=2000) -> np.ndarray:
    """(row_L, row_R) pairs whose timestamps agree within tolerance, monotonic, no reuse."""
    tl = np.asarray(rows_L_ts, dtype=np.int64)
    tr = np.asarray(rows_R_ts, dtype=np.int64)
    j, d = _nearest(tr, tl)
    pairs = []
    last = -1
    for i in np.nonzero(d <= tolerance_us)[0]:
        if j[i] > last:
            pairs.append((i, j[i]))
            last = j[i]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def find_part_files(recording_dir, part):
    d = Path(recording_dir)
    out = []
    for name in (f"*_camera_left_part{part:04d}.mp4", f"*_camera_right_part{part:04d}.mp4",
                 f"*_camera_left_part{part:04d}_pts.csv", f"*_camera_right_part{part:04d}_pts.csv"):
        hits = sorted(d.glob(name))
        if len(hits) != 1:
            raise FileNotFoundError(f"{len(hits)} files match {name} in {d}")
        out.append(hits[0])
    return tuple(out)


def _first_frame_per_row(frame_rows, n_rows):
    row2frame = np.full(n_rows, -1, dtype=np.int64)
    rows, first = np.unique(frame_rows, return_index=True)
    keep = rows >= 0
    row2frame[rows[keep]] = first[keep]
    return row2frame


def stereo_frames(recording_dir, part, start_s=None, duration_s=None, container_ts=None) -> list[FramePair]:
    """Synchronised container frame indices of a part, in ``[start_s, start_s + duration_s)``
    seconds from the left csv's first timestamp. ``container_ts`` = (left, right)
    replaces reading the videos (tests, or a cached timeline)."""
    left_mp4, right_mp4, left_csv, right_csv = find_part_files(recording_dir, part)
    pts_l, pts_r = load_pts(left_csv), load_pts(right_csv)
    if container_ts is None:
        container_ts = (container_timestamps(left_mp4), container_timestamps(right_mp4))
    r2f_l = _first_frame_per_row(frame_to_row(container_ts[0], pts_l), len(pts_l))
    r2f_r = _first_frame_per_row(frame_to_row(container_ts[1], pts_r), len(pts_r))
    t0 = pts_l[0]
    lo = t0 + (start_s or 0) * 1e6
    hi = np.inf if duration_s is None else lo + duration_s * 1e6
    out = []
    for rl, rr in pair_stereo(pts_l, pts_r):
        fl, fr = r2f_l[rl], r2f_r[rr]
        if fl >= 0 and fr >= 0 and lo <= pts_l[rl] < hi:
            out.append(FramePair(int(fl), int(fr), int(rl), int(rr), int(pts_l[rl])))
    return out
