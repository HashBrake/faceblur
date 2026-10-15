"""Temporal smoothing of 3-D hand joints with gap handling.

Per joint and axis: a constant-velocity Kalman filter forward, then a
Rauch-Tung-Striebel (RTS) smoother backward. The RTS pass makes the result
zero-phase (no lag). Missing (NaN) measurements are prediction-only steps; a
measurement whose normalised innovation exceeds ``GATE`` sigma is treated as an
outlier and skipped; after ``MAX_REJECT`` consecutive rejections a channel
accepts again so a real fast motion cannot lock the filter out.

Noise defaults: ``meas_std = 0.004`` m matches ~1 px of keypoint noise in
depth at 0.5 m. ``process_std = 5`` m/s^2 (white acceleration noise) is above
the ~4 m/s^2 peak acceleration of a 10 cm, 1 Hz hand motion, so a hand moving
at ~1 m/s is followed without bias while frame-to-frame noise is averaged out.
Measurement variance is meas_std^2 / max(conf, 0.05): low confidence, less weight.
"""
from __future__ import annotations

import numpy as np

GATE = 4.0
MAX_REJECT = 2  # after this many consecutive gated frames a channel accepts again (no lock-out)
_INIT_VEL_STD = 1.0  # m/s, prior on velocity at a channel's first measurement


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _kalman_rts(z: np.ndarray, r: np.ndarray, dt: float, q: float) -> np.ndarray:
    """Constant-velocity Kalman + RTS over (T, C) measurements z with variances r."""
    T, C = z.shape
    qa, qb, qc = q * dt**4 / 4, q * dt**3 / 2, q * dt**2
    xf = np.full((T, C, 2), np.nan)          # filtered state (p, v)
    pf = np.full((T, C, 3), np.nan)          # filtered cov (a, b, c) = [[a, b], [b, c]]
    xp = np.full((T, C, 2), np.nan)          # predicted state
    pp = np.full((T, C, 3), np.nan)          # predicted cov
    x = np.full((C, 2), np.nan)
    P = np.full((C, 3), np.nan)
    init = np.zeros(C, bool)
    rejected = np.zeros(C, int)
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in range(T):
            if t > 0:
                a, b, c = P.T
                x = np.stack([x[:, 0] + dt * x[:, 1], x[:, 1]], axis=1)
                P = np.stack([a + 2 * dt * b + dt * dt * c + qa, b + dt * c + qb, c + qc], axis=1)
            xp[t], pp[t] = x, P
            has = np.isfinite(z[t])
            new = has & ~init
            x[new] = np.stack([z[t, new], np.zeros(new.sum())], axis=1)
            P[new] = np.stack([r[t, new], np.zeros(new.sum()), np.full(new.sum(), _INIT_VEL_STD**2)], axis=1)
            init |= new
            a, b, c = P.T
            s = a + r[t]
            y = z[t] - x[:, 0]
            inlier = np.abs(y) <= GATE * np.sqrt(s)
            upd = has & ~new & (inlier | (rejected >= MAX_REJECT))
            rejected = np.where(has & ~new & ~upd, rejected + 1, 0)
            k0, k1 = a / s, b / s
            x[upd, 0] += k0[upd] * y[upd]
            x[upd, 1] += k1[upd] * y[upd]
            P[upd] = np.stack([a * (1 - k0), b * (1 - k0), c - b * b / s], axis=1)[upd]
            xf[t], pf[t] = x, P
        xs = xf.copy()
        for t in range(T - 2, -1, -1):
            a, b, c = pf[t].T
            pa, pb, pc = pp[t + 1].T
            det = pa * pc - pb * pb
            # M = Pf F^T, G = M Pp^-1
            m00, m01, m10, m11 = a + b * dt, b, b + c * dt, c
            g00 = (m00 * pc - m01 * pb) / det
            g01 = (m01 * pa - m00 * pb) / det
            g10 = (m10 * pc - m11 * pb) / det
            g11 = (m11 * pa - m10 * pb) / det
            e = xs[t + 1] - xp[t + 1]
            xs[t, :, 0] = xf[t, :, 0] + g00 * e[:, 0] + g01 * e[:, 1]
            xs[t, :, 1] = xf[t, :, 1] + g10 * e[:, 0] + g11 * e[:, 1]
    return xs[:, :, 0]


def smooth_joints(joints: np.ndarray, conf: np.ndarray, fps: float = 30.0,
                  process_std: float = 5.0, meas_std: float = 0.004, max_gap: int = 10):
    """Kalman/RTS-smooth (T, 21, 3) joints; returns (joints_s, valid, interpolated).

    A frame is missing when the wrist (joint 0) is NaN. Interior gaps of at most
    ``max_gap`` frames are filled from the smoother (valid, interpolated); longer
    gaps and leading/trailing missing frames are NaN and invalid.
    """
    T = joints.shape[0]
    z = joints.reshape(T, -1)
    w = np.maximum(np.repeat(np.nan_to_num(conf, nan=0.0), 3, axis=1), 0.05)
    r = meas_std**2 / w
    s = _kalman_rts(z, r, 1.0 / fps, process_std**2).reshape(joints.shape)

    missing = ~np.isfinite(joints[:, 0]).all(axis=1)
    valid = ~missing
    interpolated = np.zeros(T, bool)
    for a, b in _runs(missing):
        if a > 0 and b < T and b - a <= max_gap:
            valid[a:b] = True
            interpolated[a:b] = True
    s[~valid] = np.nan
    return s, valid, interpolated


def jitter_mm(joints: np.ndarray, valid: np.ndarray, interpolated: np.ndarray) -> float:
    """Jitter: median over joints of the RMS second difference, in mm.

    Second difference x[t+1] - 2 x[t] + x[t-1] per axis, over frames t where
    t-1, t, t+1 are all valid and not interpolated; RMS over those frames and
    the three axes per joint, then the median over the 21 joints.
    """
    ok = valid & ~interpolated
    sel = ok[:-2] & ok[1:-1] & ok[2:]
    d2 = (joints[2:] - 2 * joints[1:-1] + joints[:-2])[sel] * 1000.0
    rms = np.sqrt(np.nanmean(d2**2, axis=(0, 2)))
    return float(np.nanmedian(rms))
