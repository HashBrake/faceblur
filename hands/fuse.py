"""Fuse stereo-triangulated joints with MediaPipe's per-view 3-D hand shape.

Hands at 0.25-0.35 m are seen ~23 deg apart by the two cameras, so the per-view
2-D landmarks of fingers often disagree (row disagreement |v_L - v_R| of tens of
pixels) and their triangulated depth is wrong. MediaPipe's world landmarks give a
metric, hand-centred 3-D shape per view that does not depend on stereo agreement.

Per frame: the shape prior (mean of the two views' centred world landmarks) is
aligned to the triangulated joints by a weighted similarity transform (Umeyama),
trusting joints whose rows agree; then every joint is a blend of the triangulated
point and the aligned prior, or the prior alone when the stereo point is unusable.

``lam`` (0.3) is the prior's weight in the blend against the stereo weight
``w = conf * exp(-(row_dis / 8 px)^2)``: a well-matched joint (w ~ 0.9) is 75 %
stereo, a joint with 8 px row disagreement (w ~ 0.33) is about half and half.
``hard_px`` (25 px) is the row disagreement above which the stereo point is not
used at all (p90 of fingertip disagreement on true pairs is 25-47 px).
"""
from __future__ import annotations

import numpy as np

MIN_ALIGN_JOINTS = 4
MIN_ALIGN_WEIGHT = 0.05   # relative to the largest weight
FAR_FROM_PRIOR_M = 0.15
PRIOR_CONF = 0.3
FIT_SCALE_M = 0.02        # fit quality = exp(-rms / 2 cm)


def stereo_weight(row_dis, conf, sigma_px=8.0) -> np.ndarray:
    """(21,) conf * exp(-(row_dis / sigma)^2), 0 where either input is NaN."""
    w = np.asarray(conf, float) * np.exp(-(np.asarray(row_dis, float) / sigma_px) ** 2)
    return np.nan_to_num(w, nan=0.0)


def align_prior(world_l, world_r, joints_tri, w, scale_range=(0.7, 1.4)):
    """Weighted similarity fit of the world-landmark shape onto the triangulated joints.

    Returns ``(prior (21,3), scale, rms_m)`` with ``prior = scale * R @ shape + t`` in the
    triangulation's frame and ``rms_m`` the weighted RMS residual over the joints used,
    or None when fewer than 4 joints are usable (finite, weight > 0.05 x the largest).
    """
    wl, wr = np.asarray(world_l, float), np.asarray(world_r, float)
    shape = 0.5 * ((wl - wl.mean(axis=0)) + (wr - wr.mean(axis=0)))
    q = np.asarray(joints_tri, float)
    w = np.where(np.isfinite(q).all(axis=1), np.nan_to_num(np.asarray(w, float), nan=0.0), 0.0)
    use = (w > 0) & (w > MIN_ALIGN_WEIGHT * w.max(initial=0.0))
    if use.sum() < MIN_ALIGN_JOINTS:
        return None
    ww = w[use] / w[use].sum()
    P, Q = shape[use], q[use]
    mp, mq = ww @ P, ww @ Q
    Pc, Qc = P - mp, Q - mq
    U, S, Vt = np.linalg.svd((Qc * ww[:, None]).T @ Pc)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ D @ Vt
    var_p = ww @ (Pc ** 2).sum(axis=1)
    if var_p <= 0:  # degenerate (all-zero) shape
        return None
    scale = float(np.clip(np.trace(np.diag(S) @ D) / var_p, *scale_range))
    t = mq - scale * R @ mp
    prior = scale * shape @ R.T + t
    rms = float(np.sqrt(ww @ ((prior[use] - Q) ** 2).sum(axis=1)))
    return prior, scale, rms


def fuse_hand(joints_tri, conf, row_dis, world_l, world_r, lam=0.3, hard_px=25.0):
    """(joints (21,3), conf_out (21,)) for one hand in one frame; see the module docstring."""
    tri = np.asarray(joints_tri, float)
    w = stereo_weight(row_dis, conf)
    fit = align_prior(world_l, world_r, tri, w)
    if fit is None:
        return tri.copy(), w
    prior, _, rms = fit
    prior_conf = PRIOR_CONF * np.exp(-rms / FIT_SCALE_M)
    rd = np.nan_to_num(np.asarray(row_dis, float), nan=np.inf)
    with np.errstate(invalid="ignore"):
        far = ~(np.linalg.norm(tri - prior, axis=1) <= FAR_FROM_PRIOR_M)  # NaN joints count as far
    use_prior = (rd > hard_px) | far
    blend = (w[:, None] * np.nan_to_num(tri) + lam * prior) / (w + lam)[:, None]
    joints = np.where(use_prior[:, None], prior, blend)
    conf_out = np.where(use_prior, prior_conf, np.maximum(w, prior_conf))
    return joints, conf_out
