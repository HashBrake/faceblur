"""Rectified-stereo triangulation of matched keypoints, and reprojection error.

Inputs are pixels in the RECTIFIED views (what the detector sees); outputs are
3-D points in metres in the ORIGINAL left camera frame (``X_left = R_left.T @ X_rect``).

Closed form: rectified views are row-aligned, so disparity d = u_L - u_R gives
Z = f*B/d, X = (u_L - cx)*Z/f, Y = (v - cy)*Z/f with v the mean of the two rows.
Points with d < ``MIN_DISPARITY_PX`` (behind the rig, or farther than f*B/0.5 ~ 111 m)
are NaN.

In rectified views the reprojection error of this triangulation is just half the
row disagreement |v_L - v_R| per view, so ``row_disagreement`` is the main stereo
residual (and the matching cost for left/right hand pairing).
"""
from __future__ import annotations

import numpy as np

MIN_DISPARITY_PX = 0.5
REPROJ_SCALE_PX = 3.0  # conf penalty: exp(-mean reprojection error / 3 px)


def _intrinsics(rect):
    P1, P2 = rect.P_left, rect.P_right
    f = P1[0, 0]
    return f, P1[0, 2], P1[1, 2], P2[0, 2], P2[1, 2], -P2[0, 3] / f


def row_disagreement(uv_left, uv_right) -> np.ndarray:
    """|v_L - v_R| per point, pixels, (N,)."""
    return np.abs(np.asarray(uv_left, float).reshape(-1, 2)[:, 1]
                  - np.asarray(uv_right, float).reshape(-1, 2)[:, 1])


def triangulate(rect, uv_left, uv_right) -> np.ndarray:
    """(N,2) rectified left/right pixels -> (N,3) points in the ORIGINAL left frame, metres."""
    f, cxl, cyl, cxr, cyr, B = _intrinsics(rect)
    L = np.asarray(uv_left, float).reshape(-1, 2)
    R = np.asarray(uv_right, float).reshape(-1, 2)
    d = (L[:, 0] - cxl) - (R[:, 0] - cxr)
    ok = d >= MIN_DISPARITY_PX
    Z = np.where(ok, f * B / np.where(ok, d, 1.0), np.nan)
    v = 0.5 * ((L[:, 1] - cyl) + (R[:, 1] - cyr))
    X_rect = np.stack([(L[:, 0] - cxl) * Z / f, v * Z / f, Z], axis=1)
    return X_rect @ rect.R_left  # rows: R_left.T @ X_rect


def _project(P, X_rect):
    q = X_rect @ P[:, :3].T + P[:, 3]
    return q[:, :2] / q[:, 2:3]


def reprojection_error(rect, points_left_orig, uv_left_rect, uv_right_rect) -> np.ndarray:
    """(N,2) pixel distance [left, right] between rectified observations and reprojections."""
    X_rect = np.asarray(points_left_orig, float).reshape(-1, 3) @ rect.R_left.T
    obsL = np.asarray(uv_left_rect, float).reshape(-1, 2)
    obsR = np.asarray(uv_right_rect, float).reshape(-1, 2)
    eL = np.linalg.norm(_project(rect.P_left, X_rect) - obsL, axis=1)
    eR = np.linalg.norm(_project(rect.P_right, X_rect) - obsR, axis=1)
    return np.stack([eL, eR], axis=1)


def reprojection_error_original(rect, rig, points_left_orig, uv_left_rect, uv_right_rect) -> np.ndarray:
    """Same as ``reprojection_error`` but in ORIGINAL (KB fisheye) image pixels.

    ``rig`` is the ``StereoRig`` (``X_right = rig.R @ X_left + rig.t``); ``Rectification``
    does not carry the extrinsics. Observations are mapped with ``rectified_to_original``.
    """
    X = np.asarray(points_left_orig, float).reshape(-1, 3)
    obsL = rect.rectified_to_original(uv_left_rect, "left")
    obsR = rect.rectified_to_original(uv_right_rect, "right")
    eL = np.linalg.norm(rect.left.project(X) - obsL, axis=1)
    eR = np.linalg.norm(rect.right.project(X @ rig.R.T + rig.t) - obsR, axis=1)
    return np.stack([eL, eR], axis=1)


def triangulate_hand(rect, hand_left, hand_right):
    """Triangulate the 21 joints of a matched left/right ``Hand2D`` pair.

    Returns ``(joints (21,3), conf (21,), reproj (21,2))``: joints in the original
    left frame (metres), ``conf = min(conf_L, conf_R) * exp(-mean_reproj_px / 3)``
    clamped to [0, 1] (0 for joints that cannot be triangulated), and the rectified
    per-view reprojection errors.
    """
    joints = triangulate(rect, hand_left.keypoints, hand_right.keypoints)
    reproj = reprojection_error(rect, joints, hand_left.keypoints, hand_right.keypoints)
    conf = np.minimum(hand_left.conf, hand_right.conf) * np.exp(-reproj.mean(axis=1) / REPROJ_SCALE_PX)
    conf = np.clip(np.nan_to_num(conf, nan=0.0), 0.0, 1.0)
    return joints, conf, reproj
