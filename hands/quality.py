"""Per-clip quality report (quality.json): validity, reprojection, bone CV, jitter.

Two reprojection figures per hand:
  * ``reproj_px``: the stereo reprojection error, i.e. the triangulation residual of the
    RAW 2-D observations (how well the left and right views agree), over frames where
    the raw wrist is finite. This is the stereo-consistency measure and drives
    ``overall.pass.reproj``.
  * ``reproj_out_px``: the OUTPUT joints' deviation from the same observations, over
    valid, non-interpolated frames. The output is fused with the hand-shape prior,
    smoothed and bone-constrained on purpose, so it departs from the noisy observations;
    this is reported separately and is not a stereo-consistency measure.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np

from hands.skeleton import bone_cv, bone_lengths
from hands.smooth import jitter_mm

TARGET_REPROJ_PX_MEDIAN = 3.0
TARGET_BONE_CV_MAX = 0.05
HANDS = ("left", "right")


def _num(x):
    """float or None (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _nanmax(a):
    a = np.asarray(a, np.float64)
    return float(np.max(a[np.isfinite(a)])) if np.isfinite(a).any() else None


def _row_mean(r) -> np.ndarray:
    """(T,21) -> (T,) mean over finite entries, NaN where none."""
    r = np.asarray(r, np.float64)
    finite = np.isfinite(r)
    n = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(finite, r, 0.0).sum(axis=1) / n, np.nan)


def _frame_reproj(h) -> np.ndarray:
    """Per-frame mean over finite joints of the output reproj_px, valid, non-interpolated frames."""
    mean = _row_mean(h["reproj_px"])
    keep = np.asarray(h["valid"], bool) & ~np.asarray(h["interpolated"], bool) & np.isfinite(mean)
    return mean[keep]


def _stats(rp):
    return ({"median": float(np.median(rp)), "p90": float(np.percentile(rp, 90)),
             "mean": float(np.mean(rp))} if len(rp) else None)


def _hand_report(h, raw) -> dict:
    joints = np.asarray(h["joints"], np.float64)
    valid = np.asarray(h["valid"], bool)
    interp = np.asarray(h["interpolated"], bool)
    T = len(valid)
    raw_joints = np.asarray(raw["joints"], np.float64)
    raw_valid = np.isfinite(raw_joints[:, 0]).all(axis=1)

    rp = _frame_reproj(h)
    rm = _row_mean(raw["reproj_px"])
    rp_raw = rm[raw_valid & np.isfinite(rm)]
    cv = bone_cv(joints)
    cv_raw = bone_cv(raw_joints)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lengths = bone_lengths(joints)
        med_len = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan
                            for c in lengths.T]) * 1000.0
        conf = np.asarray(h["conf"], np.float64)[valid]
        mean_conf = np.nanmean(conf) if conf.size else np.nan
        jit = jitter_mm(joints, valid, interp)
        jit_raw = jitter_mm(raw_joints, raw_valid, np.zeros(T, bool))
    out = {
        "valid_share": valid.mean() if T else None,
        "interpolated_share": interp.mean() if T else None,
        "reproj_px": _stats(rp_raw),
        "reproj_out_px": _stats(rp),
        "bone_cv": [_num(v) for v in cv],
        "bone_cv_raw": [_num(v) for v in cv_raw],
        "bone_cv_max": _nanmax(cv),
        "bone_cv_raw_max": _nanmax(cv_raw),
        "bone_lengths_mm": [_num(v) for v in med_len],
        "jitter_mm": jit,
        "jitter_raw_mm": jit_raw,
        "mean_conf": mean_conf,
        "frames_valid": int(valid.sum()),
        "frames_total": int(T),
        "_reproj_out_frames": rp,
        "_reproj_frames": rp_raw,
    }
    return out


def _clean(obj):
    """Recursively convert numpy scalars and NaN to JSON-ready values."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    return obj


def quality_report(frames, hands, raw_hands, rect_size, fps_processing, meta) -> dict:
    """JSON-ready quality dict for one clip. ``rect_size`` is accepted for the driver but unused."""
    T = len(frames["frame"])
    report = {"clip": dict(meta), "processing_fps": fps_processing}
    any_valid = np.zeros(T, bool)
    pooled, pooled_out, cv_maxes, jitters = [], [], [], []
    for hand in HANDS:
        if hand not in hands:
            continue
        r = _hand_report(hands[hand], raw_hands[hand])
        pooled.append(r.pop("_reproj_frames"))
        pooled_out.append(r.pop("_reproj_out_frames"))
        any_valid |= np.asarray(hands[hand]["valid"], bool)
        if r["bone_cv_max"] is not None:
            cv_maxes.append(r["bone_cv_max"])
        if _num(r["jitter_mm"]) is not None:
            jitters.append(r["jitter_mm"])
        report[hand] = r
    rp = np.concatenate(pooled) if pooled else np.empty(0)
    reproj_median = float(np.median(rp)) if len(rp) else None
    cv_max = max(cv_maxes) if cv_maxes else None
    rp_out = np.concatenate(pooled_out) if pooled_out else np.empty(0)
    report["overall"] = {
        "reproj_px_median": reproj_median,
        "reproj_out_px_median": float(np.median(rp_out)) if len(rp_out) else None,
        "bone_cv_max": cv_max,
        "any_hand_valid_share": any_valid.mean() if T else None,
        "jitter_mm": float(np.mean(jitters)) if jitters else None,
        "targets": {"reproj_px_median": TARGET_REPROJ_PX_MEDIAN, "bone_cv_max": TARGET_BONE_CV_MAX},
        "pass": {
            "reproj": reproj_median is not None and reproj_median <= TARGET_REPROJ_PX_MEDIAN,
            "bone_cv": cv_max is not None and cv_max <= TARGET_BONE_CV_MAX,
        },
    }
    return _clean(report)


def write_quality_json(path, report: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, indent=2)
