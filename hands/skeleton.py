"""Bone-length constraint for triangulated 21-joint hands.

Stereo noise makes raw bone lengths vary from frame to frame. Each hand's 20
bone lengths are fixed to the recording's robust (median) estimate and every
frame is rebuilt from the wrist outwards along the measured bone directions.
Arrays: ``joints`` (T, 21, 3) metres with NaN for missing, ``conf`` (T, 21).
"""
from __future__ import annotations

import numpy as np

# Mirrors hands.detect.BONES (not imported: hands.detect pulls in MediaPipe).
# (parent, child) in MediaPipe order; parents always come before children.
BONES = [(0 if j % 4 == 1 else j - 1, j) for j in range(1, 21)]
_PARENT = np.array([p for p, _ in BONES])
_CHILD = np.array([c for _, c in BONES])


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    """(T, 21, 3) -> (T, 20) bone lengths; NaN where either end is NaN."""
    return np.linalg.norm(joints[:, _CHILD] - joints[:, _PARENT], axis=-1)


def reference_lengths(joints: np.ndarray, conf: np.ndarray, min_conf: float = 0.5) -> np.ndarray:
    """(20,) median length per bone over frames where both ends have conf >= min_conf.

    Falls back to all finite frames when fewer than 10 frames qualify; NaN if none.
    """
    lengths = bone_lengths(joints)
    good = (conf[:, _PARENT] >= min_conf) & (conf[:, _CHILD] >= min_conf) & np.isfinite(lengths)
    out = np.full(len(BONES), np.nan)
    for b in range(len(BONES)):
        sel = lengths[good[:, b], b]
        if len(sel) < 10:
            sel = lengths[np.isfinite(lengths[:, b]), b]
        if len(sel):
            out[b] = np.median(sel)
    return out


def constrain_bones(joints: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Rebuild every frame from the wrist with fixed bone lengths.

    child = new_parent + lengths[b] * unit(original child - original parent).
    The wrist stays; a NaN joint stays NaN and so do its descendants.
    """
    out = joints.copy()
    for b, (p, c) in enumerate(BONES):
        d = joints[:, c] - joints[:, p]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = d / np.linalg.norm(d, axis=-1, keepdims=True)
        out[:, c] = out[:, p] + lengths[b] * u
    return out


def bone_cv(joints: np.ndarray) -> np.ndarray:
    """(20,) coefficient of variation (std / mean) of finite bone lengths; NaN if < 2."""
    lengths = bone_lengths(joints)
    out = np.full(len(BONES), np.nan)
    for b in range(len(BONES)):
        sel = lengths[np.isfinite(lengths[:, b]), b]
        if len(sel) >= 2:
            out[b] = sel.std() / sel.mean()
    return out
