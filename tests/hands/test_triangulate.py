import numpy as np
import pytest

from hands.calib import build_rectification, load_calibration
from hands.conftest import CALIB_YAML
from hands.detect import Hand2D
from hands.triangulate import (reprojection_error, reprojection_error_original, row_disagreement,
                               triangulate, triangulate_hand)


@pytest.fixture(scope="module")
def rig():
    return load_calibration(CALIB_YAML)


@pytest.fixture(scope="module")
def rect(rig):
    return build_rectification(rig)


def synthetic_hand(z=0.5):
    """21 joints: wrist at origin, thumb + four fingers fanning upward, ~18 cm span, at depth z."""
    pts = [[0.0, 0.0, 0.0]]
    for k, ang in enumerate(np.radians([-60, -25, -5, 15, 35])):
        direction = np.array([np.sin(ang), -np.cos(ang), 0.1 * k])
        direction /= np.linalg.norm(direction)
        steps = (0.03, 0.06, 0.085, 0.105) if k == 0 else (0.04, 0.08, 0.11, 0.135)
        base = np.array([0.0, -0.045, 0.0]) if k else np.zeros(3)
        pts.extend(base + direction * s for s in steps)
    return np.array(pts) + [0.02, 0.05, z]


def project_rect(rect, rig, X):
    uvL = rect.original_to_rectified(rect.left.project(X), "left")
    uvR = rect.original_to_rectified(rect.right.project(X @ rig.R.T + rig.t), "right")
    return uvL, uvR


def random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-0.25, 0.25, (n, 2)), rng.uniform(0.3, 1.0, n)]


def test_synthetic_hand_span():
    H = synthetic_hand()
    span = np.linalg.norm(H[:, None] - H[None], axis=2).max()
    assert H.shape == (21, 3) and 0.15 < span < 0.22


def test_exact_triangulation_sub_mm(rect, rig):
    X = np.r_[random_points(2000), synthetic_hand()]
    uvL, uvR = project_rect(rect, rig, X)
    err = np.linalg.norm(triangulate(rect, uvL, uvR) - X, axis=1)
    near = X[:, 2] <= 0.5
    print(f"\nmax error z<=0.5 m: {err[near].max():.3e} m; max error up to 1.0 m: {err.max():.3e} m")
    assert near.sum() > 500
    assert err[near].max() < 1e-3
    assert np.isfinite(err).all()


def test_reprojection_zero_exact(rect, rig):
    X = np.r_[random_points(500, 1), synthetic_hand()]
    uvL, uvR = project_rect(rect, rig, X)
    P = triangulate(rect, uvL, uvR)
    assert reprojection_error(rect, P, uvL, uvR).max() < 1e-6
    # original pixels: the ~1e-6 px residual row misalignment of the rectification
    # (fisheye.stereoRectify) is amplified by the KB mapping to a few 1e-6 px
    assert reprojection_error_original(rect, rig, P, uvL, uvR).max() < 1e-5


def test_noise_sensitivity(rect, rig):
    rng = np.random.default_rng(2)
    X = np.c_[rng.uniform(-0.15, 0.15, (2000, 2)), np.full(2000, 0.5)]
    uvL, uvR = project_rect(rect, rig, X)
    P = triangulate(rect, uvL + rng.normal(0, 1, uvL.shape), uvR + rng.normal(0, 1, uvR.shape))
    med = np.median(np.abs(P[:, 2] - X[:, 2]))
    print(f"\nmedian |depth error| at 0.5 m, 1 px noise: {med * 1000:.2f} mm")
    assert 2e-3 < med < 8e-3


def test_invalid_disparity_nan(rect, rig):
    uvL, uvR = project_rect(rect, rig, synthetic_hand())
    P = triangulate(rect, uvR, uvL)
    assert P.shape == (21, 3) and np.isnan(P).all()
    assert np.isnan(reprojection_error(rect, P, uvR, uvL)).all()
    assert np.isnan(reprojection_error_original(rect, rig, P, uvR, uvL)).all()


def test_row_disagreement():
    assert np.allclose(row_disagreement([[1, 2], [3, 4]], [[0, 2.5], [9, 1]]), [0.5, 3.0])


def _hand(kp):
    return Hand2D(kp, np.full(21, 0.9), "right", 0.9, np.zeros((21, 3)))


def test_triangulate_hand(rect, rig):
    X = synthetic_hand()
    uvL, uvR = project_rect(rect, rig, X)
    joints, conf, reproj = triangulate_hand(rect, _hand(uvL), _hand(uvR))
    assert np.abs(joints - X).max() < 1e-6
    assert np.allclose(conf, 0.9, atol=1e-6)
    assert np.abs(reproj).max() < 1e-6

    uvR2 = uvR.copy()
    uvR2[5, 1] += 2.0
    joints, conf, reproj = triangulate_hand(rect, _hand(uvL), _hand(uvR2))
    assert conf[5] < conf[4]
    # mean-row triangulation splits the 2 px row offset evenly: 1 px in each view, 2 px total
    assert np.allclose(reproj[5], 1.0, atol=1e-6)
    assert row_disagreement(uvL, uvR2)[5] == pytest.approx(2.0)
