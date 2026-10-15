"""KB4 camera model, stereo rig convention and rectification (real calibration file)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from hands.calib import build_rectification, load_calibration
from hands.conftest import CALIB_YAML


@pytest.fixture(scope="module")
def rig():
    return load_calibration(CALIB_YAML)


@pytest.fixture(scope="module")
def rect(rig):
    return build_rectification(rig)


def _rect_pixels(rect, X_left):
    """Exact rectified pixels of left-frame points, via the rectified left frame."""
    Xr = X_left @ rect.R_left.T
    h = np.c_[Xr, np.ones(len(Xr))]
    ql, qr = h @ rect.P_left.T, h @ rect.P_right.T
    return ql[:, :2] / ql[:, 2:3], qr[:, :2] / qr[:, 2:3], Xr[:, 2]


def _random_points(n, zmin, zmax, seed):
    rng = np.random.default_rng(seed)
    z = rng.uniform(zmin, zmax, n)
    # within ~45 deg of the axis so the points stay in both kept fields of view
    xy = rng.uniform(-0.8, 0.8, (n, 2)) * z[:, None]
    return np.c_[xy + [0.06, 0.0], z]


@pytest.mark.parametrize("side", ["left", "right"])
def test_kb_round_trip_whole_image(rig, side):
    cam = getattr(rig, side)
    u, v = np.meshgrid(np.arange(0, cam.width + 1, 25.0), np.arange(0, cam.height + 1, 25.0))
    px = np.c_[u.ravel(), v.ravel()]
    rays = cam.unproject(px)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
    assert np.abs(cam.project(rays) - px).max() < 0.1


@pytest.mark.parametrize("side", ["left", "right"])
def test_kb_project_matches_opencv_fisheye(rig, side):
    cam = getattr(rig, side)
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-1, 1, (1000, 2)), rng.uniform(0.2, 2.0, 1000)]
    ref, _ = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
    assert np.abs(cam.project(pts) - ref.reshape(-1, 2)).max() < 1e-6


def test_extrinsics_convention_and_depth_identity(rig, rect):
    g = np.linspace(-0.3, 0.3, 7)
    X = np.array([[x, y, z] for z in (0.3, 0.5, 1.0, 1.5) for x in g for y in g])
    Xr = X @ rig.R.T + rig.t
    assert (X[:, 2] > 0).all() and (Xr[:, 2] > 0).all()
    pl, pr = rig.left.project(X), rig.right.project(Xr)
    assert ((pr >= 0) & (pr <= [rig.right.width - 1, rig.right.height - 1])).all()
    # right camera centre sits at +x of the left camera
    assert (-rig.R.T @ rig.t)[0] > 0.119
    rl = rect.original_to_rectified(pl, "left")
    rr = rect.original_to_rectified(pr, "right")
    disp = rl[:, 0] - rr[:, 0]
    assert (disp > 0).all()
    Z = (X @ rect.R_left.T)[:, 2]
    assert np.abs(rect.focal * rig.baseline_m / disp - Z).max() / Z.min() < 1e-6
    assert np.allclose(rect.focal * rig.baseline_m / disp / Z, 1.0, rtol=1e-6, atol=0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_rectification_round_trip(rig, rect, side):
    X = _random_points(500, 0.2, 2.0, seed=2)
    cam = getattr(rig, side)
    pts = X if side == "left" else X @ rig.R.T + rig.t
    px = cam.project(pts)
    back = rect.rectified_to_original(rect.original_to_rectified(px, side), side)
    assert np.abs(back - px).max() < 0.1


def test_rectified_ray_left_is_parallel(rect):
    X = _random_points(500, 0.2, 2.0, seed=3)
    pl, _, _ = _rect_pixels(rect, X)
    rays = rect.rectified_to_ray_left(pl)
    u = X / np.linalg.norm(X, axis=1, keepdims=True)
    ang = np.arccos(np.clip(np.sum(rays * u, axis=1), -1, 1))
    assert ang.max() < 1e-6


def test_row_alignment(rig, rect):
    X = _random_points(500, 0.2, 2.0, seed=4)
    rl = rect.original_to_rectified(rig.left.project(X), "left")
    rr = rect.original_to_rectified(rig.right.project(X @ rig.R.T + rig.t), "right")
    assert np.abs(rl[:, 1] - rr[:, 1]).max() < 0.05
    assert (rl[:, 0] - rr[:, 0] > 0).all()
    # the exact pinhole projections agree with the KB path (right: up to the
    # float precision of OpenCV's R_right, a few 1e-6 px)
    el, er, _ = _rect_pixels(rect, X)
    assert np.abs(rl - el).max() < 1e-6 and np.abs(rr - er).max() < 1e-4


def test_maps_agree_with_rectified_to_original(rect):
    rng = np.random.default_rng(5)
    px = rng.uniform([400, 300], [1200, 1000], (200, 2))
    for side, (m1, m2) in (("left", rect.map_left), ("right", rect.map_right)):
        iu, iv = px.round().astype(int).T
        from_map = np.c_[m1[iv, iu], m2[iv, iu]]
        assert np.abs(from_map - rect.rectified_to_original(np.c_[iu, iv], side)).max() < 0.01


def test_baseline_focal_and_image_shape(rig, rect):
    assert 0.119 < rig.baseline_m < 0.121
    assert rect.focal == pytest.approx(461.88, abs=0.01)
    assert rect.size == (1600, 1300)
    img = np.random.default_rng(6).integers(0, 256, (1300, 1600, 3), dtype=np.uint8)
    for side in ("left", "right"):
        out = rect.rectify(img, side)
        assert out.shape == img.shape and out.dtype == np.uint8
