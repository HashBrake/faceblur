"""Tests for hands.mano (MANO loader, joints-only model and fits)."""
from __future__ import annotations

import numpy as np
import pytest

from hands.mano import MP_FROM_MANO, TIP_VERTICES, fit_pose, fit_sequence, forward_joints, load_mano

# Inverse map: MANO-ordered (21,) index -> MediaPipe position.
MANO_FROM_MP = np.argsort(MP_FROM_MANO)
# MANO chains (3 joints each) in tip order thumb, index, middle, ring, pinky.
CHAINS = [[13, 14, 15], [1, 2, 3], [4, 5, 6], [10, 11, 12], [7, 8, 9]]


def _model(mano_dir, side="right"):
    return load_mano(side, mano_dir=mano_dir)


def _random_pose(model, rng, scale=0.3):
    return np.concatenate([rng.normal(0, 0.5, 3), model.hands_mean + rng.normal(0, scale, 45)])


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="MANO files not found at"):
        load_mano("right", mano_dir=tmp_path)


def test_mp_from_mano_is_permutation():
    assert sorted(MP_FROM_MANO) == list(range(21))
    assert len(TIP_VERTICES) == 5


def test_loader_shapes(mano_dir):
    m = _model(mano_dir)
    assert m.v_template.shape == (778, 3)
    assert m.shapedirs.shape == (778, 3, 10)
    assert m.posedirs.shape == (778, 3, 135)
    assert m.J_regressor.shape == (16, 778)
    assert m.weights.shape == (778, 16)
    assert m.kintree_table.shape == (2, 16)
    assert m.hands_mean.shape == (45,)
    assert m.hands_components.shape == (45, 45)
    for a in (m.v_template, m.shapedirs, m.posedirs, m.J_regressor, m.weights):
        assert a.dtype == np.float64
    np.testing.assert_allclose(m.J_regressor.sum(axis=1), 1.0, atol=1e-6)
    assert m.kintree_table[0, 0] in (4294967295, -1)
    assert all(m.kintree_table[0, k] < k for k in range(1, 16))


def test_rest_pose(mano_dir):
    m = _model(mano_dir)
    mp = forward_joints(m, np.zeros(10), np.zeros(48), np.zeros(3))
    mano21 = mp[MANO_FROM_MP]
    np.testing.assert_allclose(mano21[:16], m.J_regressor @ m.v_template, atol=1e-9)
    # Tips are the rest vertices and each lies nearest its own chain's distal joint.
    np.testing.assert_allclose(mano21[16:], m.v_template[TIP_VERTICES], atol=1e-12)
    distal = np.array([mano21[c[-1]] for c in CHAINS])
    for f in range(5):
        d = np.linalg.norm(distal - mano21[16 + f], axis=1)
        assert np.argmin(d) == f
    for a, b in [(1, 2), (2, 3), (3, 4)]:
        assert np.linalg.norm(mp[b] - mp[a]) < 0.06
    assert np.linalg.norm(mp[8] - mp[0]) > np.linalg.norm(mp[5] - mp[0])


def test_fit_pose_recovers_known_pose(mano_dir):
    m = _model(mano_dir)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(3):
        betas = rng.normal(0, 0.5, 10)
        pose = _random_pose(m, rng)
        trans = np.array([0.05, -0.03, 0.5])
        gt = forward_joints(m, betas, pose, trans)
        p, t, rmse = fit_pose(m, betas, gt, np.ones(21))
        err = np.linalg.norm(forward_joints(m, betas, p, t) - gt, axis=1).mean()
        errs.append(err)
        assert rmse < 0.005
    print(f"fit_pose mean joint errors (mm): {np.round(np.array(errs) * 1000, 3)}")
    assert max(errs) < 0.005


def test_fit_pose_ignores_nan_joints(mano_dir):
    m = _model(mano_dir)
    rng = np.random.default_rng(1)
    pose = _random_pose(m, rng, 0.2)
    gt = forward_joints(m, np.zeros(10), pose, np.array([0, 0, 0.4]))
    obs, conf = gt.copy(), np.ones(21)
    obs[[4, 8]] = np.nan
    conf[[4, 8]] = 0
    p, t, rmse = fit_pose(m, np.zeros(10), obs, conf)
    assert np.isfinite(rmse) and rmse < 0.005
    assert np.isfinite(p).all() and np.isfinite(t).all()


def test_fit_sequence_recovers_walk(mano_dir):
    m = _model(mano_dir)
    rng = np.random.default_rng(2)
    T = 40
    betas = rng.normal(0, 0.5, 10)
    pose = _random_pose(m, rng)
    trans = np.array([0.05, -0.03, 0.5])
    joints = np.empty((T, 21, 3))
    for i in range(T):
        pose = pose + rng.normal(0, 0.03, 48)
        trans = trans + rng.normal(0, 0.003, 3)
        joints[i] = forward_joints(m, betas, pose, trans)
    out = fit_sequence(m, joints, np.ones((T, 21)), np.ones(T, bool))
    fit = np.stack([forward_joints(m, out["betas"], out["pose"][i], out["trans"][i]) for i in range(T)])
    err = np.linalg.norm(fit - joints, axis=-1).mean()
    print(f"fit_sequence mean joint error {err * 1000:.3f} mm, {out['seconds'] / T:.4f} s/frame")
    assert out["pose"].shape == (T, 48) and out["trans"].shape == (T, 3) and out["rmse_m"].shape == (T,)
    assert err < 0.005


def test_fit_sequence_invalid_frames_nan(mano_dir):
    m = _model(mano_dir)
    rng = np.random.default_rng(3)
    T = 6
    joints = np.stack([forward_joints(m, np.zeros(10), _random_pose(m, rng, 0.1), [0, 0, 0.4]) for _ in range(T)])
    valid = np.array([1, 1, 0, 1, 0, 1], bool)
    out = fit_sequence(m, joints, np.ones((T, 21)), valid, n_shape_frames=3)
    assert np.isnan(out["pose"][~valid]).all() and np.isnan(out["rmse_m"][~valid]).all()
    assert np.isfinite(out["pose"][valid]).all()
    assert "seconds" in out


def test_left_mirrors_right(mano_dir):
    r = forward_joints(_model(mano_dir, "right"), np.zeros(10), np.zeros(48), np.zeros(3))
    lm = load_mano("left", mano_dir=mano_dir)
    assert lm.side == "left"
    left = forward_joints(lm, np.zeros(10), np.zeros(48), np.zeros(3))
    assert np.abs(left[:, 0] + r[:, 0]).max() < 0.01
    np.testing.assert_allclose(left[:, 1:], r[:, 1:], atol=0.01)
