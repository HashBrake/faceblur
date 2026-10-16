import time

import numpy as np

from hands.smooth import jitter_mm, smooth_joints

FPS = 30.0


def _sinusoid(T=150, noise=0.005, seed=0):
    t = np.arange(T) / FPS
    truth = np.zeros((T, 21, 3))
    truth[:, :, 0] = 0.1 * np.sin(2 * np.pi * t)[:, None] + np.arange(21) * 0.01
    truth[:, :, 2] = 0.5
    rng = np.random.default_rng(seed)
    return truth, truth + rng.normal(0, noise, truth.shape), np.ones((T, 21))


def test_reduces_noise_without_lag():
    truth, meas, conf = _sinusoid()
    s, valid, interp = smooth_joints(meas, conf)
    assert valid.all() and not interp.any()
    rms_noise = np.sqrt(np.mean((meas - truth) ** 2))
    rms_s = np.sqrt(np.mean((s - truth) ** 2))
    assert rms_s < rms_noise
    x, y = s[:, 0, 0] - s[:, 0, 0].mean(), truth[:, 0, 0] - truth[:, 0, 0].mean()
    xc = np.correlate(x, y, mode="full")
    assert np.argmax(xc) - (len(y) - 1) == 0


def test_short_gap_filled_long_gap_invalid():
    truth, meas, conf = _sinusoid(T=150)
    meas[40:45] = np.nan
    conf[40:45] = 0
    meas[90:120] = np.nan
    conf[90:120] = 0
    s, valid, interp = smooth_joints(meas, conf)
    assert valid[40:45].all() and interp[40:45].all()
    # Per joint the fill carries the 5 mm measurement noise; the hand position
    # (joint centroid) and the per-axis RMS error are both within 1 cm.
    assert np.sqrt(np.mean((s[40:45] - truth[40:45]) ** 2)) < 0.01
    assert np.linalg.norm(s[40:45].mean(1) - truth[40:45].mean(1), axis=-1).max() < 0.01
    assert not valid[90:120].any() and not interp[90:120].any()
    assert np.isnan(s[90:120]).all()
    assert valid[:40].all() and not interp[:40].any() and valid[120:].all()


def test_outlier_rejected():
    truth, meas, conf = _sinusoid()
    meas[70] += np.array([0.2, 0.0, 0.0])
    s, valid, _ = smooth_joints(meas, conf)
    assert valid[70]
    assert np.linalg.norm(s[70] - truth[70], axis=-1).mean() < 0.01
    assert np.linalg.norm(s[70].mean(0) - truth[70].mean(0)) < 0.01


def test_noise_free_gap_fill_accurate():
    truth, _, conf = _sinusoid(noise=0.0)
    meas = truth.copy()
    meas[40:45] = np.nan
    conf[40:45] = 0
    s, _, interp = smooth_joints(meas, conf)
    assert interp[40:45].all()
    assert np.linalg.norm(s[40:45] - truth[40:45], axis=-1).max() < 0.01


def test_leading_missing_invalid():
    truth, meas, conf = _sinusoid()
    meas[:3] = np.nan
    conf[:3] = 0
    s, valid, interp = smooth_joints(meas, conf)
    assert not valid[:3].any() and not interp[:3].any()
    assert np.isnan(s[:3]).all()
    assert valid[3:].all()


def test_jitter_static_hand():
    rng = np.random.default_rng(3)
    T = 300
    meas = np.full((T, 21, 3), 0.3) + rng.normal(0, 0.001, (T, 21, 3))
    conf = np.ones((T, 21))
    ones, zeros = np.ones(T, bool), np.zeros(T, bool)
    before = jitter_mm(meas, ones, zeros)
    assert 0.5 < before < 4
    s, valid, interp = smooth_joints(meas, conf)
    assert jitter_mm(s, valid, interp) < before


def test_fast_on_long_clip():
    rng = np.random.default_rng(4)
    meas = rng.normal(0, 0.005, (4000, 21, 3))
    t0 = time.perf_counter()
    smooth_joints(meas, np.ones((4000, 21)))
    assert time.perf_counter() - t0 < 1.0
