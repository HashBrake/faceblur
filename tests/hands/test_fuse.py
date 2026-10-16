import numpy as np

from hands.fuse import align_prior, fuse_hand, stereo_weight


def synthetic_hand(z=0.5):
    """21 joints (MediaPipe order), ~18 cm span, wrist near (0, 0.05, z)."""
    pts = [[0.0, 0.0, 0.0]]
    for k, ang in enumerate(np.radians([-60, -25, -5, 15, 35])):
        direction = np.array([np.sin(ang), -np.cos(ang), 0.1 * k])
        direction /= np.linalg.norm(direction)
        steps = (0.03, 0.06, 0.085, 0.105) if k == 0 else (0.04, 0.08, 0.11, 0.135)
        base = np.array([0.0, -0.045, 0.0]) if k else np.zeros(3)
        pts.extend(base + direction * s for s in steps)
    return np.array(pts) + [0.0, 0.05, z]


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


TRUTH = synthetic_hand()
CONF = np.full(21, 0.9)


def world(shape, offset=(0.0, 0.0, 0.0)):
    """MediaPipe-like world landmarks: the shape centred on an arbitrary point."""
    return shape - shape.mean(axis=0) + offset


def test_stereo_weight():
    rd = np.array([0.0, 8.0, np.nan] + [0.0] * 18)
    w = stereo_weight(rd, CONF)
    assert w[0] == 0.9 and abs(w[1] - 0.9 * np.exp(-1)) < 1e-12 and w[2] == 0.0


def test_exact_triangulation_and_prior():
    joints, conf = fuse_hand(TRUTH, CONF, np.zeros(21), world(TRUTH, (0.1, 0, 0)), world(TRUTH, (0, 0.2, 0)))
    assert np.abs(joints - TRUTH).max() < 1e-6
    assert (conf >= 0.9 * CONF).all()


def test_displaced_joint_rescued_by_prior():
    tri = TRUTH.copy()
    tri[8] += [0.0, 0.0, 0.20]
    rd = np.full(21, 2.0)
    rd[8] = 40.0
    joints, conf = fuse_hand(tri, CONF, rd, world(TRUTH), world(TRUTH))
    assert np.linalg.norm(joints[8] - TRUTH[8]) < 0.005
    assert conf[8] < 0.5
    others = np.delete(np.arange(21), 8)
    assert np.linalg.norm(joints[others] - TRUTH[others], axis=1).max() < 1e-6


def test_all_rows_disagree_prior_rotated():
    shape = TRUTH @ rot_z(20).T  # prior is the true shape rotated by 20 deg
    joints, conf = fuse_hand(TRUTH, CONF, np.full(21, 60.0), world(shape), world(shape))
    fit = align_prior(world(shape), world(shape), TRUTH, stereo_weight(np.full(21, 60.0), CONF))
    assert fit is not None
    _, scale, rms = fit
    assert 0.7 <= scale <= 1.4 and abs(scale - 1.0) < 1e-6 and rms < 1e-6
    assert np.linalg.norm(joints - TRUTH, axis=1).max() < 0.01
    assert np.allclose(conf, 0.3 * np.exp(-rms / 0.02))  # every joint came from the prior


def test_scale_is_clamped():
    fit = align_prior(world(TRUTH * 3), world(TRUTH * 3), TRUTH, CONF)
    assert abs(fit[1] - 0.7) < 1e-12


def test_too_few_usable_joints():
    tri = TRUTH.copy()
    tri[3:] = np.nan
    joints, conf = fuse_hand(tri, CONF, np.zeros(21), world(TRUTH), world(TRUTH))
    assert np.array_equal(joints[:3], TRUTH[:3]) and np.isnan(joints[3:]).all()
    assert np.allclose(conf, stereo_weight(np.zeros(21), CONF))
    assert align_prior(world(TRUTH), world(TRUTH), tri, CONF) is None
