import numpy as np

from hands.skeleton import BONES, bone_cv, bone_lengths, constrain_bones, reference_lengths


def _hand():
    """Straight fingers fanning out from the wrist, bones 2-9 cm."""
    j = np.zeros((21, 3))
    lens = [0.04, 0.035, 0.03, 0.025]
    for f in range(5):
        ang = np.radians(-40 + 20 * f)
        u = np.array([np.sin(ang), np.cos(ang), 0.1 * f])
        u /= np.linalg.norm(u)
        base = 0.09 if f else 0.05
        pos = np.zeros(3)
        for k, L in enumerate([base] + lens[1:] if f else [0.05, 0.04, 0.03, 0.02]):
            L = L if k else base
            pos = pos + L * u
            j[1 + 4 * f + k] = pos
    return j + np.array([0.0, 0.0, 0.5])


def _noisy(T=200, std=0.003, seed=0):
    rng = np.random.default_rng(seed)
    return _hand()[None] + rng.normal(0, std, (T, 21, 3))


def test_bones_match_detect():
    from hands.detect import BONES as DETECT_BONES
    assert BONES == DETECT_BONES


def test_constraint_fixes_bone_lengths():
    lengths = bone_lengths(_hand()[None])[0]
    assert lengths.min() > 0.0199 and lengths.max() < 0.0901
    noisy = _noisy()
    conf = np.ones((200, 21))
    assert np.mean(bone_cv(noisy) > 0.02) > 0.5
    out = constrain_bones(noisy, reference_lengths(noisy, conf))
    assert np.all(bone_cv(out) < 1e-9)
    disp = np.linalg.norm(out - noisy, axis=-1).mean()
    assert disp < 2 * 0.003
    np.testing.assert_array_equal(out[:, 0], noisy[:, 0])


def test_nan_propagates_to_descendants():
    noisy = _noisy(T=5)
    noisy[2, 6] = np.nan
    out = constrain_bones(noisy, reference_lengths(noisy, np.ones((5, 21))))
    assert np.isnan(out[2, [6, 7, 8]]).all()
    others = [i for i in range(21) if i not in (6, 7, 8)]
    assert np.isfinite(out[2, others]).all()
    assert np.isfinite(out[[0, 1, 3, 4]]).all()


def test_reference_lengths_ignores_low_confidence():
    noisy = _noisy(seed=1)
    conf = np.ones((200, 21))
    ref = reference_lengths(noisy, conf)
    bad = noisy.copy()
    for t in (5, 50, 100, 150, 199):
        bad[t] = bad[t, 0] + 10 * (bad[t] - bad[t, 0])
        conf[t] = 0.1
    np.testing.assert_allclose(reference_lengths(bad, conf), reference_lengths(np.delete(noisy, [5, 50, 100, 150, 199], 0), np.ones((195, 21))))
    np.testing.assert_allclose(reference_lengths(bad, conf), ref, rtol=0.05)


def test_bone_cv_nan_when_too_few():
    j = np.full((3, 21, 3), np.nan)
    j[0] = _hand()
    assert np.isnan(bone_cv(j)).all()
