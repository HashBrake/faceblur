from types import SimpleNamespace

import numpy as np
import pytest

from hands.calib import build_rectification, load_calibration
from hands.conftest import CALIB_YAML
from hands.match import HandTracker, assign_identity, hand_position, match_views


@pytest.fixture(scope="module")
def rig():
    return load_calibration(CALIB_YAML)


@pytest.fixture(scope="module")
def rect(rig):
    return build_rectification(rig)


def synthetic_hand(x=0.0, y=0.0, z=0.5):
    """21 joints, ~18 cm span, wrist near (x, y, z) in the original left frame."""
    pts = [[0.0, 0.0, 0.0]]
    for k, ang in enumerate(np.radians([-60, -25, -5, 15, 35])):
        direction = np.array([np.sin(ang), -np.cos(ang), 0.1 * k])
        direction /= np.linalg.norm(direction)
        steps = (0.03, 0.06, 0.085, 0.105) if k == 0 else (0.04, 0.08, 0.11, 0.135)
        base = np.array([0.0, -0.045, 0.0]) if k else np.zeros(3)
        pts.extend(base + direction * s for s in steps)
    return np.array(pts) + [x, y + 0.05, z]


def project_rect(rect, rig, X):
    uvL = rect.original_to_rectified(rect.left.project(X), "left")
    uvR = rect.original_to_rectified(rect.right.project(X @ rig.R.T + rig.t), "right")
    return uvL, uvR


def wrap(uv, label="left", score=0.9):
    return SimpleNamespace(keypoints=uv, conf=np.full(21, 0.9), label=label, score=score,
                           world=np.zeros((21, 3)))


def views(rect, rig, x, y=0.0, z=0.5, label="left", score=0.9):
    uvL, uvR = project_rect(rect, rig, synthetic_hand(x, y, z))
    return wrap(uvL, label, score), wrap(uvR, label, score)


def two_hands(rect, rig, label_a="left", label_b="right", score=0.9):
    """Hand A at x=-0.15, hand B at x=+0.15 (z=0.5). Right-view list is reversed."""
    aL, aR = views(rect, rig, -0.15, label=label_a, score=score)
    bL, bR = views(rect, rig, 0.15, label=label_b, score=score)
    return [aL, bL], [bR, aR]


def test_match_two_hands_reversed_order(rect, rig):
    L, R = two_hands(rect, rig)
    pairs = match_views(L, R, rect)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    assert all(c < 1.0 for *_, c in pairs)


def test_match_rejections(rect, rig):
    hL, hR = views(rect, rig, 0.0)
    assert match_views([hL], [], rect) == []
    assert len(match_views([hL], [hR], rect)) == 1
    neg = wrap(hL.keypoints + [50.0, 0.0])  # right view to the right of the left view
    assert match_views([hL], [neg], rect) == []
    off = wrap(hR.keypoints + [0.0, 10.0])
    assert match_views([hL], [off], rect) == []


def identity(rect, L, R, prev=None):
    pairs = match_views(L, R, rect)
    ids = assign_identity(pairs, L, R, rect, prev)
    # map to physical hand: left-view index 0 = A (x=-0.15), 1 = B
    return {k: ("A" if v[0] == 0 else "B") for k, v in ids.items()}


def test_identity_labels_agree_with_position(rect, rig):
    L, R = two_hands(rect, rig, "left", "right")
    assert identity(rect, L, R) == {"left": "A", "right": "B"}


def test_identity_position_decides_without_labels(rect, rig):
    L, R = two_hands(rect, rig, "left", "left", score=0.0)
    assert identity(rect, L, R) == {"left": "A", "right": "B"}


def test_identity_wrong_labels(rect, rig):
    L, R = two_hands(rect, rig, "right", "left")
    # documented choice: two agreeing wrong labels beat the position prior alone
    assert identity(rect, L, R) == {"left": "B", "right": "A"}
    true_a = hand_position(rect, L[0], R[1])
    true_b = hand_position(rect, L[1], R[0])
    prev = {"left": true_a + [0.03, 0, 0], "right": true_b + [0, 0.03, 0]}
    assert identity(rect, L, R, prev) == {"left": "A", "right": "B"}


def test_identity_single_hand(rect, rig):
    hL, hR = views(rect, rig, 0.1, label="left")
    pairs = match_views([hL], [hR], rect)
    assert assign_identity(pairs, [hL], [hR], rect) == {"left": (0, 0)}
    hL.label = hR.label = "right"
    assert assign_identity(pairs, [hL], [hR], rect) == {"right": (0, 0)}
    # continuity dominates: a previous left hand within 0.15 m wins over the labels
    p = hand_position(rect, hL, hR)
    assert assign_identity(pairs, [hL], [hR], rect, {"left": p + [0.05, 0, 0]}) == {"left": (0, 0)}
    # previous hand too far away: labels decide
    assert assign_identity(pairs, [hL], [hR], rect, {"left": p + [0.3, 0, 0]}) == {"right": (0, 0)}


@pytest.mark.parametrize("score", [0.9, 0.0])
def test_tracker_crossing(rect, rig, score):
    tracker = HandTracker(rect)
    for k in range(11):
        xa, xb = -0.15 + 0.03 * k, 0.15 - 0.03 * k  # A = wearer's left, crosses B
        aL, aR = views(rect, rig, xa, y=-0.05, label="left", score=score)
        bL, bR = views(rect, rig, xb, y=0.05, label="right", score=score)
        ids = tracker.update([bL, aL], [aR, bR])
        assert ids == {"left": (1, 0), "right": (0, 1)}, f"frame {k}"
    assert tracker.prev["left"][0] > tracker.prev["right"][0]  # really crossed
    for _ in range(14):
        assert tracker.update([], []) == {}
    assert set(tracker.prev) == {"left", "right"}
    for _ in range(6):
        tracker.update([], [])
    assert tracker.prev == {}
