"""Cross-view hand matching and wearer-hand identity (left/right) per stereo frame.

Hands are duck-typed: any object with ``.keypoints`` (21,2) rectified pixels,
``.conf``, ``.label`` ("left"|"right") and ``.score`` (``hands.detect.Hand2D`` fits;
it is not imported here to keep MediaPipe out of this module).

Identity score for "this pair is the wearer's LEFT hand" (higher = more left):
  * label votes: +score per view labelled "left", -score per view labelled "right"
    (two agreeing views at score 0.9 give +-1.8);
  * position prior: +POSITION_WEIGHT (1.0) for the pair with the smaller camera x;
  * continuity: CONTINUITY_WEIGHT (40 / m) * (dist to prev right - dist to prev left),
    a missing previous hand contributes 0. 5 cm of displacement = 2.0, more than one
    label vote (0.9) and the position prior.
Two agreeing wrong labels (3.6 between the two hands) beat the position prior (2.0)
on their own; continuity is what corrects them over time.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from hands.triangulate import _intrinsics, row_disagreement, triangulate

POSITION_WEIGHT = 1.0
CONTINUITY_WEIGHT = 40.0  # score per metre
SINGLE_HAND_GATE_M = 0.15
_INFEASIBLE = 1e9


def match_views(hands_left, hands_right, rect, max_row_px=8.0, depth_range_m=(0.15, 2.0)):
    """Pair left-view and right-view hands: list of (i, j, mean row disagreement px)."""
    if not hands_left or not hands_right:
        return []
    f, cxl, _, cxr, _, B = _intrinsics(rect)
    d_min, d_max = f * B / depth_range_m[1], f * B / depth_range_m[0]
    cost = np.full((len(hands_left), len(hands_right)), _INFEASIBLE)
    for i, hl in enumerate(hands_left):
        for j, hr in enumerate(hands_right):
            L, R = np.asarray(hl.keypoints, float), np.asarray(hr.keypoints, float)
            d = np.mean((L[:, 0] - cxl) - (R[:, 0] - cxr))
            if d_min <= d <= d_max:
                cost[i, j] = row_disagreement(L, R).mean()
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j])) for i, j in zip(rows, cols) if cost[i, j] <= max_row_px]


def hand_position(rect, hand_left, hand_right) -> np.ndarray:
    """Mean of triangulated wrist and middle MCP (joints 0, 9), original left frame, metres."""
    idx = [0, 9]
    P = triangulate(rect, np.asarray(hand_left.keypoints)[idx], np.asarray(hand_right.keypoints)[idx])
    return np.nanmean(P, axis=0)


def _vote(hand) -> float:
    return {"left": 1.0, "right": -1.0}.get(hand.label, 0.0) * float(hand.score)


def assign_identity(pairs, hands_left, hands_right, rect, prev=None):
    """Decide which matched pair is the wearer's left / right hand: {"left": (i, j), ...}."""
    prev = prev or {}
    pairs = list(pairs)[:2]
    if not pairs:
        return {}
    pos = [hand_position(rect, hands_left[i], hands_right[j]) for i, j, *_ in pairs]
    votes = [_vote(hands_left[i]) + _vote(hands_right[j]) for i, j, *_ in pairs]

    if len(pairs) == 1:
        i, j = pairs[0][:2]
        dists = {k: np.linalg.norm(pos[0] - p) for k, p in prev.items()}
        near = {k: d for k, d in dists.items() if d <= SINGLE_HAND_GATE_M}
        if near:
            return {min(near, key=near.get): (i, j)}
        return {"left" if votes[0] > 0 else "right": (i, j)}

    scores = []
    for k in range(2):
        s = votes[k]
        if pos[k][0] < pos[1 - k][0]:
            s += POSITION_WEIGHT
        if "left" in prev:
            s -= CONTINUITY_WEIGHT * np.linalg.norm(pos[k] - prev["left"])
        if "right" in prev:
            s += CONTINUITY_WEIGHT * np.linalg.norm(pos[k] - prev["right"])
        scores.append(s)
    left = 0 if scores[0] >= scores[1] else 1
    return {"left": tuple(pairs[left][:2]), "right": tuple(pairs[1 - left][:2])}


class HandTracker:
    """Per-frame matching + identity, carrying the previous 3-D hand positions."""

    def __init__(self, rect, max_missing=15):
        self.rect = rect
        self.max_missing = max_missing
        self.prev: dict[str, np.ndarray] = {}
        self.missing: dict[str, int] = {}

    def update(self, hands_left, hands_right):
        pairs = match_views(hands_left, hands_right, self.rect)
        ids = assign_identity(pairs, hands_left, hands_right, self.rect, self.prev)
        for key in ("left", "right"):
            if key in ids:
                i, j = ids[key]
                self.prev[key] = hand_position(self.rect, hands_left[i], hands_right[j])
                self.missing[key] = 0
            elif key in self.prev:
                self.missing[key] += 1
                if self.missing[key] >= self.max_missing:
                    del self.prev[key], self.missing[key]
        return ids
