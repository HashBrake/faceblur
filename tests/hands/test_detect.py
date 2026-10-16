"""MediaPipe Hands wrapper: conversion, skeleton constants, real-footage smoke test."""
from __future__ import annotations

import time
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from hands.calib import build_rectification, load_calibration
from hands.detect import (BONES, MEDIAPIPE_JOINTS, CropDetector, Hand2D, HandDetector, crop_window,
                          hands_from_result)


def test_blank_image_returns_empty():
    with HandDetector() as det:
        assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []


def _fake_hand(seed, label, score):
    rng = np.random.default_rng(seed)
    norm = rng.uniform(0, 1, (21, 3))
    world = rng.normal(0, 0.05, (21, 3))
    lm = NS(landmark=[NS(x=a, y=b, z=c) for a, b, c in norm])
    wlm = NS(landmark=[NS(x=a, y=b, z=c) for a, b, c in world])
    hd = NS(classification=[NS(label=label, score=score)])
    return norm, world, lm, wlm, hd


def test_hands_from_result_conversion():
    w, h = 1600, 1300
    n0, w0, lm0, wl0, hd0 = _fake_hand(0, "Left", 0.9)
    n1, w1, lm1, wl1, hd1 = _fake_hand(1, "Right", 0.7)
    res = NS(multi_hand_landmarks=[lm0, lm1], multi_hand_world_landmarks=[wl0, wl1],
             multi_handedness=[hd0, hd1])
    out = hands_from_result(res, w, h)
    assert len(out) == 2 and all(isinstance(o, Hand2D) for o in out)
    for o, n, wo, label, score in ((out[0], n0, w0, "right", 0.9), (out[1], n1, w1, "left", 0.7)):
        assert o.keypoints.shape == (21, 2) and o.keypoints.dtype == np.float64
        assert np.allclose(o.keypoints, n[:, :2] * [w, h])
        assert o.world.shape == (21, 3) and o.world.dtype == np.float64
        assert np.allclose(o.world, wo)
        assert o.conf.shape == (21,) and o.conf.dtype == np.float64
        assert np.allclose(o.conf, score)
        assert o.label == label and o.score == pytest.approx(score)


def test_hands_from_result_empty():
    res = NS(multi_hand_landmarks=None, multi_hand_world_landmarks=None, multi_handedness=None)
    assert hands_from_result(res, 640, 480) == []


def test_skeleton_constants():
    assert len(MEDIAPIPE_JOINTS) == 21 and len(set(MEDIAPIPE_JOINTS)) == 21
    assert MEDIAPIPE_JOINTS[0] == "WRIST" and MEDIAPIPE_JOINTS[20] == "PINKY_TIP"
    assert len(BONES) == 20
    assert sorted(c for _, c in BONES) == list(range(1, 21))
    for p, c in BONES:
        assert 0 <= p <= 20 and p != c
    assert sorted(c for p, c in BONES if p == 0) == [1, 5, 9, 13, 17]
    assert all(p == c - 1 for p, c in BONES if p != 0)


W, H = 1600, 1300


def _coord_image():
    """Float image whose pixel (y, x) holds (x, y): a crop's top-left pixel gives its offset."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.stack([xs, ys, np.zeros_like(xs)], axis=2)


def _hand_at(cx, cy, size, score=0.9, seed=0):
    rng = np.random.default_rng(seed)
    kp = np.array([cx, cy]) + rng.uniform(-size / 2, size / 2, (21, 2)) + 0.123456789
    return Hand2D(kp, np.full(21, score), "left", score, np.zeros((21, 3)))


class SyntheticInner:
    """Sees the full-image hands ``truth`` that lie wholly inside the image it is given."""

    def __init__(self, truth):
        self.truth, self.shapes = truth, []

    def detect(self, image):
        self.shapes.append(image.shape[:2])
        x0, y0 = image[0, 0, 0], image[0, 0, 1]
        h, w = image.shape[:2]
        out = []
        for t in self.truth:
            kp = t.keypoints - [x0, y0]
            if np.all((kp >= 0) & (kp < [w, h])):
                out.append(Hand2D(kp, t.conf.copy(), t.label, t.score, t.world))
        return out


def test_crop_maps_back_exactly_middle_and_border():
    middle, border = _hand_at(800, 650, 120, seed=1), _hand_at(30, 1270, 50, seed=2)
    inner = SyntheticInner([middle, border])
    det = CropDetector(inner, (W, H))
    prev = [_hand_at(805, 645, 120, seed=3), _hand_at(25, 1272, 50, seed=4)]
    out = det.detect(_coord_image(), prev)
    assert len(inner.shapes) == 2 and all(s[0] == s[1] and s[0] >= 256 for s in inner.shapes)
    assert inner.shapes[0][0] < W  # a real crop, not the full frame
    x0, y0, side = crop_window(prev[1].keypoints, (W, H))
    assert x0 == 0 and y0 == H - side  # border crop clamped (shifted) into the image
    assert len(out) == 2
    for t in (middle, border):
        best = min(out, key=lambda o: np.abs(o.keypoints - t.keypoints).max())
        assert np.abs(best.keypoints - t.keypoints).max() <= 1e-9


def test_crop_dedupes_same_hand_keeps_higher_score():
    a = _hand_at(800, 650, 100, score=0.6, seed=1)
    b = Hand2D(a.keypoints + 3.0, a.conf, "right", 0.8, a.world)  # same hand, 3 px off, higher score

    class TwoCrops:
        def __init__(self):
            self.calls = 0

        def detect(self, image):
            x0, y0 = image[0, 0, 0], image[0, 0, 1]
            t = (a, b)[self.calls]
            self.calls += 1
            return [Hand2D(t.keypoints - [x0, y0], t.conf, t.label, t.score, t.world)]

    det = CropDetector(TwoCrops(), (W, H))
    out = det.detect(_coord_image(), [_hand_at(790, 640, 100, seed=5), _hand_at(810, 660, 100, seed=6)])
    assert len(out) == 1 and out[0].score == 0.8 and out[0].label == "right"
    assert np.abs(out[0].keypoints - b.keypoints).max() <= 1e-9


def test_crop_without_previous_hands_uses_full_frame():
    t = _hand_at(400, 300, 100)
    inner = SyntheticInner([t])
    out = CropDetector(inner, (W, H)).detect(_coord_image(), [])
    assert inner.shapes == [(H, W)]
    assert len(out) == 1 and np.abs(out[0].keypoints - t.keypoints).max() <= 1e-9


def test_crop_with_one_previous_hand_also_searches_full_frame():
    tracked, new = _hand_at(400, 300, 100, seed=1), _hand_at(1200, 900, 100, seed=2)
    inner = SyntheticInner([tracked, new])
    out = CropDetector(inner, (W, H)).detect(_coord_image(), [_hand_at(405, 305, 100, seed=3)])
    assert inner.shapes[-1] == (H, W)
    assert len(out) == 2  # the tracked hand, seen in its crop and the full frame, counted once


@pytest.mark.real_footage
def test_real_footage_left_view(recording_ev):
    video = next(recording_ev.glob("*_camera_left_part0001.mp4"))
    yaml_path = next(recording_ev.glob("*_calibration_camera.yaml"))
    rect = build_rectification(load_calibration(yaml_path))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 600 * 30)
    frames = []
    for _ in range(60):
        ok, img = cap.read()
        assert ok
        frames.append(rect.rectify(img, "left"))
    cap.release()
    h, w = frames[0].shape[:2]

    results, t = [], 0.0
    with HandDetector() as det:
        for f in frames:
            t0 = time.perf_counter()
            results.append(det.detect(f))
            t += time.perf_counter() - t0
    fps_full = len(frames) / t

    t_half, n_half = 0.0, 0
    with HandDetector() as det:
        for f in frames:
            small = cv2.resize(f, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            n_half += bool(det.detect(small))
            t_half += time.perf_counter() - t0
    fps_half = len(frames) / t_half

    counts = [len(r) for r in results]
    hands = [hd for r in results for hd in r]
    agree = sum((hd.label == "right") == (hd.keypoints[:, 0].mean() > w / 2) for hd in hands)
    print(f"\n[T04] per-frame hand counts: {counts}")
    print(f"[T04] frames with >=1 hand: {sum(c > 0 for c in counts)}/60 (2x downscaled: {n_half}/60)")
    print("[T04] label:x-centre/w: " + ", ".join(
        f"{hd.label}:{hd.keypoints[:, 0].mean() / w:.2f}" for hd in hands))
    if hands:
        print(f"[T04] label/half agreement: {agree}/{len(hands)} = {agree / len(hands):.2f}")
        print(f"[T04] mean handedness score: {np.mean([hd.score for hd in hands]):.3f}")
    print(f"[T04] detect fps full-res {w}x{h}: {fps_full:.1f}; 2x downscaled: {fps_half:.1f}")

    assert sum(c > 0 for c in counts) >= 30
    for hd in hands:
        assert np.all((hd.keypoints >= 0) & (hd.keypoints <= [w, h]))
