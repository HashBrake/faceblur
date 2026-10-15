import json

import numpy as np

from hands.quality import quality_report, write_quality_json
from hands.skeleton import BONES, constrain_bones

T = 100
META = {"recording": "rec01", "part": 0, "start_s": 0.0, "duration_s": 3.33,
        "calibration": "calibration_camera.yaml", "n_frames": T, "fps_video": 30.0}


def _skeleton(rng, lengths):
    """(T,21,3) hands with random bone directions and (T,20) per-frame bone lengths."""
    base = rng.normal(0, 1, (T, 21, 3))
    out = np.zeros((T, 21, 3))
    out[:, 0] = [0, 0, 0.5] + rng.normal(0, 0.01, (T, 3))
    for b, (p, c) in enumerate(BONES):
        d = base[:, c] - base[:, p]
        out[:, c] = out[:, p] + lengths[:, b:b + 1] * d / np.linalg.norm(d, axis=1, keepdims=True)
    return out


def _data(reproj_scale=1.0, nan_frames=()):
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.02, 0.05, 20)
    hands, raw = {}, {}
    for i, hand in enumerate(("left", "right")):
        raw_j = _skeleton(rng, ref * (1 + 0.03 * rng.normal(size=(T, 20))))
        out_j = constrain_bones(raw_j, ref)
        valid = np.ones(T, bool)
        valid[i * 10:i * 10 + 20] = False  # left: 0..19 invalid, right: 10..29 invalid
        interp = np.zeros(T, bool)
        interp[50:53] = True
        out_j[~valid] = np.nan
        reproj = rng.uniform(0, 2, (T, 21)) * reproj_scale
        reproj[~valid] = np.nan
        for t in nan_frames:
            reproj[t, :3] = np.nan
        raw_reproj = rng.uniform(0, 1.5, (T, 21)) * reproj_scale  # raw residual, differs from output
        raw_j[5:8] = np.nan  # raw wrist missing: excluded from the raw statistics
        raw_reproj[5:8] = rng.uniform(50, 60, (3, 21))
        for t in nan_frames:
            raw_reproj[t, :3] = np.nan
        hands[hand] = {"joints": out_j, "conf": rng.uniform(0.5, 1, (T, 21)), "valid": valid,
                       "interpolated": interp, "reproj_px": reproj}
        raw[hand] = {"joints": raw_j, "reproj_px": raw_reproj}
    frames = {"frame": np.arange(T, dtype=np.int64),
              "timestamp_us": np.arange(T, dtype=np.int64) * 33333}
    return frames, hands, raw


def _frame_means(h):
    keep = h["valid"] & ~h["interpolated"]
    return np.nanmean(h["reproj_px"][keep], axis=1)


def _raw_frame_means(raw):
    keep = np.isfinite(raw["joints"][:, 0]).all(axis=1)
    return np.nanmean(raw["reproj_px"][keep], axis=1)


def test_quality_report_values(tmp_path):
    frames, hands, raw = _data(nan_frames=(60, 61))
    rep = quality_report(frames, hands, raw, (1600, 1300), 12.5, META)
    assert rep["clip"] == META
    assert rep["processing_fps"] == 12.5
    for hand in ("left", "right"):
        r = rep[hand]
        assert r["valid_share"] == 0.8
        assert r["interpolated_share"] == 0.03
        assert r["frames_valid"] == 80 and r["frames_total"] == T
        assert len(r["bone_cv"]) == 20 and len(r["bone_cv_raw"]) == 20
        assert len(r["bone_lengths_mm"]) == 20
        assert r["bone_cv_max"] < 1e-9
        assert 0.02 < r["bone_cv_raw_max"] < 0.05
        fm = _frame_means(hands[hand])
        assert abs(r["reproj_out_px"]["median"] - np.median(fm)) < 1e-12
        assert abs(r["reproj_out_px"]["p90"] - np.percentile(fm, 90)) < 1e-12
        assert abs(r["reproj_out_px"]["mean"] - np.mean(fm)) < 1e-12
        fr = _raw_frame_means(raw[hand])
        assert len(fr) == T - 3
        assert abs(r["reproj_px"]["median"] - np.median(fr)) < 1e-12
        assert abs(r["reproj_px"]["p90"] - np.percentile(fr, 90)) < 1e-12
        assert abs(r["reproj_px"]["mean"] - np.mean(fr)) < 1e-12
        assert r["jitter_mm"] > 0 and r["jitter_raw_mm"] > 0
        assert 0.5 < r["mean_conf"] < 1
        assert "implausible_joint_share" in r and r["implausible_joint_share"] is None
    ref_mm = np.median(np.linalg.norm(
        hands["left"]["joints"][20:, 1] - hands["left"]["joints"][20:, 0], axis=1)) * 1000
    assert abs(rep["left"]["bone_lengths_mm"][0] - ref_mm) < 1e-9
    pooled = np.concatenate([_frame_means(hands[h]) for h in ("left", "right")])
    pooled_raw = np.concatenate([_raw_frame_means(raw[h]) for h in ("left", "right")])
    o = rep["overall"]
    assert abs(o["reproj_out_px_median"] - np.median(pooled)) < 1e-12
    assert abs(o["reproj_px_median"] - np.median(pooled_raw)) < 1e-12
    assert o["reproj_px_median"] != o["reproj_out_px_median"]
    assert o["any_hand_valid_share"] == 0.9
    assert o["bone_cv_max"] < 1e-9
    assert o["targets"] == {"reproj_px_median": 3.0, "bone_cv_max": 0.05}
    assert o["pass"] == {"reproj": True, "bone_cv": True}
    assert o["implausible_joint_share"] is None
    assert abs(o["jitter_mm"] - (rep["left"]["jitter_mm"] + rep["right"]["jitter_mm"]) / 2) < 1e-12

    path = tmp_path / "quality.json"
    write_quality_json(path, rep)
    assert json.loads(path.read_text(encoding="utf-8"))["overall"]["pass"]["reproj"] is True


def test_quality_report_fail_flags_and_nan():
    frames, hands, raw = _data(reproj_scale=10.0)
    rng = np.random.default_rng(5)
    hands["left"]["joints"] = _skeleton(rng, 0.03 * (1 + 0.10 * rng.normal(size=(T, 20))))  # CV ~0.1
    hands["right"]["valid"][:] = False
    hands["right"]["joints"][:] = np.nan
    hands["right"]["reproj_px"][:] = np.nan
    raw["right"]["joints"][:] = np.nan
    rep = quality_report(frames, hands, raw, (1600, 1300), 10.0, META)
    assert rep["overall"]["pass"] == {"reproj": False, "bone_cv": False}
    assert rep["overall"]["reproj_px_median"] > 3.0
    r = rep["right"]
    assert r["reproj_px"] is None and r["reproj_out_px"] is None
    assert r["bone_cv_max"] is None and all(v is None for v in r["bone_cv"])
    assert all(v is None for v in r["bone_lengths_mm"])
    assert r["jitter_mm"] is None and r["mean_conf"] is None
    assert r["valid_share"] == 0.0
    json.dumps(rep, allow_nan=False)
