"""Pipeline driver: ``python -m hands.run RECORDING_DIR -o OUT [--part N] [--start S --duration S] [--mano]``.

Pass 1 detects, matches and triangulates every stereo frame pair. Per hand the
triangulated joints are fused with MediaPipe's world-landmark shape (hands.fuse),
bone-constrained, Kalman/RTS-smoothed and bone-constrained again so the output
bone lengths are exact. Pass 2 re-reads the
part and writes the review video of the OUTPUT joints. Outputs in OUT:
joints.parquet, quality.json, review.mp4 (+ mano.parquet, mano_shape.json).
joints.parquet's reproj_px column is the RAW stereo triangulation residual (same as quality.json reproj_px).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hands.calib import build_rectification, load_calibration
from hands.fuse import fuse_hand
from hands.match import HandTracker
from hands.output import (ReviewWriter, project_rectified, write_joints_parquet, write_mano_parquet,
                          write_mano_shape_json)
from hands.quality import quality_report, write_quality_json
from hands.skeleton import constrain_bones, reference_lengths
from hands.smooth import smooth_joints
from hands.timing import container_timestamps, find_part_files, stereo_frames
from hands.triangulate import reprojection_error_original, row_disagreement, triangulate_hand
from hands.video import PartReader

FPS = 30.0
HANDS = ("left", "right")
PROGRESS_EVERY = 300
REVIEW_SIZE = (1600, 650)
INTERPOLATED_MANO_CONF = 0.5  # interpolated frames have conf 0; the fit still needs weights


def _default_detector():
    from hands.detect import HandDetector
    return HandDetector()


def _reproj_px(rect, rig, joints, uv_l, uv_r) -> np.ndarray:
    """(T,21) mean of the two views' ORIGINAL-pixel reprojection errors; NaN without observation."""
    T = len(joints)
    e = reprojection_error_original(rect, rig, joints.reshape(-1, 3), uv_l.reshape(-1, 2), uv_r.reshape(-1, 2))
    return e.mean(axis=1).reshape(T, 21)


def run_pipeline(recording_dir, out_dir, part=1, start_s=0.0, duration_s=None, mano=False, mano_dir=None,
                 detector_factory=None, log=print) -> dict:
    t_run = time.perf_counter()
    rec, out = Path(recording_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    left_mp4, right_mp4, _, _ = find_part_files(rec, part)
    calib_hits = sorted(rec.glob("*_calibration_camera.yaml"))
    if len(calib_hits) != 1:
        raise FileNotFoundError(f"{len(calib_hits)} files match *_calibration_camera.yaml in {rec}")
    rig = load_calibration(calib_hits[0])
    rect = build_rectification(rig)
    image_size = (rig.left.width, rig.left.height)

    max_frames = None if duration_s is None else int((start_s + duration_s) * 31) + 200
    cts = (container_timestamps(left_mp4, max_frames), container_timestamps(right_mp4, max_frames))
    pairs = stereo_frames(rec, part, start_s, duration_s, container_ts=cts)
    if not pairs:
        raise ValueError("no synchronised stereo frames in the requested range")
    T = len(pairs)
    log(f"{rec.name} part {part}: {T} stereo frames from left frame {pairs[0].frame_left}")

    # Pass 1: detection, matching, triangulation.
    factory = detector_factory or _default_detector
    det_l, det_r = factory(), factory()
    tracker = HandTracker(rect)
    raw = {h: {"joints": np.full((T, 21, 3), np.nan),
               "uv_l": np.full((T, 21, 2), np.nan), "uv_r": np.full((T, 21, 2), np.nan),
               "conf_det": np.zeros((T, 21)), "row_dis": np.full((T, 21), np.nan),
               "world_l": np.full((T, 21, 3), np.nan), "world_r": np.full((T, 21, 3), np.nan),
               "present": np.zeros(T, bool)} for h in HANDS}
    det_seconds = 0.0
    reader = PartReader(left_mp4, right_mp4, image_size, cts)
    for t, (_, img_l, img_r) in enumerate(reader.frames(pairs)):
        t0 = time.perf_counter()
        hl = det_l.detect(rect.rectify(img_l, "left"))
        hr = det_r.detect(rect.rectify(img_r, "right"))
        det_seconds += time.perf_counter() - t0
        for hand, (i, j) in tracker.update(hl, hr).items():
            joints, _, _ = triangulate_hand(rect, hl[i], hr[j])
            r = raw[hand]
            r["joints"][t] = joints
            r["uv_l"][t], r["uv_r"][t] = hl[i].keypoints, hr[j].keypoints
            r["conf_det"][t] = np.minimum(hl[i].conf, hr[j].conf)
            r["row_dis"][t] = row_disagreement(hl[i].keypoints, hr[j].keypoints)
            r["world_l"][t], r["world_r"][t] = hl[i].world, hr[j].world
            r["present"][t] = True
        if (t + 1) % PROGRESS_EVERY == 0:
            log(f"{t + 1}/{T} frames done, detection {(t + 1) / det_seconds:.1f} fps")
    reader.close()
    for d in (det_l, det_r):
        if hasattr(d, "close"):
            d.close()
    detection_fps = T / det_seconds if det_seconds > 0 else None

    # Post-processing per hand.
    hands, raw_hands = {}, {}
    for hand in HANDS:
        r = raw[hand]
        fused, conf_f = np.full((T, 21, 3), np.nan), np.zeros((T, 21))
        for t in np.flatnonzero(r["present"]):
            fused[t], conf_f[t] = fuse_hand(r["joints"][t], r["conf_det"][t], r["row_dis"][t],
                                            r["world_l"][t], r["world_r"][t])
        conf_f = np.where(np.isfinite(fused).all(axis=2), conf_f, 0.0)
        lengths = reference_lengths(fused, conf_f)
        constrained = constrain_bones(fused, lengths)
        joints_s, valid, interpolated = smooth_joints(constrained, conf_f, fps=FPS)
        joints_out = constrain_bones(joints_s, lengths)  # exact output lengths after smoothing
        hands[hand] = {"joints": joints_out, "conf": np.where(valid[:, None], conf_f, 0.0), "valid": valid,
                       "interpolated": interpolated,
                       "reproj_px": _reproj_px(rect, rig, joints_out, r["uv_l"], r["uv_r"])}
        raw_hands[hand] = {"joints": r["joints"],
                           "reproj_px": _reproj_px(rect, rig, r["joints"], r["uv_l"], r["uv_r"])}
    frames = {"frame": np.array([p.frame_left for p in pairs], np.int64),
              "timestamp_us": np.array([p.timestamp_us for p in pairs], np.int64)}
    write_joints_parquet(out / "joints.parquet", frames,
                         {h: {**hands[h], "reproj_px": raw_hands[h]["reproj_px"]} for h in HANDS})

    # Pass 2: review video of the output joints.
    reader = PartReader(left_mp4, right_mp4, image_size, cts)
    with ReviewWriter(out / "review.mp4", size=REVIEW_SIZE, fps=FPS) as writer:
        for t, (_, img_l, img_r) in enumerate(reader.frames(pairs)):
            draw = {}
            for hand, h in hands.items():
                if h["valid"][t]:
                    uv_l, uv_r = project_rectified(rect, h["joints"][t])
                    draw[hand] = (uv_l, uv_r, h["conf"][t])
            writer.write(rect.rectify(img_l, "left"), rect.rectify(img_r, "right"), draw)
    reader.close()
    run_seconds = time.perf_counter() - t_run

    meta = {"recording": rec.name, "part": part, "start_s": start_s, "duration_s": duration_s,
            "calibration": calib_hits[0].name, "n_frames": T, "fps_video": FPS, "detection_fps": detection_fps,
            "frames_left": [pairs[0].frame_left, pairs[-1].frame_left], "mano": bool(mano)}

    if mano:
        from hands.mano import fit_sequence, load_mano
        t0 = time.perf_counter()
        fits, shapes, rmse_mm = {}, {}, {}
        for hand, h in hands.items():
            if not h["valid"].any():
                continue
            conf = np.where(h["interpolated"][:, None], INTERPOLATED_MANO_CONF, h["conf"])
            fit = fit_sequence(load_mano(hand, mano_dir), h["joints"], conf, h["valid"])
            fits[hand], shapes[hand] = fit, fit["betas"]
            rmse_mm[hand] = float(np.nanmean(fit["rmse_m"]) * 1000.0)
        if fits:
            write_mano_parquet(out / "mano.parquet", frames, fits)
            write_mano_shape_json(out / "mano_shape.json", shapes,
                                  {"recording": rec.name, "part": part, "start_s": start_s,
                                   "duration_s": duration_s, "rmse_mm": rmse_mm})
        meta["mano_seconds"] = time.perf_counter() - t0
        log(f"MANO fit: {meta['mano_seconds']:.1f} s, rmse mm {rmse_mm}")

    report = quality_report(frames, hands, raw_hands, rect.size, T / run_seconds, meta)
    write_quality_json(out / "quality.json", report)
    log(_summary(report))
    return report


def _fmt(x, spec=".2f"):
    return "n/a" if x is None else format(x, spec)


def _summary(report: dict) -> str:
    o = report["overall"]
    parts = [f"reproj median {_fmt(o['reproj_px_median'])} px (output {_fmt(o['reproj_out_px_median'])})",
             f"bone CV max {_fmt(o['bone_cv_max'], '.3f')}"]
    for hand in HANDS:
        if hand in report:
            r = report[hand]
            parts.append(f"{hand} valid {_fmt(r['valid_share'])} raw CV max {_fmt(r['bone_cv_raw_max'], '.3f')}")
    parts += [f"any-hand valid {_fmt(o['any_hand_valid_share'])}", f"jitter {_fmt(o['jitter_mm'])} mm",
              f"detection {_fmt(report['clip'].get('detection_fps'), '.1f')} fps",
              f"processing {_fmt(report['processing_fps'], '.1f')} fps"]
    return "summary: " + ", ".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m hands.run", description=__doc__.splitlines()[0])
    p.add_argument("recording_dir", help="Orbbec Ego recording folder")
    p.add_argument("-o", "--out", required=True, help="output folder (created)")
    p.add_argument("--part", type=int, default=1)
    p.add_argument("--start", type=float, default=0.0, help="seconds from the part's first left timestamp")
    p.add_argument("--duration", type=float, default=None, help="seconds (default: to the end of the part)")
    p.add_argument("--mano", action="store_true", help="also fit MANO")
    p.add_argument("--mano-dir", default=None, help="MANO model folder (default: env MANO_DIR)")
    a = p.parse_args(argv)
    run_pipeline(a.recording_dir, a.out, part=a.part, start_s=a.start, duration_s=a.duration,
                 mano=a.mano, mano_dir=a.mano_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
