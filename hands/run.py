"""Pipeline driver: ``python -m hands.run RECORDING_DIR -o OUT [--part N] [--start S --duration S] [--mano] [--world]``.

Pass 1 detects, matches and triangulates every stereo frame pair. Per hand the
triangulated joints are fused with MediaPipe's world-landmark shape (hands.fuse),
bone-constrained, Kalman/RTS-smoothed and bone-constrained again so the output
bone lengths are exact. Pass 2 re-reads the
part and writes the review video of the OUTPUT joints. Outputs in OUT:
joints.parquet, quality.json, review.mp4 (+ mano.parquet, mano_shape.json).
With --world: OUT/trajectory.txt (given, existing, or made by ORB-SLAM3 in Docker), pose.parquet,
joints_world.parquet, a "world" section in quality.json and a top-down trajectory inset in review.mp4.
joints.parquet's reproj_px column is the RAW stereo triangulation residual (same as quality.json reproj_px).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from hands.calib import build_rectification, load_calibration
from hands.fuse import fuse_hand
from hands.match import HandTracker
from hands.output import (ReviewWriter, inset_bounds, project_rectified, write_joints_parquet, write_mano_parquet,
                          write_mano_shape_json, write_pose_parquet)
from hands.quality import add_world_section, quality_report, write_quality_json
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
# A hand joint seen by a head-mounted camera is within arm's reach; depths (left-camera z) outside this
# range come from mismatched landmarks with near-zero disparity and are set to NaN after triangulation.
DEPTH_RANGE_M = (0.1, 2.0)


def _default_detector():
    from hands.detect import HandDetector
    return HandDetector()


def _default_crop_detector():
    from hands.detect import HandDetector
    return HandDetector(static_image_mode=True)  # crops move every frame: no MediaPipe tracking


def _reproj_px(rect, rig, joints, uv_l, uv_r) -> np.ndarray:
    """(T,21) mean of the two views' ORIGINAL-pixel reprojection errors; NaN without observation."""
    T = len(joints)
    e = reprojection_error_original(rect, rig, joints.reshape(-1, 3), uv_l.reshape(-1, 2), uv_r.reshape(-1, 2))
    return e.mean(axis=1).reshape(T, 21)


def _trajectory(rec: Path, out: Path, part, start_s, duration_s, trajectory_path, log) -> Path:
    """OUT/trajectory.txt: copied from ``trajectory_path``, already there, or made by ORB-SLAM3 (Docker)."""
    traj = out / "trajectory.txt"
    if trajectory_path is not None:
        if Path(trajectory_path).resolve() != traj.resolve():
            shutil.copyfile(trajectory_path, traj)
        return traj
    if traj.is_file():
        log(f"world: using existing {traj}")
        return traj
    from hands.odometry.run_orbslam3 import IMAGE
    from hands.odometry.run_orbslam3 import main as orbslam3_main
    if shutil.which("docker") is None:
        raise RuntimeError(f"--world needs a trajectory: pass --trajectory, or install Docker and build the image "
                           f"{IMAGE} as described in STATE.md 'World coordinates / odometry setup (T20)' "
                           f"(docker build -t {IMAGE} hands/odometry)")
    if duration_s is None:
        raise ValueError("--world without a trajectory needs --duration (the ORB-SLAM3 window)")
    log(f"world: running ORB-SLAM3 ({IMAGE}) into {out}")
    orbslam3_main([str(rec), "-o", str(out), "--part", str(part), "--start", str(start_s),
                   "--duration", str(duration_s)])
    return traj


def run_pipeline(recording_dir, out_dir, part=1, start_s=0.0, duration_s=None, mano=False, mano_dir=None,
                 detector_factory=None, crop=False, log=print, world=False, trajectory_path=None) -> dict:
    t_run = time.perf_counter()
    rec, out = Path(recording_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if world:  # before detection, so a missing trajectory or Docker fails fast
        from hands.imu import load_imu_calibration
        imu_hits = sorted(rec.glob("*_calibration_imu.yaml"))
        if len(imu_hits) != 1:
            raise FileNotFoundError(f"{len(imu_hits)} files match *_calibration_imu.yaml in {rec}")
        imu_calib = load_imu_calibration(imu_hits[0])
        traj_file = _trajectory(rec, out, part, start_s, duration_s, trajectory_path, log)
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
    if crop:
        from hands.detect import CropDetector
        factory = detector_factory or _default_crop_detector
        det_l, det_r = CropDetector(factory(), image_size), CropDetector(factory(), image_size)
    else:
        factory = detector_factory or _default_detector
        det_l, det_r = factory(), factory()
    hl, hr = [], []
    tracker = HandTracker(rect)
    raw = {h: {"joints": np.full((T, 21, 3), np.nan),
               "uv_l": np.full((T, 21, 2), np.nan), "uv_r": np.full((T, 21, 2), np.nan),
               "conf_det": np.zeros((T, 21)), "row_dis": np.full((T, 21), np.nan),
               "world_l": np.full((T, 21, 3), np.nan), "world_r": np.full((T, 21, 3), np.nan),
               "present": np.zeros(T, bool), "n_triangulated": 0, "n_gated": 0} for h in HANDS}
    det_seconds = 0.0
    reader = PartReader(left_mp4, right_mp4, image_size, cts)
    for t, (_, img_l, img_r) in enumerate(reader.frames(pairs)):
        t0 = time.perf_counter()
        if crop:
            hl = det_l.detect(rect.rectify(img_l, "left"), hl)
            hr = det_r.detect(rect.rectify(img_r, "right"), hr)
        else:
            hl = det_l.detect(rect.rectify(img_l, "left"))
            hr = det_r.detect(rect.rectify(img_r, "right"))
        det_seconds += time.perf_counter() - t0
        for hand, (i, j) in tracker.update(hl, hr).items():
            joints, _, _ = triangulate_hand(rect, hl[i], hr[j])
            r = raw[hand]
            finite = np.isfinite(joints).all(axis=1)
            gated = finite & ((joints[:, 2] < DEPTH_RANGE_M[0]) | (joints[:, 2] > DEPTH_RANGE_M[1]))
            joints[gated] = np.nan
            r["n_triangulated"] += int(finite.sum())
            r["n_gated"] += int(gated.sum())
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
                           "reproj_px": _reproj_px(rect, rig, r["joints"], r["uv_l"], r["uv_r"]),
                           "row_dis": r["row_dis"],
                           "implausible_joint_share": (r["n_gated"] / r["n_triangulated"]
                                                       if r["n_triangulated"] else None)}
    frames = {"frame": np.array([p.frame_left for p in pairs], np.int64),
              "timestamp_us": np.array([p.timestamp_us for p in pairs], np.int64)}
    write_joints_parquet(out / "joints.parquet", frames,
                         {h: {**hands[h], "reproj_px": raw_hands[h]["reproj_px"]} for h in HANDS})

    inset = None
    if world:
        from scipy.spatial.transform import Rotation
        from hands.world import camera_poses, joints_to_world, load_poses
        t_wc, q_wc, pose_valid = camera_poses(load_poses(traj_file), imu_calib, frames["timestamp_us"])
        write_pose_parquet(out / "pose.parquet", frames, t_wc, q_wc, pose_valid)
        write_joints_parquet(out / "joints_world.parquet", frames,
                             {h: {**hands[h], "reproj_px": raw_hands[h]["reproj_px"],
                                  "joints": joints_to_world(hands[h]["joints"], t_wc, q_wc, pose_valid)}
                              for h in HANDS}, pose_valid=pose_valid)
        heading = np.full((T, 2), np.nan)  # camera +z (forward) in the world, x/y components
        if pose_valid.any():
            heading[pose_valid] = Rotation.from_quat(q_wc[pose_valid]).apply([0.0, 0.0, 1.0])[:, :2]
        inset = (t_wc[:, :2], heading, inset_bounds(t_wc[:, :2]))
        log(f"world: pose coverage {pose_valid.mean():.3f}")

    # Pass 2: review video of the output joints.
    reader = PartReader(left_mp4, right_mp4, image_size, cts)
    with ReviewWriter(out / "review.mp4", size=REVIEW_SIZE, fps=FPS) as writer:
        for t, (_, img_l, img_r) in enumerate(reader.frames(pairs)):
            draw = {}
            for hand, h in hands.items():
                if h["valid"][t]:
                    uv_l, uv_r = project_rectified(rect, h["joints"][t])
                    draw[hand] = (uv_l, uv_r, h["conf"][t])
            frame_inset = None if inset is None else (inset[0][:t + 1], inset[1][t], inset[2])
            writer.write(rect.rectify(img_l, "left"), rect.rectify(img_r, "right"), draw, frame_inset)
    reader.close()
    run_seconds = time.perf_counter() - t_run

    meta = {"recording": rec.name, "part": part, "start_s": start_s, "duration_s": duration_s,
            "calibration": calib_hits[0].name, "n_frames": T, "fps_video": FPS, "detection_fps": detection_fps,
            "frames_left": [pairs[0].frame_left, pairs[-1].frame_left], "mano": bool(mano),
            "crop": bool(crop), "world": bool(world)}

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
    if world:
        odo = out / "odometry.json"
        odometry = json.loads(odo.read_text(encoding="utf-8")) if odo.is_file() else None
        report = add_world_section(report, t_wc, pose_valid, hands, odometry, FPS, traj_file.name)
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
    p.add_argument("--crop", action="store_true",
                   help="detect on a square crop around each previous-frame hand (default: full frame)")
    p.add_argument("--mano-dir", default=None, help="MANO model folder (default: env MANO_DIR)")
    p.add_argument("--world", action="store_true",
                   help="also write pose.parquet and joints_world.parquet (ORB-SLAM3 camera poses)")
    p.add_argument("--trajectory", default=None,
                   help="EuRoC trajectory for --world (default: OUT/trajectory.txt, else run ORB-SLAM3)")
    a = p.parse_args(argv)
    run_pipeline(a.recording_dir, a.out, part=a.part, start_s=a.start, duration_s=a.duration,
                 mano=a.mano, mano_dir=a.mano_dir, crop=a.crop, world=a.world, trajectory_path=a.trajectory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
