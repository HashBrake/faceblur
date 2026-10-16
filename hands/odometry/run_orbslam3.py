"""Run ORB-SLAM3 stereo-inertial on an exported window and read the trajectory.

ORB-SLAM3 (GPLv3) runs as a separate executable in the Docker image built from
``hands/odometry/Dockerfile`` (``docker build -t faceblur-orbslam3:4452a3c
hands/odometry``). Its EuRoC trajectory gives the pose of the IMU/body frame in
a gravity-aligned world (z up) for every tracked frame of the biggest map (the
build patches out ORB-SLAM3's re-anchoring of that file to the first keyframe).

    python -m hands.odometry.run_orbslam3 <recording_dir> -o <out_dir> --part 1 --start 600 --duration 120
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

IMAGE = "faceblur-orbslam3:4452a3c"
ORB = "/opt/orbslam3/ORB_SLAM3"
DRIVER = f"{ORB}/Examples/Stereo-Inertial/stereo_inertial_kb"
VOCAB = f"{ORB}/Vocabulary/ORBvoc.txt"


def run_orbslam3(export_dir, settings_path, out_dir, name="traj") -> Path:
    """Run the driver on ``export_dir`` (cam0/, cam1/, times.txt, imu.csv); returns ``out_dir/trajectory.txt``.

    stdout+stderr go to ``out_dir/orbslam3.log``.
    """
    export_dir, settings, out = Path(export_dir).resolve(), Path(settings_path).resolve(), Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm",
           "-v", f"{export_dir}:/export:ro", "-v", f"{settings.parent}:/settings:ro", "-v", f"{out}:/out",
           "-w", "/out", IMAGE, DRIVER, VOCAB, f"/settings/{settings.name}",
           "/export/cam0", "/export/cam1", "/export/times.txt", "/export/imu.csv", name]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    (out / "orbslam3.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    f = out / f"f_{name}.txt"
    if proc.returncode != 0 or not f.is_file():
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-30:])
        raise RuntimeError(f"ORB-SLAM3 failed (exit {proc.returncode}):\n{tail}")
    traj = out / "trajectory.txt"
    f.replace(traj)
    return traj


def load_trajectory(path):
    """(t_us int64 (N,), t_wb (N,3), q_wb (N,4) xyzw) from an EuRoC-format ORB-SLAM3 trajectory.

    Lines: ``t_ns tx ty tz qx qy qz qw``; t_ns is on the IMU clock and is rounded to us.
    """
    d = np.loadtxt(path, dtype=np.float64, ndmin=2, comments="#")
    if d.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 4))
    t_us = np.round(d[:, 0] / 1000.0).astype(np.int64)
    return t_us, d[:, 1:4], d[:, 4:8]


def trajectory_length(t_wb) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(t_wb), axis=0), axis=1).sum())


def gravity_angle_deg(t_us, q_wb, t_imu_us, accel) -> float:
    """Angle between world +z and the mean specific force rotated into the world.

    Each IMU sample inside the trajectory's time span is rotated with the pose
    nearest in time; at rest the specific force points up, so a gravity-aligned
    z-up world gives ~0 deg.
    """
    t_us, t_imu_us = np.asarray(t_us), np.asarray(t_imu_us)
    keep = (t_imu_us >= t_us[0]) & (t_imu_us <= t_us[-1])
    ti, acc = t_imu_us[keep], np.asarray(accel)[keep]
    j = np.clip(np.searchsorted(t_us, ti), 1, len(t_us) - 1)
    j = np.where(np.abs(ti - t_us[j - 1]) <= np.abs(t_us[j] - ti), j - 1, j)
    a_w = Rotation.from_quat(np.asarray(q_wb)[j]).apply(acc).mean(axis=0)
    return float(np.degrees(np.arccos(np.clip(a_w[2] / np.linalg.norm(a_w), -1.0, 1.0))))


def _load_euroc_imu(path):
    d = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    return np.round(d[:, 0] / 1000.0).astype(np.int64), d[:, 1:4], d[:, 4:7]


def main(argv=None) -> dict:
    from hands.odometry.export import _one, export_window, write_settings
    from hands.calib import load_calibration

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, required=True)
    args = ap.parse_args(argv)
    rec, out = Path(args.recording_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    exp = export_window(rec, args.part, args.start, args.duration, out / "export")
    export_s = time.perf_counter() - t0
    cam_yaml, imu_yaml = _one(rec, "*_calibration_camera.yaml"), _one(rec, "*_calibration_imu.yaml")
    rig = load_calibration(cam_yaml)
    settings = write_settings(cam_yaml, imu_yaml, (rig.left.width, rig.left.height), 30, out / "settings.yaml")

    t0 = time.perf_counter()
    traj = run_orbslam3(exp.root, settings, out)
    slam_s = time.perf_counter() - t0

    t_us, t_wb, q_wb = load_trajectory(traj)
    t_imu, _, accel = _load_euroc_imu(exp.imu)
    log = (out / "orbslam3.log").read_text(encoding="utf-8")
    lost = re.search(r"LOST_EVENTS (\d+)", log)
    maps = re.search(r"There are (\d+) maps in the atlas", log)
    report = {
        "frames_exported": exp.n_pairs, "frames_skipped": exp.n_skipped,
        "frames_with_pose": int(len(t_us)),
        "lost_events": int(lost.group(1)) if lost else None,
        "maps_in_atlas": int(maps.group(1)) if maps else None,
        "map_reinitialisations": log.count("New Map created with") - 1,
        "resets_on_tracking_failure": log.count("Reseting active map"),
        "track_local_map_failures": log.count("Fail to track local map!"),
        "trajectory_length_m": trajectory_length(t_wb),
        "gravity_angle_deg": gravity_angle_deg(t_us, q_wb, t_imu, accel) if len(t_us) > 1 else None,
        "export_s": export_s, "orbslam3_s": slam_s,
    }
    (out / "odometry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
