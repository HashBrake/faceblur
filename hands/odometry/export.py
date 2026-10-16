"""Export a recording window in TUM-VI layout and write ORB-SLAM3 settings.

Images are the ORIGINAL (unrectified) frames as 8-bit grayscale PNG: ORB-SLAM3
models the Kannala-Brandt lens itself. Every timestamp is on the IMU clock in
ns: image times are the left csv timestamps shifted by Kalibr's
``timeshift_cam_imu`` (``t_imu = t_cam + shift``), IMU rows keep their own time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hands.calib import load_calibration
from hands.imu import camera_to_imu_time, load_imu, load_imu_calibration, write_euroc_imu
from hands.timing import container_timestamps, find_part_files, stereo_frames
from hands.video import PartReader

IMU_PAD_US = 1_000_000


@dataclass
class ExportPaths:
    root: Path
    cam0: Path
    cam1: Path
    times: Path
    imu: Path
    n_pairs: int
    n_skipped: int


def _one(rec: Path, pattern: str) -> Path:
    hits = sorted(rec.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} files match {pattern} in {rec}")
    return hits[0]


def export_window(recording_dir, part, start_s, duration_s, out_dir) -> ExportPaths:
    rec, out = Path(recording_dir), Path(out_dir)
    cam0, cam1 = out / "cam0", out / "cam1"
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)
    left_mp4, right_mp4, _, _ = find_part_files(rec, part)
    rig = load_calibration(_one(rec, "*_calibration_camera.yaml"))
    calib = load_imu_calibration(_one(rec, "*_calibration_imu.yaml"))
    image_size = (rig.left.width, rig.left.height)

    max_frames = int((start_s + duration_s) * 31) + 200
    cts = (container_timestamps(left_mp4, max_frames), container_timestamps(right_mp4, max_frames))
    pairs = stereo_frames(rec, part, start_s, duration_s, container_ts=cts)
    if not pairs:
        raise ValueError("no synchronised stereo frames in the requested range")

    reader = PartReader(left_mp4, right_mp4, image_size, cts)
    times_ns, skipped = [], 0
    try:
        for pair in pairs:
            try:
                _, img_l, img_r = next(reader.frames([pair]))
            except ValueError:
                skipped += 1
                continue
            t_ns = int(camera_to_imu_time(pair.timestamp_us, calib)) * 1000
            cv2.imwrite(str(cam0 / f"{t_ns}.png"), cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY))
            cv2.imwrite(str(cam1 / f"{t_ns}.png"), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))
            times_ns.append(t_ns)
    finally:
        reader.close()
    if not times_ns:
        raise ValueError("no stereo pair could be decoded")

    times = out / "times.txt"
    times.write_bytes("".join(f"{t}\n" for t in times_ns).encode())  # LF: the driver does not strip "\r"
    t0_us, t1_us = times_ns[0] // 1000 - IMU_PAD_US, times_ns[-1] // 1000 + IMU_PAD_US
    t_imu, gyro, accel = load_imu(_one(rec, "*_imu.csv"), t0_us, t1_us, calib=calib)
    imu = out / "imu.csv"
    write_euroc_imu(imu, t_imu, gyro, accel)
    return ExportPaths(out, cam0, cam1, times, imu, len(times_ns), skipped)


def _real(x) -> str:
    """A float that OpenCV's YAML reader types as real (ORB-SLAM3 checks isReal())."""
    s = repr(float(x))
    return s if "." in s else s.replace("e", ".0e")


def _matrix(name: str, T: np.ndarray) -> str:
    # dt f: ORB-SLAM3 reads these with cv::Mat::at<float>; values keep full precision in the text.
    data = ", ".join(_real(v) for v in np.asarray(T, dtype=np.float64).ravel())
    return f"{name}: !!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [{data}]\n"


def write_settings(calib_camera_yaml, calib_imu_yaml, image_size, fps, out_path) -> Path:
    """ORB-SLAM3 stereo-inertial KannalaBrandt8 settings (File.version 1.0) from our calibration."""
    rig = load_calibration(calib_camera_yaml)
    imu = load_imu_calibration(calib_imu_yaml)
    w, h = int(image_size[0]), int(image_size[1])
    T_c2_c1 = np.eye(4)  # X_right = R X_left + t
    T_c2_c1[:3, :3], T_c2_c1[:3, 3] = rig.R, rig.t
    T_c1_c2 = np.linalg.inv(T_c2_c1)  # the yaml rotations are orthonormal only to ~1e-8
    T_b_c1 = np.linalg.inv(imu.T_cam_imu)

    lines = ["%YAML:1.0", 'File.version: "1.0"', 'Camera.type: "KannalaBrandt8"']
    for i, cam in ((1, rig.left), (2, rig.right)):
        lines += [f"Camera{i}.fx: {_real(cam.fx)}", f"Camera{i}.fy: {_real(cam.fy)}",
                  f"Camera{i}.cx: {_real(cam.cx)}", f"Camera{i}.cy: {_real(cam.cy)}"]
        lines += [f"Camera{i}.k{j + 1}: {_real(cam.k[j])}" for j in range(4)]
        lines += [f"Camera{i}.overlappingBegin: 0", f"Camera{i}.overlappingEnd: {w - 1}"]
    lines += [f"Camera.width: {w}", f"Camera.height: {h}", f"Camera.fps: {int(round(fps))}",
              "Camera.RGB: 1", "Stereo.ThDepth: 40.0"]
    text = "\n".join(lines) + "\n" + _matrix("Stereo.T_c1_c2", T_c1_c2) + _matrix("IMU.T_b_c1", T_b_c1)
    text += "\n".join([
        f"IMU.NoiseGyro: {_real(imu.gyr_noise_density)}",
        f"IMU.NoiseAcc: {_real(imu.acc_noise_density)}",
        f"IMU.GyroWalk: {_real(imu.gyr_random_walk)}",
        f"IMU.AccWalk: {_real(imu.acc_random_walk)}",
        f"IMU.Frequency: {_real(imu.update_rate)}",
        "ORBextractor.nFeatures: 1500",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20",
        "ORBextractor.minThFAST: 7",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -3.5",
        "Viewer.ViewpointF: 500.0",
    ]) + "\n"
    out = Path(out_path)
    out.write_text(text, encoding="utf-8")
    return out
