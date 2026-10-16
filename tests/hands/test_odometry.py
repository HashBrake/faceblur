"""ORB-SLAM3 export, settings writer and trajectory reader (hands.odometry)."""
from __future__ import annotations

import re

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hands.calib import load_calibration
from hands.conftest import CALIB_YAML
from hands.imu import load_imu_calibration
from hands.odometry.export import export_window, write_settings
from hands.odometry.run_orbslam3 import gravity_angle_deg, load_trajectory, trajectory_length
from hands.timing import load_pts

IMU_YAML = CALIB_YAML.parent / "calibration_imu.yaml"


def _text_matrix(text, name):
    m = re.search(rf"{re.escape(name)}: !!opencv-matrix.*?data: \[([^\]]*)\]", text, re.S)
    return np.array([float(v) for v in m.group(1).split(",")]).reshape(4, 4)


def test_write_settings_matches_calibration(tmp_path):
    out = write_settings(CALIB_YAML, IMU_YAML, (1600, 1300), 30, tmp_path / "settings.yaml")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("%YAML:1.0\n")
    rig = load_calibration(CALIB_YAML)
    imu = load_imu_calibration(IMU_YAML)

    T_rl = np.eye(4)
    T_rl[:3, :3], T_rl[:3, 3] = rig.R, rig.t
    T_c1_c2 = _text_matrix(text, "Stereo.T_c1_c2")
    np.testing.assert_allclose(T_c1_c2, np.linalg.inv(T_rl), atol=1e-9)
    assert abs(T_c1_c2[0, 3] - 0.1202) < 0.001
    assert abs(T_c1_c2[1, 3]) < 0.01 and abs(T_c1_c2[2, 3]) < 0.01

    T_b_c1 = _text_matrix(text, "IMU.T_b_c1")
    np.testing.assert_allclose(T_b_c1, np.linalg.inv(imu.T_cam_imu), atol=1e-9)
    R, t = imu.T_cam_imu[:3, :3], imu.T_cam_imu[:3, 3]
    np.testing.assert_allclose(T_b_c1[:3, 3], -R.T @ t, atol=1e-9)

    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_READ)
    assert fs.isOpened()
    assert fs.getNode("Camera.type").string() == "KannalaBrandt8"
    for j in range(4):
        assert fs.getNode(f"Camera1.k{j + 1}").isReal()
        assert fs.getNode(f"Camera1.k{j + 1}").real() == pytest.approx(rig.left.k[j], abs=1e-15)
        assert fs.getNode(f"Camera2.k{j + 1}").real() == pytest.approx(rig.right.k[j], abs=1e-15)
    assert fs.getNode("Camera1.fx").real() == pytest.approx(rig.left.fx)
    assert fs.getNode("IMU.Frequency").isReal() and fs.getNode("IMU.Frequency").real() == 1000.0
    for key in ("Camera.fps", "Camera.width", "Camera.height", "Camera.RGB", "ORBextractor.nFeatures",
                "Camera1.overlappingEnd"):
        assert fs.getNode(key).isInt(), key
    assert fs.getNode("Camera.width").real() == 1600 and fs.getNode("Camera1.overlappingEnd").real() == 1599
    assert fs.getNode("ORBextractor.nFeatures").real() == 1500
    for key in ("IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk", "Stereo.ThDepth"):
        assert fs.getNode(key).isReal(), key
    assert fs.getNode("IMU.GyroWalk").real() == pytest.approx(imu.gyr_random_walk)
    m = fs.getNode("Stereo.T_c1_c2").mat()
    assert m.shape == (4, 4) and m.dtype == np.float32  # ORB-SLAM3 reads it with at<float>
    np.testing.assert_allclose(m, T_c1_c2, atol=1e-6)
    fs.release()


def test_load_trajectory_round_trip(tmp_path):
    t_ns = np.array([345600123456000, 345600156789000, 345600190123000], dtype=np.int64)
    pos = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.3, 0.4, 1.0]])
    quat = Rotation.from_euler("xyz", [[0, 0, 0], [10, 0, 0], [0, 20, 30]], degrees=True).as_quat()
    lines = [f"{t:.6f} " + " ".join(f"{v:.9f}" for v in (*p, *q)) for t, p, q in zip(t_ns.astype(float), pos, quat)]
    path = tmp_path / "f_traj.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    t_us, t_wb, q_wb = load_trajectory(path)
    assert t_us.dtype == np.int64 and t_us.tolist() == (t_ns // 1000).tolist()
    np.testing.assert_allclose(t_wb, pos, atol=1e-9)
    np.testing.assert_allclose(q_wb, quat, atol=1e-9)
    assert trajectory_length(t_wb) == pytest.approx(1.5)


def test_gravity_angle_of_tilted_world():
    t_us = np.array([0, 1000, 2000], dtype=np.int64)
    up_body = np.array([0.0, 0.0, 9.81])
    # Body tilted 10 deg about x relative to a frame where gravity is along world z.
    q = np.tile(Rotation.from_euler("x", 10, degrees=True).as_quat(), (3, 1))
    t_imu = np.arange(0, 2001, 100, dtype=np.int64)
    acc = np.tile(up_body, (len(t_imu), 1))
    assert gravity_angle_deg(t_us, q, t_imu, acc) == pytest.approx(10.0, abs=1e-6)
    acc_level = Rotation.from_quat(q[0]).inv().apply(np.tile(up_body, (len(t_imu), 1)))
    assert gravity_angle_deg(t_us, q, t_imu, acc_level) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.real_footage
def test_export_window_real_2s(recording_ev, tmp_path):
    name = recording_ev.name
    exp = export_window(recording_ev, 1, 600, 2, tmp_path)
    assert b"\r" not in exp.times.read_bytes()
    times = np.loadtxt(exp.times, dtype=np.int64, ndmin=1)
    assert 58 <= exp.n_pairs <= 60 and len(times) == exp.n_pairs
    assert np.all(np.diff(times) > 0)
    assert sorted(p.stem for p in exp.cam0.glob("*.png")) == sorted(str(t) for t in times)
    assert sorted(p.stem for p in exp.cam1.glob("*.png")) == sorted(str(t) for t in times)
    img = cv2.imread(str(exp.cam0 / f"{times[0]}.png"), cv2.IMREAD_UNCHANGED)
    assert img.shape == (1300, 1600) and img.dtype == np.uint8

    # Image times = left csv timestamps + Kalibr shift (us -> ns).
    calib = load_imu_calibration(recording_ev / f"{name}_calibration_imu.yaml")
    shift_us = round(calib.timeshift_cam_imu_s * 1e6)
    pts_l = load_pts(recording_ev / f"{name}_camera_left_part0001_pts.csv")
    t_cam_us = times // 1000 - shift_us
    assert np.all(times % 1000 == 0)
    assert np.isin(t_cam_us, pts_l).all()
    lo = pts_l[0] + 600_000_000
    assert np.all((t_cam_us >= lo) & (t_cam_us < lo + 2_000_000))

    imu = np.loadtxt(exp.imu, delimiter=",", comments="#", ndmin=2)
    t_imu_ns = imu[:, 0].astype(np.int64)
    assert imu.shape[1] == 7 and np.all(np.diff(t_imu_ns) > 0)
    assert t_imu_ns[0] <= times[0] - 1_000_000_000 + 2_000_000
    assert t_imu_ns[-1] >= times[-1] + 1_000_000_000 - 2_000_000
    assert t_imu_ns[0] >= times[0] - 1_000_000_000 and t_imu_ns[-1] <= times[-1] + 1_000_000_000
    assert np.median(np.linalg.norm(imu[:, 4:7], axis=1)) == pytest.approx(9.81, abs=0.1)
