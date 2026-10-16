"""IMU csv loading, Kalibr IMU calibration, time shift, EuRoC export."""
from __future__ import annotations

import time

import numpy as np
import pytest

from hands.conftest import CALIB_YAML
from hands.imu import (
    EUROC_HEADER,
    camera_to_imu_time,
    gravity_check,
    imu_to_camera_time,
    load_imu,
    load_imu_calibration,
    write_euroc_imu,
)
from hands.timing import load_pts

IMU_YAML = CALIB_YAML.parent / "calibration_imu.yaml"
HEADER = "timestamp_us\t,x\t,y\t,z\t,type\n"


def _synthetic_csv(path):
    rows = [
        "300,0.1,-8.4,-4.3,accel", "300,0.01,0.02,-0.1,gyro",
        "100,1.0,2.0,3.0,accel", "100,0.4,0.5,0.6,gyro",  # out of order
        "200,7.0,8.0,9.0,accel",  # gyro row missing
    ]
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_load_imu_merges_sorts_and_drops_incomplete(tmp_path):
    t, gyro, accel = load_imu(_synthetic_csv(tmp_path / "imu.csv"))
    assert t.dtype == np.int64 and t.tolist() == [100, 300]
    np.testing.assert_array_equal(accel, [[1.0, 2.0, 3.0], [0.1, -8.4, -4.3]])
    np.testing.assert_array_equal(gyro, [[0.4, 0.5, 0.6], [0.01, 0.02, -0.1]])


def test_load_imu_window(tmp_path):
    p = _synthetic_csv(tmp_path / "imu.csv")
    assert load_imu(p, t0_us=150)[0].tolist() == [300]
    assert load_imu(p, t1_us=299)[0].tolist() == [100]
    assert load_imu(p, t0_us=100, t1_us=300)[0].tolist() == [100, 300]
    assert len(load_imu(p, t0_us=301)[0]) == 0


def test_calibration_applied_matches_hand_computation(tmp_path):
    calib = load_imu_calibration(IMU_YAML)
    t, gyro, accel = load_imu(_synthetic_csv(tmp_path / "imu.csv"), calib=calib)
    raw_g = np.array([[0.4, 0.5, 0.6], [0.01, 0.02, -0.1]])
    raw_a = np.array([[1.0, 2.0, 3.0], [0.1, -8.4, -4.3]])
    for i in range(2):
        np.testing.assert_allclose(accel[i], np.linalg.solve(calib.M_acc, raw_a[i] - calib.acc_bias), rtol=0, atol=1e-12)
        np.testing.assert_allclose(gyro[i], np.linalg.solve(calib.M_gyr, raw_g[i] - calib.gyr_bias), rtol=0, atol=1e-12)
    assert not np.allclose(accel, raw_a)


def test_load_imu_calibration_real_yaml():
    c = load_imu_calibration(IMU_YAML)
    assert c.T_cam_imu.shape == (4, 4)
    np.testing.assert_allclose(c.T_cam_imu[:3, 3], [0.048, 0.005, -0.02184], rtol=0, atol=1e-12)
    R = c.T_cam_imu[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    np.testing.assert_array_equal(c.T_cam_imu[3], [0, 0, 0, 1])
    assert c.timeshift_cam_imu_s == pytest.approx(-0.012430748059884415, abs=1e-15)
    assert c.update_rate == 1000.0
    assert c.acc_bias.shape == c.gyr_bias.shape == c.gravity.shape == (3,)
    assert c.M_acc.shape == c.M_gyr.shape == (3, 3)
    assert c.acc_noise_density == pytest.approx(1.4131801092262099e-03)
    assert c.gyr_random_walk == pytest.approx(1.7866262500917867e-06)
    assert np.linalg.norm(c.gravity) == pytest.approx(9.81, abs=0.01)


def test_time_conversion_round_trip():
    c = load_imu_calibration(IMU_YAML)
    t_cam = np.array([347_470_640_146, 347_470_673_487], dtype=np.int64)
    t_imu = camera_to_imu_time(t_cam, c)
    assert t_imu.dtype == np.int64
    assert (t_imu - t_cam).tolist() == [-12431, -12431]  # t_imu = t_cam + timeshift
    assert np.array_equal(imu_to_camera_time(t_imu, c), t_cam)


def test_gravity_check():
    a = np.array([[0.0, 0.0, 9.81], [0.0, 9.0, 0.0], [10.0, 0.0, 0.0]])
    assert gravity_check(a) == pytest.approx(9.81)


def test_write_euroc_imu_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    t = np.array([347_466_372_850, 347_466_373_844, 347_466_374_850], dtype=np.int64)
    gyro, accel = rng.normal(size=(3, 3)), rng.normal(size=(3, 3)) * 9.8
    p = tmp_path / "data.csv"
    write_euroc_imu(p, t, gyro, accel)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == EUROC_HEADER
    assert lines[0].startswith("#timestamp [ns],w_RS_S_x [rad s^-1]")
    assert [int(line.split(",")[0]) for line in lines[1:]] == (t * 1000).tolist()
    vals = np.array([[float(v) for v in line.split(",")[1:]] for line in lines[1:]])
    np.testing.assert_array_equal(vals[:, :3], gyro)
    np.testing.assert_array_equal(vals[:, 3:], accel)


@pytest.mark.real_footage
def test_real_window_600_720s(recording_ev):
    name = recording_ev.name
    t_first = int(load_pts(recording_ev / f"{name}_camera_left_part0001_pts.csv")[0])
    t0, t1 = t_first + 600_000_000, t_first + 720_000_000
    calib = load_imu_calibration(recording_ev / f"{name}_calibration_imu.yaml")
    start = time.perf_counter()
    t, gyro, accel = load_imu(recording_ev / f"{name}_imu.csv", t0, t1, calib=calib)
    elapsed = time.perf_counter() - start
    raw_norm = gravity_check(load_imu(recording_ev / f"{name}_imu.csv", t0, t0 + 10_000_000)[2])
    print(f"rows={len(t)} load_s={elapsed:.2f} median_dt={np.median(np.diff(t))} "
          f"accel_norm raw(10s)={raw_norm:.4f} calibrated={gravity_check(accel):.4f}")
    assert elapsed < 20.0
    assert 110_000 <= len(t) <= 130_000
    assert t[0] >= t0 and t[-1] <= t1
    assert np.all(np.diff(t) > 0)
    assert abs(np.median(np.diff(t)) - 1000) <= 50
    assert abs(gravity_check(accel) - 9.81) <= 0.3
    assert np.isfinite(gyro).all() and np.isfinite(accel).all()
