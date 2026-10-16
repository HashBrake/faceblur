"""Tests for hands.world: trajectory interpolation, camera poses, joints in world coordinates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hands.imu import camera_to_imu_time, load_imu_calibration
from hands.world import Trajectory, camera_poses, interpolate_poses, joints_to_world, load_poses

IMU_YAML = Path(__file__).parent / "data" / "calibration_imu.yaml"
DT_US = 33_000


def circle_trajectory(n=20, t0=1_000_000, radius=1.0):
    """n poses on a circle (z = 1.6 m), yaw along the tangent, DT_US apart."""
    t = t0 + DT_US * np.arange(n, dtype=np.int64)
    a = np.linspace(0, np.pi, n)
    pos = np.column_stack([radius * np.cos(a), radius * np.sin(a), np.full(n, 1.6)])
    rot = Rotation.from_euler("zyx", np.column_stack([a + np.pi / 2, 0.1 * np.sin(a), 0.05 * a]))
    return Trajectory(t, pos, rot)


def write_euroc(path, traj: Trajectory):
    q = traj.rotations.as_quat()
    lines = [f"{t * 1000}.000000 " + " ".join(f"{v:.12f}" for v in (*p, *qq))
             for t, p, qq in zip(traj.t_us, traj.positions, q)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_poses_roundtrip(tmp_path):
    traj = circle_trajectory()
    write_euroc(tmp_path / "trajectory.txt", traj)
    got = load_poses(tmp_path / "trajectory.txt")
    assert got.t_us.dtype == np.int64 and np.array_equal(got.t_us, traj.t_us)
    assert np.allclose(got.positions, traj.positions, atol=1e-11)
    assert np.allclose((got.rotations.inv() * traj.rotations).magnitude(), 0, atol=1e-9)


def test_world_joints_at_pose_timestamps():
    """(a) world joints == R_wb R_bc X + R_wb t_bc + t_wb at the pose timestamps."""
    calib = load_imu_calibration(IMU_YAML)
    traj = circle_trajectory()
    shift = int(camera_to_imu_time(0, calib))
    t_cam = traj.t_us - shift
    rng = np.random.default_rng(3)
    joints = rng.uniform(-0.2, 0.2, (len(t_cam), 21, 3)) + [0.0, 0.1, 0.45]
    t_wc, q_wc, valid = camera_poses(traj, calib, t_cam)
    assert valid.all()
    world = joints_to_world(joints, t_wc, q_wc, valid)
    T_bc = np.linalg.inv(calib.T_cam_imu)
    R_bc, t_bc = T_bc[:3, :3], T_bc[:3, 3]
    for k in range(len(t_cam)):
        R_wb = traj.rotations[k].as_matrix()
        expect = joints[k] @ (R_wb @ R_bc).T + R_wb @ t_bc + traj.positions[k]
        assert np.abs(world[k] - expect).max() <= 1e-9


def test_interpolation_midpoints_and_gaps():
    """(b) mid-timestamps: midpoint position, slerp midpoint rotation; outside span / hole -> invalid."""
    traj = circle_trajectory()
    mid = traj.t_us[:-1] + DT_US // 2
    pos, quat, valid = interpolate_poses(traj, mid)
    assert valid.all()
    assert np.abs(pos - (traj.positions[:-1] + traj.positions[1:]) / 2).max() <= 1e-12
    slerp = Slerp(traj.t_us.astype(np.float64), traj.rotations)
    expect = slerp(mid.astype(np.float64))
    assert (Rotation.from_quat(quat).inv() * expect).magnitude().max() <= 1e-9
    # exact pose timestamps and the span ends are valid
    _, q_at, v_at = interpolate_poses(traj, traj.t_us)
    assert v_at.all() and (Rotation.from_quat(q_at).inv() * traj.rotations).magnitude().max() <= 1e-9

    outside = np.array([traj.t_us[0] - 10_000, traj.t_us[-1] + 10_000])
    p_out, q_out, v_out = interpolate_poses(traj, outside)
    assert not v_out.any() and np.isnan(p_out).all() and np.isnan(q_out).all()

    keep = np.ones(len(traj.t_us), bool)
    keep[8:14] = False  # poses 7 and 14 are 7 * 33 ms = 231 ms apart
    holed = Trajectory(traj.t_us[keep], traj.positions[keep], traj.rotations[keep])
    gap_mid = (traj.t_us[7] + traj.t_us[14]) // 2
    near_edge = traj.t_us[7] + 20_000
    _, _, v_gap = interpolate_poses(holed, np.array([gap_mid, near_edge]))
    assert v_gap.tolist() == [False, True]
    _, _, v_empty = interpolate_poses(Trajectory(np.zeros(0, np.int64), np.zeros((0, 3)), None), np.array([0]))
    assert not v_empty.any()


def test_joints_to_world_nan_where_invalid():
    joints = np.ones((3, 21, 3))
    t_wc = np.array([[1.0, 2.0, 3.0], [np.nan] * 3, [0.0, 0.0, 0.0]])
    q_wc = np.array([[0, 0, 0, 1.0], [np.nan] * 4, [0, 0, 0, 1.0]])
    valid = np.array([True, False, True])
    w = joints_to_world(joints, t_wc, q_wc, valid)
    assert np.allclose(w[0], [2.0, 3.0, 4.0]) and np.isnan(w[1]).all() and np.allclose(w[2], 1.0)
