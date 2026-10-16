"""Hands in world coordinates: camera poses from the ORB-SLAM3 trajectory.

The trajectory (EuRoC ``t_ns tx ty tz qx qy qz qw``) is the IMU/body pose in a
gravity-aligned world (z up) on the IMU clock. Frame timestamps are on the
camera clock; ``camera_to_imu_time`` converts them. The left camera pose is
``T_world_cam = T_world_body * T_b_c1`` with ``T_b_c1 = inv(T_cam_imu)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from hands.imu import ImuCalibration, camera_to_imu_time
from hands.odometry.run_orbslam3 import load_trajectory


@dataclass
class Trajectory:
    t_us: np.ndarray        # (N,) int64, IMU clock, sorted
    positions: np.ndarray   # (N, 3) t_wb, metres
    rotations: Rotation     # N rotations R_wb


def load_poses(trajectory_path) -> Trajectory:
    t_us, t_wb, q_wb = load_trajectory(trajectory_path)
    order = np.argsort(t_us, kind="stable")
    return Trajectory(t_us[order], t_wb[order], Rotation.from_quat(q_wb[order]))


def interpolate_poses(traj: Trajectory, t_query_us, max_gap_us: int = 50_000):
    """(t_wb (M,3), q_wb (M,4) xyzw, valid (M,)): linear position, slerp rotation.

    ``valid`` is False outside the trajectory span and when the nearer of the two
    bracketing poses is more than ``max_gap_us`` from the query; those rows are NaN.
    """
    tq = np.asarray(t_query_us, np.int64).reshape(-1)
    M, t = len(tq), traj.t_us
    pos, quat, valid = np.full((M, 3), np.nan), np.full((M, 4), np.nan), np.zeros(M, bool)
    if len(t) == 0:
        return pos, quat, valid
    inside = (tq >= t[0]) & (tq <= t[-1])
    j1 = np.clip(np.searchsorted(t, tq, side="left"), 0, len(t) - 1)
    j0 = np.where(t[j1] == tq, j1, np.maximum(j1 - 1, 0))
    nearest = np.minimum(np.abs(tq - t[j0]), np.abs(t[j1] - tq))
    valid = inside & (nearest <= max_gap_us)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return pos, quat, valid
    a, b = j0[idx], j1[idx]
    span = (t[b] - t[a]).astype(np.float64)
    alpha = np.divide((tq[idx] - t[a]).astype(np.float64), span, out=np.zeros(idx.size), where=span > 0)
    pos[idx] = traj.positions[a] + alpha[:, None] * (traj.positions[b] - traj.positions[a])
    r0, r1 = traj.rotations[a], traj.rotations[b]
    rel = (r0.inv() * r1).as_rotvec()
    quat[idx] = (r0 * Rotation.from_rotvec(alpha[:, None] * rel)).as_quat()
    return pos, quat, valid


def camera_poses(traj: Trajectory, calib: ImuCalibration, timestamp_cam_us, max_gap_us: int = 50_000):
    """(t_wc (M,3), q_wc (M,4) xyzw, valid) of the left camera for camera-clock frame timestamps."""
    t_wb, q_wb, valid = interpolate_poses(traj, camera_to_imu_time(timestamp_cam_us, calib), max_gap_us)
    T_bc = np.linalg.inv(calib.T_cam_imu)
    R_bc, t_bc = T_bc[:3, :3], T_bc[:3, 3]
    t_wc, q_wc = np.full_like(t_wb, np.nan), np.full_like(q_wb, np.nan)
    if valid.any():
        R_wb = Rotation.from_quat(q_wb[valid])
        t_wc[valid] = R_wb.apply(t_bc) + t_wb[valid]
        q_wc[valid] = (R_wb * Rotation.from_matrix(R_bc)).as_quat()
    return t_wc, q_wc, valid


def joints_to_world(joints_cam, t_wc, q_wc, valid) -> np.ndarray:
    """(T,21,3) camera-frame joints -> world: ``X_w = R_wc X_c + t_wc``; NaN where ``valid`` is False."""
    joints_cam = np.asarray(joints_cam, np.float64)
    valid = np.asarray(valid, bool)
    out = np.full_like(joints_cam, np.nan)
    if valid.any():
        R = Rotation.from_quat(np.asarray(q_wc)[valid]).as_matrix()
        out[valid] = np.einsum("tij,tkj->tki", R, joints_cam[valid]) + np.asarray(t_wc)[valid][:, None, :]
    return out
