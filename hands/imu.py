"""IMU samples and IMU calibration of an Orbbec Ego recording.

``*_imu.csv``: header ``timestamp_us\t,x\t,y\t,z\t,type`` (tabs and commas
mixed), then plain comma rows that alternate ``accel`` (specific force, m/s^2)
and ``gyro`` (rad/s) with the same timestamp, about 1000 Hz, sorted by time.
The timestamps are on the camera pts clock (same us epoch as ``*_pts.csv``).
The file is ~500 MB per recording, so it is streamed in blocks and reading
stops once past the requested window.

``*_calibration_imu.yaml`` is Kalibr-style: ``cam0.T_cam_imu`` has its
translation in millimetres (converted to metres here), ``timeshift_cam_imu`` is
in seconds, and the ``misalignment`` IMU model reads measured = M * true + bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import yaml

_COLUMNS = ["timestamp_us", "x", "y", "z", "type"]
_TYPES = {"timestamp_us": pa.int64(), "x": pa.float64(), "y": pa.float64(), "z": pa.float64(), "type": pa.string()}
EUROC_HEADER = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")


@dataclass
class ImuCalibration:
    T_cam_imu: np.ndarray  # (4, 4), translation in metres
    timeshift_cam_imu_s: float
    acc_bias: np.ndarray
    gyr_bias: np.ndarray
    M_acc: np.ndarray
    M_gyr: np.ndarray
    acc_noise_density: float
    acc_random_walk: float
    gyr_noise_density: float
    gyr_random_walk: float
    update_rate: float
    gravity: np.ndarray


def load_imu_calibration(yaml_path) -> ImuCalibration:
    d = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    cam, imu = d["cam0"], d["imu0"]
    T = np.array(cam["T_cam_imu"], dtype=np.float64)
    T[:3, 3] /= 1000.0  # mm -> m
    arr = lambda v: np.array(v, dtype=np.float64)  # noqa: E731
    return ImuCalibration(
        T_cam_imu=T,
        timeshift_cam_imu_s=float(cam["timeshift_cam_imu"]),
        acc_bias=arr(imu["AccBias"]),
        gyr_bias=arr(imu["GyrBias"]),
        M_acc=arr(imu["M_acc"]),
        M_gyr=arr(imu["M_gyr"]),
        acc_noise_density=float(imu["accelerometer"]["noise_density"]),
        acc_random_walk=float(imu["accelerometer"]["random_walk"]),
        gyr_noise_density=float(imu["gyroscope"]["noise_density"]),
        gyr_random_walk=float(imu["gyroscope"]["random_walk"]),
        update_rate=float(imu["update_rate"]),
        gravity=arr(d["gravity"]),
    )


def load_imu(csv_path, t0_us=None, t1_us=None, calib=None):
    """(t_us (N,) int64, gyro (N,3), accel (N,3)), one row per timestamp with both samples.

    Timestamps with only one of accel/gyro are dropped; rows are sorted and kept
    when ``t0_us <= t <= t1_us``. With ``calib`` both sensors are corrected as
    ``true = M^-1 (measured - bias)``.
    """
    reader = pacsv.open_csv(
        str(csv_path),
        read_options=pacsv.ReadOptions(skip_rows=1, column_names=_COLUMNS, block_size=8 << 20),
        convert_options=pacsv.ConvertOptions(column_types=_TYPES),
    )
    kept = []
    for batch in reader:
        if batch.num_rows == 0:
            continue
        t = batch.column(0)
        if t1_us is not None and pc.min(t).as_py() > t1_us:
            break  # the file is sorted: nothing later can be in the window
        mask = None
        if t0_us is not None:
            mask = pc.greater_equal(t, t0_us)
        if t1_us is not None:
            m1 = pc.less_equal(t, t1_us)
            mask = m1 if mask is None else pc.and_(mask, m1)
        kept.append(batch if mask is None else batch.filter(mask))
    table = pa.Table.from_batches(kept, schema=reader.schema)

    def sensor(name):
        sub = table.filter(pc.equal(table.column("type"), name))
        ts = sub.column("timestamp_us").to_numpy()
        xyz = np.column_stack([sub.column(c).to_numpy() for c in ("x", "y", "z")]).astype(np.float64)
        return ts, xyz.reshape(-1, 3)

    ta, acc = sensor("accel")
    tg, gyr = sensor("gyro")
    t, ia, ig = np.intersect1d(ta, tg, return_indices=True)  # sorted, unique
    acc, gyr = acc[ia], gyr[ig]
    if calib is not None:
        acc = np.linalg.solve(calib.M_acc, (acc - calib.acc_bias).T).T
        gyr = np.linalg.solve(calib.M_gyr, (gyr - calib.gyr_bias).T).T
    return t.astype(np.int64), gyr, acc


def camera_to_imu_time(t_cam_us, calib: ImuCalibration):
    """Kalibr convention: ``t_imu = t_cam + timeshift_cam_imu`` (int64 us)."""
    return np.asarray(t_cam_us, dtype=np.int64) + np.int64(round(calib.timeshift_cam_imu_s * 1e6))


def imu_to_camera_time(t_imu_us, calib: ImuCalibration):
    """Kalibr convention: ``t_cam = t_imu - timeshift_cam_imu`` (int64 us)."""
    return np.asarray(t_imu_us, dtype=np.int64) - np.int64(round(calib.timeshift_cam_imu_s * 1e6))


def gravity_check(accel) -> float:
    """Median accelerometer norm; ~9.81 m/s^2 for a correctly scaled IMU."""
    return float(np.median(np.linalg.norm(np.asarray(accel, dtype=np.float64), axis=1)))


def write_euroc_imu(path, t_us, gyro, accel) -> None:
    """EuRoC ``imu0/data.csv``: ns timestamp, gyro xyz, accel xyz."""
    t_ns = np.asarray(t_us, dtype=np.int64) * 1000
    g = np.asarray(gyro, dtype=np.float64)
    a = np.asarray(accel, dtype=np.float64)
    lines = [EUROC_HEADER]
    for i in range(len(t_ns)):
        lines.append(f"{t_ns[i]}," + ",".join(repr(float(v)) for v in (*g[i], *a[i])))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
