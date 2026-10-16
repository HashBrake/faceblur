"""Stereo calibration: KB4 fisheye camera model and pinhole stereo rectification.

The Orbbec Ego ``*_calibration_camera.yaml`` lists ``cam_0`` / ``IR_L`` (the
reference, identity extrinsics) and ``cam_1`` / ``IR_R``, both with the
Kannala-Brandt model ``distortion_model: "KB"`` (k1..k4; k5, k6, p1, p2 are
zero). KB4 is exactly OpenCV's ``cv2.fisheye`` model, so ``D = [k1, k2, k3, k4]``.

Extrinsics convention (verified): the yaml's ``rotation`` / ``translation`` of
cam_1 map LEFT-camera coordinates to RIGHT-camera coordinates,
``X_right = R @ X_left + t``, with t given in millimetres (here stored in
metres). With t = [-120.17, -0.49, -0.16] mm the right camera centre in the
left frame is ``-R.T @ t`` = +120 mm along x, i.e. IR_R really is to the right
of IR_L, and a point in front of both cameras has positive disparity
u_left - u_right > 0. The opposite reading (``X_left = R @ X_right + t``)
would put IR_R 120 mm to the LEFT, contradicting its name.

Rectification: the rotations come from ``cv2.fisheye.stereoRectify``; the
pinhole projections are built here. fx ~ 504 on a 1600 px image means the raw
view reaches ~90 deg off-axis, which a pinhole image cannot hold, so the
rectified views keep only +-``half_fov_deg`` horizontally (default 60 deg; the
raw image beyond that is dropped). At 1600 px wide this gives
f = 800 / tan(60 deg) ~ 461.9 px. Both rectified views share f and the
principal point (output centre); the right projection is
``P_right = [[f, 0, cx, -f*B], [0, f, cy, 0], [0, 0, 1, 0]]`` in the rectified
left frame, so disparity u_L - u_R = f*B/Z > 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    k: np.ndarray  # (4,) k1..k4
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def D(self) -> np.ndarray:
        return np.asarray(self.k, dtype=np.float64).reshape(4)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.hypot(x, y)
        th = np.arctan2(r, z)
        th2 = th * th
        k1, k2, k3, k4 = self.D
        thd = th * (1 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
        safe = r > 1e-15
        scale = np.where(safe, thd / np.where(safe, r, 1.0), 1.0 / z)
        return np.stack([self.fx * scale * x + self.cx, self.fy * scale * y + self.cy], axis=1)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        mx = (px[:, 0] - self.cx) / self.fx
        my = (px[:, 1] - self.cy) / self.fy
        thd = np.hypot(mx, my)
        k1, k2, k3, k4 = self.D
        th = thd.copy()
        for _ in range(50):
            t2 = th * th
            f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd
            df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            step = f / df
            th -= step
            if np.all(np.abs(step) < 1e-12):
                break
        safe = thd > 1e-15
        s = np.where(safe, np.sin(th) / np.where(safe, thd, 1.0), 0.0)
        return np.stack([s * mx, s * my, np.cos(th)], axis=1)


@dataclass
class StereoRig:
    left: Camera
    right: Camera
    R: np.ndarray  # (3, 3), X_right = R @ X_left + t
    t: np.ndarray  # (3,) metres

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.t))


def _camera(c: dict) -> Camera:
    i, d = c["intrinsics"], c["distortion"]
    return Camera(float(i["fx"]), float(i["fy"]), float(i["cx"]), float(i["cy"]),
                  np.array([d["k1"], d["k2"], d["k3"], d["k4"]], dtype=np.float64),
                  int(c["image_width"]), int(c["image_height"]))


def load_calibration(yaml_path) -> StereoRig:
    cams = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))["cameras"]
    is_left = [c.get("name") == "IR_L" or c.get("id") == "cam_0" for c in cams]
    left = cams[is_left.index(True)]
    right = cams[is_left.index(False)]
    ext = right["extrinsics"]
    return StereoRig(_camera(left), _camera(right),
                     np.array(ext["rotation"], dtype=np.float64),
                     np.array(ext["translation"], dtype=np.float64) / 1000.0)


@dataclass
class Rectification:
    left: Camera
    right: Camera
    map_left: tuple
    map_right: tuple
    P_left: np.ndarray
    P_right: np.ndarray
    R_left: np.ndarray
    R_right: np.ndarray
    size: tuple  # (w, h)
    focal: float

    def _side(self, side: str):
        if side == "left":
            return self.left, self.R_left, self.P_left
        if side == "right":
            return self.right, self.R_right, self.P_right
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def rectify(self, image_bgr: np.ndarray, side: str) -> np.ndarray:
        self._side(side)
        m1, m2 = self.map_left if side == "left" else self.map_right
        return cv2.remap(image_bgr, m1, m2, interpolation=cv2.INTER_LINEAR)

    def rectified_to_ray_left(self, pixels_rect: np.ndarray) -> np.ndarray:
        px = np.asarray(pixels_rect, dtype=np.float64).reshape(-1, 2)
        Kinv = np.linalg.inv(self.P_left[:, :3])
        rays = np.c_[px, np.ones(len(px))] @ Kinv.T @ self.R_left  # R_left.T applied to rows
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def original_to_rectified(self, pixels_orig: np.ndarray, side: str) -> np.ndarray:
        cam, R, P = self._side(side)
        q = cam.unproject(pixels_orig) @ R.T @ P[:, :3].T
        return q[:, :2] / q[:, 2:3]

    def rectified_to_original(self, pixels_rect: np.ndarray, side: str) -> np.ndarray:
        cam, R, P = self._side(side)
        px = np.asarray(pixels_rect, dtype=np.float64).reshape(-1, 2)
        rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(P[:, :3]).T @ R  # R.T applied
        return cam.project(rays)


def build_rectification(rig: StereoRig, size=None, half_fov_deg: float = 60.0) -> Rectification:
    in_size = (rig.left.width, rig.left.height)
    w, h = size if size is not None else in_size
    R1, R2, *_ = cv2.fisheye.stereoRectify(
        rig.left.K, rig.left.D, rig.right.K, rig.right.D, in_size, rig.R, rig.t,
        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=(w, h))
    f = (w / 2) / np.tan(np.radians(half_fov_deg))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    P1 = np.c_[K, np.zeros(3)]
    P2 = np.c_[K, [-f * rig.baseline_m, 0.0, 0.0]]
    maps = [cv2.fisheye.initUndistortRectifyMap(cam.K, cam.D, Rr, K, (w, h), cv2.CV_32FC1)
            for cam, Rr in ((rig.left, R1), (rig.right, R2))]
    return Rectification(rig.left, rig.right, tuple(maps[0]), tuple(maps[1]), P1, P2,
                         np.asarray(R1), np.asarray(R2), (w, h), float(f))
