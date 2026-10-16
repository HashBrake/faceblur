"""Outputs: per-frame joints Parquet, camera pose Parquet, MANO Parquet/JSON, and the review video.

Joints are metres in the ORIGINAL left camera frame. The review video shows the
two rectified views at half size side by side with each hand's skeleton drawn.
"""
from __future__ import annotations

import json
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hands.skeleton import BONES

JOINT_COLUMNS = [f"j{k:02d}_{a}" for k in range(21) for a in "xyz"]
CONF_COLUMNS = [f"c{k:02d}" for k in range(21)]
POSE_COLUMNS = [f"pose_{k:02d}" for k in range(48)]
TRANS_COLUMNS = ["trans_x", "trans_y", "trans_z"]
POSE_PARQUET_COLUMNS = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
INSET_SIZE = 220

HAND_COLORS = {"left": (0, 200, 0), "right": (0, 140, 255)}
MIN_DRAW_CONF = 0.2


def _base_columns(frames: dict, hand: str) -> dict:
    T = len(frames["frame"])
    return {
        "frame": pa.array(np.asarray(frames["frame"], np.int64), pa.int64()),
        "timestamp_us": pa.array(np.asarray(frames["timestamp_us"], np.int64), pa.int64()),
        "hand": pa.array([hand] * T, pa.string()),
    }


def _concat(parts: list[dict]) -> pa.Table:
    names = list(parts[0])
    return pa.table({n: pa.concat_arrays([p[n] for p in parts]) for n in names})


def _f64(a) -> pa.Array:
    return pa.array(np.asarray(a, np.float64), pa.float64())


def write_joints_parquet(path, frames: dict, hands: dict, pose_valid=None) -> None:
    """One row per (frame, hand) for every frame, in "left", "right" order.

    With ``pose_valid`` ((T,) bool) a last ``pose_valid`` column is added (joints_world.parquet).
    """
    parts = []
    for hand in ("left", "right"):
        if hand not in hands:
            continue
        h = hands[hand]
        T = len(frames["frame"])
        reproj = np.asarray(h["reproj_px"], np.float64)
        finite = np.isfinite(reproj)
        n = finite.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, np.where(finite, reproj, 0.0).sum(axis=1) / n, np.nan)
        cols = _base_columns(frames, hand)
        cols["valid"] = pa.array(np.asarray(h["valid"], bool), pa.bool_())
        cols["interpolated"] = pa.array(np.asarray(h["interpolated"], bool), pa.bool_())
        cols["reproj_px"] = _f64(mean)
        joints = np.asarray(h["joints"], np.float64).reshape(T, 63)
        for i, name in enumerate(JOINT_COLUMNS):
            cols[name] = _f64(joints[:, i])
        conf = np.asarray(h["conf"], np.float64)
        for i, name in enumerate(CONF_COLUMNS):
            cols[name] = _f64(conf[:, i])
        if pose_valid is not None:
            cols["pose_valid"] = pa.array(np.asarray(pose_valid, bool), pa.bool_())
        parts.append(cols)
    pq.write_table(_concat(parts), path, compression="snappy")


def write_pose_parquet(path, frames: dict, t_wc, q_wc, valid) -> None:
    """pose.parquet: ``frame, timestamp_us, pose_valid, tx, ty, tz, qx, qy, qz, qw`` of T_world_cam."""
    cols = {"frame": pa.array(np.asarray(frames["frame"], np.int64), pa.int64()),
            "timestamp_us": pa.array(np.asarray(frames["timestamp_us"], np.int64), pa.int64()),
            "pose_valid": pa.array(np.asarray(valid, bool), pa.bool_())}
    vals = np.hstack([np.asarray(t_wc, np.float64), np.asarray(q_wc, np.float64)])
    for i, name in enumerate(POSE_PARQUET_COLUMNS):
        cols[name] = _f64(vals[:, i])
    pq.write_table(pa.table(cols), path, compression="snappy")


def read_joints_parquet(path) -> pa.Table:
    return pq.read_table(path)


def write_mano_parquet(path, frames: dict, mano: dict) -> None:
    parts = []
    for hand in ("left", "right"):
        if hand not in mano:
            continue
        m = mano[hand]
        cols = _base_columns(frames, hand)
        cols["valid"] = pa.array(np.asarray(m["valid"], bool), pa.bool_())
        pose = np.asarray(m["pose"], np.float64)
        for i, name in enumerate(POSE_COLUMNS):
            cols[name] = _f64(pose[:, i])
        trans = np.asarray(m["trans"], np.float64)
        for i, name in enumerate(TRANS_COLUMNS):
            cols[name] = _f64(trans[:, i])
        parts.append(cols)
    pq.write_table(_concat(parts), path, compression="snappy")


def write_mano_shape_json(path, shapes: dict, meta: dict) -> None:
    out = {hand: [float(v) for v in np.asarray(s, np.float64)] for hand, s in shapes.items()}
    out.update(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def draw_skeleton(image_bgr: np.ndarray, uv: np.ndarray, conf: np.ndarray, color_bgr) -> None:
    """Draw bones and joints in place; skips joints with conf < 0.2 or NaN pixels."""
    uv = np.asarray(uv, np.float64)
    ok = (np.asarray(conf) >= MIN_DRAW_CONF) & np.isfinite(uv).all(axis=1)
    pts = [tuple(int(round(c)) for c in p) if o else None for p, o in zip(uv, ok)]
    for p, c in BONES:
        if pts[p] is not None and pts[c] is not None:
            cv2.line(image_bgr, pts[p], pts[c], color_bgr, 2, cv2.LINE_AA)
    for j, pt in enumerate(pts):
        if pt is not None:
            cv2.circle(image_bgr, pt, 5 if j == 0 else 3, color_bgr, -1, cv2.LINE_AA)


def inset_bounds(xy) -> tuple:
    """Square (x0, y0, side) world box around the finite (N,2) points with a 10% margin (min 0.5 m)."""
    xy = np.asarray(xy, np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        return (-0.25, -0.25, 0.5)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    side = max(float((hi - lo).max()) * 1.2, 0.5)
    c = (lo + hi) / 2
    return (float(c[0] - side / 2), float(c[1] - side / 2), side)


def draw_trajectory_inset(image_bgr, origin, track_xy, heading_xy, bounds, size: int = INSET_SIZE) -> None:
    """Top-down world x/y inset at pixel ``origin`` (top-left): the trace so far, the current
    position (last finite point of ``track_xy``) as a dot and a heading line. World +y is up."""
    ox, oy = origin
    roi = image_bgr[oy:oy + size, ox:ox + size]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.rectangle(image_bgr, (ox, oy), (ox + size - 1, oy + size - 1), (200, 200, 200), 1)
    x0, y0, side = bounds
    xy = np.asarray(track_xy, np.float64).reshape(-1, 2)
    px = np.column_stack([ox + (xy[:, 0] - x0) / side * (size - 1),
                          oy + size - 1 - (xy[:, 1] - y0) / side * (size - 1)])
    ok = np.isfinite(px).all(axis=1)
    seg = ok[:-1] & ok[1:]
    if seg.any():
        a, b = px[:-1][seg].round().astype(np.int32), px[1:][seg].round().astype(np.int32)
        cv2.polylines(image_bgr, [np.stack([p, q]) for p, q in zip(a, b)], False, (255, 220, 0), 1,
                      cv2.LINE_AA)
    if ok.any():
        cur = px[np.flatnonzero(ok)[-1]]
        c = tuple(int(v) for v in cur.round())
        h = np.asarray(heading_xy, np.float64)
        n = np.linalg.norm(h) if np.isfinite(h).all() else 0.0
        if n > 1e-6:
            end = (int(round(cur[0] + 25 * h[0] / n)), int(round(cur[1] - 25 * h[1] / n)))
            cv2.line(image_bgr, c, end, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(image_bgr, c, 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(image_bgr, f"{side:.1f} m", (ox + 4, oy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                1, cv2.LINE_AA)


def project_rectified(rect, joints_left_orig: np.ndarray):
    """(21,3) original-left-frame points -> (uv_left, uv_right) rectified pixels; NaN if behind."""
    X_rect = np.asarray(joints_left_orig, np.float64).reshape(-1, 3) @ rect.R_left.T
    out = []
    for P in (rect.P_left, rect.P_right):
        q = X_rect @ P[:, :3].T + P[:, 3]
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        uv[~(q[:, 2] > 0)] = np.nan
        out.append(uv)
    return out[0], out[1]


class ReviewWriter:
    """Writes rectified left|right at half size, side by side, as H.264 mp4."""

    def __init__(self, path, size=(1600, 650), fps: float = 30.0):
        self.size = tuple(size)
        self.n = 0
        w, h = self.size
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-y", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, left_bgr_rect: np.ndarray, right_bgr_rect: np.ndarray, hands: dict, inset=None) -> None:
        """``inset``: optional (track_xy, heading_xy, bounds), drawn bottom-right of the left view."""
        left, right = left_bgr_rect.copy(), right_bgr_rect.copy()
        for hand, (uv_l, uv_r, conf) in hands.items():
            color = HAND_COLORS[hand]
            draw_skeleton(left, uv_l, conf, color)
            draw_skeleton(right, uv_r, conf, color)
        w, h = self.size
        half = (w // 2, h)
        frame = np.hstack([cv2.resize(left, half, interpolation=cv2.INTER_AREA),
                           cv2.resize(right, half, interpolation=cv2.INTER_AREA)])
        if inset is not None:
            draw_trajectory_inset(frame, (half[0] - INSET_SIZE - 8, h - INSET_SIZE - 8), *inset)
        cv2.putText(frame, str(self.n), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
            self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
