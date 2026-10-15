"""Outputs: per-frame joints Parquet, MANO Parquet/JSON, and the review video.

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


def write_joints_parquet(path, frames: dict, hands: dict) -> None:
    """One row per (frame, hand) for every frame, in "left", "right" order."""
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
        parts.append(cols)
    pq.write_table(_concat(parts), path, compression="snappy")


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

    def write(self, left_bgr_rect: np.ndarray, right_bgr_rect: np.ndarray, hands: dict) -> None:
        left, right = left_bgr_rect.copy(), right_bgr_rect.copy()
        for hand, (uv_l, uv_r, conf) in hands.items():
            color = HAND_COLORS[hand]
            draw_skeleton(left, uv_l, conf, color)
            draw_skeleton(right, uv_r, conf, color)
        w, h = self.size
        half = (w // 2, h)
        frame = np.hstack([cv2.resize(left, half, interpolation=cv2.INTER_AREA),
                           cv2.resize(right, half, interpolation=cv2.INTER_AREA)])
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
