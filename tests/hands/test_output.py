import json

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hands.calib import build_rectification, load_calibration
from hands.conftest import CALIB_YAML
from hands.output import (CONF_COLUMNS, JOINT_COLUMNS, ReviewWriter, draw_skeleton,
                          project_rectified, read_joints_parquet, write_joints_parquet,
                          write_mano_parquet, write_mano_shape_json)

T = 5


@pytest.fixture(scope="module")
def rect():
    return build_rectification(load_calibration(CALIB_YAML))


def _frames():
    return {"frame": np.arange(10, 10 + T, dtype=np.int64),
            "timestamp_us": np.arange(T, dtype=np.int64) * 33333 + 1000}


def _hand(rng, invalid_frame=None):
    joints = rng.normal(0, 0.05, (T, 21, 3)) + [0, 0, 0.5]
    conf = rng.uniform(0, 1, (T, 21))
    valid = np.ones(T, bool)
    reproj = rng.uniform(0, 3, (T, 21))
    reproj[1, :5] = np.nan
    if invalid_frame is not None:
        joints[invalid_frame] = np.nan
        valid[invalid_frame] = False
        reproj[invalid_frame] = np.nan
    return {"joints": joints, "conf": conf, "valid": valid,
            "interpolated": np.array([False, False, True, False, False]), "reproj_px": reproj}


def _hand_layout():
    """Plausible 21-joint hand (metres) around a wrist at z = 0.5 m."""
    pts = [np.array([0.0, 0.05, 0.5])]
    for f in range(5):
        base = np.array([-0.04 + 0.02 * f, 0.0, 0.5])
        for k in range(4):
            pts.append(base + [0.0, -0.025 * k, -0.005 * k])
    return np.array(pts)


def test_joints_parquet_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    hands = {"left": _hand(rng, invalid_frame=3), "right": _hand(rng)}
    path = tmp_path / "joints.parquet"
    write_joints_parquet(path, _frames(), hands)
    t = read_joints_parquet(path)

    expected = (["frame", "timestamp_us", "hand", "valid", "interpolated", "reproj_px"]
                + JOINT_COLUMNS + CONF_COLUMNS)
    assert t.column_names == expected
    assert len(JOINT_COLUMNS) == 63 and len(CONF_COLUMNS) == 21
    types = {"frame": pa.int64(), "timestamp_us": pa.int64(), "hand": pa.string(),
             "valid": pa.bool_(), "interpolated": pa.bool_()}
    for name in t.column_names:
        assert t.schema.field(name).type == types.get(name, pa.float64()), name
    assert t.num_rows == 2 * T

    d = t.to_pydict()
    assert d["hand"] == ["left"] * T + ["right"] * T
    assert d["frame"] == list(_frames()["frame"]) * 2
    assert d["timestamp_us"] == list(_frames()["timestamp_us"]) * 2
    assert d["valid"][3] is False and all(d["valid"][i] for i in range(2 * T) if i != 3)
    assert d["interpolated"][2] is True and d["interpolated"][0] is False
    for r, hand in enumerate(["left"] * T + ["right"] * T):
        i = r % T
        h = hands[hand]
        got = np.array([d[c][r] for c in JOINT_COLUMNS], dtype=float)
        np.testing.assert_allclose(got, h["joints"][i].reshape(-1), atol=1e-12)
        if i == 3 and hand == "left":
            assert np.isnan(got).all()
        np.testing.assert_allclose([d[c][r] for c in CONF_COLUMNS], h["conf"][i], atol=1e-12)
        row = h["reproj_px"][i]
        exp = row[np.isfinite(row)].mean() if np.isfinite(row).any() else np.nan
        np.testing.assert_allclose(d["reproj_px"][r], exp, atol=1e-12)
    assert np.isnan(d["reproj_px"][3])


def test_joints_parquet_right_only(tmp_path):
    path = tmp_path / "joints.parquet"
    write_joints_parquet(path, _frames(), {"right": _hand(np.random.default_rng(1))})
    t = read_joints_parquet(path)
    assert t.num_rows == T
    assert set(t.column("hand").to_pylist()) == {"right"}


def test_mano_parquet_and_shape_json(tmp_path):
    rng = np.random.default_rng(2)
    mano = {h: {"pose": rng.normal(size=(T, 48)), "trans": rng.normal(size=(T, 3)),
                "valid": np.array([True, True, False, True, True])} for h in ("left", "right")}
    path = tmp_path / "mano.parquet"
    write_mano_parquet(path, _frames(), mano)
    t = pq.read_table(path)
    pose_cols = [f"pose_{k:02d}" for k in range(48)]
    assert t.column_names == (["frame", "timestamp_us", "hand", "valid"] + pose_cols
                              + ["trans_x", "trans_y", "trans_z"])
    assert t.num_rows == 2 * T
    d = t.to_pydict()
    assert d["hand"] == ["left"] * T + ["right"] * T
    np.testing.assert_allclose([d[c][T + 1] for c in pose_cols], mano["right"]["pose"][1], atol=1e-12)
    np.testing.assert_allclose([d[c][0] for c in ("trans_x", "trans_y", "trans_z")],
                               mano["left"]["trans"][0], atol=1e-12)
    assert d["valid"][2] is False

    shapes = {"left": rng.normal(size=10), "right": rng.normal(size=10)}
    jpath = tmp_path / "shape.json"
    write_mano_shape_json(jpath, shapes, {"recording": "rec01"})
    j = json.loads(jpath.read_text(encoding="utf-8"))
    assert j["recording"] == "rec01"
    np.testing.assert_allclose(j["left"], shapes["left"], atol=1e-12)
    assert len(j["right"]) == 10


def test_draw_skeleton():
    uv = np.array([[100.0, 160.0]] + [[60 + 20 * (j // 4), 140 - 20 * (j % 4)] for j in range(20)])
    img = np.zeros((200, 200, 3), np.uint8)
    draw_skeleton(img, uv, np.ones(21), (0, 200, 0))
    assert img.any()

    img = np.zeros((200, 200, 3), np.uint8)
    draw_skeleton(img, uv, np.zeros(21), (0, 200, 0))
    assert not img.any()

    nan_uv = uv.copy()
    nan_uv[::2] = np.nan
    img = np.zeros((200, 200, 3), np.uint8)
    draw_skeleton(img, nan_uv, np.ones(21), (0, 140, 255))
    draw_skeleton(img, np.full((21, 2), np.nan), np.ones(21), (0, 140, 255))


def test_project_rectified_disparity(rect):
    X = np.array([[0.02, -0.01, 0.5]])
    uv_l, uv_r = project_rectified(rect, X)
    z_rect = (rect.R_left @ X[0])[2]
    f = rect.P_left[0, 0]
    B = -rect.P_right[0, 3] / f
    assert abs((uv_l[0, 0] - uv_r[0, 0]) - f * B / z_rect) < 1e-6
    assert abs(uv_l[0, 1] - uv_r[0, 1]) < 1e-9

    behind, _ = project_rectified(rect, np.array([[0.0, 0.0, -0.5], [np.nan, 0.0, 0.5]]))
    assert np.isnan(behind).all()


def test_review_writer(tmp_path, rect):
    w, h = rect.size
    assert (w, h) == (1600, 1300)
    grad = np.tile(np.linspace(0, 255, w).astype(np.uint8)[None, :, None], (h, 1, 3))
    uv_l, uv_r = project_rectified(rect, _hand_layout())
    assert np.isfinite(uv_l).all() and np.isfinite(uv_r).all()
    path = tmp_path / "review.mp4"
    with ReviewWriter(path) as rw:
        for _ in range(12):
            rw.write(grad, grad, {"left": (uv_l, uv_r, np.ones(21))})
    assert path.is_file()
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (650, 1600, 3)
        n += 1
    cap.release()
    assert n == 12
