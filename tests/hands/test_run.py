"""Tests for the pipeline driver (hands.run) and the part reader (hands.video)."""
from __future__ import annotations

import json
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
import pytest

from hands.calib import build_rectification, load_calibration
from hands.conftest import CALIB_YAML
from hands.detect import Hand2D
from hands.output import CONF_COLUMNS, JOINT_COLUMNS
from hands.run import main, run_pipeline

STEM = "Orbbec_Ego_SYNTH_19700101_000000"
N_ROWS = 70          # csv rows and video frames per camera
RIGHT_OFFSET = 10    # right csv row 0 == left csv row 10
START_S, DURATION_S = 0.32, 2.0  # left rows 10..69: the 60 rows both cameras cover
FPS = 30.0
CORRUPT_FRAME, CORRUPT_JOINT = 20, 8  # left hand, right view: joint 8 at 0.6 px disparity (~90 m deep)


@pytest.fixture(scope="module")
def rig():
    return load_calibration(CALIB_YAML)


@pytest.fixture(scope="module")
def rect(rig):
    return build_rectification(rig)


def synthetic_hand(x, y=0.0, z=0.5):
    """21 joints (MediaPipe order), bones 2-9 cm, wrist near (x, y + 0.05, z), original left frame."""
    pts = [[0.0, 0.0, 0.0]]
    for k, ang in enumerate(np.radians([-60, -25, -5, 15, 35])):
        direction = np.array([np.sin(ang), -np.cos(ang), 0.1 * k])
        direction /= np.linalg.norm(direction)
        steps = (0.03, 0.06, 0.085, 0.105) if k == 0 else (0.04, 0.08, 0.11, 0.135)
        base = np.array([0.0, -0.045, 0.0]) if k else np.zeros(3)
        pts.extend(base + direction * s for s in steps)
    return np.array(pts) + [x, y + 0.05, z]


def truth(k):
    """Wearer's (left, right) hand joints at stereo frame k."""
    return synthetic_hand(-0.12 + 0.02 * k / FPS), synthetic_hand(0.12)


def project_rect(rect, rig, X):
    uvL = rect.original_to_rectified(rect.left.project(X), "left")
    uvR = rect.original_to_rectified(rect.right.project(X @ rig.R.T + rig.t), "right")
    return uvL, uvR


def _write_video(path, n, seed):
    w, h = 1600, 1300
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-r", "30", "-y", str(path)]
    grad = np.tile(np.linspace(0, 200, w).astype(np.uint8)[None, :, None], (h, 1, 3))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        frame = grad.copy()
        frame[:, :, (i + seed) % 3] = (i * 3) % 256
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def make_recording(root):
    rec = root / STEM
    rec.mkdir()
    shutil.copy(CALIB_YAML, rec / f"{STEM}_calibration_camera.yaml")
    rng = np.random.default_rng(7)
    base = 1_000_000_000 + 33_333 * np.arange(N_ROWS + RIGHT_OFFSET) + rng.integers(-300, 301, N_ROWS + RIGHT_OFFSET)
    for side, ts in (("left", base[:N_ROWS]), ("right", base[RIGHT_OFFSET:])):
        np.savetxt(rec / f"{STEM}_camera_{side}_part0001_pts.csv", ts, fmt="%d", header="timestamp_us",
                   comments="")
        _write_video(rec / f"{STEM}_camera_{side}_part0001.mp4", N_ROWS, 0 if side == "left" else 1)
    return rec


class FakeDetector:
    """Returns both synthetic hands, projected into this view's rectified pixels with 0.3 px noise."""

    def __init__(self, view, rect, rig, seed, partner=None):
        self.view, self.rect, self.rig = view, rect, rig
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.partner = partner  # left-view detector: corrupt one joint against its output
        self.last = None

    def detect(self, image):
        assert image.shape == (1300, 1600, 3)
        out = []
        for label, X in zip(("left", "right"), truth(self.calls)):
            uv = project_rect(self.rect, self.rig, X)[0 if self.view == "left" else 1]
            uv = uv + self.rng.normal(0, 0.3, uv.shape)
            if self.partner is not None and label == "left" and self.calls == CORRUPT_FRAME:
                uv[CORRUPT_JOINT] = self.partner.last[0].keypoints[CORRUPT_JOINT] - [0.6, 0.0]
            out.append(Hand2D(uv, np.full(21, 0.9), label, 0.9, X - X.mean(axis=0)))
        self.calls += 1
        self.last = out
        return out


def factory(rect, rig, made, corrupt=False):
    def make():
        partner = made[0] if (made and corrupt) else None
        d = FakeDetector("left" if not made else "right", rect, rig, seed=len(made), partner=partner)
        made.append(d)
        return d
    return make


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    return make_recording(tmp_path_factory.mktemp("rec"))


@pytest.fixture(scope="module")
def synthetic_run(recording, rect, rig, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    made = []
    logs = []
    report = run_pipeline(recording, out, start_s=START_S, duration_s=DURATION_S,
                          detector_factory=factory(rect, rig, made, corrupt=True), log=logs.append)
    return out, report, made, logs


def test_synthetic_end_to_end(synthetic_run):
    out, report, made, logs = synthetic_run
    q = json.loads((out / "quality.json").read_text(encoding="utf-8"))
    T = q["clip"]["n_frames"]
    assert T >= 55
    assert len(made) == 2 and made[0].view == "left"
    assert made[0].calls == T and made[1].calls == T
    assert q["clip"]["recording"] == STEM and q["clip"]["calibration"].endswith("_calibration_camera.yaml")
    assert q["clip"]["mano"] is False and q["clip"]["detection_fps"] > 0
    assert q["clip"]["frames_left"][1] - q["clip"]["frames_left"][0] == T - 1
    assert logs[-1].startswith("summary:")

    t = pq.read_table(out / "joints.parquet")
    assert t.num_rows == 2 * T
    assert t.column_names == (["frame", "timestamp_us", "hand", "valid", "interpolated", "reproj_px"]
                              + JOINT_COLUMNS + CONF_COLUMNS)
    d = t.to_pydict()
    assert d["hand"] == ["left"] * T + ["right"] * T

    o = q["overall"]
    assert o["pass"]["reproj"] is True and o["pass"]["bone_cv"] is True
    assert q["left"]["valid_share"] >= 0.9 and q["right"]["valid_share"] >= 0.9
    assert o["reproj_px_median"] < 1.0 and o["reproj_out_px_median"] is not None
    assert q["left"]["reproj_px"]["median"] < 1.0 and q["left"]["reproj_px"]["p90"] >= 0
    assert q["left"]["reproj_out_px"]["median"] >= 0
    assert report["overall"]["reproj_px_median"] == o["reproj_px_median"]
    assert o["bone_cv_max"] < 1e-9  # re-constrained after smoothing: exact bone lengths

    # Depth gate: the one ~90 m joint (left hand) is NaN in the raw stats; output joint 8 stays finite.
    n_present = round(q["left"]["valid_share"] * T)  # synthetic: every frame present, none interpolated
    assert 0 < q["left"]["implausible_joint_share"] <= 1 / (21 * n_present) + 1e-9
    assert q["right"]["implausible_joint_share"] == 0
    assert o["implausible_joint_share"] is not None
    assert q["left"]["bone_cv_raw_max"] < 0.5
    assert q["clip"]["crop"] is False
    assert 0 <= q["right"]["fingertip_row_p90_px"] < 3.0  # 0.3 px noise per view

    # Parquet reproj_px is the raw stereo residual: pooled over rows where it is finite (the raw-finite
    # frames quality.py pools), its median equals quality.json's overall.reproj_px_median.
    rp = np.array(d["reproj_px"], np.float64)
    assert np.isclose(np.median(rp[np.isfinite(rp)]), o["reproj_px_median"], rtol=0, atol=1e-6)

    joints = np.array([[d[c][r] for c in JOINT_COLUMNS] for r in range(2 * T)]).reshape(2, T, 21, 3)
    errs = []
    for h in range(2):
        for k in range(T):
            r = h * T + k
            if d["valid"][r] and not d["interpolated"][r]:
                errs.append(np.linalg.norm(joints[h, k] - truth(k)[h], axis=1).mean())
    assert len(errs) >= 0.9 * 2 * T
    assert np.mean(errs) < 0.005, np.mean(errs)
    assert np.isfinite(joints[0, CORRUPT_FRAME, CORRUPT_JOINT]).all()

    cap = cv2.VideoCapture(str(out / "review.mp4"))
    assert cap.isOpened()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (650, 1600, 3)
        n += 1
    cap.release()
    assert n == T


def test_synthetic_mano(recording, rect, rig, tmp_path, mano_dir):
    out = tmp_path / "out"
    report = run_pipeline(recording, out, start_s=START_S, duration_s=DURATION_S, mano=True, mano_dir=mano_dir,
                          detector_factory=factory(rect, rig, []), log=lambda s: None)
    T = report["clip"]["n_frames"]
    assert report["clip"]["mano"] is True and report["clip"]["mano_seconds"] > 0
    t = pq.read_table(out / "mano.parquet")
    assert t.num_rows == 2 * T
    assert sum(c.startswith("pose_") for c in t.column_names) == 48
    j = json.loads((out / "mano_shape.json").read_text(encoding="utf-8"))
    assert len(j["left"]) == 10 and len(j["right"]) == 10
    assert j["recording"] == STEM and j["part"] == 1
    for hand in ("left", "right"):
        # MediaPipe-layout synthetic hand vs MANO geometry: ~9-10 mm measured, so 20 mm (allowed by T10).
        assert j["rmse_mm"][hand] < 20.0, j["rmse_mm"]


def test_crop_path_wired(recording, tmp_path):
    """crop=True wraps the factory's detectors: with no hands every call gets the full rectified frame."""
    seen = []

    class Empty:
        def detect(self, image):
            seen.append(image.shape)
            return []

    report = run_pipeline(recording, tmp_path / "out", start_s=START_S, duration_s=DURATION_S, crop=True,
                          detector_factory=Empty, log=lambda s: None)
    T = report["clip"]["n_frames"]
    assert report["clip"]["crop"] is True
    assert seen == [(1300, 1600, 3)] * (2 * T)


def test_main_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "usage" in out and "--crop" in out


def test_part_reader_size_check(recording, rig):
    from hands.timing import container_timestamps, find_part_files, stereo_frames
    from hands.video import PartReader
    left, right, *_ = find_part_files(recording, 1)
    cts = (container_timestamps(left), container_timestamps(right))
    pairs = stereo_frames(recording, 1, START_S, DURATION_S, container_ts=cts)
    reader = PartReader(left, right, (800, 600), cts)
    with pytest.raises(ValueError, match="differs from the calibration"):
        next(reader.frames(pairs))
    reader.close()


@pytest.mark.real_footage
def test_real_smoke(recording_ev, tmp_path):
    report = run_pipeline(recording_ev, tmp_path / "out", part=1, start_s=600, duration_s=3.0)
    q = json.loads((tmp_path / "out" / "quality.json").read_text(encoding="utf-8"))
    assert 85 <= q["clip"]["n_frames"] <= 92
    assert q["overall"]["any_hand_valid_share"] > 0.5
    o = report["overall"]
    cv_raw = max(report[h]["bone_cv_raw_max"] or 0 for h in ("left", "right"))
    print(f"\nREAL SMOKE: reproj median {o['reproj_px_median']} px (output {o['reproj_out_px_median']}), "
          f"bone CV max raw {cv_raw} constrained {o['bone_cv_max']}, jitter {o['jitter_mm']} mm, "
          f"detection {report['clip']['detection_fps']} fps, processing {report['processing_fps']} fps, "
          f"valid L {report['left']['valid_share']} R {report['right']['valid_share']}")
