"""The patch pass: which rows it masks, how long it holds them, and what it writes.

CPU only and no detector: the rows are handed in, the way the check would
report them, so these tests measure the selection, the hold and the write
pass, not the detectors (`tests/test_verify.py` covers those)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from faceblur.patch import held, patch, to_mask
from faceblur.settings import Settings
from faceblur.video import read_frames

from conftest import audio_streams, run_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = Settings(encoder="x264")      # never the GPU in these tests


def row(frame, x, y, px=30, **marks):
    """One face row as `verify._row` writes it, centre and size, no landmarks."""
    return {"kind": "face", "frame": frame, "px": px, "x": x, "y": y, "w": px, "h": px,
            "score": 0.9, "landmarks": None, "applied": 0.05, **marks}


# ------------------------------------------------------------ without video

def test_to_mask_leaves_out_hands_zone_and_other_kinds():
    rows = [row(1, 10, 10), row(1, 50, 50, hand=0.8), row(2, 10, 10, zone=0.7),
            row(3, 10, 10, px=8), {**row(4, 10, 10), "kind": "screen"}]
    chosen = to_mask(rows)
    # The 8 px face is kept: the patch has no size floor.
    assert [(r["frame"], r["px"]) for r in chosen] == [(1, 30), (3, 8)]


def test_hold_extends_either_side_and_clips_to_the_video():
    boxes = held([row(1, 20, 20)], n=10, hold=2)
    assert sorted(boxes) == [0, 1, 2, 3]
    boxes = held([row(5, 20, 20)], n=10, hold=2)
    assert sorted(boxes) == [3, 4, 5, 6, 7]
    assert all(len(v) == 1 for v in boxes.values())
    assert sorted(held([row(5, 20, 20)], n=10, hold=0)) == [5]


def test_hold_does_not_stack_the_same_face():
    # Found on two consecutive frames: frames 4 to 7 each get one box, not two.
    boxes = held([row(5, 20, 20), row(6, 21, 20)], n=20, hold=2)
    assert sorted(boxes) == [3, 4, 5, 6, 7, 8]
    assert all(len(v) == 1 for v in boxes.values())
    # A face elsewhere on the frame is a second box.
    boxes = held([row(5, 20, 20), row(5, 90, 60)], n=20, hold=1)
    assert all(len(v) == 2 for v in boxes.values())


# ---------------------------------------------------------------- with video

@pytest.fixture(scope="module")
def copy_clip(tmp_path_factory) -> Path:
    """Four seconds of test pattern at 10 fps, 128x96, a keyframe every 10
    frames and no B-frames, the way the pipeline's copies are encoded."""
    out = tmp_path_factory.mktemp("patch") / "clip_blurred.mp4"
    r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=128x96:rate=10:duration=4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-bf", "0",
                    "-g", "10", "-keyint_min", "10", "-sc_threshold", "0",
                    "-pix_fmt", "yuv420p", str(out)])
    assert r.returncode == 0, r.stderr
    return out


def box_change(a: np.ndarray, b: np.ndarray, x: int, y: int, half: int = 10) -> float:
    return float(np.abs(a[y - half:y + half, x - half:x + half].astype(np.float32)
                        - b[y - half:y + half, x - half:x + half].astype(np.float32)).mean())


def test_patch_masks_the_rows_and_only_them(copy_clip, tmp_path):
    out = tmp_path / "patched.mp4"
    rows = [row(5, 40, 48), row(5, 100, 48, hand=0.9), row(15, 100, 48, zone=0.8)]
    record = patch(copy_clip, copy_clip, out, SETTINGS, hold=2, rows=rows)
    assert record["error"] == "" and record["status"] == "done"
    assert out.is_file()
    before, after = list(read_frames(copy_clip)), list(read_frames(out))
    assert len(before) == len(after) == 40
    # The face, held two frames either side.
    for i in range(3, 8):
        assert box_change(before[i], after[i], 40, 48) > 8, i
    # Outside the hold, and on the boxes set aside, nothing beyond encoder noise.
    for i in (0, 1, 2, 8, 9):
        assert box_change(before[i], after[i], 40, 48) < 3, i
    for i in (5, 15):
        assert box_change(before[i], after[i], 100, 48) < 3, i
    # Stretches with no mask are copied from the copy, so they decode identical.
    for i in range(20, 40):
        assert np.array_equal(before[i], after[i]), i
    assert record["frames_copied"] > 0
    # No audio, and the record beside the output says what was done.
    assert audio_streams(out) == []
    saved = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    assert saved["masked"] == 1 and saved["set_aside_hand"] == 1
    assert saved["set_aside_zone"] == 1 and saved["rows_found"] == 3
    assert saved["frames_touched"] == 5 and saved["frames_found"] == 1
    assert saved["masked_by_size"]["24-40 px"] == 1
    assert saved["frames"] == 40 and saved["output_audio"] is False
    assert saved["faceblur_version"] and "check_seconds" in saved


def test_nothing_to_mask_copies_the_file_byte_for_byte(copy_clip, tmp_path):
    out = tmp_path / "patched.mp4"
    record = patch(copy_clip, copy_clip, out, SETTINGS, rows=[row(5, 40, 48, hand=0.9)])
    assert record["masked"] == 0 and record["error"] == ""
    assert out.read_bytes() == copy_clip.read_bytes()
    assert json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))["masked"] == 0


def test_a_failed_patch_leaves_no_output_and_says_why(tmp_path):
    out = tmp_path / "patched.mp4"
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nothing here")
    record = patch(bad, bad, out, SETTINGS, rows=[row(1, 10, 10)])
    assert record["error"] and record["status"] == "failed"
    assert not out.exists()
    assert json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))["error"]


def test_the_check_feeds_the_patch(video_with_face, tmp_path):
    """End to end, detectors included: an unblurred copy shows its face on
    every frame, so the check finds it and the patch masks it. The one test
    here that runs the detectors, on ten frames."""
    out = tmp_path / "patched.mp4"
    record = patch(video_with_face, video_with_face, out, SETTINGS, hold=2)
    assert record["error"] == "", record["error"]
    assert record["frames_checked"] == 10
    assert record["masked"] >= 5 and record["frames_touched"] == 10
    before, after = list(read_frames(video_with_face)), list(read_frames(out))
    assert all(box_change(a, b, 320, 320, 30) > 8 for a, b in zip(before, after))


# ------------------------------------------------------------- deliver.py

@pytest.fixture(scope="module")
def deliver():
    spec = importlib.util.spec_from_file_location("deliver", ROOT / "cloud" / "deliver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deliver_picks_the_patched_copy_only_when_asked_and_finished(deliver):
    out = {"a/x_blurred.mp4": 10, "a/x_blurred.mp4.json": 1,
           "a/quarantine/y_blurred.mp4": 20, "a/quarantine/y_blurred.mp4.json": 1,
           "a/quarantine/z_blurred.mp4": 30, "a/quarantine/z_blurred.mp4.json": 1}
    patched = {"a/y_blurred.mp4": 21, "a/y_blurred.mp4.json": 1,
               "a/z_blurred.mp4": 31}                         # z never finished
    # Without the flag nothing changes.
    got = deliver.pick("a/y.mp4", out, patched, "_blurred", False)
    assert got["video"] == "out/a/quarantine/y_blurred.mp4" and not got["patched"]
    # With it, a held back copy with a finished patch comes from patched/.
    got = deliver.pick("a/y.MP4", out, patched, "_blurred", True)
    assert got["video"] == "patched/a/y_blurred.mp4" and got["size"] == 21
    assert got["records"] == [("out/a/quarantine/y_blurred.mp4.json", "a/y.MP4.json"),
                              ("patched/a/y_blurred.mp4.json", "a/y.MP4.patch.json")]
    # A patch with no record is not used; a delivered copy is never patched.
    assert deliver.pick("a/z.mp4", out, patched, "_blurred", True)["video"] == \
        "out/a/quarantine/z_blurred.mp4"
    got = deliver.pick("a/x.mp4", out, patched, "_blurred", True)
    assert got["video"] == "out/a/x_blurred.mp4" and not got["held"]
    assert deliver.pick("a/missing.mp4", out, patched, "_blurred", True) is None
    # At the top of the tree too.
    assert deliver.pick("w.mp4", {"quarantine/w_blurred.mp4": 5,
                                  "quarantine/w_blurred.mp4.json": 1},
                        {"w_blurred.mp4": 6, "w_blurred.mp4.json": 1},
                        "_blurred", True)["video"] == "patched/w_blurred.mp4"
