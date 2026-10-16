"""The adaptive detection stride (`faceblur/stride.py`).

Detection skips a frame only where nothing was seen for a while and the camera
is nearly still. These tests hold the policy to its rules, hold production to
the masks it drew without the policy on a clip with quiet wall stretches, and
hold the evaluation harness's replay to the frames production skips.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from faceblur import batch
from faceblur.batch import _chunks, _mask_key, detect_job, run_video, serial_submit
from faceblur.pipeline import STATUS_DONE
from faceblur.segments import frame_times
from faceblur.settings import Settings, SettingsError
from faceblur.stride import AdaptiveStride, skip_next
from faceblur.video import probe

from eval import measure

from conftest import run_ffmpeg

ON = Settings(adaptive_stride=True, adaptive_stride_max=2, adaptive_stride_shift=4.0,
              adaptive_stride_quiet=3)
FACE = ["face"]


def _drive(settings, frames):
    """Feed (strong, weak, shift) per frame through one policy; the skipped indices."""
    policy = AdaptiveStride(settings)
    skipped = []
    for i, (strong, weak, shift) in enumerate(frames):
        if policy.skip(shift):
            skipped.append(i)
        else:
            policy.saw(strong, weak)
    return skipped


# ------------------------------------------------------------ the policy

def test_never_skips_until_quiet_frames_were_empty():
    assert _drive(ON, [([], [], 0.0)] * 3) == []
    # The fourth frame follows three empty detected ones.
    assert _drive(ON, [([], [], 0.0)] * 4) == [3]


def test_never_skips_after_a_strong_or_a_weak_candidate():
    for strong, weak in ((FACE, []), ([], FACE)):
        frames = [(strong, weak, 0.0)] + [([], [], 0.0)] * 3
        # After the candidate on frame 0, frames 1, 2, 3 are detected.
        assert _drive(ON, frames) == []
        assert _drive(ON, frames + [([], [], 0.0)]) == [4]
    assert not skip_next(0, 0, 0.0, ON)


def test_never_two_skips_in_a_row_at_max_2():
    skipped = _drive(ON, [([], [], 0.0)] * 40)
    assert skipped
    assert all(b - a >= 2 for a, b in zip(skipped, skipped[1:]))
    assert not skip_next(10, 1, 0.0, ON)
    # At max 3 two in a row are allowed, never three.
    three = _drive(ON.with_changes(adaptive_stride_max=3), [([], [], 0.0)] * 40)
    assert any(b - a == 1 for a, b in zip(three, three[1:]))
    assert not any(c - a == 2 for a, c in zip(three, three[2:]))


def test_a_skipped_frame_is_not_quiet_evidence():
    # Detected 0, 1, 2, skipped 3, detected 4 with a face: three empties after
    # it are needed again, and the skip on 3 did not count towards them.
    frames = [([], [], 0.0)] * 4 + [(FACE, [], 0.0)] + [([], [], 0.0)] * 4
    assert _drive(ON, frames) == [3, 8]


def test_never_skips_when_the_camera_moves_or_the_shift_is_missing():
    assert _drive(ON, [([], [], 4.0)] * 20) == []
    assert _drive(ON, [([], [], 9.0)] * 20) == []
    assert _drive(ON, [([], [], None)] * 20) == []
    assert _drive(ON, [([], [], 3.99)] * 5) == [3]


def test_max_1_or_off_never_skips():
    assert _drive(ON.with_changes(adaptive_stride_max=1), [([], [], 0.0)] * 30) == []
    assert _drive(ON.with_changes(adaptive_stride=False), [([], [], 0.0)] * 30) == []


# ------------------------------------------------------------ settings

def test_settings_validation():
    with pytest.raises(SettingsError):
        Settings(adaptive_stride_max=0)
    with pytest.raises(SettingsError):
        Settings(adaptive_stride_quiet=0)
    with pytest.raises(SettingsError):
        Settings(adaptive_stride_shift=-1.0)
    with pytest.raises(SettingsError):
        Settings(adaptive_stride_shift=10.0, fast_shift=10.0)
    Settings(adaptive_stride_max=1, adaptive_stride_quiet=1, adaptive_stride_shift=0.0)
    Settings(adaptive_stride_shift=20.0, fast_shift=0.0)       # fast_shift off
    d = Settings().to_dict()
    for name in ("adaptive_stride", "adaptive_stride_max", "adaptive_stride_shift",
                 "adaptive_stride_quiet"):
        assert name in d


# ------------------------------------------------------------ production

def test_detect_job_detects_every_frame_while_the_face_is_there(video_with_face):
    info, times = probe(video_with_face), frame_times(video_with_face)
    n = len(times)
    got = detect_job({"src": str(video_with_face), "info": info, "times": times,
                      "settings": Settings(device="cpu"), "start": 0, "end": n})
    assert got["skipped_adaptive"] == 0
    assert sorted(got["strong"]) == list(range(n))
    assert all(got["strong"][i] for i in range(n))


@pytest.fixture(scope="module")
def face_wall_face(tmp_path_factory, lena) -> Path:
    """The clip of tests/test_overlap.py: a grey wall for three seconds, a
    still face for one, the wall for four more, 10 fps."""
    out = tmp_path_factory.mktemp("clips") / "face_wall_face.mp4"
    face = "scale=160:-2,pad=640:640:240:240:gray,setsar=1,format=yuv420p"
    result = run_ffmpeg([
        "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=gray:s=640x640:r=10:d=3",
        "-loop", "1", "-framerate", "10", "-t", "1", "-i", str(lena),
        "-f", "lavfi", "-i", "color=c=gray:s=640x640:r=10:d=4",
        "-filter_complex",
        f"[0:v]setsar=1,format=yuv420p[a];[1:v]{face}[b];[2:v]setsar=1,format=yuv420p[c];"
        "[a][b][c]concat=n=3:v=1[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
        "-g", "10", "-keyint_min", "10", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the clip: {result.stderr[-500:]}")
    return out


def _masks_of(src, out, monkeypatch, adaptive: bool):
    real = batch._write_and_join
    seen = []

    def spy(*args, **kwargs):
        seen.append(args[6])
        return real(*args, **kwargs)

    monkeypatch.setattr(batch, "_write_and_join", spy)
    record = run_video(src, out, Settings(device="cpu", chunk_seconds=1,
                                          adaptive_stride=adaptive), serial_submit)
    monkeypatch.setattr(batch, "_write_and_join", real)
    assert record.status == STATUS_DONE, record.error
    return record, [_mask_key(f) for f in seen[0]]


def test_run_video_masks_are_the_same_with_the_policy_on(tmp_path, face_wall_face, monkeypatch):
    on, masks_on = _masks_of(face_wall_face, tmp_path / "on.mp4", monkeypatch, True)
    off, masks_off = _masks_of(face_wall_face, tmp_path / "off.mp4", monkeypatch, False)
    assert off.frames_skipped_adaptive == 0
    assert on.frames_skipped_adaptive > 0
    assert any(masks_off)
    assert masks_on == masks_off
    assert on.frames_with_mask == off.frames_with_mask


# ------------------------------------------------------------ the replay

def _sequence(n):
    """Per frame strong, weak and camera shift: faces, a weak box, a pan."""
    strong = {i: ["face"] for i in list(range(10, 15)) + [50, 51, 52]}
    weak = {70: ["weak"]}
    shifts = {i: ((6.0, 0.0) if 33 <= i <= 40 else (1.0, 1.0)) for i in range(n)}
    return strong, weak, shifts


def test_the_replay_skips_the_frames_detect_job_skips(monkeypatch):
    n, fps = 100, 10.0
    settings = Settings(chunk_seconds=3, zone=False, device="cpu")
    strong, weak, shifts = _sequence(n)
    chunk = max(1, int(round(settings.chunk_seconds * fps)))

    # Production: detect_job on run_video's ranges, with the models, the
    # decode and the shift faked from the sequence.
    looked = []

    def fake_filter(raw, settings_, shape):
        looked.append(raw)
        return list(strong.get(raw, []))

    bank = SimpleNamespace(detect_raw=lambda frame: frame, compute={}, model_hashes={})
    monkeypatch.setattr(batch, "_bank", lambda s: bank)
    monkeypatch.setattr(batch, "decode_range",
                        lambda src, times, a, b, info, hw: iter(range(a, b)))
    monkeypatch.setattr(batch, "downscale", lambda frame: frame)
    monkeypatch.setattr(batch, "estimate_shift",
                        lambda prev, cur, side: (0.0, 0.0) if prev is None else shifts[cur])
    monkeypatch.setattr(batch, "filter_candidates", fake_filter)
    monkeypatch.setattr(batch, "weak_candidates",
                        lambda raw, s, shape: list(weak.get(raw, [])))
    info = SimpleNamespace(height=4, width=4, fps=fps)
    total = 0
    for a, b in _chunks(n, chunk):
        total += detect_job({"src": "x.mp4", "info": info, "times": [], "settings": settings,
                             "start": a, "end": b})["skipped_adaptive"]
    production = sorted(looked)

    # The replay, from a cache of the same sequence.
    looked.clear()
    monkeypatch.setattr(measure, "filter_candidates", fake_filter)
    monkeypatch.setattr(measure, "weak_candidates",
                        lambda raw, s, shape: list(weak.get(raw, [])))
    got = {}

    class Tracker:
        def run(self, per, weak_, shifts_, shape):
            got["per"], got["weak"] = per, weak_
            return per, []

    monkeypatch.setattr(measure, "tracker_for", lambda s: Tracker())
    cache = {"shape": (4, 4), "frames": {i: i for i in range(n)},
             "shifts": {i: shifts[i] for i in range(n)}}
    measure.tracked(cache, settings, fps=fps)
    replay = sorted(looked)

    assert production == replay
    skipped = sorted(set(range(n)) - set(replay))
    assert total == len(skipped) > 0
    # Every chunk's first frame is detected, nothing near a candidate or in the
    # pan is skipped, and the skipped frames carry empty lists.
    for a in range(0, n, chunk):
        assert a in replay
    for i in list(range(10, 18)) + list(range(33, 41)) + list(range(50, 56)) + [70, 71, 72, 73]:
        assert i in replay, i
    assert all(got["per"][i] == [] and got["weak"][i] == [] for i in skipped)
    assert all(got["per"][i] for i in strong)

    # Off, the replay looks at every frame.
    looked.clear()
    measure.tracked(cache, settings.with_changes(adaptive_stride=False), fps=fps)
    assert sorted(looked) == list(range(n))
