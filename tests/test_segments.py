"""Segment planning, and the phased pipeline end to end on a short clip."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.batch import run_video, serial_submit
from faceblur.pipeline import STATUS_DONE, sidecar_path
from faceblur.segments import Segment, frame_times, keyframes, plan_segments, split_long, verify
from faceblur.settings import Settings
from faceblur.video import part_path

FAST = Settings(det_sizes=(320,), verify=False, min_track=1, tail=1, chunk_seconds=0.5,
                min_copy_seconds=0.3)


def test_plan_copies_clean_gops_and_encodes_masked_ones():
    masked = [False] * 30
    masked[12] = True
    keys = [0, 10, 20]
    segs = plan_segments(masked, keys, 30, min_copy_frames=5)
    assert segs == [Segment(0, 10, True), Segment(10, 20, False), Segment(20, 30, True)]


def test_plan_merges_neighbouring_runs_of_one_kind():
    masked = [False] * 40
    masked[15] = True
    masked[25] = True
    keys = [0, 10, 20, 30]
    segs = plan_segments(masked, keys, 40, min_copy_frames=5)
    assert segs == [Segment(0, 10, True), Segment(10, 30, False), Segment(30, 40, True)]


def test_short_clean_runs_are_encoded_with_their_neighbours():
    masked = [False] * 30
    masked[5] = True
    masked[25] = True
    keys = [0, 10, 20]
    segs = plan_segments(masked, keys, 30, min_copy_frames=15)
    assert segs == [Segment(0, 30, False)]


def test_plan_covers_every_frame_exactly_once():
    rng = np.random.default_rng(3)
    masked = list(rng.random(500) < 0.1)
    keys = sorted(set(rng.integers(0, 500, 40).tolist()) | {0})
    segs = plan_segments(masked, keys, 500, min_copy_frames=20)
    assert segs[0].start == 0 and segs[-1].end == 500
    assert all(a.end == b.start for a, b in zip(segs, segs[1:]))
    assert all(s.frames > 0 for s in segs)
    for s in segs:
        if s.copy:
            assert not any(masked[s.start:s.end])


def test_copy_clean_off_yields_one_encode_segment():
    segs = plan_segments([False] * 30, [0, 10, 20], 30, 5, copy_clean=False)
    assert segs == [Segment(0, 30, False)]


def test_frame_times_and_keyframes_of_a_clip(video_with_audio):
    times = frame_times(video_with_audio)
    assert len(times) == 30
    assert times == sorted(times)
    keys = keyframes(video_with_audio, times)
    assert keys[0] == 0
    assert all(0 <= k < 30 for k in keys)


def test_phased_run_writes_a_verified_file(video_with_audio, tmp_path):
    dst = tmp_path / "out.mp4"
    record = run_video(video_with_audio, dst, FAST, serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.frames == 30
    assert dst.exists()
    assert not part_path(dst).exists()
    assert verify(dst, 30) == ""
    assert sidecar_path(dst).exists()
    # The clip holds no faces, so every frame was copied and matches the source.
    assert record.frames_copied == 30
    a, b = cv2.VideoCapture(str(video_with_audio)), cv2.VideoCapture(str(dst))
    n = 0
    while True:
        ok1, x = a.read()
        ok2, y = b.read()
        if not (ok1 and ok2):
            break
        assert np.array_equal(x, y)
        n += 1
    assert n == 30


def test_phased_run_keeps_the_audio(video_with_audio, tmp_path):
    from conftest import audio_codec
    dst = tmp_path / "out.mp4"
    record = run_video(video_with_audio, dst, FAST, serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert audio_codec(dst) == "aac"


def test_phased_run_without_copying_encodes_everything(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    record = run_video(video_silent, dst, FAST.with_changes(copy_clean=False), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.frames_copied == 0
    assert verify(dst, 20) == ""


def test_phased_run_skips_an_existing_output(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    dst.write_bytes(b"x")
    assert run_video(video_silent, dst, FAST, serial_submit).status == "skipped"


def test_phased_run_reports_progress_for_both_phases(video_silent, tmp_path):
    seen = []
    run_video(video_silent, tmp_path / "out.mp4", FAST, serial_submit,
              on_progress=lambda s, d, n: seen.append((s, d, n)))
    stages = [s for s, _, _ in seen]
    assert "detecting" in stages and "writing" in stages
    assert seen[-1][1] == seen[-1][2]


def test_a_broken_file_fails_cleanly(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    record = run_video(broken, tmp_path / "out.mp4", FAST, serial_submit)
    assert record.status == "failed"
    assert "Could not read this video" in record.error


def test_long_encode_segments_split_at_keyframes():
    segs = [Segment(0, 100, False), Segment(100, 130, True)]
    out = split_long(segs, keys=[0, 30, 60, 90, 100, 120], max_frames=35)
    assert out == [Segment(0, 30, False), Segment(30, 60, False), Segment(60, 90, False),
                   Segment(90, 100, False), Segment(100, 130, True)]


def test_split_leaves_a_segment_without_inner_keyframes_alone():
    assert split_long([Segment(0, 100, False)], keys=[0], max_frames=35) == [Segment(0, 100, False)]


def test_progress_ticks_inside_a_range_not_only_at_its_end(video_long, tmp_path):
    """The window shows movement within a range, not one jump per range."""
    seen = []
    run_video(video_long, tmp_path / "out.mp4", FAST.with_changes(chunk_seconds=100),
              serial_submit, on_progress=lambda s, d, n: seen.append((s, d)))
    detect = [d for s, d in seen if s == "detecting"]
    assert detect[0] == 0
    assert len(set(detect)) > 3
    assert detect == sorted(detect)


# ---- pieces only join cleanly when they share one reordering delay

def _clip_with_b_frames(path):
    from conftest import run_ffmpeg
    r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=160x120:rate=10:duration=3",
                    "-c:v", "libx264", "-preset", "veryfast", "-bf", "2", "-g", "10",
                    "-pix_fmt", "yuv420p", str(path)])
    assert r.returncode == 0, r.stderr
    return path


def test_reorder_delay_is_zero_without_b_frames(video_silent):
    from faceblur.segments import reorder_delay
    assert reorder_delay(video_silent) == 0


def test_reorder_delay_counts_the_b_frame_hold_back(tmp_path):
    from faceblur.segments import reorder_delay
    assert reorder_delay(_clip_with_b_frames(tmp_path / "bf.mp4")) >= 1


def test_a_source_with_b_frames_is_encoded_whole(tmp_path):
    """No copied pieces: they would carry a delay the encoded pieces lack."""
    src = _clip_with_b_frames(tmp_path / "bf.mp4")
    dst = tmp_path / "out.mp4"
    record = run_video(src, dst, FAST, serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.reorder_delay >= 1
    assert record.frames_copied == 0
    assert record.join_attempts == 1
    assert verify(dst, 30, 2.9) == ""


def test_encoded_pieces_carry_no_b_frames():
    from faceblur.video import video_codec_args
    assert video_codec_args("x264", 12, "fast", 16)[-2:] == ["-bf", "0"]
    assert video_codec_args("nvenc", 12, "fast", 16)[-2:] == ["-bf", "0"]


def test_a_retry_after_a_bad_join_reports_done(video_silent, tmp_path, monkeypatch):
    """The first join fails its check, the second passes: the record says done
    and carries no error from the first attempt."""
    import faceblur.batch as batch
    calls = []
    real = batch.verify

    def flaky(part, n, seconds=None):
        calls.append(1)
        return "length 1.000 s, expected 2.000 s" if len(calls) == 1 else real(part, n, seconds)

    monkeypatch.setattr(batch, "verify", flaky)
    dst = tmp_path / "out.mp4"
    record = run_video(video_silent, dst, FAST, serial_submit)
    assert record.status == STATUS_DONE
    assert record.error == ""
    assert record.join_attempts == 2
    assert record.frames_copied == 0
    assert dst.exists() and sidecar_path(dst).exists()
