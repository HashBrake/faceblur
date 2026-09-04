"""process_video end to end: audio passthrough, progress, cancel, audit record."""
from __future__ import annotations

import json
import threading
import time

import pytest

from conftest import audio_codec, audio_streams, stream_lines
from faceblur import STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, STATUS_STOPPED
from faceblur.pipeline import output_path, process_video, sidecar_path
from faceblur.settings import Settings
from faceblur.video import part_path

FAST = Settings(det_sizes=(320,), verify=False, min_track=1, tail=1)


def test_audio_passes_through_untouched(video_with_audio, tmp_path):
    """Phase 1 verify: the output holds one audio stream in the source codec."""
    assert audio_codec(video_with_audio) == "aac"

    dst = tmp_path / "out.mp4"
    record = process_video(video_with_audio, dst, FAST)

    assert record.status == STATUS_DONE, record.error
    assert dst.exists()
    assert len(audio_streams(dst)) == 1
    assert audio_codec(dst) == "aac"


def test_a_source_without_audio_gives_an_output_without_audio(video_silent, tmp_path):
    """The Ego camera files carry no audio track."""
    dst = tmp_path / "out.mp4"
    record = process_video(video_silent, dst, FAST)
    assert record.status == STATUS_DONE, record.error
    assert audio_streams(dst) == []
    assert any("Video:" in line for line in stream_lines(dst))


def test_frame_count_and_size_survive(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    record = process_video(video_silent, dst, FAST)
    assert record.status == STATUS_DONE, record.error
    assert record.frames == 20
    assert record.resolution == "320x240"


def test_the_sidecar_holds_the_audit_record(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    record = process_video(video_silent, dst, FAST)
    side = sidecar_path(dst)
    assert side.exists()

    saved = json.loads(side.read_text(encoding="utf-8"))
    assert saved["source"] == video_silent.name
    assert saved["output"] == "out.mp4"
    assert saved["status"] == STATUS_DONE
    assert saved["resolution"] == "320x240"
    assert saved["engine"] == "yunet"
    assert saved["settings"]["det_sizes"] == [320]
    assert saved["model_sha256"]["yunet"].startswith("c0aa2a66")
    assert saved["wall_seconds"] > 0
    assert saved == record.to_dict()


def test_progress_reports_both_passes(video_silent, tmp_path):
    seen = []
    process_video(video_silent, tmp_path / "out.mp4", FAST,
                  on_progress=lambda stage, done, total: seen.append((stage, done, total)))

    stages = [s for s, _, _ in seen]
    assert "detecting" in stages and "writing" in stages
    assert stages.index("detecting") < stages.index("writing")
    detecting = [d for s, d, _ in seen if s == "detecting"]
    assert detecting == sorted(detecting)
    assert detecting[-1] == 20


def test_cancel_stops_quickly_and_leaves_no_file(video_long, tmp_path):
    """Phase 1 verify: cancel stops within 2 seconds and leaves no output file."""
    dst = tmp_path / "out.mp4"
    cancel = threading.Event()
    started = threading.Event()

    def trip():
        started.wait(10)
        time.sleep(0.4)
        cancel.set()

    waker = threading.Thread(target=trip, daemon=True)
    waker.start()

    def on_progress(stage, done, total):
        started.set()

    t0 = time.time()
    record = process_video(video_long, dst, FAST, on_progress=on_progress, cancel=cancel)
    elapsed = time.time() - t0

    assert record.status == STATUS_STOPPED
    assert elapsed - 0.4 < 2.0, f"took {elapsed:.1f}s to stop"
    assert not dst.exists()
    assert not sidecar_path(dst).exists()


def test_cancel_during_the_write_pass_leaves_no_file(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    cancel = threading.Event()

    def on_progress(stage, done, total):
        if stage == "writing" and done >= 3:
            cancel.set()

    record = process_video(video_silent, dst, FAST, on_progress=on_progress, cancel=cancel)
    assert record.status == STATUS_STOPPED
    assert not dst.exists()


def test_an_unreadable_file_fails_without_raising(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    record = process_video(broken, tmp_path / "out.mp4", FAST)

    assert record.status == STATUS_FAILED
    assert "Could not read this video" in record.error
    assert not (tmp_path / "out.mp4").exists()


def test_an_existing_output_is_skipped(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    dst.write_bytes(b"already here")
    record = process_video(video_silent, dst, FAST)
    assert record.status == STATUS_SKIPPED
    assert dst.read_bytes() == b"already here"


def test_replace_existing_overwrites(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    dst.write_bytes(b"already here")
    record = process_video(video_silent, dst, FAST.with_changes(replace_existing=True))
    assert record.status == STATUS_DONE, record.error
    assert dst.read_bytes() != b"already here"


def test_output_path_names_the_blurred_copy(tmp_path):
    assert output_path(tmp_path / "clip.mov", tmp_path / "out").name == "clip_blurred.mp4"
    assert output_path(tmp_path / "clip.mp4", tmp_path / "out", "_x").name == "clip_x.mp4"


@pytest.mark.parametrize("mode", ["blur", "pixelate", "solid"])
def test_every_mode_writes_a_playable_file(video_silent, tmp_path, mode):
    dst = tmp_path / f"{mode}.mp4"
    record = process_video(video_silent, dst, FAST.with_changes(mode=mode))
    assert record.status == STATUS_DONE, record.error
    assert dst.stat().st_size > 0
    assert any("Video: h264" in line for line in stream_lines(dst))


def test_no_part_file_remains_after_success(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    record = process_video(video_silent, dst, FAST)
    assert record.status == STATUS_DONE, record.error
    assert dst.exists()
    assert not part_path(dst).exists()


def test_no_part_file_remains_after_cancel(video_silent, tmp_path):
    dst = tmp_path / "out.mp4"
    cancel = threading.Event()

    def on_progress(stage, done, total):
        if stage == "writing" and done >= 2:
            cancel.set()

    process_video(video_silent, dst, FAST, on_progress=on_progress, cancel=cancel)
    assert not dst.exists()
    assert not part_path(dst).exists()


def test_the_finished_file_appears_only_at_the_end(video_silent, tmp_path):
    """While writing, only the .part file exists. The real name appears once."""
    dst = tmp_path / "out.mp4"
    seen = []

    def on_progress(stage, done, total):
        if stage == "writing":
            seen.append((dst.exists(), part_path(dst).exists()))

    record = process_video(video_silent, dst, FAST, on_progress=on_progress)
    assert record.status == STATUS_DONE, record.error
    assert all(not final for final, _ in seen)
    assert any(part for _, part in seen)
    assert dst.exists() and not part_path(dst).exists()
