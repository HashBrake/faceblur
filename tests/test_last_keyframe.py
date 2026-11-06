"""Sources whose last frame is a keyframe, so the last group of pictures is
one frame.

Found on an Orbbec camera file of 37,591 frames whose frame 37,590 is a
keyframe: the tail probe in run_video seeks to that frame, the millisecond
timestamp aims just past it, ffmpeg drops it as before the seek, nothing
comes out, and the whole camera failed with "Could not read this video".
"""
from __future__ import annotations

from pathlib import Path

from faceblur import h5
from faceblur.batch import run_video, serial_submit
from faceblur.pipeline import STATUS_DONE
from faceblur.segments import decode_range, frame_times, keyframes
from faceblur.settings import Settings
from faceblur.video import probe

REAL = Path(r"D:\Player One Internal videos\AZER764009K\Orbbec_Ego_AZER764009K_19700101_000048"
            r"\Orbbec_Ego_AZER764009K_19700101_000048_camera_left_part0001.mp4")


def test_the_fixture_ends_in_a_one_frame_gop(video_last_keyframe):
    times = frame_times(video_last_keyframe)
    assert len(times) == 20
    assert keyframes(video_last_keyframe, times)[-1] == 19


def test_decode_from_the_last_keyframe_yields_one_frame(video_last_keyframe):
    times = frame_times(video_last_keyframe)
    keys = keyframes(video_last_keyframe, times)
    info = probe(video_last_keyframe)
    frames = list(decode_range(video_last_keyframe, times, keys[-1], len(times), info))
    assert len(frames) == 1
    # It is the last frame: the same picture a decode of the whole clip ends on.
    whole = list(decode_range(video_last_keyframe, times, 0, len(times), info))
    assert len(whole) == 20
    assert (frames[0] == whole[-1]).all()


def test_run_video_writes_every_frame(video_last_keyframe, tmp_path):
    out = tmp_path / "last_keyframe_blurred.mp4"
    record = run_video(video_last_keyframe, out, Settings(mask=("face",)), serial_submit,
                       workers=1)
    assert record.status == STATUS_DONE, record.error
    assert record.frames == 20
    assert record.source_frames_dropped == 0
    assert len(frame_times(out)) == 20


def test_the_h5_runner_finishes_with_every_frame(video_last_keyframe, tmp_path):
    fixture = tmp_path / "last_keyframe.h5"
    h5.pack([video_last_keyframe], ["rgb_ego_center_left"], None, fixture)
    results = h5.run(fixture, tmp_path / "run", Settings(mask=("face",)), workers=1)
    record = results[0]["record"]
    assert record.status == STATUS_DONE, record.error
    assert record.frames == 20
