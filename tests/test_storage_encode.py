"""The hot storage encode: HEVC with NVENC or libx265, keyframes, B-frames, audio."""
from __future__ import annotations

import av
import pytest
from av.video.frame import PictureType

from faceblur.video import VideoError, part_path
from storage import checks, encode
from storage.ffmpeg import nvenc_works
from storage.probe import frame_pts_seconds, probe
from conftest import audio_codec, run_ffmpeg

EXPECTED_ENCODER = {
    ("hevc", "nvenc"): "hevc_nvenc",
    ("hevc", "software"): "libx265",
    ("av1", "nvenc"): "av1_nvenc",
    ("av1", "software"): "libaom-av1",
}


@pytest.mark.parametrize("codec", encode.CODECS)
@pytest.mark.parametrize("encoder", ["nvenc", "software"])
def test_codec_args(codec, encoder):
    args = encode.codec_args(codec, encoder, 24)
    assert args[args.index("-c:v") + 1] == EXPECTED_ENCODER[(codec, encoder)]
    assert args[args.index("-g") + 1] == "60"
    if codec == "hevc":
        assert args[args.index("-tag:v") + 1] == "hvc1"


def test_resolve_encoder():
    assert encode.resolve_encoder("hevc", "software") == "software"
    assert encode.resolve_encoder("hevc", "nvenc") == "nvenc"
    assert encode.resolve_encoder("hevc", "auto") in ("nvenc", "software")


def _frame_types(path):
    """(key_frame flags, picture types) of every decoded frame."""
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    return [f.key_frame for f in frames], [f.pict_type for f in frames]


def _assert_stored_copy(src, dst):
    facts, src_facts = probe(dst), probe(src)
    assert facts.codec == "hevc"
    assert facts.frames == 120
    assert abs(facts.duration_s - src_facts.duration_s) <= 1 / src_facts.fps
    keys, types = _frame_types(dst)
    key_at = [i for i, k in enumerate(keys) if k]
    assert len(key_at) >= 2
    assert key_at[0] == 0
    gaps = [b - a for a, b in zip(key_at, key_at[1:])] + [len(keys) - key_at[-1]]
    assert max(gaps) <= encode.GOP
    assert PictureType.B in types


def test_encode_software(video_long, tmp_path):
    dst = tmp_path / "out.mp4"
    result = encode.encode_file(video_long, dst, "hevc", "software", quality=28)
    assert dst.is_file()
    assert not part_path(dst).exists()
    assert result.encoder == "software"
    assert result.size_bytes == dst.stat().st_size
    _assert_stored_copy(video_long, dst)


@pytest.mark.skipif(not nvenc_works("hevc_nvenc"), reason="no HEVC NVENC on this machine")
def test_encode_nvenc(video_long, tmp_path):
    dst = tmp_path / "out.mp4"
    result = encode.encode_file(video_long, dst, "hevc", "nvenc", quality=28)
    assert result.encoder == "nvenc"
    assert not part_path(dst).exists()
    _assert_stored_copy(video_long, dst)


def test_encode_failure_leaves_nothing(tmp_path):
    dst = tmp_path / "out.mp4"
    with pytest.raises(VideoError):
        encode.encode_file(tmp_path / "missing.mp4", dst, "hevc", "software")
    assert not dst.exists()
    assert not part_path(dst).exists()


def test_encode_keeps_audio(video_with_audio, tmp_path):
    dst = tmp_path / "out.mp4"
    encode.encode_file(video_with_audio, dst, "hevc", "software", quality=28)
    assert audio_codec(dst) == "aac"


@pytest.mark.parametrize("codec", encode.CODECS)
def test_nvenc_sets_maxrate_software_does_not(codec):
    nvenc = encode.codec_args(codec, "nvenc", 28)
    software = encode.codec_args(codec, "software", 28)
    assert "-maxrate" in nvenc and "-bufsize" in nvenc
    assert "-maxrate" not in software and "-bufsize" not in software


def test_encode_keeps_source_timestamps(tmp_path):
    """The camera's 30.0085 fps on time base 1/90000 keeps every frame's pts.

    ffmpeg rounds this rate to 30/1 for r_frame_rate; an encoder that takes its
    time base from that writes exactly 30 fps and the copy drifts.
    """
    src = tmp_path / "camera_rate.mp4"
    r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=320x240:rate=119130000/3969853:duration=4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-video_track_timescale", "90000", str(src)])
    assert r.returncode == 0, r.stderr
    dst = tmp_path / "out.mp4"
    encode.encode_file(src, dst, "hevc", "software", quality=28)
    assert frame_pts_seconds(dst) == pytest.approx(frame_pts_seconds(src), abs=1e-6)
    assert probe(dst).duration_s == pytest.approx(probe(src).duration_s, abs=1e-6)
    assert checks.duration(probe(src), probe(dst)).passed
