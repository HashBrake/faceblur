"""The hot storage encode: HEVC with NVENC or libx265, keyframes, B-frames, audio."""
from __future__ import annotations

import av
import pytest
from av.video.frame import PictureType

from faceblur.video import VideoError, part_path
from storage import encode
from storage.ffmpeg import nvenc_works
from storage.probe import probe
from conftest import audio_codec

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
