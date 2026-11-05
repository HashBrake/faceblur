"""DF2 layout, HDF5 I/O primitives and `python -m faceblur.h5 pack`.

The committed fixture tests/data/df2_small.h5 was built with these commands,
run from the repository root (out/ is gitignored, so the inputs stay out of git):

    FF=$(.venv/Scripts/python.exe -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
    mkdir -p out/df2_fixture
    $FF -y -f lavfi -i testsrc=size=320x240:rate=10:duration=1.2 \
        -c:v libx264 -preset ultrafast -pix_fmt yuv420p out/df2_fixture/left.mp4
    $FF -y -f lavfi -i testsrc2=size=320x240:rate=10:duration=1.2 \
        -c:v libx264 -preset ultrafast -pix_fmt yuv420p out/df2_fixture/right.mp4
    .venv/Scripts/python.exe -m faceblur.h5 pack out/df2_fixture/left.mp4 \
        out/df2_fixture/right.mp4 --audio tests/data/speech_16k.wav --out tests/data/df2_small.h5

Its audio is the committed speech sample (tests/test_vad.py says how that was
made), 75200 samples of 16 kHz mono, so the VAD has speech to find in it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from faceblur import df2, h5io
from faceblur.h5 import main as h5_main

from conftest import DATA

FIXTURE = DATA / "df2_small.h5"
LEFT = df2.camera_path("rgb_ego_center_left")
RIGHT = df2.camera_path("rgb_ego_center_right")


def decode_all(mp4: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def assert_same_tree(a: h5py.Group, b: h5py.Group) -> None:
    """Every member of `a` is in `b` with the same dtype, shape, values and attrs, and nothing more."""
    assert set(a.keys()) == set(b.keys()), a.name
    assert set(a.attrs.keys()) == set(b.attrs.keys()), a.name
    for key in a.attrs:
        assert np.array_equal(np.asarray(a.attrs[key]), np.asarray(b.attrs[key])), f"{a.name} attr {key}"
    for name in a:
        x, y = a[name], b[name]
        if isinstance(x, h5py.Group):
            assert isinstance(y, h5py.Group), y.name
            assert_same_tree(x, y)
        else:
            assert isinstance(y, h5py.Dataset), y.name
            assert x.dtype == y.dtype, x.name
            assert x.shape == y.shape, x.name
            assert np.array_equal(x[()], y[()]), x.name
            assert set(x.attrs.keys()) == set(y.attrs.keys()), x.name
            for key in x.attrs:
                assert np.array_equal(np.asarray(x.attrs[key]), np.asarray(y.attrs[key])), f"{x.name} attr {key}"


@pytest.fixture(scope="module")
def packed_pair(tmp_path_factory, video_silent, video_with_audio, sine_wav) -> Path:
    out = tmp_path_factory.mktemp("df2") / "pair.h5"
    assert h5_main(["pack", str(video_silent), str(video_with_audio),
                    "--audio", str(sine_wav), "--out", str(out)]) == 0
    return out


@pytest.mark.parametrize("which", ["video_silent", "video_with_audio"])
def test_round_trip_gives_the_same_frames(which, request, tmp_path):
    source = request.getfixturevalue(which)
    h5_path = tmp_path / "one.h5"
    assert h5_main(["pack", str(source), "--out", str(h5_path)]) == 0
    remuxed = tmp_path / "remuxed.mp4"
    with h5py.File(h5_path, "r") as f:
        group = f[LEFT]
        h5io.read_video_group(group, remuxed)
        pts = group[df2.PTS][:]
        ns = group[df2.TIMESTAMPS_NS][:]
        fps = group.attrs[df2.ATTR_FPS]
        frame_count = int(group.attrs[df2.ATTR_FRAME_COUNT])

    want, got = decode_all(source), decode_all(remuxed)
    assert len(want) == len(got) > 0
    assert all(np.array_equal(w, g) for w, g in zip(want, got))
    assert np.all(np.diff(pts) > 0)
    assert frame_count == len(got)
    assert ns[1] - ns[0] == pytest.approx(1e9 / fps, rel=1e-3)


def test_remux_keeps_keyframes_and_extradata(video_silent, tmp_path):
    import av

    h5_path = tmp_path / "one.h5"
    h5_main(["pack", str(video_silent), "--out", str(h5_path)])
    remuxed = tmp_path / "remuxed.mp4"
    with h5py.File(h5_path, "r") as f:
        h5io.read_video_group(f[LEFT], remuxed)
        keyframe = f[LEFT][df2.KEYFRAME][:]
        extradata = bytes(f[LEFT].attrs[df2.ATTR_EXTRADATA])
    with av.open(str(remuxed)) as c:
        stream = c.streams.video[0]
        assert bytes(stream.codec_context.extradata) == extradata
        flags = [p.is_keyframe for p in c.demux(stream) if p.size]
    assert flags == list(keyframe)
    assert flags[0]


def test_two_mp4s_get_the_ego_centre_names(packed_pair):
    with h5py.File(packed_pair, "r") as f:
        assert LEFT in f and RIGHT in f
        assert h5io.camera_groups(f) == ["rgb_ego_center_left", "rgb_ego_center_right"]


def test_cameras_flag_overrides_names(video_silent, video_with_audio, tmp_path):
    out = tmp_path / "named.h5"
    # Given out of CAMERAS order, plus one name the layout does not know.
    h5_main(["pack", str(video_silent), str(video_with_audio), str(video_silent),
             "--cameras", "zz_extra,rgb_wrist_right,rgb_wrist_left", "--out", str(out)])
    with h5py.File(out, "r") as f:
        assert h5io.camera_groups(f) == ["rgb_wrist_left", "rgb_wrist_right", "zz_extra"]
        assert LEFT not in f


def test_audio_is_int16_pcm(packed_pair, video_silent, tmp_path):
    with h5py.File(packed_pair, "r") as f:
        assert h5io.has_audio(f)
        pcm = f[f"{df2.AUDIO_GROUP}/{df2.AUDIO_PCM}"]
        assert pcm.dtype == np.int16
        assert pcm.ndim == 2 and pcm.shape[1] == 1
        assert abs(pcm.shape[0] - 16000) <= 400
        assert pcm.attrs[df2.ATTR_SAMPLE_RATE] == 16000
        assert pcm.attrs[df2.ATTR_CHANNELS] == 1
        assert np.abs(pcm[()]).max() > 1000  # a tone, not silence
    silent = tmp_path / "silent.h5"
    h5_main(["pack", str(video_silent), "--out", str(silent)])
    with h5py.File(silent, "r") as f:
        assert not h5io.has_audio(f)


def test_external_links_are_found(tmp_path):
    with h5py.File(FIXTURE, "r") as f:
        assert h5io.external_links(f) == []
    linked = tmp_path / "linked.h5"
    with h5py.File(FIXTURE, "r") as src, h5py.File(linked, "w") as dst:
        h5io.copy_passthrough(src, dst)
        dst["dataset/meta/outside"] = h5py.ExternalLink("elsewhere.h5", "/secret")
        dst["dataset/observation/state/alias"] = h5py.SoftLink("/dataset/meta/scene")
    with h5py.File(linked, "r") as f:
        assert sorted(h5io.external_links(f)) == ["dataset/meta/outside", "dataset/observation/state/alias"]


def test_copy_passthrough_carries_everything_but_image_and_audio(tmp_path):
    out = tmp_path / "pass.h5"
    with h5py.File(FIXTURE, "r") as src, h5py.File(out, "w") as dst:
        h5io.copy_passthrough(src, dst)
    with h5py.File(FIXTURE, "r") as src, h5py.File(out, "r") as dst:
        assert df2.IMAGE_GROUP not in dst
        assert df2.AUDIO_GROUP not in dst
        assert dict(dst.attrs) == dict(src.attrs)
        for path in (df2.META_GROUP, df2.ACCESSORIES_GROUP, df2.TIMELINE_GROUP, df2.STATE_GROUP, df2.ENV_GROUP):
            assert_same_tree(src[path], dst[path])
        # Nothing else under observation either.
        assert set(dst[df2.OBSERVATION_GROUP].keys()) == {"timeline", "state", "env"}


def test_video_outside_the_allowed_group_is_named(tmp_path):
    with h5py.File(FIXTURE, "r") as f:
        assert h5io.has_video_outside(f, df2.IMAGE_GROUP) == []
    planted = tmp_path / "planted.h5"
    with h5py.File(FIXTURE, "r") as src, h5py.File(planted, "w") as dst:
        h5io.copy_passthrough(src, dst, skip=())
        dst.create_dataset(f"{df2.ACCESSORIES_GROUP}/mocap/{df2.VIDEO}", data=np.zeros(3, np.uint8))
    with h5py.File(planted, "r") as f:
        assert h5io.has_video_outside(f, df2.IMAGE_GROUP) == [f"{df2.ACCESSORIES_GROUP}/mocap/{df2.VIDEO}"]


def test_committed_fixture_is_complete(tmp_path):
    assert FIXTURE.stat().st_size < 1_000_000
    with h5py.File(FIXTURE, "r") as f:
        assert f.attrs["df2_version"] == "fixture-2026-11"
        assert h5io.camera_groups(f) == ["rgb_ego_center_left", "rgb_ego_center_right"]
        assert h5io.has_audio(f)
        assert len(f[df2.TIMELINE_GROUP][df2.TIMELINE_TIMESTAMPS_NS]) == 12
        for path in df2.PASSTHROUGH_GROUPS:
            assert any(isinstance(obj, h5py.Dataset) for obj in f[path].values()), path
        frames = {}
        for name in h5io.camera_groups(f):
            group = f[df2.camera_path(name)]
            assert len(group[df2.VIDEO]) == 12
            mp4 = tmp_path / f"{name}.mp4"
            h5io.read_video_group(group, mp4)
            frames[name] = decode_all(mp4)
            assert len(frames[name]) == 12
    # The two cameras show different pictures.
    assert not np.array_equal(frames["rgb_ego_center_left"][0], frames["rgb_ego_center_right"][0])


def test_pack_runs_as_a_module(video_silent, tmp_path):
    out = tmp_path / "cli.h5"
    result = subprocess.run([sys.executable, "-m", "faceblur.h5", "pack", str(video_silent), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert LEFT in result.stdout
    assert out.is_file()


def test_packets_stream_into_a_resizable_dataset(video_long, tmp_path, monkeypatch):
    # 120 packets in batches of 50: two full batches and a partial last one.
    monkeypatch.setattr(h5io, "BATCH", 50)
    h5_path = tmp_path / "long.h5"
    assert h5_main(["pack", str(video_long), "--out", str(h5_path)]) == 0
    remuxed = tmp_path / "long.mp4"
    with h5py.File(h5_path, "r") as f:
        group = f[LEFT]
        for name in (df2.VIDEO, df2.PTS, df2.DTS, df2.KEYFRAME):
            assert group[name].chunks is not None, name
            assert group[name].maxshape == (None,), name
        assert int(group.attrs[df2.ATTR_FRAME_COUNT]) == len(group[df2.VIDEO]) == len(group[df2.TIMESTAMPS_NS])
        h5io.read_video_group(group, remuxed)
    want, got = decode_all(video_long), decode_all(remuxed)
    assert len(want) == len(got) == 120
    assert all(np.array_equal(w, g) for w, g in zip(want, got))


def test_pack_of_an_unreadable_file_fails_in_one_line_and_leaves_nothing(tmp_path, capsys):
    import struct

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(struct.pack(">I", 24) + b"ftypisom" + struct.pack(">I", 512) + b"isomiso2")
    assert bad.stat().st_size == 24
    out = tmp_path / "bad.h5"
    assert h5_main(["pack", str(bad), "--out", str(out)]) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "cannot read" in err and "bad.mp4" in err
    assert not out.exists()
    assert not (tmp_path / "bad.h5.part").exists()


def test_pack_refuses_a_resource_fork_before_opening_it(tmp_path, capsys, monkeypatch):
    fork = tmp_path / "._x.mp4"
    fork.write_bytes(b"\x00\x05\x16\x07")
    opened = []
    monkeypatch.setattr(h5io, "write_video_group", lambda *a: opened.append(a))
    out = tmp_path / "fork.h5"
    assert h5_main(["pack", str(fork), "--out", str(out)]) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "._x.mp4" in err
    assert opened == []
    assert not out.exists() and not (tmp_path / "fork.h5.part").exists()
