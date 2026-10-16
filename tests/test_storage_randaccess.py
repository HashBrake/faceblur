"""Random access: seeded seeks with OpenCV and PyAV return exactly the sequential frame."""
from __future__ import annotations

import json

import av
import pytest

from conftest import run_ffmpeg
from storage import randaccess
from storage.encode import encode_file


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """300 frames of testsrc at 30 fps in H.264, and its libx265 copy (gop 60, B-frames)."""
    d = tmp_path_factory.mktemp("randaccess")
    src, dst = d / "src.mp4", d / "hevc.mp4"
    r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=320x240:rate=30:duration=10",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)])
    assert r.returncode == 0, r.stderr
    encode_file(src, dst, "hevc", "software", quality=20)
    return dst, src


@pytest.fixture(scope="module")
def result(pair):
    hevc, h264 = pair
    return randaccess.measure(hevc, h264, seeks=100, seed=0)


def test_hevc_copy_has_b_frames(pair):
    with av.open(str(pair[0])) as container:
        kinds = [int(f.pict_type) for f in container.decode(video=0)]
    assert kinds.count(3) >= 1


def test_every_seek_is_exact(result):
    for name in ("converted", "source"):
        for lib in ("opencv", "pyav"):
            entry = result["files"][name][lib]
            assert (entry["exact"], entry["total"]) == (100, 100), (name, lib, entry["offsets"])
    assert len(result["indices"]) == 100


def test_sequential_counts(result):
    for name in ("converted", "source"):
        for lib in ("opencv", "pyav"):
            assert result["files"][name][lib]["frames"] == 300


def test_keyframe_interval_of_the_copy(result):
    assert result["files"]["converted"]["pyav"]["keyframe_interval"] <= 60


def test_main_exit_and_json(pair, tmp_path, capsys, monkeypatch):
    # Timing is not asserted on a shared machine: the ratio bound is lifted so the
    # exit code here reflects exactness only.
    monkeypatch.setattr(randaccess, "MAX_RATIO", float("inf"))
    out = tmp_path / "ra.json"
    assert randaccess.main([str(pair[0]), "--source", str(pair[1]), "--seeks", "20",
                            "--seed", "1", "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {"seeks", "seed", "indices", "files", "ratio", "passed"} <= data.keys()
    assert set(data["files"]) == {"converted", "source"}
    assert set(data["ratio"]) == {"opencv", "pyav"}
    for lib in ("opencv", "pyav"):
        assert {"frames", "exact", "total", "mean_ms", "median_ms"} <= data["files"]["converted"][lib].keys()
    assert data["passed"] is True
    assert "ratio opencv" in capsys.readouterr().out
