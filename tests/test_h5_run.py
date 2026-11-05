"""`python -m faceblur.h5 run`: a DF2 file in, a de-identified episode.h5 out.

Every run here uses one worker, so no pool is started and the GPU holds one
detector process at most. The two-camera fixture is tests/data/df2_small.h5
(test_h5io.py says how it was made); the face test packs the conftest face
clip into a one-camera file of its own.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from faceblur import audit, batch, df2, h5, h5io
from faceblur.pipeline import AuditRecord
from faceblur.settings import Settings

from test_h5io import FIXTURE, assert_same_tree, decode_all

ALL_KINDS = ("face", "text", "plate", "screen")
CAMERA_DATASETS = (df2.VIDEO, df2.PTS, df2.DTS, df2.KEYFRAME, df2.TIMESTAMPS_NS,
                   df2.INTRINSICS, df2.EXTRINSICS)


def _vlen(obj) -> bool:
    return isinstance(obj, h5py.Dataset) and h5py.check_vlen_dtype(obj.dtype) is not None


def _holds_vlen(group: h5py.Group) -> bool:
    found = []
    group.visititems(lambda _, obj: found.append(True) if _vlen(obj) else None)
    return bool(found)


def assert_same_except(a: h5py.Group, b: h5py.Group, skip: tuple[str, ...] = (),
                       extra: tuple[str, ...] = ()) -> None:
    """`a` and `b` hold the same tree, leaving out the `skip` paths on both
    sides and the `extra` paths `b` may add. A group that leads to a skipped
    or extra path may carry more attrs in `b` (the root's faceblur_* ones);
    those of `a` must be there, equal. Everywhere else the attrs match exactly.

    Whole subtrees go to test_h5io.assert_same_tree. The camera packets are a
    vlen dataset numpy cannot compare in one go, so a subtree holding them is
    walked here and the packets compared one by one."""
    def path(group, name):
        return f"{group.name.rstrip('/')}/{name}".lstrip("/")

    here = a.name.strip("/")
    leads = here == "" or any(s.startswith(here + "/") for s in skip + extra)
    if not leads:
        assert set(a.attrs.keys()) == set(b.attrs.keys()), a.name
    for key in a.attrs:
        assert np.array_equal(np.asarray(a.attrs[key]), np.asarray(b.attrs[key])), f"{a.name} attr {key}"
    left = {n for n in a if path(a, n) not in skip}
    right = {n for n in b if path(b, n) not in skip and path(b, n) not in extra}
    assert left == right, a.name
    for name in left:
        x, y = a[name], b[name]
        p = path(a, name)
        if isinstance(x, h5py.Group):
            assert isinstance(y, h5py.Group), p
            if _holds_vlen(x) or any(s.startswith(p + "/") for s in skip + extra):
                assert_same_except(x, y, skip, extra)
            else:
                assert_same_tree(x, y)
        elif _vlen(x):
            assert y.dtype == x.dtype and y.shape == x.shape, p
            assert all(np.array_equal(u, v) for u, v in zip(x[()], y[()])), p
            assert set(x.attrs.keys()) == set(y.attrs.keys()), p
        else:
            assert x.dtype == y.dtype and x.shape == y.shape, p
            assert np.array_equal(x[()], y[()]), p
            assert set(x.attrs.keys()) == set(y.attrs.keys()), p
            for key in x.attrs:
                assert np.array_equal(np.asarray(x.attrs[key]), np.asarray(y.attrs[key])), f"{p} attr {key}"


def packet_sha(group: h5py.Group) -> str:
    h = hashlib.sha256()
    for packet in group[df2.VIDEO][:]:
        h.update(packet.tobytes())
    return h.hexdigest()


def decoded(group: h5py.Group, mp4: Path) -> list[np.ndarray]:
    h5io.read_video_group(group, mp4)
    return decode_all(mp4)


@pytest.fixture(scope="module")
def two_file(tmp_path_factory):
    out = tmp_path_factory.mktemp("run_default")
    # Outside a container: the header's image_digest must come out None.
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("FACEBLUR_IMAGE_DIGEST", raising=False)
        results = h5.run(FIXTURE, out, Settings(mask=ALL_KINDS), workers=1)
    return out / "episode.h5", results


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    out = tmp_path_factory.mktemp("run_spec")
    h5.run(FIXTURE, out, Settings(mask=ALL_KINDS), workers=1, spec_layout=True)
    return out / "episode.h5"


def test_release_has_no_audio_no_raw_video_and_no_links(two_file, tmp_path):
    path, _ = two_file
    assert path.exists()
    assert not (path.parent / ".work").exists()
    assert not (path.parent / "episode.h5.part").exists()
    with h5py.File(FIXTURE, "r") as src, h5py.File(path, "r") as out:
        assert not h5io.has_audio(out)
        assert h5io.has_video_outside(out, df2.IMAGE_GROUP) == []
        assert h5io.external_links(out) == []
        for camera in h5io.camera_groups(src):
            a, b = src[df2.camera_path(camera)], out[df2.camera_path(camera)]
            assert packet_sha(a) != packet_sha(b), camera
            assert len(decoded(b, tmp_path / f"{camera}_out.mp4")) == \
                len(decoded(a, tmp_path / f"{camera}_in.mp4")), camera


def test_every_other_group_is_copied_unchanged(two_file):
    path, _ = two_file
    with h5py.File(FIXTURE, "r") as src, h5py.File(path, "r") as out:
        assert_same_except(src, out, skip=(df2.IMAGE_GROUP, df2.AUDIO_GROUP),
                           extra=(df2.IMAGE_GROUP,))
        assert out.attrs["faceblur_release"]
        assert out.attrs["faceblur_version"] == AuditRecord.faceblur_version
        assert out.attrs["faceblur_layout"] == "two-file"


def test_every_camera_is_present_with_its_datasets_and_attrs(two_file):
    path, _ = two_file
    with h5py.File(FIXTURE, "r") as src, h5py.File(path, "r") as out:
        assert h5io.camera_groups(out) == h5io.camera_groups(src)
        for camera in h5io.camera_groups(src):
            a, b = src[df2.camera_path(camera)], out[df2.camera_path(camera)]
            for name in CAMERA_DATASETS:
                assert name in b, f"{camera}/{name}"
            assert set(b.keys()) == set(a.keys())
            assert set(b.attrs.keys()) == set(a.attrs.keys())
            for name in (df2.INTRINSICS, df2.EXTRINSICS):
                assert np.array_equal(a[name][()], b[name][()])
            for key in (df2.ATTR_WIDTH, df2.ATTR_HEIGHT, df2.ATTR_CODEC):
                assert a.attrs[key] == b.attrs[key]


def test_run_returns_per_frame_and_timestamps_for_each_camera(two_file):
    _, results = two_file
    with h5py.File(FIXTURE, "r") as src:
        assert [r["camera"] for r in results] == h5io.camera_groups(src)
        for r in results:
            n = len(src[df2.camera_path(r["camera"])][df2.TIMESTAMPS_NS])
            assert r["record"].status == "done"
            assert len(r["per_frame"]) == r["frames"] == n
            assert len(r["timestamps_ns"]) == n
            assert r["seconds"] > 0


def test_spec_layout_keeps_the_input_and_adds_the_sibling_group(spec):
    with h5py.File(FIXTURE, "r") as src, h5py.File(spec, "r") as out:
        assert_same_except(src, out, extra=(df2.RELEASE_IMAGE_GROUP,))
        assert h5io.has_audio(out)
        assert out.attrs["faceblur_layout"] == "spec"
        for camera in h5io.camera_groups(src):
            group = out[f"{df2.RELEASE_IMAGE_GROUP}/{camera}"]
            assert df2.VIDEO in group
            assert packet_sha(group) != packet_sha(src[df2.camera_path(camera)])


def test_a_face_is_masked_and_the_corners_are_not(video_with_face, tmp_path):
    fixture = tmp_path / "face.h5"
    h5.pack([video_with_face], ["rgb_ego_center_left"], None, fixture)
    out = tmp_path / "run"
    results = h5.run(fixture, out, Settings(mask=("face",)), workers=1)
    assert results[0]["record"].frames_with_mask > 0
    path = df2.camera_path("rgb_ego_center_left")
    with h5py.File(fixture, "r") as src, h5py.File(out / "episode.h5", "r") as dst:
        before = decoded(src[path], tmp_path / "in.mp4")[5].astype(np.int16)
        after = decoded(dst[path], tmp_path / "out.mp4")[5].astype(np.int16)
    face = np.abs(before[240:400, 240:400] - after[240:400, 240:400]).mean()
    assert face > 20
    for ys, xs in ((slice(0, 80), slice(0, 80)), (slice(0, 80), slice(560, 640)),
                   (slice(560, 640), slice(0, 80)), (slice(560, 640), slice(560, 640))):
        assert np.abs(before[ys, xs] - after[ys, xs]).mean() < 4


def test_a_failed_camera_leaves_no_file(monkeypatch, tmp_path, capsys):
    def failed(src, dst, settings, submit, **kwargs):
        return AuditRecord(source=Path(src).name, output=Path(dst).name,
                           status="failed", error="boom")

    monkeypatch.setattr(batch, "run_video", failed)
    out = tmp_path / "run"
    assert h5.main(["run", str(FIXTURE), "--out", str(out), "--workers", "1"]) == 1
    assert "boom" in capsys.readouterr().err
    assert not (out / "episode.h5").exists()
    assert not (out / "episode.h5.part").exists()
    assert not (out / ".work").exists()


def test_a_file_with_no_camera_is_refused(tmp_path, capsys):
    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as f:
        f.require_group(df2.META_GROUP)
    assert h5.main(["run", str(empty), "--out", str(tmp_path / "run"), "--workers", "1"]) == 2
    assert "no camera" in capsys.readouterr().err
    assert not (tmp_path / "run" / "episode.h5").exists()


def test_cli_prints_the_output_path(tmp_path, capsys):
    out = tmp_path / "run"
    assert h5.main(["run", str(FIXTURE), "--out", str(out), "--workers", "1"]) == 0
    printed = capsys.readouterr().out
    assert str(out / "episode.h5") in printed
    assert "ms/frame" in printed
    assert (out / "episode.h5").exists()


# --- audit.jsonl ----------------------------------------------------------------

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def audit_lines(out: Path) -> list[dict]:
    return audit.read(out / "audit.jsonl")


def of_type(lines: list[dict], kind: str) -> list[dict]:
    return [line for line in lines if line["type"] == kind]


def test_audit_verifies_and_every_line_matches_the_schema(two_file):
    path, _ = two_file
    lines = audit_lines(path.parent)
    assert audit.verify(path.parent / "audit.jsonl") == (True, len(lines), "")
    schema = audit.load_schema()
    for line in lines:
        audit.validate_line(line, schema)


def test_audit_header_names_the_input_settings_and_every_model(two_file):
    path, results = two_file
    header = audit_lines(path.parent)[0]
    assert header["type"] == "header"
    assert header["input_sha256"] == audit.sha256_file(FIXTURE)
    assert header["input_name"] == FIXTURE.name
    with h5py.File(FIXTURE, "r") as src:
        assert header["cameras"] == h5io.camera_groups(src)
    # Compared in its JSON form: the tuples of Settings come back as lists.
    expected = json.loads(audit.canonical(Settings(mask=ALL_KINDS).to_dict()))
    assert header["settings"] == expected
    models = header["models"]
    for r in results:
        assert set(r["record"].model_sha256) <= set(models)
        for name, digest in r["record"].model_sha256.items():
            assert models[name] == digest
    assert "silero_vad" in models
    assert all(HEX64.match(v) for v in models.values())
    assert header["image_digest"] is None
    assert header["tool_version"] == AuditRecord.faceblur_version


def test_audit_lists_the_speech_that_was_removed(two_file):
    path, _ = two_file
    speech = of_type(audit_lines(path.parent), "speech")
    assert len(speech) == 1
    assert speech[0]["start_s"] == 0.0
    assert 2.9 <= speech[0]["end_s"] <= 3.1
    assert speech[0]["pad_ms"] == 300


def test_audit_has_one_frame_line_per_frame_and_a_summary_per_camera(two_file):
    path, results = two_file
    lines = audit_lines(path.parent)
    frames = of_type(lines, "frame")
    assert len(frames) == sum(r["frames"] for r in results)
    for r in results:
        mine = [line for line in frames if line["camera"] == r["camera"]]
        assert [line["frame"] for line in mine] == list(range(r["frames"]))
        assert [line["pts_ns"] for line in mine] == [int(t) for t in r["timestamps_ns"]]
    for line in frames:
        assert set(line["counts"]) == set(audit.CLASSES)
        for kind, boxes in line["boxes"].items():
            assert len(boxes) == line["counts"][kind]
    done = of_type(lines, "camera_done")
    assert [d["camera"] for d in done] == [r["camera"] for r in results]
    for d, r in zip(done, results):
        assert d["record"]["status"] == "done"
        assert d["frames"] == r["frames"]
    footer = lines[-1]
    assert footer["type"] == "footer"
    assert footer["status"] == "done"
    assert footer["output_name"] == "episode.h5"
    assert footer["output_sha256"] == audit.sha256_file(path)


def test_audit_records_the_face_boxes(video_with_face, tmp_path):
    fixture = tmp_path / "face.h5"
    h5.pack([video_with_face], ["rgb_ego_center_left"], None, fixture)
    out = tmp_path / "run"
    h5.run(fixture, out, Settings(mask=("face",)), workers=1)
    lines = audit_lines(out)
    assert audit.verify(out / "audit.jsonl")[0]
    # Packed without audio: no speech, and no VAD model in the header.
    assert of_type(lines, "speech") == []
    assert "silero_vad" not in lines[0]["models"]
    with h5py.File(fixture, "r") as src:
        group = src[df2.camera_path("rgb_ego_center_left")]
        width, height = int(group.attrs[df2.ATTR_WIDTH]), int(group.attrs[df2.ATTR_HEIGHT])
    faces = [line for line in of_type(lines, "frame") if line["counts"]["face"] >= 1]
    assert faces
    for line in faces:
        for x, y, w, h in line["boxes"]["face"]:
            assert 0 <= x and 0 <= y and w > 0 and h > 0
            assert x + w <= width and y + h <= height


def test_a_failed_run_still_leaves_a_verifying_audit(monkeypatch, tmp_path):
    def failed(src, dst, settings, submit, **kwargs):
        return AuditRecord(source=Path(src).name, output=Path(dst).name,
                           status="failed", error="boom")

    monkeypatch.setattr(batch, "run_video", failed)
    monkeypatch.setenv("FACEBLUR_IMAGE_DIGEST", "sha256:" + "ab" * 32)
    out = tmp_path / "run"
    with pytest.raises(h5.RunFailed):
        h5.run(FIXTURE, out, Settings(mask=ALL_KINDS), workers=1)
    lines = audit_lines(out)
    assert audit.verify(out / "audit.jsonl") == (True, len(lines), "")
    assert lines[0]["image_digest"] == "sha256:" + "ab" * 32
    assert lines[-1]["type"] == "footer"
    assert lines[-1]["status"] == "failed"
    assert "boom" in lines[-1]["error"]
    assert lines[-1]["output_sha256"] == "0" * 64
    for line in lines:
        audit.validate_line(line)


def test_verify_command_passes_a_good_audit_and_fails_a_flipped_byte(two_file, tmp_path, capsys):
    path, _ = two_file
    good = path.parent / "audit.jsonl"
    assert h5.main(["verify", str(good)]) == 0
    assert "ok" in capsys.readouterr().out
    data = bytearray(good.read_bytes())
    # A digit inside the first frame line's pts or box, not its hash.
    at = data.index(b'"pts_ns":') + len(b'"pts_ns":')
    data[at] = ord("9") if data[at] != ord("9") else ord("8")
    bad = tmp_path / "audit.jsonl"
    bad.write_bytes(bytes(data))
    assert h5.main(["verify", str(bad)]) == 1
    assert "problem" in capsys.readouterr().out


def test_device_cpu_puts_every_model_on_the_cpu(tmp_path, capsys):
    # T15: the plate, zone and hand models follow --device too, not only the
    # face and text models, and the audit's compute says so.
    assert h5.main(["selfcheck", "--device", "cpu"]) == 0
    printed = capsys.readouterr().out
    providers = dict(re.findall(r"(?m)^(\w+): (\w+ExecutionProvider)$", printed))
    assert {"yunet", "ppocr", "lpd_yunet"} <= set(providers), printed
    assert set(providers.values()) == {"CPUExecutionProvider"}, printed
    out = tmp_path / "run"
    assert h5.main(["run", str(FIXTURE), "--out", str(out), "--device", "cpu",
                    "--workers", "1"]) == 0
    records = [line["record"] for line in of_type(audit_lines(out), "camera_done")]
    assert records
    for record in records:
        assert "lpd_yunet" in record["compute"], record["compute"]
        assert set(record["compute"].values()) == {"CPUExecutionProvider"}, record["compute"]
