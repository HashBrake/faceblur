"""What the source carried besides pictures, and what the copy carries on.

Package M1. Two questions an auditor asks about a file they cannot open: was
there sound, and what did the container know. The record answers both, and
answers the second with **names only**, because a metadata tag is where a
camera writes a location, a device serial or an account name, and a record
that copied those would be the leak it exists to rule out.

The last test here is the one that matters most. It is not about the record
at all: it checks that the blurred copy does not inherit the source's tags,
which would put a location in the file this tool hands over.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conftest import run_ffmpeg
from faceblur.batch import run_video, serial_submit
from faceblur.pipeline import STATUS_DONE, sidecar_path
from faceblur.settings import Settings
from faceblur.video import ffmpeg_exe, has_audio, probe, source_metadata

FAST = dict(device="cpu", copy_clean=False, chunk_seconds=0, encode_seconds=0)
# A tag with something in it nobody should be handed on. The name is ordinary
# and the value is the kind of thing a camera writes.
SECRET = "51.5007,-0.1246"


@pytest.fixture(scope="module")
def tagged(tmp_path_factory) -> Path:
    """A short clip carrying a location tag and a comment."""
    out = tmp_path_factory.mktemp("meta") / "tagged.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                         "-i", "color=size=320x240:rate=10:duration=1",
                         "-metadata", f"location={SECRET}",
                         "-metadata", f"comment=filmed by somebody, {SECRET}",
                         "-c:v", "libx264", "-preset", "ultrafast",
                         "-pix_fmt", "yuv420p", str(out)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-400:])
    return out


@pytest.fixture(scope="module")
def tagged_no_bframes(tmp_path_factory) -> Path:
    """The same, without B-frames, so the run is allowed to copy stretches."""
    out = tmp_path_factory.mktemp("meta") / "tagged_copy.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                         "-i", "color=size=320x240:rate=10:duration=2",
                         "-metadata", f"location={SECRET}",
                         "-c:v", "libx264", "-preset", "ultrafast", "-bf", "0",
                         "-g", "5", "-pix_fmt", "yuv420p", str(out)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-400:])
    return out


def tags_of(path: Path) -> str:
    """Everything ffmpeg will say about a file's metadata, as one string."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(path),
                        "-map_metadata", "0", "-f", "ffmetadata", "-"],
                       capture_output=True, timeout=120)
    return r.stdout.decode(errors="replace")


# ------------------------------------------------------------- what is read

def test_the_names_are_read_and_the_values_are_not(tagged):
    audio, names = source_metadata(tagged)
    assert "comment" in names, f"the comment tag was not seen: {names}"
    assert audio is False
    for name in names:
        assert SECRET not in name, "a value reached the names"


def test_the_probe_carries_them(tagged):
    info = probe(tagged)
    assert info.tags and "comment" in info.tags
    assert info.audio is False


def test_a_file_with_sound_says_so(video_with_audio, video_silent):
    assert has_audio(video_with_audio) is True
    assert has_audio(video_silent) is False
    assert probe(video_with_audio).audio is True
    assert probe(video_silent).audio is False


def test_a_file_ffmpeg_cannot_read_gives_an_empty_answer(tmp_path):
    """`probe` is where an unreadable file is reported. A metadata line is not
    worth a second way to fail."""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert source_metadata(broken) == (False, ())


# ---------------------------------------------------------- what is recorded

def test_the_record_says_what_the_source_carried(tmp_path, tagged):
    out = tmp_path / "out.mp4"
    record = run_video(tagged, out, Settings(**FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.source_audio is False
    assert "comment" in record.source_tags_present


def test_no_tag_value_is_anywhere_in_the_written_record(tmp_path, tagged):
    """The record is a document that gets sent to people."""
    out = tmp_path / "out.mp4"
    record = run_video(tagged, out, Settings(**FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    written = sidecar_path(out).read_text(encoding="utf-8")
    assert SECRET not in written
    assert "51.5007" not in written
    assert json.loads(written)["source_tags_present"] == record.source_tags_present


def test_an_old_record_without_these_fields_still_reads():
    """New fields default to empty, so a record written before M1 loads."""
    from faceblur.pipeline import AuditRecord

    record = AuditRecord(source="a.mp4", output="b.mp4", status=STATUS_DONE)
    assert record.source_audio is False
    assert record.source_tags_present == []


# ------------------------------------------------- and what the copy carries

def test_the_blurred_copy_does_not_inherit_the_source_tags(tmp_path, tagged):
    """The one that matters. A location tag surviving into the copy would put
    the filming location in the file this tool hands over, and no amount of
    blurring would take it out again."""
    assert SECRET in tags_of(tagged), "the fixture did not carry the tag"
    out = tmp_path / "clean.mp4"
    record = run_video(tagged, out, Settings(**FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert out.is_file()
    carried = tags_of(out)
    assert SECRET not in carried, f"the copy carried the source's tag values: {carried}"
    assert "location" not in carried.lower()


def test_a_copied_stretch_does_not_carry_the_tags_either(tmp_path, tagged_no_bframes):
    """The copy path is the one to worry about: those stretches are the
    source's own packets, taken without re-encoding. Nothing is masked in
    this clip, so every frame goes through it."""
    from faceblur.segments import reorder_delay

    assert reorder_delay(tagged_no_bframes) == 0, "the fixture must be copyable"
    out = tmp_path / "copied.mp4"
    record = run_video(tagged_no_bframes, out,
                       Settings(device="cpu", copy_clean=True, chunk_seconds=0,
                                encode_seconds=0), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.frames_copied > 0, "this run was supposed to copy stretches"
    carried = tags_of(out)
    assert SECRET not in carried, f"a copied stretch carried the tag: {carried}"
