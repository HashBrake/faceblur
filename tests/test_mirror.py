"""Recreating a folder, and a pool that loses a worker."""
from __future__ import annotations

import multiprocessing
import os

import pytest

from faceblur.batch import PoolSubmit, WorkerLost
from faceblur.mirror import MirrorError, check_target, copy_files, plan


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "Shoot 01"
    (root / "day1" / "cam a").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "notes.txt").write_text("hello")
    (root / "day1" / "clip.MP4").write_bytes(b"video")
    (root / "day1" / "cam a" / "clip.mp4").write_bytes(b"video")
    (root / "day1" / "cam a" / "clip.mov").write_bytes(b"other video")
    (root / "day1" / "cam a" / "thumb.jpg").write_bytes(b"jpg")
    return root


def test_every_mp4_is_blurred_and_everything_else_is_copied(tree, tmp_path):
    out = tmp_path / "out"
    mirror = plan(tree, out)
    assert mirror.target == out / "Shoot 01"
    rel = lambda pairs: sorted(str(s.relative_to(tree)).replace("\\", "/") for s, _ in pairs)
    assert rel(mirror.videos) == ["day1/cam a/clip.mp4", "day1/clip.MP4"]
    # Only MP4. A .mov is copied as it is.
    assert rel(mirror.files) == ["day1/cam a/clip.mov", "day1/cam a/thumb.jpg", "notes.txt"]
    for src, dst in mirror.videos + mirror.files:
        assert dst.relative_to(mirror.target) == src.relative_to(tree)


def test_copy_makes_every_folder_and_never_copies_a_video(tree, tmp_path):
    mirror = plan(tree, tmp_path / "out")
    copied, problems = copy_files(mirror)
    assert (copied, problems) == (3, [])
    assert (mirror.target / "empty").is_dir()
    assert (mirror.target / "notes.txt").read_text() == "hello"
    assert (mirror.target / "day1" / "cam a" / "clip.mov").exists()
    # The originals of the videos must never land in the copy.
    assert not (mirror.target / "day1" / "clip.MP4").exists()
    assert not (mirror.target / "day1" / "cam a" / "clip.mp4").exists()


def test_copy_leaves_existing_files_unless_asked(tree, tmp_path):
    mirror = plan(tree, tmp_path / "out")
    copy_files(mirror)
    (mirror.target / "notes.txt").write_text("changed")
    assert copy_files(mirror)[0] == 0
    assert (mirror.target / "notes.txt").read_text() == "changed"
    assert copy_files(mirror, replace=True)[0] == 3
    assert (mirror.target / "notes.txt").read_text() == "hello"


def test_the_report_sits_beside_the_copy_not_inside_it(tree, tmp_path):
    mirror = plan(tree, tmp_path / "out")
    assert mirror.report_path.parent == mirror.target.parent


@pytest.mark.parametrize("where", ["self", "inside", "parent"])
def test_a_copy_that_would_overwrite_or_loop_is_refused(tree, where):
    target = {"self": tree, "inside": tree / "day1" / "x",
              "parent": tree.parent}[where]
    with pytest.raises(MirrorError):
        check_target(tree, target)
    with pytest.raises(MirrorError):
        plan(tree, target.parent if where == "self" else target)


def _die(job):
    if job == "die":
        os._exit(3)
    return job


def test_a_worker_that_dies_is_reported_not_waited_for():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2) as pool:
        submit = PoolSubmit(pool)
        submit.POLL_SECONDS = 0.2
        assert submit(_die, ["a", "b"], 2) == ["a", "b"]
        with pytest.raises(WorkerLost):
            submit(_die, ["a", "die", "b"], 2)
