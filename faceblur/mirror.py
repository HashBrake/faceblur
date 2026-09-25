"""Recreate a folder as it is, with every MP4 in it replaced by its blurred copy.

The copy of the folder has the same name, the same subfolders, including empty
ones, and the same files under the same names. Every file that is not an MP4
is copied byte for byte. Every MP4 goes through the pipeline and its blurred
copy takes the original's name, with no suffix and no audit record beside it,
so that nothing in the copy of the folder is a file the original did not have.
The audit records go in one report file beside the copy instead.

The original of an MP4 is never copied across, not even when blurring it
fails. A missing file is a gap somebody notices. An unblurred file under the
original's name is a leak nobody does.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MIRROR_EXT = frozenset({".mp4"})


class MirrorError(ValueError):
    """The folder cannot be recreated where it was asked to be."""


@dataclass
class MirrorPlan:
    source: Path                  # the folder the user chose
    target: Path                  # its copy, output folder / source name
    folders: list[Path] = field(default_factory=list)                # relative
    files: list[tuple[Path, Path]] = field(default_factory=list)     # copied as they are
    videos: list[tuple[Path, Path]] = field(default_factory=list)    # blurred

    @property
    def report_path(self) -> Path:
        return self.target.parent / f"{self.target.name}_faceblur_report.json"


def is_mirrored_video(path: Path) -> bool:
    return path.suffix.lower() in MIRROR_EXT


def target_for(source: Path, out_dir: Path) -> Path:
    return Path(out_dir) / Path(source).name


def check_target(source: Path, target: Path) -> None:
    """Refuse a copy that would overwrite the source or sit inside it.

    Inside it, the walk would find the copy it is writing and copy it again.
    """
    src = Path(source).resolve()
    dst = Path(target).resolve()
    if dst == src:
        raise MirrorError("The copy would replace the original folder. "
                          "Choose a different output folder.")
    if src in dst.parents:
        raise MirrorError("The output folder is inside the folder you chose. "
                          "Choose an output folder outside it.")
    if dst in src.parents:
        raise MirrorError("The folder you chose is inside the copy it would make. "
                          "Choose a different output folder.")


def plan(source: Path, out_dir: Path) -> MirrorPlan:
    """Walk `source` and say where everything in it goes."""
    source = Path(source)
    target = target_for(source, out_dir)
    check_target(source, target)
    result = MirrorPlan(source=source, target=target)
    for root, dirs, files in os.walk(source):
        dirs.sort()
        here = Path(root)
        relative = here.relative_to(source)
        result.folders.append(relative)
        for name in sorted(files):
            src = here / name
            dst = target / relative / name
            if is_mirrored_video(src):
                result.videos.append((src, dst))
            else:
                result.files.append((src, dst))
    return result


def copy_files(mirror: MirrorPlan, replace: bool = False,
               cancel: Optional[Callable[[], bool]] = None) -> tuple[int, list[str]]:
    """Make every folder, then copy every file that is not a video.

    A file already there is left alone unless `replace`. Returns how many
    files were copied and one line per file that could not be.
    """
    for relative in mirror.folders:
        (mirror.target / relative).mkdir(parents=True, exist_ok=True)
    copied, problems = 0, []
    for src, dst in mirror.files:
        if cancel is not None and cancel():
            break
        if dst.exists() and not replace:
            continue
        try:
            shutil.copy2(src, dst)
            copied += 1
        except OSError as exc:
            problems.append(f"{src.relative_to(mirror.source)}: {exc}")
    for relative in mirror.folders:
        # Folder times last, because writing into a folder moves its time.
        try:
            shutil.copystat(mirror.source / relative, mirror.target / relative)
        except OSError:
            pass
    return copied, problems
