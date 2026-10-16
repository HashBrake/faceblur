r"""Which ffmpeg the storage tools run, and what it can do.

The pipeline's ffmpeg is the one imageio-ffmpeg bundles (gyan.dev's 7.1
essentials build). A newer or fuller build can be dropped into `tools/` at the
repo root without touching the pipeline: the storage tools prefer it. `tools/`
is gitignored, so a downloaded build never enters git.

There is no ffprobe, so capabilities are read from `ffmpeg -encoders` and
`ffmpeg -filters`, and NVENC is tried for real, because a listed encoder only
means the build has it, not that this GPU and driver can run it.

VMAF, as T02 runs it. The working filter string with this build is::

    libvmaf=log_fmt=json:log_path='C\:/full/path/vmaf.json'

with the distorted file as the first input and the reference as the second.
The built-in default model is vmaf_v0.6.1; naming it is not needed. On Windows
the log path takes forward slashes, the drive colon escaped as `\:`, and the
whole path in single quotes; an unquoted path fails to parse at either level
of escaping. The pooled score is `pooled_metrics.vmaf.mean` in the JSON log.
Note that vmaf_v0.6.1 does not reach 100 on identical inputs with little
motion: the 320x240 test clip against itself scores about 98.6.
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from faceblur import video
from faceblur.video import _no_window

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Path to ffmpeg: a build under `tools/` if there is one, else the bundled one."""
    if TOOLS_DIR.is_dir():
        for pattern in ("ffmpeg*.exe", "**/bin/ffmpeg.exe"):
            for candidate in sorted(TOOLS_DIR.glob(pattern)):
                if candidate.is_file():
                    return str(candidate)
    return video.ffmpeg_exe()


def run(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", *args], capture_output=True,
                          timeout=timeout, **_no_window())


def _listed(kind: str) -> set[str]:
    """Names in the table `ffmpeg -encoders` or `ffmpeg -filters` prints.

    Every row is flags, then the name, then a description. The legend above
    the table has a `=` in its second column, and that is how it is skipped.
    """
    r = run([f"-{kind}"], timeout=60)
    names = set()
    for line in r.stdout.decode(errors="replace").splitlines():
        cols = line.split()
        if len(cols) >= 2 and cols[1] != "=":
            names.add(cols[1])
    return names


@lru_cache(maxsize=None)
def has_encoder(name: str) -> bool:
    return name in _listed("encoders")


@lru_cache(maxsize=None)
def has_filter(name: str) -> bool:
    return name in _listed("filters")


@lru_cache(maxsize=None)
def nvenc_works(encoder: str = "hevc_nvenc") -> bool:
    """True when this NVIDIA encoder runs here: one frame, encoded to nowhere."""
    try:
        r = run(["-v", "error", "-f", "lavfi", "-i", "color=size=256x256:rate=1",
                 "-frames:v", "1", "-c:v", encoder, "-f", "null", "-"], timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
