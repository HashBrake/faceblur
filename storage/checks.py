"""Checks that a re-encoded file can replace its source.

A stored copy is only trusted when it has every frame, the same length, decodes
without a single error, and still looks like the source. The first three are
cheap and run first; VMAF is slow, so it runs last and only on a file that has
passed the rest. On a long recording VMAF is sampled in a few windows spread
over the beginning, middle and end instead of scoring every frame.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import av

from faceblur.video import VideoError
from storage.ffmpeg import run
from storage.probe import VideoFacts, probe


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class CheckReport:
    checks: list[Check]
    frames: int
    duration_s: float
    vmaf_mean: float | None
    vmaf_min: float | None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def describe(self) -> str:
        return "\n".join(f"{c.name}: {'ok' if c.passed else 'FAILED'} ({c.detail})"
                         for c in self.checks)


@dataclass(frozen=True)
class VmafResult:
    mean: float
    per_window: list[float]
    frames: int


def frame_count(src: VideoFacts, dst: VideoFacts) -> Check:
    return Check("frame_count", dst.frames == src.frames,
                 f"source {src.frames}, copy {dst.frames}")


def duration(src: VideoFacts, dst: VideoFacts) -> Check:
    # Half a frame: rounding in the timestamps passes, a whole missing frame
    # does not (a tolerance of one frame would let a copy one frame short pass).
    tolerance = 0.5 / src.fps if src.fps else 0.0
    diff = abs(dst.duration_s - src.duration_s)
    return Check("duration", diff <= tolerance,
                 f"source {src.duration_s:.3f} s, copy {dst.duration_s:.3f} s, "
                 f"allowed {tolerance:.3f} s")


def decodes_cleanly(path: Path) -> Check:
    """Every frame decodes in PyAV, and ffmpeg's own decode prints no error."""
    path = Path(path)
    try:
        expected = probe(path).frames
        with av.open(str(path)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
    except (av.FFmpegError, OSError, VideoError) as exc:
        return Check("decodes_cleanly", False, f"decode failed: {exc}")
    if decoded != expected:
        return Check("decodes_cleanly", False, f"decoded {decoded} frames of {expected}")
    try:
        r = run(["-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                timeout=6 * 3600)
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("decodes_cleanly", False, f"ffmpeg decode failed: {exc}")
    errors = r.stderr.decode(errors="replace").strip()
    if r.returncode != 0 or errors:
        shown = errors[-300:] or f"exit code {r.returncode}"
        return Check("decodes_cleanly", False, f"ffmpeg reported: {shown}")
    return Check("decodes_cleanly", True, f"{decoded} frames")


def _vmaf_window(reference: Path, distorted: Path, log: Path, threads: int,
                 start: float | None, length: float | None) -> tuple[float, int]:
    """One libvmaf run: (pooled mean, frames scored). Distorted first, reference second."""
    cut = [] if start is None else ["-ss", f"{start:.3f}", "-t", f"{length:.3f}"]
    quoted = "'" + log.resolve().as_posix().replace(":", r"\:") + "'"
    r = run(["-v", "error", *cut, "-i", str(distorted), *cut, "-i", str(reference),
             "-lavfi", f"libvmaf=log_fmt=json:log_path={quoted}:n_threads={threads}",
             "-f", "null", "-"], timeout=6 * 3600)
    if r.returncode != 0:
        tail = r.stderr.decode(errors="replace").strip()[-500:]
        raise VideoError(f"VMAF failed for {distorted}: {tail}")
    data = json.loads(log.read_text())
    return float(data["pooled_metrics"]["vmaf"]["mean"]), len(data["frames"])


def vmaf(reference: Path, distorted: Path, windows: int = 3, window_s: float = 60.0,
         threads: int = 4) -> VmafResult:
    """VMAF of `distorted` against `reference`, whole file or sampled in windows."""
    reference, distorted = Path(reference), Path(distorted)
    total = probe(reference).duration_s
    if total <= windows * window_s:
        spans = [(None, None)]
    elif windows == 1:
        spans = [((total - window_s) / 2, window_s)]
    else:
        spans = [(i * (total - window_s) / (windows - 1), window_s) for i in range(windows)]
    scores, counts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, length) in enumerate(spans):
            score, n = _vmaf_window(reference, distorted, Path(tmp) / f"vmaf{i}.json",
                                    threads, start, length)
            scores.append(score)
            counts.append(n)
    frames = sum(counts)
    mean = sum(s * n for s, n in zip(scores, counts)) / frames if frames else 0.0
    return VmafResult(mean=mean, per_window=scores, frames=frames)


def vmaf_check(result: VmafResult, floor: float = 95.0) -> Check:
    return Check("vmaf", result.mean >= floor,
                 f"mean {result.mean:.2f}, min window {min(result.per_window):.2f}, "
                 f"floor {floor}")


def run_all(src: Path, dst: Path, floor: float = 95.0, windows: int = 3,
            window_s: float = 60.0) -> CheckReport:
    """Every check, cheapest first. VMAF is skipped when an earlier check failed."""
    src_facts, dst_facts = probe(src), probe(dst)
    checks = [frame_count(src_facts, dst_facts), duration(src_facts, dst_facts),
              decodes_cleanly(dst)]
    vmaf_mean = vmaf_min = None
    if all(c.passed for c in checks):
        result = vmaf(src, dst, windows=windows, window_s=window_s)
        checks.append(vmaf_check(result, floor))
        vmaf_mean, vmaf_min = result.mean, min(result.per_window)
    else:
        checks.append(Check("vmaf", False, "skipped: earlier check failed"))
    return CheckReport(checks=checks, frames=dst_facts.frames,
                       duration_s=dst_facts.duration_s, vmaf_mean=vmaf_mean,
                       vmaf_min=vmaf_min)
