"""The adaptive stride on the single-pass path, `pipeline.plan` / `process_video`.

The same policy as `batch.detect_job` (tests/test_stride.py), with one policy
for the whole file: a skipped stride frame never reaches the bank, its
candidates are empty, and the plan and the record count it.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from faceblur import pipeline
from faceblur.pipeline import STATUS_DONE, plan, process_video
from faceblur.settings import Settings
from faceblur.stride import AdaptiveStride, shift_px

from conftest import run_ffmpeg

N = 30
ON = Settings(device="cpu", zone=False, adaptive_stride=True, camera_comp=True,
              adaptive_stride_quiet=3, adaptive_stride_max=2, adaptive_stride_shift=4.0)


@pytest.fixture(scope="module")
def still_wall(tmp_path_factory) -> Path:
    """Three seconds of a still texture, 10 fps. A still camera on a texture
    measures a few hundredths of a pixel; flat grey would measure an exact
    zero, which is no estimate (see stride.shift_px)."""
    out = tmp_path_factory.mktemp("clips") / "still_wall.mp4"
    wall = ("geq=lum='128+50*sin(X/9)*cos(Y/13)+30*sin((X+2*Y)/23)':cb=128:cr=128,"
            "setsar=1,format=yuv420p")
    result = run_ffmpeg([
        "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x320:r=10:d=3",
        "-vf", wall, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
        "-pix_fmt", "yuv420p", str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the clip: {result.stderr[-500:]}")
    return out


class FakeBank:
    """detect_raw returns the index of the frame it was called on."""

    def __init__(self, got):
        self.got = got
        self.called = []
        self.compute, self.model_hashes = {}, {}

    def detect_raw(self, frame):
        self.called.append(self.got["index"])
        return self.got["index"]


@pytest.fixture
def faked(monkeypatch):
    """A fake bank and tracker; candidates come from `strong` and `weak` by frame."""
    got = {"strong": {}, "weak": {}, "shifts": [], "index": None}
    bank = FakeBank(got)
    real_shift, real_read = pipeline.estimate_shift, pipeline.read_frames

    def read(src):
        # Tells the bank which frame it is looking at.
        for i, frame in enumerate(real_read(src)):
            got["index"] = i
            yield frame

    def shift(prev, small, side):
        s = real_shift(prev, small, side)
        got["shifts"].append(s)
        return s

    class Tracker:
        def run(self, per, weak, shifts, shape):
            got["per"], got["weak_per"] = per, weak
            return [[] for _ in per], []

    monkeypatch.setattr(pipeline, "estimate_shift", shift)
    monkeypatch.setattr(pipeline, "read_frames", read)
    monkeypatch.setattr(pipeline, "filter_candidates",
                        lambda raw, s, shape: list(got["strong"].get(raw, [])))
    monkeypatch.setattr(pipeline, "weak_candidates",
                        lambda raw, s, shape: list(got["weak"].get(raw, [])))
    monkeypatch.setattr(pipeline, "tracker_for", lambda s: Tracker())
    return bank, got


def _expected(settings, got):
    """The frames the policy skips, driven by hand over the measured shifts."""
    policy, skipped = AdaptiveStride(settings), []
    for i, s in enumerate(got["shifts"]):
        if policy.skip(shift_px(s)):
            skipped.append(i)
        else:
            policy.saw(got["strong"].get(i, []), got["weak"].get(i, []))
    return skipped


def test_plan_skips_quiet_still_frames_without_calling_the_bank(still_wall, faked):
    bank, got = faked
    p = plan(still_wall, ON, bank)
    skipped = sorted(set(range(N)) - set(bank.called))
    assert len(got["shifts"]) == N
    # The quiet run-up is detected, then every other frame at max 2.
    assert skipped == list(range(3, N, 2))
    assert skipped == _expected(ON, got)
    assert p.skipped_adaptive == len(skipped)
    assert all(got["per"][i] == [] and got["weak_per"][i] == [] for i in skipped)


def test_plan_off_calls_the_bank_on_every_stride_frame(still_wall, faked):
    bank, got = faked
    p = plan(still_wall, ON.with_changes(adaptive_stride=False), bank)
    assert bank.called == list(range(N))
    assert p.skipped_adaptive == 0


def test_plan_without_camera_comp_counts_as_moving(still_wall, faked):
    bank, got = faked
    p = plan(still_wall, ON.with_changes(camera_comp=False), bank)
    assert bank.called == list(range(N))
    assert p.skipped_adaptive == 0


@pytest.mark.parametrize("kind", ["strong", "weak"])
def test_a_candidate_resets_the_quiet_count(still_wall, faked, kind):
    bank, got = faked
    face = SimpleNamespace(verified=False)
    got[kind][6] = [face]
    p = plan(still_wall, ON, bank)
    skipped = sorted(set(range(N)) - set(bank.called))
    # 3 and 5 skipped; the candidate on 6 needs 7, 8, 9 detected again.
    assert skipped[:3] == [3, 5, 10]
    assert all(i in bank.called for i in (6, 7, 8, 9))
    assert skipped == _expected(ON, got)
    assert p.skipped_adaptive == len(skipped)
    assert got["per" if kind == "strong" else "weak_per"][6] == [face]


def test_process_video_records_the_skipped_count(still_wall, faked, tmp_path):
    bank, got = faked
    record = process_video(still_wall, tmp_path / "on.mp4", ON, bank=bank, write_sidecar=False)
    assert record.status == STATUS_DONE, record.error
    assert record.frames_skipped_adaptive == len(range(3, N, 2))
    assert record.frames_skipped_adaptive == N - len(bank.called)
    off = process_video(still_wall, tmp_path / "off.mp4", ON.with_changes(adaptive_stride=False),
                        bank=FakeBank(got), write_sidecar=False)
    assert off.status == STATUS_DONE, off.error
    assert off.frames_skipped_adaptive == 0
