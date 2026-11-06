"""T19, long-run defect 2a: a screen is kept clear only where a hand is on it.

A MacBook showing CAD sat under `zone_kept` for thousands of frames because
the wearer's hand, busy with something beside it, had its grown reach over
the screen. With `zone_screen_needs_hand` (the default) a screen is protected
as a face is, by the hand quads only; the audit split (`h5._zone_kept`) and
the output check (`verify.residual_job`) ask the same question `redact` does.
No real footage in here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import run_ffmpeg
from faceblur import batch, h5
from faceblur.detect import Detection
from faceblur.hands import RECT_SCALE, Hand
from faceblur.settings import Settings
from faceblur.verify import plan_jobs, residual_job
from faceblur.zone import ZoneFrame

SHAPE = (640, 640)
SCREEN = Detection(250.0, 260.0, 140.0, 120.0, 0.8, None, "laptop", True, 0.8, "screen")


def square(cx, cy, half):
    return ((cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half))


def hand(cx, cy, palm=130.0) -> Hand:
    return Hand(square(cx, cy, palm * RECT_SCALE / 2), 0.9,
                (cx - palm / 2, cy - palm / 2, palm, palm), 0.95)


# ---------------------------------------------------------- the audit split

def test_the_audit_keeps_a_screen_under_a_hand_and_not_under_the_reach_alone():
    s = Settings(mask=("face", "screen"))
    reach_only = ZoneFrame(live=(square(320, 320, 250),), hands=())
    assert h5._zone_kept([SCREEN], reach_only, SHAPE, s) == [False]
    remembered = ZoneFrame(remembered=(square(320, 320, 250),))
    assert h5._zone_kept([SCREEN], remembered, SHAPE, s) == [False]
    under_hand = ZoneFrame(live=(square(320, 320, 250),), hands=(square(320, 320, 110),))
    assert h5._zone_kept([SCREEN], under_hand, SHAPE, s) == [True]
    old = Settings(mask=("face", "screen"), zone_screen_needs_hand=False)
    assert h5._zone_kept([SCREEN], reach_only, SHAPE, old) == [True]


def test_text_in_the_reach_is_still_kept():
    s = Settings(mask=("text", "screen"))
    line = Detection(280.0, 300.0, 80.0, 30.0, 0.9, None, "text", True, 0.9, "text",
                     ((280, 300), (360, 300), (360, 330), (280, 330)))
    reach_only = ZoneFrame(live=(square(320, 320, 250),), hands=())
    assert h5._zone_kept([line, SCREEN], reach_only, SHAPE, s) == [True, False]


# --------------------------------------------------------- the output check

@pytest.fixture(scope="module")
def copy(tmp_path_factory, video_with_face):
    """The face clip re-encoded and not masked: the screen is a miss there."""
    out = tmp_path_factory.mktemp("t19") / "copy.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-i", str(video_with_face),
                         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
                         "-pix_fmt", "yuv420p", str(out)])
    assert result.returncode == 0, result.stderr[-400:]
    return out


class _NoFaces:
    def detect(self, frame):
        return []


class _Screens:
    def detect(self, frame):
        return [SCREEN]


def _rows(src, out, settings, where):
    hands = {i: [where] for i in range(10)}
    [job] = plan_jobs(Path(src), Path(out), settings, chunk=1000, hands=hands)
    result = residual_job(job)
    rows = [row for row in result["found"] if row["kind"] == "screen"]
    assert rows, "the unmasked screen must be found"
    return rows


def test_the_check_sets_aside_a_screen_under_a_hand_and_calls_the_reach_a_miss(
        video_with_face, copy, monkeypatch):
    monkeypatch.setattr(batch, "_bank", lambda settings: _NoFaces())
    monkeypatch.setattr(batch, "_screens", lambda settings: _Screens())
    s = Settings(device="cpu", mask=("face", "screen"), hand_rule=False)
    # The hand on the screen: its 1.6-palm quad covers all of it.
    on = _rows(video_with_face, copy, s, hand(320.0, 320.0))
    assert all("zone" in row for row in on)
    # The hand below it: the reach covers 62 percent of the screen, the hand none.
    below = _rows(video_with_face, copy, s, hand(320.0, 560.0))
    assert all("zone" not in row for row in below)
    old = Settings(device="cpu", mask=("face", "screen"), hand_rule=False,
                   zone_screen_needs_hand=False)
    assert all("zone" in row for row in _rows(video_with_face, copy, old, hand(320.0, 560.0)))
