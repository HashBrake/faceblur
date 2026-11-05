"""Text and plate strides (T13a).

The field-test pilot profiled text at 160 ms a frame and plates at 112 ms,
the two largest stages of 242 ms end to end. Neither moves on its own and
both holds bridge gaps of six frames, so the text detector runs on every
`text_stride`-th frame and the plate detector on every `plate_stride`-th.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from conftest import run_ffmpeg
from faceblur.batch import detect_job
from faceblur.detect import Detection
from faceblur.plates import PlateDetector
from faceblur.segments import frame_times
from faceblur.video import probe
from faceblur.settings import Settings, SettingsError
from faceblur.text import hold as hold_text

from test_text_mask import text_frame


@pytest.fixture(scope="module")
def text_clip(tmp_path_factory) -> Path:
    """Six frames of pasted text at 10 fps."""
    folder = tmp_path_factory.mktemp("strides")
    still = folder / "text.png"
    cv2.imwrite(str(still), text_frame())
    out = folder / "text.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-loop", "1", "-i", str(still),
                         "-frames:v", "6", "-r", "10", "-c:v", "libx264", "-preset",
                         "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the text clip: {result.stderr[-500:]}")
    return out


def run_job(clip: Path, monkeypatch, **changes) -> tuple[dict, list[int]]:
    """detect_job over the whole clip, serial, with the plate detector counted."""
    calls: list[int] = []

    def counted(self, frame):
        calls.append(1)
        return []

    monkeypatch.setattr(PlateDetector, "detect", counted)
    info, times = probe(clip), frame_times(clip)
    settings = Settings(mask=("text", "plate"), zone=False, device="cpu",
                        plate_device="cpu", **changes)
    got = detect_job({"src": str(clip), "info": info, "times": times,
                      "settings": settings, "start": 0, "end": len(times)})
    return got, calls


def test_text_on_even_frames_and_plates_on_every_third(text_clip, monkeypatch):
    got, calls = run_job(text_clip, monkeypatch, text_stride=2, plate_stride=3)
    assert sorted(got["text"]) == [0, 2, 4]
    assert len(calls) == 2, "plates on frames 0 and 3"


def test_strides_of_one_detect_every_frame(text_clip, monkeypatch):
    got, calls = run_job(text_clip, monkeypatch, text_stride=1, plate_stride=1)
    assert sorted(got["text"]) == list(range(6))
    assert len(calls) == 6


def test_the_hold_fills_the_frames_between_strided_sightings():
    line = Detection(10, 10, 100, 20, 0.9, None, "ppocr", True, 0.9, "text")
    held = hold_text({0: [line], 2: [line], 4: [line]}, Settings(), 6)
    assert sorted(held) == list(range(6)), "1 and 3 filled, 5 by the tail"


def test_strides_are_settings():
    s = Settings()
    assert (s.text_stride, s.plate_stride, s.zone_stride) == (2, 3, 1)
    recorded = s.to_dict()
    assert recorded["text_stride"] == 2 and recorded["plate_stride"] == 3
    for bad in ({"text_stride": 0}, {"plate_stride": 0}):
        with pytest.raises(SettingsError):
            Settings(**bad)
