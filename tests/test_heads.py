"""The head backstop (faceblur/heads.py): a face mask from a YOLOX person box
for faces the face detectors cannot see. No real footage in here."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import run_ffmpeg
from faceblur import heads
from faceblur.detect import Detection
from faceblur.redact import POLYGON, shape_of
from faceblur.settings import Settings, SettingsError

SHAPE = (1000, 1000)   # height, width


def person(x, y, w, h, score=0.8) -> Detection:
    return Detection(x, y, w, h, score, None, "person", True, score, "screen")


def face(x, y, w, h) -> Detection:
    return Detection(x, y, w, h, 0.9, None, "track", True, 0.9, "face")


# ------------------------------------------------------------------ geometry

def test_the_head_is_the_top_of_the_box_and_its_middle_across():
    s = Settings()
    head = heads.head_region(person(100, 200, 200, 500), s)
    # Width 0.6 of 200 = 120, centred: x 140..260. Height 0.22 of 500 = 110.
    assert (head.x, head.y, head.w, head.h) == pytest.approx((140, 200, 120, 110))
    assert head.kind == "face" and head.source == "person_head"
    assert [p for c in head.quad for p in c] == pytest.approx(
        [140, 200, 260, 200, 260, 310, 140, 310])
    assert shape_of(head) == POLYGON, "masked as the rectangle, not an ellipse"


def test_the_head_is_clipped_to_the_frame():
    head = heads.head_region(person(300, -20, 200, 500), Settings(), SHAPE)
    assert head.y == 0 and head.x == pytest.approx(340)
    assert head.y + head.h == pytest.approx(-20 + 110)


def test_a_box_cut_by_a_side_edge_puts_the_head_against_that_edge():
    """The pilot's AM 032259 frame 2653: a colleague leaning in from the left
    rim, his face in the top-left corner of the box [0, 20, 706, 921]."""
    s = Settings()
    shape = (1300, 1600)
    left = heads.head_region(person(0, 20, 706, 921), s, shape)
    assert (left.x, left.y) == pytest.approx((0, 20))
    assert left.w == pytest.approx(0.6 * 706)
    assert left.h == pytest.approx(left.w), "as tall as wide: the box is not the body"
    right = heads.head_region(person(1000, 270, 600, 832), s, shape)
    assert right.x + right.w == pytest.approx(1600)
    middle = heads.head_region(person(430, 295, 365, 703), s, shape)
    assert middle.x == pytest.approx(430 + 0.2 * 365)
    assert middle.h == pytest.approx(0.22 * 703)


def test_a_face_without_corners_is_still_an_ellipse():
    assert shape_of(face(0, 0, 50, 50)) != POLYGON


# ------------------------------------------------------------------ exclusions

def test_by_default_only_a_box_cut_by_a_side_edge_gives_a_head():
    shape = (1300, 1600)
    middle = person(430, 295, 365, 703)
    left = person(0, 20, 706, 921)
    right = person(1000, 270, 600, 832)
    assert Settings().head_edge_only is True
    assert heads.regions([middle], shape, Settings()) == []
    assert len(heads.regions([left, right], shape, Settings())) == 2
    assert len(heads.regions([middle], shape, Settings(head_edge_only=False))) == 1


def test_the_wearer_is_left_out_at_the_edge_too():
    # Reaching the bottom band and the right edge, starting low: the wearer's arm.
    arm = person(1175, 1045, 425, 252)
    assert heads.regions([arm], (1300, 1600), Settings()) == []


def test_a_box_starting_at_the_top_edge_has_no_head_in_view():
    """Long-run defect 6: a person box cut at the top is a torso or legs."""
    shape = (1000, 1600)
    assert Settings().head_top_margin == pytest.approx(0.01)
    assert heads.regions([person(0, 0, 500, 700)], shape, Settings()) == []
    assert heads.regions([person(0, 9, 500, 700)], shape, Settings()) == []
    found = heads.regions([person(0, 40, 500, 700)], shape, Settings())
    assert len(found) == 1 and found[0].y == pytest.approx(40)


def square(cx, cy, half):
    return ((cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half))


def test_a_box_over_one_of_the_wearers_hands_has_no_head():
    """Long-run defect 6: the wearer's own forearm near the lens."""
    shape = (1000, 1600)
    arm = person(1300, 200, 300, 400)                  # cut by the right edge
    assert len(heads.regions([arm], shape, Settings())) == 1, "no hands: kept"
    assert len(heads.regions([arm], shape, Settings(), ())) == 1
    # The head region's centre is inside the hand quad.
    head = heads.head_region(arm, Settings(), shape)
    on_head = (square(head.cx, head.cy, 40),)
    assert heads.regions([arm], shape, Settings(), on_head) == []
    # A hand quad over the box, IoU 0.33 with it, away from the head region.
    over = (square(1450, 500, 100),)
    assert heads.regions([arm], shape, Settings(), over) == []
    # A small hand elsewhere in the box: IoU below 0.1, centre outside.
    small = (square(1450, 550, 30),)
    assert len(heads.regions([arm], shape, Settings(), small)) == 1
    # A hand far away.
    assert len(heads.regions([arm], shape, Settings(), (square(200, 800, 100),))) == 1


def test_the_new_head_settings_are_validated_and_recorded():
    for bad in (dict(head_top_margin=-0.1), dict(head_top_margin=1.0),
                dict(head_hand_iou=0.0), dict(head_hand_iou=1.5)):
        with pytest.raises(SettingsError):
            Settings(**bad)
    recorded = Settings().to_dict()
    assert recorded["head_top_margin"] == 0.01 and recorded["head_hand_iou"] == 0.1


def test_a_wide_box_on_the_bottom_edge_is_the_wearer():
    s = Settings(head_edge_only=False)
    wearer = person(200, 400, 600, 600)        # 0.6 of the width, bottom at 1000
    assert heads.regions([wearer], SHAPE, s) == []
    floating = person(200, 200, 600, 600)      # same box, bottom at 800
    found = heads.regions([floating], SHAPE, s)
    assert len(found) == 1 and found[0].source == "person_head"
    close = person(400, 300, 300, 700)         # on the edge, narrow, starts high: someone close
    assert len(heads.regions([close], SHAPE, s)) == 1
    legs = person(400, 600, 300, 400)          # on the edge, narrow, starts low: the wearer's legs
    assert heads.regions([legs], SHAPE, s) == []
    lap = person(300, 550, 450, 340)           # bottom at 0.89 H: still the bottom band
    assert heads.regions([lap], SHAPE, s) == []


def test_a_head_a_face_already_covers_is_dropped():
    head = heads.head_region(person(100, 200, 200, 500), Settings())
    centre = face(head.cx - 20, head.cy - 20, 40, 40)
    elsewhere = face(700, 700, 60, 60)
    assert heads.covered(head, [centre])
    assert not heads.covered(head, [elsewhere])
    per_frame = [[centre], [elsewhere], []]
    held = {0: [head], 1: [head], 2: [head]}
    kept = heads.uncovered(held, per_frame)
    assert 0 not in kept
    assert kept[1] == [head] and kept[2] == [head]


def test_a_face_overlapping_by_iou_covers_the_head():
    head = heads.head_region(person(100, 200, 200, 500), Settings())
    # Same size, shifted so its centre is outside the region: IoU 0.48/1.52.
    shifted = face(head.x + 0.52 * head.w, head.y, head.w, head.h)
    assert not (head.x <= shifted.cx <= head.x + head.w)
    assert heads.covered(head, [shifted])
    # Shifted further: IoU 0.4/1.6, centre outside, not covered.
    further = face(head.x + 0.6 * head.w, head.y, head.w, head.h)
    assert not heads.covered(head, [further])


def test_another_head_region_does_not_count_as_a_face():
    head = heads.head_region(person(100, 200, 200, 500), Settings())
    assert heads.uncovered({0: [head]}, [[head]]) == {0: [head]}


# ------------------------------------------------------------------ hold

def test_seen_once_is_not_held():
    head = heads.head_region(person(100, 200, 200, 500), Settings())
    assert heads.hold({0: [head]}, Settings(), 20) == {}


def test_seen_twice_is_held_across_the_gap_and_given_a_tail():
    s = Settings()
    head = heads.head_region(person(100, 200, 200, 500), s)
    held = heads.hold({0: [head], 2: [head]}, s, 20)
    assert sorted(held) == list(range(0, 2 + s.head_tail + 1))
    assert all(len(v) == 1 and v[0].source == "person_head" for v in held.values())


# ------------------------------------------------------------------ settings

def test_the_backstop_is_off_on_the_desktop():
    assert Settings().head_backstop is False


def test_head_settings_are_in_the_record():
    d = Settings(head_backstop=True).to_dict()
    for name in ("head_backstop", "head_edge_only", "head_conf", "head_frac", "head_width_frac",
                 "head_wearer_frac", "head_wearer_top", "head_wearer_bottom", "head_min_run", "head_gap", "head_tail",
                 "head_link_iou"):
        assert name in d
    assert d["head_backstop"] is True
    assert d["head_conf"] == 0.5 and d["head_frac"] == 0.22


@pytest.mark.parametrize("change", [
    {"head_conf": 0}, {"head_conf": 1.5}, {"head_frac": 0}, {"head_width_frac": 1.2},
    {"head_wearer_frac": 0}, {"head_wearer_top": 0}, {"head_wearer_bottom": 1.1},
    {"head_link_iou": 0}, {"head_min_run": 0},
    {"head_gap": -1}, {"head_tail": -1}, {"head_edge_only": "yes"}])
def test_head_settings_are_validated(change):
    with pytest.raises(SettingsError):
        Settings(**change)


# ------------------------------------------------------------------ end to end

SIZE = 640
BODY = (460, 120, 120, 400)          # x, y, w, h of the drawn "person"


def drawn_person() -> np.ndarray:
    frame = np.full((SIZE, SIZE, 3), 200, np.uint8)
    x, y, w, h = BODY
    frame[y:y + h, x:x + w] = 40
    frame[y:y + 80, x + 30:x + 90] = 170          # the "head"
    return frame


@pytest.fixture(scope="module")
def person_clip(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("person")
    still = folder / "person.png"
    cv2.imwrite(str(still), drawn_person())
    out = folder / "person.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-loop", "1", "-i", str(still),
                         "-frames:v", "6", "-r", "10", "-c:v", "libx264", "-preset",
                         "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the person clip: {result.stderr[-500:]}")
    return out


def decode_all(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_a_person_the_face_detectors_miss_has_the_head_filled(person_clip, tmp_path,
                                                               monkeypatch):
    """YOLOX does not call a drawn rectangle a person, so the person box is
    handed in; everything after it is the real pipeline."""
    from faceblur.batch import run_video, serial_submit
    from faceblur.pipeline import STATUS_DONE
    from faceblur.screens import ScreenDetector

    monkeypatch.setattr(ScreenDetector, "detect_people",
                        lambda self, frame: [person(*BODY)])
    # The drawn person stands clear of the frame edge, so edge-only is off.
    settings = Settings(mask=("face",), head_backstop=True, head_edge_only=False,
                        zone=False, copy_clean=False)
    out = tmp_path / "person_masked.mp4"
    per_frame: list = []
    record = run_video(person_clip, out, settings, serial_submit, per_frame_out=per_frame)
    assert record.status == STATUS_DONE, record.error
    assert record.heads > 0 and record.head_frames >= 2
    assert sum(1 for d in per_frame[2] if d.kind == "face") >= 1
    assert record.settings["head_backstop"] is True

    source = decode_all(person_clip)[2].astype(np.int16)
    frame = decode_all(out)[2].astype(np.int16)
    x, y, w, h = BODY
    hx0, hx1 = int(x + w * 0.2), int(x + w * 0.8)      # 484..556
    hy1 = int(y + h * 0.22)                              # 208
    head = frame[y + 4:hy1 - 4, hx0 + 4:hx1 - 4]
    assert (np.abs(head) <= 8).all(axis=2).mean() >= 0.95, "the head region is filled"
    body = (slice(hy1 + 30, y + h - 10), slice(x + 10, x + w - 10))
    diff = np.abs(frame[body] - source[body]).mean()
    assert diff < 3, f"the rest of the person moved by {diff:.2f}"


def test_without_the_backstop_nothing_is_masked(person_clip, tmp_path, monkeypatch):
    from faceblur.batch import run_video, serial_submit
    from faceblur.screens import ScreenDetector

    monkeypatch.setattr(ScreenDetector, "detect_people",
                        lambda self, frame: [person(*BODY)])
    settings = Settings(mask=("face",), zone=False, copy_clean=False)
    record = run_video(person_clip, tmp_path / "plain.mp4", settings, serial_submit)
    assert record.heads == 0 and record.head_frames == 0


def test_edge_only_by_default_leaves_a_person_clear_of_the_edge(person_clip, tmp_path,
                                                                 monkeypatch):
    from faceblur.batch import run_video, serial_submit
    from faceblur.screens import ScreenDetector

    monkeypatch.setattr(ScreenDetector, "detect_people",
                        lambda self, frame: [person(*BODY)])
    settings = Settings(mask=("face",), head_backstop=True, zone=False, copy_clean=False)
    record = run_video(person_clip, tmp_path / "edge_only.mp4", settings, serial_submit)
    assert record.heads == 0 and record.head_frames == 0
