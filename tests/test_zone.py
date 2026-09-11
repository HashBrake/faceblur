"""The handled zone: the region nothing is ever masked in.

The owner's rule of 2026-09-11 makes this the one promise in the project that
is absolute. Every other number here is a rate, and a miss is a leak to be
driven down; inside the zone a single destroyed pixel is the rule broken. So
these tests are about the promise rather than about a threshold: that the zone
is subtracted from every mask, that turning it off changes nothing else, that
memory expires and moves with the camera, and that a face is never protected
by memory.

The models are not tested here. `tests/test_hands.py` covers the palm detector
and the landmark model, and what this adds is geometry and bookkeeping, which
is where a quiet mistake would live.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceblur.detect import Detection
from faceblur.hands import RECT_SCALE, Hand
from faceblur.redact import alpha_with_zone, build_alpha, redact
from faceblur.settings import Settings, SettingsError
from faceblur.zone import (REMEMBERED_PROTECTS, ZoneFrame, alpha_for, build,
                           describe, hand_scale, inside_frame, moved, overlaps,
                           polygon_area, polygon_for, qualifies, share, windows)

SHAPE = (600, 800)          # a small frame, so the tests are quick


def hand(cx=400.0, cy=500.0, size=140.0, score=0.9, presence=0.95) -> Hand:
    """A hand whose quad is the square MediaPipe would hand on, upright."""
    half = size * RECT_SCALE / 2
    quad = ((cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half))
    return Hand(quad, score, (cx - size / 2, cy - size / 2, size, size), presence)


def square(cx, cy, half):
    return ((cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half))


def face(x=100.0, y=100.0, size=60.0) -> Detection:
    """A face box with landmarks that actually sit in it.

    `ellipse_for` blends the box centre with the landmark centre, so landmarks
    somewhere else drag the mask somewhere else, which is a fine way to write a
    test that passes for the wrong reason.
    """
    lm = tuple((x + size * dx, y + size * dy) for dx, dy in
               ((0.3, 0.35), (0.7, 0.35), (0.5, 0.55), (0.35, 0.75), (0.65, 0.75)))
    return Detection(x, y, size, size, 0.9, lm)


def screen(x=100.0, y=100.0, w=200.0, h=150.0) -> Detection:
    return Detection(x, y, w, h, 0.9, None, "tv", True, 0.9, "screen")


# --------------------------------------------------------------- the windows

def test_the_windows_cover_the_whole_frame():
    """A hand in a corner is a hand. A grid that stopped short of an edge
    would protect the middle of the picture and nothing else."""
    boxes = windows(SHAPE, 256)
    h, w = SHAPE
    covered = np.zeros(SHAPE, bool)
    for x0, y0, x1, y1 in boxes:
        covered[y0:y1, x0:x1] = True
    assert covered.all()
    assert all(x1 <= w and y1 <= h for _, _, x1, y1 in boxes)


def test_a_frame_smaller_than_one_window_gets_one_window():
    assert windows((100, 100), 512) == [(0, 0, 100, 100)]


def test_the_windows_overlap_so_a_hand_on_a_seam_is_still_seen():
    boxes = windows((1300, 1600), 512)
    xs = sorted({x0 for x0, _, _, _ in boxes})
    assert min(b - a for a, b in zip(xs, xs[1:])) < 512, "no overlap at all"


# ------------------------------------------------------------ whose hand

def test_a_large_hand_qualifies_wherever_it_is():
    s = Settings()
    assert qualifies(hand(400, 100, size=s.zone_min_hand_px + 10), SHAPE, s)


def test_a_small_hand_low_in_the_frame_qualifies():
    """The camera is on the chest, so the wearer's hands come in from below."""
    s = Settings()
    low = hand(400, SHAPE[0] - 20, size=40)
    assert qualifies(low, SHAPE, s)


def test_a_small_hand_high_in_the_frame_does_not():
    """A bystander across the table. This is the rule that can be wrong, and
    the audit under docs/audits is where the cases it gets wrong are written."""
    s = Settings()
    assert not qualifies(hand(400, 60, size=40), SHAPE, s)


# --------------------------------------------------------------- the polygon

def test_the_polygon_reaches_past_the_hand_to_what_it_holds():
    s = Settings()
    h = hand()
    poly = polygon_for(h, s, SHAPE)
    assert polygon_area(poly) > polygon_area(h.quad), "a card in hand needs the room"
    assert hand_scale(s) == pytest.approx(RECT_SCALE * s.zone_scale)


def test_the_polygon_is_clipped_to_the_frame():
    poly = polygon_for(hand(10.0, 10.0, size=200.0), Settings(), SHAPE)
    assert all(0 <= x <= SHAPE[1] and 0 <= y <= SHAPE[0] for x, y in poly)


def test_a_clipped_polygon_is_still_convex_enough_to_fill():
    """`cv2.fillConvexPoly` fills nonsense for a non convex quad, and the zone
    is drawn with it, so a clamped corner must not fold the shape."""
    poly = polygon_for(hand(0.0, 0.0, size=300.0), Settings(), SHAPE)
    canvas = alpha_for(SHAPE, [poly], 0.0)
    assert canvas.any()


# ---------------------------------------------------------------- the memory

def test_a_handled_thing_stays_protected_after_the_hand_leaves():
    s = Settings(zone_memory=1.0)
    zones = build({0: [hand()]}, None, s, 40, 10.0, SHAPE)
    assert zones[0].live and not zones[0].remembered
    assert not zones[5].live and zones[5].remembered, "the hand went, the thing stayed"


def test_the_memory_expires():
    s = Settings(zone_memory=1.0)
    zones = build({0: [hand()]}, None, s, 40, 10.0, SHAPE)
    assert zones[9].remembered
    assert not zones[20].remembered, "ten frames at 10 fps is one second"


def test_a_live_hand_refreshes_the_memory_rather_than_stacking_on_it():
    s = Settings(zone_memory=1.0)
    zones = build({0: [hand()], 5: [hand()]}, None, s, 40, 10.0, SHAPE)
    assert len(zones[6].remembered) == 1, "one object, one polygon"
    assert zones[13].remembered, "refreshed at frame 5, so still alive at 13"


def test_a_polygon_is_live_or_remembered_and_never_both():
    """They are drawn into different canvases and a polygon in both would be
    counted twice, which is harmless for the alpha and misleading for the
    share."""
    s = Settings(zone_memory=1.0)
    zones = build({0: [hand()], 3: [hand()]}, None, s, 20, 10.0, SHAPE)
    for z in zones:
        for poly in z.live:
            assert not any(overlaps(poly, other) for other in z.remembered)


def test_the_remembered_zone_moves_with_the_camera():
    """It has to follow the object, not the pixels. `estimate_shift` says a
    static thing sits dx further right in the next frame, so the polygon does
    too."""
    s = Settings(zone_memory=2.0)
    shifts = [(0.0, 0.0)] + [(4.0, 0.0)] * 39
    zones = build({0: [hand(cx=400.0)]}, shifts, s, 40, 10.0, SHAPE)
    first = np.mean([p[0] for p in zones[1].remembered[0]])
    later = np.mean([p[0] for p in zones[6].remembered[0]])
    assert later - first == pytest.approx(20.0, abs=1e-6), "five frames at 4 px"


def test_a_remembered_polygon_that_leaves_the_frame_is_dropped():
    s = Settings(zone_memory=10.0)
    shifts = [(0.0, 0.0)] + [(-200.0, 0.0)] * 39
    zones = build({0: [hand(cx=100.0, size=60.0)]}, shifts, s, 40, 10.0, SHAPE)
    assert not zones[20].remembered


def test_memory_of_zero_seconds_remembers_nothing():
    zones = build({0: [hand()]}, None, Settings(zone_memory=0.0), 10, 10.0, SHAPE)
    assert zones[0].live and not zones[0].remembered
    assert not zones[3].live and not zones[3].remembered


# ------------------------------------------------- what each zone protects

def test_the_reach_protects_text_and_screens_and_the_hand_protects_a_face():
    """The two are not the same promise, and the audit of 2026-09-11 is why.

    What the wearer reaches over is what they are holding, so text and screens
    in it are part of the task. A face inside somebody's reach is usually
    their own face: on `004310` frame 836 a canteen worker's hand near the
    bottom of the frame grew a polygon 40 px upwards and covered their face at
    144 px, and the zone was protecting a stranger from redaction.
    """
    z = ZoneFrame(live=(square(400, 400, 150),), hands=(square(400, 400, 60),))
    assert z.for_kind("screen") == z.live
    assert z.for_kind("text") == z.live
    assert z.for_kind("face") == z.hands
    assert z.for_kind("face", face_needs_hand=False) == z.live


def test_a_remembered_polygon_protects_text_and_screens_only():
    """A person who walks into the space where a card was put down is still a
    person. This is the line the owner's rule draws and the one place the two
    protections differ."""
    z = ZoneFrame(remembered=(square(400, 400, 100),))
    assert z.for_kind("face") == ()
    assert set(REMEMBERED_PROTECTS) == {"text", "screen"}
    assert z.for_kind("screen") == z.remembered
    assert z.for_kind("text") == z.remembered


# ------------------------------------------------------ applying it to a mask

def test_a_face_under_the_wearers_hand_is_not_masked():
    s = Settings()
    f = face(380, 380, 60)
    z = ZoneFrame(live=(square(410, 410, 200),), hands=(square(410, 410, 150),))
    assert build_alpha(SHAPE, [f], s).any(), "it would have been masked"
    assert not (alpha_with_zone(SHAPE, [f], s, z) >= 0.5).any()


def test_a_face_inside_the_reach_but_not_under_a_hand_is_masked():
    """The leak the audit found, as a test. A hand low in the frame grows a
    polygon that reaches a face above it; the face is somebody's, not a
    picture of one, and it must still be destroyed."""
    s = Settings()
    f = face(380, 380, 60)
    z = ZoneFrame(live=(square(410, 410, 200),), hands=(square(410, 600, 60),))
    assert (alpha_with_zone(SHAPE, [f], s, z) >= 0.5).any()


def test_a_screen_inside_the_reach_is_still_protected():
    """The other half: what the hand is reaching over is what it is holding."""
    s = Settings(mask=("face", "screen"))
    sc = screen(350, 350, 120, 100)
    z = ZoneFrame(live=(square(410, 410, 250),), hands=(square(410, 600, 60),))
    assert build_alpha(SHAPE, [sc], s).any()
    assert not (alpha_with_zone(SHAPE, [sc], s, z) >= 0.5).any()


def test_a_face_inside_a_remembered_polygon_is_masked():
    s = Settings()
    f = face(380, 380, 60)
    z = ZoneFrame(remembered=(square(410, 410, 150),))
    assert (alpha_with_zone(SHAPE, [f], s, z) >= 0.5).any()


def test_a_screen_inside_a_remembered_polygon_is_not_masked():
    s = Settings(mask=("face", "screen"))
    sc = screen(350, 350, 120, 100)
    z = ZoneFrame(remembered=(square(410, 410, 200),))
    assert build_alpha(SHAPE, [sc], s).any()
    assert not (alpha_with_zone(SHAPE, [sc], s, z) >= 0.5).any()


def test_a_face_and_a_screen_in_one_remembered_polygon_get_different_answers():
    """The case the two canvases exist for: on one frame, the same polygon
    protects the screen and not the face."""
    s = Settings(mask=("face", "screen"))
    f = face(360, 360, 60)
    sc = screen(500, 500, 120, 100)
    z = ZoneFrame(remembered=(square(450, 450, 250),))
    alpha = alpha_with_zone(SHAPE, [f, sc], s, z) >= 0.5
    assert alpha[int(f.cy), int(f.cx)], "the face is masked"
    assert not alpha[int(sc.cy), int(sc.cx)], "the screen is not"


def test_half_a_face_behind_a_hand_keeps_the_half_that_is_not():
    """Applied to the alpha rather than by dropping the detection, so a partly
    covered face is partly masked. Dropping it would have left the whole face
    visible."""
    s = Settings()
    f = face(300, 300, 200)
    z = ZoneFrame(live=(square(300, 400, 200),), hands=(square(300, 400, 200),))
    alpha = alpha_with_zone(SHAPE, [f], s, z) >= 0.5
    assert alpha.any(), "the top half is still masked"
    assert not alpha[420, 400], "the part behind the hand is not"


def test_the_zone_protects_all_of_its_polygon_and_not_just_the_middle():
    """The bug the output side check caught on three of four sample files.

    A shape drawn at its true size and then blurred is half covered at its own
    boundary, so the zone was drawn shrunk by the feather and faded inwards:
    two pixels inside the edge of a 200 px square the alpha read 0.137, which
    left 86 percent of a mask alive in a band about twelve pixels wide around
    every hand. The promise held in the middle of the zone and nowhere near
    its edge, which is exactly where a hand meets the thing it is holding.
    """
    s = Settings()
    poly = square(300, 300, 100)
    alpha = alpha_for(SHAPE, [poly], s.feather)
    inside = alpha[201:400, 201:400]
    assert inside.min() > 0.99, "the zone has to cover all of its own polygon"
    assert alpha[300, 402] > 0.5, "and err outwards, not inwards"
    assert alpha[300, 300 + 100 + int(4 * s.feather)] == 0.0, "but not for ever"


def test_a_face_at_the_very_edge_of_a_live_polygon_is_still_not_masked():
    """The same bug seen from the pipeline's side: a hand holding a card has
    the card at the edge of the polygon, not in its middle."""
    s = Settings()
    poly = square(300, 300, 120)
    f = face(x=300 - 30, y=300 + 120 - 60, size=55)      # sitting on the boundary
    z = ZoneFrame(live=(poly,), hands=(poly,))
    assert build_alpha(SHAPE, [f], s).any(), "it would have been masked"
    assert (alpha_with_zone(SHAPE, [f], s, z) >= 0.5).sum() == 0


def test_the_zone_off_reproduces_the_alpha_exactly():
    """The test that says the zone is subtractive and changed nothing else."""
    s = Settings()
    dets = [face(100, 100, 60), face(400, 300, 80)]
    plain = build_alpha(SHAPE, dets, s)
    assert np.array_equal(alpha_with_zone(SHAPE, dets, s, None), plain)
    assert np.array_equal(alpha_with_zone(SHAPE, dets, s, ZoneFrame()), plain)


def test_an_empty_zone_costs_nothing_and_changes_nothing():
    s = Settings()
    frame = np.full((*SHAPE, 3), 120, np.uint8)
    a, _ = redact(frame, [face()], s)
    b, _ = redact(frame, [face()], s, zone=ZoneFrame())
    assert np.array_equal(a, b)


def test_a_frame_fully_inside_the_zone_is_returned_untouched():
    """Not "almost untouched": the source pixel is copied through, which is
    what makes the zone a promise rather than a preference."""
    s = Settings()
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (*SHAPE, 3), dtype=np.uint8)
    z = ZoneFrame(live=(square(400, 300, 400),), hands=(square(400, 300, 400),))
    out, alpha = redact(frame, [face(350, 250, 80)], s, zone=z)
    assert not alpha.any()
    assert np.array_equal(out, frame)


# ---------------------------------------------------------------- the record

def test_the_build_keeps_the_hand_beside_its_reach():
    """`build` has to hand both out: the reach for text and screens, the hand
    itself for faces."""
    zones = build({0: [hand()]}, None, Settings(), 5, 10.0, SHAPE)
    assert zones[0].live and zones[0].hands
    assert polygon_area(zones[0].live[0]) > polygon_area(zones[0].hands[0])


def test_the_zone_share_is_the_share_of_the_frame_it_protects():
    z = ZoneFrame(live=(square(400, 300, 100),))
    assert share(SHAPE, z) == pytest.approx(200 * 200 / (SHAPE[0] * SHAPE[1]), rel=0.02)
    assert share(SHAPE, ZoneFrame()) == 0.0
    assert share(SHAPE, None) == 0.0


def test_the_summary_counts_frames_and_hands():
    zones = [ZoneFrame(live=(square(400, 300, 50),)), ZoneFrame(), ZoneFrame()]
    out = describe(zones, SHAPE)
    assert out["zone_frames"] == 1
    assert out["hands_per_frame_mean"] == pytest.approx(1 / 3, rel=1e-3)
    assert out["zone_share_max"] > out["zone_share_mean"] > 0


# ---------------------------------------------------------------- settings

def test_the_zone_is_on_by_default():
    """The rule says nothing handled is ever masked. A zone that had to be
    switched on would make that a preference."""
    assert Settings().zone is True


def test_every_zone_setting_is_validated_and_recorded():
    for bad in (dict(zone_window=8), dict(zone_stride=0), dict(zone_nms=0.0),
                dict(zone_min_hand_px=-1), dict(zone_edge_frac=1.5),
                dict(zone_scale=0.0), dict(zone_memory=-1.0),
                dict(zone_gate_changed=2.0), dict(zone_device="tpu")):
        with pytest.raises(SettingsError):
            Settings(**bad)
    recorded = Settings().to_dict()
    for name in ("zone", "zone_window", "zone_stride", "zone_nms", "zone_min_hand_px",
                 "zone_edge_frac", "zone_scale", "zone_memory", "zone_gate_changed",
                 "zone_device"):
        assert name in recorded, f"{name} is not in the audit record"


# ------------------------------------------------------------------ helpers

def test_two_polygons_that_touch_overlap_and_two_that_do_not_do_not():
    assert overlaps(square(100, 100, 50), square(140, 140, 50))
    assert not overlaps(square(100, 100, 50), square(400, 400, 50))


def test_a_polygon_moved_by_the_camera_keeps_its_shape():
    poly = square(100, 100, 50)
    assert polygon_area(moved(poly, 30, -10)) == pytest.approx(polygon_area(poly))


def test_a_polygon_off_the_edge_is_outside_the_frame():
    assert inside_frame(square(100, 100, 50), SHAPE)
    assert not inside_frame(square(-500, 100, 50), SHAPE)


# ---------------------------------------------- the report's own numbers

def test_the_orientation_tables_come_from_a_command_and_not_from_a_note():
    """Report 17.7's tables are printed by `eval.zone --orientation`.

    The pass that first produced them worked them out in a throwaway script,
    which is the one thing this project's own rule forbids: every number in
    the report has to come from something a third party can re-run. This
    fails if the command stops answering, or if the audit CSVs lose the
    columns it reads.
    """
    from eval.zone import orientation_tables, pointing

    text = orientation_tables()
    assert "The wearer's hands" in text and "A bystander's hands" in text
    assert "zone_orientation" in text
    assert "179 rows from 4 files" in text, "the committed audit is four files of 179"


def test_the_angle_is_recovered_from_the_quad_the_audit_committed():
    """Straight up the frame is zero, and the quad carries the rotation, so
    nothing has to be detected again to score the rule."""
    from eval.zone import pointing

    up = "100,100 200,100 200,300 100,300"        # top edge above bottom edge
    assert abs(pointing(up)) < 1e-6
    down = "100,300 200,300 200,100 100,100"
    assert abs(abs(pointing(down)) - 180) < 1e-6
    assert pointing("1,2 3,4") is None


# ------------------------------------------- the harness measures what ships

def test_the_hands_cache_is_rebuilt_when_the_zone_code_moves(tmp_path, monkeypatch):
    """The fingerprint carries `faceblur/zone.py` itself.

    The lesson of 2026-09-08: a cache that outlives the code that filled it
    makes every number after it describe a build that no longer exists. The
    hands cache decides what the zone protects in every F2 number, so it gets
    the same rule as the face cache.
    """
    import eval.zone as ez

    settings = Settings(mask=("face",))
    first = ez.hands_fingerprint(settings)
    fake = tmp_path / "faceblur"
    fake.mkdir()
    (fake / "zone.py").write_bytes(b"# a zone module that says something else\n")
    monkeypatch.setattr(ez, "REPO", tmp_path)
    assert ez.hands_fingerprint(settings) != first


def test_the_hands_cache_follows_the_settings_that_decide_what_is_cached():
    """And only those. A rebuild is an hour of GPU time on the long file, so
    a setting that is applied after the hands are found must not force one."""
    import eval.zone as ez

    base = Settings(mask=("face",))
    assert ez.hands_fingerprint(base) == ez.hands_fingerprint(Settings(mask=("face",)))
    narrower = Settings(mask=("face",), zone_window=384)
    assert ez.hands_fingerprint(narrower) != ez.hands_fingerprint(base)
    grown = Settings(mask=("face",), zone_scale=2.0)
    assert ez.hands_fingerprint(grown) == ez.hands_fingerprint(base)


def test_a_cached_hand_rebuilds_the_same_zone_as_the_hand_itself():
    """`zone_from_cache` is `build` with the models replaced by a file.

    If these two ever disagree the sweep scores a zone the pipeline would not
    draw, which is the fault report 17.4 found in the first place.
    """
    import eval.zone as ez

    settings = Settings(mask=("face",))
    h = hand()
    live = build({4: [h]}, None, settings, 8, 30.0, SHAPE)
    data = {"shape": list(SHAPE), "frames": {"4": [h.to_dict()]}}
    from_file = ez.zone_from_cache(data, settings, 8, 30.0, None)
    assert len(from_file) == len(live) == 8
    assert any(z.live for z in from_file)
    for a, b in zip(live, from_file):
        assert np.allclose(np.asarray(a.live, float),
                           np.asarray(b.live, float), atol=0.1)
        assert np.allclose(np.asarray(a.hands, float),
                           np.asarray(b.hands, float), atol=0.1)


def test_the_harness_says_so_when_it_has_no_hands_cache():
    """Silence would read as a zone of nothing, which scores more off face
    masking than the build produces. The note travels with the numbers."""
    from eval.measure import zones_for

    cache = {"shape": list(SHAPE), "frames": {}}
    zones, note = zones_for(None, Settings(mask=("face",)), cache, 3)
    assert len(zones) == 3 and not any(z.live for z in zones)
    assert "cache" in note
    off = Settings(mask=("face",), zone=False)
    assert zones_for(None, off, cache, 3)[1] == "the zone is off"
