"""Text false positives: what is not a line of writing is not filled solid.

Package T13c. The field-test pilot (docs/privacy_module_fieldtest.md, defect
3) found most of the text mask on things: a grey rag and black cloth found
as squares up to 220 px, cups, a tool case, ceiling lights. `text.false_text`
drops a quad between the merge of the scales and the hold, by rules each
behind a setting, and the report's "Text false positives" has what each one
removed and cost on hand-labelled pilot frames.

The model is not trusted to produce a rag on a synthetic frame: on a drawn
frame it finds the drawn words and nothing in a noise patch or a white one.
So each rule is tested twice, once through the real model on a drawn frame,
and once through `TextDetector.detect` with the probability map built by
hand, which is the only way to put a square region in front of the rules
and see it go.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.settings import Settings, SettingsError
from faceblur.text import TextDetector, bands, false_text, shrunk, unclipped

SIZE = (480, 640)
PATCH = (380, 220, 200, 200)          # x, y, w, h of the cloth
LINES = (((40, 90), "INVOICE 4471"), ((40, 160), "JANE DOE"))
OFF = dict(text_min_aspect=0.0, text_bright_max=255.0, text_ink_min=0.0)


def sides(det) -> tuple[float, float]:
    (_, _), (a, b), _ = cv2.minAreaRect(np.asarray(det.quad, np.float32))
    return max(a, b), min(a, b)


def is_square(det, settings: Settings) -> bool:
    long_side, short_side = sides(det)
    return short_side > settings.text_square_px and long_side < settings.text_min_aspect * short_side


def overlaps(det, box, share=0.5) -> bool:
    """`det`'s box covers at least `share` of `box` (x, y, w, h)."""
    x, y, w, h = box
    ix = max(0.0, min(x + w, det.x + det.w) - max(x, det.x))
    iy = max(0.0, min(y + h, det.y + det.h) - max(y, det.y))
    return ix * iy >= share * w * h


def drawn(patch: np.ndarray | None = None, background: int = 128) -> np.ndarray:
    frame = np.full((*SIZE, 3), background, np.uint8)
    if patch is not None:
        x, y, w, h = PATCH
        frame[y:y + h, x:x + w] = patch[..., None]
    for origin, line in LINES:
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    return frame


def line_boxes() -> list[tuple[int, int, int, int]]:
    """Where the drawn lines' ink is, read back from a frame with only them."""
    ink = (drawn().min(axis=2) < 80).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((1, 41), np.uint8)))
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, count) if stats[i, 4] > 500]
    assert len(boxes) == 2
    return boxes


def detector_on(prob: np.ndarray, settings: Settings) -> TextDetector:
    """A TextDetector whose map is `prob` at every scale, at scale 1."""
    det = object.__new__(TextDetector)
    det.settings = settings
    det.map_for = lambda frame, size: (prob, 1.0)
    return det


def lines_map() -> np.ndarray:
    prob = np.zeros(SIZE, np.float32)
    for x, y, w, h in line_boxes():
        prob[y + 4:y + h - 4, x + 2:x + w - 2] = 0.95
    return prob


@pytest.fixture(scope="module")
def model():
    return TextDetector(Settings(device="cpu"))


# ------------------------------------------------- (a) cloth beside two lines

def test_the_model_keeps_two_lines_beside_a_noise_patch_and_no_square(model):
    settings = model.settings
    patch = np.random.default_rng(3).integers(0, 256, (200, 200), dtype=np.uint8)
    kept = model.detect(drawn(patch))
    for box in line_boxes():
        assert any(overlaps(d, box) for d in kept), f"the line at {box} was lost"
    assert not [d for d in kept if is_square(d, settings)], "a square quad was kept"


def cloth_map(seed: int = 0) -> np.ndarray:
    """The lines, and a rag the way DB sees one: a speckled region of ink."""
    prob = lines_map()
    x, y, w, h = PATCH
    speck = np.random.default_rng(seed).random((h, w)) < 0.85
    prob[y:y + h, x:x + w] = np.where(speck, 0.9, 0.0)
    return prob


def test_a_square_region_of_speckled_ink_is_dropped_and_the_lines_survive():
    frame = drawn(np.random.default_rng(3).integers(0, 256, (200, 200), dtype=np.uint8))
    without = detector_on(cloth_map(), Settings(**OFF)).detect(frame)
    assert any(overlaps(d, PATCH) for d in without), "without the rules the rag is found"

    settings = Settings()
    kept = detector_on(cloth_map(), settings).detect(frame)
    assert not any(overlaps(d, PATCH, 0.2) for d in kept), "the rag is still masked"
    assert not [d for d in kept if is_square(d, settings)]
    for box in line_boxes():
        assert any(overlaps(d, box) for d in kept), f"the line at {box} was lost"


def test_a_block_of_stacked_lines_found_as_one_region_is_kept():
    """Three lines DB ran together are a square too. The bands in the map
    are what tells them from a rag, and that is what the exception is for:
    a label or a page held up is the text the rule exists to mask."""
    prob = np.zeros(SIZE, np.float32)
    for k in range(3):
        prob[200 + 44 * k:232 + 44 * k, 200:360] = 0.95
    prob[200:320, 200:204] = 0.95            # what ran the three together
    frame = np.full((*SIZE, 3), 128, np.uint8)
    block = (200, 200, 160, 120)
    kept = detector_on(prob, Settings()).detect(frame)
    assert any(overlaps(d, block) for d in kept), "a block of lines was dropped"
    assert all(is_square(d, Settings()) for d in kept if overlaps(d, block))
    no_exception = detector_on(prob, Settings(text_square_bands=0)).detect(frame)
    assert not any(overlaps(d, block) for d in no_exception)


def test_a_tilted_line_is_not_called_square_by_its_upright_box():
    """The rule reads the quad's own sides. A long line at 45 degrees has an
    upright box that is square and over 60 px, and it is still a line."""
    prob = np.zeros(SIZE, np.float32)
    cv2.line(prob, (150, 350), (350, 150), 0.95, 16)
    frame = np.full((*SIZE, 3), 128, np.uint8)
    kept = detector_on(prob, Settings()).detect(frame)
    assert len(kept) == 1
    assert kept[0].w > 60 and kept[0].h > 60 and kept[0].w < 1.5 * kept[0].h


# ------------------------------------------------- (b) a washed-out patch

WHITE = (270, 190, 100, 100)


def white_frame() -> np.ndarray:
    frame = np.full((*SIZE, 3), 180, np.uint8)
    x, y, w, h = WHITE
    frame[y:y + h, x:x + w] = 250
    return frame


def test_the_model_keeps_no_quad_over_a_white_patch(model):
    kept = model.detect(white_frame())
    assert not any(overlaps(d, WHITE, 0.2) for d in kept)


def test_a_washed_out_region_is_dropped_by_its_brightness():
    prob = np.zeros(SIZE, np.float32)
    x, y, w, h = WHITE
    prob[y:y + h, x:x + w] = 0.95
    frame = white_frame()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = detector_on(prob, Settings(**OFF)).detect(frame)
    assert len(found) == 1 and overlaps(found[0], WHITE), "without the rules it is found"

    # The grown quad takes in the wall: its mean is about 200. The rule reads
    # the region the map marked, which is the patch at 250.
    only_bright = Settings(**dict(OFF, text_bright_max=240.0))
    assert false_text(found[0], grey, prob, 1.0, only_bright) == "bright"
    assert false_text(found[0], grey, prob, 1.0, Settings(**OFF)) is None
    assert not any(overlaps(d, WHITE, 0.2)
                   for d in detector_on(prob, Settings()).detect(frame))


def test_dark_text_on_a_light_page_is_not_washed_out():
    prob = lines_map()
    frame = drawn(background=235)
    kept = detector_on(prob, Settings()).detect(frame)
    for box in line_boxes():
        assert any(overlaps(d, box) for d in kept), f"the line at {box} was lost"


# ------------------------------------------------- the ink rule, off by default

def test_the_ink_rule_drops_a_sparse_region_when_it_is_turned_on():
    prob = lines_map()
    sparse = np.random.default_rng(1).random((40, 300)) < 0.5
    prob[300:340, 100:400] = np.where(sparse, 0.95, 0.0)
    prob[300:340, 100:400][::2] = 0.95                  # one connected region
    frame = np.full((*SIZE, 3), 128, np.uint8)
    region = (100, 300, 300, 40)
    assert any(overlaps(d, region) for d in detector_on(prob, Settings()).detect(frame))
    strict = Settings(text_ink_min=0.9)
    assert not any(overlaps(d, region) for d in detector_on(prob, strict).detect(frame))


# ------------------------------------------------- the pieces

def test_shrunk_undoes_the_unclip():
    for rect in (((100.0, 80.0), (120.0, 20.0), 0.0), ((60.0, 60.0), (50.0, 50.0), 30.0)):
        grown = cv2.boxPoints(unclipped(rect, 1.6))
        (_, _), (w, h), _ = cv2.minAreaRect(shrunk(grown, 1.6))
        assert sorted((w, h)) == pytest.approx(sorted(rect[1]), abs=0.01)


def test_bands_counts_stacked_lines():
    mask = np.ones((60, 40), bool)
    ink = np.zeros((60, 40), bool)
    assert bands(ink, mask) == 0
    ink[5:15] = ink[25:35] = ink[45:55] = True
    assert bands(ink, mask) == 3
    speckle = np.random.default_rng(0).random((60, 40)) < 0.85
    assert bands(speckle, mask) == 1


# ------------------------------------------------- (d) the settings

NEW = ("text_min_aspect", "text_square_px", "text_square_bands", "text_bright_max",
       "text_ink_min")


def test_every_new_setting_is_recorded():
    recorded = Settings().to_dict()
    for name in NEW:
        assert name in recorded, f"{name} is not in the audit record"
        assert recorded[name] == getattr(Settings(), name)


@pytest.mark.parametrize("bad", [
    {"text_min_aspect": -0.1}, {"text_square_px": -1}, {"text_square_bands": -1},
    {"text_bright_max": -1.0}, {"text_bright_max": 256.0},
    {"text_ink_min": -0.1}, {"text_ink_min": 1.1},
])
def test_every_new_setting_is_validated(bad):
    with pytest.raises(SettingsError):
        Settings(**bad)


def test_the_defaults_are_the_ones_the_measurement_chose():
    """Report, "Text false positives": shape on at 1.5 and 40 px with the
    stacked-lines exception; brightness at 240; the ink rule, a box threshold
    of 0.7 and a run of 4 frames each lost real text on the labelled set and
    are not taken."""
    s = Settings()
    assert (s.text_min_aspect, s.text_square_px, s.text_square_bands) == (1.5, 40, 2)
    assert s.text_bright_max == 240.0 and s.text_ink_min == 0.0
    assert s.text_box_thresh == 0.6 and s.text_min_run == 2
