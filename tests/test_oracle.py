"""The object oracle: its cache, and the rules that turn it into a number.

The oracle itself is 1.4 GB of weights in a third environment and none of that
is tested here. What is tested is everything around it that can be wrong
quietly: a cache that goes stale without saying so, which is the failure
`eval/common.py` had on 2026-09-08 and which made every number for a week
describe a build that no longer existed; and the two rules that decide what
counts as an oracle screen, which are what a recall number is actually made
of.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval import oracle_owl                                        # noqa: E402
from eval.oracle_owl import (CARD_PROMPTS, MODEL, PROMPTS, REVISION,  # noqa: E402
                             SCREEN_PROMPTS, WEIGHTS_BYTES, WEIGHTS_SHA256,
                             WeightsWrong, boxes_for, cache_path, fingerprint,
                             verify_weights)
from eval.screens import MIN_SHARE, oracle_screens                 # noqa: E402


def frame_rows(*rows):
    """A cache holding one frame, in the shape `oracle_owl.run` writes."""
    return {"shape": [1000, 1000], "stride": 5,
            "frames": {"0": [dict(r) for r in rows]}}


def row(label="a television", score=0.5, box=(100, 100, 300, 300)):
    return {"label": label, "score": score, "box": list(box)}


# ------------------------------------------------------------- the cache

def test_the_cache_says_what_it_depends_on():
    """A cache that cannot tell it is stale is worse than no cache: it keeps
    answering, with last week's answer."""
    assert fingerprint(5, 0.1) == fingerprint(5, 0.1)
    assert fingerprint(5, 0.1) != fingerprint(10, 0.1), "the stride is in it"
    assert fingerprint(5, 0.1) != fingerprint(5, 0.2), "the threshold is in it"


def test_editing_the_oracle_invalidates_its_cache(monkeypatch, tmp_path):
    """The module's own source is in the fingerprint, so changing a prompt or
    the way a box is written down rebuilds rather than misleads. An edit to a
    comment costs a rerun; a silently wrong measurement costs more."""
    before = fingerprint(5, 0.1)
    fake = tmp_path / "oracle_owl.py"
    fake.write_text(Path(oracle_owl.__file__).read_text(encoding="utf-8") + "\n# a change\n",
                    encoding="utf-8")
    monkeypatch.setattr(oracle_owl, "__file__", str(fake))
    assert fingerprint(5, 0.1) != before


def test_a_cache_from_another_fingerprint_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle_owl, "CACHE_DIR", tmp_path)
    video = tmp_path / "clip.mp4"
    path = cache_path(video)
    path.write_text(json.dumps({"fingerprint": "not the one", "frames": {}}),
                    encoding="utf-8")
    assert oracle_owl.load(video) is None


def test_a_cache_that_is_not_json_is_not_used(tmp_path, monkeypatch):
    """A run killed part way through used to be able to leave one of these.
    It writes through a temporary file now, and this is the belt."""
    monkeypatch.setattr(oracle_owl, "CACHE_DIR", tmp_path)
    video = tmp_path / "clip.mp4"
    cache_path(video).write_text("{ half a file", encoding="utf-8")
    assert oracle_owl.load(video) is None


def test_the_prompts_cover_every_question_the_later_packages_ask():
    """Screens for the recall number, cards for the veto, the rest for the
    surface column. A prompt missing here is a question nobody can answer."""
    for group in (SCREEN_PROMPTS, CARD_PROMPTS):
        for prompt in group:
            assert prompt in PROMPTS
    assert len(set(PROMPTS)) == len(PROMPTS), "a prompt is listed twice"


def test_rows_can_be_read_back_by_prompt_and_by_score():
    data = frame_rows(row(score=0.5), row("a trading card", 0.2), row("a sign", 0.9))
    assert len(boxes_for(data, 0)) == 3
    assert len(boxes_for(data, 0, SCREEN_PROMPTS)) == 1
    assert len(boxes_for(data, 0, floor=0.4)) == 2
    assert boxes_for(data, 999) == []


# ------------------------------------------- what counts as an oracle screen

def test_a_screen_too_small_to_read_is_not_counted():
    """The shipped class is not trying to find a monitor 0.1 percent of the
    frame across a hall, and counting one against it would be measuring a
    promise nobody made."""
    big = frame_rows(row(box=(0, 0, 200, 200)))          # 4 % of the frame
    small = frame_rows(row(box=(0, 0, 40, 40)))          # 0.16 %
    assert len(oracle_screens(big, 0, 0.3)) == 1
    assert oracle_screens(small, 0, 0.3) == []


def test_the_size_floor_is_the_share_of_the_frame_it_says_it_is():
    side = int((MIN_SHARE * 1000 * 1000) ** 0.5)
    over = frame_rows(row(box=(0, 0, side + 2, side + 2)))
    under = frame_rows(row(box=(0, 0, side - 2, side - 2)))
    assert len(oracle_screens(over, 0, 0.3)) == 1
    assert oracle_screens(under, 0, 0.3) == []


def test_overlapping_proposals_for_one_screen_are_counted_once():
    """OWLv2 does no suppression of its own and returns several boxes for one
    object. Left in, recall stops being about screens and becomes about how
    many proposals the shipped rule happened to cover."""
    data = frame_rows(row(score=0.6, box=(100, 100, 300, 300)),
                      row(score=0.5, box=(105, 104, 302, 298)),
                      row(score=0.4, box=(98, 96, 297, 305)))
    kept = oracle_screens(data, 0, 0.3)
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.6), "the surest proposal survives"


def test_two_screens_side_by_side_are_counted_twice():
    data = frame_rows(row(box=(0, 0, 200, 200)), row(box=(700, 700, 900, 900)))
    assert len(oracle_screens(data, 0, 0.3)) == 2


def test_only_the_screen_prompts_count_as_a_screen():
    """A card and a sign are in the cache for other work packages, and a
    recall number for screens must not be made of them."""
    data = frame_rows(row("a trading card", 0.9, (0, 0, 300, 300)),
                      row("a sign", 0.9, (400, 400, 700, 700)),
                      row("a laptop", 0.9, (0, 700, 300, 950)))
    kept = oracle_screens(data, 0, 0.3)
    assert [d.source for d in kept] == ["laptop"]


def test_the_floor_is_applied_before_anything_else():
    data = frame_rows(row(score=0.2, box=(0, 0, 300, 300)))
    assert oracle_screens(data, 0, 0.3) == []
    assert len(oracle_screens(data, 0, 0.15)) == 1


def test_an_empty_frame_answers_nothing_rather_than_raising():
    assert oracle_screens(frame_rows(), 0, 0.3) == []
    assert oracle_screens({"shape": [10, 10], "frames": {}}, 5, 0.3) == []


# ------------------------------------------------------------- the weights

def test_the_pinned_weights_are_the_ones_the_documentation_names():
    """One hash, two places, and they have to agree.

    `models/README.md` is where a reader looks for what a model is; the
    constant in `eval/oracle_owl.py` is what the code checks against. Two
    copies of a number drift, so this fails when they do.
    """
    readme = (REPO / "models" / "README.md").read_text(encoding="utf-8")
    assert WEIGHTS_SHA256 in readme, (
        "the oracle weight hash in eval/oracle_owl.py is not in models/README.md")
    assert str(WEIGHTS_BYTES) in readme
    assert REVISION in readme and MODEL in readme


def test_weights_that_are_the_wrong_size_stop_the_run(monkeypatch, tmp_path):
    """A detector with altered weights does not fail, it answers differently,
    and two hours later the answer is in a report attributed to a model nobody
    can identify. So it stops before it starts."""
    wrong = tmp_path / "model.safetensors"
    wrong.write_bytes(b"not the weights")
    monkeypatch.setattr(oracle_owl, "weights_path", lambda: wrong)
    with pytest.raises(WeightsWrong) as exc:
        verify_weights()
    assert str(WEIGHTS_BYTES) in str(exc.value)


def test_weights_that_are_missing_say_how_to_fetch_them(monkeypatch):
    monkeypatch.setattr(oracle_owl, "weights_path", lambda: None)
    with pytest.raises(WeightsWrong) as exc:
        verify_weights()
    assert REVISION in str(exc.value)


def test_weights_that_match_the_pin_are_accepted(monkeypatch, tmp_path):
    import hashlib

    body = b"x" * 32
    good = tmp_path / "model.safetensors"
    good.write_bytes(body)
    monkeypatch.setattr(oracle_owl, "weights_path", lambda: good)
    monkeypatch.setattr(oracle_owl, "WEIGHTS_BYTES", len(body))
    monkeypatch.setattr(oracle_owl, "WEIGHTS_SHA256", hashlib.sha256(body).hexdigest())
    assert verify_weights() == hashlib.sha256(body).hexdigest()
