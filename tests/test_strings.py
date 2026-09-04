"""The wording rules in Section 7, checked automatically.

The asd-ste100 skill is not installed in this environment. These tests enforce
the mechanical rules from that section so they cannot drift. Judgement rules,
such as active voice, were applied by hand when ui/strings.py was written.
"""
from __future__ import annotations

import re

import pytest

from ui import strings as S

# Every user facing string constant in the module.
TEXTS = {
    name: value
    for name, value in vars(S).items()
    if name.isupper() and isinstance(value, str) and not name.startswith("ACC_")
}
ALL_TEXTS = dict(TEXTS)
ALL_TEXTS.update({
    name: value for name, value in vars(S).items()
    if name.startswith("ACC_") and isinstance(value, str)
})

BANNED_WORDS = [
    "seamless", "powerful", "smart", "ai-powered", "ai powered", "effortless",
    "blazing", "cutting edge", "state of the art", "simply", "just",
]
PHRASAL_VERBS = ["kick off", "pick out", "set up the", "fill out", "check out"]

# One word for one thing. These synonyms must never appear.
BANNED_SYNONYMS = ["clip", "footage", "movie"]


def sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


@pytest.mark.parametrize("name", sorted(ALL_TEXTS))
def test_no_dashes(name):
    """No em dashes and no en dashes anywhere in the UI."""
    value = ALL_TEXTS[name]
    assert "—" not in value, f"{name} holds an em dash"
    assert "–" not in value, f"{name} holds an en dash"


@pytest.mark.parametrize("name", sorted(ALL_TEXTS))
def test_no_emoji(name):
    """No emoji in UI text. The status marks are geometric shapes, not emoji."""
    for char in ALL_TEXTS[name]:
        point = ord(char)
        assert not (0x1F300 <= point <= 0x1FAFF), f"{name} holds an emoji"
        assert not (0x1F000 <= point <= 0x1F0FF), f"{name} holds an emoji"
        assert point != 0xFE0F, f"{name} holds an emoji variation selector"


@pytest.mark.parametrize("name", sorted(TEXTS))
def test_at_most_twenty_words_per_sentence(name):
    for sentence in sentences(TEXTS[name]):
        words = [w for w in re.split(r"\s+", sentence) if w]
        assert len(words) <= 20, f"{name} has a {len(words)} word sentence: {sentence}"


@pytest.mark.parametrize("name", sorted(TEXTS))
def test_no_marketing_words(name):
    lowered = TEXTS[name].lower()
    for word in BANNED_WORDS:
        assert word not in lowered, f"{name} holds the marketing word {word!r}"


@pytest.mark.parametrize("name", sorted(TEXTS))
def test_no_phrasal_verbs_with_a_single_word_alternative(name):
    lowered = TEXTS[name].lower()
    for phrase in PHRASAL_VERBS:
        assert phrase not in lowered, f"{name} holds the phrasal verb {phrase!r}"


@pytest.mark.parametrize("name", sorted(TEXTS))
def test_one_word_for_one_thing(name):
    """The input is a video or a folder. Never a clip, footage or a movie."""
    words = re.findall(r"[a-z]+", TEXTS[name].lower())
    for banned in BANNED_SYNONYMS:
        assert banned not in words, f"{name} holds the word {banned!r}"


@pytest.mark.parametrize("name", sorted(TEXTS))
def test_sentence_case(name):
    """No Title Case and no ALL CAPS.

    A label may hold a proper noun, a placeholder or the app name, so the test
    only refuses a run of capitalised words and a run of shouting.
    """
    value = TEXTS[name]
    # A known acronym is not shouting.
    plain = value.replace("FaceBlur", "")
    for acronym in ("CPU", "GB", "MB", "KB", "TB", "PC"):
        plain = plain.replace(acronym, "")
    assert not re.search(r"\b[A-Z]{3,}\b", plain), f"{name} shouts: {value}"
    words = re.findall(r"\b[A-Za-z]+\b", re.sub(r"\{[^}]*\}", "", value))
    capitalised_run = 0
    for index, word in enumerate(words):
        if index and word[0].isupper() and word not in ("FaceBlur", "PC", "CPU", "Nth"):
            capitalised_run += 1
            assert capitalised_run < 3, f"{name} looks like Title Case: {value}"
        else:
            capitalised_run = 0


def test_every_reference_string_from_the_plan_is_present():
    """Section 7 lists reference strings. They must match exactly."""
    expected = {
        "DROP_ZONE_EMPTY": "Drop a video or a folder here",
        "CHOOSE_FILES": "Choose files",
        "CHOOSE_FOLDER": "Choose a folder",
        "OUTPUT_LABEL": "Output folder",
        "CHOOSE_OUTPUT": "Choose output folder",
        "START": "Blur faces",
        "STOP": "Stop",
        "ENGINE_STANDARD": "Standard",
        "ENGINE_STANDARD_HELP": "One detector. Faster. May miss small or turned faces.",
        "ENGINE_MAX": "Maximum coverage",
        "ENGINE_MAX_HELP": "Two detectors. About twice as slow. Finds more faces.",
        "STRIDE_HELP": "Detect faces on every Nth frame. 1 is safest. Higher is faster.",
        "REPLACE_EXISTING": "Replace existing files",
        "STATUS_WAITING": "Waiting",
        "STATUS_DETECTING": "Detecting",
        "STATUS_WRITING": "Writing",
        "STATUS_DONE": "Done",
        "STATUS_FAILED": "Failed",
        "STATUS_STOPPED": "Stopped",
        "STATUS_SKIPPED": "Already done",
        "STOP_CONFIRM": "Stop now? The app deletes unfinished files.",
        "SUMMARY": "{done} videos done. {failed} failed.",
        "OPEN_OUTPUT": "Open output folder",
        "ERR_UNREADABLE": ("Could not read this video. Check that the file is not "
                           "open in another program."),
        "ERR_NO_SPACE": "Not enough disk space in the output folder.",
        "ERR_FFMPEG": "Could not write the blurred copy. See the log file for details.",
    }
    for name, text in expected.items():
        assert getattr(S, name) == text, f"{name} drifted from the plan"


def test_every_status_word_has_a_mark():
    """No information is carried by colour alone, so every status shows a mark."""
    for word in (S.STATUS_WAITING, S.STATUS_DETECTING, S.STATUS_WRITING,
                 S.STATUS_DONE, S.STATUS_FAILED, S.STATUS_STOPPED, S.STATUS_SKIPPED):
        assert word in S.STATUS_MARK
        assert S.STATUS_MARK[word].strip()


def test_errors_say_what_happened_then_what_to_do():
    for name in ("ERR_UNREADABLE", "ERR_OUTPUT_UNWRITABLE", "ERR_MODELS",
                 "INPUT_NONE_FOUND"):
        parts = sentences(getattr(S, name))
        assert len(parts) >= 2, f"{name} does not say what to do next"


def test_the_help_text_does_not_upgrade_a_hedge_to_a_fact():
    assert "may miss" in S.ENGINE_STANDARD_HELP.lower()
    assert "may miss" in S.ABOUT_TEXT.lower()


@pytest.mark.parametrize("seconds,expected", [
    (10, "Less than a minute left."),
    (60, "About 1 minute left."),
    (140, "About 2 minutes left."),
    (3600, "About 1 hour left."),
    (4200, "About 1 hour 10 minutes left."),
    (7200, "About 2 hours left."),
    (8100, "About 2 hours 15 minutes left."),
])
def test_time_left_reads_as_a_sentence(seconds, expected):
    assert S.time_left(seconds) == expected


@pytest.mark.parametrize("count,size,expected", [
    (1, 1024 ** 3, "1 video, 1.0 GB"),
    (12, 9 * 1024 ** 3, "12 videos, 9.0 GB"),
])
def test_input_summary_counts_videos(count, size, expected):
    assert S.input_summary(count, size) == expected
