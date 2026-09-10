"""The kinds of sensitive thing this tool can mask, and what each one is.

Until 2026-09-10 there was one kind and the code said "face" everywhere. The
scope is now three, each a switch the user can throw by itself, so the code
has to carry the kind with every box instead of assuming it.

A kind is not a detector. It is what the user asked to have hidden, and it
decides three things the rest of the pipeline needs to know:

- **the shape of the mask.** A face is hidden by an ellipse over the eyes,
  nose and mouth, because the corners of its box are hair and background. A
  line of text is hidden by the quadrilateral the text sits in, because text
  is not oval and the corners of its box are the page. `redact.py` reads
  `shape` and nothing else.
- **whether it is ready.** A kind with no detector is not offered: a checkbox
  that masks nothing is worse than no checkbox. `available()` is what the
  window and the command line list, and it is derived from what is actually
  wired up, never from a hand-kept list.
- **what to call it.** One name for the audit record and the command line,
  one for a person reading the window.

Adding a kind is: write the detector, add the row here, and the switch, the
record, the gate and the report follow.
"""
from __future__ import annotations

from dataclasses import dataclass

# Mask geometry. ELLIPSE is the identity ellipse of `redact.ellipse_for`;
# POLYGON is the four corners a detector gives, masked as they come.
ELLIPSE = "ellipse"
POLYGON = "polygon"


@dataclass(frozen=True)
class Kind:
    """One class of sensitive thing."""

    name: str          # what the audit record and --mask call it
    label: str         # what the window calls it
    shape: str         # ELLIPSE or POLYGON
    summary: str       # one line, for --help and the window's help text
    ready: bool        # is there a detector behind it yet

    def __str__(self) -> str:
        return self.name


FACE = Kind(
    name="face",
    label="Faces",
    shape=ELLIPSE,
    summary="every face, hidden by an ellipse over the eyes, nose and mouth",
    ready=True,
)

TEXT = Kind(
    name="text",
    label="Personal text",
    shape=POLYGON,
    summary="names, addresses, phone numbers and handwriting, leaving card "
            "names, prices and grading labels alone",
    ready=False,
)

SCREEN = Kind(
    name="screen",
    label="Screens",
    shape=POLYGON,
    summary="a phone, monitor or television as an object, whatever is on it",
    ready=False,
)

# Order matters: it is the order the window lists them in, and faces are the
# one most runs want.
KINDS: tuple[Kind, ...] = (FACE, TEXT, SCREEN)
BY_NAME: dict[str, Kind] = {k.name: k for k in KINDS}


class UnknownKind(ValueError):
    """A name that is not one of the kinds this build knows."""


def available() -> tuple[Kind, ...]:
    """The kinds with a detector behind them, in listing order."""
    return tuple(k for k in KINDS if k.ready)


def kind(name: str) -> Kind:
    try:
        return BY_NAME[name]
    except KeyError:
        raise UnknownKind(
            f"{name!r} is not something this build can mask. It knows "
            f"{', '.join(k.name for k in KINDS)}.") from None


def parse(text: str) -> tuple[str, ...]:
    """Read a --mask value such as 'face,screen' into kind names.

    Order and duplicates are dropped: the result is in listing order, so the
    audit record reads the same however the flag was typed.
    """
    wanted = {part.strip() for part in text.split(",") if part.strip()}
    if not wanted:
        raise UnknownKind("--mask needs at least one kind of thing to mask.")
    for name in sorted(wanted):
        found = kind(name)
        if not found.ready:
            raise UnknownKind(
                f"This build cannot mask {name} yet: {found.summary}.")
    return tuple(k.name for k in KINDS if k.name in wanted)
