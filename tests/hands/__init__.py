"""Tests for the hands package.

pyproject runs pytest in its default "prepend" import mode and tests/ has no
__init__.py, so pytest imports this directory as the top-level package
``hands``, which would shadow the real package at the repo root. This file
therefore runs the real ``hands/__init__.py`` in its place and puts the real
package directory first on ``__path__``: ``import hands`` and
``from hands.x import y`` reach the real code, while pytest still finds the test
modules (``hands.test_*``, ``hands.conftest``) in this directory.
"""
from pathlib import Path

_REAL_INIT = Path(__file__).resolve().parents[2] / "hands" / "__init__.py"
__path__ = [str(_REAL_INIT.parent), *__path__]
exec(compile(_REAL_INIT.read_text(encoding="utf-8"), str(_REAL_INIT), "exec"))
