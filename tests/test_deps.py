"""Checks that the h5py and jsonschema pins agree everywhere they are written.

The privacy desensitization module reads HDF5 with h5py and validates audit
records with jsonschema. pyproject.toml is the source of truth for their
versions; the installed packages and both requirements files must match it,
so the Windows venv and the Linux CUDA image run the same code.
"""
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["h5py", "jsonschema"]


def _pyproject_pins():
    """Return {name: version} for every exact pin in [project] dependencies."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return dict(dep.split("==") for dep in data["project"]["dependencies"])


def _requirements_pins(path):
    """Return {name: version} for every pin line in a requirements file."""
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return dict(l.split("==") for l in lines if l and not l.startswith("#"))


@pytest.mark.parametrize("name", PACKAGES)
def test_installed_version_matches_pyproject(name):
    assert version(name) == _pyproject_pins()[name]


def test_jsonschema_in_test_extra_matches_main_pin():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert f"jsonschema=={_pyproject_pins()['jsonschema']}" in data["project"]["optional-dependencies"]["test"]


@pytest.mark.parametrize("path", ["requirements.txt", "cloud/requirements-cuda.txt"])
@pytest.mark.parametrize("name", PACKAGES)
def test_requirements_files_carry_same_pins(path, name):
    assert _requirements_pins(path)[name] == _pyproject_pins()[name]
