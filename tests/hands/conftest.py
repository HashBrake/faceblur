"""Shared paths and fixtures for the hands tests."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

CALIB_YAML = Path(__file__).parent / "data" / "calibration_camera.yaml"
FULL_FOOTAGE = Path(r"D:\Player One - Phase 0 full")
RECORDING_EV = FULL_FOOTAGE / "AZER76400EV" / "Orbbec_Ego_AZER76400EV_19700105_003108"
MANO_DIR = Path(os.environ.get("MANO_DIR", r"D:\Claude Projects\Player One Face Removal\cloud\mano"))

# The main faceblur venv has no MediaPipe or PyYAML. There the main suite still
# collects and these tests are left out; they run in the hands .venv.
if importlib.util.find_spec("mediapipe") is None or importlib.util.find_spec("yaml") is None:
    collect_ignore_glob = ["test_*.py"]


def pytest_configure(config):
    config.addinivalue_line("markers", "real_footage: needs the full footage drive")


@pytest.fixture
def recording_ev() -> Path:
    if not RECORDING_EV.is_dir():
        pytest.skip("full footage not present")
    return RECORDING_EV


@pytest.fixture
def mano_dir() -> Path:
    if not ((MANO_DIR / "MANO_LEFT.pkl").is_file() and (MANO_DIR / "MANO_RIGHT.pkl").is_file()):
        pytest.skip(f"MANO files not found at {MANO_DIR}")
    return MANO_DIR
