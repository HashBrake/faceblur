"""The hands environment has the pinned packages and the calibration fixture."""
from __future__ import annotations

import importlib

import yaml

from hands.conftest import CALIB_YAML


def test_hands_package_imports():
    import hands

    assert hands.__all__ == []


def test_mediapipe_pinned_version():
    import mediapipe

    assert mediapipe.__version__ == "0.10.21"


def test_numpy_pinned_version():
    import numpy

    assert numpy.__version__.startswith("1.26")


def test_support_packages_import():
    for name in ("pyarrow", "yaml", "scipy"):
        importlib.import_module(name)
    import cv2

    assert hasattr(cv2, "fisheye")
    assert callable(cv2.fisheye.undistortPoints)


def test_calibration_yaml_has_two_kb_cameras():
    calib = yaml.safe_load(CALIB_YAML.read_text(encoding="utf-8"))
    cameras = calib["cameras"]
    assert len(cameras) == 2
    assert all(cam["distortion_model"] == "KB" for cam in cameras)
