"""Stereo hand tracking for Orbbec Ego recordings.

The package undistorts the two Kannala-Brandt (KB) fisheye views, finds 2-D hand
keypoints in each view with MediaPipe Hands, triangulates them to 3-D with the
stereo calibration, smooths the tracks over time, and can optionally fit a MANO
hand model to the result.
"""

__all__ = []
