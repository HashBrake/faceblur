"""2-D hand keypoints per view with MediaPipe Hands (legacy solution, CPU).

Handedness: MediaPipe labels hands as if the image were mirrored (selfie view).
Our egocentric cameras are not mirrored, so the label is swapped: MediaPipe
"Left" is the wearer's right hand and "Right" the wearer's left hand.

Per-joint confidence: MediaPipe does not fill per-landmark visibility/presence
for hands, so ``conf`` is the hand's handedness score replicated per joint.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

MEDIAPIPE_JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# (parent, child): each finger is a chain wrist -> base -> ... -> tip.
BONES = [(0 if j % 4 == 1 else j - 1, j) for j in range(1, 21)]

_SWAP = {"Left": "right", "Right": "left"}


@dataclass
class Hand2D:
    keypoints: np.ndarray  # (21, 2) float64 pixels, x right, y down
    conf: np.ndarray       # (21,) float64, handedness score replicated
    label: str             # "left" | "right": the WEARER's hand
    score: float           # handedness score
    world: np.ndarray      # (21, 3) float64 metres, hand-centred (MediaPipe world landmarks)


def hands_from_result(results, width: int, height: int) -> list[Hand2D]:
    if not results.multi_hand_landmarks:
        return []
    out = []
    for lms, wlms, hd in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks,
                             results.multi_handedness):
        kp = np.array([[p.x * width, p.y * height] for p in lms.landmark], dtype=np.float64)
        world = np.array([[p.x, p.y, p.z] for p in wlms.landmark], dtype=np.float64)
        cls = hd.classification[0]
        score = float(cls.score)
        out.append(Hand2D(kp, np.full(21, score, dtype=np.float64), _SWAP[cls.label], score, world))
    return out


class HandDetector:
    """One MediaPipe Hands instance; tracking state is per instance, so use one per view."""

    def __init__(self, max_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, static_image_mode=False):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode, max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def detect(self, image_bgr: np.ndarray) -> list[Hand2D]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return hands_from_result(self._hands.process(rgb), w, h)

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
