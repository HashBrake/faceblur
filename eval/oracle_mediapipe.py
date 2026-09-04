"""Third face detector and hand regions, from MediaPipe. Runs in .venv-eval.

    .venv-eval\\Scripts\\python.exe eval\\oracle_mediapipe.py VIDEO --stride 10

Writes eval/cache/<stem>.oracle.json with, for every stride-th frame, BlazeFace
boxes and the convex hull of every hand. MediaPipe is Apache 2.0. It is not a
dependency of the app; it only supplies evidence for the harness.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import mediapipe as mp

    faces = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                      min_detection_confidence=0.3)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=4,
                                     min_detection_confidence=0.5)
    video = Path(args.video)
    out = Path(args.out) if args.out else REPO / "eval" / "cache" / f"{video.stem}.oracle.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result = {"video": video.name, "stride": args.stride, "shape": [H, W], "frames": {}}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            entry = {"faces": [], "hands": []}
            fr = faces.process(rgb)
            for d in fr.detections or []:
                bb = d.location_data.relative_bounding_box
                entry["faces"].append([bb.xmin * W, bb.ymin * H, bb.width * W,
                                       bb.height * H, float(d.score[0])])
            hr = hands.process(rgb)
            for hand in hr.multi_hand_landmarks or []:
                pts = np.array([[lm.x * W, lm.y * H] for lm in hand.landmark], np.float32)
                hull = cv2.convexHull(pts).reshape(-1, 2)
                entry["hands"].append([[float(x), float(y)] for x, y in hull])
            result["frames"][str(i)] = entry
        i += 1
    cap.release()
    out.write_text(json.dumps(result), encoding="utf-8")
    n = len(result["frames"])
    print(f"{video.name}: {n} frames, "
          f"{sum(len(f['faces']) for f in result['frames'].values())} blazeface boxes, "
          f"{sum(len(f['hands']) for f in result['frames'].values())} hands -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
