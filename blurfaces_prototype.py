#!/usr/bin/env python3
"""Blur faces in video for PII removal. See README.md."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).parent / "models" / "yunet.onnx"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class Detector:
    def __init__(self, conf, nms, scales):
        if not MODEL.exists():
            sys.exit(f"Missing model file: {MODEL}\nRun setup again (see README).")
        self.net = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), conf, nms, 5000)
        self.scales = scales

    def detect(self, frame):
        """Return list of [x, y, w, h, score] in original frame coords."""
        h, w = frame.shape[:2]
        out = []
        for s in self.scales:
            img = frame if s == 1.0 else cv2.resize(frame, (int(w * s), int(h * s)))
            ih, iw = img.shape[:2]
            self.net.setInputSize((iw, ih))
            _, faces = self.net.detect(img)
            if faces is None:
                continue
            for f in faces:
                out.append([f[0] / s, f[1] / s, f[2] / s, f[3] / s, f[-1]])
        return nms_merge(out)


def nms_merge(boxes, thr=0.35):
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    area = b[:, 2] * b[:, 3]
    order = b[:, 4].argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return b[keep].tolist()


def propagate(per_frame, window, grow):
    """Spread each detection forward and backward `window` frames, growing the box
    each step to cover motion. Fills gaps where the detector blinks."""
    n = len(per_frame)
    out = [list(f) for f in per_frame]
    for i, boxes in enumerate(per_frame):
        for x, y, w, h, s in boxes:
            for d in range(1, window + 1):
                g = 1.0 + grow * d
                nw, nh = w * g, h * g
                nx, ny = x - (nw - w) / 2, y - (nh - h) / 2
                for j in (i - d, i + d):
                    if 0 <= j < n:
                        out[j].append([nx, ny, nw, nh, s])
    return [nms_merge(f, 0.6) for f in out]


def redact(frame, boxes, pad, mode, strength):
    if not boxes:
        return frame
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh, _ in boxes:
        px, py = bw * pad, bh * pad
        x0, y0 = int(max(0, x - px)), int(max(0, y - py))
        x1, y1 = int(min(w, x + bw + px)), int(min(h, y + bh + py))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    if not mask.any():
        return frame

    if mode == "solid":
        out = frame.copy()
        out[mask > 0] = 0
        return out

    # Both remaining modes destroy information by downsampling, so the original
    # pixels cannot be recovered from the output.
    k = max(1, int(round(min(h, w) / strength)))
    small = cv2.resize(frame, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    if mode == "pixelate":
        cover = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:  # blur
        cover = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        cover = cv2.GaussianBlur(cover, (0, 0), k)

    m3 = cv2.merge([mask] * 3) > 0
    return np.where(m3, cover, frame)


def process(src, dst, args, det):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return {"file": src.name, "error": "could not open"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Pass 1: detect
    t0 = time.time()
    per_frame, idx = [], 0
    last = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            last = det.detect(frame)
        per_frame.append(last if idx % args.stride == 0 else [])
        idx += 1
        if args.progress and total and idx % 50 == 0:
            pct = 100 * idx / total
            print(f"\r  {src.name}: detecting {pct:5.1f}%", end="", flush=True)
    cap.release()
    raw_hits = sum(1 for f in per_frame if f)
    raw_boxes = sum(len(f) for f in per_frame)

    window = max(args.persist, args.stride)
    per_frame = propagate(per_frame, window, args.grow)
    cov_hits = sum(1 for f in per_frame if f)
    detect_s = time.time() - t0

    # Pass 2: redact and encode, muxing audio from the original
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           "-i", str(src),
           "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", args.preset,
           "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-movflags", "+faststart", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(str(src))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = redact(frame, per_frame[i] if i < len(per_frame) else [],
                         args.pad, args.mode, args.strength)
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            i += 1
            if args.progress and total and i % 50 == 0:
                print(f"\r  {src.name}: writing   {100*i/total:5.1f}%", end="", flush=True)
    finally:
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if args.progress:
        print("\r" + " " * 60 + "\r", end="")
    if proc.returncode != 0:
        return {"file": src.name, "error": f"ffmpeg failed: {err.strip()[:400]}"}

    return {
        "file": src.name, "output": dst.name, "frames": i,
        "resolution": f"{w}x{h}", "fps": round(fps, 3),
        "frames_with_detection": raw_hits, "detections": raw_boxes,
        "frames_covered_after_propagation": cov_hits,
        "settings": {"conf": args.conf, "stride": args.stride, "persist": args.persist,
                     "pad": args.pad, "mode": args.mode, "scales": args.scales},
        "detect_seconds": round(detect_s, 1),
    }


def main():
    p = argparse.ArgumentParser(
        description="Blur every detected face in a video (PII removal).")
    p.add_argument("input", help="video file or folder of videos")
    p.add_argument("-o", "--output", default=None,
                   help="output file or folder (default: <input>_blurred)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="detection threshold; lower = more faces caught, more false blurs (default 0.25)")
    p.add_argument("--scales", default="1.0,1.75",
                   help="detection scales; add e.g. 2.5 for very small/distant faces (default 1.0,1.75)")
    p.add_argument("--stride", type=int, default=1,
                   help="detect every Nth frame; >1 is faster and leans on propagation (default 1)")
    p.add_argument("--persist", type=int, default=6,
                   help="frames to carry a box forward AND backward past its detection (default 6)")
    p.add_argument("--grow", type=float, default=0.06,
                   help="box growth per propagated frame, to cover motion (default 0.06)")
    p.add_argument("--pad", type=float, default=0.30,
                   help="box padding as a fraction of face size (default 0.30)")
    p.add_argument("--mode", choices=["blur", "pixelate", "solid"], default="blur")
    p.add_argument("--strength", type=int, default=28,
                   help="higher = stronger destruction (default 28)")
    p.add_argument("--crf", type=int, default=20, help="x264 quality, lower = better (default 20)")
    p.add_argument("--preset", default="medium", help="x264 preset (default medium)")
    p.add_argument("--no-progress", dest="progress", action="store_false")
    p.add_argument("--report", default=None, help="write a JSON audit record here")
    args = p.parse_args()
    args.scales = [float(s) for s in args.scales.split(",") if s.strip()]

    src = Path(args.input)
    if src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in VIDEO_EXT)
        outdir = Path(args.output) if args.output else src.parent / f"{src.name}_blurred"
    elif src.is_file():
        files = [src]
        outdir = Path(args.output).parent if args.output else src.parent
    else:
        sys.exit(f"Not found: {src}")
    if not files:
        sys.exit(f"No videos found in {src}")
    outdir.mkdir(parents=True, exist_ok=True)

    det = Detector(args.conf, 0.3, args.scales)
    results = []
    for n, f in enumerate(files, 1):
        if src.is_file() and args.output:
            dst = Path(args.output)
        else:
            dst = outdir / f"{f.stem}_blurred.mp4"
        print(f"[{n}/{len(files)}] {f.name}")
        r = process(f, dst, args, det)
        results.append(r)
        if "error" in r:
            print(f"  FAILED: {r['error']}")
        else:
            print(f"  -> {dst}  ({r['detections']} detections over {r['frames']} frames, "
                  f"{r['frames_covered_after_propagation']}/{r['frames']} frames redacted, "
                  f"{r['detect_seconds']}s detect)")

    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2))
        print(f"Report: {args.report}")
    if any("error" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
