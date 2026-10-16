"""Random access check: seeded seeks on a stored copy and on its source.

A hot-storage copy must serve any frame on request. This decodes both files
in order with OpenCV and with PyAV, then seeks to the same seeded random frame
indices in each file with each library, and checks that every seek returns
exactly the frame a sequential decode gives at that index (per library), and
that seeking the copy is at most twice as slow as seeking the source.

    python -m storage.randaccess CONVERTED --source H264 [--seeks 100] [--seed 0] [--json PATH]

Exit 0 when every seek was exact and both mean-seek ratios are <= 2.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import av
import cv2

LIBS = ("opencv", "pyav")
MAX_RATIO = 2.0


def _digest(bgr) -> str:
    return hashlib.sha256(bgr.tobytes()).hexdigest()


def sequential_hashes(path: Path, lib: str) -> dict:
    """Hash of every frame in decode order; PyAV also gives each frame's pts and the keyframes."""
    hashes, pts, keyframes = [], [], []
    if lib == "opencv":
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                hashes.append(_digest(frame))
        finally:
            cap.release()
    else:
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                hashes.append(_digest(frame.to_ndarray(format="bgr24")))
                pts.append(frame.pts)
                if frame.key_frame:
                    keyframes.append(i)
    return {"hashes": hashes, "pts": pts, "keyframes": keyframes}


def _opencv_seeks(path: Path, indices: list[int]) -> list[tuple[str | None, float]]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    out = []
    try:
        for i in indices:
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            ms = (time.perf_counter() - t0) * 1000
            out.append((_digest(frame) if ok else None, ms))
    finally:
        cap.release()
    return out


def _pyav_seeks(path: Path, indices: list[int], pts: list[int]) -> list[tuple[str | None, float]]:
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for i in indices:
            target = pts[i]
            t0 = time.perf_counter()
            container.seek(target, stream=stream, backward=True)
            digest = None
            for frame in container.decode(stream):
                if frame.pts == target:
                    digest = _digest(frame.to_ndarray(format="bgr24"))
                    break
                if frame.pts is not None and frame.pts > target:
                    break
            ms = (time.perf_counter() - t0) * 1000
            out.append((digest, ms))
    return out


def seek_frames(path: Path, lib: str, indices: list[int],
                pts: list[int] | None = None) -> list[tuple[str | None, float]]:
    """(sha256 of the frame returned, seek + read ms) for each index, one reader reused."""
    if lib == "opencv":
        return _opencv_seeks(Path(path), indices)
    return _pyav_seeks(Path(path), indices, pts)


def _keyframe_interval(keyframes: list[int]) -> float | None:
    gaps = [b - a for a, b in zip(keyframes, keyframes[1:])]
    return statistics.mean(gaps) if gaps else None


def measure(converted: Path, source: Path, seeks: int = 100, seed: int = 0) -> dict:
    files = {"converted": Path(converted), "source": Path(source)}
    seq = {(name, lib): sequential_hashes(path, lib) for name, path in files.items() for lib in LIBS}
    frames = min(len(s["hashes"]) for s in seq.values())
    rng = random.Random(seed)
    indices = [rng.randrange(frames) for _ in range(seeks)]
    result = {"seeks": seeks, "seed": seed, "indices": indices, "files": {}, "ratio": {}}
    for name, path in files.items():
        result["files"][name] = {"path": str(path)}
        for lib in LIBS:
            s = seq[(name, lib)]
            where = {h: k for k, h in reversed(list(enumerate(s["hashes"])))}
            got = seek_frames(path, lib, indices, s["pts"])
            ms = [t for _, t in got]
            exact = sum(1 for i, (h, _) in zip(indices, got) if h == s["hashes"][i])
            offsets = [where[h] - i if h in where else None
                       for i, (h, _) in zip(indices, got) if h != s["hashes"][i]]
            entry = {"frames": len(s["hashes"]), "exact": exact, "total": len(indices),
                     "mean_ms": statistics.mean(ms) if ms else 0.0,
                     "median_ms": statistics.median(ms) if ms else 0.0,
                     "offsets": offsets}
            if lib == "pyav":
                entry["keyframes"] = len(s["keyframes"])
                entry["keyframe_interval"] = _keyframe_interval(s["keyframes"])
            result["files"][name][lib] = entry
    for lib in LIBS:
        src = result["files"]["source"][lib]["mean_ms"]
        conv = result["files"]["converted"][lib]["mean_ms"]
        result["ratio"][lib] = conv / src if src else float("inf")
    exact = all(result["files"][n][lib]["exact"] == len(indices) for n in files for lib in LIBS)
    result["passed"] = exact and all(r <= MAX_RATIO for r in result["ratio"].values())
    return result


def _table(result: dict) -> str:
    lines = ["file       lib     frames  exact     mean ms  median ms"]
    for name in ("converted", "source"):
        for lib in LIBS:
            e = result["files"][name][lib]
            lines.append(f"{name:<10} {lib:<7} {e['frames']:>6}  {e['exact']:>3}/{e['total']:<4}"
                         f" {e['mean_ms']:>8.2f}  {e['median_ms']:>9.2f}")
    for lib in LIBS:
        lines.append(f"ratio {lib} converted/source mean: {result['ratio'][lib]:.3f}")
    lines.append("PASS" if result["passed"] else "FAIL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m storage.randaccess", description=__doc__.splitlines()[0])
    parser.add_argument("converted", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seeks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = measure(args.converted, args.source, args.seeks, args.seed)
    print(_table(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
