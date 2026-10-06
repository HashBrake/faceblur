"""Assemble the finished dataset in the bucket, without downloading anything.

    python cloud/deliver.py SOURCE_FOLDER [--apply]

The delivered tree, `deliver/`, is the source folder exactly, file for file
and name for name, with three changes decided by the owner on 2026-10-06:

- every video is its blurred copy under the original name. A copy the gate
  held back is delivered too: it is blurred, and the list of the frames that
  still show something goes to `faceblur-records/held_back.csv`;
- `*_audio.wav` and `*_audio_pts.csv` are not delivered at all;
- a video with no frames in it (the recorder's 24 byte stubs) goes as it is,
  because there is nothing in it to blur. A real video that has no finished
  copy is never delivered raw: it is listed in `faceblur-records/not_delivered.txt`.

FaceBlur's own records go beside the tree, not into it, under
`faceblur-records/` at the same relative paths.

Videos are copied inside the bucket (an S3 server side copy); the other files
are uploaded from SOURCE_FOLDER. Without `--apply` it only says what it would
do. Configuration as in stream_run.py (S3_* and BUCKET, from cloud/greennode.env).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from faceblur.pipeline import output_path  # noqa: E402
from faceblur.settings import VIDEO_EXT, Settings  # noqa: E402

AUDIO = ("_audio.wav", "_audio_pts.csv")
EMPTY = 1024          # bytes; the recorder's stubs are 24


def load_env() -> None:
    env = HERE / "greennode.env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    e = os.environ
    e.update({"RCLONE_CONFIG_VS_TYPE": "s3", "RCLONE_CONFIG_VS_PROVIDER": "Other",
              "RCLONE_CONFIG_VS_ENDPOINT": e["S3_ENDPOINT"],
              "RCLONE_CONFIG_VS_ACCESS_KEY_ID": e["S3_ACCESS_KEY"],
              "RCLONE_CONFIG_VS_SECRET_ACCESS_KEY": e["S3_SECRET_KEY"]})


def rclone(*args: str, stdin: str | None = None) -> str:
    r = subprocess.run(["rclone", *args], input=stdin, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rclone {' '.join(args[:3])}: {r.stderr.strip()[-400:]}")
    return r.stdout


def listing(remote: str) -> dict[str, int]:
    try:
        out = rclone("lsjson", remote, "--recursive", "--files-only")
    except RuntimeError:
        return {}
    return {e["Path"]: e["Size"] for e in json.loads(out or "[]")}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    load_env()
    b = f"vs:{os.environ['BUCKET']}"
    suffix = Settings().suffix

    local = {p.relative_to(args.source).as_posix(): p.stat().st_size
             for p in args.source.rglob("*") if p.is_file()}
    uploaded, out = listing(f"{b}/in"), listing(f"{b}/out")

    copies, records, stubs, missing, audio_kept = [], [], [], [], []
    for rel, size in sorted(local.items()):
        p = PurePosixPath(rel)
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        if size < EMPTY:
            stubs.append(rel)
            continue
        name = output_path(Path(p.name), Path("."), suffix).name
        for where in (p.parent / name, p.parent / "quarantine" / name):
            if str(where) in out and f"{where}.json" in out:
                copies.append((str(where), rel))
                records.append((f"{where}.json", f"{rel}.json"))
                break
        else:
            missing.append(rel)

    others = [rel for rel in local
              if PurePosixPath(rel).suffix.lower() not in VIDEO_EXT
              and not rel.endswith(AUDIO)]
    dropped = [rel for rel in local if rel.endswith(AUDIO)]
    print(f"videos: {len(copies)} blurred copies, {len(stubs)} empty stubs as they are, "
          f"{len(missing)} not delivered")
    print(f"other files: {len(others)} delivered, {len(dropped)} audio files left out")

    # Every record: a held back copy's says what the check still found, and
    # any of them may say the source carried sound, which must not be shipped.
    rows = []
    for src, rel in copies:
        r = json.loads(rclone("cat", f"{b}/out/{src}.json"))
        if r.get("source_audio"):
            audio_kept.append(rel)
        if "quarantine" in PurePosixPath(src).parts:
            rows.append([rel, r.get("residual_faces", 0), r.get("residual_max_px", 0),
                         r.get("residual_frames", 0)])
    print(f"held back but delivered blurred: {len(rows)}")
    if audio_kept:
        print(f"STOP: {len(audio_kept)} copies carry the source's audio track, e.g. {audio_kept[:3]}")
        return 2
    if not args.apply:
        print("dry run; --apply to do it")
        return 0

    for src, rel in copies:
        rclone("copyto", f"{b}/out/{src}", f"{b}/deliver/{rel}")
    for src, rel in records:
        rclone("copyto", f"{b}/out/{src}", f"{b}/faceblur-records/{rel}")
    for rel in stubs:
        rclone("copyto", str(args.source / rel), f"{b}/deliver/{rel}")
    filters = "\n".join(["- *.mp4", "- *.MP4", *[f"- *{a}" for a in AUDIO], "+ **"])
    rclone("copy", str(args.source), f"{b}/deliver", "--filter-from", "-",
           "--transfers", "8", stdin=filters)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["video", "faces_left", "largest_px", "frames_with_a_face"])
    w.writerows(rows)
    rclone("rcat", f"{b}/faceblur-records/held_back.csv", stdin=buf.getvalue())
    rclone("rcat", f"{b}/faceblur-records/not_delivered.txt", stdin="\n".join(missing) + "\n")

    # Nothing raw may be in the delivered tree: every delivered video must be
    # a copy that came out of the pipeline, or a stub.
    delivered = listing(f"{b}/deliver")
    expected = {rel: out[src] for src, rel in copies}
    raw = [rel for rel, size in delivered.items()
           if PurePosixPath(rel).suffix.lower() in VIDEO_EXT and size >= EMPTY
           and (expected.get(rel) != size or uploaded.get(rel) == size)]
    leaked_audio = [rel for rel in delivered if rel.endswith(AUDIO)]
    print(f"delivered: {len(delivered)} files; raw videos found: {len(raw)}; "
          f"audio files found: {len(leaked_audio)}")
    return 1 if raw or leaked_audio else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
