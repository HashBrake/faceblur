"""Assemble the finished dataset in the bucket, without downloading anything.

    python cloud/deliver.py SOURCE_FOLDER [--apply]

The delivered tree, at the root of its own bucket (`--to`, default the public
bucket `faceblur-blurred`, every object public-read), is the source folder
exactly, file for file
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

With `--patched` (owner, 2026-10-16), a held back copy that the patch pass
(`cloud/patch_run.py`) finished is delivered from `patched/` instead, and its
patch record goes to `faceblur-records/` as `<video>.patch.json`, beside the
record that held it back. Without the flag nothing changes.

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


def pick(rel: str, out: dict, patched: dict, suffix: str,
         use_patched: bool = False) -> dict | None:
    """Where one source video's delivered copy comes from, and its records.

    Paths are relative to the work bucket. The delivered copy is the one
    under `out/`, beside the source's path or in `quarantine/` there, and
    only once its record stands beside it: the copy is uploaded first, so a
    record means the copy is whole. With `use_patched`, a held back copy that
    the patch pass (`cloud/patch_run.py`) has finished is delivered patched
    instead, from `patched/` at the same path without `quarantine/`, and its
    patch record goes to the records as well as the one that held it back.
    A patch whose record is missing never finished and is not used.

    None when the video has no finished copy at all.
    """
    p = PurePosixPath(rel)
    name = output_path(Path(p.name), Path("."), suffix).name
    for where in (p.parent / name, p.parent / "quarantine" / name):
        w = str(where)
        if w not in out or f"{w}.json" not in out:
            continue
        held = "quarantine" in where.parts
        records = [(f"out/{w}.json", f"{rel}.json")]
        fixed = str(p.parent / name)
        if held and use_patched and fixed in patched and f"{fixed}.json" in patched:
            return {"video": f"patched/{fixed}", "size": patched[fixed],
                    "held": True, "patched": True,
                    "records": records + [(f"patched/{fixed}.json", f"{rel}.patch.json")]}
        return {"video": f"out/{w}", "size": out[w], "held": held, "patched": False,
                "records": records}
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--to", default="faceblur-blurred",
                    help="the bucket the delivered tree goes to; it holds nothing else")
    ap.add_argument("--acl", default="public-read",
                    help="ACL for every delivered object (the owner chose a public link)")
    ap.add_argument("--patched", action="store_true",
                    help="deliver a held back copy from patched/ where the patch pass "
                         "(cloud/patch_run.py) finished it")
    args = ap.parse_args(argv)
    load_env()
    b = f"vs:{os.environ['BUCKET']}"
    suffix = Settings().suffix

    local = {p.relative_to(args.source).as_posix(): p.stat().st_size
             for p in args.source.rglob("*") if p.is_file()}
    uploaded, out = listing(f"{b}/in"), listing(f"{b}/out")
    patched = listing(f"{b}/patched") if args.patched else {}

    copies, records, stubs, missing, audio_kept = [], [], [], [], []
    for rel, size in sorted(local.items()):
        p = PurePosixPath(rel)
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        if size < EMPTY:
            stubs.append(rel)
            continue
        got = pick(rel, out, patched, suffix, args.patched)
        if got is None:
            missing.append(rel)
            continue
        copies.append((got, rel))
        records.extend(got["records"])

    others = [rel for rel in local
              if PurePosixPath(rel).suffix.lower() not in VIDEO_EXT
              and not rel.endswith(AUDIO)]
    dropped = [rel for rel in local if rel.endswith(AUDIO)]
    print(f"videos: {len(copies)} blurred copies, {len(stubs)} empty stubs as they are, "
          f"{len(missing)} not delivered")
    print(f"other files: {len(others)} delivered, {len(dropped)} audio files left out")

    # Every record: a held back copy's says what the check still found, and
    # any of them may say the source carried sound, which must not be shipped.
    rows, bad_patches = [], []
    for got, rel in copies:
        r = json.loads(rclone("cat", f"{b}/{got['records'][0][0]}"))
        if r.get("source_audio"):
            audio_kept.append(rel)
        if got["patched"]:
            # The patch writes pictures only, and says so; a record that says
            # otherwise, or that the patch failed, stops the delivery the same way.
            pr = json.loads(rclone("cat", f"{b}/{got['records'][1][0]}"))
            if pr.get("output_audio"):
                audio_kept.append(rel)
            if pr.get("error") or pr.get("status") != "done":
                bad_patches.append(rel)
        if got["held"]:
            row = [rel, r.get("residual_faces", 0), r.get("residual_max_px", 0),
                   r.get("residual_frames", 0)]
            if args.patched:
                row += [pr.get("masked", 0) if got["patched"] else ""]
            rows.append(row)
    print(f"held back but delivered blurred: {len(rows)}"
          + (f", {sum(1 for g, _ in copies if g['patched'])} of them patched"
             if args.patched else ""))
    if audio_kept:
        print(f"STOP: {len(audio_kept)} copies carry the source's audio track, e.g. {audio_kept[:3]}")
        return 2
    if bad_patches:
        print(f"STOP: {len(bad_patches)} patched copies have a record that says the patch "
              f"failed, e.g. {bad_patches[:3]}")
        return 2
    if not args.apply:
        print("dry run; --apply to do it")
        return 0

    d = f"vs:{args.to}"
    acl = ["--s3-acl", args.acl] if args.acl else []
    rclone("mkdir", d)

    for got, rel in copies:
        rclone("copyto", f"{b}/{got['video']}", f"{d}/{rel}", *acl)
    for src, rel in records:
        rclone("copyto", f"{b}/{src}", f"{b}/faceblur-records/{rel}")
    for rel in stubs:
        rclone("copyto", str(args.source / rel), f"{d}/{rel}", *acl)
    filters = "\n".join(["- *.mp4", "- *.MP4", *[f"- *{a}" for a in AUDIO], "+ **"])
    rclone("copy", str(args.source), d, "--filter-from", "-", "--transfers", "8", *acl,
           stdin=filters)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["video", "faces_left", "largest_px", "frames_with_a_face"]
               + (["patched_faces_masked"] if args.patched else []))
    w.writerows(rows)
    rclone("rcat", f"{b}/faceblur-records/held_back.csv", stdin=buf.getvalue())
    rclone("rcat", f"{b}/faceblur-records/not_delivered.txt", stdin="\n".join(missing) + "\n")

    # Nothing raw may be in the delivered tree: every delivered video must be
    # a copy that came out of the pipeline, or a stub.
    delivered = listing(d)
    expected = {rel: got["size"] for got, rel in copies}
    raw = [rel for rel, size in delivered.items()
           if PurePosixPath(rel).suffix.lower() in VIDEO_EXT and size >= EMPTY
           and (expected.get(rel) != size or uploaded.get(rel) == size)]
    leaked_audio = [rel for rel in delivered if rel.endswith(AUDIO)]
    print(f"delivered: {len(delivered)} files; raw videos found: {len(raw)}; "
          f"audio files found: {len(leaked_audio)}")
    return 1 if raw or leaked_audio else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
