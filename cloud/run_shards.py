"""Run the pipeline on every card of a multi GPU machine at once.

The pipeline itself uses one card. This splits the videos under one folder
into one share per card, balanced by file size, and runs `cli.py` once per
card with `CUDA_VISIBLE_DEVICES` set to that card. Every share writes into the
same output folder with the folder structure kept, so the result looks as
though one run had done the lot.

    python cloud/run_shards.py /data/in /data/out --gpus 8 -- --workers 6 --quarantine

Everything after `--` goes to `cli.py` unchanged. Run it again after a crash
and only the videos with no finished copy are shared out: a delivered copy,
or a quarantined copy with its record, counts as finished.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from cli import destination, find_videos  # noqa: E402
from faceblur.pipeline import sidecar_path  # noqa: E402
from faceblur.settings import Settings  # noqa: E402


def finished(dst: Path) -> bool:
    """A delivered copy, or a held back one whose record was written after it."""
    if dst.exists():
        return True
    held = dst.parent / "quarantine" / dst.name
    return held.is_file() and sidecar_path(held).is_file()


def share_out(files: list[Path], n: int) -> list[list[Path]]:
    """Largest first, each to the share with the least so far."""
    shares: list[list[Path]] = [[] for _ in range(n)]
    loads = [0] * n
    for f in sorted(files, key=lambda p: p.stat().st_size, reverse=True):
        k = loads.index(min(loads))
        shares[k].append(f)
        loads[k] += f.stat().st_size
    return shares


def link_share(files: list[Path], root: Path, folder: Path) -> None:
    """The share as a folder of links at the same relative paths, so that
    `cli.py --recursive` mirrors the original structure in the output."""
    shutil.rmtree(folder, ignore_errors=True)
    for f in files:
        link = folder / f.relative_to(root)
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(f, link)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, passthrough = argv[:k], argv[k + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--shards", type=Path, default=None,
                        help="where the per card folders of links go (default: OUTPUT/_shards)")
    args = parser.parse_args(argv)

    root = args.input.resolve()
    out = args.output.resolve()
    shards_dir = (args.shards or out / "_shards").resolve()
    logs = out / "_logs"
    logs.mkdir(parents=True, exist_ok=True)

    suffix = Settings().suffix
    files = [f for f in find_videos(root, True) if out not in f.parents]
    todo = [f for f in files if not finished(destination(f, root, out, True, suffix))]
    print(f"{len(files)} videos, {len(files) - len(todo)} already finished, "
          f"{len(todo)} to do on {args.gpus} cards", flush=True)
    if not todo:
        return 0

    procs = []
    for k, share in enumerate(share_out(todo, args.gpus)):
        if not share:
            continue
        folder = shards_dir / f"gpu{k}"
        link_share(share, root, folder)
        cmd = [sys.executable, str(HERE.parent / "cli.py"), str(folder), "-o", str(out),
               "--recursive", "--no-progress",
               "--report", str(logs / f"gpu{k}_report.json"), *passthrough]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        log = open(logs / f"gpu{k}.log", "a", encoding="utf-8")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {len(share)} videos, "
                  f"{sum(f.stat().st_size for f in share) / 1e9:.1f} GB\n")
        log.flush()
        procs.append((k, len(share), subprocess.Popen(cmd, env=env, stdout=log,
                                                      stderr=subprocess.STDOUT), log))
        print(f"card {k}: {len(share)} videos", flush=True)

    started = time.time()
    while any(p.poll() is None for _, _, p, _ in procs):
        time.sleep(60)
        done = sum(finished(destination(f, root, out, True, suffix)) for f in todo)
        hours = (time.time() - started) / 3600
        running = sum(p.poll() is None for _, _, p, _ in procs)
        print(f"{time.strftime('%H:%M')} {done}/{len(todo)} finished, "
              f"{running} cards running, {hours:.2f} h", flush=True)

    failed = 0
    for k, n, p, log in procs:
        log.close()
        print(f"card {k}: exit {p.returncode}", flush=True)
        failed += p.returncode != 0
    left = [f for f in todo if not finished(destination(f, root, out, True, suffix))]
    print(f"{len(todo) - len(left)} of {len(todo)} finished, {len(left)} not", flush=True)
    for f in left:
        print(f"  not finished: {f.relative_to(root)}", flush=True)
    return 1 if failed or left else 0


if __name__ == "__main__":
    sys.exit(main())
