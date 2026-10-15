"""Pick the hot-storage quality from a measured size-vs-VMAF curve.

Each sample file is encoded at every quality on the list, and each copy is
scored against its source with VMAF over the whole file. The chosen quality is
the one with the smallest mean size, as a share of the source, among those
whose worst file still scores at least the floor. The encode and measure
stages run separately, so slow encodes are never repeated: an existing copy is
reused. The same run works on any machine; the label says which one it was.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from faceblur.video import VideoError
from storage import checks
from storage.encode import CODECS, ENCODERS, GOP, encode_file
from storage.probe import probe


@dataclass(frozen=True)
class Point:
    file: str
    codec: str
    quality: int
    encoder: str | None
    size_bytes: int
    ratio: float
    vmaf: float
    frames_ok: bool
    duration_ok: bool
    encode_seconds: float | None
    encode_fps: float | None


@dataclass(frozen=True)
class Summary:
    codec: str
    quality: int
    files: int
    mean_ratio: float
    min_ratio: float
    max_ratio: float
    min_vmaf: float
    mean_vmaf: float
    mean_fps: float | None
    all_ok: bool


@dataclass(frozen=True)
class Choice:
    quality: int
    mean_ratio: float
    min_vmaf: float
    meets_floor: bool
    meets_target: bool


def summarise(points: list[Point]) -> dict[tuple[str, int], Summary]:
    """One summary per (codec, quality) over every file measured at it."""
    groups: dict[tuple[str, int], list[Point]] = {}
    for p in points:
        groups.setdefault((p.codec, p.quality), []).append(p)
    out = {}
    for (codec, quality), group in sorted(groups.items()):
        ratios = [p.ratio for p in group]
        scores = [p.vmaf for p in group]
        fps = [p.encode_fps for p in group if p.encode_fps is not None]
        out[(codec, quality)] = Summary(
            codec=codec, quality=quality, files=len(group),
            mean_ratio=sum(ratios) / len(ratios), min_ratio=min(ratios),
            max_ratio=max(ratios), min_vmaf=min(scores),
            mean_vmaf=sum(scores) / len(scores),
            mean_fps=sum(fps) / len(fps) if fps else None,
            all_ok=all(p.frames_ok and p.duration_ok for p in group))
    return out


def choose(summaries: list[Summary], floor: float = 95.0,
           target: float = 0.70) -> Choice | None:
    """The smallest mean size whose worst file reaches the floor, else the best achievable."""
    if not summaries:
        return None
    ok = [s for s in summaries if s.all_ok]
    passing = [s for s in ok if s.min_vmaf >= floor]
    if passing:
        best = min(passing, key=lambda s: (s.mean_ratio, s.quality))
    else:
        best = max(ok or summaries, key=lambda s: (s.min_vmaf, -s.quality))
    return Choice(quality=best.quality, mean_ratio=best.mean_ratio, min_vmaf=best.min_vmaf,
                  meets_floor=bool(passing), meets_target=best.mean_ratio <= target)


def encode_name(src: Path, codec: str, quality: int) -> str:
    return f"{src.stem}.{codec}.q{quality}.mp4"


def plan(inputs: list[Path], codecs: list[str], qualities: dict[str, list[int]],
         av1_files: int | None) -> list[tuple[Path, str, int]]:
    """Every (input, codec, quality) to encode. AV1 runs on the smallest files only."""
    jobs = []
    for codec in codecs:
        files = inputs
        if codec == "av1" and av1_files is not None:
            files = sorted(sorted(inputs, key=lambda p: p.stat().st_size)[:av1_files])
        jobs += [(src, codec, q) for src in files for q in qualities[codec]]
    return jobs


def encode_stage(jobs, workdir: Path, encoder: str, gop: int, log=print) -> None:
    """Encode every job not already in `workdir`, with a sidecar of its speed and encoder."""
    workdir.mkdir(parents=True, exist_ok=True)
    for src, codec, quality in jobs:
        dst = workdir / encode_name(src, codec, quality)
        if dst.exists():
            continue
        result = encode_file(src, dst, codec=codec, encoder=encoder, quality=quality, gop=gop)
        dst.with_suffix(".json").write_text(json.dumps(
            {"seconds": result.seconds, "encoder": result.encoder,
             "size_bytes": result.size_bytes}))
        ratio = result.size_bytes / src.stat().st_size
        log(f"{dst.name}: {result.seconds:.1f} s, {ratio:.2f} of the source")


def measure_stage(jobs, workdir: Path, threads: int, log=print) -> list[Point]:
    """Check and score every job whose encode exists. Missing encodes are skipped."""
    points = []
    for src, codec, quality in jobs:
        dst = workdir / encode_name(src, codec, quality)
        if not dst.exists():
            continue
        a, b = probe(src), probe(dst)
        sidecar = dst.with_suffix(".json")
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        seconds = meta.get("seconds")
        score = checks.vmaf(src, dst, windows=1, window_s=1e9, threads=threads).mean
        point = Point(file=src.name, codec=codec, quality=quality,
                      encoder=meta.get("encoder"), size_bytes=b.size_bytes,
                      ratio=b.size_bytes / a.size_bytes, vmaf=score,
                      frames_ok=checks.frame_count(a, b).passed,
                      duration_ok=checks.duration(a, b).passed,
                      encode_seconds=seconds,
                      encode_fps=a.frames / seconds if seconds else None)
        log(f"{dst.name}: VMAF {score:.2f}, {point.ratio:.2f} of the source")
        points.append(point)
    return points


def _flag(encoder: str | None) -> str:
    return "cq" if encoder == "nvenc" else "crf"


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def write_markdown(path: Path, data: dict) -> None:
    """The curve as plain Markdown: inputs, one table per codec, the chosen quality."""
    points = [Point(**p) for p in data["points"]]
    summaries = summarise(points)
    codecs = sorted({p.codec for p in points}, key=CODECS.index)
    encoders = {c: sorted({p.encoder or "unknown" for p in points if p.codec == c})
                for c in codecs}
    lines = [f"# Hot storage size-vs-VMAF curve: {data['label']}", "",
             f"- Measured: {data['created']}",
             f"- VMAF floor (worst file): {data['floor']:g}",
             f"- Size target (mean, share of H.264): {data['target']:.2f}"]
    lines += [f"- {c.upper()} encoder: {', '.join(encoders[c])}" for c in codecs]
    lines += ["", "## Inputs", "", "| file | size (MB) | frames | duration (s) |",
              "|---|---|---|---|"]
    lines += [f"| {i['name']} | {i['size_bytes'] / 1e6:.1f} | {i['frames']} | "
              f"{i['duration_s']:.1f} |" for i in data["inputs"]]
    for codec in codecs:
        lines += ["", f"## {codec.upper()}", "",
                  "| quality | files | mean size ratio | min..max ratio | min VMAF | "
                  "mean VMAF | encode fps |", "|---|---|---|---|---|---|---|"]
        for (c, _), s in summaries.items():
            if c != codec:
                continue
            mark = "" if s.all_ok else " (failed a frame or duration check)"
            lines.append(f"| {s.quality}{mark} | {s.files} | {s.mean_ratio:.2f} | "
                         f"{s.min_ratio:.2f}..{s.max_ratio:.2f} | {s.min_vmaf:.1f} | "
                         f"{s.mean_vmaf:.1f} | {_fmt(s.mean_fps, '.1f')} |")
        choice = data["choice"].get(codec)
        if choice is None:
            continue
        flag = _flag(encoders[codec][0])
        met = "met" if choice["meets_target"] else "not met"
        detail = (f"min VMAF {choice['min_vmaf']:.1f}, mean size "
                  f"{choice['mean_ratio']:.2f} of H.264 (target {data['target']:.2f} {met})")
        if choice["meets_floor"]:
            sentence = f"Chosen {codec.upper()} quality: {flag} {choice['quality']} — {detail}"
        else:
            sentence = (f"Chosen {codec.upper()} quality: no quality reaches "
                        f"{data['floor']:g}; best achievable is {flag} {choice['quality']} "
                        f"— {detail}")
        lines += ["", sentence]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m storage.calibrate",
        description="Measure size against VMAF over sample files and pick a quality.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/hot_storage_curve"))
    parser.add_argument("--codecs", default="hevc")
    parser.add_argument("--hevc-qualities", default="20,22,24,26,28,30,32,34")
    parser.add_argument("--av1-qualities", default="24,28,32,36,40")
    parser.add_argument("--av1-files", type=int, default=None)
    parser.add_argument("--encoder", choices=ENCODERS, default="auto")
    parser.add_argument("--stage", choices=("encode", "measure", "all"), default="all")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--floor", type=float, default=95.0)
    parser.add_argument("--target", type=float, default=0.70)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gop", type=int, default=GOP)
    try:
        args = parser.parse_args(argv)
        qualities = {"hevc": _ints(args.hevc_qualities), "av1": _ints(args.av1_qualities)}
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    except ValueError as exc:
        print(f"Bad quality list: {exc}", file=sys.stderr)
        return 2
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    unknown = [c for c in codecs if c not in CODECS]
    if not codecs or unknown:
        print(f"Unknown codec: {', '.join(unknown) or '(none given)'}", file=sys.stderr)
        return 2
    inputs = sorted(args.input_dir.glob("*.mp4")) if args.input_dir.is_dir() else []
    if not inputs:
        print(f"No mp4 files in {args.input_dir}", file=sys.stderr)
        return 2
    workdir = args.workdir or args.input_dir / "calib"
    jobs = plan(inputs, codecs, qualities, args.av1_files)
    try:
        if args.stage in ("encode", "all"):
            encode_stage(jobs, workdir, args.encoder, args.gop)
        if args.stage == "encode":
            return 0
        points = measure_stage(jobs, workdir, args.threads)
        facts = {src: probe(src) for src in inputs}
    except VideoError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summaries = summarise(points)
    choice = {}
    for codec in codecs:
        picked = choose([s for (c, _), s in summaries.items() if c == codec],
                        args.floor, args.target)
        choice[codec] = asdict(picked) if picked else None
    data = {"label": args.label,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "floor": args.floor, "target": args.target,
            "inputs": [{"name": p.name, "size_bytes": f.size_bytes, "frames": f.frames,
                        "duration_s": f.duration_s} for p, f in facts.items()],
            "points": [asdict(p) for p in points], "choice": choice}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_markdown(Path(f"{out}.md"), data)
    for codec, picked in choice.items():
        if picked:
            print(f"{codec}: quality {picked['quality']}, mean size {picked['mean_ratio']:.2f}, "
                  f"min VMAF {picked['min_vmaf']:.2f}, floor met {picked['meets_floor']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
