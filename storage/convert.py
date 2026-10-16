"""Convert one stored video to HEVC (or AV1), checked, and only on request.

The default is a dry run: the file is encoded beside the source, every check
runs, the result goes in the manifest, and the copy is deleted. Only with
`--apply`, and only when every check passed, does the copy replace the source
under the same name, by one rename on the same drive, so there is never a
moment without a whole file at that path. Nothing here touches the cloud.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from faceblur.settings import Settings
from faceblur.video import VideoError, part_path
from storage import checks
from storage.checks import CheckReport
from storage.encode import CODECS, ENCODERS, encode_file, resolve_encoder
from storage.manifest import MANIFEST_NAME, ManifestRow, append_row, now_utc, sha256_of
from storage.probe import probe


class SameCodecError(ValueError):
    """The source is already in the target codec."""


def default_quality(codec: str, encoder: str) -> int:
    """The calibrated quality for a codec on an encoder already resolved from `auto`.

    NVENC and libx265 use different scales, so each has its own setting. AV1 is
    not a default and keeps a placeholder.
    """
    if codec == "hevc":
        return Settings().hevc_cq if encoder == "nvenc" else Settings().hevc_crf
    return 24


def temp_path(src: Path, codec: str) -> Path:
    """The converted copy, beside the source so the final rename stays on one drive."""
    return src.parent / f"{src.stem}.{codec}.tmp.mp4"


def convert(src: Path, apply: bool = False, codec: str = "hevc", encoder: str = "auto",
            quality: int | None = None, gop: int | None = None, manifest: Path | None = None,
            keep: bool = False, floor: float = 95.0, windows: int = 3,
            window_s: float = 60.0, log=print) -> tuple[ManifestRow, CheckReport | None]:
    src = Path(src)
    if quality is None:
        encoder = resolve_encoder(codec, encoder)
        quality = default_quality(codec, encoder)
    gop = Settings().gop if gop is None else gop
    before = probe(src)
    if before.codec == codec:
        raise SameCodecError(f"{src} is already {codec}")
    sha_before = sha256_of(src)
    tmp = temp_path(src, codec)
    try:
        result = encode_file(src, tmp, codec=codec, encoder=encoder, quality=quality, gop=gop)
        report = checks.run_all(src, tmp, floor, windows, window_s)
        log(report.describe())
        passed = report.passed
        status = ("applied" if apply else "dry-run") if passed else "failed"
        manifest = Path(manifest) if manifest else src.parent / MANIFEST_NAME
        try:
            row_path = src.resolve().relative_to(manifest.resolve().parent).as_posix()
        except ValueError:
            row_path = str(src)
        row = ManifestRow(
            path=row_path, sha256_before=sha_before,
            sha256_after=sha256_of(tmp) if passed else sha_before,
            size_before=before.size_bytes, size_after=result.size_bytes,
            codec_before=before.codec, codec_after=codec, frames=report.frames,
            duration_s=report.duration_s, vmaf=report.vmaf_mean, vmaf_min=report.vmaf_min,
            checks_passed=passed, status=status, encoder=result.encoder,
            quality=quality, converted_at=now_utc())
        if apply and passed:
            # Rename first: a row says "applied" only for a file that was replaced.
            os.replace(tmp, src)
        append_row(manifest, row)
        ratio = result.size_bytes / before.size_bytes if before.size_bytes else 0.0
        vmaf_text = "n/a" if report.vmaf_mean is None else f"{report.vmaf_mean:.2f}"
        log(f"{status}: {src.name} {before.size_bytes} -> {result.size_bytes} bytes "
            f"({ratio:.2f} of the source), VMAF {vmaf_text}")
        if not (apply and passed) and not keep:
            tmp.unlink(missing_ok=True)
        return row, report
    except BaseException:
        tmp.unlink(missing_ok=True)
        part_path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m storage.convert",
        description="Re-encode a video for hot storage. A dry run unless --apply.")
    parser.add_argument("src", type=Path)
    parser.add_argument("--apply", action="store_true",
                        help="replace the source when every check passes")
    parser.add_argument("--codec", choices=CODECS, default="hevc")
    parser.add_argument("--encoder", choices=ENCODERS, default="auto")
    parser.add_argument("--quality", type=int, default=None)
    parser.add_argument("--gop", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="keep the converted copy")
    parser.add_argument("--floor", type=float, default=95.0)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    try:
        args = parser.parse_args(argv)
        if args.windows < 1:
            parser.error("--windows must be at least 1")
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    if not args.src.is_file():
        print(f"No such file: {args.src}", file=sys.stderr)
        return 2
    try:
        row, _ = convert(args.src, apply=args.apply, codec=args.codec, encoder=args.encoder,
                         quality=args.quality, gop=args.gop, manifest=args.manifest,
                         keep=args.keep, floor=args.floor, windows=args.windows,
                         window_s=args.window_seconds)
    except (SameCodecError, VideoError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if row.checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
