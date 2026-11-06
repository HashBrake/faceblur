"""The DF2 command line: `python -m faceblur.h5 pack ...` and `python -m faceblur.h5 run ...`.

`pack` builds a DF2 file from camera mp4s, because the partner has not sent a
real one yet and the module needs something to read. Besides the cameras it
writes small deterministic datasets in every other group of the tree, so the
pass-through copy has something to carry and the tests something to compare.

`run` is the privacy module itself. Each camera's packets are remuxed to an
mp4, put through the same file pipeline the desktop app uses (batch.run_video),
and the blurred mp4 is stored back as packets. The input is never modified.
The default output is a release file with no audio and no raw video; with
--spec-layout it is instead a copy of the input with the blurred cameras added
beside the raw ones, the single-file form the partner's spec describes.
Either way it is written under a temporary name and renamed only when every
camera finished, so a failed run leaves no episode.h5 that looks complete.

Beside it `run` writes audit.jsonl (faceblur/audit.py), line by line as the
run goes, so even a run that dies leaves a chain that verifies up to where it
stopped. `verify` checks such a file: the chain, and every line against
docs/audit_schema.json.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import av
import h5py
import numpy as np
from jsonschema import ValidationError

from . import audit, batch, detect, df2, h5io, hands, plates, screens, text, vad
from .classes import UnknownKind
from .classes import parse as parse_mask
from .pipeline import STATUS_DONE, STATUS_FAILED, AuditRecord
from .settings import Settings

FIXTURE_VERSION = "fixture-2026-11"
AUDIT_NAME = "audit.jsonl"
OUTPUT_NAME = "episode.h5"
# Written into the container image at build time; the env var wins over it.
IMAGE_DIGEST_FILE = Path("/etc/faceblur_image_digest")
# The hand pass on every 2nd frame in `run` and `selfcheck`: the pilot
# profiled it at 78 ms a frame and the zone has memory (zone_memory) over the
# frame between. The desktop CLI keeps Settings' 1.
H5_ZONE_STRIDE = 2
# `run` and `selfcheck` also turn on the head backstop (faceblur/heads.py);
# the desktop default stays off.


def _dummy_passthrough(f: h5py.File, frames: int) -> None:
    """One small dataset in each pass-through group, the same every time."""
    for i, path in enumerate(df2.PASSTHROUGH_GROUPS):
        group = f.require_group(path)
        if path.startswith(df2.META_GROUP):
            data = group.create_dataset("text", data=np.bytes_(f"fixture text for {path}"))
        elif "/imu/" in path or "/emg/" in path:
            # Wearable sensors sample faster than the cameras.
            values = np.arange(frames * 4 * 6, dtype=np.float32).reshape(frames * 4, 6) + i
            data = group.create_dataset("samples", data=values)
        else:
            values = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6) + i
            data = group.create_dataset("values", data=values)
        data.attrs["source"] = "faceblur.h5 pack"


class RunFailed(Exception):
    """A run that cannot finish. `code` is the exit status the CLI returns."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def pack(mp4s: list[Path], cameras: list[str], audio: Path | None, out: Path) -> list[str]:
    """Write a DF2 file. Returns the groups written, for the report.

    Written to `<out>.part` and renamed when complete, so a file that cannot
    be read leaves nothing behind. Raises RunFailed (code 1) for a macOS
    `._` resource fork, refused before it is opened, and for a file the
    demuxer cannot read.
    """
    for mp4 in mp4s:
        if Path(mp4).name.startswith("._"):
            raise RunFailed(f"{mp4}: refused: a macOS resource fork, not a video", 1)
    out = Path(out)
    part = out.with_name(out.name + ".part")
    written = []
    try:
        with h5py.File(part, "w") as f:
            f.attrs["df2_version"] = FIXTURE_VERSION
            first_timestamps = None
            for mp4, name in zip(mp4s, cameras):
                path = df2.camera_path(name)
                _read_or_fail(mp4, h5io.write_video_group, f.require_group(path), mp4)
                if first_timestamps is None:
                    first_timestamps = f[path][df2.TIMESTAMPS_NS][:]
                written.append(path)
            if audio is not None:
                _read_or_fail(audio, h5io.write_audio_group, f, audio)
                written.append(df2.AUDIO_GROUP)
            timeline = f.require_group(df2.TIMELINE_GROUP)
            timeline.create_dataset(df2.TIMELINE_TIMESTAMPS_NS, data=first_timestamps)
            written.append(df2.TIMELINE_GROUP)
            _dummy_passthrough(f, len(first_timestamps))
            written += list(df2.PASSTHROUGH_GROUPS)
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)
    return written


def _read_or_fail(source: Path, write, *args) -> None:
    """`write(*args)`, with a file the demuxer cannot read as one RunFailed line."""
    try:
        write(*args)
    except (av.error.FFmpegError, OSError) as exc:
        message = getattr(exc, "strerror", None) or str(exc)
        raise RunFailed(f"{source}: cannot read: {message}", 1) from None


# Every attr read_video_group needs to remux a camera group.
_REQUIRED_ATTRS = (df2.ATTR_CODEC, df2.ATTR_WIDTH, df2.ATTR_HEIGHT,
                   df2.ATTR_TIME_BASE_NUM, df2.ATTR_TIME_BASE_DEN, df2.ATTR_EXTRADATA)


def _check_camera(f: h5py.File, camera: str) -> None:
    """RunFailed (code 2) for a camera group that lacks an attr the remux needs."""
    attrs = f[df2.camera_path(camera)].attrs
    for attr in _REQUIRED_ATTRS:
        if attr not in attrs:
            raise RunFailed(f"{camera}: camera group has no {attr} attr", 2)


# What write_video_group makes from the blurred mp4. Everything else in an
# input camera group (calibration, whatever the partner adds) is copied over.
_REWRITTEN = (df2.VIDEO, df2.PTS, df2.DTS, df2.KEYFRAME, df2.TIMESTAMPS_NS)
_REWRITTEN_ATTRS = (df2.ATTR_CODEC, df2.ATTR_WIDTH, df2.ATTR_HEIGHT, df2.ATTR_FPS,
                    df2.ATTR_TIME_BASE_NUM, df2.ATTR_TIME_BASE_DEN, df2.ATTR_PIX_FMT,
                    df2.ATTR_EXTRADATA, df2.ATTR_FRAME_COUNT)


def _store_camera(src: h5py.Group, dst: h5py.Group, blurred: Path) -> None:
    """The blurred mp4 as packets in `dst`, with the rest of `src` beside them."""
    h5io.write_video_group(dst, blurred)
    for name in src:
        if name in _REWRITTEN:
            continue
        # write_video_group puts placeholder calibration in; the input's wins.
        if name in dst:
            del dst[name]
        src.copy(src[name], dst, name=name)
    for key, value in src.attrs.items():
        if key not in _REWRITTEN_ATTRS:
            dst.attrs[key] = value


def _blur_camera(f: h5py.File, camera: str, work: Path, settings: Settings,
                 submit, workers: int) -> dict:
    """Run one camera through the file pipeline. Returns its result dict."""
    raw = work / f"{camera}.mp4"
    blurred = work / f"{camera}_blurred.mp4"
    group = f[df2.camera_path(camera)]
    h5io.read_video_group(group, raw)
    per_frame: list = []
    zones: list = []
    started = time.time()
    try:
        record = batch.run_video(raw, blurred, settings, submit, workers=workers,
                                 write_sidecar=False, per_frame_out=per_frame,
                                 zones_out=zones)
    except batch.WorkerLost as exc:
        record = AuditRecord(source=raw.name, output=blurred.name, status=STATUS_FAILED,
                             error=str(exc))
    if record.status != STATUS_DONE:
        raise RunFailed(f"{camera}: {record.status}: {record.error}", 1)
    return {"camera": camera, "record": record, "per_frame": per_frame, "zones": zones,
            "shape": (int(group.attrs[df2.ATTR_HEIGHT]), int(group.attrs[df2.ATTR_WIDTH])),
            "frames": len(per_frame), "seconds": round(time.time() - started, 2),
            "timestamps_ns": group[df2.TIMESTAMPS_NS][:], "mp4": blurred}


def _write_output(input_path: Path, part: Path, results: list[dict], spec_layout: bool) -> None:
    """Write `part`: the release file, or the input copy with a sibling group."""
    if spec_layout:
        # The raw cameras and the audio stay, by the spec's design.
        shutil.copyfile(input_path, part)
    with h5py.File(input_path, "r") as src, h5py.File(part, "r+" if spec_layout else "w") as out:
        if spec_layout:
            parent = df2.RELEASE_IMAGE_GROUP
        else:
            parent = df2.IMAGE_GROUP
            h5io.copy_passthrough(src, out)
        for result in results:
            camera = result["camera"]
            _store_camera(src[df2.camera_path(camera)],
                          out.require_group(f"{parent}/{camera}"), result["mp4"])
        out.attrs["faceblur_release"] = True
        out.attrs["faceblur_version"] = AuditRecord.faceblur_version
        out.attrs["faceblur_layout"] = "spec" if spec_layout else "two-file"


def _model_files(settings: Settings, audio: bool) -> dict[str, Path]:
    """Every model file this run will load, by the name its record uses.

    The header is written before any detector is built, so this follows the
    same conditions the detectors are built under (DetectorBank, detect_job,
    the output check) instead of asking them.
    """
    files: dict[str, Path] = {}
    if settings.wants("face"):
        cf = settings.engine in ("centerface", "both") or (
            settings.engine == "yunet" and settings.verify)
        if settings.engine in ("yunet", "both"):
            files["yunet"] = detect.YUNET_DYNAMIC_MODEL
        if cf:
            files["centerface"] = detect.CENTERFACE_MODEL
        if cf and settings.engine == "yunet" and settings.confirm_on_crops \
                and settings.third_opinion:
            files["ultraface"] = detect.ULTRAFACE_MODEL
        if settings.check_output and settings.hand_rule:
            # The output check's hand rule, under the hand models' own names.
            files["palm"] = hands.PALM_MODEL
            files["hand_landmark"] = hands.LANDMARK_MODEL
    if settings.wants("screen"):
        files["yolox"] = screens.SCREEN_MODEL
    if settings.wants("text"):
        files["ppocr"] = text.TEXT_MODEL
    if settings.wants("plate"):
        files["lpd_yunet"] = plates.PLATE_MODEL
    if settings.zone:
        files["zone_palm"] = hands.PALM_MODEL
        files["zone_landmark"] = hands.LANDMARK_MODEL
    if audio:
        files["silero_vad"] = vad.VAD_MODEL
    return files


def _image_digest() -> str | None:
    """The container image the run is in, when the image says so."""
    digest = os.environ.get("FACEBLUR_IMAGE_DIGEST")
    if digest:
        return digest
    if IMAGE_DIGEST_FILE.is_file():
        return IMAGE_DIGEST_FILE.read_text(encoding="utf-8").strip()
    return None


def _json_safe(value):
    """`value` with NaN and infinities as None: the audit's JSON has no form for them."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _zone_kept(dets, zone, shape: tuple[int, int], settings: Settings) -> list[bool]:
    """Per detection: did the handled zone leave it clear?

    The output check's test (`verify.covered_by_zone` at `zone_cover`) on the
    polygons `redact` cuts out for that kind: faces (and screens, with
    `zone_screen_needs_hand`) by the hands (or the live reach), text, plates
    and otherwise screens by the live reach and the remembered polygons.
    One crop per polygon group per frame.
    """
    if not zone or not dets:
        return [False] * len(dets)
    from .verify import covered_by_zone, zone_crop
    from .zone import reach_protects

    protects = reach_protects(settings)
    crops: dict[bool, object] = {}
    kept = []
    for d in dets:
        group = d.kind in protects
        if group not in crops:
            crops[group] = zone_crop(shape, zone.for_kind(d.kind, settings.zone_face_needs_hand,
                                                          protects))
        kept.append(covered_by_zone(d, crops[group]) >= settings.zone_cover)
    return kept


def _audit_camera(writer: audit.AuditWriter, result: dict, settings: Settings) -> None:
    """One frame line per frame of a finished camera, then its summary line.

    `counts`, `boxes` and `quads` are the masks painted; a detection the
    handled zone left clear goes under `zone_kept` instead.
    """
    camera = result["camera"]
    zones = result.get("zones") or []
    for i, dets in enumerate(result["per_frame"]):
        counts = {c: 0 for c in audit.CLASSES}
        boxes: dict[str, list] = {}
        quads: dict[str, list] = {}
        kept: dict[str, list] = {}
        zone = zones[i] if i < len(zones) else None
        for d, in_zone in zip(dets, _zone_kept(dets, zone, result["shape"], settings)):
            if in_zone:
                kept.setdefault(d.kind, []).append([d.x, d.y, d.w, d.h])
                continue
            counts[d.kind] += 1
            boxes.setdefault(d.kind, []).append([d.x, d.y, d.w, d.h])
            if d.quad is not None and d.kind in ("text", "plate"):
                quads.setdefault(d.kind, []).append([list(p) for p in d.quad])
        writer.frame(camera, i, int(result["timestamps_ns"][i]), counts, boxes, quads or None,
                     kept or None)
    writer.camera_done(camera, result["frames"], result["seconds"],
                       _json_safe(result["record"].to_dict()))


def run(input_path: Path, out_dir: Path, settings: Settings, workers: int = 2,
        spec_layout: bool = False) -> list[dict]:
    """De-identify every camera of `input_path` into `out_dir/episode.h5`.

    Returns one dict per camera: camera, record, per_frame, zones, shape, frames, seconds,
    timestamps_ns. Raises RunFailed when there is no camera or the input holds
    a soft or external link (code 2, before audit.jsonl is opened) or a
    camera did not finish (code 1); then no episode.h5 is written, and
    `out_dir/audit.jsonl` (written whenever a camera was found) ends with a
    failed footer.
    """
    input_path, out_dir = Path(input_path), Path(out_dir)
    with h5py.File(input_path, "r") as f:
        cameras = h5io.camera_groups(f)
        if not cameras:
            raise RunFailed(f"{input_path} has no camera group under {df2.IMAGE_GROUP}.", 2)
        # The pass-through copy would follow a soft or external link and put
        # what it points at into the release.
        links = h5io.external_links(f)
        if links:
            raise RunFailed(f"{input_path} holds links the release would follow: "
                            f"{', '.join(links[:5])}", 2)
        # Before any audit line or detector: a broken group fails in one line.
        for camera in cameras:
            _check_camera(f, camera)
        pcm = vad.read_pcm(f)
    work = out_dir / ".work"
    part = out_dir / "episode.h5.part"
    final = out_dir / OUTPUT_NAME
    work.mkdir(parents=True, exist_ok=True)
    writer = audit.AuditWriter(out_dir / AUDIT_NAME)
    pool = None
    try:
        models = {name: detect.model_sha256(path)
                  for name, path in _model_files(settings, pcm is not None).items()}
        writer.header(audit.sha256_file(input_path), input_path.name, cameras, models,
                      settings.to_dict(), _image_digest(), AuditRecord.faceblur_version)
        if pcm is not None:
            # Audio is quick, and the release drops it (D-B): the record of
            # what was spoken goes in before the slow part of the run.
            for start, end in vad.speech_segments(pcm[0], pcm[1], settings):
                writer.speech(start, end, settings.vad_pad_ms)
            pcm = None
        if workers == 1:
            submit = batch.serial_submit
        else:
            # Built the way cli.run_parallel builds it, one pool for the whole run.
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(workers, initializer=batch.init_pool_worker,
                                initargs=(workers,))
            submit = batch.PoolSubmit(pool)
        results = []
        with h5py.File(input_path, "r") as f:
            for camera in cameras:
                result = _blur_camera(f, camera, work, settings, submit, workers)
                _audit_camera(writer, result, settings)
                results.append(result)
        _write_output(input_path, part, results, spec_layout)
        part.replace(final)
        writer.footer(audit.sha256_file(final), final.name, "done")
    except BaseException as exc:
        # The footer says the run did not finish; there is no output to hash.
        writer.footer(audit.GENESIS, "", "failed", str(exc) or type(exc).__name__)
        raise
    finally:
        writer.close()
        if pool is not None:
            pool.terminate()
            pool.join()
        part.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    for result in results:
        del result["mp4"]  # gone with the work folder
    return results


def _describe(result: dict) -> str:
    """One line per camera for the CLI."""
    frames = result["frames"]
    ms = 1000 * result["seconds"] / frames if frames else 0.0
    return (f"{result['camera']}: {frames} frames, {result['seconds']:.1f} s, "
            f"{ms:.0f} ms/frame, masked {100 * result['record'].masked_mean:.2f}% of the picture")


def verify_audit(path: Path) -> tuple[bool, int, str]:
    """The chain, then every line against the schema: (ok, lines, the first problem)."""
    ok, lines, problem = audit.verify(path)
    if not ok:
        return ok, lines, problem
    schema = audit.load_schema()
    for line in audit.read(path):
        try:
            audit.validate_line(line, schema)
        except ValidationError as exc:
            return False, lines, f"schema at seq {line.get('seq')}: {exc.message}"
    return True, lines, ""


def _side_devices(device: str) -> dict[str, str]:
    """--device for the plate, zone and hand models too.

    Those follow their own settings (default "auto"), so without this a
    `--device cpu` run still put the plate model on the GPU.
    """
    return {"plate_device": device, "zone_device": device, "hand_device": device}


def selfcheck(device: str) -> int:
    """Every masking model loaded and run once on a flat frame.

    The container's HEALTHCHECK: a model that does not load, or a provider
    that fails on its first inference, shows here and not an hour into a run.
    A flat frame finds nothing, which is fine: the point is that it runs.
    """
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    try:
        settings = Settings(mask=("face", "text", "plate", "screen"), device=device,
                            zone_stride=H5_ZONE_STRIDE, head_backstop=True,
                            **_side_devices(device))
        for detector in (detect.DetectorBank(settings), screens.ScreenDetector(settings),
                         text.TextDetector(settings), plates.PlateDetector(settings)):
            detector.detect(frame)
            for name, provider in detector.compute.items():
                print(f"{name}: {provider}")
    except Exception as exc:  # any failure is the answer the check exists to give
        print(f"selfcheck failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m faceblur.h5", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="build a DF2 file from camera mp4s")
    p.add_argument("mp4", nargs="+", type=Path)
    p.add_argument("--cameras", help="comma-separated camera names, one per mp4")
    p.add_argument("--audio", type=Path, help="a sound file for the audio group")
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="write a de-identified episode.h5 from a DF2 file")
    r.add_argument("input", type=Path)
    r.add_argument("--out", type=Path, required=True, help="folder for episode.h5")
    r.add_argument("--spec-layout", action="store_true",
                   help="copy the input and add the blurred cameras beside the raw ones")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--mask", default="face,text,plate,screen")
    r.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    r.add_argument("--check-output", action="store_true")
    v = sub.add_parser("verify", help="check an audit.jsonl: hash chain and schema")
    v.add_argument("audit", type=Path)
    c = sub.add_parser("selfcheck", help="load every masking model and run it on one frame")
    c.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args = ap.parse_args(argv)

    if args.command == "selfcheck":
        return selfcheck(args.device)

    if args.command == "verify":
        ok, lines, problem = verify_audit(args.audit)
        print(f"ok: {lines} lines" if ok else f"problem: {problem} ({lines} lines verified)")
        return 0 if ok else 1

    if args.command == "run":
        try:
            mask = parse_mask(args.mask)
        except UnknownKind as exc:
            ap.error(str(exc))
        settings = Settings(mask=mask, device=args.device, check_output=args.check_output,
                            zone_stride=H5_ZONE_STRIDE, head_backstop=True,
                            **_side_devices(args.device))
        try:
            results = run(args.input, args.out, settings, args.workers, args.spec_layout)
        except RunFailed as exc:
            print(exc, file=sys.stderr)
            return exc.code
        for result in results:
            print(_describe(result))
        print(args.out / "episode.h5")
        return 0

    cameras = args.cameras.split(",") if args.cameras else list(df2.PACK_ORDER)
    if len(cameras) < len(args.mp4):
        ap.error(f"{len(args.mp4)} mp4s but only {len(cameras)} camera names")
    try:
        written = pack(args.mp4, cameras, args.audio, args.out)
    except RunFailed as exc:
        print(exc, file=sys.stderr)
        return exc.code
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
