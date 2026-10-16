# Hot storage: calibrated defaults and the converter run on the four samples

- Date: 2026-10-16 (libx265 measurement 06:52 local, converter run 07:04 local, UTC+7)
- Machine: the local PC, NVIDIA GeForce RTX 3070 (driver 595.79), ffmpeg 7.1 (gyan.dev, via imageio-ffmpeg)
- Code: the commit that adds this file (task T06, parent 9370dc0)
- Inputs: the four blurred H.264 samples in `faceblur-hot-storage-data/blurred/` (927.0 MB in all)

## Defaults (faceblur/settings.py)

| setting | value | source |
|---|---|---|
| `hevc_cq` | 28 | NVENC HEVC; min VMAF 96.2 at 0.19 of H.264 on the RTX 3070 (`hot_storage_curve.md`), 96.5 at 0.18 on the RTX 5090 (`hot_storage_curve_5090.md`) |
| `hevc_crf` | 22 | libx265 fallback, measured below |
| `gop` | 60 | the brief: a keyframe at least every 60 frames |

`storage.convert` resolves `--encoder auto` first, then takes `hevc_cq` for NVENC or `hevc_crf` for libx265, so the manifest's `encoder` and `quality` always agree. `--gop` defaults to `Settings().gop`.

## libx265 crf measurement

Command (through the GPU lock), on a copy of 004100 alone (105.7 MB, 984 frames, 32.8 s):

```
python -m storage.calibrate <t06/x265_in> --out <t06/curve_x265> --codecs hevc \
  --hevc-qualities 20,22,24 --encoder software --stage all --workdir <t06/x265_work> \
  --label "libx265 medium, RTX-3070 PC CPU, 004100 only" --threads 4
```

The whole run took 192 s, well inside the 45 min bound.

| crf | size (MB) | size ratio | VMAF | encode fps |
|---|---|---|---|---|
| 20 | 21.8 | 0.206 | 97.89 | 10.9 |
| 22 | 15.9 | 0.150 | 96.48 | 45.7 |
| 24 | 11.8 | 0.112 | 94.54 | 49.9 |

The rule: the crf whose VMAF on 004100 is nearest to NVENC cq 28's on the same file (96.58, `hot_storage_curve.json`) while staying at or above 95. crf 22 is 0.10 away; crf 20 is 1.31 away; crf 24 is below 95. **Chosen: `hevc_crf = 22`.**
The crf 20 encode ran at 10.9 fps against 46-50 fps for the other two. It was the first encode of the run (cold file cache, and the CPU is shared with other sessions; the lock holds the GPU, not the CPU), so that figure is not a property of crf 20.

## Converter run

Driver `faceblur-hot-storage-data/t06/run_convert.py`, run once through the GPU lock. All defaults: `--codec hevc --encoder auto` (resolved to `nvenc`), quality `hevc_cq` 28, gop 60, VMAF floor 95, 3 windows of 60 s (all four files are under 180 s, so VMAF scores each file whole and the min window equals the mean).

1. Dry run, for each file in `blurred/`:
   `python -m storage.convert <blurred/file> --manifest <converted/hot_storage_manifest.parquet>`
2. Copy the four files into `converted/` (emptied first).
3. Apply, for each copy: `python -m storage.convert <converted/file> --apply --manifest <same>`

Every one of the 8 commands exited 0.

### Apply pass

| file | size before (MB) | size after (MB) | ratio | VMAF mean | VMAF min | frames | duration (s) | checks passed | apply wall (s) | dry-run wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 003939 | 207.7 | 40.7 | 0.196 | 96.56 | 96.56 | 2033 | 67.739 | yes | 75.2 | 94.3 |
| 004100 | 105.7 | 20.7 | 0.196 | 96.58 | 96.58 | 984 | 32.779 | yes | 67.2 | 46.6 |
| 004310 | 128.9 | 22.1 | 0.171 | 96.19 | 96.19 | 938 | 31.245 | yes | 63.8 | 39.7 |
| 005035 | 484.6 | 94.9 | 0.196 | 99.11 | 99.11 | 3971 | 132.328 | yes | 143.0 | 179.0 |
| **total** | **927.0** | **178.5** | **0.193** | | | 7926 | 264.1 | 4/4 | 349.2 | 359.6 |

Files are named by their timestamp: `ego_AZER76400FE_19700101_<ts>_camera_left_part0001_blurred.mp4`. Exact totals: 926,957,324 bytes before, 178,463,293 after. Wall time includes the encode and every check (frame count, duration, full decode, VMAF). The dry run and the apply pass produced byte-identical encodes (same sizes and SHA-256 per file), and the VMAF numbers equal the calibration's cq 28 points for each file.

After the run, `probe()` on each file in `converted/` reports codec `hevc` with the same frame count as its source.

### Sources and manifest

- The four sources in `blurred/` were left unchanged: size and SHA-256 of each were the same before and after the run, and no temporary file was left beside them.
- `converted/hot_storage_manifest.parquet` holds 8 rows: 4 `dry-run` (paths in `blurred/`) and 4 `applied` (paths in `converted/`). Every row has `checks_passed` true, VMAF at least 95, `codec_after` hevc, `encoder` nvenc, `quality` 28.

### Decode check fix

The first attempt of this run failed every file on `decodes_cleanly`: ffmpeg's null-output decode reported `non monotonically increasing dts to muxer`. The same message appears for the untouched H.264 sources, so it is not a defect of the encodes. By default ffmpeg puts the null muxer's timestamps on a 1/30 grid from `r_frame_rate`; the camera's 30.0085 fps clock then puts two frames on one tick. The check now decodes with `-fps_mode passthrough -enc_time_base -1`, as the encoder already does, and a test with a clock slightly faster than 30 fps covers it. That attempt's manifest was discarded with the rest of `converted/` before the run above.

## Random access (T07)

Driver `faceblur-hot-storage-data/t07/run.py`, run once through the GPU lock together with the full test suite:

```
python -m storage.randaccess <converted/..._005035_...> --source <blurred/..._005035_...> \
  --seeks 100 --seed 0 --json <t07/randaccess_005035.json>
```

File 005035 (3971 frames, 132.3 s): the NVENC HEVC copy (cq 28, gop 60, B-frames) against its H.264 source. Seed 0, 100 frame indices from `random.Random(0)`, the same for both files and both libraries. Each seek is timed from the seek call to the decoded BGR frame, with one reader reused across the 100 seeks; a seek is exact when the frame's SHA-256 equals that library's own sequential decode at that index. The run exited 0 in 287 s.

### OpenCV (`CAP_PROP_POS_FRAMES` + `read()`)

| file | frames seen | exact seeks | mean seek (ms) | median seek (ms) |
|---|---|---|---|---|
| HEVC copy | 3971 | 100/100 | 80.0 | 79.3 |
| H.264 source | 3971 | 100/100 | 137.8 | 142.0 |

Ratio HEVC / H.264 mean: **0.58**.

### PyAV (`container.seek(pts, backward=True)`, decode forward to the pts)

| file | frames seen | exact seeks | mean seek (ms) | median seek (ms) |
|---|---|---|---|---|
| HEVC copy | 3971 | 100/100 | 242.7 | 252.6 |
| H.264 source | 3971 | 100/100 | 727.6 | 756.5 |

Ratio HEVC / H.264 mean: **0.33**.

Both ratios are at or below 2.0: the bound holds. Seeking the copy is faster than seeking the source.

Plain `CAP_PROP_POS_FRAMES` was exact on 100 of 100 seeks on the HEVC copy, despite its B-frames, so no reader-side remedy was needed. The same held on the synthetic libx265 clip in `tests/test_storage_randaccess.py` (300 frames, gop 60, B-frames).

Why HEVC seeks compare as they do: a seek decodes from the keyframe at or before the target, so its cost grows with the distance to that keyframe. The sequential PyAV pass counted 67 keyframes in the copy (one every 60 frames) against 28 in the source (one every 150 frames, mean interval 144.4), so the copy decodes on average about 30 frames per seek against about 75 for the source, which outweighs HEVC's higher per-frame decode cost.

## CSV → Parquet (T08)

`storage/csv2parquet.py` turns the IMU (`*_imu.csv`) and timestamp (`*_pts.csv`) side files into Parquet without losing a byte.

Schemas, written with `pyarrow.parquet`, zstd level 9, dictionary encoding on:

- IMU: `timestamp_us` int64, `x`, `y`, `z` float64, `type` dictionary<int32, string>.
- pts (camera left, camera right, audio): `timestamp_us` int64.

The schema metadata key `csv2parquet` holds a JSON object with what a CSV parser forgets, detected from each file rather than assumed: the exact header line (the IMU header `timestamp_us\t,x\t,y\t,z\t,type` keeps its tabs), the line terminator, the decimal places of each float column (taken from the first data row), whether the file ends with a terminator, the column names and a format version (1).

Lossless means byte-identical: after each file is written, the Parquet is read back from disk, the CSV is regenerated from it in batches and compared with the source chunk by chunk, so no whole file is held twice. If a single byte differs the Parquet is deleted and the file is reported `failed`. That makes the converter refuse, rather than approximate, anything the stored facts cannot reproduce: a float with a different number of decimals than the first row, scientific notation (`1e-05`), a CRLF line in an LF file, leading zeros or a plus sign, empty fields, an unknown header. Negative zero (`-0.000000`) survives, since float64 keeps the sign. A file without a final newline round-trips byte for byte (the flag is stored); a header-only file is converted too.

```
python -m storage.csv2parquet PATH... [--tree ROOT] [--out DIR] [--apply]
```

Files given and every `*_imu.csv` / `*_pts.csv` under `--tree` (recursive) are converted; other files are skipped with a message. The Parquet goes beside the CSV, or with `--out` mirrored under DIR at its path relative to `--tree` (a file given outside the tree lands at its own name). `--apply` deletes the CSV after a passing round trip, and only when the Parquet is beside it, never with `--out` elsewhere. One line per file (status, bytes before and after, ratio after/before), a totals line, exit 0 only when every file passed.

### Real recording folder

`D:\Player One - Phase 0 full\AZER76400FF\Orbbec_Ego_AZER76400FF_19700104_222421`, converted with `--tree` to `--out` in the data folder (`faceblur-hot-storage-data\parquet\...`), on the RTX 3070 PC, CPU only. Wall time per file is a separate timed `convert` of that file alone (write, read back, full byte comparison).

| file | rows | size before (bytes) | size after (bytes) | ratio | round trip ok | wall s |
|---|---|---|---|---|---|---|
| `..._imu.csv` | 2,380,160 | 112,888,986 | 14,490,683 | 0.128 | yes | 6.55 |
| `..._audio_pts.csv` | 27,889 | 362,570 | 132,748 | 0.366 | yes | 0.26 |
| `..._camera_left_part0001_pts.csv` | 35,622 | 463,099 | 165,986 | 0.358 | yes | 0.21 |
| `..._camera_right_part0001_pts.csv` | 35,620 | 463,073 | 165,981 | 0.358 | yes | 0.15 |
| **total** | 2,479,291 | 114,177,728 | 14,955,398 | **0.131** | 4/4 | 5.55 (one CLI run) |

The CLI run over the folder exited 0 in 5.55 s with a peak working set of 447 MB. A second process, without the storage code, regenerated each CSV from its Parquet in plain Python (`f"{v:.6f}"` per value) and its sha256 matched the source for all four files. Sizes and modification times of all nine files in the read-only recording folder were the same before and after; `--apply` was not used, so every CSV is still in place.

## --codec hevc in the blur pipeline (T09)

`cli.py --codec {h264,hevc}` (setting `codec`, default `h264`) picks the codec of the blurred copy. `h264` is unchanged, with exactly the same ffmpeg command line as before. `hevc` writes the copy straight in the hot-storage format, so no second `storage.convert` pass is needed.

- Settings: `CODECS = ("h264", "hevc")`, validated. `to_dict()`, and so the audit record in the sidecar, now holds `codec`, `hevc_cq`, `hevc_crf` and `gop`.
- One source of truth for the arguments: `faceblur.video.video_codec_args(..., codec="hevc")` returns `storage.encode.codec_args("hevc", ...)`, the same set the calibration was measured with (NVENC `hevc_nvenc` cq `hevc_cq`=28, or libx265 crf `hevc_crf`=22, keyframe every `gop`=60 frames, B-frames on, `-tag:v hvc1`). The import happens inside the hevc branch, so the h264 path never loads `storage`.
- `--encoder auto` uses NVENC when `hevc_nvenc` works here (`nvenc_available("hevc_nvenc")`). The CPU fallback name `x264`, used by `--encoder x264` and by the batch retry when the GPU encoder refuses, maps to libx265 for hevc.
- No stream copies for hevc: `want_copy` needs `codec == "h264"`, because a copied H.264 piece of the source cannot be joined to HEVC pieces. Every frame is encoded. The encode still runs in `encode_seconds` pieces joined by the concat demuxer, and that join verifies with B-frames on, so the one-segment fallback was **not** needed. Before the real run, three 150-frame NVENC HEVC pieces were joined and checked: `segments.verify` came back empty, and all 450 frames decoded in order (max pixel difference 1). libx265 pieces joined the same way.

Command (RTX 3070 PC, through the GPU lock, otherwise default settings, the same settings that made the earlier H.264 copy):

```
python cli.py ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4 -o <data>/t09/out --codec hevc --workers 2 --no-progress
python -m faceblur.verify <source> <copy>
```

| | value |
|---|---|
| file | `ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4` (1600x1300, 30.02 fps) |
| source bytes | 59,057,133 (H.264) |
| H.264 copy bytes | 105,690,318 |
| HEVC copy bytes | 21,133,541 (`hevc`, NVENC) |
| ratio HEVC / H.264 copy | 0.200 (0.358 of the source) |
| frames (source / copy) | 984 / 984 |
| duration (source / copy) | 32.778 s / 32.778 s |
| `segments.verify` | empty (pass), also with the length check |
| pipeline join | 1 attempt, 0 frames copied, status done, sidecar `settings.codec` = `hevc` |
| `faceblur.verify`, HEVC copy | exit 1: 7 faces on untouched pixels (6 at 40 px and over, largest 220 px) over 7 frames; zone gate: 881 frames over 1.00 percent changed, worst 9.84 percent |
| `faceblur.verify`, H.264 copy (same arguments) | exit 1: 11 faces on untouched pixels (largest 179 px) over 11 frames, 3 more set aside as the wearer's hands; zone gate: 0 frames over, worst 0.099 percent |
| pipeline wall seconds | 184.1 (sidecar: detect 124.1, encode 47.2, wall 183.4) |
| verify wall seconds | 226.2 (HEVC copy), 252.3 (H.264 copy) |

Faces: the HEVC copy has no misses that the H.264 copy lacks. It has fewer (7 against 11), and both copies fail the face gate at default settings, so the codec adds no face misses. The two lists differ frame by frame because a lossy copy changes the "applied" share that `verify` measures on each box.

Zone gate: this one fails on HEVC only. It counts zone pixels that moved between the source and the copy, and its 1 percent threshold assumes a near-lossless copy (H.264 crf 12 / cq 16). At cq 28 the compression noise alone moves up to 9.8 percent of the zone's pixels. With `--check-output` or `--quarantine`, every HEVC copy would therefore be held back for `zone`. This needs a follow-up: compare the zone against a reference that knows the codec, or skip that gate for lossy storage copies. It is not changed here.
