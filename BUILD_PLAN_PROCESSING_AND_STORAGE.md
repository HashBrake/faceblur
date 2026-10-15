# Build plan: better processing at lower compute, and compressed hot storage

Written 2026-10-16 after the first full Phase 0 run (119 h of footage on eight
RTX 5090 VMs) and the hand toolkit pilot. Numbers are from that run unless
marked *estimate*. Prices: GreenNode RTX 5090 VM about 23,100 VND (~$0.89)
per hour; vStorage about 28 VND per GB per day (~$32 per TB per month).

## Where we start

| | Today |
|---|---|
| Face blur | ~2 VM-hours per hour of footage, ~$1.80 |
| GPU busy | 50-85 %; one-core steps (tracker, zone, encode) leave it idle |
| Blurred size | 21 % larger than the raw (H.264 at near-lossless CQ 16, no B-frames) |
| Hand toolkit (EgoSmith) | left camera only, monocular, ~0.5-1.5 VM-h per hour *estimate*, pilot running |
| Footage passes | blur and hands each download, decode and write the video separately |
| Failures seen | a 12 h stall in the tracker's stitch step; 13 sources cut off mid-frame; slices finishing hours apart |

## Part 1. Processing: as good as possible, compute-efficient

The order is by value per effort. Each item has a pass/fail measurement, and
nothing ships without a before/after on the same footage (`docs/report.md`).

### 1.1 Robustness (do first, ~2 days)

1. **Cut-off sources handled in the pipeline.** Probe each file's tail; if the
   last frame does not decode, work on the decodable frames and record it,
   instead of failing (today: `cloud/repair_tail.py` by hand).
2. **Watchdog per stage.** Any stage that runs more than k times its expected
   time for the video's length is logged and stopped, and the video is
   retried once and then listed. The 12 h stall would have cost 1 h.
3. **One shared work queue instead of fixed slices.** Machines take the next
   video from the bucket with a claim marker. Finished machines stop taking
   work, so the run ends at the same time on every machine. Today the
   slices finished 01:00 to 05:00 apart.
4. **Delete through the API.** An IAM service account (the owner creates it)
   lets `cloud/gn_api.py` delete each VM the minute its queue is empty. The
   console session lasts 12 h and blocked deletion overnight.

### 1.2 Blur: same quality, ~40 % less compute (~1 week)

1. **Remove the one-core bottlenecks.** Profile the tracker, the zone and the
   segment encode. Vectorise the tracker with NumPy and cache per-track
   frame ranges (the stitch fix gave 100x+ on one recording). Run the encode
   for blurred segments on NVENC sessions in parallel with detection of the
   next chunk. Target: GPU above 85 % with one lane, no lane juggling.
2. **Check only what needs checking.** Today every video is detected, written,
   checked, re-tracked, written and checked again. Instead, recheck and
   rewrite only the segments where the check found faces, plus a margin.
   *Estimate*: second-chance cost drops by 60-80 %, and the total by ~30 %.
3. **Faster detectors.** Export YuNet and the confirm detector to TensorRT FP16
   on the 5090, batch frames across the clip. *Estimate*: 1.5-2.5x on the
   detection stage. Gate: the same faces found on the benchmark set
   (`eval/`), and no drop in recall.
4. **Smarter stride.** Detect every frame only where faces are present or
   the camera moves fast. Elsewhere detect every 2nd-3rd frame and track
   between. Gate: recall on the benchmark set within 0.5 %.

Target: ~1.2 VM-h per hour of footage (from ~2), with recall equal or better.

### 1.3 Hands: much better 3-D, at modest cost (~2 weeks after the pilot)

The partner toolkit uses one camera and guesses depth. Our headset has a
calibrated stereo pair (12 cm baseline) and an IMU.

1. **Use the calibration.** Undistort (KB model) before inference, so hands at
   the image edge are not distorted.
2. **Both cameras.** Run the hand detector and a 2-D keypoint model on left
   and right. Triangulate the 21 joints: about 4 mm depth error at 0.5 m
   *estimate*, against several cm monocular.
3. **Fit MANO to the triangulated joints over time.** One hand shape per
   person and recording, fixed bone lengths, smoothing, short gaps filled,
   long gaps flagged. Use HaWoR only as the initial guess and where one view
   is occluded.
4. **World coordinates.** Stereo-inertial odometry (Basalt, OpenVINS or
   ORB-SLAM3) gives the head's path, so hands are placed in the room.
5. **Quality per recording.** Left/right reprojection agreement, jitter,
   bone-length variance and % valid frames go into a per-recording score,
   so people only review the bad ones.
6. **Licences.** HaWoR is CC BY-NC-ND and MANO is non-commercial. The owner
   decides before commercial use. A fully commercial route is MediaPipe or
   RTMPose 2-D keypoints plus our triangulation, with joints only and no MANO.

Compute *estimate*: 1-3 VM-h per hour of footage for both cameras, the fitting
and the odometry.

### 1.4 One pass per video (~1 week, after 1.2 and 1.3)

Today each tool downloads, decodes (and the hand toolkit writes ~50 GB of
JPEGs per hour) on its own. Instead, one job per video:

download once → NVDEC decode once → frames shared in GPU memory by face
detection, hand detection and keypoints → blurred video encoded once
(HEVC, see Part 2) → hand outputs written as Parquet.

This saves the second download and decode, and the JPEG detour (disk and
~15 min per hour of video). *Estimate*: 25-35 % less wall time than running
the two pipelines one after another.

### 1.5 Hardware choice (1 day)

Benchmark one hour of footage on 5090, 4090, L40S and H100 with the finished
pipeline, and pick the lowest cost per hour of footage. Ask GreenNode for a
volume price once the yearly volume is known.

### Expected result

| | Today | After Part 1 *estimate* |
|---|---|---|
| Blur | ~2 VM-h / h, $1.80 | ~1.2 VM-h / h, ~$1.10 |
| Blur + stereo hands | n/a | ~2-3 VM-h / h, ~$2-2.70 |
| Overnight babysitting | console sign-ins, manual fixes | none: queue, watchdog, API deletion |

## Part 2. Compression for hot storage

Hot storage holds what people and training jobs read often. The raw footage
is not kept. The goal is the fewest TB-months at no visible loss and with
fast random access to frames.

### 2.1 Video

1. **Codec.** HEVC (H.265) via NVENC as the default hot format: ~30-50 % fewer
   bits than H.264 at the same quality, and decoded everywhere (ffmpeg,
   OpenCV, PyAV, browsers on most systems). AV1 (NVENC AV1 on the 5090) is
   another ~10-20 % smaller, but decoding is slower and less universal.
   Adopt it only if the training data loaders decode it at the needed
   speed (test with dav1d).
2. **Quality target, measured.** Tune the constant-quality setting per codec so
   VMAF against the blurred H.264 is ≥ 95 (visually lossless) on a test set
   covering every device and lighting. Store the setting in `settings.py`.
3. **Random access.** A keyframe every 1-2 s (GOP 30-60) with B-frames. The
   data loaders seek to arbitrary frames, so long GOPs that save 3-5 % but
   make seeks slow are not worth it.
4. **Where.** Written directly by the blur step (Part 1.4). The copy-unblurred-
   segments trick goes away, since H.264 pieces cannot join HEVC ones; the
   whole video is re-encoded once on NVENC, which is fast. For the current
   1.25 TB tree, a one-off convert: ~20-30 VM-h (~$18-27), file by file
   with checks.
5. **Checks per file before the H.264 copy is deleted.** Same frame count,
   duration within a frame, decodes clean end to end, VMAF on 3 sampled
   minutes ≥ 95, checksum recorded.

### 2.2 Everything else

| Data | Today | Hot format | Expected saving |
|---|---|---|---|
| IMU logs (`*_imu.csv`, ~1.3 GB per recording) | CSV text | Parquet (zstd), float32 columns | 5-10x |
| Frame timestamps (`*_pts.csv`) | CSV | Parquet, or into the video's metadata | 5x+ |
| Audio (`*_audio.wav`, not delivered today) | WAV | not kept, or FLAC (lossless, ~2x) / Opus if needed | 2-10x |
| Hand outputs (MANO JSON per chunk) | JSON text | one Parquet per video | 5-10x |
| Hand skeleton videos | MPEG-4 Part 2 | not stored: re-rendered on demand from the joints | 100 % |
| Calibration YAML | small | as is | — |

### 2.3 Layout

- One folder per recording, as today, so a recording is one prefix to fetch.
- A per-tree manifest (Parquet): file, size, checksum, codec, frames,
  duration, VMAF sample, held-back status. Tools read the manifest, not the
  bucket listing.
- The delivered names stay the same, so links and scripts keep working.

### Expected result for Phase 0 *estimate*

| | Today | Hot format |
|---|---|---|
| Video | ~1.2 TB (blurred H.264) | ~0.7-0.9 TB (HEVC) |
| IMU + timestamps | ~0.1 TB | ~0.01-0.02 TB |
| Hand outputs | ~0.06-0.4 TB | ~0.01-0.05 TB |
| **Total** | **~1.3-1.7 TB** | **~0.75-0.95 TB** |
| Storage per month | ~$42-54 | ~$24-30 |

## Schedule

| Step | Effort | Depends on |
|---|---|---|
| 1.1 Robustness | 2 days | service account for API deletion |
| 2 Compression test on 4 files (VMAF, decode speed) | 0.5 day | — |
| 2 Convert the Phase 0 tree to HEVC + Parquet | 1 day of runs | test passed |
| 1.2 Blur efficiency | 1 week | — |
| 1.3 Stereo hands | 2 weeks | hand pilot result, licence decision |
| 1.4 One pass per video | 1 week | 1.2, 1.3 |
| 1.5 Hardware benchmark | 1 day | 1.4 |

About 5 weeks of work for one person, plus a few hundred dollars of test
compute. The compression part pays for itself in about a month of storage.
