# Privacy desensitization module: measurements

Measurements taken while building the privacy desensitization module
(licence plates, text, screens and the rest on top of the face pipeline),
2026-11-05. Each task that needs a number to choose a default writes it here,
with the setup that produced it, so that the defaults in
`faceblur/settings.py` can be traced to something. Later tasks append
sections.

## Summary of the numbers

One line per measurement the module was asked for; the section named at the
end of each line has the setup and the caveats.

- **Plate recall on pasted plates**: at the default `plate_size` 1280, 16/16
  plates of 160 and 240 px found on grey and on a real frame, 0/16 of 60 and
  100 px; no plate-sized false find on 40 real frames. Licence plates.
- **Text coverage**: on the pasted-text clip, 100 % of the ink pixels are
  under the solid fill in all 6 frames, and no pixel more than 40 px from
  the text moves by more than 5.7 levels; the detector costs 157 ms a frame
  on DirectML at the default sizes, 58 ms at 1920 alone. Text.
- **Re-ID after a deblur attempt**: a solid face fill leaves 0/60 faces
  matched under both attacks, against 2/60 for the old blur 6 and 1/60 for
  pixelate 6, so faces are filled solid. Face obfuscation against a deblur
  attempt.
- **Speed per frame per camera**: 246 to 284 ms a frame, 259 ms over 7926
  frames of four cameras, two workers on the RTX 3070; one hour of one
  30 fps camera is about 7.8 hours. Speed per frame per camera.
- **Tracker before and after the forward walk**: recall 60/60 both,
  continuity 98.43 % to 98.52 %, exposure counts unchanged. Tracker: forward
  extension.
- **Docker build and run**: cold build 945 s, rebuild 35 to 45 s; the GPU
  run on the two-camera test file exits 0 in 18 s with every model on CUDA
  and an audit that verifies. Docker image.
- **Voice**: Silero VAD v6.2.3 finds the one spoken sentence of the test
  sample as (0.0, 2.956) s with the 300 ms pad. Voice.

## Licence plates

Task T06b. Detector: LPD-YuNet (`models/lpd_yunet_dynamic.onnx`, see
`models/README.md`) through `faceblur/plates.py`, at the zoo's score threshold
0.8 and NMS 0.3, `plate_min_px` 24, `plate_pad` 0.10. The tables below were
measured with `plate_max_frac` 0.5; the default is now 0.25 (see the end of
this section).

### Setup

- **The plate.** The plate in `tests/data/plate_cc0.jpg`, cut at the
  hand-measured box (136, 411, 182, 64) and scaled to each width with its
  aspect kept (60 px wide is 21 px tall, 240 px wide is 84 px tall).
- **Backgrounds.** A flat grey (128) 1600x1300 frame, and frame 300 of
  `footage/ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4` (1600x1300,
  read with cv2), the camera's own size.
- **Positions.** Eight centres, a 4 by 2 grid at 15, 38, 62 and 85 percent
  across and 30 and 70 percent down.
- **Found** means a detection whose box has IoU of at least 0.5 with the
  pasted rectangle.
- `plate_size` is the long side the frame is resized to before it is padded
  to a multiple of 32; 1600 is the frame's raw size.

### Recall, found / 8

| plate_size | background | 60 px | 100 px | 160 px | 240 px |
|---|---|---|---|---|---|
| 1280 (default) | grey | 0/8 | 0/8 | 8/8 | 8/8 |
| 1280 (default) | real frame | 0/8 | 0/8 | 8/8 | 8/8 |
| 1600 (raw) | grey | 0/8 | 4/8 | 8/8 | 8/8 |
| 1600 (raw) | real frame | 0/8 | 4/8 | 8/8 | 8/8 |
| 1920 | grey | 0/8 | 8/8 | 8/8 | 8/8 |
| 1920 | real frame | 0/8 | 8/8 | 8/8 | 8/8 |
| 2560 | grey | 8/8 | 8/8 | 8/8 | 8/8 |
| 2560 | real frame | 6/8 | 8/8 | 8/8 | 8/8 |

The model finds a plate once it is about 120 px wide **at its input**,
whatever the frame: 160 px in the frame is 128 px at 1280, 100 px is 120 px
at 1920, 60 px is 96 px at 2560. Making the frame bigger is the only lever on
small plates and it costs time and false finds (below).

### False finds on real frames, no plate pasted

40 frames of the same file (every 25th frame, 1600x1300; an indoor washroom,
no vehicles), DirectML.

| plate_size | frames with a find | finds | finds under 400 px wide |
|---|---|---|---|
| 1280 (default) | 5 | 5 | 0 |
| 1600 (raw) | 10 | 12 | 0 |
| 1920 | 22 | 31 | 8 |
| 2560 | 34 | 75 | 26 |

At 1280 every false find is a wide band 524 to 829 px across (the mirror
row, a wall), scoring 0.82 to 0.93. Those would be filled solid if they last
`plate_min_run` frames; `plate_max_frac` 0.5 (800 px here) lets most of them
through. From 1920 up the false finds include plate-sized boxes, such as the
"NO SMOKING" sign on frame 100 (147x80, 0.94).

### Speed, ms per frame at 1600x1300

Mean of 20 calls after 3 warm-up calls, the whole `PlateDetector.detect`
(resize, pad, inference, decode, NMS).

| plate_size | DirectML (RTX 3070) | CPU |
|---|---|---|
| 1280 (default) | 62.9 | 141.5 |
| 1600 (raw) | 106.9 | 229.3 |
| 1920 | 149.1 | not measured |
| 2560 | 319.4 | not measured |

### Choice: `plate_size` 1280

1280 is kept. It is the only size with no plate-sized false find on the 40
real frames, it costs 63 ms a frame on DirectML against 107 at the raw size,
and it finds every plate of 160 px and over. The raw size buys half of the
100 px plates for 70 percent more time and twice the false finds; 1920 buys
all of them at 2.4 times the time and plate-sized false finds on 8 frames in
40.

**What this costs.** At 1280 a plate under about 150 px wide in a
1600x1300 frame is not found. Taking the camera's 1600 px as about 100 degrees across, a
52 cm plate is about 115 px wide at 3 m, so plates more than a few metres
away will be missed.
The T12 field test must measure recall on real plates before plate masking
is relied on.

### Choice: `plate_max_frac` 0.25

At 1280 every false find on the 40 real frames was a band 524 to 829 px
across, over 0.3 of the 1600 px frame width, and at 0.5 most of them would
have been filled solid. 0.25 (400 px here) removes all of them. A real plate
wider than a quarter of a 1600 px frame is on a car the wearer is touching.
The pasted plates up to 240 px (0.15 of the width) are unaffected, so the
recall table holds at 0.25.

## Tracker: forward extension

### What changed

`Tracker._extend_forwards` (faceblur/track.py) is the mirror of
`_extend_backwards`. After linking, gap filling and the backward walk, each
established and sure track walks on from its last detection over boxes no
kept track owns: confirmed boxes, and weak boxes down to `conf_weak_long`.
The prediction is the last box it stood on moved by the camera, the match is
`_affinity`, the reach is `_gap_ok` from the last box taken. Unlike the
backward walk it also stops `weak_run` frames after the last confirmed
detection, so a hand confirmed once as a face is not followed for fifty
frames. During linking a track already continues on weak boxes, but through
the velocity prediction of `_match`; the walk adds the boxes that prediction
missed, typically a face that slows or turns as it leaves the picture.
Tails still start after the extended last frame.

### Measurement, 004100 sample

`.venv/Scripts/python.exe -m eval.measure footage/ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4`,
default settings, the same raw-detection cache for both (the cache key does
not cover track.py). "Before" is commit d41a155. The full-precision numbers
come from `eval.measure.evaluate` on the same inputs with and without the
walk; the printed lines match them.

Printed line, before:

    recall 100.0%  off-face mean 0.22% max 5.78% (strict 0.34%, detections 0.04%)  exposed 40+ 85 fr, 24-40 55 fr  hands n/a  continuity 98.4%  masked mean 0.56% p95 1.72% max 5.78%  tracks 30  [large 64+: 24/24  medium 32-63: 28/28  small <32: 8/8]

Printed line, after:

    recall 100.0%  off-face mean 0.23% max 5.78% (strict 0.35%, detections 0.04%)  exposed 40+ 85 fr, 24-40 55 fr  hands n/a  continuity 98.5%  masked mean 0.56% p95 1.72% max 5.78%  tracks 30  [large 64+: 24/24  medium 32-63: 28/28  small <32: 8/8]

| number | before | after | change |
|---|---|---|---|
| recall | 1.0000 (60/60) | 1.0000 (60/60) | 0 |
| off_face mean | 0.0022407 | 0.0022519 | +0.0000113 |
| off_face max | 0.057755 | 0.057755 | 0 |
| off_face_strict mean | 0.0034499 | 0.0034664 | +0.0000166 |
| off_face_detections mean | 0.00044630 | 0.00044636 | +0.00000006 |
| hand_damage_wearer | n/a (no hands cache) | n/a (no hands cache) | |
| exposed_40 | 85 | 85 | 0 |
| exposed_24_40 | 55 | 55 | 0 |
| continuity | 0.98433 | 0.98517 | +0.00084 |
| masked mean | 0.0055597 | 0.0055824 | +0.0000226 |
| masked p95 | 0.017175 | 0.017175 | 0 |
| masked max | 0.057755 | 0.057755 | 0 |
| tracks | 30 | 30 | 0 |
| track length mean | 42.77 | 43.43 | +0.67 |
| recall by size | 24/24, 28/28, 8/8 | 24/24, 28/28, 8/8 | 0 |

Acceptance: recall and continuity are not lower, exposed_40 and
exposed_24_40 are not higher, off_face rose by 0.0000113 (limit 0.001) and
masked mean by 0.0000226 (limit 0.002). The first variant passed, so no
stricter variant was tried and the walk is on with no new setting.

The gain on this clip is small: tracks grew by 20 frames in all and
continuity rose 0.08 points, while the exposure counts did not move. Most
weak boxes past a track's end were already taken by the velocity-based
continuation during linking. hand_damage_wearer could not be read: the
004100 sample has no hands cache, and building one was outside this task.

## Face obfuscation against a deblur attempt

Task T08, decision D-C. The face mask shipped as `mode` "blur" at
`strength` 6: the region shrunk to about six blocks across the face, scaled
back up linearly and blurred (`redact._cover_region`). Blur is a linear
filter and diffusion-era deblurring recovers part of it, so the brief was:
keep this mask only if a face recogniser still fails after a deblur attempt,
otherwise fill faces solid.

### Setup

- **Faces.** The 60 largest detector sightings (at least 48 px, landmarks
  present, not track-filled) of the 004100 sample, picked as
  `eval.reid.mask_curve` picks them, from its raw cache and the shipped
  tracker settings. 60 faces of at least 48 px were available, so the size
  floor was not lowered.
- **Recogniser.** SFace (`models/sface.onnx`, `eval.reid.Recogniser`), five
  landmark alignment. Each face is embedded from the source, masked with
  `redact.redact(frame, [face], settings)` with that face as the only box,
  attacked, and embedded again with the same landmarks and alignment, so the
  mask and the attack are the only difference between the two crops.
- **Threshold.** 0.582, the score 99 percent of stranger pairs on this
  footage stay below (`threshold_fmr1` in the sample's `reid.json`). The
  published SFace threshold, 0.363, is reached by 15.5 percent of strangers
  here and would count noise as matches.
- **Attacks**, both on the box the mask was drawn in (region bounds plus
  twice the feather, as `_cover_region` cuts it), written back into a copy of
  the frame:
  - *deconvolve*: upsample x2 with `INTER_CUBIC`; unsharp mask, frame + 1.5 x
    (frame - Gaussian sigma 3); Richardson-Lucy, 10 iterations, float32 with
    clipping to [0, 1] and an epsilon of 1e-6, `cv2.filter2D`, Gaussian PSF of
    sigma k/2 in frame pixels where k = max(2, round(2 x block_scale /
    strength)) is the mask's own block size (sigma k on the x2 grid);
    downsample back with `INTER_AREA`. The attacker is handed the exact block
    size.
  - *sharpen*: upsample x2 cubic, bilateral filter (d 9, colour 40, space 5),
    the same unsharp mask, downsample back with `INTER_AREA`. No knowledge of
    the mask.
- **Pass criterion.** At most 1 percent of the faces still match their source
  after an attack, which at 60 faces means none, under both attacks.

### Result

    .venv\Scripts\python.exe -m pytest tests/test_reid_deblur.py -s -p no:cacheprovider

| mask | attack | still matched | worst score | passes |
|---|---|---|---|---|
| blur 6 | deconvolve | 2/60 | 0.594 | no |
| blur 6 | sharpen | 2/60 | 0.613 | no |
| pixelate 6 | deconvolve | 1/60 | 0.607 | no |
| pixelate 6 | sharpen | 1/60 | 0.631 | no |
| solid | deconvolve | 0/60 | 0.531 | yes |
| solid | sharpen | 0/60 | 0.550 | yes |

For reference, without any attack (`mask_curve`, same faces and threshold,
in the sample's `reid.json`): blur 6 matched 2/60 (worst 0.593), pixelate 6
1/60 (worst 0.645), solid 0/60 (worst 0.552). Blur 6 already sat on the
threshold before the attack; the attacks pushed the worst score of each mode
up by 0.00 to 0.02 for blur and did not move the count.

**How many of the 60 the number is about.** A face counts as identifiable in
the source when SFace matches it, at the same threshold, to the same track in
another frame (`eval.reid.verdict`'s self-match, against up to 40 evenly
sampled other sightings of that track). 45 of the 60 are; the other 15 the
recogniser cannot place even untouched, so a mask cannot make them less
identifiable to it. The solid result is 0 of those 45 as well.

Solid is not zero by construction. The aligned SFace crop around a face
still holds the hair, ears and background outside the ellipse, which the
mask leaves exact, and those pixels alone carry a face to 0.55 against its
own source, close to the threshold. That margin is what the ellipse size
(`ellipse_w`, `ellipse_h`) buys, and it is the next thing to watch if the
threshold or the recogniser changes.

### Decision: `face_fill` "solid"

Blur 6 fails, so `Settings.face_fill` (new, "mode" or "solid") defaults to
"solid" and `redact.fill_mode("face", settings)` returns "solid" whatever
`mode` says. Every face is filled solid on every path that masks through
`redact`: the command line and the UI today, and the HDF5 module when it
starts masking, since it is to mask through the same code. `face_fill` "mode"
gives the old behaviour back for anyone who chooses it, and is recorded in
the audit settings like every other setting. Pixelate 6 also fails, so it is
not an alternative. Screens still follow `mode`; text and plates were already
solid.

`tests/test_reid_deblur.py` asserts that the mask the shipped default selects
passes both attacks, so a change back to blur or pixelate as the default
fails the suite while this measurement says what it says. The test skips
with its reason when the 004100 footage or its raw cache is absent.

## Docker image

`cloud/Dockerfile` now builds two images from shared layers. `privacy`, the
default target, is the de-identification module alone:

    docker build -f cloud/Dockerfile -t faceblur-privacy .
    docker run --gpus all -v <in>:/in:ro -v <out>:/out faceblur-privacy

`notebook` (`--target notebook`) is the GreenNode image as before, JupyterLab
and sshd as root.

What the privacy stage does:

- Runs as `faceblur`, uid 10001, which owns `/out` and nothing else. No
  JupyterLab, no SSH server, no rclone, no exposed port.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`: every model is in the
  image and nothing may fetch one at run time.
- `cloud/privacy_entry.sh` first runs `sha256sum -c models/SHA256SUMS` and
  stops on any mismatch, so the weights that run are the ones reviewed.
  `models/make_sums.py` writes that file; `tests/test_models.py` checks it
  against the hashes pinned there.
- Every `/in/*.h5` and `/in/*.hdf5` goes through `python -m faceblur.h5 run`.
  One input writes `/out/episode.h5` and `/out/audit.jsonl`, several write
  `/out/<stem>/`. `FACEBLUR_WORKERS` (default 2) and `FACEBLUR_ARGS` (e.g.
  `--spec-layout --mask face`) come in through `docker run -e`. Each audit is
  then checked with `faceblur.h5 verify`. Exit 2: no input; 1: a run failed or
  an audit did not verify.
- `HEALTHCHECK` runs `python -m faceblur.h5 selfcheck` every 5 minutes: every
  masking model (YuNet, CenterFace, UltraFace, YOLOX, PP-OCR, LPD-YuNet) loads
  and runs once on a flat 640x480 frame, and the provider each one ran on is
  printed. On this PC's `.venv`, `selfcheck --device cpu` takes about 4.5 s.

### Built and run on this PC

Docker Desktop 29.5.3 on Windows 11 with WSL 2, an RTX 3070 (8 GB), and the
`nvidia` runtime that Docker Desktop provides.

| | |
|---|---|
| Build, cold (base image pulled, pip install) | 945 s |
| Rebuild after a code or config change (cached layers) | 35-45 s |
| Image size | 9.77 GB on disk, 3.59 GB content (compressed) |
| `id -u` in the container | 10001 |
| `docker inspect` healthcheck | `[CMD python -m faceblur.h5 selfcheck] 5m0s 2m0s 1m0s 0s 1` |
| `selfcheck`, `--gpus all` | exit 0, about 7 s, every model on `CUDAExecutionProvider` |
| `selfcheck`, no GPU | exit 0, all on the CPU (see below) |
| `/app/cloud` in the image | no `mano`, no `*.env` |

The `--gpus all` run on `tests/data/df2_small.h5` (2 cameras, 12 frames each)
exited 0 after 18 s in total, including the sha256 check, container start and
model load. It wrote `/out/episode.h5` (139 KB) and `/out/audit.jsonl`
(29 lines), and the audit verified both in the container and on the host
(`ok: 29 lines`). Per camera: `rgb_ego_center_left` took 433 ms/frame and
masked nothing; `rgb_ego_center_right` took 416 ms/frame and masked text on
all 12 frames. Most of that time is the first inference on each session at a
new shape, so 12 frames say nothing about steady-state speed. Every model the
`camera_done` records list as `compute` ran on `CUDAExecutionProvider`:
yunet, centerface, ultraface, yolox, ppocr, lpd_yunet, zone_palm and
zone_landmark. No model fell back to the CPU, and the log has no onnxruntime
EP error. The run without `--gpus all` was not needed.

The notebook target was not built. It shares `base` and only adds time.

Errors found by running the image, and fixed before this was committed:

- `sha256sum -c models/SHA256SUMS` reads the names in the file from the
  folder it runs in, `/app`, not from `models/`. With bare names every model
  "could not be read" and the container exited 1 before any run.
  `models/make_sums.py` now writes `models/<file>`, and the test's
  re-implementation resolves names from the repo root, the way sha256sum does.
- `faceblur.h5 verify` reads `docs/audit_schema.json`, which the image did not
  carry. The run had finished, but verify raised FileNotFoundError and the
  entry script exited 1. `base` now copies the schema, and `.dockerignore`
  lets it in.
- `.dockerignore` let `cloud/mano/` (the licensed MANO hand models, which
  must never ship) in through `!cloud`. It is now excluded again after the
  allows, like `*.env`.

One finding is left for its own task. Without a GPU, onnxruntime falls back
from CUDA to the CPU for every model. `selfcheck` still prints
`CUDAExecutionProvider` for yunet, centerface and ultraface, while yolox,
ppocr and lpd_yunet correctly say CPU. The face backends copy
`pool.provider` when they are built (`faceblur/detect.py`, the
`self.provider = self.pool.provider` lines) and do not see the session the
pool creates later, so the audit's `compute` can say CUDA for a face model
that ran on the CPU. In the GPU run above the label was right, because CUDA
really ran.

Fixed in T14 and T15. `provider` on the three face backends is now a property
that reads the pool each time, like `TextDetector` and `PlateDetector`
already did, so `compute` names the session that actually ran. A second
mislabel turned up beside it: `faceblur.h5 run --device cpu` and
`selfcheck --device cpu` still ran the plate model on DirectML, because the
plate, zone and hand models follow `plate_device`, `zone_device` and
`hand_device` (default "auto"), not `device`. In the HDF5 module `--device`
now sets all four; the desktop `cli.py` is unchanged. Checked by
`tests/test_provider.py` (a backend whose pool changes provider after it was
built reports the new one; a CPU bank reports `CPUExecutionProvider` for
yunet, centerface and ultraface; on this PC an "auto" bank reports
`DmlExecutionProvider` after one frame) and by a test in
`tests/test_h5_run.py`: `selfcheck --device cpu` prints only
`CPUExecutionProvider`, lpd_yunet included, and every `camera_done` record
of a `run --device cpu` on `tests/data/df2_small.h5` lists only
`CPUExecutionProvider` in `compute`. Without the `--device` fix that test
fails on the plate line.

The audit names no image yet: the entry script warns that neither
`/etc/faceblur_image_digest` nor `FACEBLUR_IMAGE_DIGEST` is set. Pass
`-e FACEBLUR_IMAGE_DIGEST=$(docker inspect --format '{{.Id}}' faceblur-privacy)`
to record it.

## Text

Task T11. Detector: PP-OCRv3 text detection
(`models/text_detection_ppocrv3.onnx`) through `faceblur/text.py`, default
settings: `text_sizes` (1920, 2560), filled solid whatever `mode` says.

### Coverage on pasted text

The clip `tests/test_text_mask.py` builds and runs, re-created from its own
helpers (`text_frame`, `run`, `decode_all`, imported from a scratch script):
three lines of black text ("INVOICE 4471", "JANE DOE", "PAID 12.50",
OpenCV Hershey simplex at scale 1.2, 3 px stroke) on a light panel on a grey
wall, 640x480, 6 frames at 10 fps, libx264 crf 12. It goes through
`batch.run_video` with `mask` text, the zone off, nothing copied, on the CPU,
and the copy is decoded and compared with the drawn frame.

- **Ink** is every pixel darker than 80 in the drawn frame, grown by 2 px
  each way: 14817 pixels. **Filled** means within 8 levels of black in the
  copy.
- **Far** is every pixel more than 40 px from any ink: 228250 pixels, 74 %
  of the frame.

| frame | ink filled solid | far, mean change | far, changed by more than 3 | far, largest change |
|---|---|---|---|---|
| 0 to 5, each | 100.00 % | 2.67 levels | 0.35 % | 5.7 levels |

Every frame gives the same numbers, because the clip is one still. The
run's record says `text_lines` 18 and `text_frames` 6, which is the three
lines found on each of the 6 frames. Text takes 10.5 % of the frame on
average. The source clip decodes to the drawn frame exactly, so the
2.67-level shift on far pixels comes from the copy's own re-encode, not the
mask; no far pixel moves by more than 5.7 levels, and the test's limit is a
mean of 3.

This shows that clean printed text is found and covered completely. It says
nothing about text that is small, slanted, handwritten or smeared by motion;
the T12 field test has to measure that on real footage.

### Speed, ms per frame at 1600x1300

Frame 300 of `footage/ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4`,
the frame the plate tables use. Mean of 20 calls after 3 warm-up calls, the
whole `TextDetector.detect` (resize to each size, inference, contours,
merging across sizes), one process at a time. The frame holds no text the
detector keeps, so this is the cost of looking, which every frame pays.

| text_sizes | DirectML (RTX 3070) | CPU |
|---|---|---|
| (1920, 2560) (default) | 156.6 | 600.1 |
| (1920,) | 57.5 | 211.3 |

The 2560 pass is 99 ms of the 157 on DirectML, about 63 percent of the text
cost, and the text detector at the default sizes costs two and a half times
the plate detector (63 ms) on the same card. It is the first place to look
for speed (decision D23); what dropping 2560 costs in recall on small text
has not been measured here.

## Speed per frame per camera

Task T09a, reported here in T11. A DF2 file packed from the four sample
mp4s, one per head-worn camera (`out/fixture4.h5`, 575 MB, 1600x1300,
30 fps), through `python -m faceblur.h5 run` with two workers, `mask`
face,text,plate,screen, the zone on, DirectML on the RTX 3070. Time is wall
time per camera; cameras run one after another.

| camera | frames | time | ms per frame | face | text | plate | screen | all |
|---|---|---|---|---|---|---|---|---|
| rgb_ego_center_left | 2033 | 511 s | 251 | 0.84 % | 0.87 % | 0.08 % | 0.03 % | 1.80 % |
| rgb_ego_center_right | 984 | 242 s | 246 | 0.57 % | 1.00 % | 0.05 % | 0.43 % | 2.00 % |
| rgb_ego_side_left | 938 | 267 s | 284 | 2.38 % | 1.15 % | 0.01 % | 0.01 % | 3.52 % |
| rgb_ego_side_right | 3971 | 1017 s | 256 | 0.73 % | 0.85 % | 0.43 % | 0.28 % | 2.20 % |
| all four | 7926 | 2050 s | 259 | | | | | |

The kind columns are the mean share of the frame masked for that kind; the
last is the union. `rgb_ego_side_left` has the most face area and is the
slowest.

**An hour of footage.** One hour of one 30 fps camera is 108000 frames; at
259 ms a frame that is about 7.8 hours on this PC. A four-camera hour is
about 31 hours. The output was 924 MB from the 575 MB input, and
`frames_copied` was 0: a mask reached every group of pictures, so every
frame was re-encoded at the pipeline's high quality instead of copied.

**The audit on one camera.** The 004310 file alone (938 frames) ran in
4 min 35 s and wrote 941 audit lines (header, 938 frames, `camera_done`,
footer), 1.06 MB, which `faceblur.h5 verify` checked in 1.7 s. Summed over
its frame lines the counts are face 18218, text 3093, screen 11 and plate 7.
The face count is every mask drawn, interpolated and tail boxes included,
from 114 tracks and 7125 raw detections, so it is far above the number of
faces a viewer would count; boxes that fill a gap can overlap.

## Voice

Task T09b. The release carries no audio (decision D-B); the audit lists
where the removed audio held speech. Silero VAD v6.2.3
(`models/silero_vad.onnx`, MIT, sha256 1a153a22...8788e3) through
`faceblur/vad.py`, on the CPU, threshold 0.5, `vad_pad_ms` 300, minimum
speech 250 ms, minimum silence 100 ms.

The sample, `tests/data/speech_16k.wav`, is 4.7 s of 16 kHz mono: one
synthesized sentence, "The quick brown fox jumps over the lazy dog.", voiced
from about 0.11 s to 2.59 s, then 2 s of digital silence.

| | start | end |
|---|---|---|
| voice in the sample | about 0.11 s | about 2.59 s |
| segment found, 300 ms pad included | 0.0 s | 2.956 s |

One segment. It covers all the speech and ends 0.37 s after the voice: the
300 ms pad and about two of the model's 32 ms windows. The start is clamped
at 0 because the pad reaches before the file. Finding it took 0.19 s.

What the audit holds: one `speech` line per segment, `start_s`, `end_s` and
`pad_ms`, written right after the header and before any frame, so even a run
that dies during the video records what audio was removed. The `models` map
names `silero_vad` with its sha256 whenever the input has an audio group.
The audit says when someone spoke, never what was said or who said it; no
wearer and bystander split is made (decision D-B).

## Strides from the pilot

Task T13a, after the field-test pilot (`docs/privacy_module_fieldtest.md`,
defects 5, 9 and 10). The pilot's single-process stage profile put text at
160 ms a frame, plates at 112, the zone at 78 and faces at 45, for 242 ms a
frame end to end at 2 workers. Three changes:

- `text_stride` 2: the text detector runs on every 2nd frame. `text.hold`
  closes gaps of up to `text_gap` (6) frames, so a line seen on frames 0, 2
  and 4 is masked on 1 and 3 as well (`tests/test_strides.py`).
- `plate_stride` 3: the plate detector runs on every 3rd frame, bridged by
  `plate_gap` (6) the same way.
- `zone_stride` 2 in `python -m faceblur.h5 run` and `selfcheck` only. The
  desktop CLI keeps `Settings().zone_stride` 1.

And one precision rule: a plate box taller than `plate_aspect_max` (0.8) of
its width is dropped. Plates are 2:1 to 5:1 wide; the pilot's false plates
were 166x282 portrait boxes on bins.

`python -m faceblur.h5 run AM_032259_L.h5 --workers 2`, one 1600x1300 camera,
4864 frames, on the same PC (RTX 3070, DirectML). Masked shares are mean
percent of the picture over all frames.

| run | ms/frame | seconds | detect_s | encode_s | mask_face% | mask_text% | mask_plate% | mask_screen% | zone_prevented% | text_lines | plates | plate_frames | tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot (strides 1/1/1) | 252 | 1227.5 | 1043.49 | 146.23 | 0.75 | 0.85 | 0.12 | 0.27 | 0.279 | 23681 | 167 | 280 | 93 |
| T13a (text 2, plate 3, zone 2) | 152 | 741.3 | 583.71 | 126.82 | 0.76 | 0.68 | 0.004 | 0.27 | 0.216 | 11857 | 12 | 7 | 93 |

The run is 40 percent faster: 152 ms a frame against 252, 12.5 minutes of
wall time against 20.5. Faces and screens are unchanged (the same 93 tracks).
`text_lines` halves because the detector looks at half the frames; the text
mask falls from 0.85 to 0.68 percent of the picture, with 4566 of 4864
frames still carrying a text mask. Which part of that drop is false lines
that no longer reach `text_min_run` and which part is real lines left
uncovered on a frame between two detections has not been measured; a
contact-sheet pass on this output would say. The
plate mask falls from 0.12 to 0.004 percent and from 280 frames to 7, which is
the aspect rule removing the bins: 12 plate boxes are left. The audit
verifies (4867 lines).

## Text false positives

Package T13c, for defect 3 of the field-test pilot
(`docs/privacy_module_fieldtest.md`): most of the text mask was on things
rather than writing. `text.false_text` now screens each frame's quads after
the two scales are merged and before `text.hold`, by rules that each have a
setting and an off value.

### The labelled set

21 frames decoded from the pilot fixtures (kept outside git, with the
scripts, under `FaceBlur-work\privacy_fieldtest\t13c\`): FE left 0, 251, 754,
1005, 1256, 1507, 2009, 2261; FE right 2261, 2512; 9K left 0, 251, 501, 1253,
1503; AM left 884, 1326, 1768, 2210, 2653, 4421. Every box the shipped
detector finds on them (72) was drawn and labelled once by eye
(`labels.json`):

- 14 real text regions: five TOTAL drawer labels and a TOTAL package (FE
  754), a "2" on the tube in the wearer's hand (FE 2261), the label on the
  extinguisher bucket (9K 0, 251), the EXIT sign (AM 884), the wearer's
  watch (AM 1326, 1768, 2210), wall lettering (AM 2210).
- 49 false boxes: rags (5), cups, bottles and the sink drain (11), metal
  plates and brackets (8), ceiling lights (3), pegboard holes (9), door
  glass (4), a tool case, a charger, a phone, a clip, a tape reel and other
  parts.
- 7 unsure (printed packages, a bag, a battery sticker, the door glass with
  lettering on it), counted in neither.

A real region counts as covered when one kept quad's box covers at least
half of it. A false box counts as removed when no kept quad's box covers
half of it. Real text the detector never finds (the battery label the
wearer reads in AM 1326 and 1768, the bin labels in 9K 1253) is out of
scope for this measurement: no screening rule can bring it back.

### Per frame, each rule alone

The real `quads_from`, `merged` and `false_text` on the frames' cached
probability maps (`measure.py`, `frame_results.json`):

| rule | real covered | false removed |
|---|---|---|
| none (as shipped) | 14/14 | 0/49 |
| `text_box_thresh` 0.7 | 10/14 | 23/49 |
| shape 1.5 / 60 px, bands exception | 14/14 | 10/49 |
| shape 1.5 / 60 px, no exception | 14/14 | 12/49 |
| shape 1.5 / 50 px, bands exception | 14/14 | 11/49 |
| shape 1.5 / 45 px, bands exception | 14/14 | 15/49 |
| **shape 1.5 / 40 px, bands exception** | **14/14** | **16/49** |
| shape 1.5 / 40 px, no exception | 14/14 | 20/49 |
| shape 1.5 / 30 px, bands exception | 14/14 | 19/49 |
| shape 2.0 / 40 px, bands exception | 14/14 | 21/49 |
| shape 2.0 / 30 px, no exception | 14/14 | 31/49 |
| brightness 240 | 14/14 | 0/49 |
| brightness 200 | 14/14 | 2/49 |
| brightness 180 | 14/14 | 5/49 |
| brightness 150 | 13/14 | 8/49 |
| ink 0.26 or lower | 14/14 | 0/49 |
| ink 0.70 | 12/14 | 13/49 |
| ink 0.75 | 10/14 | 27/49 |
| ink 0.80 | 6/14 | 42/49 |

`text_box_thresh` 0.7 loses the "2", the extinguisher label, a watch and
the wall lettering, all scored 0.60 to 0.69. The ink rule
(`text_ink_min`, the share of the marked region at or above `text_thresh`)
does not separate: the rags are as solid in the DB map as the words (0.60
to 0.84 against 0.67 to 0.85), and every threshold that removes a false
box loses a real one. Both are off.

The shape rule reads the quad's own sides, not its upright box, so a tilted
line is not a square. Its thresholds were not taken at the table's
maximum. The unclip grows both sides by the same distance, which squeezes
the aspect: a word twice as long as it is tall comes out at 1.48, three
times as long at 1.9. Aspect 2.0 would drop a three-letter word seen up
close, and the EXIT sign here is only 2.4 times closer than that. At 1.5
only one- and two-character marks are at risk. The short-side floor of 40
px sits 1.7 times above the largest square-ish real quad in the set (the
"2", 31x23). 30 px removes three more false boxes with a margin of only 1.3.
The stacked-lines exception keeps 4 false boxes. It is kept on, because a
label or a page found as one region is the text the rule exists to mask.
The brightness rule reads the region the map marked (the quad shrunk back
by the unclip; the grown quad takes in the dark wall around a light). Even
so, the lights average only 163 to 212 there. The white watch strap is at
165 to 173, and black print on white paper averages about 195. So 240 stays:
it removes nothing here and only drops a region too washed out to hold
dark ink.

### Over time, with the hold

`text_min_run` can only be measured over time. For each labelled frame the
detector ran on every second frame in a window of 16 frames each side (the
pipeline's `text_stride` 2), then the rules, then `text.hold`. The held
quads at the labelled frame were checked (`temporal.py`,
`temporal_results.json`):

| rules | `text_min_run` | real covered | false removed |
|---|---|---|---|
| none | 2 (as shipped) | 12/14 | 10/49 |
| none | 3 | 11/14 | 11/49 |
| none | 4 | 11/14 | 13/49 |
| defaults | **2** | **12/14** | **24/49** |
| defaults | 3 | 11/14 | 28/49 |
| defaults | 4 | 10/14 | 31/49 |

The shipped hold already loses the EXIT sign and one watch sighting: each
is seen once in its window. The rules lose nothing more. A run of 3 loses a
second watch sighting and a run of 4 also loses the wall lettering, so
`text_min_run` stays 2. The pasted-text test keeps its 6-frame clip.

### Defaults

`text_min_aspect` 1.5, `text_square_px` 40, `text_square_bands` 2,
`text_bright_max` 240, `text_ink_min` 0 (off). `text_box_thresh` stays 0.6
and `text_min_run` stays 2. On the labelled frames this removes 14 more of
the 49 false boxes than the shipped code at no cost in real text. What
is left is the pegboard holes (thin rows, aspect 4 to 7), the door glass,
and small parts. The shape rule cannot see those, and the ink and box
threshold rules cost real text.

### End to end

`python -m faceblur.h5 run AM_032259_L.h5 --workers 2`, same PC and fixture
as the T13a row above:

| run | ms/frame | seconds | detect_s | encode_s | mask_face% | mask_text% | mask_plate% | mask_screen% | text_lines | text_frames | frames with a text mask |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T13a | 152 | 741.3 | 583.71 | 126.82 | 0.76 | 0.68 | 0.004 | 0.27 | 11857 | | 4566 |
| T13c | 150 | 728.2 | 580.02 | 117.45 | 0.76 | 0.44 | 0.004 | 0.27 | 9880 | 4270 | 3684 |

The text mask falls from 0.68 to 0.44 percent of the picture, down 35
percent. `text_lines` falls by 17 percent and frames carrying a text mask
from 4566 to 3684. The rules cost no measurable time (150 ms a frame against
152). Faces, plates and screens are unchanged. Going by the labelled frames,
the false text still masked on this camera is mostly pegboard holes and door glass;
the output has not been checked frame by frame.

## Head backstop

T13d, defect 1 of docs/privacy_module_fieldtest.md: faces at the fisheye rim,
turned 60 to 90 degrees or cut by the frame edge, are not found by YuNet or
CenterFace. The backstop (faceblur/heads.py) reads the COCO class `person`
from the YOLOX screen model, which already runs, so it adds no model and
costs one more decode of the same inference. Each person box gives a head
region, masked as a face polygon (a rectangle, not an ellipse):

- the top `head_frac` (0.22) of the box, its middle `head_width_frac` (0.6)
  across;
- for a box cut by the left or right edge of the frame, the region sits
  against that edge and is as tall as it is wide. The lens tips a person
  leaning in from the rim so his head is in the top corner on the cut side;
- not for the wearer: a box that reaches the bottom band (bottom at 0.85 H
  or lower) and is wider than 0.5 of the frame or starts below 0.5 H. The
  brief's rule (on the bottom edge and wider than half the frame) let
  through the wearer's legs, lap and arms on every pilot frame looked at:
  they are 0.24 to 0.65 of the width, start at 0.55 H or lower and one ends
  at 0.89 H;
- dropped on a frame where a tracked face already covers it (IoU 0.3, or
  the face centre inside the region), checked against the tracker's final
  masks;
- held like plates: `head_min_run` 2, `head_gap` 6, `head_tail` 3,
  `head_link_iou` 0.3;
- left to the handled zone like every face.

It is off on the desktop (`head_backstop` False) and on in the HDF5 `run`
and `selfcheck`. By default (`head_edge_only` True) only a person box cut
by the left or right frame edge gives a head region; see "Decision: edge
only" below.

### The two pilot leaks

| clip, frame | before | after |
|---|---|---|
| AM 032259 frame 2653: colleague's half face at the left edge, about 150x300 px | visible | covered. Person box [0, 20, 706, 921] at 0.73; the region is [0, 38, 509, 509] and hides eyes, nose, mouth and chin; hair above y 38 still shows |
| 9K 000106 frames 1253, 2255, 2095: seated person at the right edge, profile about 70 px | visible | **still visible**. YOLOX finds no person there, not even at conf 0.3. The only boxes on those frames are the wearer's own lap and arms, correctly left out |

So the backstop closes the cut-off face at the rim, but not the
side-on seated person. That one needs a different fix: (b) or (c) of
defect 1, rotated or padded edge bands for the face detector.

A first version with the brief's geometry (centred strip, wearer = wide box
on the bottom edge) did not cover AM 2653: the strip landed on the wall
beside the colleague's head. It also masked the wearer's legs on most
frames: face mask 2.31 percent, 3260 frames carrying a head mask. The
numbers below are for the shipped rules.

### Extra mask

`python -m faceblur.h5 run <fixture> --workers 2`, same PC as T13a/T13c:

| clip | face mask, pilot | face mask, T13d | `heads` | `head_frames` | frames | ms/frame |
|---|---|---|---|---|---|---|
| AM 032259 L | 0.75 % | 1.78 % | 3843 | 1653 | 4864 | 151 |
| 9K 000106 L | 0.17 % | 0.33 % | 876 | 706 | 2757 | 145 |

T13a had AM at 0.76 percent. The backstop more than doubles the face mask
on AM: head masks are on a third of its frames. It costs no measurable time
(151 ms a frame against T13c's 150).

### False masks seen on the contact sheets

Two sheets a run, 8 frames that carry a head mask, under
`FaceBlur-work\privacy_fieldtest\t13d\`:

- objects on the pegboard read as a person at 0.5 to 0.63: the red bag (AM
  frames 0, 3922, 4252);
- the wearer's arm or hand reaching up at the right edge, which starts high
  so the wearer rule keeps it (AM 2960, 4863; 9K 2756);
- torsos of people cut by a side edge with their head out of frame, where
  the edge-anchored square lands on a shirt (AM 2165, 2576);
- a loose box whose top strip lands above a head the face tracker already
  masks, on the wall (AM 4252; 9K 0, 106, 283);
- a pink cushion (9K 432).

Real heads the face detectors did not mask that the backstop did: AM 2653,
distant seated people on 9K (816, 1667, 1906; small, partly covered) and a
person on a ladder (AM 4863).

### What it would add on 004100

`eval.measure` replays cached face detections and cannot see the
backstop, so this was measured directly
(`privacy_fieldtest\t13d\measure_004100.py`). Face masks came from
`batch.run_video(mask=("face",))` on
`footage/ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4`
(workers 2, backstop off, face mask 0.567 %). YOLOX people were read on
every 10th frame, giving head regions under the shipped wearer rule. A
region a face mask on that frame already covers does not count; the rest
are the backstop's additions:

| `head_conf` | head regions | not covered by a face mask |
|---|---|---|
| 0.5 | 163 | 84 (51.5 %) |
| 0.6 | 135 | 70 (51.9 %) |
| 0.7 | 114 | 52 (45.6 %) |

I looked at 20 additions sampled at random (seed 0) at 0.5:

- **20 of 20 are on a real person.** None is a coat, bag or other object.
  This clip is a washroom full of people.
- **11 of 20 regions contain at least part of a head**: hair, a face, a face
  in the mirror.
- **9 of 20 sit on a back or shoulder.** People bent over a sink, or tipped
  by the tilted camera, have their head at the side of the box, not the top.

At 0.7 the same check gives 20 of 20 on people and about 8 of 20 holding a
head. Raising the threshold drops real people and does not move the region
onto the head. **`head_conf` stays 0.5.** The task's rule raises it when
more than half the additions are objects, and none were. The 70 percent
mark is met for "on a real person" (100 percent) but not for "covers the
head" (55 percent). The objects that do occur were on the pilot clip's
pegboard, at 0.5 to 0.63, and a threshold of 0.7 would also drop the
colleague at AM 2653 (0.73) close to the line.

### Decision: edge only

The measurements above show the backstop works for the case it was built
for: a person cut by the side edge of the frame. Everywhere else its head
strip lands on a back, a shoulder or the wall about half the time, and the
face consensus already masks most of those faces. So `head_edge_only`
(default True) makes a head box only from a person box within 1 percent of
the left or right edge. The wearer rule still applies. With
`head_edge_only` False it behaves as in the tables above.

Re-run with the HDF5 defaults (backstop on, edge only), same PC:

| clip | face mask, pilot | all boxes | edge only | `heads` | `head_frames` | ms/frame |
|---|---|---|---|---|---|---|
| AM 032259 L | 0.75 % | 1.78 % | 1.56 % | 1263 (was 3843) | 688 (was 1653) | 151 |
| 9K 000106 L | 0.17 % | 0.33 % | 0.20 % | 51 (was 876) | 35 (was 706) | 145 |

- AM 2653 is still covered: region [0, 38, 509, 509] over the colleague's face.
- These false masks are gone, because the box does not touch a side edge:
  - AM 0, 3922, 4252: pegboard bag; at 4252 also the strip beside a masked head;
  - 9K 0, 106, 283: strip above a seated, already masked head;
  - 9K 432: cushion.
- These false masks remain, all from boxes cut by a side edge:
  - AM 2165, 2576: torso of a person whose head is out of frame (371x371 and 465x468);
  - AM 2960, 4863 and 9K 2756: the wearer's arm or hand at the right edge, reaching up, which starts too high for the wearer rule.
- Lost against "all boxes": the small real heads away from the edge, which
  the face consensus also misses. They are the distant seated people on 9K
  (816, 1667, 1906) and the person on a ladder at AM 4863.

The remaining edge masks are what the AM 0.75 to 1.56 percent buys. The 9K
side-on seated profile stays open. It needs a face pass on rotated or
padded edge bands (defect 1, fixes b and c), not a person box.

### Limits

- A head region is a guess from a box. Bent, seated and sideways people
  often have the head outside the top strip.
- A person YOLOX does not see is not helped at all: the 9K seated profile.
- The wearer's own arm reaching up from the bottom corner still gets a mask.
