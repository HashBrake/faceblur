# Privacy desensitization module: measurements

Measurements taken while building the privacy desensitization module
(licence plates, text, screens and the rest on top of the face pipeline),
2026-11-05. Each task that needs a number to choose a default writes it here,
with the setup that produced it, so that the defaults in
`faceblur/settings.py` can be traced to something. Later tasks append
sections.

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
