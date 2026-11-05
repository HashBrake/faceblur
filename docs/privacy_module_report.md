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
