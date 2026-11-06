# Privacy module field test: pilot on real footage (T12a)

This is the first run of the HDF5 privacy module (`python -m faceblur.h5`) on
the real head-camera recordings in `D:\Player One Internal videos`, ahead of
the 2-3 hour long run. It checks that the tool runs end to end on real files,
measures where the time goes, and lists, from looking at the output, what the
module gets wrong. The long run comes after the fixes below.

- Date: 2026-11-06. The pilot runs took 03:46-04:41.
- Machine: Windows 11, 32 GB RAM (about 13 GB free during the run), RTX 3070
  8 GB shared with the desktop, DirectML (every model reported
  `DmlExecutionProvider`), `--workers 2`.
- Code: `b447d72` (faceblur 0.2.0), default settings, `--mask face,text,plate,screen`.
- None of the footage is in git. Fixtures, outputs, contact sheets and
  scripts are in `D:\Claude Projects\FaceBlur-work\privacy_fieldtest\`.

## Selection and why

About 6 minutes of recording across 4 camera streams, one session per device,
plus two broken files:

| Entry | Source | Cameras | Length | Why |
|---|---|---|---|---|
| `FE_015252_LR` | FE `ego_AZER76400FE_19700101_015252` | left + right (`rgb_ego_center_left,rgb_ego_center_right`) | 92 s, 2764 frames each | the two-camera case; workshop bench, one far person |
| `9K_000106_L` | 9K `Orbbec_Ego_AZER764009K_19700101_000106` | left | 92 s, 2757 frames | office lab, one seated person, many labelled parts bins |
| `AM_032259_L` | AM `Orbbec_Ego_AZER76400AM_19700101_032259` | left | 162 s, 4864 frames | the most people: a colleague close to the wearer, seated people, a whiteboard |
| `corrupt_AM_103249` | AM `..._19700111_103249_camera_left_part0001.mp4` | left | 24-byte file (only an `ftyp` box) | one of the "Invalid data" files |
| `junk_AM_033011` | AM `._Orbbec_Ego_AZER76400AM_19700101_033011_camera_left_part0001.mp4` | left | 4 KB AppleDouble file | the macOS `._` junk files |

## The driver, and how to resume

The scripts are in `privacy_fieldtest\scripts\`. Run them from the work
folder with the repo's `.venv\Scripts\python.exe`:

- `drive.py selection.json` handles each entry in turn:
  - It packs `in\<name>.h5` if that file is missing. Packing goes through a
    `.h5.part` name, renamed only on success.
  - It then runs `python -m faceblur.h5 run in\<name>.h5 --out out\<name> --workers 2`,
    unless `out\<name>\episode.h5` already exists.
  - It logs one JSON line per step to `logs\drive.jsonl`: name, step, frames,
    wall seconds, exit code, start, end, and the last lines of output on
    failure. The full output goes to `logs\<name>.log`.
  - A failed pack is logged and the entry is skipped. The driver never stops
    on one entry, and writes `logs\DONE` at the end.

  **To resume, run `python scripts\drive.py selection.json` again.** Finished
  entries are skipped.
- `summarise.py [--md]` prints the per-camera table below from every
  `out\*\audit.jsonl`.
- `sheets.py <name>` makes the contact sheets:
  - It remuxes each input and output camera to mp4 with
    `h5io.read_video_group`, into `out\<name>\view\`.
  - It samples 40 frames per hour of footage (at least 12) evenly.
  - It writes `sheets\<name>_<camera>_<k>.jpg`, 4 rows a sheet. Each row has
    the source on the left, 800 px wide, with the audit boxes drawn (face red,
    text yellow, plate magenta, screen cyan) and the counts printed. The
    output is on the right at the same width.
  - `sheets\<name>_<camera>_flick_<k>.jpg` holds 3 consecutive output frames
    at 5 random points per camera, for checking flicker.
- `crop.py <name> <camera> <frame> x y w h` writes a full-resolution
  source | output crop, used to confirm a finding from a sheet.
- `maskcheck.py <name> <camera>` takes 30 frames that have detections. It
  reports whether `out[i]` lines up with `src[i]` (the best offset in -3..3)
  and how many audit boxes are visibly changed in the output.
- `profile.py <mp4>` produces the stage profile below.

## Results per camera

All five audits verify (`python -m faceblur.h5 verify`: 5532, 2760 and 4867
lines ok), and every run exited 0. Masked shares are mean percent of the
picture over all frames. `zone_prevented%` is `masked_in_zone_prevented`.

| run | camera | frames | seconds | ms/frame | mask_face% | mask_text% | mask_plate% | mask_screen% | zone_share% | zone_prevented% | tracks | text_lines | plates | screens | plate_frames | max f/t/p/s | detect_s | encode_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9K_000106_L | left | 2757 | 680.0 | 247 | 0.17 | 1.12 | 0.32 | 1.75 | 9.5 | 0.201 | 43 | 12391 | 452 | 544 | 572 | 7/49/3/2 | 572.62 | 89.15 |
| AM_032259_L | left | 4864 | 1227.5 | 252 | 0.75 | 0.85 | 0.12 | 0.27 | 9.9 | 0.279 | 93 | 23681 | 167 | 730 | 280 | 10/62/2/3 | 1043.49 | 146.23 |
| FE_015252_LR | left | 2764 | 636.5 | 230 | 0.08 | 1.28 | 0.04 | 0.30 | 18.5 | 0.538 | 37 | 11624 | 125 | 308 | 175 | 5/15/2/2 | 561.51 | 56.85 |
| FE_015252_LR | right | 2764 | 631.5 | 228 | 0.07 | 1.29 | 0.05 | 0.43 | 16.3 | 0.510 | 34 | 11149 | 169 | 289 | 258 | 4/16/2/2 | 552.75 | 61.28 |

Over the whole pilot: 13,149 frames in 3,187 s of wall time, or 242 ms a
frame. Each output `episode.h5` is 1.36-1.53 times the size of its input,
because the video is re-encoded.

`maskcheck.py` was run on all four streams. Source and output frames line up
(offset 0 on 30 of 30 sampled frames in each stream). In the output, 62-74 %
of the audit's text boxes and 74-90 % of its face boxes are visibly filled.
The rest are inside the handled zone. The zone keeps them clear on purpose,
but the audit still lists them as masks (defect 7).

## Stage profile

`scripts\profile.py` was run on 60 frames evenly sampled from the AM 032259
left file, in one process on DirectML. Each stage ran alone on every frame
after two warm-up calls. The table gives the mean ms per frame:

| Stage | ms/frame (1 process) |
|---|---|
| `DetectorBank.detect_raw` (YuNet + CenterFace + UltraFace, faces) | 44.7 |
| `ScreenDetector.detect` (YOLOX) | 5.5 |
| `TextDetector.detect` at (1920, 2560) | 159.9 |
| `TextDetector.detect` at (1920,) (comparison only, not in the sum) | 75.0 |
| `PlateDetector.detect` at 1280 | 111.8 |
| `zone.HandFinder.detect` | 78.1 |
| `redact.redact` with the frame's own detections | 15.5 |
| `motion.downscale` x2 + `estimate_shift` | 5.2 |
| cv2 sequential decode | 4.7 |
| **Sum of the stages the pipeline runs** | **425.4** |
| Encode, from the pilot records (`encode_seconds / frames`) | 20.6-32.3 (end to end, 2 workers) |

The profile matches the pilot. 425 ms of detection work per frame, split
over 2 workers, is 213 ms. The run records give `detect_seconds / frames`
of 203-215 ms, and encoding adds 21-32 ms, which makes the 228-252 ms end to
end. Text takes 38 % of the work, plates 26 %, the zone 18 % and faces 11 %.

Two numbers differ from `docs/privacy_module_report.md`. The plate model took
112 ms here, against 63 ms there, and text at (1920,) took 75 ms against 58.
Both were measured on their own while only the desktop was using the GPU. On
the 60-frame sample, text at (1920,) found 130 lines, against 250 at both
scales.

## Robustness: the corrupt file and the `._` junk files

- `pack` fails on both files, the 24-byte one and the 4 KB `._` one. It
  exits 1 with a raw PyAV traceback ending in
  `av.error.InvalidDataError: [Errno 1094995529] Invalid data found when processing input`.
  When run directly with `--out X.h5`, **`pack` leaves a 10 KB `X.h5`
  behind**, holding a camera group that has no attrs.
- `run` on that leftover file exits 1 with an uncaught traceback
  (`KeyError: "Unable to synchronously open attribute (can't locate attribute: 'codec')"`),
  not with the module's own `RunFailed` message. It still writes an
  `audit.jsonl` ending in a `failed` footer, and that audit verifies
  (2 lines).
- The driver logged both entries in `drive.jsonl` with `exit 1`,
  `skipped: true` and the exception text, removed the `.part` file, and moved
  on to the next entry. Nothing crashed.
- Scale of the problem: 10 of the 49 left-camera files cannot be opened (9K
  002148, 003651, 003658, 003705, 022835; AM 19700101_233326,
  19700110_033156, 19700111_095706, 103242, 103249). There are also 10 `._`
  files beside AM mp4s. The full list is in `logs\durations.txt`.

## Defects found by eye

I looked at every contact sheet (28 in `sheets\`) and at crops where a
sheet was unclear. Frame numbers are indices in the audit and in the
remuxed mp4s. Sizes are in source pixels (frames are 1600x1300). Defects are
ranked by severity: privacy leak > wrong mask inside the zone > false mask >
audit > speed > cosmetics.

### 1. Leak: faces at the fisheye edge, turned sideways or cut off by the frame, are not masked

- Evidence:
  - `AM_032259_L_left_01.jpg`, row 3, frame 2653: a colleague stands next to
    the wearer with his face at the left edge, about 150x300 px and half in
    the frame. Eyes, nose and mouth are plainly visible
    (`crop_AM_032259_L_left_2653_0_0.jpg`). The audit counts no face (f0) and
    the output shows the whole face.
  - `9K_000106_L_left_01.jpg` frame 1253, `9K_000106_L_left_02.jpg` frame
    2255, and flicker frames 2094-2096: a person seated at the right edge,
    turned about 90 degrees by the lens, profile about 70 px, f0, not masked
    (`crop_9K_000106_L_left_1253_1250_950.jpg`,
    `crop_9K_000106_L_left_2255_1250_950.jpg`). The same person is masked
    when facing the camera upright (frames 0 and 251).
- Likely cause: YuNet and CenterFace are trained on upright faces. At the
  fisheye rim, faces are turned 60-90 degrees, and a face cut off by the
  frame edge loses the landmarks the confirmation step needs (9K records
  `unconfirmed_frames` 1256 of 2757). See `faceblur/detect.py`
  (`DetectorBank`, `filter_candidates`).
- Proposed fixes:
  - (a) A backstop from a model that already runs: the YOLOX screen model
    also detects the COCO class `person`. Mask the head of every person box
    outside the zone (the top fifth of the box, grown by `zone_face_scale`),
    and track it like a face.
  - (b) Run the face detector on the left and right edge bands rotated by
    +-90 degrees.
  - (c) Pad the frame with a reflected border before face detection, so a
    face cut off by the edge keeps its shape.
- Test: build a fixture frame from a test face image, rotated 90 degrees and
  placed half outside the left edge of a 1600x1300 frame. The pipeline's
  output must change the face pixels. No real footage goes in the test.
- Status: partly fixed in T13d. A head backstop from YOLOX `person` boxes cut by the side edge of the frame now covers the AM 032259 frame 2653 half face, raising the AM face mask from 0.75 to 1.56 percent. The 9K 000106 seated profile is still visible, because YOLOX finds no person there; it needs fix (b) or (c). See the report, "Head backstop".

### 2. Leak: printed notices and a whiteboard are left readable

- Evidence:
  - `AM_032259_L_left_01.jpg` frames 1768, 2210, 2653 and 3095
    (`crop_AM_032259_L_left_2210_150_100.jpg`), and `AM_032259_L_left_02.jpg`
    frame 4421: an A4 notice hangs on the pegboard with its heading
    bar and lines of print visible. It is never masked.
  - `AM_032259_L_left_02.jpg` frame 4863
    (`crop_AM_032259_L_left_4863_0_150.jpg`): a whiteboard at x 0-400,
    y 300-750 carries handwritten tables of numbers, partly legible at full
    resolution. It is never masked.
- Likely cause: the text detector keeps only lines it is confident about
  (`text_min_px` 12, `text_box_thresh` 0.6). Small, low-contrast print and
  thin marker strokes stay below the box threshold. There is no page or
  board class, so a page is masked only when its lines are. See
  `faceblur/text.py` and the `text_*` settings.
- Proposed fix, either of:
  - A page rule: mask the whole of any bright, flat quadrilateral that holds
    two or more text candidates, counting candidates the box threshold
    dropped. `text_max_frac` already treats pages as a separate case.
  - For whiteboards: inside large, bright, low-texture regions, run the text
    map with `text_box_thresh` lowered to 0.45.
- Test: a synthetic frame with a white board carrying a grid of 14 px
  handwritten-style digits at 30 % contrast. The output must fill the whole
  board.
- Status: not fixed (T13e). The text model gives no response at all on the notice or the whiteboard (the probability map is 0 there, also on 4x upscaled and contrast-stretched crops), so neither the page rule nor the lowered board threshold covers any part of them. No rule ships. A fix needs a page or board finder that does not rely on PP-OCR. See the report, "Page rule".

### 3. False mask: surface texture masked as text, in squares up to 220 px

- Evidence:
  - FE left, `FE_015252_LR_left_00.jpg`: frames 0 and 251, a grey rag on the
    bench, about 180x200 px, filled as a black square or triangle; frame
    754, drawer contents.
  - FE left, `FE_015252_LR_left_01.jpg`: frames 1005 and 1256, the sink
    drain, cups and a bowl; frame 1507, a 220x220 px square over tools on the
    table.
  - FE left, `FE_015252_LR_left_02.jpg`: frames 2009 and 2261, the ceiling
    light at the top left, about 100 px.
  - FE right: `FE_015252_LR_right_02.jpg` frame 2512, the rag, 220x200 px;
    frame 2261.
  - 9K frame 1253: rows of pegboard holes, three boxes up to 263x47 px.
  - AM frame 2653: the door glass.
  - Each camera logged 11-24k text lines in 1.5-2.7 minutes of footage. The
    mean text mask is 0.85-1.29 % of the frame, the most of any kind.
- Likely cause: PP-OCRv3 DB at 2560 px fires on high-frequency texture such
  as cloth, perforated metal and bright highlights, and a `text_min_run` of
  2 frames lets most of it through. See `faceblur/text.py`, `text_sizes`,
  `text_box_thresh` and `text_min_run`.
- Proposed fixes:
  - A shape rule: text lines are long and thin. Drop a single quad with an
    aspect ratio under 1.5 and a short side over 60 px, unless it holds
    several stacked lines.
  - Raise `text_min_run` from 2 to 4.
  - Drop quads that are washed out to white, such as ceiling lights.
- Test: on frames 0, 251, 1507 and 2512 of the FE pilot (kept as fixtures
  outside git), or on a synthetic image of crumpled cloth, no text mask over
  60 px square may appear. Count text masks per frame before and after the
  fix.
- Status: partly fixed in T13c. The shape rule (aspect under 1.5, short side over 40 px, stacked lines excepted) drops the rags, cups, cases and tools; a run of 4 and a box threshold of 0.7 lost real text and were not taken; pegboard holes and door glass remain. See the report, "Text false positives".

### 4. False mask: objects masked as faces

- Evidence:
  - `9K_000106_L_left_00.jpg` frame 501: a plastic bag on a desk, filled
    with a 140 px black circle.
  - 9K flicker frames 1060-1061: the green parts bins.
  - `FE_015252_LR_right_01.jpg` frame 1256: a red object at the left edge,
    about 90 px.
  - `AM_032259_L_left_00.jpg` frame 884: a red object at the bottom right.
  - `AM_032259_L_left_01.jpg` frame 3095 and `AM_032259_L_left_02.jpg`
    frames 3537 and 3979: boxes and tools on the bench, filled with black
    circles of about 100 px.
- Likely cause: a single CenterFace or YuNet hit gets confirmed on a
  coloured blob, and face tracking then carries it forward. See the
  thresholds and `verify` in `faceblur/detect.py`.
- Proposed fix: for a face candidate with no face-like landmark layout (eyes
  above the mouth, inside the box) and no YOLOX person box around it,
  require the higher confirmation score.
- Test: with the bag and the red object as fixtures (outside git), no face
  masks appear, and the upright face in 9K frame 0 is still masked.

### 5. False mask: parts bins and brackets masked as licence plates

- Evidence:
  - `9K_000106_L_left_02.jpg` frames 2004, 2255 and 2505: a portrait box of
    166x282 px over the blue bins, filled.
  - `FE_015252_LR_right_00.jpg` frames 502 and 754: a metal plate on the
    wall, about 190x200 px.
  - 9K logged 452 plate detections in 92 s. No real licence plate appears on
    any sheet; all the pilot footage is indoors.
- Likely cause: `faceblur/plates.py` applies `plate_max_frac` to the width
  only and does not check the aspect ratio.
- Proposed fix: reject a plate quad that is taller than 0.8 of its width.
  Real plates are 2:1 to 5:1 wide.
- Test: a unit test in `tests/` gives the plate filter a portrait quad and a
  4:1 quad. The portrait quad is dropped and the 4:1 quad is kept.

### 6. False mask: a wall and window blurred as a screen

- Evidence:
  - `FE_015252_LR_right_01.jpg` frame 1758: a 450x600 px box covering a
    wall, a door and a whiteboard at the left edge, blurred.
  - `AM_032259_L_left_02.jpg` frame 3979: two boxes at the left edge, one
    on a real monitor and one on the wall around it.
- Likely cause: YOLOX labels large flat panels at the fisheye rim as `tv`.
  The FE box is close to the `screen_max_area` limit of 0.12. See
  `faceblur/screens.py`.
- Proposed fix, either of:
  - Drop screen boxes that touch the image rim and contain no
    high-contrast inner rectangle.
  - Lower `screen_max_area` to 0.08.
- Test: with that frame as a fixture (outside git), no screen mask appears.

### 7. Audit: boxes inside the handled zone are listed as masks

- Fixed in T13b: `boxes` and `counts` now list only what was painted; boxes the zone left clear are under `zone_kept`.

- Evidence: `maskcheck.py` finds that 10-38 % of the audit's text and face
  boxes in each stream have output pixels identical to the source. The crops
  show these are items in the wearer's hand, left clear by the zone as
  designed. For example, in FE left frame 502 the tool in hand carries a text
  box and a screen box and is not masked. The audit line still lists both
  boxes under `boxes`, and the `run_video` docstring says `per_frame_out`
  "receives the masks the copy was written with".
- Likely cause: `_audit_camera` in `faceblur/h5.py` writes every detection in
  `per_frame`, before `redact` removes the ones inside the zone.
- Proposed fix: write zone-suppressed boxes under a separate key
  (`zone_kept`) or with a flag, so `boxes` lists exactly what was painted.
- Test: a test in `tests/` runs a fixture with one detection inside a forced
  zone. The audit's `boxes` must leave it out and `zone_kept` must list it.

No item inside the zone was masked. Everything in the wearer's hands stayed
clear: the glue bottle, cutter, scissors, battery packs, the label being read
in AM 1768, and the heat gun in AM 4421. The wearer's watch (AM frames 0,
884, 1326) carries a text box and is correctly left clear.

### 8. Flicker: short masks blink in and out

- Evidence:
  - FE left flicker frames 1060 -> 1061: two small masks in the sink
    disappear. Frames 165 -> 166: a mask appears on the tool case.
  - FE right 1658 -> 1659 -> 1660: a mask on the bench changes from an
    L-shape to a rectangle, and a new one appears on the drill press arm.
  - 9K 165 -> 166: a mask over the desk disappears. 9K 1061 -> 1062: the
    false face mask on the bins disappears.
  - Every blinking mask was a false one (defects 3 and 4). No real face or
    real text blinked.
- Likely cause: false detections near the run threshold, with a short
  `text_tail` of 3.
- Fix: this should go away with the fixes for defects 3 and 4. No separate
  change is proposed.

### 9. Speed: 228-252 ms a frame

See the Speed section below.

### 10. Robustness: pack leaves a half-written file and run crashes with a traceback

- Evidence: see the Robustness section above.
- Likely cause: `pack` in `faceblur/h5.py` opens the output file directly
  and does not catch `av.error`. `run` does not check that a camera group
  carries the attrs it reads.
- Proposed fixes:
  - `pack` writes to `out.part` and renames it only on success. It catches
    `av.error.FFmpegError` and exits 1 with `<file>: cannot read: <message>`,
    leaving no file behind.
  - `run` raises `RunFailed(..., 2)` when a camera group is missing `codec`.
  - `pack` refuses any file whose name starts with `._`.
- Tests in `tests/`:
  - Packing a 24-byte file that holds only `ftyp` exits 1 with a one-line
    message and leaves no output file.
  - `run` on a hand-made file with an empty camera group exits 2.

### 11. Cosmetics

- No colour shift, no black borders and no frame offset were seen. Source
  and output have the same frame count on all four streams.
- The release file is 1.36-1.53 times the size of the input.
- A note on the contact sheets: a solid black fill on the black pegboard or
  a dark desk cannot be seen at sheet scale. 9K frames 1002, 1253, 1503 and
  1754 looked unmasked on the sheets, but a pixel check shows them filled
  (output mean under 3 grey levels inside the boxes). Run `maskcheck.py` or
  look at a crop before calling something a miss.

## Speed

The pilot measured 242 ms a frame overall (228-252 per stream). At that
rate, one hour of one 30 fps camera takes about 7.3 hours on this PC.

Changes for the long run, with their cost from the profile. Each saving is
the single-process ms divided by 2 workers:

| Change | Setting | Saves | Cost |
|---|---|---|---|
| Run text at both scales on every 2nd frame and let the hold fill the gap | new `text_stride` (2) | about 40 ms | a sign first seen on an odd frame is masked one frame late; `text_gap` 6 covers the rest |
| Run plates on every 3rd frame | new `plate_stride` (3) | about 37 ms | a plate is masked up to 2 frames late; the pilot had no real plates |
| Run the zone on every 2nd frame | existing `zone_stride` 2 | about 20 ms | the zone lags one frame behind fast hand motion; `zone_memory` 3 s already holds it |
| Run text at (1920,) only, instead of `text_stride` | existing `text_sizes` | about 42 ms | 48 % fewer text lines on the sample; small, far text is lost, which makes defect 2 worse |

Proposal: `text_stride` 2, `plate_stride` 3 and `zone_stride` 2, keeping
both text scales. The expected speed is about 250 - 40 - 37 - 20, so **about
155 ms a frame**, or 4.6 hours per camera-hour. Fixing defect 3 means fewer
text quads, which also cuts a little from `redact` and encoding. The two
stride settings do not exist yet, so this needs a code task. `zone_stride`
2 is available now and on its own gives about 230 ms a frame.

## Selection for the long run

The proposal is about 2.2 hours of left-camera recording from all three
devices, plus one left+right pair. It leaves out the two one-hour files:
- 9K `004024` is a 10 GB single file, which would be one long run that cannot
  be split.
- FE `19700104_000732` is only 848 MB for an hour. That bitrate suggests a
  mostly static or dark recording, so it should be looked at first.

| Device | Session | Cameras | Length |
|---|---|---|---|
| 9K | `Orbbec_Ego_AZER764009K_19700101_000048` | left + right | 1253 s |
| AM | `Orbbec_Ego_AZER76400AM_19700101_033011` | left | 1825 s |
| AM | `Orbbec_Ego_AZER76400AM_19700110_041646` | left | 1672 s |
| FE | `ego_AZER76400FE_19700101_234012` | left | 1303 s |
| 9K | `Orbbec_Ego_AZER764009K_19700101_024425` | left | 1095 s |
| FE | `ego_AZER76400FE_19700102_124120` | left | 623 s |

That is 7,771 s (2.16 h) of recording. With the right camera of 000048, it
is 9,024 s (2.51 h) of camera time, about 270,700 frames.

- Compute: **about 18.2 hours** at the measured 242 ms a frame, or **about
  11.7 hours** at the proposed 155 ms.
- Disk: about 15 GB of fixtures and 22 GB of outputs (using the pilot's
  output-to-input size ratio of 1.45). D: has 942 GB free.
- Long sweeps need checkpoints, and the driver already provides them: each
  entry has its own `out\<name>\episode.h5`, and a rerun skips finished
  entries.
