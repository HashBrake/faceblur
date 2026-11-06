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

## Long run (T12b, GreenNode)

The long run processed the selection above in the cloud, with the fixes made
since the pilot. The material is in
`D:\Claude Projects\FaceBlur-work\privacy_fieldtest\cloud\`: `summary.md`,
`out\<entry>\audit.jsonl`, `sheets\` (42 sheets), `maskcheck\`, `logs\` and
`progress.log`. The analysis scripts and the full-resolution crops are in
`cloud\analysis\`: `audit_stats.py`, `audit_stats2.py`, `lookup.py` and
`crop_src.py`. None of the footage is in git.

- What ran: code `02c2e36`, which adds the strides, the plate aspect rule,
  the text false-positive rules, the head backstop for edge persons only,
  and the `zone_kept` split in the audit. The run used default settings and
  `--mask face,text,plate,screen`. Image
  `vcr.vngcloud.vn/110156-faceblur/faceblur-privacy:field`, digest
  `sha256:6b3249056453d935aee28b58077936b322a17ad711150a583ddaeb83e131d64d`.
- Where: VM `faceblur-field-1` (g7-standard-24x64-1rtx5090, HCM03-1B), one
  RTX 5090 on CUDA, `--workers 5`.
- When: created 2026-11-06 11:33, deleted 17:10. Deletion was confirmed at
  17:11.
- Cost: 5.62 h at $0.90, about $5.05. Storage of about 59 GB costs about
  1,660 VND a day.
- Volume: 6 entries, 7 camera streams, 270,760 frames, 9,025 s of camera
  time at 30 fps. `progress.log` reports 308,760 frames, which is a
  summing error; the audits add up to 270,760.
- Results: every entry exited 0 and every audit verifies (18,695 to 75,178
  lines).
- Speed: 52-70 ms a frame end to end per stream, 62.5 ms over the whole run
  (16,925 s of run time). One camera-hour takes about 1.9 hours, against
  7.3 hours in the pilot. The GPU averaged 37 % utilisation (peak 79 %),
  23 GB of memory (peak 29 GB) and 152 W.
- Storage: 24.24 GB in and 35.17 GB out, in `vs:faceblur/fieldtest/`. The
  outputs are 1.45 times the input size, as in the pilot.
- Rerun: the left camera of `9K_000048_LR` was rerun one frame short (37,590
  frames against 37,591 in the source). The cause is the
  single-keyframe-last-GOP defect, which is being fixed as T18.

### Results per camera

| run | camera | frames | ms/frame | mask_face% | mask_text% | mask_plate% | mask_screen% | masked_mean% | zone_share% | heads | head_frames | text_lines | plates | screens | max f/t/p/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FE_124120_L | left | 18692 | 52 | 0.05 | 0.53 | 0.00 | 1.19 | 1.76 | 1.2 | 58 | 73 | 23033 | 13 | 4028 | 6/11/1/4 |
| 9K_024425_L | left | 32861 | 58 | 0.71 | 0.36 | 0.03 | 0.55 | 1.63 | 20.7 | 2614 | 3220 | 23890 | 462 | 4690 | 26/9/2/5 |
| 9K_000048_LR | left | 37590 | 70 | 0.29 | 0.39 | 0.04 | 1.99 | 2.69 | 22.0 | 1535 | 1816 | 41433 | 474 | 11170 | 11/47/2/7 |
| 9K_000048_LR | right | 37584 | 62 | 0.43 | 0.44 | 0.12 | 2.76 | 3.72 | 21.2 | 1613 | 1916 | 61432 | 975 | 13306 | 9/61/2/8 |
| FE_234012_L | left | 39096 | 63 | 1.52 | 0.29 | 0.10 | 1.62 | 3.51 | 12.1 | 9305 | 4267 | 24593 | 1505 | 9862 | 19/10/2/5 |
| AM_041646_L | left | 50163 | 63 | 0.74 | 0.47 | 0.03 | 0.41 | 1.63 | 11.7 | 2952 | 2851 | 75429 | 923 | 4792 | 24/14/2/5 |
| AM_033011_L | left | 54774 | 62 | 0.21 | 0.48 | 0.02 | 1.10 | 1.80 | 7.1 | 953 | 874 | 63975 | 768 | 8598 | 11/40/2/5 |

DirectML and CUDA give the same output. Entries 1 and 2 were also run on the
office PC (RTX 3070, DirectML) at 121 and 174 ms a frame, against 52 and 58
ms on the 5090. On both entries the masked shares agree within 0.02
percentage points, and the head, plate and screen counts within 0.3 %.

**Mask check.** `maskcheck.py` sampled 30 frames that have detections in each
stream. Source and output line up: offset 0 on 29 or 30 of the 30 frames,
and the odd one is off by 1. The share of painted audit boxes whose pixels
visibly changed in the output was:

- text: 31/42 to 77/78
- face: 9/13 to 114/115
- screen: 1/1, but 2/5, 3/13 and 6/18 in three streams

Since T13b, `boxes` lists only what was painted, so the expected share is
100 %. The check counts a box as unchanged when it is mostly flat or dark,
because a black fill on a black shirt or a blur over a dark monitor changes
nothing visible. It also counts boxes clipped to zero area at the frame
edge as unchanged, which causes the `Mean of empty slice` warnings in three
of the files. The check therefore understates the masking. The low screen
counts are mostly blurred, switched-off monitors. The check found no frame
where a painted box was missing from the output.

### Audit statistics

These come from `analysis\audit_stats.py` and `audit_stats2.py`, run over
every frame line. Sizes are the long side in source pixels; frames are
1600x1300. "Backstop-shaped" counts frames that hold a face box of 100 px or
more that is square within 6 % and touches the left, right or top edge. It
is a lower bound for the backstop: the record's `head_frames` is the true
count.

Faces and zone:

| stream | frames with a face | face boxes <24 / 24-40 / 40-80 / 80+ | frames with >=8 faces | frames with text | zone_kept non-empty | head_frames (record) | backstop-shaped frames |
|---|---|---|---|---|---|---|---|
| FE_124120 L | 975 (5.2 %) | 307 / 227 / 460 / 610 | 0 | 16026 (85.7 %) | 717 (3.8 %) | 73 | 62 |
| 9K_024425 L | 15200 (46.3 %) | 38416 / 20259 / 6588 / 5107 | 3278 | 20072 (61.1 %) | 9184 (27.9 %) | 3220 | 1424 |
| 9K_000048 L | 15080 (40.1 %) | 5343 / 5912 / 5153 / 6754 | 10 | 27747 (73.8 %) | 14798 (39.4 %) | 1816 | 580 |
| 9K_000048 R | 13654 (36.3 %) | 5160 / 5044 / 4479 / 5262 | 4 | 31196 (83.0 %) | 16503 (43.9 %) | 1916 | 1038 |
| FE_234012 L | 28900 (73.9 %) | 9406 / 22576 / 24274 / 30980 | 1499 | 23353 (59.7 %) | 8694 (22.2 %) | 4267 | 2650 |
| AM_041646 L | 42139 (84.0 %) | 56664 / 65642 / 42235 / 14281 | 5034 | 36992 (73.7 %) | 16501 (32.9 %) | 2851 | 1047 |
| AM_033011 L | 15451 (28.2 %) | 5125 / 7551 / 10719 / 7702 | 102 | 45993 (84.0 %) | 10458 (19.1 %) | 874 | 317 |

Plates and screens:

| stream | plate boxes <80 / 80-150 / 150-300 / 300+ | mean plate box | frames with a screen | screen area, mean over all frames | screen area when present, median / p90 / max | frames with a screen under zone_kept |
|---|---|---|---|---|---|---|
| FE_124120 L | 0 / 31 / 0 / 7 | 20,183 px2 | 4193 (22.4 %) | 1.10 % | 3.6 / 10.0 / 13.5 % | 85 |
| 9K_024425 L | 10 / 62 / 167 / 561 | 69,800 px2 | 4162 (12.7 %) | 0.50 % | 2.3 / 11.7 / 17.2 % | 1627 |
| 9K_000048 L | 13 / 73 / 215 / 398 | 55,158 px2 | 8060 (21.4 %) | 2.02 % | 8.0 / 18.3 / 41.1 % | 3297 |
| 9K_000048 R | 1 / 166 / 494 / 1584 | 61,681 px2 | 9117 (24.3 %) | 2.86 % | 11.0 / 23.7 / 47.1 % | 2928 |
| FE_234012 L | 0 / 331 / 2943 / 565 | 36,469 px2 | 10121 (25.9 %) | 1.67 % | 6.4 / 11.1 / 25.2 % | 1606 |
| AM_041646 L | 330 / 374 / 875 / 430 | 29,013 px2 | 5674 (11.3 %) | 0.39 % | 2.1 / 8.0 / 18.8 % | 2651 |
| AM_033011 L | 8 / 499 / 998 / 137 | 24,398 px2 | 10915 (19.9 %) | 1.07 % | 4.5 / 10.8 / 39.6 % | 170 |

What the numbers say:

- **Plates.** All the footage is indoors, and no sheet shows a real licence
  plate. Every plate box is therefore false. They are large (mean 20,000 to
  70,000 px2, mostly 150-300+ px), so each one is a big black fill. The
  aspect rule leaves 0-228 axis-aligned boxes per stream taller than 0.8 of
  their width, because the rule applies to the rotated quad, not to the
  box.
- **Screens.** The 1-2.9 % mean screen share is realistic only for the 9K
  office scenes. There the sheets show real monitors and laptops in most
  screen boxes, with only one false box out of every 8-11. Over all 42
  sheets I counted about 16 false screen boxes out of about 47. The false
  ones were the CNC bed, chair backs, a jacket, a freezer, a sink, a clock,
  a keyboard area, a gate, a pegboard and whiteboards. FE_124120 and
  FE_234012 are a workshop and a lab, so their 1.1-1.7 % is mostly false
  boxes. The per-frame sum reaches 41-47 % on 9K_000048, because overlapping
  boxes are summed. Per 2-minute block, screen share swings from 0.0 % to
  13.6 % as the wearer turns towards the desks.
- **Screens kept by the zone.** In 9K_000048 L, 3,297 of the 8,060 frames
  with a screen also have a screen box under `zone_kept`. Defect 2 explains
  why that matters.
- **False face bursts.** In 9K_024425, 3,278 frames have 8 or more face
  boxes, while the sheets show at most 3-4 real people. These frames are
  defect 7.

### Defects found by eye

I looked at all 42 sheets: 30 regular sheets and 12 flicker sheets. Frame
numbers are audit indices. Sizes are in source pixels. A sheet halves the
source, and a flicker tile shrinks it 2.67 times.

The outputs are not stored locally, so a crop at full resolution shows the
source with the audit's painted boxes drawn on it. Only FE_124120 and
9K_024425 have local fixtures. Each finding is marked:

- **[full-res]**: confirmed on a full-resolution crop in
  `cloud\analysis\crops\`.
- **[audit]**: confirmed from the audit lines with `lookup.py`.
- **[sheet]**: rests on the sheet alone.

Defects are ranked as in the pilot: privacy leak > wrong mask inside the
zone > false mask > flicker > audit > speed.

#### 1. Leak: faces turned sideways at the fisheye rim are still not masked

- Evidence:
  - **[full-res]** `9K_024425_L_left_00.jpg`, frame 8962
    (`crops\9K_024425_L_8962_0_940.jpg`). Two colleagues at the left rim are
    turned about 90 degrees by the lens: one seated face of about 45 px at
    x 0-50, y 960-1010, and a man in a green shirt bending over, face of
    about 70 px cut by the left edge at y 1230-1300. The frame has no face
    box at all.
  - **[full-res]** Frame 0 of the same stream (`crops\9K_024425_L_0_0_700.jpg`):
    the seated person, about 40 px, rotated, unmasked. The same desk is in
    view in many frames of the stream.
  - **[sheet]** `9K_000048_LR_left_02.jpg`, frame 23132: a colleague's face
    cut by the bottom-left edge, about 120x140 px, eye and hair visible. The
    only face box in the frame is on a tape roll (`lookup.py`).
  - **[sheet]** `9K_000048_LR_left_03.jpg` frame 34698 and
    `9K_000048_LR_right_03.jpg` frame 34692: people seated at the top rim,
    heads of about 40-50 px, f0.
  - **[sheet]** `AM_033011_L_left_00.jpg`, frame 8648: far seated people of
    about 20 px, f0.
- Likely cause: unchanged from the pilot. YuNet and CenterFace miss faces
  rotated 60-90 degrees. The T13d backstop only fires when YOLOX finds the
  `person`, and YOLOX also misses rotated, cut-off people. Where it does find
  them, it works: AM_041646 frame 47375 and FE_234012 frame 24058 have
  their rim persons masked.
- Proposed fix: pilot fix (b), running the face detectors on the left and
  right 25 % bands rotated by +-90 degrees, with the hits mapped back. This
  costs about 2 x 45 ms single-process on the bands. If that is too slow,
  try fix (c), the reflected border.
- Test: a fixture frame with a face rotated 90 degrees and half outside the
  left edge of a 1600x1300 frame. The output must change the face pixels.
  An upright face in the same frame must still be masked once.

#### 2. Leak: real screens left readable, kept by the zone or not detected

- Evidence:
  - **[full-res]** `FE_124120_L_left_02.jpg`, frame 18691
    (`crops\FE_124120_L_18691_550_200.jpg`): a MacBook showing CAD work,
    about 220x235 px, sits under `zone_kept` (`[1083, 396, 218, 235]`) and
    is not blurred. The wearer's right hand is near it but is handling an
    insole, not the laptop.
  - **[audit]** Frames that have a screen box under `zone_kept`: 3,297 in
    9K_000048 L, 2,928 in 9K_000048 R, 2,651 in AM_041646, 1,627 in
    9K_024425 and 1,606 in FE_234012.
  - **[audit]** In `9K_000048_LR_right_flick_00.jpg`, frames 31234 -> 31236,
    the same screen box moves from `boxes` to `zone_kept`. Defect 9 covers
    this blink.
  - **[sheet]** In `9K_000048_LR_left_02.jpg` frames 28915 and 31806,
    `9K_000048_LR_right_02.jpg` frames 28910 and 31801, and
    `9K_000048_LR_right_flick_01.jpg` frames 31845-31847, a large curved
    monitor shows a datasheet with tables, at about 500x450 px. It is never
    in a screen box. On the laptop next to it, the PCB layout is also
    unblurred in frames 31806 and 31801. The same monitor is detected and
    blurred in frames 37589 and 37583, where it is further away.
  - **[sheet]** `9K_000048_LR_left_01.jpg` frame 20240 and `_right_01.jpg`
    frame 20237: a monitor at the left rim showing a picture, not boxed.
  - **[full-res]** 9K_024425 frame 25248
    (`crops\9K_024425_L_25248_0_300.jpg`): of two monitors side by side,
    the second one, about 90x90 px and showing a white page, is not boxed.
- Likely causes:
  - (a) The zone is applied to screens like any other kind. `zone_scale`
    1.5 around a hand that rests near a laptop covers the laptop's screen.
  - (b) Close monitors are probably dropped by `screen_max_area` 0.12, or
    missed because YOLOX sees them at a steep fisheye angle.
- Proposed fixes:
  - Keep a screen under the zone only when a hand box overlaps the screen
    box itself. A zone grown around the hand is not enough.
  - Measure the dropped YOLOX `tv` and `laptop` boxes on frame 31806 before
    changing `screen_max_area`.
- Test: a fixture with a hand box next to, but not on, a screen box. The
  screen must be painted and must not be listed in `zone_kept`. A hand
  overlapping the screen keeps it clear.
- Status: (a) fixed in T19 (`zone_screen_needs_hand`): screens kept by the zone on FE_015252 fell from 396 to 196 (L) and 211 to 174 (R), see `docs/privacy_module_report.md`, "Zone and backstop fixes from the long run". (b) is open.

#### 3. Leak: labels, whiteboards and signs are left readable (pilot defect 2)

- Evidence, all **[sheet]**:
  - `AM_033011_L_left_02.jpg` frame 25945: two shipping labels on cartons,
    "Carton # 7 / Made in China / Reference / Size / Quantity", at about
    280x260 px. They are readable on the sheet and not masked.
  - `AM_033011_L_left_03.jpg` frame 40359: a whiteboard at the top left
    with handwritten tables of numbers, legible on the sheet and not
    masked.
  - `AM_041646_L_left_04.jpg` frames 44588 and 47375: two whiteboards with
    diagrams, about 360x240 px each, not masked.
  - `AM_033011_L_left_00.jpg` frame 0 and `_02.jpg` frame 31711: "PT LAB"
    wall lettering.
  - `FE_124120_L_left_02.jpg` frame 16992: large wall letters.
- Likely cause: unchanged from the pilot (T13e). PP-OCR gives no response on
  the notice or the board. Large wall letters are above the text detector's
  scale.
- Proposed fix: a page and board finder that does not depend on PP-OCR, for
  example a bright quadrilateral with ink strokes. Before building it, ask
  the owner whether whiteboards and room signage must be masked; see the
  Recommendation.
- Test: as in pilot defect 2.

#### 4. Leak, partial: close faces are under-masked

- Evidence:
  - **[sheet]** `FE_234012_L_left_00.jpg`, frame 3007: a colleague in a
    headset, turned down, about 200x250 px. The face box is
    `[237, 460, 172, 206]` **[audit]**, and the ellipse covers the glasses,
    nose and mouth. The cheek, ear and glasses arm at x 50-230 stay visible.
  - **[full-res]** In 9K_024425 frame 25248 the face box (47x74) covers only
    the front half of a profile. The painted ellipse is grown by 0.35, which
    probably covers it.
- Likely cause: the face box covers only the frontal landmarks. For a
  tilted or profile head the ellipse misses the side of the head.
- Proposed fix: for faces of 150 px or more, grow the ellipse to the YOLOX
  head (the top of the `person` box) when one overlaps the face.
- Test: a fixture with a 200 px profile face and a person box; the output
  must change the ear region.

#### 5. Wrong mask inside the zone: items in the wearer's hands get masked, and the zone barely works with gloves in the dark

- Evidence:
  - **[full-res]** `FE_124120_L_left_01.jpg`, frame 6797
    (`crops\FE_124120_L_6797_500_450.jpg`): the gloved left hand holds a red
    and green insole. A text box of 140x142 px on it is painted black, and
    there is no text on it. `zone_kept` is empty.
  - **[sheet]** Also `FE_124120_L_left_00.jpg` frames 0, 1699 and 13593:
    small masks on the tool in the hands.
  - **[sheet]** `AM_033011_L_left_04.jpg` frame 49007: a face box of 90x110
    px on the dustpan the wearer is holding, a black circle.
  - **[sheet]** `AM_033011_L_left_01.jpg` frame 17297: the tray in hand.
  - **[sheet]** `9K_000048_LR_right_00.jpg` frame 2891: the pin header of the
    PCB being held.
  - **[sheet]** `AM_041646_L_left_04.jpg` frame 50162: a 435x460 screen blur
    over the gate the wearer is moving.
  - **[audit]** FE_124120 has `zone_share` 1.2 % and `zone_kept` in 3.8 % of
    frames, against 16-18 % for the FE pilot. The wearer works in white
    gloves under a single bright lamp.
- Likely cause: the palm and landmark models find gloved, overexposed hands
  poorly, so there is often no zone. Plain detector false positives on the
  object then get painted.
- Proposed fix: measure HandFinder recall on 50 FE_124120 frames with gloved
  hands. If recall is low, add a fallback zone from the wearer's forearms
  entering the bottom of the frame. Separately, the shape rule from T13c
  should also drop a text quad of 140x142 that holds no stacked lines.
  Check why it passed here.
- Test: frame 6797 as a fixture, kept outside git: no mask on the insole.

#### 6. False mask: the head backstop paints large squares on torsos, legs and the wearer's own hands

- Evidence:
  - **[sheet + audit]** `AM_033011_L_left_03.jpg`, frame 34593: a face box
    of 441x441 at the top-left corner covers a standing person's trousers
    and a chair. The head is out of frame.
  - **[sheet + audit]** `FE_234012_L_left_02.jpg`, frame 30073: a 364x362
    box covers a colleague's striped torso. The head is above the frame.
  - **[sheet + audit]** The wearer's own hand or arm, close to the lens:
    - `9K_000048_LR_left_flick_00.jpg` frames 27562-27564: 433-446 px.
    - `AM_033011_L_left_04.jpg` frame 54773: 412 px over the right hand.
    - `FE_234012_L_left_03.jpg` frame 39095: 290 px.
  - **[audit]** In 9K_000048 L frames 2653-2655, a 158x170 box at the right
    edge sits on the wearer's hand holding the soldering iron. It alternates
    `boxes` -> `zone_kept` -> `boxes`.
  - The backstop-shaped boxes are mostly over 300 px: 1,594 of 2,684 in
    FE_234012 and 805 of 1,064 in 9K_000048 R.
- Likely cause: a person box cut by the top or side edge has its "head" band
  taken from the top of the visible part, which is a torso or legs when the
  head is outside the frame. YOLOX also labels the wearer's own forearm
  near the lens as `person`. The `head_wearer_*` exclusion only covers the
  bottom-centre region.
- Proposed fixes:
  - Fire the backstop only when the person box is cut by the side edge and
    its top edge lies inside the frame, so the head is visible.
  - Drop person boxes that overlap a hand or arm the zone sees.
  - Cap the backstop box at the `head_width_frac` of the person's width, as
    measured below the cut.
- Test: a fixture where a person box starts at y=0 (cut at the top). No head
  mask may appear. A person box cut at the left edge with its top inside the
  frame still gets one.
- Status: fixed in T20 (`head_top_margin`, `head_hand_iou`), covered by unit tests. The size cap is not done. FE_015252 has no such case (heads unchanged), so a long-run rerun is still needed to measure it.

#### 7. False mask: bursts of false faces on pegboards, clutter and motion blur

- Evidence:
  - **[full-res]** 9K_024425 frame 23898
    (`crops\9K_024425_L_23898_300_100.jpg`): 14 face boxes. Three are real
    (two seated people and a man with a raised arm); 11 are tools and parts
    on the pegboard, 40-60 px each, painted as black circles.
  - **[audit]** 3,278 frames of this stream have 8 or more face boxes.
  - **[full-res]** Frame 11949 (`crops\9K_024425_L_11949_1200_120.jpg`): a
    227x238 face box on a power strip and pegboard at the right edge. There
    is no person.
  - **[sheet]** False faces also appear on:
    - a tape roll: `9K_000048_LR_left_02.jpg` frame 23132, 96x108 px
    - a mug: `AM_033011_L_left_01.jpg` frame 20180, 88x103 px
    - a bench grinder: `FE_234012_L_left_03.jpg` frame 36088
    - the wearer's own forearm: `AM_041646_L_left_02.jpg` frames 22294 and
      27868
    - six blobs in a motion-blurred frame: `AM_041646_L_left_02.jpg` frame
      27868
- Likely cause: pilot defect 4 is not fixed, and is worse on cluttered
  pegboards and blurred frames.
- Proposed fix: pilot defect 4's fix, a higher confirmation score for a
  candidate with no landmark layout and no person box around it. Also
  suppress new face tracks in frames whose global motion exceeds
  `fast_shift`.
- Test: frames 23898 and 11949 as fixtures, kept outside git. At most the 3
  real faces may be masked in 23898, and none in 11949.

#### 8. False mask: non-screens blurred as screens, and plates that are never real

- Evidence, screens:
  - **[full-res]** 9K_024425 frame 20911
    (`crops\9K_024425_L_20911_700_0.jpg`): an 822x290 screen box over the
    pegboard.
  - **[sheet]**
    - the CNC bed: `FE_234012_L_left_00.jpg` frames 0 and 6015
    - chair backs: `FE_124120_L_left_01.jpg` frames 8496 and 10195
    - a jacket: `9K_000048_LR_left_03.jpg` frame 34698
    - a freezer: `AM_033011_L_left_00.jpg` frame 2883
    - a sink and a clock: `AM_041646_L_left_01.jpg` frames 11147 and 13934
    - a keyboard area: `9K_000048_LR_right_02.jpg` frame 31801, 390x521
- Evidence, plates:
  - **[sheet]** `9K_000048_LR_right_01.jpg` frame 11564: a 347x246 plate box
    on the pegboard at the top edge, filled black.
  - **[sheet]** `AM_033011_L_left_02.jpg` frame 23062: a plate box on the
    wearer's own shirt print.
  - **[sheet]** `FE_234012_L_left_flick_00.jpg` frame 25247: a 318x168 plate
    box on the yellow fence.
  - **[audit]** The plate table above: every plate box is false, they are
    large, and 0-228 per stream are portrait-shaped.
- Likely cause:
  - Screens: YOLOX `tv` fires on large dark rectangles at the rim (pilot
    defect 6).
  - Plates: the LPD model fires on printed patterns and dark panels. The
    aspect rule checks the quad, but the painted box is the quad's
    bounding box.
- Proposed fixes:
  - Plates: apply `plate_aspect_max` to the axis-aligned box as well, and
    reject plate boxes over 300 px long.
  - Screens: require a screen box to contain a bright or textured inner
    rectangle, or an edge-to-interior contrast step (pilot defect 6). A
    flat-dark test would drop switched-off monitors, which is acceptable.
- Test: a unit test giving the plate filter a quad whose bounding box is
  portrait; it is dropped. A fixture with the pegboard frame 20911: no
  screen mask.

#### 9. Flicker: masks blink when the zone toggles

- Evidence:
  - **[audit]** 9K_000048 L frames 2653 -> 2654 -> 2655: a head mask shows,
    then disappears, then shows again, because the box moves into
    `zone_kept` for one frame (defect 6).
  - **[audit]** 9K_000048 R frames 31234 -> 31236: a 303x412 screen blur
    disappears in the same way.
  - **[sheet]** `9K_000048_LR_left_flick_00.jpg` frames 16968-16970 and
    25247-25249: small text masks on the bench and on the wearer's shirt
    print come and go.
  - **[sheet]** `9K_024425_L_left_flick_00.jpg` frames 27562 -> 27563: a
    small blurred patch appears at the bottom in a frame where the audit
    counts are all 0. I could not check this without the output, so it is
    not verified.
  - No real face blinked in any flicker sheet. Close faces in FE_234012
    2653-2655 and 16968-16970 and AM_041646 16968-16970 stay masked in all
    three frames.
- Likely cause: the zone runs every 2nd frame (`zone_stride` 2), and its
  decision is applied per frame without hysteresis.
- Proposed fix: hold a zone decision for `zone_memory` in both directions,
  so that a box kept clear, or painted, changes state only after 2
  consecutive frames.
- Test: a fixture where the zone covers a box on frames 0, 2 and 4 but not
  on 1 and 3. The box's state must not alternate.
- Status: the stride gap is fixed in T21 (hands carried over skipped frames): on FE_015252 R, screen flips fell from 37 to 12, and face flips fell from 9 to 6 or 7. L screen flips stayed at 35. Those come from the hand detector changing between looked-at frames, and hysteresis is not added.

#### 10. Audit and run records

- **[audit]** `progress.log` reports 308,760 frames. The audits add up to
  270,760.
- **[audit]** `maskcheck` warns `Mean of empty slice` on boxes clipped to
  zero area at the edge. The audit can carry such boxes for clipped person
  heads. Clamp them, or drop zero-area boxes before writing the audit.

#### 11. Speed and cost

- **[audit]** 62.5 ms a frame end to end with 5 workers on one RTX 5090. One
  camera-hour costs about 1.9 GPU-hours, about $1.70 at $0.90/h.
- The GPU averaged 37 % utilisation, so the run is limited by the CPU or by
  the pipeline, not by the GPU.
- More workers or a 2-GPU flavour are the next lever; this was not measured
  here.

#### Cosmetics

- No colour shift or frame offset was seen. Frame counts match, except the
  T18 frame noted above.
- Text masks on the wearer's own shirt print and watch are painted, mostly
  invisibly (black on black). See `AM_033011` and `AM_041646`. These are
  harmless, but they add to the text counts.

### What improved since the pilot

- **Speed.** 242 ms a frame in the pilot. Locally the strides gave 121-174
  ms, and the cloud gave 52-70 ms a frame. The long run needed 4.7 h of
  run time, against the 11.7 h estimated for the PC.
- **Text false positives (pilot 3), reduced.** The mean text mask fell from
  0.85-1.29 % of the frame to 0.29-0.53 %. The rags, cups and tool cases are
  gone from the sheets. False text squares of 150-230 px remain on parts
  bins (`9K_024425_L_left_00.jpg` frame 5975, `_02.jpg` frame 29873), on
  cardboard, on a plant, on chair legs, and on held items (defect 5).
- **Plates (pilot 5), reduced.** The mean plate mask fell from 0.04-0.32 %
  to 0.00-0.12 %. False plates remain (defect 8).
- **Edge faces (pilot 1), partly fixed.** The backstop masks real rim
  persons in AM_041646 frame 47375 and FE_234012 frame 24058. Rotated rim
  faces are still missed (defect 1), and the backstop adds defect 6.
- **Audit (pilot 7), fixed.** `boxes` lists only what was painted, and
  `zone_kept` lists what the zone kept clear. Every finding above could be
  checked against it.
- **Flicker (pilot 8), mostly gone.** No real face blinked. The remaining
  blinks come from the zone (defect 9).
- **Robustness (pilot 10).** No input in this run was corrupt. Seven
  streams of 10-30 minutes each ran without a failure.
- **Not improved:** whiteboards and notices (pilot 2, now defect 3), objects
  masked as faces (pilot 4, now defect 7), and walls masked as screens
  (pilot 6, now defect 8).

### Recommendation

Fix before the partner integration, in this order:

1. **Defect 2 (a): stop the zone keeping screens unless a hand touches the
   screen box.** This is a small change with a clear test, and it closes a
   leak verified at full resolution.
2. **Defect 1: rotated band face detection at the left and right rims.** It
   costs speed, but it is the leak seen most often. Measure the cost on
   the 9K_024425 fixture, which is local.
3. **Defect 6: limit the backstop to person boxes whose top is in the
   frame, and exclude the wearer's arms.** It removes the largest false
   masks, up to 441x441 px.
4. **Defect 9: zone hysteresis.**
5. **Defect 8, plates:** check the plate box shape, and cap plate box size.
6. **Defect 5:** check zone recall with gloves before changing anything.

Defects 7 and 8 (screens) are false masks. They cost picture area, but they
do not leak anything. Do them after the items above.

Questions for the owner:

- Must whiteboards, carton labels and room signage ("PT LAB") be masked?
  PP-OCR does not see them, so this needs a new page and board finder.
- Should the wearer's own clothing print (for example "GO EVERYWHERE WITH
  WHEREVER ATELIER", "GREEN") and their watch face be kept or masked? This
  run masked them as text.
- Is a blurred switched-off monitor or TV acceptable? This run blurred them.
- Does the partner need faster than about 1.9 GPU-hours per camera-hour? If
  so, measure a 2-GPU flavour or more workers next. The GPU sat at 37 %.
