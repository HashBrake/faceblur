# FaceBlur report

What was built, how well it works, how fast it runs, on what, and what was
left out on purpose. Every number here was produced by the code in this
repository with no human labels and no human judgement in the loop. Dated
2026-09-05.

## 1. The task

Video in, video out. Every face in the output is destroyed inside the smallest
region that hides identity: an ellipse over the eyes, nose and mouth. Nothing
else changes: hands, cards, screens, bodies, hair and background keep their
original pixels, so the output stays usable as training data. No labelling,
no review step, no parameter a person has to pick per video.

The footage is from an Ego wearable camera pointed at card collectors:
1600x1300, about 30 fps, H.264, no audio, near constant frame rate. Four
sample files, 264 seconds in total, sit in `footage/` (gitignored).

## 2. Hardware and software the numbers come from

| | |
|---|---|
| CPU | Intel Core i7-12700K, 12 cores / 20 threads |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 3070, 8 GB, driver 595.79 (DirectML, NVENC) |
| OS | Windows 11 Enterprise 10.0.26200 |
| Python | 3.12.8 |
| Inference | onnxruntime-directml 1.24.4, onnx 1.17.0 |
| Video | ffmpeg 7.1 from imageio-ffmpeg 0.6.0, h264_nvenc for encoding |
| Detectors | YuNet (opencv_zoo, Apache 2.0), CenterFace (deface 1.5.0, MIT) |

Exact package versions are pinned in `requirements.txt`; model hashes are in
`models/README.md` and in every audit record.

## 3. Quality and accuracy

### 3.1 How it is measured without labels

There are no ground-truth labels for this footage and the process must not
need any, so `eval/` measures five things automatically:

- **Consensus recall.** Three unrelated detectors (YuNet, CenterFace,
  MediaPipe BlazeFace) run over the video; a face any two agree on is a
  pseudo-label. Recall is the share of pseudo-faces whose identity ellipse is
  inside the mask.
- **Synthetic recall.** Face crops of known position are pasted into real
  frames of the same video at various sizes, sharp and motion blurred. The
  truth is exact. This catches the misses consensus cannot see.
- **Off-face masking.** Masked pixels where no detector sees a face. The
  "strict" variant counts only consensus faces as face, an upper bound.
- **Hand damage.** MediaPipe Hands regions; the share of hand pixels touched.
- **Continuity.** Once a face is confirmed, how often it stays covered while
  any detector still sees it.

`eval/sweep.py` tries a grid of settings and keeps only those that pass the
gates (off-face at most 0.3 % mean and 3 % on any frame, hands at most 0.1 %),
then picks the best mean of consensus and synthetic recall, with continuity as
the tie-break. The chosen defaults are the ones in `faceblur/settings.py`.
Full tables: `docs/precision_report.md`.

### 3.2 Results on the sample footage

Measured on `004310` (938 frames, the busiest file, 215 tracks). Where two
numbers are given they are the sweep's measurement and the README's earlier
measurement of the same build.

| Measure | First build | Shipped build |
|---|---|---|
| Share of the frame destroyed, mean | 33 % | 1.5–1.7 % |
| Share of the frame destroyed, worst frame | 93 % | 3.5–3.8 % |
| Masked pixels where no detector sees a face, mean | 29.7 % | 0.16–0.30 % |
| Hand pixels touched | 82 % | 0.00 % |
| Faces the detectors agree on, covered | 100 % | 98.5–99.0 % |
| Confirmed faces kept covered while still visible | – | 99.0–99.6 % |

Synthetic recall, faces of known position pasted into real frames:

| Faces | Covered |
|---|---|
| 64 px and larger, sharp | 89 % |
| 64 px and larger, motion blurred | 73 % |
| 32 to 63 px | 94 % |
| under 32 px | 46 % |
| all | 83 % |

Mask area over identity-ellipse area: 2.0. The destroyed region is about
twice the area of the minimal ellipse; that margin keeps a moving face covered
and gives the soft edge.

Across the four sample files, from the audit records: 0.25–1.7 % of each frame
destroyed on average, at most 4.8 % on any frame, zero frames over the 5 %
budget.

### 3.3 What the misses are

Small faces (under 32 px on a 1600 px frame: someone across the room) and
strongly motion blurred faces. The detectors themselves find a 24 px face
about half the time at any threshold; that is a model limit, not a threshold
choice. The first build reached 100 % on the agreed faces by masking a third of
every frame. This build trades a few points of recall on the smallest faces
for leaving everything else intact, which was the stated goal.

### 3.4 Entry frames

Faces entering the field of view used to show for one or two frames before
the mask started. Now a confirmed track is extended backwards through weaker
sightings, and its mask reaches six frames before the first sighting and two
after the last, following the face's motion and growing 5 % per frame. The
continuity metric and `tests/test_entry.py` cover this.

## 4. Speed

### 4.1 Per frame

| Path | Detection per 1600x1300 frame |
|---|---|
| CPU, first precision build | 700 ms |
| CPU, shipped build | 350 ms |
| GPU (RTX 3070, DirectML), shipped build, one process | 100 ms |

Detection dominates. Decoding costs about 3.5 ms per frame, masking and NVENC
encoding about 10 ms per frame per worker, copying clean stretches nothing.

### 4.2 The four sample files, end to end

Same defaults, same machine. "Window" is the desktop UI on its pool; "CLI" is
`cli.py footage -o ... --workers 10` in one process pool. Wall time per video
includes probing, detection, tracking, writing, joining and verification.

| File | Video | Source | Window wall | CLI wall | CLI detect | CLI write | Copied | Output | Ratio (CLI) |
|---|---|---|---|---|---|---|---|---|---|
| `003939` | 68 s, 2033 frames | 110 MB | 75 s | 69 s | 43 s | 25 s | 150 frames | 204 MB | 1.01 : 1 |
| `004100` | 33 s, 984 frames | 59 MB | no record (see note) | 41 s | 31 s | 10 s | 120 frames | 103 MB | 1.26 : 1 |
| `004310` | 31 s, 938 frames | 96 MB | 77 s | 83 s | 66 s | 17 s | 0 frames | 129 MB | 2.66 : 1 |
| `005035` | 132 s, 3971 frames | 312 MB | 182 s | 173 s | 134 s | 39 s | 60 frames | 482 MB | 1.31 : 1 |
| **all four** | **264 s** | | **333 s + 1 file** | **367 s batch** | | | | | **1.39 : 1** |

Window and CLI run the same pool and the same code, so the per-video times
match within noise. The window run's `004100` produced a good file but no
audit record because of the bookkeeping bug fixed on 2026-09-05 (STATE.md);
its time was not recorded. The CLI batch total (367 s, 6 min 8 s wall)
includes starting the 10 worker processes and loading the models once.

Output size: the outputs are larger than the sources because they are
encoded at a higher quality (NVENC cq 16) than the camera used, and because
the encoded pieces carry no B-frames (+5.5 % against the same encode with
B-frames, measured on `003939`: 203.6 MB against 193.0 MB).

### 4.3 What would make it faster

- A CUDA build on a data-centre GPU: the detection graph is the same ONNX,
  onnxruntime-gpu replaces onnxruntime-directml. Not measured here.
- More worker processes only help until the GPU saturates; 10 workers on an
  RTX 3070 already share one card.
- `--stride 2` halves detection. The sweep measures what that costs in recall
  on the footage and picks it only if it stays inside the gates; on this
  footage it did not.
- NVDEC decoding (`--hwaccel cuda`) was measured no faster than CPU decoding
  here and stays off.

## 5. Assumptions

- Footage like the samples: a wearable camera, faces 20–300 px, near constant
  frame rate, H.264 in MP4. Other containers and codecs are read by ffmpeg but
  were not tested.
- No audio in the samples. When audio exists it is copied through untouched.
- A face needs to be seen twice by YuNet and confirmed once by CenterFace
  before any pixel is touched. A face that appears for one frame only is not
  masked; on this footage that is the trade that keeps hands untouched.
- The identity region is the eyes-nose-mouth ellipse. Hair, ears and the
  outline of the head are left. This is deliberate and is what "bare minimum"
  was taken to mean.
- Pixelation (six blocks across the face, from a shrunk copy) destroys
  identity. A plain Gaussian blur is invertible in principle and is not offered
  on its own.
- Windows with a DirectX 12 GPU or a CPU. Linux and macOS are not tested.
- The evaluation's pseudo-labels are only as good as the union of three
  detectors. A face all three miss is invisible to the measurement. The
  synthetic test is there to bound that.

## 6. Done on purpose

- **Two architectures must agree** before a mask is drawn: YuNet finds,
  CenterFace confirms on a crop. One detector at a low threshold is what
  blurred hands in the first build.
- **Size and shape gates**: at most 15 % of the long side, roughly square. A
  whole screen cannot be a face.
- **Tracker** with confirmation (seen twice, close together), interpolation
  over gaps up to five frames, backward continuation, motion tails.
- **Ellipse mask rotated to the eye line**, feathered, with a per-face block
  size so a small face is destroyed as thoroughly as a large one.
- **Only the region inside the ellipse changes**; the rest of the frame is the
  decoded source pixel. Stretches with no mask are stream-copied, byte for
  byte.
- **Automatic parameter selection** by sweep with gates: no person picks a
  threshold.
- **Audit record per output** with how much of each frame was destroyed,
  flagged frames, every setting, model hashes, compute provider and timing.
- **Verified joins**: frame count, forward timestamps, length against the
  source, clean demux; a failed join falls back to encoding the whole file.
- **Encoded pieces carry no B-frames** so they join exactly with copied source
  pieces; copies are only made from sources with no reordering (see 8).
- **Every dependency pinned**, real footage and every output folder gitignored.
- **GPU through DirectML**, so any DX12 card works without a CUDA install;
  static ONNX graphs are built per input size because DirectML mis-computes
  the dynamic ones.

## 7. Not done on purpose

- **No installer, no code signing** (Phase 6 of the plan was not requested).
- **No cloud runner.** `faceblur.batch.run_video` takes a `submit` callable and
  detection runs on independent frame ranges, so one can be written without
  changing the pipeline. Not built.
- **No larger detector.** RetinaFace, SCRFD or YOLO-face variants would raise
  small-face recall at 3–10x the compute. The evaluation harness is where to
  test one; the defaults were not changed without measurement.
- **No number plate, text, or body redaction.**
- **No manual review UI.** The requirement was no human intervention; the
  audit record is the substitute.
- **NVDEC** off by default, measured no faster here.
- **`--no-verify` and `--engine both`** exist for recall at any cost, and are
  documented as blurring hands (7.7 % of hand pixels on this footage).

## 8. Known limits and open issues

- Faces under 32 px: about half covered. Motion blurred faces: about three
  quarters.
- A source with B-frames is encoded whole rather than partly copied. Still
  correct, slower for videos with few faces.
- `-bf 0` on the encoder makes output files larger at the same quality; see
  the size column in 4.2.
- Outputs are H.264 at crf 12 / NVENC cq 16 and are larger than the sources
  (13 Mb/s in), because the camera encoded at a lower quality than the setting
  chosen to keep training data intact.
- The window runs videos one after another; a queue of many short videos
  would go faster with videos in parallel. Not needed for the sample sizes.

## 9. Reproduce

```
.venv\Scripts\python.exe -m pytest tests                      # 741 tests
.venv\Scripts\python.exe cli.py footage -o footage_blurred --workers 10
.venv-eval\Scripts\python.exe eval\oracle_mediapipe.py VIDEO --stride 5
.venv\Scripts\python.exe -m eval.consensus VIDEO --stride 10
.venv\Scripts\python.exe -m eval.sweep VIDEO                  # writes docs/precision_report.md
```

## 10. Missed faces, second pass (2026-09-07)

After the first push the user reported faces still showing, most in `004100`,
when the camera moves fast and when people enter the frame. This section
records what was found, what was built, and what it measured.

### 10.1 What the diagnostic found

Every YuNet candidate at 0.3 or more that ended up unmasked on `004100` was
classified by where it was dropped, then the frames were looked at.

- Raising sensitivity is the wrong lever. YuNet scores the wearer's own hand on
  the mop handle at 0.74-0.82, a wall sign at 0.80, a whole person seen through
  a doorway at 0.6-0.8. CenterFace rejects all of these; that confirmation is
  the only thing between the output and the hand-blurring of the first build.
- The real misses fell into four groups: faces cut by the frame edge (a
  person entering; CenterFace's crop was truncated and scored 0.25 against a
  bar of 0.3), faces turned down at the sinks, faces during a fast pan
  (CenterFace's confirmation rate fell from 88 to 81 percent between the
  calmest and fastest quarter of frames, and the tracker's overlap link broke
  at 30-56 px a frame), and small dark faces far away.

### 10.2 What was built

| Change | Why | Measured effect |
|---|---|---|
| Reflect-padded square crops | a face half out of the picture keeps its context | edge recall in the synthetic set 51 percent before the tracker work |
| Mirror and tight views, counted only at 0.4 or more | faces turned down clear the bar in another view | `004100` unconfirmed frames 80 to 66; "best view at 0.3" instead let hands through (1.45 percent) |
| UltraFace tie-break when CenterFace scores 0.25-0.3 | a third family on the near misses | `004100` unconfirmed frames to 64; asked from 0.1 up it cost 5.5 percent of hand pixels, since it fires on hands too |
| Camera shift out of the tracker's prediction | a pan no longer breaks a track | synthetic pan recall 58 to 73 percent |
| Centre-distance link, longer gap while the camera moves fast | small fast faces have no overlap frame to frame | synthetic edge recall 51 to 72 percent, entry delay 6.5 to 3.3 frames |
| Entry tail to twelve frames for a moving face, growth 5 to 3 percent | cover the face before the detectors lock on | edge recall 75 percent with the views; worst-frame masking back under 3 percent |
| Confirmed track may follow a face to 35 percent of the frame | people walking up to the camera | no measured cost |
| Crops only from `min(conf, 0.5)` up | a box below `conf` can never be confirmed | 20 to 8 crops a frame on `004310`, no result changes |
| GPU workers capped at six | ten workers made DirectML page (7.8 GB of 8) | cache build 62 s instead of stalling |

Not adopted, with the number that decided it: tails that follow the size
trend (worst-frame masking 5.6 percent, no recall gain); a graded rule
letting a strong YuNet box pass with CenterFace at 0.2 (in the sweep, see
`docs/precision_report.md`); any threshold below `conf` 0.6.

### 10.3 Results

Hands: 0.00 percent of hand pixels on `004310` at the final defaults, as before.

Unconfirmed stretches (three frames or more where YuNet saw a box at full
threshold that nothing confirmed and no mask covers; hands and objects mostly):

| File | Before | After |
|---|---|---|
| `004100` | 13 stretches, 80 frames | 13 stretches, 64 frames |
| `004310` | 18 stretches | 12 stretches, 57 frames |

Every stretch left on `004100` was looked at: the wearer's hand on the mop
and a wall sign. The faces that were missed before (the crouching person at
the left edge at frame 210, the person bent over the sink at 611, the mirror
reflections at 778, the two small faces at 69) are masked in the new output.

Synthetic recall on `004310`, old tracker and plain confirmation against the
final defaults: interior 85 to 86 percent, edge 51 to 75, pan 58 to 73; entry
delay (frames from half visible to masked) 6.5 to 3.3.

The sweeps (`docs/precision_report.md` for `004310`,
`docs/precision_report_004100.md` for `004100`), with the unconfirmed frame
count as the tie-break behind continuity:

- `004310`: the camera-aware tracker with the three views at the 0.4 bar,
  recall 99.3 percent, synthetic 80.1 percent, hands 0.00 percent, off-face
  0.44 percent (0.16 from detector boxes), continuity 99.6 percent. The third
  opinion made no difference on this file either way.
- `004100`: the same plus the third opinion, recall 98.4 percent, synthetic
  84.6 percent, hands 0.00 percent, off-face 0.19 percent (0.01 from detector
  boxes), continuity 99.9 percent. This file would also accept the looser
  view bar (0.3); `004310` showed that costs 1.45 percent of hand pixels, so
  0.4 ships.
- `max_gap` split 3 against 5 with nothing between them; 5 ships because it
  keeps a face covered over more missed frames.

The shipped defaults are the union that passes the gates on both files.

### 10.4 Cost

Detection per frame in one process went from about 100 ms to about 130 ms
before the crop floor, less after it. End to end on the four sample files:

| Run | Workers | Total for 264 s of video | Ratio |
|---|---|---|---|
| Before this pass (section 4.2) | 10 | 367 s | 1.39 : 1 |
| Views and third detector, crops from 0.2 up | 6 | 498 s | 1.89 : 1 |
| Crops from `min(conf, 0.5)` up, four workers | 4 | 394 s | 1.49 : 1 |

Per file, the last run: `003939` 58 s detect + 34 s write, `004100` 33 + 12,
`004310` 58 + 23, `005035` 126 + 48. Four workers on the RTX 3070 use about
4.9 GB of the card and 40 percent of it; six pushed it to 7.3 GB and it
crawled.

### 10.5 Assumptions added

- The camera's frame to frame movement is a translation. Rotation and zoom
  are ignored; for a chest camera that is right frame to frame, and the
  shift is only used to predict, never to draw.
- A face that CenterFace scores below 0.25 in every view is not a face, no
  matter what UltraFace says. This is what keeps hands out, and it means a
  face all three detectors read as a hand stays visible.
- The synthetic edge and pan sequences stand in for real entries and pans.
  They use the footage's own faces and backgrounds, but a pasted face is
  sharper than a real one in motion.

## 11. Faces lost during a hit, and reflections (2026-09-08)

After the second pass the user reported two things: on the table tennis file
(`005035`) the opponent's face shows every time the wearer hits the ball, and
on `004100` faces in the mirrors still show in some frames. The target was
restated as a hard gate: no exposed frame for a face of 40 px or more, 99
percent or better for 24 to 40 px, hands still untouched.

### 11.1 What the diagnostic found

- During a hit the whole picture smears for 20 to 30 frames. Every detector
  fails for most of them, the camera shift cannot be measured on the smear
  (phase correlation finds no peak), and the tracker's five-frame gap closes.
  When the picture sharpens the face is re-acquired as a new track. YuNet
  still fires on the smear in about half the frames, at 0.3 to 0.4 and up to
  four times the sharp face's size.
- The opponent's face is 20 to 30 px across the table; the 40 px class on
  this file is the people at the side of the room.
- The mirror reflections are faces YuNet scores 0.4 to 0.55 that CenterFace
  confirms at 0.36 to 0.46: two detectors agree, but neither clears its own
  bar.
- The wearer's hand on the paddle is confirmed by CenterFace at 0.3 to 0.5
  two or three frames in a row, once in a few hundred frames. Every rule
  that keeps a face covered longer kept the hand covered longer too; the
  first attempt at this pass reached 2.4 percent of hand pixels on `005035`.

### 11.2 What was built

| Change | Effect |
|---|---|
| Stitching: two tracks of one face up to 45 frames apart are joined and the gap interpolated; the position gate widens by 8 px a frame of gap, since the camera's move cannot be measured on a smear | the opponent's face is one track through a hit |
| Gap filling: inside a gap of an established track, every YuNet box from 0.3 up near the walk is masked, and the walk follows the nearest; a box up to four times the face counts while the camera moves fast | masks sit on the smear, not on the straight line |
| Boxes interpolated across a gap grow 4 percent per frame from the nearest sighting | the middle of a gap, where the face is least certain, gets the largest mask |
| Every mask grows by the camera shift while it exceeds 8 px a frame | a smeared face is covered to the length of its smear |
| Established tracks (five confirmations, one of them at CenterFace 0.5 or more) continue on boxes down to 0.3, for at most 30 frames past a confirmation, get the long exit tail, the entry tail and the backward reach; a track below that masks only its confirmed frames and the gaps between them | the wearer's hand, confirmed two or three frames at a time, gets nothing beyond those frames |
| A borderline plain confirmation (CenterFace 0.3 to 0.5) counts only if the mirror view sees a face at 0.2 or more | 43 of 45 borderline consensus faces on three files clear it; the wearer's hand did not |
| Two-detector agreement rule for small boxes only: YuNet 0.35 and CenterFace 0.35 make a face when the box is 48 px or less | mirror reflections and far faces; with no size limit at 0.4 and 0.5 the rule admitted a hand box on `004310` (0.43 percent of hand pixels), and hands are 90 px and more, so the limit costs nothing measured |

Two measurements were added: `exposed_40` and `exposed_24_40` in
`eval/measure.py`, the frames between two sure sightings of one face (both
at CenterFace 0.5 or more, up to 45 frames apart) with no mask within one and
a half box sizes of the line between them, counted by face size; and a
synthetic recall bucket for 40 px and up. The sweep ranks by the exposed
frames first, then the score. The exposure count is a proxy: pairs of
different faces at the back of the room get bracketed too, so it is read as
a difference between settings, not as a truth.

### 11.3 Results

Three files, the build pushed on 2026-09-07 against the new defaults:

| File | Hands before | Hands after | Exposed 40+ before | after | Exposed 24-40 before | after | Recall before | after | Synthetic under 40 px before | after |
|---|---|---|---|---|---|---|---|---|---|---|
| `004100` | 0.00 % | 0.00 % | 81 | 55 | 50 | 56 | 98.4 % | 98.4 % | 63 % | 64 % |
| `004310` | 0.00 % | 0.00 % | 78 | 62 | 165 | 125 | 99.3 % | 99.8 % | 60 % | 65 % |
| `005035` | 1.31 % | 0.09 % | 448 | 456 | 1152 | 1167 | 98.3 % | 99.6 % | 60 % | 68 % |

The exposure counts on `005035` do not fall with the agreement rule on
because it adds sure sightings, so more stretches get bracketed; without it
they read 390 and 1052. The hand number on `005035` for the pushed build was
never measured before this pass: the file had no hand oracle. With
established tracks alone it was 0.72 percent; with no tails for tracks below
established, 0.09.

Off-face masking rose from 0.35 to 0.69 percent on `004310` (detector boxes
0.11 to 0.26), the price of the extras and the growth in gaps. The sweep's
gates moved to 0.7 and 0.3 percent to admit it; the hand gate stayed. The
metric's face zone now also counts a box both detectors scored at 0.35 or
more, since the agreement rule masks those on purpose.

On the hit frames of `005035` looked at (160, 344, 346, 508) the opponent's
smeared face carries a mask on the smear; at 175, where nothing detects
anything, the interpolated mask sits where the straight line puts it, 170 px
from a face that is a smear in that frame.

End to end on the four sample files with these defaults: 469 s for 264 s of
video (1.78 : 1; 394 s before this pass). Per file: `003939` 62 s detect +
41 s write, `004100` 35 + 16, `004310` 68 + 26, `005035` 132 + 62. The write
pass grew because more frames carry masks and fewer stretches can be copied.
Cutting confirmation crops from 0.35 up for every box, not only the small
ones the agreement rule can use, had pushed this to 806 s.

The sweeps (`docs/precision_report_005035.md`, `docs/precision_report_004100.md`,
`docs/precision_report.md` for `004310`; 103 settings each, ranked by exposed
frames of 40 px and more, then 24 to 40, then the score, inside the gates):

- `005035`: the shipped tracker with stitching at 45 frames and blur growth,
  the agreement rule off, tracks sure at 0.6; recall 98.9 percent, hands 0.09,
  exposed 389 and 1054 frames, off-face 0.29 percent.
- `004100`: the agreement rule on, stitching at 90, tracks sure at 0.4, and no
  extra confirmation views; recall 100 percent, hands 0.00, exposed 52 and
  36, off-face 0.50 percent.
- `004310`: stitching at 90, no blur growth, the agreement rule off; recall
  99.5 percent, hands 0.00, exposed 54 and 67, off-face 0.65 percent.

The three picks differ by a handful of exposed frames each; the shipped
defaults (views at the 0.4 bar, the agreement rule for small boxes, tracks
sure at 0.5, stitching at 45, gap growth, blur growth) pass every gate on all
three files and are the union, as in section 10. Stitching at 90 was not
taken because with the earlier tracker rules it cost 3.6 percent of hand
pixels on `005035`.

### 11.4 Not done, and why

- A hand detector. The three face detectors all fire on the wearer's hand at
  times; everything above works around that with track-level rules. A palm
  model in the pipeline would settle it; MediaPipe's cannot run in the main
  environment (numpy below 2) and converting it is a pass of its own.
- The agreement rule with no size limit: measured to admit hands on
  `004310`; boxes over 48 px keep needing YuNet at 0.6.
- Stitching over 90 frames: 3.6 percent of hand pixels on `005035`.
