# FaceBlur report

What was built, how well it works, how fast it runs, on what, and what was
left out on purpose. Started 2026-09-05; each later pass adds a section rather
than rewriting an earlier one, so a number and the day it was measured stay
together.

Every number here was produced by the code in this repository with no human
labels and no human judgement in the loop, with two exceptions, sections 14.1
and 15.2. Both are the same problem: deciding whether a rule that sets a
detection aside is safe needs to know what those detections actually are, and
nothing in this repository knows. 108 boxes from the output-side check and 85
screen runs were labelled by eye, once, by one person; both sections say so
and report the ones that could not be called. Nothing in the pipeline or in
the rest of the evaluation depends on a label.

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
| Face detectors | YuNet (opencv_zoo, Apache 2.0), CenterFace (deface 1.5.0, MIT), UltraFace (Linzaer, MIT) |
| Screen detector | YOLOX-tiny (Megvii, Apache 2.0), section 15 |
| Hand models | MediaPipe palm detector and hand landmark model (Apache 2.0), section 14 |
| Evaluation only | MediaPipe face oracle, SFace recogniser (Apache 2.0). Neither ships |

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
- **No number plate or body redaction.** Text is not built either, and is
  the piece still owed; screens are built and measured, section 15.
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
- The output-side check holds all four sample files back, correctly: they
  still show faces. Sections 13 and 14.

## 9. Reproduce

```
.venv\Scripts\python.exe -m pytest tests                      # 1084 tests
.venv\Scripts\python.exe cli.py footage -o footage_blurred --workers 10
.venv\Scripts\python.exe cli.py footage -o out --mask face,screen   # section 15
.venv-eval\Scripts\python.exe eval\oracle_mediapipe.py VIDEO --stride 5
.venv\Scripts\python.exe -m eval.consensus VIDEO --stride 10
.venv\Scripts\python.exe -m eval.sweep VIDEO                  # writes docs/precision_report.md
.venv\Scripts\python.exe -m eval.reid VIDEO BLURRED           # section 12
.venv\Scripts\python.exe -m faceblur.verify VIDEO BLURRED --workers 4   # sections 13 and 14
```

### 9.1 What the packaged app carries, and how that is checked

The packaged app is where a promise is easiest to break quietly. A model can
be committed, wired up, tested and still left out of `build\faceblur.spec`,
and the only symptom is a checkbox in someone else's copy of the app that
masks nothing. That is what happened to the hand models and the screen model:
both landed in the code and both were missing from `dist\FaceBlur` for the
five days between 2026-09-05 and this rebuild.

Two tests now stand where memory used to. `tests/test_models.py` asserts that
every `.onnx` in `models/` is either named in the spec's `datas` or listed in
`NOT_SHIPPED` with the reason the build does not need it, and that every one
of them has a pinned sha256. The comments in the spec are stripped before the
search, so a file named only in a comment does not count as shipped.

Three models are in `NOT_SHIPPED`. `sface.onnx` is the face recogniser, which
is evaluation only and has its own test keeping it out. `centerface.onnx` and
`ultraface.onnx` are the static graphs `models/make_dynamic.py` derives the
dynamic ones from; they are committed for provenance and nothing opens them at
run time. `centerface.onnx` was in the spec and was taken out with this pass,
which is 7 MB of a file the code cannot load.

**The packaged app is now the command line as well.** `FaceBlur.exe` with no
arguments opens the window, as a double click always did. With arguments it is
`cli.py`, so a machine with no Python on it can run a batch, and so this check
is a command rather than a person clicking through a window. A windowed build
owns no console, so `faceblur_app._attach_console` takes the shell's own
console or its redirected output before the run starts, and prints nothing if
there is neither.

The check, run on this rebuild:

```
dist\FaceBlur\FaceBlur.exe footage -o footage_blurred_r1 --mask face,screen --workers 4
```

| File | Frames | Screens found | Frames a screen mask reached | Frame destroyed, mean | Worst frame | Over budget |
|---|---|---|---|---|---|---|
| `003939` | 2033 | 30 | 45 | 0.90 % | 4.5 % | 0 |
| `004100` | 984 | 116 | 167 | 0.93 % | 5.8 % | 6 |
| `004310` | 938 | 9 | 11 | 2.33 % | 8.9 % | 24 |
| `005035` | 3971 | 584 | 848 | 0.99 % | 13.3 % | 40 |

Four copies, four records, every one carrying `screens` above zero, in 556 s
for 264 s of video. The `005035` row reproduces section 15.4 exactly: 0.99
percent of the average frame, a worst frame of 13.3 percent, 40 frames over
budget and 848 frames carrying a screen mask. The packaged build and the
source build are the same build.

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

## 12. Is anybody still identifiable (2026-09-08)

Every number before this section asks whether a mask covered a box. None of
them asks the question the work is actually for: after the pipeline has run,
can a machine still tell who these people are. Coverage is a proxy for that,
and a proxy can be right about itself and wrong about the thing it stands
for — six blocks across a face was an assumption nobody had tested.

So this pass measures identity directly, with a face recogniser
(`models/sface.onnx`, SFace, Apache 2.0, the model opencv_zoo pairs with
YuNet; it takes the five landmarks YuNet already produces). It embeds an
aligned face into 128 numbers; two embeddings of one person point the same
way, of two people they do not. `eval/reid.py` is the harness, and it is
evaluation only: the packaged build does not carry the recogniser, and
`tests/test_models.py` fails if it ever does. A tool that redacts faces has
no business shipping a face recogniser.

### 12.1 What a probe can decide, and what it cannot

The leak test embeds a face from the source frame and the same rectangle of
the finished copy. Same frame, same box, same alignment, so the only
difference between the two crops is what the pipeline did. A face it masked
scores what a stranger scores; a face it missed scores near one.

The first run said 1160 of 8060 sightings on `004310` still looked like the
source face, the worst at 0.987. The pictures said otherwise: those faces are
plainly blurred in the copy. The aligned crop around a 16 px face is mostly
hair, shoulders and the room behind, those pixels are identical in both
frames whatever the mask did, and they clear any threshold on their own.

Two controls now decide whether a probe means anything, both against the same
source face:

- **the floor**: the same frame masked here and now, which is what a mask
  this pipeline applies correctly leaves behind. Where the floor alone clears
  the threshold, nothing about that face can be read off the copy.
- **the self match**: the best the same person reaches in any other frame of
  the source, untouched. A face the recogniser cannot match to itself
  anywhere else in the video is a face it cannot identify at all, and whether
  the copy of it scores 0.2 or 0.7 says nothing about anybody's privacy.
  Taking the best of every other sighting makes this a lenient test, since
  the best of dozens of comparisons reaches a stranger's score by itself, and
  that is deliberate: setting a probe aside is the unsafe direction for a
  claim about privacy, so only the clearest cases are set aside. It removes
  five or six percent of them.

`verdict` in `eval/reid.py` applies the two, and the harness reports the set
aside counts beside the leaks rather than hiding them. Each leak also carries
`applied`: how much the copy changed the box against how much the mask
applied here changes it. Near zero means no mask reached that frame; near one
means the mask ran and identity survived it. They are different failures.

### 12.2 Results on the four sample files

**Can the recogniser identify anyone at all.** Two sightings of one track are
the same person by construction; two sightings a frame apart and a box apart
cannot be. The two distributions say where the threshold sits on this
footage, and it is not where the publication puts it.

| File | Same-person pairs | Stranger pairs | Stranger median | Threshold at 1 stranger in 100 | Same people recognised there |
|---|---|---|---|---|---|
| `003939` | 13245 | 1057 | +0.105 | +0.482 | 45 % |
| `004100` | 6984 | 485 | +0.204 | +0.582 | 18 % |
| `004310` | 31233 | 17052 | +0.243 | +0.601 | 11 % |
| `005035` | 60943 | 9457 | +0.230 | +0.568 | 16 % |

`003939` is the file with people at conversational distance, and there the
recogniser works: strangers score 0.105, the same person 0.443, and 45
percent of same-person pairs are recognised at a threshold one stranger pair
in a hundred reaches. On the other three, faces are 16 to 40 px across a room
and it recognises one same-person pair in six or fewer. That is the ceiling
on what a mask can be asked to hide there, and it is the reason section 12.4
does not chase smaller faces.

**Does the mask hide identity.** The sixty largest faces of each file,
embedded from the source frame and again from the same frame after `redact`
has run on it. Same alignment, so the mask is the only difference. Counts are
faces still matched at that file's threshold:

| Mask | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| blur, 4 blocks | 0/60 | 0/60 | 0/60 | 0/60 |
| blur, 6 blocks (shipped) | 0/60 | 2/60 | 2/60 | 0/60 |
| blur, 8 blocks | 0/60 | 2/60 | 2/60 | 1/60 |
| blur, 12 blocks | 10/60 | 1/60 | 5/60 | 5/60 |
| pixelate, 6 blocks | 0/60 | 1/60 | 0/60 | 3/60 |
| solid | 0/60 | 0/60 | 0/60 | 1/60 |

`strength` is how many blocks lie across a face, so 12 keeps four times the
detail of 6. On `003939`, where the recogniser can actually identify people,
6 blocks defeat it on all sixty faces and 12 blocks leave ten recognisable.
The default of 6 was an assumption ("6 leaves nothing to recognise" in
`settings.py`); it is now measured, and so is the fact that the next setting
up would have been a mistake. A wider ellipse helps too — 1.4 x 1.45 leaves
nothing matched on any file — but the shipped 1.1 x 1.15 already does on the
file that matters, at a third of the pixels destroyed.

**What is left in the finished copies.** Every sighting of every track, in
the source and in the blurred copy of that same frame:

| File | Sightings | Crop mostly room | Not identifiable in the source | Can decide | Still matched | Worst |
|---|---|---|---|---|---|---|
| `003939` | 3262 | 190 | 44 | 3028 | 173 (5.7 %) | +0.797 |
| `004100` | 1241 | 48 | 67 | 1126 | 97 (8.6 %) | +0.751 |
| `004310` | 8110 | 576 | 482 | 7052 | 671 (9.5 %) | +0.814 |
| `005035` | 12010 | 608 | 633 | 10769 | 232 (2.2 %) | +0.725 |

An untouched face scores 0.99 against itself. Nothing in any of the four
copies comes near that: the worst is 0.814, and every leak carries `applied`
of 0.56 or more, which is to say the mask ran on every one of them. The row
above sets the scale for how much of this is noise: a solid black ellipse,
which cannot carry a face at all, still clears the threshold on one of the
sixty largest faces of `005035`. What
these rows report is not a face the pipeline missed. It is a blurred face
whose crop still leans the right way, and they sit where the measurement is
weakest — by size on `004310`, 24 of 2235 sightings of 40 px and over, 170 of
1725 between 24 and 40, and 477 of 3092 under 24 px, in a size class where
the recogniser is wrong about the *untouched* source seven times in eight.

### 12.3 The caches this harness had been reading were stale

The first leak run pointed at four frames of `004100` where an 85 px face
sat unmasked in the copy, and the box diff against the source agreed: those
pixels had not been touched. They had. The evaluation cache holds every
detector's raw candidates for every frame, and it was built halfway through
the pass of 2026-09-07, before that pass changed which boxes get a
confirmation crop. The cache version number had been bumped in the same
session, so nothing said the file was out of date, and every measurement
since described a build that no longer existed.

The cache now carries a fingerprint of `faceblur/detect.py` and of the
settings that decide what it holds, and a mismatch rebuilds it
(`eval/common.py`, `tests/test_eval_cache.py`). An edit to a comment in
`detect.py` costs a rebuild, about a minute a video on four workers; a
silently wrong measurement costs more.

Rebuilt, and with the pseudo-labels rebuilt on top of them, the shipped
defaults measure:

| File | Recall | Off-face mean (detector boxes) | Exposed 40+ | Exposed 24-40 | Hands | Continuity |
|---|---|---|---|---|---|---|
| `004100` | 100.0 % | 0.18 % (0.04 %) | 91 | 55 | 0.00 % | 99.0 % |
| `004310` | 99.7 % | 0.77 % (0.29 %) | 62 | 116 | 0.00 % | 99.5 % |
| `005035` | 99.7 % | 0.43 % (0.13 %) | 435 | 1196 | 0.09 % | 93.5 % |

The synthetic sets are cut from those same pseudo-labels, so they were
rebuilt too (`eval/synthetic.py` now stamps a set with the labels it came
from). On the shipped defaults they cover 78.7 percent of pasted faces on
`004310`, 87.8 on `004100` and 86.2 on `005035`; on `004310`, 83 percent of
faces 64 px and over, 79 of 32 to 63 px, 65 under 32 px, 86 sharp against 65
motion blurred, 81 entering at the frame edge at 2.6 frames' delay, and 86
during a pan.

Section 11 reported 98.4 / 99.8 / 99.6 percent recall and 55 / 62 / 456
exposed frames of 40 px and more for the same settings on the same files.
Recall and off-face masking read better on honest evidence and the exposure
proxy reads worse on `004100`, where it went from 55 frames to 91. The
differences are the crop rule that moved: which boxes get a confirmation
crop decides which faces are confirmed at the margin. The table above is the
current build measured on its own evidence, and it replaces section 11's.

### 12.4 Scanning at more pixels: measured, not taken

The standing open item was that faces under 32 px are covered about half the
time, and that a larger scan would raise it. Two ways to do that were
measured on 60 frames of `004310` that carry a hand oracle, against the
shipped scan (the whole frame at 1280 and at 1920):

| Scan | Boxes found | under 24 px | Boxes on a hand | Time a frame |
|---|---|---|---|---|
| shipped | 574 | 363 | 0 | 155 ms |
| plus a 2560 pass | 593 | 373 | 0 | 256 ms |
| 2x2 tiles at 1440 | 603 | 373 | 1 | 316 ms |

Tiling doubles the cost of detection, buys five percent more boxes, nearly
all of them the smallest, and puts one on the wearer's hand — the failure
the whole precision rebuild exists to prevent. The extra whole-frame pass is
cheaper and clean on hands, and still costs 65 percent more time for 3
percent more boxes.

Neither is worth it, and section 12.2 says why more plainly than the cost
does: at those sizes the recogniser cannot identify anybody in the untouched
source either. The tiling code was written for this pass and is removed
again; what remains of it is this table.

### 12.5 What this does not prove

- The recogniser only ever sees faces the source side detectors find. A leak
  count of zero is not proof of anonymity; it is proof that no face this
  pipeline can find survives its own mask. The size curve bounds what the
  faces it cannot find could give away, and that bound is the argument, not
  the leak count on its own.
- SFace is one recogniser, trained on portrait photographs. A better model, a
  model trained on this kind of camera, or a person who knows the people in
  the room may do what it cannot. "Not identifiable by SFace at four metres"
  is the claim; "anonymous" is not.
- The threshold is re-read off this footage rather than taken from the
  publication, because a wide angle camera at four metres is not a portrait
  set. It is set where one stranger pair in a hundred clears it, which is
  strict on the leak side and honest about the cost: at that threshold the
  recogniser also fails to recognise most same-person pairs.
- The mask curve and the ellipse curve run on the largest faces in each file,
  where the recogniser works at all. They say what the mask does to a face it
  can see, not what it does on average.

## 13. What the copy still shows (2026-09-08)

Everything above reads the source. The detectors report what they found, the
tracker what it kept, the audit how much was destroyed, and the whole
evaluation harness compares all of it against the source as well. None of it
can see a face that every part of it missed: a miss is invisible to the thing
that missed it, and coverage measured that way is a statement about what the
pipeline noticed, not about what it left behind.

`faceblur/verify.py` reads the output instead. It detects on the finished
copy, and for every box asks whether anything happened there. It needs no
labels, no pseudo-labels, no recogniser, no second environment and nobody's
time, and it is the only check in the project that can find a face nothing
else in it can.

### 13.1 The rule

A detector fires on a blurred face too, so a box in the copy proves nothing by
itself. Three numbers separate the cases, all inside the box:

- what the copy differs from the source by,
- what a mask applied here and now would differ by,
- what the encoder moved on its own, measured per frame as the median
  difference over the whole frame. Every pixel of the copy differs from the
  source a little, because the file is re-encoded.

The last one is subtracted from the other two. It matters more than it sounds:
on the first real find — a 40 px face reflected in a washroom mirror on
`004100`, plainly visible in the copy — a mask would have moved the box by
14.6 and the encoder had moved it by 3.6, so the raw ratio read 0.25, a hair
from being waved through as "the detector is firing on our own blur". With the
encoder's own noise off both sides it reads 0.05, which is what it is.

A box a mask would barely change even so — a flat wall, a dark corner — is
counted as undecidable rather than called either way.

### 13.2 Why it finds anything at all

It runs the same detectors that produced the copy, so it is not an independent
witness. Two things make it find misses anyway:

- **It is stricter than the pipeline.** One confirmed detection is enough to
  flag a frame. The pipeline needs a track: several confirmations, the
  agreement rules, a length. Everything that keeps hands out of the mask also
  lets a real face through when it is seen briefly, and the copy shows it.
- **The pixels are not the same pixels.** The copy is re-encoded, so a face at
  the confirmation margin can resolve differently in it. The mirror face at
  frame 660 of `004100` is detected in the copy and not in the source.

What it cannot do is find a face all three detectors miss in the copy as well.
It closes the gap between "confirmed as a track" and "detected at all"; it
does not close the gap below detection. Section 12's size curve is what bounds
that one.

### 13.3 Holding a copy back

`--quarantine` turns the check into a gate: a copy that still shows a face of
`quarantine_min_px` or more (24 by default) is moved into a `quarantine`
folder beside the output instead of shipping, and its audit record names the
frames. The file is not deleted — it is the only blurred copy of that video,
and whoever picks it up decides whether to cut those frames, run it again with
other settings, or look at them. For a training set, losing a video or a few
frames of one costs data volume; shipping a face costs something else.

### 13.4 What it found on the sample footage

The four sample files, blurred with the shipped defaults, checked frame by
frame:

| File | Frames | Boxes still visible | Frames they sit in | 40+ px | 24-40 px | under 24 px | Stretches |
|---|---|---|---|---|---|---|---|
| `003939` | 2033 | 8 | 7 | 6 | 2 | 0 | 5 |
| `004100` | 984 | 8 | 8 | 6 | 1 | 1 | 7 |
| `004310` | 938 | 17 | 16 | 8 | 3 | 6 | 14 |
| `005035` | 3971 | 94 | 89 | 52 | 21 | 21 | 58 |

That is 127 boxes over 120 frames of 7926, or one frame in 66. Every one of
them was checked by eye on the difference map for the two files looked at in
detail, and the finds are of three kinds.

**Faces the run missed.** On `004100`: a 98 px face at the left edge of frame
642, a 104 px face at frame 812, a 43 px face at frames 398 and 399, and a
40 px face reflected in a washroom mirror at frame 660. Nothing was masked at
any of them. These are the reason the check exists — no measurement that
reads the source finds them, because the pipeline needs a track before it
masks and these were seen too briefly to make one. The mirror face at 660 is
sharper still: the copy's detectors find it and the source's do not, because
re-encoding moved it over the confirmation line.

**The wearer's own hand.** On `005035`, the table tennis file, the biggest
finds are the hand holding the bat: 144 px at frame 1382, 118 at 1630, 99 at
1379, 91 at 770. The detectors call a hand a face on this footage — section
10 measured YuNet scoring the wearer's hand on a mop at 0.74 to 0.82 — and
the pipeline keeps them out of the mask with track-level rules that this
check does not have. So it reports them. The check inherits the detectors'
weakness; it does not correct it.

> The sentence that stood here said that on that file most of its finds are
> hands rather than faces. That was true of the finds named above, which are
> the largest ones, and section 14.1 shows it is not true of the finds: on
> `005035` there are 34 faces to 16 hands. It was written from the top of the
> list rather than from the list.

**Faces on other surfaces.** A person on a television at frame 2487 of
`005035`. Untouched, correctly reported, and a judgement call as to whether
anyone would want it masked.

The borderline band is real and worth stating: a verified miss on `004100`
reads 0.19 and a masked face on `005035` reads 0.30, so a rule at 0.3 keeps
every miss it has been shown and admits the occasional masked face along with
them. It errs towards reporting, which is the safe direction for a gate, and
the ratio is in the record so a borderline case can be looked at.

Checking `004100`, 984 frames, costs 32 s on four workers of an RTX 3070
against the 37 s its detection pass took: the same work again, near enough,
since both are one detection per frame and the check decodes two files
instead of one.

With `--quarantine` on and the default of 24 px, all four sample files would
be held back. That is the honest state of this pipeline, not a fault of the
gate: every one of these files still shows something face-shaped that nothing
masked, and until this pass there was no way to know it.

## 14. Telling a hand from a face in the output-side check (2026-09-09)

Section 13 left one thing open: the check that reads the finished copy runs
the same face detectors that produced it, and they call the wearer's own hand
a face. The pipeline keeps hands out of the mask with track-level rules — a
hand is confirmed two or three frames at a time and never earns a sure track
— and a check that reads one frame at a time has no counterpart to them. It
was called the obvious next piece of work, and the thing standing between
`--quarantine` and being usable unattended.

Half of that turned out to be right.

### 14.1 What the check was actually finding

Before building anything, every one of the 108 boxes the check reports on the
four sample files was looked at, one at a time, on a crop of the source around
it. That is one person's eye on 108 pictures, several of them motion blurred
and none of them large, so six could not be called at all and are reported
apart from the rest. It is the only ground truth this question has, and
nothing below means anything without it.

| File | Boxes | A face | The wearer's hand | Neither | Could not say |
|---|---|---|---|---|---|
| `003939` | 8 | 6 | 2 | 0 | 0 |
| `004100` | 8 | 7 | 0 | 1 | 0 |
| `004310` | 17 | 5 | 2 | 6 | 4 |
| `005035` | 75 | 34 | 16 | 23 | 2 |
| **all** | **108** | **52** | **20** | **30** | **6** |

At 40 px and over, where identity is at stake at all, it is 30 faces, 16
hands, 7 neither and 3 unsayable.

Section 13.4 said that on the table tennis file most of the check's finds are
the hand holding the bat. That was true of the finds it named, which were the
largest ones, and it is not true of the finds: on `005035` there are 34 faces
against 16 hands, and over the four files the check is finding real faces
about two and a half times as often as it is finding hands. The rest, "neither",
is a television, a picture on a wall, a flat wall, and motion smears.

So the hand rule is worth having, and it is **not** what stands between the
gate and unattended use. What stands there is that these four copies still
show around fifty faces that nothing masked. No rule about hands moves that.

### 14.2 The rule

`faceblur/hands.py`. Two models, both MediaPipe's, both Apache 2.0, converted
from the tflite in its wheel once and offline by `models/make_hands.py`; the
conversion agrees with the tflite to 1.5e-4 on every raw output. MediaPipe
itself runs only in `.venv-eval`, and this has to run in the main environment,
which has onnxruntime and nothing else. Neither model is loaded unless
`--check-output` or `--quarantine` is on, and then only for a box the check
has already flagged, so the pipeline's own cost is untouched.

**A fixed window, not a multiple of the box.** The first attempt cropped a
multiple of the flagged box, the way the face confirmation cuts its crops, and
it was unstable in a way that is easy to miss: the same face read 0.00 hand at
one multiple and 1.00 at the next. What the palm detector makes of a crop
depends on how large a hand is inside it, and a multiple of the box holds that
constant only when the box is the hand. A fixed pixel window downscaled to the
model's own 192 puts a hand at the size the model was trained for whatever
size the box around it happens to be. Two windows, 384 and 512 px: one 512
window sets aside 12 of the 20 hands, both together 13, a third at 768 adds
none.

**A second model, because the first is loose.** Of the 138 boxes the palm
detector put on the crops taken around those 108 finds, the landmark model
rejects 97, and two crops in five carry one it rejects. That matters more than
the raw rate, because the region a palm box hands on is 2.6 times its own
size: an invented hand covers whatever is near it. On `004100` frame 450 the
palm detector put a box on a man's shoulder and its region covered the real
missed face beside it. So every palm box is warped to 224x224 and put to
MediaPipe's hand-landmark model, which returns how sure it is that the
rectangle holds a hand, and only a confirmed one counts. The score needs no
sigmoid, whatever MediaPipe's graph does with it downstream: this model
already returns a probability, and a sigmoid would compress black (0.007) and
a hand (0.89) into 0.50 and 0.71 and leave nothing to threshold.

That second model is the whole difference, and the clearest way to see it is
to run the same 1050 combinations of window set, palm floor, region and
coverage with it and without it:

|  | combinations that lose no face | best of those | how many reach it |
|---|---|---|---|
| palm detector alone | 256 of 1050 | 11 of 20 hands | 4 |
| with the confirmation at 0.7 | **1050 of 1050** | **13 of 20 hands** | 24 |

Without the confirmation there is a rule that reaches 11 hands and loses no
face, and it is a needle: from it, growing the region from the palm box to
twice the palm box loses seven faces, and tightening the coverage from 0.3 to
0.7 drops it from 11 hands to 1. With the confirmation nothing in that grid
loses a face at all, and the geometry stops mattering — which is what makes
this safe to put behind a gate rather than merely good on this footage.

**Where the settings sit.** Hands set aside of 20, then faces lost of 52, over
the palm detector's floor and the landmark model's confirmation, with the two
windows that ship, the region MediaPipe itself uses and a coverage of half the
box. The shipped setting is the bold row at palm 0.4:

| confirmation | palm 0.3 | palm 0.4 | palm 0.5 | palm 0.6 | palm 0.7 |
|---|---|---|---|---|---|
| 0.50 | 14, **2** | 14, **2** | 13, **2** | 12, **2** | 11, **2** |
| 0.60 | 14, **1** | 14, **1** | 13, **1** | 12, **1** | 11, **1** |
| 0.65 | 14, 0 | 14, 0 | 13, 0 | 12, 0 | 11, 0 |
| **0.70** | 13, 0 | 13, 0 | 12, 0 | 11, 0 | 10, 0 |
| 0.75 | 11, 0 | 11, 0 | 11, 0 | 10, 0 | 9, 0 |
| 0.80 | 10, 0 | 10, 0 | 10, 0 | 9, 0 | 8, 0 |

The confirmation is the setting that decides, and 0.7 is one step inside where
it stops losing faces. Around the shipped setting — the two windows, palm
floor 0.4, MediaPipe's own region, coverage half the box — every single step
holds: palm floor 0.3 gives 13 hands and no face, 0.7 gives 10; region 1.6
gives 12, 2.0 gives 13; coverage 0.3 and 0.7 both give 13. Nothing on that
plateau costs a face.

### 14.3 What it does

Run over the four sample files, with the shipped defaults:

| File | Boxes | Reported as faces | Set aside as hands | Held back? |
|---|---|---|---|---|
| `003939` | 8 | 7 | 1 | yes |
| `004100` | 8 | 8 | 0 | yes |
| `004310` | 17 | 15 | 2 | yes |
| `005035` | 75 | 64 | 11 | yes |
| **all** | **108** | **94** | **14** | |

Against the labels: **13 of the 20 hands set aside, 0 of the 52 faces, 0 of
the 30 that are neither, and 1 of the 6 that could not be called.** The seven
hands it leaves are reported as faces, as before.

A hand stays in `residual_list`, marked with the coverage that decided it. A
rule that quietly dropped what it disagreed with would be worth nothing to an
auditor. It is left out of `residual_faces`, out of `residual_by_size`, out of
`residual_runs`, and out of what the gate reads.

**No file is released by this.** All four are still held back, and correctly:
every one of them still shows faces nothing masked. What the rule buys is a
record that says which finds are hands, and a gate that will not hold a copy
back for a hand alone.

### 14.4 Cost

Nothing, within the noise of the measurement, which was the one genuine
surprise of this pass.

`005035`, 3971 frames, 75 flagged boxes, four workers on an RTX 3070, the
whole check timed end to end in one process so the three runs are comparable:

| | Seconds | Faces reported | Hands set aside |
|---|---|---|---|
| hand rule off | 341 | 75 | — |
| hand models on the GPU | **338** | 64 | 11 |
| hand models on the CPU | 385 | 64 | 11 |

On `004100`, 984 frames and 8 flagged boxes, there is nothing to see either:
34 to 36 seconds whichever way round, repeated.

Two models, two windows each and a landmark pass per palm box sounds like a
lot until the arithmetic: the rule runs on the boxes the check has already
flagged, which is 75 frames of 3971 on the worst file and 8 of 984 on the
next. Both models are small — 192x192 and 224x224 — beside a 1600x1300
detection pass.

The expectation going in was the opposite, and the setting `hand_device` is
what is left of it. Each worker already holds about 850 MB of face-detector
graphs, four of them share an 8 GB card, and section 10.4 records six workers
reaching 7.3 GB and DirectML paging; two more models per worker looked like it
would tip that over. It does not: on the GPU the rule is free, and moving it
to the CPU to save the memory costs 13 percent instead. `hand_device` stays,
defaulting to `auto` with the rest, for a machine where the card is smaller.

Both devices set aside the same 11 hands and report the same 64 faces. A gate
whose answer depended on which device it ran on would not be one.

### 14.5 What it does not do

- **Seven hands still read as faces**, one on `003939` and six on `005035`.
  They are the ones where neither window shows the palm detector enough of a
  hand to get a confirmed box over the flagged one — a fist gripping a bat,
  side on, in motion. One more is available at a confirmation of 0.65 rather
  than 0.7, at no measured cost in faces, and it was not taken: 0.65 is the
  last value before the cliff, and one step is the margin this rule is worth
  spending on a fourteenth hand.
- **The 30 finds that are neither a face nor a hand are untouched.** A person
  on a television, a picture on a wall, a flat wall, a motion smear. They hold
  copies back exactly as before. A screen detector would be a project of its
  own and is not obviously wanted: masking a television is a policy question,
  not a detection one.
- **The labelling is one pass by one pair of eyes** over 108 crops, 9 to 187
  px, several motion blurred. Six could not be called and are counted apart;
  the rule flagged one of those six. A second person would not agree with
  every call, and the numbers above should be read with that in them.
- **The rule reads the source frame, not the copy.** Nothing masks a hand, so
  it looks the same in both, and the source is the sharper of the two. If a
  future change ever masked hands on purpose, this would have to change with
  it.

## 15. Screens as objects (2026-09-10)

The scope changed on 2026-09-10 from faces to faces, personal text and
screens, each a switch of its own (`FACEBLUR_BUILD_PLAN.md`, "Scope change").
This section is the screen class. Text is not built.

A screen needs none of the machinery the face side has. There is no identity
to weigh: whatever is on the glass is hidden, so the only question is where
the glass is. One general purpose object detector answers it.

### 15.1 The model, and the first run

YOLOX-tiny, Apache 2.0, from Megvii's release, 20 MB. Licence came first:
most YOLO derivatives in common use are AGPL and this project cannot ship
them. Of the permissive detectors that remained it is the smallest whose
decode matches one already here — anchor free, three strides, the same shape
as YuNet's — so `faceblur/screens.py` could be checked against a frame rather
than trusted. It is a COCO detector and three of its eighty classes are
glass: tv, laptop and cell phone.

The decode was checked by eye first: on `005035` frame 309 it puts a box on
the wall mounted television at 0.85. Two things about the input were found by
measurement rather than read from a page, and both are recorded in
`models/README.md`: the frame is letterboxed to the **top left** on grey 114,
not centred, and it is fed as **raw 0 to 255 BGR with no mean and no standard
deviation** — normalising it the ImageNet way drops that same score from 0.85
to 0.004.

Then it was run end to end, and it failed:

| `005035`, 3971 frames | Faces only | Faces and screens, first try |
|---|---|---|
| Share of the frame destroyed, mean | 1.7 % | **2.84 %** |
| Worst frame | 4.8 % | **42.7 %** |
| Frames over the 5 % budget | 0 | **200** |

### 15.2 What it was actually masking

Every one of the 85 detection runs the model produced across the four sample
files, at a floor of 0.5, was looked at on the frame it came from. 54 are a
real screen, 29 are not, 2 could not be called.

The 29 are almost all one thing: **the blue table tennis table**. A COCO
detector calls any large flat rectangle a television, and this footage is
made of them. The table is called a laptop, a television and a phone by
turns. Washroom mirrors, glass walls and dark doorways make up the rest.

Score does not separate them. The table reads **0.88** covering a third of
the frame; the real television across the hall reads **0.85** covering one
percent. A bigger model does not separate them either: YOLOX-s, at 36 MB,
misses the real television at frame 425 outright, scores the one at 309 at
0.48 against tiny's 0.85, and still calls the glass wall a television.

Two things do separate them, and neither is the detector's opinion.

- **Size.** The false runs are 2.4 to 35 percent of the frame; the wall
  television and the phones are 0.4 to 1.6.
- **Persistence.** 14 of the 29 false runs last a single frame and not one
  real screen does. A table looks like a laptop from some angles and not
  others, so the detector calls it one and then stops. A monitor on a wall
  stays a monitor.

### 15.3 The two rules, and why both

Precision over the 83 runs that could be called, and the real screens lost,
for a minimum run length against a cap on the box's share of the frame:

| Seen in | cap 2 % | cap 4 % | cap 8 % | cap 12 % | cap 25 % | no cap |
|---|---|---|---|---|---|---|
| 1 frame | 92 %, lost 5 | 91 %, lost 2 | 87 %, lost 1 | 82 %, lost 0 | 68 %, lost 0 | 65 %, lost 0 |
| 2 frames | 92 %, lost 5 | 91 %, lost 2 | 90 %, lost 1 | 89 %, lost 0 | 79 %, lost 0 | 78 %, lost 0 |
| **3 frames** | 92 %, lost 5 | 91 %, lost 2 | 91 %, lost 2 | **91 %, lost 1** | 84 %, lost 1 | 83 %, lost 1 |
| 4 frames | 95 %, lost 18 | 93 %, lost 16 | 93 %, lost 16 | 93 %, lost 15 | 87 %, lost 15 | 87 %, lost 15 |
| 6 frames | 95 %, lost 35 | 91 %, lost 33 | 91 %, lost 33 | 92 %, lost 32 | 85 %, lost 32 | 85 %, lost 32 |

The shipped rule is **seen in 3 frames, cap 12 percent** (`screen_min_run`,
`screen_max_area`), and the point of the table is what it says about doing
either alone.

A cap alone has to be tight to work: 4 percent, for 91 percent precision. At
4 percent it throws away a real monitor covering 10.6 percent of the frame,
and that is the worst place to lose one, because a screen that large and that
close is the one most likely to be showing something a person could read.

With persistence carrying the precision instead, the cap can sit at 12
percent for the same 91 percent, and that monitor is kept. Persistence is
what makes the looser cap safe.

Going further costs too much. At 4 frames precision rises two points and
fifteen more real screens are lost, because most sightings of the wall
television are short bursts of three or four frames as the camera swings past
it. Losing them loses coverage of a screen the run does mask elsewhere.

### 15.4 What it costs, and what is left

Share of each frame the screen mask destroys, with the runs grouped, the
short ones dropped, gaps filled and the tail added:

| Rule | `003939` | `004100` | `004310` | `005035` | Worst frame |
|---|---|---|---|---|---|
| floor 0.35, no cap, no persistence | 0.44 % | 0.46 % | 0.10 % | 0.66 % | 78 % |
| floor 0.50, no cap, no persistence | 0.15 % | 0.23 % | 0.01 % | 0.30 % | 35 % |
| floor 0.50, cap 4 %, no persistence | 0.08 % | 0.38 % | 0.07 % | 0.32 % | 6 % |
| **floor 0.50, cap 12 %, seen in 3** | **0.04 %** | **0.34 %** | **0.01 %** | **0.24 %** | **12 %** |

The face mask on the same four files destroys 0.25 to 1.7 percent of each
frame and at most 4.8 percent of one. The screen mask is smaller than that on
average and larger at its worst, and the worst is one frame with a large
monitor properly masked in it.

What is left, and is not going to be fixed by tuning:

- **9 percent of what it masks is not a screen.** The survivors are small:
  a 2.4 percent patch of a washroom wall on `003939` for 10 frames, a 0.4
  percent patch on `005035` for 27. They are cheap in pixels and they are
  still wrong.
- **One real screen of 54 is lost** by the persistence rule, and it is a two
  frame sighting of a monitor at 5.2 percent of the frame.
- **A screen the detector never finds at all is not counted here**, exactly
  as with faces. There is no oracle for screens on this footage, so unlike
  the face numbers there is no recall figure against an independent witness,
  only precision against one person's eye over 85 runs.
- **The rule is tuned on one venue.** The table tennis hall is what makes the
  size cap necessary and what sets its value. A room with a large monitor
  close to the camera and no large flat furniture would want a looser cap;
  there is no footage here to set one on.
- **A screen is masked as an object, whatever is on it.** A monitor showing a
  scoreboard is destroyed as thoroughly as one showing a spreadsheet of
  names. That is the scope decision of 2026-09-10, taken so that nothing has
  to judge what is on the glass, and the cost is over masking.

## 16. Screens in the output side check (2026-09-10)

Section 15 built the screen class and left one of the four conditions a kind
has to meet before it may be called ready unmet: nothing checked whether a
screen the run missed was still in the finished copy. Every screen number in
section 15 reads the source and inherits the pipeline's own blindness, exactly
as every face number did before section 13.

This section closes that. `faceblur/verify.py` now reads the copy for whatever
the run was asked to mask.

### 16.1 What the check finds, and what holds a copy back

**The rule is the face rule, asked of a different shape.** `applied` takes its
region from `redact.region_for`, so the control masks a screen's quad where it
masked a face's ellipse, and the comparison happens inside it. Nothing else in
that function was ever face specific. A masked screen still reads as a screen
to YOLOX, because a pixelated rectangle is a flat rectangle, and that is what
the ratio is for: near one the mask ran, near zero nothing reached it.

**One rule is new: how long was it there.** A face needs none, because one
frame of a face is a face. A screen does, because the detector calls a table a
laptop for one frame and then stops (section 15.2). So screen finds are
grouped into runs by `screens.runs_in`, the same grouping and the same
settings the pipeline uses on the source side, and a run has to last
`screen_gate_min_run` checked frames before it decides anything. The length is
counted in checked frames, so at `check_stride` 2 a run of three spans six
source frames.

**Hands are not asked about a screen.** Nothing mistakes a hand for a
television, and a hand in front of a monitor does not stop it being a monitor.

Run over the four sample files and the copies of them work package R1 wrote
with `--mask face,screen`:

| File | Frames checked | Faces left | Screen boxes left | Frames they sit in | Largest | Runs over the floor | Runs of 3 or more | Held back for |
|---|---|---|---|---|---|---|---|---|
| `003939` | 2033 | 7 | 11 | 11 | 525 px | 7 | 1 | face, screen |
| `004100` | 984 | 9 | 14 | 14 | 252 px | 5 | 3 | face, screen |
| `004310` | 938 | 16 | 10 | 10 | 314 px | 7 | 0 | face |
| `005035` | 3971 | 67 | 129 | 127 | 640 px | 85 | 7 | face, screen |

All four are held back for faces, as section 13.4 said they would be. **Three
of the four would be held back for a screen alone**: `003939`, `004100` and
`005035`. `004310` would not, and it is the file with the fewest screens in
it.

**Persistence is what makes this gate usable, and the size floor is not.**
Without the run rule, every one of the four would be held back for a screen:
the runs over the floor column is never zero. With it, three are. On `005035`
the rule does most of its work: 56 of the 85 runs last a single checked frame
and only 7 reach three.

The floor did nothing at all. `screen_gate_min_px` is 48 and the smallest
screen find on any of the four files is 95 px, so not one find was set aside
by it. That is not the floor working, it is the floor never being tested: a
COCO detector at a confidence of 0.5 does not fire on small glass, so there
are no small screen finds on this footage to set aside. A venue with a phone
on a table across a room would exercise it and there is none here. Read the
48 as untested rather than as measured.

**Cost, and what it does to the face numbers.** The same check run twice on
each copy, once with screens in it and once without:

| File | Faces only | With screens | Faces reported, both ways |
|---|---|---|---|
| `003939` | 68.6 s | 71.2 s | 7 |
| `004100` | 33.4 s | 34.4 s | 9 |
| `004310` | 53.4 s | 54.4 s | 16 |
| `005035` | 146.3 s | 154.9 s | 67 |

Two to six percent, not the second detection pass the extra model suggests:
YOLOX-tiny runs at 416 by 416 against the face bank's 1280 and 1920, and both
files are already decoded. The face count is identical either way on every
file, so nothing in section 13 or section 14 moves.

Those face counts are close to but not the same as section 14.3's 7, 8, 15 and
64. These are different copies. Section 14.3 read `footage_blurred_v8`, made
with faces alone; this reads the R1 copies, made with faces and screens, so
the pixels a face detector sees in them are not the same pixels. The check
itself is unchanged, which is what the identical face counts with and without
the screen pass show.

**Does the gate catch what the pipeline knowingly throws away?** This is the
question worth asking of a gate, and it can be asked without labels. The
pipeline masks nothing it has not seen `screen_min_run` times, and section
15.4 records that costing one real screen of 54. Every screen dropped that way
is by definition still in the copy, so the check should be reporting it.
Re-detecting screens on the four sources, grouping them the shipped way, and
matching each dropped run against the runs the check found on the copy:

| File | Source runs | Masked | Dropped by persistence | Dropped runs the check reports |
|---|---|---|---|---|
| `003939` | 12 | 4 | 8 | 5 |
| `004100` | 10 | 5 | 5 | 5 |
| `004310` | 5 | 1 | 4 | 4 |
| `005035` | 147 | 49 | 98 | 68 |
| **all** | **174** | **59** | **115** | **82** |

**The check reports 71 percent of what the pipeline's own persistence rule set
aside**, and it does so with no knowledge of what that rule did. The 33 it
does not report are runs the copy's re-encoded pixels no longer show a screen
at, which is the same effect that lets the check find faces the source side
misses, running the other way.

**The two frame monitor of section 15.4 is caught.** Across the four files
there is exactly one dropped two frame sighting of a monitor at about five
percent of the frame: `003939`, frames 611 and 612, 358 px, 4.81 percent. The
check finds it on the copy as a run of four checked frames from 606 to 612,
which is over `screen_gate_min_run`, so it holds that copy back. The one real
screen section 15.4 says the pipeline loses is the one the gate stops.

Section 15.4 gives that monitor as 5.2 percent of the frame and this
measurement gives 4.81 percent for the largest box of the run. Nothing here
explains the difference and it is not worth another pass to chase: it is the
only candidate on any of the four files by a wide margin, the next nearest two
frame sightings being 2.55 and 7.31 percent.

**A rule that was suspected and is not happening.** `runs_in` groups by label,
so one object the detector calls a television and then a laptop would split
into two short runs and be dropped by persistence for the wrong reason. Over
the 115 dropped runs there is exactly one pair of different labels overlapping
in time, so on this footage that is not what is costing screens. It is worth
knowing before anybody tunes `screen_min_run` again.

**A number that is not what it looks like.** `screen_max_area` is 0.12 and two
of the dropped runs measure 13.12 and 12.43 percent of the frame. The cap is
checked on the detector's own box and `screen_pad` grows it by six percent
afterwards, so a box at the cap is masked at up to about 13.4 percent of the
frame. The setting bounds what may be called a screen, not what is destroyed.

### 16.2 Screen recall against an independent witness

Section 15.4 had to say screens have no recall figure at all: one detector,
nobody to check it, and a screen it never finds counted nowhere. This is the
witness, and the number is not good.

**The oracle.** OWLv2, open vocabulary, Apache 2.0, pinned to one revision,
run on the CPU in a third environment at 4.2 s a frame, stride 5 over the four
sample files: 1587 frames, about two hours. It takes phrases rather than a
fixed class list, so one model answers for televisions, monitors, phones,
cards, slabs, badges, documents, handwriting and signs, which is what work
packages F0, T1 and T2 were going to need as well. `eval/oracle_owl.py` holds
the phrases and the revision; `models/README.md` holds the hash.

Three checks before it was used for anything, because a witness nobody has
questioned is not a witness:

- **Its coordinates are right.** OWLv2's processor pads a frame to square, and
  a post-processor that unpads wrongly returns boxes that look plausible and
  sit in the wrong place. On frame 309 of `005035` the oracle puts a
  television at xyxy [213, 233, 349, 339] and the shipped YOLOX puts one at
  [203, 227, 358, 349]. Two families agreeing within ten pixels is the check.
- **It does not share the failure that made the size cap necessary.** A COCO
  detector calls any large flat rectangle a television and the blue table
  tennis table reads as a laptop at 0.88 over a third of the frame (section
  15.2). Across all four files at a score of 0.2 the oracle puts exactly one
  box over 12 percent of the frame. It does not call the table anything.
- **It rejects the false positives section 15.4 admits to.** On the washroom
  file the shipped rule masks 10 boxes and the oracle calls none of them a
  screen. Those are the 2.4 percent patch of washroom wall that section 15.4
  reports as wrong.

**Recall and precision**, counted per sighting on the frames the oracle
looked at, over screens covering 0.4 percent of the frame or more, with the
oracle's own overlapping proposals suppressed first at an IoU of 0.5 and a
match counted at an IoU of 0.3:

| File | Oracle screens | Recall at oracle floor 0.3 | Precision |
|---|---|---|---|
| `003939` washroom | 17 | 0 % | 0 % |
| `004100` corridor | 43 | 2 % | 3 % |
| `004310` canteen | 105 | 0 % | 0 % |
| `005035` table tennis | 491 | 23 % | 61 % |

`eval/screens.py` prints the same table at six oracle thresholds, because the
answer moves with that choice and a single number would hide it. On `005035`
recall runs from 9 percent at a floor of 0.1 to 25 percent at 0.4, and
precision the other way, from 68 percent down to 56.

**What is causing it is not the two rules section 15.3 tuned.** For every
oracle screen the shipped rule does not mask, the same code was asked which
step lost it:

| File | Oracle screens | Masked | Never seen by the detector | Thrown out by the size cap | Dropped by persistence |
|---|---|---|---|---|---|
| `003939` | 17 | 0 | 17 | 0 | 0 |
| `004100` | 43 | 1 | 42 | 0 | 0 |
| `004310` | 105 | 0 | 105 | 0 | 0 |
| `005035` | 491 | 113 | 369 | 0 | 9 |

The size cap costs nothing at all and `screen_min_run` costs nine sightings on
one file and none on the other three. Section 15.3 chose both carefully and
neither is what is holding the class back. What is holding it back is that
YOLOX-tiny at `screen_conf` 0.5 does not report most of what the oracle calls
a screen.

**Whether a threshold would fix that depends on the venue.** At each place the
oracle sees a screen and the shipped detector reported nothing, YOLOX was
asked what it scored there with no floor at all:

| File | 0.50 and over | 0.25 to 0.50 | 0.01 to 0.25 | Nothing at all |
|---|---|---|---|---|
| `003939` | 0 % | 12 % | 35 % | 53 % |
| `004100` | 0 % | 5 % | 70 % | 26 % |
| `004310` | 0 % | 0 % | 40 % | 60 % |
| `005035` | 15 % | 18 % | 58 % | 9 % |

On the table tennis hall the detector sees something at 91 percent of them,
almost all of it under the shipped floor, so `screen_conf` is the lever there
and the trade against the table is measurable. On the canteen and the washroom
it sees nothing at all at 53 to 60 percent of them, and no threshold reaches
those.

**Read both numbers as bounds, in both directions.** The washroom has no
screens in it. The oracle finds 17 there at a floor of 0.3 and 3 at 0.4, and
they are almost certainly mirrors and bright tiles, so part of what this
section counts as a missed screen is not one and the recall figure is
pessimistic. Equally, a screen neither model finds is in neither column. The
oracle is one model's opinion at a threshold somebody chose, exactly the
caveat section 12.5 gives the recogniser, and where the two disagree neither
is right by definition.

**Nothing can adjudicate between the oracle and the eye.** Precision against
the oracle is 56 to 68 percent on the one file with enough data and section
15.2 gives 91 percent against one person's eye over 85 runs. Those two cannot
be reconciled, because **the 85 labels were never committed**. Only the counts
survive, in section 15.2. The same is true of the 108 boxes of section 14.1.
Both audits are honest about being one person's eye, and neither can be
re-scored against anything, now or later. A future by eye audit should write
its labels to a file beside the report.

### 16.2.1 There are no cards in this footage

The oracle was built with `a trading card`, `a graded card slab` and `a binder
page of trading cards` among its phrases, because work packages F0 and T2 are
about cards and needed regions to work from. Over 1587 frames spread across
all four sample files it reports:

| Score floor | Card prompt hits | Television hits, for scale |
|---|---|---|
| 0.10 | 137 | 3213 on `005035` alone |
| 0.15 | 11 | |
| 0.20 | 0 | |
| 0.30 | 0 | |

Zero at any score a reader would act on. The four sample files are a washroom
being cleaned, a corridor being mopped, a canteen and a table tennis hall, and
the wearer appears to be a facilities worker. Everything already in this
report is consistent with that once it is pointed at: the wearer's hand on a
mop in section 10, the face in a washroom mirror in section 13.4, the blue
table tennis table in section 15.2.

This is not a fault in anything. It is a fact about the sample footage that no
document said plainly, and it decides what the remaining work is worth:

- **F0, faces printed on cards, has nothing to measure.** Decision D1 asks
  whether to leave a printed face on a card alone. On this footage the
  question is unobservable either way.
- **T2, the card and slab veto, has nothing to tune.** Its design in the build
  plan is a decision tree driven by how much card text sits above the size
  floor, and here there is no card text at all.
- **The screen and text work is unaffected**, and text is worth more than
  expected: the canteen has signage and packaging, and a person in the
  washroom file wears a shirt with text on it, which is the case section 7
  item 25 of the build plan names as personal text on clothing.

Section 11 of `FACEBLUR_BUILD_PLAN_V2.md` asks the owner whether the footage
holds sports cards with real faces or only game cards. The measured answer is
neither.

### 16.3 What each kind destroys, measured apart

Work package 4.3 of `FACEBLUR_BUILD_PLAN_V2.md`. The audit record had one
number for how much of a frame was destroyed, and once two kinds can mask at
once that number stops answering the question anybody asks of it. A screen
mask is large by design and a face mask is small on purpose, so a frame three
percent destroyed may be 0.4 percent of face and the rest glass, and the five
percent budget that was set on faces no longer means what it says.

`redact.masked_shares` builds each kind's alpha on the same canvas and gives
the share of the frame it covers. `encode_job` and `process_video` carry one
list per kind through to the record as `masked_mean_by_kind`,
`masked_max_by_kind` and `frames_over_budget_by_kind`. The union numbers keep
their names and their meaning. When a frame carries only one kind, which is
most frames, the union alpha is that kind's mask and nothing extra is
computed.

The four sample files, run with `--mask face,screen` and the shipped defaults:

| File | Frames | Union mean | Face | Screen | Union worst | Face | Screen | Over budget: union | face | screen |
|---|---|---|---|---|---|---|---|---|---|---|
| `003939` | 2033 | 0.90 % | 0.85 % | 0.05 % | 4.46 % | 4.46 % | 3.29 % | 0 | 0 | 0 |
| `004100` | 984 | 0.93 % | 0.51 % | 0.43 % | 5.84 % | 5.84 % | 4.70 % | 6 | 6 | 0 |
| `004310` | 938 | 2.33 % | 2.31 % | 0.01 % | 8.94 % | 8.94 % | 1.05 % | 24 | 23 | 0 |
| `005035` | 3971 | 0.99 % | 0.69 % | 0.31 % | 13.30 % | 8.33 % | 13.30 % | 40 | 9 | 23 |

**The table tennis file is the case the split exists for.** Read the union
alone and `005035` has 40 frames over the five percent budget. The face mask
puts 9 of them there. Twenty three are the screen mask, correctly masking a
large close monitor, and the remaining 8 are over budget only because the two
kinds together cross the line where neither does alone. A reader of the old
record would have seen the face class four times as far over budget as it is.

**The face column reproduces the faces only run exactly.** Section 15.4
measured `005035` with faces alone at 0.69 percent of the average frame, a
worst frame of 8.3 percent and 9 frames over budget. The face column of a run
that also masked screens reads 0.69 percent, 8.33 percent and 9. That is the
check that this split is measuring what it claims to: adding a second kind
does not move the first kind's numbers by anything.

The screen column does not reproduce section 15.4 as cleanly. It reads 0.05,
0.43, 0.01 and 0.31 percent against 15.4's 0.04, 0.34, 0.01 and 0.24. Nothing
here explains the difference. These numbers come from the shipped pipeline and
are written into every audit record, so re-running the command reproduces
them; 15.4's came from a one-off comparison harness that is not in the
repository and cannot be re-run. Prefer these, and treat 15.4's screen shares
as the shape of the answer rather than the answer.

**The sweep was never at risk.** The build plan expected the gates in
`eval/sweep.py` to be diluted by screen masks too. They are not, and the code
is what says so: `eval/measure.evaluate` builds its own per frame list from
the raw face cache and the tracker and never runs the screen detector, so no
screen mask can reach `off_face_mean`, `off_face_max`,
`off_face_detections_mean` or `hand_damage` however many kinds a run is asked
for. `tests/test_classes.py::test_the_sweep_measures_faces_and_never_sees_a_screen`
pins that, so a later change which starts feeding screens into the harness has
to face the question rather than move every gate quietly.

### 16.4 What this does not do

- **It reports what YOLOX calls a screen.** The check inherits the detector's
  weakness exactly as the face side inherits its detectors': a table that
  reads as a laptop for three checked frames on the copy will hold a copy
  back. There is no oracle for screens on this footage, so there is no way
  here to say how many of the 82 are real. Work package E1 is what would.
- **A screen over the size cap is not reported, on purpose.**
  `residual_job` calls the same `screens.detect` the pipeline calls, so
  `screen_max_area` applies on the copy side too and a monitor over 12 percent
  of the frame is invisible to the check as well as to the run. That is scope
  rather than a miss: the pipeline decided not to mask it, and a gate that
  held the copy back for a thing the run was never going to mask would hold
  back every file with a table in it. It does mean the check cannot tell you
  about the one screen the cap is most likely to be wrong about. The cap is
  the place to argue, not the gate.
- **There is a recall number now, and it is a bound.** Section 16.2 has it. A
  screen neither the shipped detector nor the oracle finds is still in nobody's
  column.
- **The size floor is untested**, as above.
- **The window cannot run this at all.** `--check-output` and `--quarantine`
  are command line only; `ui/app.py` never sets either, so no run started from
  the window is checked or held back. That predates this pass and is not
  changed by it.
- **Per kind shares are of the frame, not of each other.** Two kinds may
  overlap, so the shares can sum to more than the union. A face on a
  television is masked by both classes and counted in both columns.
- **The by eye labels of section 15.2 are not used here and are not
  re-scored.** Nothing in this section rests on a label.

### 16.5 Screens outside the handled zone

Held for work package S2. Section 16.2 says the shipped screen class reports
most of what the oracle sees at a score under `screen_conf` 0.5, and the rule
of 2026-09-11 changes what a lower floor costs: a masked table is a cost to
the environment rather than to privacy, and a handled screen is protected by
the zone whatever the floor. S2 re-measures at 0.5, 0.4, 0.3 and 0.25 with the
zone on and moves the default only if precision holds.

### 16.6 Fixes from the review of that pass (2026-09-11)

Seven things the last pass got wrong or left undone, found by reviewing it
rather than by running it. Two changed behaviour, three changed tests, two
changed documents.

**The exit code was not the gate's answer.** `python -m faceblur.verify`
returned 1 whenever it reported any box at all, so a single frame flicker on a
table failed the command while the gate itself would have shipped the copy. A
script could not tell "look at this" from "do not ship this", which is the
only question an exit code is good for. It now follows `held_for` and nothing
else, and everything found is still printed and still in the record.

Run on the `004310` copy from work package R1, which the check finds 10 screen
boxes in, none of them lasting the three checked frames the gate asks for:

| Command | What it reports | The gate | Exit |
|---|---|---|---|
| `--mask screen` | 10 screen boxes over the size floor, 7 runs, none of 3 | ships | 0 |
| `--mask face,screen` | the same, plus the faces | held back for face | 1 |

Before this change the first row exited 1.

**A row was not enough to rebuild the detection it came from.** `_row` carried
a centre, a long side and a score. Package F1 seeds the tracker with the
check's own finds, and a seed with no width, height or landmarks masks an
upright ellipse of guessed shape over a face that is turned. Every row of
every kind now carries `w`, `h` and `landmarks`, rounded the way
`Detection.to_dict` rounds them, and `verify.detection_from_row` turns one
back into a `Detection`. A row written before today has none of those, so
`detection_from_row` falls back to a square box with no landmarks rather than
raising on somebody's old record.

**Two tests were guarding by substring.**
`test_the_sweep_measures_faces_and_never_sees_a_screen` asserted that the word
`screens` did not appear in the evaluation harness source, and
`test_the_oracle_never_runs_in_the_pipeline` searched `faceblur/*.py` for
`torch`. Both fail on a comment, which makes the rule impossible to write
about in the file it guards, and both pass on an import assembled at run time,
which is the case worth catching. They now parse the file with `ast` and check
the module and symbol names actually imported, walking into function bodies
because that is where this codebase puts its expensive imports.

**The oracle trusted its own cache.** A revision pin says which commit to
fetch; it does not say that what is on the disk is still that commit, and a
detector with altered weights does not fail, it answers differently. Two hours
later the answer is in a report attributed to a model nobody can identify.
`eval/oracle_owl.verify_weights` hashes `model.safetensors` before the model
is built and stops if the size or the sha256 does not match the pin. It costs
about a second on 620 MB, once per run.

The pin lives as a constant in `eval/oracle_owl.py` rather than being read out
of `models/README.md` at run time, which is what the build plan asked for.
Parsing a markdown document to decide whether to trust a model is worse than
one constant plus a test that the two agree, so
`test_the_pinned_weights_are_the_ones_the_documentation_names` fails if the
README ever drifts from the code.

**Three documents were stale.** Section 9 said 999 tests. The README's screen
caveats listed what was measured without saying what it means, and now say it
in one sentence: the screen class misses most of the screens an independent
witness sees and holds most copies back for the ones it finds, so it is fit
for best effort masking with the gate off and not for unattended use until S2.
`STATE.md` was 847 lines with the S1 findings in four places, and was
rewritten to about half that, around the rule rather than around the history.

**One behaviour was kept rather than fixed**, and written down instead.
`residual_job` calls the same `screens.detect` the pipeline calls, so
`screen_max_area` applies on the copy side too and a monitor over 12 percent
of the frame is invisible to the check as well as to the run. That is scope
rather than a miss: the pipeline decided not to mask it, and a gate that held
a copy back for something the run was never going to mask would hold back
every file with a table in it. It does mean the check is silent about the one
screen the cap is most likely to be wrong about, and section 16.4 now says so.

## 17. The handled zone (2026-09-11)

The owner's rule of 2026-09-11: *anything the collector is interacting with or
handling is not blurred at all; everything else that carries information is
blurred.* Work package H1 is the first half of that sentence. Every other part
of this pipeline finds things to destroy; `faceblur/zone.py` finds the one
region that is never destroyed, and `redact.redact` subtracts it from every
mask at the last step.

Applying it last is deliberate. Detectors, tracker and holds run exactly as
they did, so the record can say both what would have been masked and what was,
turning the zone off reproduces the old alpha array for array, and a face half
behind a hand loses the half that is behind the hand rather than being dropped
whole.

### 17.1 How the zone is built

MediaPipe's palm detector takes 192 px and the wearer's hands on this camera
run 150 to 500 px, so the whole frame downscaled to 192 puts a hand at 20 px
where the model sees nothing. The frame is covered by 30 sliding windows of
512 px at half overlap, palm boxes are pooled across windows and suppressed at
an IoU of 0.5, and only then does the landmark model confirm each survivor at
`hand_presence` 0.7, the plateau section 14.2 measured. Suppressing before
confirming matters: a palm proposal costs about 2 ms and a confirmation about
7 ms, and a hand lands in four overlapping windows.

A hand qualifies as the wearer's when its palm box is at least
`zone_min_hand_px` (120) across **or** its rectangle reaches the bottom
`zone_edge_frac` (0.25) of the frame. In egocentric footage the wearer's hands
are the closest of their kind and enter from below. Section 17.5 is what that
rule gets wrong.

### 17.2 What it covers, and what it costs

Measured over the four sample files with `--mask face,screen`:

| File | Frames with a zone | Zone share, mean | max | Hands per frame | Mask the zone took back |
|---|---|---|---|---|---|
| `003939` | 2033 of 2033 | 8.98 % | 36.41 % | 1.62 | 0.036 % |
| `004100` | 982 of 984 | 7.17 % | 34.13 % | 1.59 | 0.008 % |
| `004310` | 938 of 938 | 10.12 % | 17.66 % | 2.62 | 0.015 % |
| `005035` | 3971 of 3971 | 10.68 % | 34.84 % | 1.15 | 0.035 % |

**The zone is on every frame of every file.** In egocentric footage the
wearer's hands are essentially never out of shot.

**And it takes almost nothing back out of the masks**: 0.008 to 0.036 percent
of the frame. The face mask share barely moves, from 0.85 to 0.83 percent on
`003939` and not at all on `005035`. Read carelessly that says the zone does
nothing, and it is worth saying plainly what it does say: the track level hand
rules were already keeping masks off hands, which is what section 14 measured
at 0.00 to 0.09 percent of hand pixels. There was almost nothing inside the
zone to take back.

What the zone changes is not how much is protected today but what kind of
statement the protection is. It was three tuned thresholds holding; it is now
structural, and that is what makes package F2 possible at all: every setting
section 10 rejected was rejected for masking the wearer's hand, and the zone
subtracts the hand from the mask whatever the threshold says.

The cost is real. The detect phase roughly doubles, 966 s against 556 s for
the same four files, about 60 ms a frame. `zone_stride` 2 was measured on
`004310`:

| | Detect | Wall | Hands per frame | Zone share | Took back |
|---|---|---|---|---|---|
| `zone_stride` 1 | 108.6 s | 149.4 s | 2.62 | 10.12 % | 0.007 % |
| `zone_stride` 2 | 89.4 s | 130.1 s | 1.30 | 9.41 % | 0.005 % |

Eighteen percent off the detect phase for half the hand evidence, with the
zone share barely moving because memory fills the gaps. That is a bad trade
for the one promise that is absolute, so the default stays at 1.

### 17.3 The precision gate, and two defects it found

The output side check gained a second question: **was anything destroyed
inside the handled zone?** It rebuilds the zone from the source frames, hands
being unmasked in both and the source sharper, and counts pixels that moved
further than the encoder's own noise plus `FLAT_DIFF`. Over
`zone_gate_changed` holds the copy back, for `zone`.

It earned its keep twice before anybody looked at a frame.

**The zone was fading the wrong way.** A shape drawn at its true size and then
blurred is half covered at its own boundary, so masks are drawn grown by the
feather and blurred. The zone was drawn **shrunk** and faded inwards: two
pixels inside the edge of a 200 px square the alpha read 0.137, so 86 percent
of a mask survived in a band about twelve pixels wide all the way around every
hand. That band is the worst place to lose the promise, because it is exactly
where a hand meets the thing it is holding. The gate reported it on three of
the four files. Drawn grown, the smallest alpha anywhere inside a polygon went
from 0.004 to 0.992.

**The check counted the rule working as the rule failing.** Residual faces
went *up* with the zone on, from 7 to 13 on `003939`, because a face the zone
protects stays in the copy and the check called it a miss. A find inside the
zone is now set aside exactly as a hand is: it stays in `residual_list` marked
with the coverage that decided it, and leaves every count the gate reads.
`residual_in_zone` is the count.

The three populations the gate separates, on `004310`:

| Copy | Frames over the gate | Worst frame |
|---|---|---|
| Made with `--no-zone` | 32 | 17.692 % |
| Made with the zone fading inwards | 13 | 1.881 % |
| Made with the zone as it ships | 0 | 0.501 % |

An order of magnitude between a copy that broke the rule and one that keeps
it, which is what makes the gate worth having.

**`zone_gate_changed` is 1.0 percent, and the first guess of 0.5 was a
shipping defect.** At 0.5 the clean `004310` read 0.501 on one frame and was
quarantined for `zone` by a thousandth: a gate that fires on a good copy is
worse than no gate, because the next person turns it off. The four clean
readings and a broken one, which is the distribution the threshold should have
been set from in the first place:

| Copy | Worst frame |
|---|---|
| `003939`, clean | 0.097 % |
| `004100`, clean | 0.099 % |
| `005035`, clean | 0.299 % |
| `004310`, clean | 0.501 % |
| `004310` made with `--no-zone` | 17.692 % |

There is a factor of thirty between the worst clean reading and a copy that
actually broke the rule, and nothing at all in between. 1.0 percent sits in
that gap with a two times margin over the worst clean file and seventeen times
under the broken one. All four files now hold back for nothing.

### 17.4 Is the zone on the right thing

The output side check sorts what it finds into the wearer's hands and faces
the run missed, using MediaPipe's two hand models and three face detectors,
none of which know the zone exists. So it can mark the zone's homework.

**The zone contains 44 of the 52 boxes the hand rule calls hands**, on copies
made before the face region was tightened. The eight it misses are mostly on
`005035`, where 11 of 17 land inside: the table tennis file, where a fist
gripping a bat is the case section 14.5 already records both hand models
failing on. The two mechanisms fail on the same thing, which is worth knowing
before either is trusted alone.

On the copies that actually ship, with `zone_face_scale` at 1.6, the same
comparison reads **13 of 21**. The two are not the same measurement and should
not be quoted against each other: tightening the face region masks more of the
wearer's hands, so fewer hand shaped boxes survive in the copy for the hand
rule to find at all, and the denominator falls from 52 to 21 with the
numerator. The agreement rate falls from 85 to 62 percent, and what that says
is that the boxes which survive are the harder ones.

The build plan asked for the 20 hands and 52 faces of section 14.1 to be
checked against the zone by frame number. **That run's record was never
committed**, as section 16.2 records of its labels, so the comparison above is
rebuilt from a check anybody can re-run instead.

Masked pixels inside MediaPipe's own hand regions, on the copies that actually
shipped, from `eval/zone.py --copy`:

| File | Frames with hands | Hand damage | Worst frame |
|---|---|---|---|
| `004100` | 117 | 0.021 % | 0.209 % |
| `004310` | 183 | 0.078 % | 0.224 % |

Under the sweep's 0.1 percent gate, and **not zero, which the build plan
expected**. The expectation was wrong rather than the code: MediaPipe's hulls
are every hand in the frame and the rule protects the wearer's handled zone,
so a mask landing on a bystander's hand is permitted now where it was not
before. `hand_damage` has therefore changed meaning and is measuring two
things at once. Package F2 should score hulls that intersect the zone
separately from those that do not before it leans on that gate.

### 17.5 The audit, and the leak it found

One hundred frames were rendered with the zone drawn on them and looked at
once, by one person, and every one of the 179 polygons on them carries a
verdict. **The labels are committed**, in
`docs/audits/zone_faces_set_aside_2026-09-11.csv` and
`docs/audits/zone_hands_*.csv`, which is the standing rule that the two
earlier audits in this project broke and the reason neither can be re-scored.

The `zone_hands` files were committed empty by the first pass and filled in by
the second, which is why section 17.7 has a number this section does not.

The informative frames are not the uniform sample. They are the ones where the
check set a **face** aside as handled, because that is the zone declining to
redact somebody. There are 56 of those across the four files; 27 were looked
at, in eight distinct situations.

| Verdict | Cases |
|---|---|
| A bystander's face | 20 |
| The wearer's own hand, called a face by the detectors | 7 |
| Not examined | 29 |

**Twenty of twenty-seven were the zone protecting a stranger's face**, which
is the worst failure this tool can have. Three situations:

- `004310` frames 830 to 836: a canteen worker in a green apron. Their own hand
  rests low in the frame, a 56 px palm whose rectangle dips to y=1016 past the
  edge rule's y=975, so it qualifies as the wearer's. Grown by `zone_scale`, it
  reached 40 px upward and covered their face at 144 px, zone cover 1.0.
- `004310` frame 577: the same worker, another moment, 124 px, fully covered.
- `003939` frames 1083 to 1100: a man washing his hands at a sink. His face is
  beside his own hand, 132 px, 90 percent covered.

The seven correct cases are all the wearer's own open hand being called a face
by the detectors, where the zone and the hand rule agree.

**The fix separates two promises that were never the same.** What the wearer
reaches *over* is what they are holding, so text and screens in it are part of
the task. But a face inside somebody's reach is usually their own face. A face
is therefore protected only where a hand is actually on it, and text and
screens keep the full reach and its memory. `zone_face_needs_hand` reverses it.

That left a second question, because MediaPipe's hand rectangle is already 2.6
palm boxes, which is the crop its landmark model wants rather than the hand's
silhouette: at 2.6 a 56 px palm still makes a 146 px quad that covers a 144 px
face standing beside it. Swept against the 27 labelled cases:

| Palm multiple | Bystander faces still protected | The wearer's hands kept |
|---|---|---|
| 1.0 | 1 of 20 | 4 of 7 |
| 1.3 | 1 of 20 | 6 of 7 |
| **1.6** | **3 of 20** | **7 of 7** |
| 2.0 | 5 of 20 | 7 of 7 |
| 2.6 | 14 of 20 | 7 of 7 |

`zone_face_scale` ships at 1.6, the knee of that table.

**What it actually did, on the copies that ship.** The sweep above scores
coverage on the source frames; running the whole pipeline and the check at 1.6
gives the answer including every other rule:

| | Before the fix | After |
|---|---|---|
| Audited bystander faces the zone protects | 20 of 20 | **1 of 20** |
| Audited hands of the wearer's own the zone protects | 7 of 7 | **4 of 7** |

The leak is closed to one case. It cost three of the wearer's own hands, which
the face detectors call faces and which are now masked because a 1.6 palm box
region does not cover enough of the box to set it aside. That is the same
trade running the other way and it is not free. Masking the wearer's hand is a
precision failure against the rule; it is a smaller one than declining to
redact a stranger, and the track level hand rules and `hand_damage` both say
it stays small in pixels: 0.021 and 0.078 percent of hand pixels.

The one bystander face still protected, and the three hands now masked, are
the same hard case from opposite sides: a face and a hand close enough
together that no region separates them.

**What it costs.** A printed face on a card held in the hand is now masked,
because the card sits in the reach and not under the hand. That reverses
decision D1 of the build plan. It cannot be tested here, because there are no
cards in this footage (section 16.2.1), and the trade was taken deliberately:
masking a mass produced photograph on a card is a smaller harm than leaving a
bystander's real face in a copy. `zone_face_needs_hand=False` gives D1 back.

**One of the twenty is still protected** at 1.6, on the copies that ship, and
it is the hardest case: a face directly beside its owner's own hand, where any
region that covers the hand covers part of the face. Nothing here fixes that,
and it is the honest residual of this package, together with the three of the
wearer's own hands the same tightening now leaves masked.

### 17.6 What this does not do

- **The check tests only the region where the promise is absolute**, which
  since the face fix is the hand itself rather than its reach. A face may now
  be masked inside the reach on purpose, so measuring the reach would report
  the rule working as the rule broken. The reach still protects text and
  screens, and the check cannot rebuild the remembered half of it, because
  memory crosses frame range boundaries and each range runs in its own worker
  with no memory of the one before. What the reach prevented is reported by
  the pipeline instead, as `masked_in_zone_prevented`.
- **The remembered zone drifts.** `motion.estimate_shift` measures a
  translation, so under camera rotation or zoom a remembered polygon slides
  off its object. The memory is three seconds for that reason, and the drift
  is not separately measured.
- **Whose hands is a size and position rule**, not an understanding of who is
  doing what. On a uniform sample it is wrong 12 percent of the time, section
  17.7; on the pre-selected set of faces the zone protected it was wrong 20
  times in 27 before the fix and once after it. Those two numbers measure
  different populations and the uniform one is the one to quote.
- **29 of the 56 faces the zone set aside were not examined.** They are in the
  CSV marked as such.
- **One venue each.** Every number here comes from a washroom, a corridor, a
  canteen and a table tennis hall.

### 17.7 Follow ups (2026-09-11)

Four things the review of section 17 asked for. One was a shipping defect and
is folded into 17.3; the rest are here.

**Whose hands, on frames nobody chose.** Section 17.5 measured the rule on the
faces the zone set aside, which is a set selected for being failures, and 17.6
admitted the unbiased number did not exist. It does now. All 179 polygons on
the 100 uniformly sampled frames were labelled and committed:

| Verdict | Rows |
|---|---|
| The wearer's hand | 153 |
| A bystander's hand | 20 |
| Could not be called | 6 |

**"Whose hands" is wrong on 20 of the 173 rows that could be called, 12
percent**, against 74 percent on the pre-selected set. Both numbers are real
and they are not comparable. Two things in the breakdown are worth keeping:
`004100` contributes three bystander polygons from a single frame, in which a
man bends over at the left and the wearer's own hand on the mop is not covered
at all; and `005035` contributes none, because nobody else comes close enough.

**The wrist to knuckle direction does not separate them.** `hands.rect_for`
already computes the angle that puts the wrist to middle knuckle line upright,
and the idea was that the wearer's hands point up the frame while a bystander
facing them points down. Scored against the 179 labels, by recovering the
angle from each committed quad:

| | n | min | median | p90 | max |
|---|---|---|---|---|---|
| The wearer's hands | 153 | 1 deg | 23 deg | 66 deg | 96 deg |
| A bystander's hands | 20 | 7 deg | 35 deg | 78 deg | 105 deg |

The two ranges lie on top of each other.

| `zone_orientation` | The wearer's hands kept | A bystander's dropped |
|---|---|---|
| 60 deg | 135 of 153 | 6 of 20 |
| 80 deg | 147 of 153 | 1 of 20 |
| 100 deg | 153 of 153 | 1 of 20 |
| 120 deg and up | 153 of 153 | 0 of 20 |

At the 100 degrees that was proposed the rule does nothing; at 60 it buys six
bystanders for eighteen of the wearer's own hands, which is the wrong trade
for a promise that is absolute. **The setting is not shipped.** The
measurement refutes the hypothesis rather than the threshold, so a value that
had no effect would be a knob with nothing behind it. `zone_face_needs_hand`
stays as it is, and the re-sweep that was conditional on this table was not
run.

**The window grid is cut by a fifth, and the first attempt at it was wrong.**
Thirty windows of 512 px at half overlap cover the frame; the wearer's arms
come up from below, so the top rows can go. Cutting them by laying a smaller
grid over what is left moved hands per frame by 44 percent on `005035` while
saving the same time, because every window lands somewhere new and what the
palm model makes of a crop depends on where the hand sits inside it. Skipping
rows and leaving the rest where they were:

| File | 30 windows | 24 windows (0.15) | 18 windows (0.2) |
|---|---|---|---|
| `003939` | 57 ms | 46 ms | 36 ms |
| `004100` | 66 ms | 52 ms | 41 ms |
| `004310` | 69 ms | 52 ms | 40 ms |
| `005035` | 66 ms | 46 ms | 35 ms |

Hands per frame and zone share are identical to two decimal places at every
setting on every file, and re-running the output side check with the cut on
reproduces the previous run exactly: the same faces, screens, set aside counts,
hand agreement and gate readings.

`zone_top_frac` ships at **0.15**, not the 0.2 the numbers allow. At 0.2 the
topmost window starts at y=512 and no palm in the top 39 percent of the frame
can be found at all; at 0.15 it is the top 20 percent. Nothing on this footage
holds anything up to look at it, and the collector footage this tool is for is
people doing exactly that, so the blind region is worth more than the extra 18
percent. The table is here for whoever wants to take it.

One check on the premise: the highest polygon in the audit has its top edge at
y=114, which looks like a hand at the top of the frame and is not. Its palm is
at y=776 and the polygon is grown by nearly four palm widths. All four
settings find that hand.
