# FaceBlur

FaceBlur hides sensitive things in video. Give it a video or a folder of videos.
It writes a copy of each one to an output folder you choose, with the things you
asked for destroyed.

FaceBlur runs on your PC. Your video never leaves the machine.

## What it does and does not do

FaceBlur masks three kinds of thing, and each one is a switch of its own.
Masking faces does not force masking anything else.

| Kind | What it masks | State |
|---|---|---|
| Faces | An oval over the eyes, nose and mouth | Built and measured. On by default |
| Screens | A phone, monitor or television as an object, whatever is on it | Built and measured for precision. Off by default |
| Personal text | Names, addresses, phone numbers and handwriting, leaving card names, prices and grading labels alone | Not built. The switch refuses with a message that says so |

`--mask face,screen` on the command line, or the checkboxes in the window,
choose what a run hides. Nothing is ever on by default except faces: widening
what the tool destroys is a decision on the day.

For faces, FaceBlur masks the smallest region that hides a person's identity.
It leaves hair, hands, bodies and objects alone. The rest of the picture is
untouched, so a blurred copy stays usable as training data.

FaceBlur does not redact number plates or bodies, and it does not strip
container metadata.

FaceBlur does not change audio. It copies the audio track from the source into
the blurred copy, untouched. The four sample files carry none.

FaceBlur may miss a face. Read the next section before you rely on it.

## How well it works

Every number below comes from an automatic evaluation with no human labels. The
method is in `eval/` and the full tables are in `docs/precision_report.md`.
Measured on a 31 second Ego camera file, 938 frames, 1600x1300.

| Measure | First build | This build (2026-09-08) |
|---|---|---|
| Share of the frame destroyed, mean | 33% | 2.3% |
| Share of the frame destroyed, worst frame | 93% | 5.8% |
| Masked pixels where no detector sees a face, mean | 29.7% | 0.77% (0.29% from detector boxes, the rest tails and gap fills) |
| Hand pixels touched | 82% | 0.00% (0.09% on the table tennis file) |
| Faces the detectors agree on, covered | 100% | 99.7% |
| Confirmed faces kept covered while still visible | | 99.5% |
| Largest 60 faces still recognised by a face recogniser after the mask | | 2 of 60 |

Recall against faces of known position, pasted into real frames of the same
video, with the same settings; the set now includes faces entering at the
frame edge and faces during a camera pan:

| Faces | Covered |
|---|---|
| 64 px and larger | 83% |
| 32 to 63 px | 79% |
| under 32 px | 65% |
| sharp, any size | 86% |
| motion blurred, any size | 65% |
| entering at the frame edge | 81%, masked 2.6 frames after half visible |
| during a camera pan | 86% |

Small and motion blurred faces carry the misses. The detectors themselves find a
24 px face about half the time, at any threshold. On a 1600 px frame a 24 px face
is a person far across the room.

This file is the hardest of the four: the same measurement on the other two
files with pseudo-labels covers 88% and 86% of pasted faces against 79% here.

Coverage is a proxy for the thing that matters, so this build also measures the
thing itself: a face recogniser (SFace) embeds every face of the source and the
same rectangle of the blurred copy. The mask defeats it on the faces it can
recognise at all, and the shipped strength of six blocks across a face is the
setting that does so — twelve blocks leaves ten of sixty large faces still
recognisable. On this footage the recogniser identifies people at
conversational distance and mostly fails on faces across a room, which bounds
what the misses below can give away. `docs/report.md` section 12 has the
numbers and what they do not prove.

The first build reached 100 percent on the agreed faces by masking a third of
every frame. This build masks 2 percent and keeps hands untouched; the misses
left are small and blurred faces. `docs/precision_audit.md` explains the first
rebuild, `docs/report.md` sections 10 and 11 the two passes since.

Check the blurred copies before you share them.

## How it decides what is a face

Pixels are destroyed only when several independent checks agree:

1. YuNet finds a candidate at a threshold of 0.6, scanning at 1280 and 1920 px.
2. CenterFace, a different architecture, must fire on the same spot. It looks
   at a crop around each candidate rather than the whole frame. A crop that
   runs past the frame edge is filled with the picture reflected, so a face
   half out of the picture still sits in context. CenterFace also sees the
   crop's mirror image and a tighter crop; those views count only when they
   are more sure (0.4) than the plain view needs to be (0.3), because every
   extra view is an extra chance for a hand to slip through.
3. When CenterFace is unsure, scoring between 0.25 and 0.3, a third detector
   of a third family, UltraFace, breaks the tie on the same crop. It is never
   asked about a box CenterFace scored below 0.25, which is where hands and
   signs land: UltraFace itself fires on hands, so it can only ever confirm,
   never overrule.
4. Two detectors agreeing count as one being sure, for small boxes: a YuNet
   box of 48 px or less at 0.35 that CenterFace also scores at 0.35 is a face
   (mirror reflections, far faces). Hands are large, so they never take this
   road.
5. The box must be a plausible face: at most 15 percent of the frame's long
   side, and roughly square. A confirmed track may follow a face that grows
   past that, to 35 percent, as a person walks up to the camera; a box that
   large can never start a track.
6. A tracker links detections across frames. A face must be seen at least twice,
   close together, before any pixel is touched. The camera's own movement
   between frames, measured by phase correlation, is taken out of the
   prediction, so a pan does not break a track, and a detection may join a
   track by centre distance when a small fast face has no overlap frame to
   frame. Gaps of up to five frames are interpolated, ten while the camera
   moves fast. Two tracks of one face up to 45 frames apart, where the
   detectors lost it (a hit at table tennis smears the picture for a second),
   are joined and the gap interpolated, following whatever YuNet still saw of
   the smear. Every mask grows by the camera's shift while it moves fast.
   A track becomes established after five confirmations, one of them sure
   (CenterFace 0.5 or more); only then may YuNet alone keep it going at a
   lower threshold, forwards and backwards in time, and only then does the
   mask reach before the first sighting (six frames, twelve for a face moving
   in from the edge) and after the last (four). A track below that masks its
   confirmed frames and the gaps between them, nothing more: the wearer's
   own hand gets confirmed two or three frames at a time, and used to be
   masked for thirty frames around them.
7. The mask is an ellipse fitted to the box and rotated to the eye line, with a
   soft edge. Inside it the pixels are replaced from a copy shrunk to six blocks
   across, so the face cannot come back.

The thresholds were chosen by `eval/sweep.py`: among settings that keep
masking outside anything a detector calls a face under 0.7 percent of the frame
(0.3 percent counting detector boxes alone, the rest being tails and gap
fills) and hands untouched, the fewest frames with a known face of 40 px or
more left visible, then the highest recall.

Every blurred copy comes with an audit record that says how much of each frame
was destroyed and flags any frame over 5 percent.

## How it decides what is a screen

Screens need none of that machinery. There is no identity to weigh: whatever is
on the glass is hidden, so the only question is where the glass is. One object
detector, YOLOX-tiny, answers it, and two rules keep it honest.

1. **A screen has to be seen in three frames** before any of it is masked
   (`screen_min_run`). A COCO detector calls any large flat rectangle a
   television, and on this footage the blue table tennis table is called a
   laptop, a television and a phone by turns. It calls a table one for a frame
   and then stops; a monitor on a wall stays a monitor. On the sample footage
   14 of the 29 false runs last a single frame and not one real screen does.
2. **A box over 12 percent of the frame is not a screen** (`screen_max_area`).
   The table reads 0.88 over a third of the frame and the real television
   across the hall reads 0.85 over one percent, so the score separates nothing
   and the size does.

Both rules are needed. A size cap alone has to sit at 4 percent to reach the
same precision, and at 4 percent it throws away a real monitor covering 10.6
percent of the frame, which is the worst screen to lose: one that large and
that close is the one most likely to show something a person could read. With
persistence carrying the precision the cap can sit three times looser and keep
that monitor.

Four things to know before you switch screens on:

- **Recall is low, and it is the detector rather than the rules.** Measured
  against an independent witness (OWLv2, evaluation only), the screen class
  masks 0 to 23 percent of the screens that witness sees. The size cap costs
  none of that and the persistence rule costs 2 percent: what limits it is
  that YOLOX-tiny at a confidence of 0.5 does not report most of them. Both
  numbers are one model's opinion against another's, so read them as bounds.
  `docs/report.md` section 16.2 has the tables.
- **9 percent of what it masks is not a screen.** The survivors are small
  patches: 2.4 percent of a frame of washroom wall for 10 frames, 0.4 percent
  for 27. They are cheap in pixels and they are still wrong.
- **The rules are tuned on one venue.** The table tennis hall is what makes the
  size cap necessary and what sets its value. A room with a large monitor close
  to the camera and no large flat furniture would want a looser cap, and there
  is no footage here to set one on.
- **A screen over the size cap is invisible to the check as well.** The check
  runs the same detector with the same 12 percent cap, so a monitor larger
  than that is neither masked nor reported. That is scope, not a miss, but it
  is the screen the cap is most likely to be wrong about.
- **A screen the check reports may be a table.** `--check-output` now reads the
  copy for screens as well as faces, and it inherits the detector's weakness
  the way it inherits the face detectors': it reports what YOLOX calls a
  screen. The gate asks for persistence before it acts on one, and the record
  names every find either way.

The screen mask destroys 0.01 to 0.34 percent of the average frame on the
sample files, and at most 12 percent of one, which is a large monitor properly
masked. `docs/report.md` section 15 has the tables.

**What sections 16.1 and 16.2 mean together, in one sentence: the screen class
misses most of the screens an independent witness sees, and holds most copies
back for the ones it does find.** So it is fit for best effort masking with the
gate off, and it is not fit for unattended use with the gate on, because it
will quarantine nearly everything while still leaving screens in the copies it
releases. Switching screens on is worth doing; relying on it is not, yet.

## Set up

You need Python 3.12 on Windows.

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You do not install ffmpeg. The `imageio-ffmpeg` package carries it.

## Use the window

```
.venv\Scripts\python.exe -m ui.app
```

Drop a video or a folder on the window. Choose an output folder. Press Blur
faces. Each row shows one video and its progress. Press Stop to end the run. Stop
deletes unfinished files.

## Use the command line

```
faceblur INPUT [-o OUTPUT] [--mask face,screen]
         [--engine yunet|centerface|both] [--conf F]
         [--det-sizes 1280,1920] [--stride N] [--no-verify] [--tta 0|1|2]
         [--no-third] [--no-camera] [--max-face F] [--min-track N]
         [--max-gap N] [--tail N] [--pad F]
         [--mode blur|pixelate|solid] [--workers N] [--device auto|gpu|cpu]
         [--encoder auto|nvenc|x264] [--chunk-seconds S] [--no-copy]
         [--hwaccel none|cuda]
         [--check-output] [--check-stride N] [--quarantine] [--quarantine-px N]
         [--no-hand-rule] [--report PATH] [--recursive] [--no-progress]
```

```
.venv\Scripts\python.exe cli.py C:\ego -o C:\ego_blurred --workers 8
```

INPUT is a video or a folder. A folder run reads every video in it. OUTPUT
defaults to a folder named `<input>_blurred` next to the input. The exit code is
1 when any video failed.

`--mask` chooses what to hide. Faces are the default. `screen` is built and can
be added, as `--mask face,screen`. `text` is named in the interface and refused
with a message that says so, because a switch that masks nothing is worse than
no switch. The window shows the same three as checkboxes, with the one that is
not built switched off and labelled.

`--no-verify` and `--engine both` find more faces and also blur hands and
objects. On the Ego footage they touched 7.7 percent of hand pixels. Use them
only when a missed face costs more than a damaged frame.

`--check-output` runs the detectors over the finished copy as well. A face
found there, on pixels the run never changed, is a face the run missed, and
nothing that reads the source can see it. It costs a second detection pass,
about the same again as the first: 32 seconds for a 33 second file on this
PC. `--quarantine` turns that into a gate: a copy that still shows a face of
24 px or more (`--quarantine-px`) is moved into a `quarantine` folder beside
the output instead of shipping, with its audit record naming the frames. Use
both for anything that leaves the machine.

The check looks for what the run was asked to mask. With `--mask face,screen`
it reads the copy for screens too, at about 2 to 6 percent on top of the face
check, and the record says how many are left and where. A screen holds a copy
back only when it was **there**: the finds are grouped into runs the way the
pipeline groups them, and a run has to last `screen_gate_min_run` checked
frames and clear `screen_gate_min_px` on its long side. One frame of a table
that looked like a laptop holds nothing back, and without that rule every
copy would be held. A face needs no such rule, because one frame of a face is
a face. The record's `held_back_for` names the kinds that stopped a copy, and
the command line prints them.

The check uses the same detectors as the rest of the pipeline, and they call
the wearer's own hand a face. So before it reports anything it asks
MediaPipe's hand models, which ship with the build, whether the box is a hand;
one that is stays in the record with the coverage that decided it and does not
hold the copy back. `--no-hand-rule` turns that off. On the four sample files
the rule sets aside 13 of the 20 hands and none of the 52 faces.

Read what is left with the size in mind. The check finds 108 boxes over 7926
frames of the sample footage, and every one of them was looked at by eye: 52
are faces the run really did miss, 20 are the wearer's hand, 30 are neither —
a television, a picture on a wall, a motion smear — and 6 could not be called.
`docs/report.md` sections 13 and 14 go through them. Faces outnumber hands two
and a half to one, so a copy the gate holds back is usually being held back
for the right reason.

## What it writes

- `<name>_blurred.mp4`, the blurred copy, H.264 at crf 12 with the source audio.
- `<name>_blurred.mp4.json`, the audit record: frames, tracks, how much of each
  frame was masked (mean, p95, max), frames over the mask budget, the same
  three split by kind, every setting, the sha256 of each model, and the time
  taken. Read the split when more than one kind is on: a screen mask is large
  by design, and on the table tennis file the union says 40 frames are over
  the 5 percent budget where the face mask puts 9 of them there. With screens on it also holds
  `screens`, the detections the model made, and `screen_frames`, the frames a
  screen mask reached after the persistence rule. After `--check-output` it
  holds what a second pass over the copy found: frames checked, faces still
  visible by size, the frame stretches they sit in, how many of the boxes it
  found were the wearer's hand, the screens still visible and the runs they
  form, and `held_back_for`, the kinds that stopped the copy shipping. It
  keeps up to 200 rows of each kind, and every count in it is over every row.
- `quarantine\<name>_blurred.mp4` and its record, when `--quarantine` held a
  copy back. The file is not deleted: it is the only blurred copy of that
  video, and the record says which frames stopped it.

## Evaluate a new batch

The harness needs a second environment for MediaPipe, which supplies the third
face detector and the hand regions, and a third for the object oracle, which
is what gives screens a recall number:

```
py -3.12 -m venv .venv-eval
.venv-eval\Scripts\python.exe -m pip install -r requirements-eval.txt

py -3.12 -m venv .venv-oracle
.venv-oracle\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
.venv-oracle\Scripts\python.exe -m pip install -r requirements-oracle.txt
```

Three environments rather than two because MediaPipe pins numpy below 2 and
torch wants a newer one, and every face number in the report was measured
through MediaPipe. Loosening that pin to save disk would put the face side's
evidence at risk for nothing.

Then, for a video:

```
.venv-eval\Scripts\python.exe eval\oracle_mediapipe.py VIDEO --stride 5
.venv\Scripts\python.exe -m eval.consensus VIDEO --stride 10
.venv\Scripts\python.exe -m eval.sweep VIDEO
.venv\Scripts\python.exe -m eval.misses VIDEO --images SOME_FOLDER
.venv\Scripts\python.exe -m eval.reid VIDEO BLURRED_COPY
.venv\Scripts\python.exe -m faceblur.verify VIDEO BLURRED_COPY --workers 4
.venv-oracle\Scripts\python.exe eval\oracle_owl.py VIDEO --stride 5
.venv\Scripts\python.exe -m eval.screens VIDEO
```

`eval.oracle_owl` runs an open vocabulary detector (OWLv2, Apache 2.0) that
knows nothing about this pipeline, and writes what it sees as boxes. It is
slow, about 4 seconds a frame on the processor, and it never ships and never
runs in the pipeline. `eval.screens` scores the shipped screen class against
it, which is the only recall number screens have.

`eval.misses` lists the stretches where YuNet saw a box at full threshold that
nothing confirmed and no mask covers, with one annotated frame each. The same
list is in every audit record as `unconfirmed_runs`. Most are hands and
objects; a change that finds more faces makes the count fall while the gates
hold.

`faceblur.verify` is the same check `--check-output` runs, on copies that are
already written. It needs no second environment, no labels and no oracle: it
detects on the copy and reports the boxes nothing was done to, having first
asked the hand models which of them are the wearer's own hand. `--mask
face,screen` tells it what the run was asked to mask, so that a copy nobody
asked to have screens masked in is not reported as missing one. Its exit code
is 1 when anything is left.

`eval.reid` asks the question coverage only stands in for: can a face
recogniser still put the same name to anybody in the blurred copy. It reports
where the recogniser's threshold sits on this footage, what the mask does to a
face it can see, and which sightings of the finished copy still match the
source. `docs/report.md` section 12 explains what those numbers can and cannot
prove.

The sweep writes `docs/precision_report.md` and prints the settings that pass
the gates. No step asks a person for anything.

## Speed

FaceBlur uses the GPU when it finds one. Both detectors run through onnxruntime
with DirectML on Windows, so any DirectX 12 card works, and with CUDA on a cloud
machine. Encoding uses NVENC on an NVIDIA card and x264 otherwise, at matching
quality.

Measured on a 1600x1300 file, per frame:

| Path | Detection | Notes |
|---|---|---|
| CPU, first precision build | 700 ms | both detectors, whole frame, two sizes |
| CPU, this build | 350 ms | confirmation on crops around candidates |
| GPU, this build | 110 ms | RTX 3070 through DirectML, three confirmation views and the third detector |

End to end on the four sample files (264 s of video) with four workers: 469 s,
about 1.8 seconds of processing per second of video. The busiest file (163
tracks, every frame masked) runs at 2.6 to 1. The window uses the same pool.
`docs/report.md` has the table.

Three things make a batch scale:

- **Frame ranges.** Detection runs on ranges of `chunk_seconds` (default 15) in
  parallel worker processes. Ranges need no overlap, because the tracker runs
  once over the whole timeline afterwards.
- **Copied stretches.** A group of pictures with no mask is copied from the
  source byte for byte, so it keeps its original quality and costs nothing to
  encode. Only stretches that hold a mask are re-encoded, and long ones are cut
  at keyframes so several workers share them. Copied and encoded pieces only
  join exactly when they carry the same frame reordering delay, so the encoder
  writes no B-frames and a source that has them is encoded whole. The audit
  record's `reorder_delay` says which happened.
- **A verified join.** The pieces are joined at keyframes with the source
  audio. The result is checked for a clean demux, the exact frame count, forward
  timestamps and the source's length. If a join with copied pieces fails that
  check, the video is encoded end to end instead, without asking anyone.

`--workers N` sets the worker processes; the default is half the cores, and
at most four when they share a GPU, since each holds every model's graphs
(about 850 MB on an RTX 3070) and the desktop keeps about 2 GB of the card. `--chunk-seconds`, `--no-copy`,
`--device cpu` and `--encoder x264` turn each piece off when you need to. The
audit record says which processor each model ran on and how many frames were
copied untouched.

Detecting on every second frame (`--stride 2`) halves detection time. The
automatic sweep measures what it costs in recall on your footage and only picks
it if it stays inside the gates.

## Build the packaged app

```
.venv\Scripts\python.exe -m PyInstaller build\faceblur.spec --noconfirm
```

The result is `dist\FaceBlur\`. The evaluation harness is not part of it.
DirectML travels with the onnxruntime package, so the packaged app uses the GPU
too.

## Models

| Name | Model | Licence | Size | What it is for |
|---|---|---|---|---|
| `yunet` | YuNet from opencv_zoo | Apache 2.0 | 232 KB | finds face candidates |
| `centerface` | CenterFace from the deface package | MIT | 7.3 MB | confirms them on a crop |
| `ultraface` | UltraFace RFB-320 from Linzaer | MIT | 1.3 MB | breaks ties when CenterFace is unsure |
| `yolox` | YOLOX-tiny from Megvii | Apache 2.0 | 20 MB | finds screens, when screens are switched on |
| `palm_detection` | MediaPipe palm detector | Apache 2.0 | 4.6 MB | proposes a hand in the output side check |
| `hand_landmark` | MediaPipe hand landmark model | Apache 2.0 | 11 MB | confirms it is a hand |

`models/README.md` records where each file came from, its sha256, its input
layout and how its output decodes. `models/sface.onnx`, the face recogniser, is
evaluation only and never enters the packaged app; a test fails if it does.

## Versions

These exact versions passed the tests. `requirements.txt` pins every one.

| Package | Version |
|---|---|
| Python | 3.12.8 |
| opencv-python-headless | 4.13.0.92 |
| numpy | 2.5.2 |
| onnx | 1.17.0 |
| onnxruntime-directml | 1.24.4 |
| imageio-ffmpeg | 0.6.0 (ffmpeg 7.1, with NVENC) |
| PySide6 | 6.11.2 |
| pytest | 9.0.2 |
| pyinstaller | 6.22.2 |

The evaluation harness needs `requirements-eval.txt` in its own environment,
because MediaPipe pins numpy below 2.

## Run the tests

```
.venv\Scripts\python.exe -m pytest tests\
```

## Report and handover

`docs/report.md` holds the measurements: quality, accuracy, speed, the hardware
they were taken on, the assumptions, and what was done and not done on
purpose. `STATE.md` says where the project stands for whoever picks it up.

## History

`FACEBLUR_BUILD_PLAN.md` is the original plan. `docs/recall_report.md` is the
Phase 3 measurement of the first build, with hand labels. `docs/precision_audit.md`
is the audit that found why that build destroyed hands and counters, and
`docs/precision_report.md` is the automatic measurement of this one.
