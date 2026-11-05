# FaceBlur

FaceBlur hides sensitive things in video. Give it a video or a folder of videos.
It writes a copy of each one to an output folder you choose, with the things you
asked for destroyed.

FaceBlur runs on your PC. Your video never leaves the machine.

## What it does and does not do

One rule decides everything: **whatever the wearer is handling is never
touched, and everything else that carries information is destroyed.** A card
or a phone in their hands stays readable because that is the task being
recorded. A sign on the wall behind them does not.

FaceBlur masks three kinds of thing, and each one is a switch of its own.
Masking faces does not force masking anything else.

| Kind | What it masks | State |
|---|---|---|
| Faces | An oval over the eyes, nose and mouth | Built and measured. On by default |
| Screens | A phone, monitor or television as an object, whatever is on it | Built and measured for precision. Off by default |
| Personal text | Writing on anything the wearer is not handling, whatever it says | The detector is built and measured, nothing masks text yet. The switch refuses with a message that says so |

`--mask face,screen` on the command line, or the checkboxes in the window,
choose what a run hides. Nothing is ever on by default except faces: widening
what the tool destroys is a decision on the day.

For faces, FaceBlur masks the smallest region that hides a person's identity.
It leaves hair, bodies and objects alone. The rest of the picture is
untouched, so a blurred copy stays usable as training data.

## The handled zone

Around the wearer's hands, and around what those hands are holding, nothing is
ever masked. That region is found on every frame by two hand models and
subtracted from every mask at the last step, so it is not a threshold that
usually holds but a region that is taken back out. It stays for about three
seconds after a hand leaves, moving with the camera, so a thing put down on a
table is still protected while it is being worked through.

On the sample footage it covers 7 to 11 percent of each frame, and the wearer's
hands are in shot on essentially every frame of every file. It costs about 45
ms a frame, which roughly doubles the detection pass; `--no-zone` turns it off
and is for measuring what it costs, not for a copy that ships.

Whose hands it counts as the wearer's is a rule about size and position, not
an understanding of who is doing what, and on a uniform sample of 179 hands it
is wrong 12 percent of the time. The labels behind that are committed under
`docs/audits/`.

Two things it does **not** promise, both measured and both in
`docs/report.md` section 17:

- **A face is protected only where a hand is actually on it**, not everywhere
  the hand can reach. Without that split the zone was found protecting
  bystanders' faces from redaction in 20 of 27 cases that were looked at by
  eye. One of those twenty is still protected: a face directly beside its
  owner's own hand, where no region separates the two. The cost of the fix is
  that three of seven of the wearer's own hands, which the face detectors
  mistake for faces, are now masked.
- **A printed face on a card held in the hand is masked**, because the card
  sits in the hand's reach rather than under the hand itself. Set
  `zone_face_needs_hand` to false to protect it instead, and read section 17.5
  before doing so.

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

**Recreate the whole folder.** Choose one folder and tick this box. The app
makes a copy of the folder, with the same name, inside the output folder: every
subfolder, including empty ones, and every file, under the same names. Each
`.mp4` in it, at any depth, is replaced by its blurred copy under the original
name. Every other file, including other video formats, is copied byte for byte.
Nothing is added inside the copy: the audit records go in one
`<folder>_faceblur_report.json` beside it. Two rules follow from "same names":

- A copy the check does not trust stays in place instead of going to
  `quarantine`, and the summary and the report name it.
- The original of an MP4 is never copied in, not even when blurring it fails.
  That file is missing from the copy and its row says why.

The app refuses an output folder inside the chosen folder, and one that would
put the copy on top of the original.

**Workers** is capped at 4 when the models run on the GPU. Each worker loads
every model, so more than that only uses memory. Before 2026-09-26 the window
allowed any number up to the core count and remembered 10. Two busy workers
already take this PC from 9 GB free to under 2 GB, and a worker that died
left the run waiting for it forever. A worker that dies now fails that one
video with a message, and the next video gets a new pool.

## Use the command line

```
faceblur INPUT [-o OUTPUT] [--mask face,screen]
         [--engine yunet|centerface|both] [--conf F]
         [--det-sizes 1280,1920] [--stride N] [--no-verify] [--tta 0|1|2]
         [--no-third] [--no-camera] [--max-face F] [--min-track N]
         [--max-gap N] [--tail N] [--pad F]
         [--mode blur|pixelate|solid] [--workers N] [--device auto|gpu|cpu]
         [--encoder auto|nvenc|x264] [--chunk-seconds S] [--no-copy]
         [--hwaccel none|cuda] [--no-zone]
         [--check-output] [--check-stride N] [--quarantine] [--quarantine-px N]
         [--second-chance N] [--no-hand-rule] [--report PATH] [--recursive]
         [--no-progress]
```

```
.venv\Scripts\python.exe cli.py C:\ego -o C:\ego_blurred --workers 8
```

INPUT is a video or a folder. A folder run reads every video in it. OUTPUT
defaults to a folder named `<input>_blurred` next to the input. The exit code is
1 when any video failed.

`--no-zone` masks inside the handled zone as well. It exists to measure what
the zone costs and a copy that ships should never be made with it.

`--mask` chooses what to hide. Faces are the default. `screen` is built and can
be added, as `--mask face,screen`. `text` is named in the interface and refused
with a message that says so, because a switch that masks nothing is worse than
no switch. The window shows the same three as checkboxes, with the one that is
not built switched off and labelled.

`--no-verify` and `--engine both` find more faces and also blur hands and
objects. The handled zone takes the wearer's hands out of every mask and it
is not enough to make these safe: it protects a face only where a hand is on
it, and it only knows the hands its own palm detector found. Measured again
with the zone on, they put masks on 0.39 percent of the wearer's hand pixels
on one sample file and 1.41 percent on another, against a limit of 0.1
(report section 19). Use them only when a missed face costs more than a
damaged frame.

`--check-output` runs the detectors over the finished copy as well. A face
found there, on pixels the run never changed, is a face the run missed, and
nothing that reads the source can see it. It costs a second detection pass,
about the same again as the first: 32 seconds for a 33 second file on this
PC. `--quarantine` turns that into a gate: a copy that still shows a face of
24 px or more (`--quarantine-px`) is moved into a `quarantine` folder beside
the output instead of shipping, with its audit record naming the frames. Use
both for anything that leaves the machine.

While a copy is being checked it is named `<name>.mp4.unchecked`, and it takes
its real name, or goes to `quarantine`, only once the check has spoken. A file
with that ending in an output folder is one a run did not finish checking:
it was never delivered and can be deleted.

`--second-chance N` uses what the check found. A face it finds in the copy
is a face this run missed, on pixels the run never changed, and it goes back
in as a detection: the file is written again from the source, never from the
copy, and checked again. No threshold is lowered. It costs another write and
another check, about the same again as the first pass. On one sample file it
took the faces still visible in the copy from 8 to 1; on another, from 13 to
9, because most of what is left there is a face a mask already partly covers
and putting the same box back cannot help. Report section 20.

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

The storage tools in `storage/` shrink finished copies for hot storage:
`python -m storage.convert FILE [--apply]` re-encodes to HEVC (NVENC cq 28, or
libx265 crf 22), checks and replaces only with `--apply`; `storage.manifest TREE`
lists a tree in Parquet; `storage.calibrate DIR` measures the size-vs-VMAF curve;
`storage.randaccess COPY --source H264` checks random seeks; `storage.csv2parquet
--tree DIR [--out DIR] [--apply]` turns IMU and pts CSVs into Parquet byte for
byte; `cli.py ... --codec hevc` writes the blurred copy as HEVC directly. See
`docs/hot_storage_convert.md` and `docs/report.md` section 27.

## What it writes

- `<name>_blurred.mp4`, the blurred copy, H.264 at crf 12 with the source audio.
- `<name>_blurred.mp4.json`, the audit record: frames, tracks, how much of each
  frame was masked (mean, p95, max), frames over the mask budget, the same
  three split by kind, every setting, the sha256 of each model, and the time
  taken. It also says whether the source had sound and lists the **names** of
  the metadata tags the source carried, never their values, so that a reader
  who cannot be handed the footage can still answer both questions. The
  blurred copy carries none of those tags: a location written in a container
  survives any amount of blurring, and there is a test that it does not
  survive this (report section 23). Read the split when more than one kind is on: a screen mask is large
  by design, and on the table tennis file the union says 40 frames are over
  the 5 percent budget where the face mask puts 9 of them there. With screens on it also holds
  `screens`, the detections the model made, and `screen_frames`, the frames a
  screen mask reached after the persistence rule. It always holds the handled
  zone's share of the frame, the hands found per frame, and
  `masked_in_zone_prevented`, the share of the frame the zone took back out of
  the masks. After `--check-output` it
  holds what a second pass over the copy found: frames checked, faces still
  visible by size, the frame stretches they sit in, how many of the boxes it
  found were the wearer's hand, the screens still visible and the runs they
  form, and `held_back_for`, the kinds that stopped the copy shipping. It
  keeps up to 200 rows of each kind, and every count in it is over every row.
- `quarantine\<name>_blurred.mp4` and its record, when `--quarantine` held a
  copy back. The file is not deleted: it is the only blurred copy of that
  video, and the record says which frames stopped it.

## The HDF5 module

The same pipeline also runs as the privacy stage of the partner's
post-processing chain, where a recording arrives as one HDF5 file in their
DF2 layout instead of a folder of mp4s. `faceblur/h5.py` is that stage.

### The command

```
.venv\Scripts\python.exe -m faceblur.h5 pack CAM1.mp4 CAM2.mp4 [--cameras a,b] [--audio SOUND] --out IN.h5
.venv\Scripts\python.exe -m faceblur.h5 run IN.h5 --out FOLDER [--spec-layout] [--workers 2] [--mask face,text,plate,screen] [--device auto|gpu|cpu] [--check-output]
.venv\Scripts\python.exe -m faceblur.h5 verify FOLDER\audit.jsonl
.venv\Scripts\python.exe -m faceblur.h5 selfcheck [--device auto|gpu|cpu]
```

- `pack` builds a DF2 file from camera mp4s. Without `--cameras` the files
  are named in the order the head-worn cameras come in
  (`rgb_ego_center_left`, `rgb_ego_center_right`, `rgb_ego_side_left`,
  `rgb_ego_side_right`, then the rest).
- `run` writes `episode.h5` and `audit.jsonl` into `--out` and never touches
  the input. By default `episode.h5` is a separate release file, so there
  are two files in the end, the untouched input and the release;
  `--spec-layout` instead writes a copy of the input with the de-identified
  cameras added beside the raw ones, the single-file form the partner's spec
  describes. `--workers` defaults to 2: each worker holds every model on the
  GPU, text at 2560 px and the plate model add to that, and the 8 GB card on
  this PC holds two. `--device` and `--check-output` mean what they mean on
  the command line above. The output is written under a temporary name and
  renamed only when every camera finished, so a failed run leaves no
  `episode.h5` that looks complete.
- `verify` checks an audit file's hash chain and every line against the
  schema.
- `selfcheck` loads every masking model, runs it once on a flat frame and
  prints the processor each one ran on.

Exit codes: `run` returns 0 when done, 1 when a camera did not finish and 2
when the input has no camera; `verify` and `selfcheck` return 1 on any
problem.

In Docker the stage is one command (`docs/privacy_module_report.md`, section
Docker image, has the build):

```
docker run --gpus all -v <in>:/in:ro -v <out>:/out faceblur-privacy
```

It checks the weights against `models/SHA256SUMS` first and stops on a
mismatch, runs every `/in/*.h5` and `/in/*.hdf5` (one input writes
`/out/episode.h5`, several write `/out/<stem>/`), then verifies each audit.
`-e FACEBLUR_WORKERS=N` sets the workers (default 2), `-e FACEBLUR_ARGS="..."`
adds run flags such as `--spec-layout --mask face`, and
`-e FACEBLUR_IMAGE_DIGEST=$(docker inspect --format '{{.Id}}' faceblur-privacy)`
names the image in the audit. The container exits 2 when `/in` holds no
input and 1 when a run failed or an audit did not verify.

### The input layout

The module reads the DF2 tree. Cameras are groups under
`dataset/observation/image/<camera>/`, with twelve names in the partner's
order: `rgb_wrist_left`, `rgb_wrist_right`, `rgb_wrist_left_up`,
`rgb_wrist_left_down`, `rgb_wrist_right_up`, `rgb_wrist_right_down`,
`rgb_ego_center_left`, `rgb_ego_center_right`, `rgb_ego_side_left`,
`rgb_ego_side_right`, `ir_ego_left`, `ir_ego_right`. Any group there that
holds `video` is processed, the known names first and in that order, an
unknown name after them.

A camera group holds `video`, the encoded H.264 packets exactly as demuxed
from the camera's mp4, one variable-length uint8 element per packet, so the
file stays the size of the mp4 and the original can be remuxed bit for bit.
Beside it are `pts`, `dts`, `keyframe`, `timestamps_ns` (one per packet),
`intrinsics` (3x3) and `extrinsics` (4x4), and attrs for the codec, width,
height, fps, time base, pixel format, the avcC extradata and the frame
count. Audio, when present, is `dataset/observation/audio/pcm`, int16
[samples, channels], with `sample_rate` and `channels` attrs. Everything
else in the file (meta, state, env, timeline, IMU, EMG, mocap) is never
interpreted and is passed through.

The partner's spec gives the group tree but not what sits inside a camera
group, so the names inside are this project's choice. Every path and name
lives in one module, `faceblur/df2.py`, which is the only file to change
when the partner's sample arrives and differs. No partner sample exists yet,
which is why `pack` exists: the tests and the measurements run on files it
packed from the four sample mp4s.

### The outputs

`episode.h5`, the default release file, holds the de-identified video of
every camera the input had, in the same group form, and every other group of
the input copied through unchanged with its attrs. It holds no audio
dataset, no raw video and no external links; a test checks all three. Its
root attrs `faceblur_release`, `faceblur_version` and `faceblur_layout`
(`two-file` or `spec`) say what wrote it. With `--spec-layout` the file is a
copy of the input, raw cameras and audio included, with the de-identified
cameras under `dataset/observation/image_deidentified/`; it is for
integration tests with the partner, not for release.

What is masked, with the default `--mask face,text,plate,screen`:

- Faces are filled solid. A blurred or pixelated face was still matched by a
  face recogniser after a deblur attempt and a solid one was not
  (`docs/privacy_module_report.md`, section Face obfuscation against a
  deblur attempt).
- Text and plates are filled solid, because a blurred line keeps its length
  and its word breaks.
- Screens follow `mode`, blur by default.
- The handled zone is never masked, whatever is found inside it.

Speed, on a file packed from the four sample mp4s (1600x1300, 30 fps), two
workers, DirectML on the RTX 3070:

| Camera | Frames | Time | Per frame | Masked, mean |
|---|---|---|---|---|
| rgb_ego_center_left | 2033 | 511 s | 251 ms | 1.80 % |
| rgb_ego_center_right | 984 | 242 s | 246 ms | 2.00 % |
| rgb_ego_side_left | 938 | 267 s | 284 ms | 3.52 % |
| rgb_ego_side_right | 3971 | 1017 s | 256 ms | 2.20 % |
| all four | 7926 | 2050 s | 259 ms | |

Cameras run one after another, so the times add. At about 260 ms a frame,
one hour of one 30 fps camera takes about 7.8 hours on this PC, and an hour
with four cameras about 31 hours. The 575 MB input made a 924 MB
`episode.h5`: no stretch was copied untouched, because text and faces reach
nearly every frame, and every re-encoded frame is written at high quality.
Text detection at two sizes is the costliest detector, 157 ms a frame
on its own (report, section Text).

### The audit record

`audit.jsonl`, beside `episode.h5`, records what was removed and by what.
The partner's chain keeps the release and discards the input, so this file
is the only evidence left. It holds one JSON object per line, written and
flushed as the run goes. Every line carries `seq`, `prev`, `hash`, `type`
and `time`, where `hash` is the sha256 of the previous line's hash followed
by this line in canonical JSON. Changing, removing or reordering a line
breaks every hash after it, and a run that dies leaves a chain that verifies
up to where it stopped.

| Line | When | Fields |
|---|---|---|
| `header` | first | `input_sha256`, `input_name`, `cameras`, `models`, `settings`, `image_digest`, `tool_version` |
| `speech` | one per stretch of speech, before any frame | `start_s`, `end_s`, `pad_ms` |
| `frame` | one per frame per camera | `camera`, `frame`, `pts_ns`, `counts`, `boxes`, optional `quads`, optional `zone_kept` |
| `camera_done` | after each camera | `camera`, `frames`, `seconds`, `record` (the file pipeline's audit record) |
| `footer` | last, `done` or `failed` | `output_sha256`, `output_name`, `status`, `error` |

`docs/audit_schema.json` describes every field. `python -m faceblur.h5
verify audit.jsonl` walks the chain and validates each line against the
schema; it prints `ok: N lines`, or the first problem and exits 1. For one
938-frame camera the file is 941 lines, about 1 MB, and verifies in under 2
seconds.

`counts` and `boxes` are the masks painted into the copy, of each kind,
including the boxes the tracker filled into gaps and tails, not only what a
detector saw on that frame, so two boxes of one track can overlap. `quads`
gives the rotated outline of text and plates. A box at least half inside the
handled zone (what the wearer holds or touches) is left clear in the copy, so
it is not in `counts` or `boxes`; the optional `zone_kept` lists those boxes
per kind, in the same `[x, y, w, h]` form. The `speech` lines say which
stretches of the removed audio held speech (Silero VAD, padded by `pad_ms`
each side), so whoever needs the speech knows where to look in the input.
`models` maps every model the run loaded to the sha256 of its file, under
the names `yunet`, `centerface`, `ultraface`, `yolox`, `ppocr`, `lpd_yunet`,
`zone_palm`, `zone_landmark`, `palm`, `hand_landmark` and `silero_vad`,
whichever the settings used. `image_digest` names the container image, from
`FACEBLUR_IMAGE_DIGEST` or the file `/etc/faceblur_image_digest`, and is
null outside a container.

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
| `ppocr` | PP-OCRv3 text detection from opencv_zoo | Apache 2.0 | 2.4 MB | finds lines of text, when text is switched on |
| `lpd_yunet` | LPD-YuNet from opencv_zoo | Apache 2.0 | 4.1 MB | finds licence plates, when plates are switched on |
| `silero_vad` | Silero VAD v6.2.3 | MIT | 2.3 MB | lists the speech in the audio the HDF5 module removes |

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
| h5py | 3.16.0 |
| jsonschema | 4.26.0 |
| imageio-ffmpeg | 0.6.0 (ffmpeg 7.1, with NVENC) |
| PySide6 | 6.11.2 |
| pytest | 9.0.2 |
| pyinstaller | 6.22.2 |

The evaluation harness needs `requirements-eval.txt` in its own environment,
because MediaPipe pins numpy below 2.

The storage tools need `requirements-storage.txt` in the same `.venv`: av 19.0.1, pyarrow 26.0.0.

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
