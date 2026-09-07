# FaceBlur

FaceBlur blurs faces in video. Give it a video or a folder of videos. It writes a
blurred copy of each one to an output folder you choose.

FaceBlur runs on your PC. Your video never leaves the machine.

## What it does and does not do

FaceBlur masks the smallest region that hides a person's identity: an oval over
the eyes, nose and mouth. It leaves hair, hands, bodies and objects alone. The
rest of the picture is untouched, so a blurred copy stays usable as training
data.

FaceBlur redacts faces. It does not redact number plates, bodies or text.

FaceBlur does not change audio. It copies the audio track from the source into
the blurred copy, untouched.

FaceBlur may miss a face. Read the next section before you rely on it.

## How well it works

Every number below comes from an automatic evaluation with no human labels. The
method is in `eval/` and the full tables are in `docs/precision_report.md`.
Measured on a 31 second Ego camera file, 938 frames, 1600x1300.

| Measure | First build | This build |
|---|---|---|
| Share of the frame destroyed, mean | 33% | 1.7% |
| Share of the frame destroyed, worst frame | 93% | 3.8% |
| Masked pixels where no detector sees a face, mean | 29.7% | 0.30% |
| Hand pixels touched | 82% | 0.00% |
| Faces the detectors agree on, covered | 100% | 99.0% |
| Confirmed faces kept covered while still visible | | 99.6% |

Recall against faces of known position, pasted into real frames of the same
video, with the same settings:

| Faces | Covered |
|---|---|
| 64 px and larger, sharp | 89% |
| 64 px and larger, motion blurred | 73% |
| 32 to 63 px | 94% |
| under 32 px | 46% |

Small and motion blurred faces carry the misses. The detectors themselves find a
24 px face about half the time, at any threshold. On a 1600 px frame a 24 px face
is a person far across the room.

The first build reached 100 percent on the agreed faces by masking a third of
every frame. This build gives up a few points of recall on the smallest faces to
leave everything else intact. `docs/precision_audit.md` explains why.

Check the blurred copies before you share them.

## How it decides what to blur

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
faceblur INPUT [-o OUTPUT] [--engine yunet|centerface|both] [--conf F]
         [--det-sizes 1280,1920] [--stride N] [--no-verify] [--max-face F]
         [--min-track N] [--max-gap N] [--tail N] [--pad F]
         [--mode blur|pixelate|solid] [--workers N] [--device auto|gpu|cpu]
         [--encoder auto|nvenc|x264] [--chunk-seconds S] [--no-copy]
         [--hwaccel none|cuda] [--tta 0|1|2] [--no-third] [--no-camera]
         [--report PATH] [--recursive] [--no-progress]
```

```
.venv\Scripts\python.exe cli.py C:\ego -o C:\ego_blurred --workers 8
```

INPUT is a video or a folder. A folder run reads every video in it. OUTPUT
defaults to a folder named `<input>_blurred` next to the input. The exit code is
1 when any video failed.

`--no-verify` and `--engine both` find more faces and also blur hands and
objects. On the Ego footage they touched 7.7 percent of hand pixels. Use them
only when a missed face costs more than a damaged frame.

## What it writes

- `<name>_blurred.mp4`, the blurred copy, H.264 at crf 12 with the source audio.
- `<name>_blurred.mp4.json`, the audit record: frames, tracks, how much of each
  frame was masked (mean, p95, max), frames over the mask budget, every setting,
  the sha256 of each model, and the time taken.

## Evaluate a new batch

The harness needs a second environment for MediaPipe, which supplies the third
face detector and the hand regions:

```
py -3.12 -m venv .venv-eval
.venv-eval\Scripts\python.exe -m pip install -r requirements-eval.txt
```

Then, for a video:

```
.venv-eval\Scripts\python.exe eval\oracle_mediapipe.py VIDEO --stride 5
.venv\Scripts\python.exe -m eval.consensus VIDEO --stride 10
.venv\Scripts\python.exe -m eval.sweep VIDEO
.venv\Scripts\python.exe -m eval.misses VIDEO --images SOME_FOLDER
```

`eval.misses` lists the stretches where YuNet saw a box at full threshold that
nothing confirmed and no mask covers, with one annotated frame each. The same
list is in every audit record as `unconfirmed_runs`. Most are hands and
objects; a change that finds more faces makes the count fall while the gates
hold.

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

## Detectors

| Name | Model | Licence | Size |
|---|---|---|---|
| `yunet`, finds faces | YuNet from opencv_zoo | Apache 2.0 | 232 KB |
| `centerface`, confirms them | CenterFace from the deface package | MIT | 7.3 MB |
| `ultraface`, breaks ties | UltraFace RFB-320 from Linzaer | MIT | 1.3 MB |

`models/README.md` records where each file came from and its sha256.

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
