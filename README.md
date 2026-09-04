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
| Share of the frame destroyed, mean | 33% | 1.3% |
| Share of the frame destroyed, worst frame | 93% | 3.5% |
| Masked pixels where no detector sees a face, mean | 29.7% | 0.16% |
| Hand pixels touched | 82% | 0.00% |
| Faces the detectors agree on, covered | 100% | 98.4% |
| Confirmed faces kept covered while still visible | | 99.3% |

Recall against faces of known position, pasted into real frames of the same
video, with the same settings:

| Faces | Covered |
|---|---|
| 64 px and larger, sharp | 96 to 100% |
| 64 px and larger, motion blurred | 50 to 100% |
| 32 to 63 px | 68% |
| under 32 px | 56% |

Small and motion blurred faces carry the misses. The detectors themselves find a
24 px face about half the time, at any threshold. On a 1600 px frame a 24 px face
is a person far across the room.

The first build reached 100 percent on the agreed faces by masking a third of
every frame. This build gives up a few points of recall on the smallest faces to
leave everything else intact. `docs/precision_audit.md` explains why.

Check the blurred copies before you share them.

## How it decides what to blur

Pixels are destroyed only when several independent checks agree:

1. YuNet finds a candidate at a threshold of 0.5, scanning at 1280 and 1920 px.
2. CenterFace, a different architecture, must fire on the same spot.
3. The box must be a plausible face: at most 15 percent of the frame's long
   side, and roughly square.
4. A tracker links detections across frames. A face must be seen at least three
   times, close together, before any pixel is touched. Once a track is confirmed, YuNet
   alone may keep it going at a lower threshold, if its box overlaps where the
   track predicts the face to be. A hand can never start a track. Gaps of up to
   five frames are interpolated. The mask reaches two frames past each end, at
   the same size.
5. The mask is an ellipse fitted to the box and rotated to the eye line, with a
   soft edge. Inside it the pixels are replaced from a copy shrunk to six blocks
   across, so the face cannot come back.

The thresholds were chosen by `eval/sweep.py`: the highest recall that keeps
masking outside anything a detector calls a face under 0.3 percent of the frame,
and hands untouched.

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
         [--mode blur|pixelate|solid] [--workers N] [--report PATH]
         [--recursive] [--no-progress]
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
```

The sweep writes `docs/precision_report.md` and prints the settings that pass
the gates. No step asks a person for anything.

## Speed

On one worker, both detectors at 1280 and 1920 px cost about 0.7 s per
1600x1300 frame, and near lossless encoding adds about 0.2 s. Raise `--workers`
to process several videos at once. Each worker caps its threads so a pool does
not oversubscribe the CPU.

## Build the packaged app

```
.venv\Scripts\python.exe -m PyInstaller build\faceblur.spec --noconfirm
```

The result is `dist\FaceBlur\`, about 380 MB unpacked, 150 MB zipped. The
evaluation harness is not part of it.

## Detectors

| Name | Model | Licence | Size |
|---|---|---|---|
| `yunet`, finds faces | YuNet from opencv_zoo | Apache 2.0 | 232 KB |
| `centerface`, confirms them | CenterFace from the deface package | MIT | 7.3 MB |

`models/README.md` records where each file came from and its sha256.

## Run the tests

```
.venv\Scripts\python.exe -m pytest tests\
```

## History

`FACEBLUR_BUILD_PLAN.md` is the original plan. `docs/recall_report.md` is the
Phase 3 measurement of the first build, with hand labels. `docs/precision_audit.md`
is the audit that found why that build destroyed hands and counters, and
`docs/precision_report.md` is the automatic measurement of this one.
