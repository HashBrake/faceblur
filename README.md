# FaceBlur

FaceBlur blurs faces in video. Give it a video or a folder of videos. It writes a
blurred copy of each one to an output folder you choose.

FaceBlur runs on your PC. Your video never leaves the machine.

## What it does and does not do

FaceBlur redacts faces. It does not redact number plates, bodies or text.

FaceBlur does not change audio. It copies the audio track from the source into
the blurred copy, untouched. A reviewer who cares about voices needs a different
tool.

FaceBlur may miss a face. Read the next section before you rely on it.

## How many faces it finds

Phase 3 measured this on three real Ego camera files. A person labelled 207 faces
by hand in 42 frames. A face counts as covered when the mask covers at least 80
percent of its rectangle.

| Setting | Faces covered | Share of the frame destroyed |
|---|---|---|
| Defaults | 75.8% | 33% |
| Defaults with `--persist 14` | 93.2% | 55% |
| Defaults with `--persist 14 --pad 0.6` | 98.1% | 70% |
| `--engine both --conf 0.15 --pad 0.45 --persist 10` | 99.5% | 79% |

Read those two columns together. Every setting that finds more faces also
destroys more of the picture. On this footage the settings that reach 99 percent
blur about four fifths of every frame.

The hand placed rectangles carry tens of pixels of error at small face sizes, so
the first column understates what the pipeline covers. `docs/recall_report.md`
gives the full method, an upper bound for each row, and the numbers by face size.

Check the blurred copies before you share them.

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

Advanced settings hold the detector, the frame step, the redaction style and the
number of videos to process at once.

## Use the command line

```
faceblur INPUT [-o OUTPUT] [--engine yunet|centerface|both] [--conf F]
         [--det-sizes 640,1280] [--stride N] [--persist N] [--pad F]
         [--mode blur|pixelate|solid] [--workers N] [--report PATH]
         [--recursive] [--no-progress]
```

Run it through the virtual environment:

```
.venv\Scripts\python.exe cli.py C:\ego -o C:\ego_blurred --workers 8
```

INPUT is a video or a folder. A folder run reads every video in it. Add
`--recursive` to walk subfolders and mirror them in the output folder. OUTPUT
defaults to a folder named `<input>_blurred` next to the input. The exit code is
1 when any video failed.

To find more faces, raise `--persist` first. It costs no extra time.

```
.venv\Scripts\python.exe cli.py C:\ego -o C:\ego_blurred --persist 14
```

## What it writes

For each video, FaceBlur writes two files to the output folder:

- `<name>_blurred.mp4`, the blurred copy, H.264 video with the source audio.
- `<name>_blurred.mp4.json`, the audit record.

The audit record holds the source name, the frame count, the resolution, the
frame rate, how many frames held a detection, how many frames the pipeline
masked, every setting used, the sha256 of each model, and the time taken. Give
this file to a compliance reviewer.

## How it works

FaceBlur reads each video twice.

The first pass detects faces. It scales each frame so its long side matches each
detection size, runs every enabled detector at each size, maps the boxes back,
and merges them. It then copies each box forward and backward in time, growing it
as it goes, so a detector that fires on three frames out of four still yields a
mask on all four.

The second pass destroys the pixels and encodes. It shrinks the whole frame, then
grows it back, then paints that copy inside the padded rectangles. Shrinking
throws information away, so the face cannot come back from the output. A plain
blur is a linear filter and can be partly undone, so FaceBlur never ships a mode
that only blurs.

Masks are rectangles, not ellipses. An ellipse drawn inside the box leaves the
corners, and hairlines and ears live in the corners.

## Speed

Measured on one core, on a 31 second file of 1600x1300 video.

| Stage | Seconds |
|---|---|
| Detect | 73 |
| Encode and mux | 187 |
| Total | 261 |

That is about 8.5 hours of one core for one hour of footage. Encoding costs more
than detection. Raise `--workers` to process several videos at once.

## Detectors

FaceBlur ships two, and runs them through `--engine`.

| Name | Model | Licence | Size |
|---|---|---|---|
| `yunet`, the default | YuNet from opencv_zoo | Apache 2.0 | 232 KB |
| `centerface` | CenterFace from the deface package | MIT | 7.3 MB |

`--engine both` runs the two and merges the result. It costs about twice the
detection time. `models/README.md` records where each file came from and its
sha256. `tests/test_models.py` checks both hashes.

## Build the packaged app

The packaged app needs no Python on the target machine.

```
.venv\Scripts\python.exe -m PyInstaller build\faceblur.spec --noconfirm
```

The result is `dist\FaceBlur\`. Zip that folder and give it to the user. They
unpack it and run `FaceBlur.exe`.

| Measure | Size |
|---|---|
| Unpacked folder | 381 MB |
| Zip | 151 MB |

Most of that is three things: OpenCV at 99 MB, Qt at 92 MB and ffmpeg at 84 MB.
The two models take 15 MB.

## Run the tests

```
.venv\Scripts\python.exe -m pytest tests\
```

`tests/manual_ui.md` holds the checks that need a person and a screen.

## Versions

These exact versions passed the tests. `requirements.txt` pins every one.

| Package | Version |
|---|---|
| Python | 3.12.8 |
| opencv-python-headless | 4.13.0.92 |
| numpy | 2.5.2 |
| onnxruntime | 1.29.0 |
| imageio-ffmpeg | 0.6.0 (ffmpeg 7.1) |
| PySide6 | 6.11.2 |
| pytest | 9.0.2 |
| pyinstaller | 6.22.2 |

## Known limits

No detector finds every face. Faces at a sharp angle, faces smaller than about 10
pixels, faces behind glass and faces in mirrors get missed. Propagation lowers the
miss rate. It cannot cover a face the detector never fires on.

FaceBlur also blurs things that are not faces. On the Ego footage it masked a row
of sinks. The plan accepts that trade, because a missed face is a data leak and an
extra blur is not.

Audio passes through untouched.
