# FaceBlur build plan

Handoff document for Claude Code. Read the whole file before writing code. Every phase ends with a verify step. Do not start the next phase until the verify step passes.

## TL;DR

Build a Windows desktop tool that takes a video file or a folder of videos, blurs every face, and writes the blurred copies to an output folder the user picks. Faces only. Recall matters more than speed: one missed face is a data leak, one extra blur is cosmetic. Stack: Python, OpenCV YuNet for detection (Apache 2.0, 232 KB model, already validated), CenterFace from `deface` as a second detector for an ensemble mode, ffmpeg through `imageio-ffmpeg` for encoding and audio passthrough, PySide6 for the UI, PyInstaller for packaging. A working prototype of the core pipeline ships with this plan (`blurfaces_prototype.py`). Start from it.

## 1. Decisions already made

These came from the product owner. Do not reopen them.

| Question | Decision |
|---|---|
| Where it runs | On the user's Windows PC. Video never leaves the machine. |
| What to redact | Faces only. No plates, no bodies, no text. |
| Recall vs speed | Never miss a face. Low threshold, padded boxes, temporal propagation. Accept slower processing and some over-blurring. |
| Footage type | Egocentric wearable camera ("Ego" camera) footage. Close faces, motion blur, mixed indoor and outdoor, long recordings, batches of many files. |
| Delivery order | Core library first, then CLI, then UI. All in one repo. |
| Workflow | Drag a file or folder onto the app, or pick it with a button. Pick an output folder. Press start. Blurred copies appear in the output folder with the same file names plus a suffix. |
| Tool choices | Prefer open source tools with a track record. Avoid AGPL code (this runs inside a company). |

## 2. What is already validated

A prototype was built and run before this plan was written. Its results set the defaults below. Treat the numbers as indicative: they came from one cloud CPU and one crowd photo, not from Ego footage.

The prototype (`blurfaces_prototype.py`) does: two passes over each video (detect, then redact and encode), multi-scale YuNet detection, NMS merge, temporal propagation of boxes forward and backward with box growth per frame, padded rectangular masks, three redaction modes (downsample+blur, pixelate, solid), ffmpeg encode with audio copied from the source, and a JSON audit record per file. It ran end to end on a synthetic 120-frame test video and redacted 120 of 120 frames.

Detector quality check on a 548x342 crowd photo (`messi5.jpg` from the OpenCV samples), threshold 0.25, YuNet, CPU:

| Detection long side (px) | Faces found | Time per frame |
|---|---|---|
| 320 | 32 | 8 ms |
| 480 | 18 | 9 ms |
| 640 | 11 | 16 ms |
| 960 | 7 | 34 ms |
| 1280 | 11 | 63 ms |
| 1920 | 9 | 202 ms |
| 2560 | 21 | 373 ms |

Two things to take from this table. First, cost grows with the square of the detection size, so the pipeline must detect at fixed absolute sizes, not at the source resolution. A 4K source and a 540p source should cost the same to scan. Second, small working sizes find more tiny faces (with more false positives) and large working sizes find the large faces. No single size wins, so the default is two sizes. The counts include false positives. There is no ground truth for this photo. The real recall measurement happens in Phase 5 on Ego footage.

Speed of the prototype at 960x540 with scales 1.0 and 1.75: about 5 frames per second on one CPU core. At native scale only: about 33 frames per second. This is the central cost tradeoff and the reason the plan includes batch parallelism and an optional GPU path.

Uncertainty flags:

- CenterFace (the `deface` detector) was not benchmarked here. Its inclusion rests on the `deface` project's track record, not on measurements from this session.
- YuNet recall on profile faces and on motion-blurred faces was not measured. The synthetic test applied Gaussian blur to some frames and the detector still fired, but that is weak evidence.
- All timing numbers are from a Linux container. The user's Windows machine will differ.

## 3. Architecture

Three layers in one repo:

```
faceblur/
  faceblur/            core library (no UI code, no print statements)
    detect.py          detector backends: YuNet, CenterFace, ensemble
    track.py           NMS, temporal propagation
    redact.py          mask building and pixel destruction
    video.py           decode, encode, audio mux via ffmpeg
    pipeline.py        process_video(src, dst, settings, on_progress) -> AuditRecord
    settings.py        dataclass with defaults and validation
  models/
    yunet.onnx         Apache 2.0, from opencv_zoo (sha256 below)
    centerface.onnx    MIT, copied from the deface package
  cli.py               argparse front end over pipeline.py
  ui/
    app.py             PySide6 window
    strings.py         every user-visible string in one file
  tests/
  build/
    faceblur.spec      PyInstaller
  README.md
```

The core library must be callable from the CLI and the UI with the same `process_video` function. Progress goes through a callback, not through stdout. Cancellation goes through a threading.Event that the pipeline checks between frames.

## 4. Tool choices and why

### Face detector

Primary: YuNet through `cv2.FaceDetectorYN` (OpenCV 4.13 confirmed to have it). Reasons: Apache 2.0 model and code, 232 KB, runs on CPU at usable speed, validated in this session, no extra dependency beyond opencv-python-headless.

Model file: `face_detection_yunet_2023mar.onnx` from the opencv_zoo repository. The GitHub raw URL returns a Git LFS pointer, not the model. Fetch from the LFS media endpoint:

```
https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
sha256: 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
size: 232589 bytes
```

Commit the file to the repo. Verify the sha256 in a test. A copy ships with this plan.

Secondary: CenterFace, taken from the `deface` package (MIT, pip installable, `centerface.bnmerged.onnx` bundled in the wheel). `deface` is the most cited open source tool built for exactly this job. Its last release is v1.5.0 from October 2023, so treat it as a source of a proven model and reference code, not as a maintained dependency. Vendor the ONNX file and write a thin wrapper that runs it through `onnxruntime`. Do not depend on the `deface` package at runtime.

Ensemble mode (`engine = both`): run both detectors, union the boxes, NMS merge. Two detectors with different training and architecture miss different faces. This costs about double the detection time and is the setting to use when recall matters most.

Rejected: Ultralytics YOLO face models (AGPL 3.0, a problem inside a company), InsightFace SCRFD (model zoo is licensed for non-commercial research use), MediaPipe BlazeFace (weak on profile and small faces; acceptable as a third backend later, not now).

### Video decode and encode

Decode with OpenCV `VideoCapture`. Encode by piping raw BGR frames into ffmpeg (`libx264`, `crf 20`, `yuv420p`, `+faststart`) with the source file as a second input and `-map 1:a? -c:a copy` so audio passes through untouched when present. ffmpeg comes from `imageio-ffmpeg`, which bundles a static binary in the wheel. The user does not install ffmpeg separately.

Known risk: rotation metadata. Phone and wearable files often carry a rotation tag. OpenCV applies it on decode in recent versions. ffmpeg also applies it on decode for the audio input, which does not matter, but if the pipeline ever switches to ffmpeg decode, both paths must agree. Add a test with a rotated sample file in Phase 5.

Known risk: variable frame rate sources. Piping frames at the nominal fps from `CAP_PROP_FPS` drifts audio sync on VFR files. If Ego camera files are VFR, switch to `-vsync passthrough` with explicit timestamps or re-mux with `-fps_mode`. Check one real Ego file with `ffprobe` in Phase 0.

### UI framework

Choice: PySide6 (Qt for Python, LGPL 3.0). Reasons: native Windows file and folder dialogs, drag-and-drop of both files and folders delivers full paths reliably through `QMimeData.urls()`, single language for the whole app, mature PyInstaller support, threading model (`QThread` plus signals) that fits a long-running job with progress and cancel.

Alternative considered: `pywebview` (HTML/CSS front end in a native window). It gives more visual control and exposes dropped file paths through `pywebviewFullPath`. Whether a dropped folder yields a path on the Windows EdgeChromium backend was not confirmed in this session. Since folder drop is core to the workflow, that uncertainty decides against it. If the product owner later wants a web-style UI, revisit with a spike that tests folder drop first.

Rejected: Electron and Tauri (a second language plus a Python sidecar to package), a local web server plus browser tab (browsers cannot return a folder path for output selection).

### GPU (optional, Phase 6)

`onnxruntime-directml` runs ONNX models on any DirectX 12 GPU on Windows without CUDA. `deface` already supports it through `--execution-provider`. YuNet through OpenCV DNN stays on CPU. If detection speed on the user's machine is too slow after batch parallelism, move the YuNet call from `cv2.FaceDetectorYN` to `onnxruntime` with the DirectML provider and reimplement YuNet's decode step (priors and NMS) in numpy. That is real work, so it is the last phase.

### Packaging

PyInstaller one-folder build. Bundle both ONNX models and the `imageio-ffmpeg` binary as data files. Target: a zip the user unpacks and runs by double-clicking `FaceBlur.exe`. No Python install on the target machine.

## 5. Detection pipeline specification

Implement exactly this. The prototype already does most of it.

Pass 1, detect. For every frame (or every `stride` frames), resize the frame so its long side equals each value in `det_sizes` (default `640, 1280`), run every enabled detector at each size, map boxes back to source coordinates, merge with NMS at IoU 0.35. Store per-frame box lists in memory. Boxes are small, so a two-hour video fits.

Propagate. For each detection at frame t, copy the box to frames t minus k through t plus k for k up to `persist` (default 6 frames, and never less than `stride`). Grow the copied box by `grow` per frame of distance (default 6 percent) so it still covers a face that moves. Re-run NMS at IoU 0.6 per frame to collapse duplicates. This step is what turns per-frame detection into "never miss a face": a detector that fires on 3 frames out of 4 still yields a mask on all 4.

Pass 2, redact and encode. For every frame, build a mask from the union of padded rectangles (`pad` default 30 percent of face width and height on each side). Rectangles, not ellipses: an ellipse inscribed in the box leaves the corners, and hairlines and ears live in the corners. Fill the mask from a cover image and pipe the frame to ffmpeg.

Cover image, by `mode`:

- `blur` (default): downsample the whole frame by factor `strength` (default 28, relative to the short side), upsample back, then Gaussian blur. The downsample discards information. A plain Gaussian blur is a linear filter and can be partly inverted, so do not ship a mode that only blurs.
- `pixelate`: same downsample, nearest-neighbor upsample.
- `solid`: black.

Audit record. For each output file write a JSON sidecar: source name, output name, frame count, resolution, fps, frames with at least one raw detection, total detections, frames covered after propagation, every setting used, detector engine and model sha256, wall time. This is the evidence a compliance reviewer asks for.

Defaults table:

| Setting | Default | Notes |
|---|---|---|
| engine | `yunet` | `centerface`, `both` |
| conf | 0.25 | YuNet score threshold. Lower catches more, blurs more non-faces. |
| det_sizes | 640, 1280 | Long side in pixels. Add 1920 for very small distant faces at 3x cost. |
| stride | 1 | Detect every frame. 2 or 3 is the speed knob when the owner accepts it. |
| persist | 6 | Frames carried forward and backward. |
| grow | 0.06 | Box growth per propagated frame. |
| pad | 0.30 | Mask padding. |
| mode | `blur` | |
| strength | 28 | |
| crf | 20 | |
| output suffix | `_blurred` | |

## 6. CLI specification

```
faceblur INPUT [-o OUTPUT] [--engine yunet|centerface|both] [--conf F] [--det-sizes 640,1280]
         [--stride N] [--persist N] [--pad F] [--mode blur|pixelate|solid]
         [--workers N] [--report PATH] [--no-progress]
```

INPUT is a file or a folder. Folder input processes every file with a video extension, non-recursive by default, `--recursive` to walk. OUTPUT defaults to a sibling folder named `<input>_blurred`. `--workers` runs that many videos in parallel processes (default: half the CPU count, minimum 1). Exit code 1 if any file failed. The prototype's argparse block is the starting point.

## 7. UI specification

One window. No menus beyond the standard window controls and an About entry. No settings screen in the first version; advanced settings live behind a disclosure control on the main window.

### Screen layout, top to bottom

1. Drop zone. Large dashed region. Accepts one or more files, or one folder. Label inside: "Drop a video or a folder here". Below it a button: "Choose files" and a second button: "Choose a folder". After a drop or a pick, the zone shows the count and total size: "12 videos, 8.4 GB".
2. Output folder row. Read-only path field, button "Choose output folder". Default: a folder named `<source folder>_blurred` next to the source. If the user does not change it, the app creates it.
3. Advanced settings, collapsed by default, labeled "Advanced settings". Contains: engine (radio: Standard, Maximum coverage), detect every N frames (spinbox, default 1), redaction style (radio: Blur, Pixelate, Black box), workers (spinbox). Each control has one line of help text under it in the wording rules below.
4. Start button, primary style, right-aligned. Disabled until both an input and an output folder exist. Label: "Blur faces". While running it becomes "Stop".
5. Progress area. One row per file: file name, a progress bar, a status word (Waiting, Detecting, Writing, Done, Failed, Stopped). Above the rows, an overall bar and an estimate: "3 of 12 done. About 40 minutes left." Time estimates come from measured throughput on the files already done, not from a guess before the first file finishes. Show "Estimating..." until then.
6. On completion: a summary line, "12 videos done. 0 failed." and a button "Open output folder".

### Behavior

- Processing runs in a worker process pool, not in the UI thread. The window stays responsive.
- Stop finishes the current frame, kills the workers, deletes partial output files, and marks remaining rows Stopped. It never leaves a half-written file in the output folder.
- If an output file already exists, skip it and mark the row "Already done" unless the user ticked "Replace existing files" in advanced settings.
- Failures show the ffmpeg or decode error in a tooltip on the Failed status and in the audit log. The batch continues with the next file.
- Keyboard: Ctrl+O choose files, Ctrl+Shift+O choose folder, Ctrl+Enter start, Escape stop (with confirmation while running).
- Light and dark appearance both supported. Use Qt's palette roles, not hard-coded colors, except for the primary button.
- The window remembers the last output folder and the advanced settings between runs (QSettings).

### Design requirements (from the apple-design skill)

Run the `apple-design` skill on the finished UI as a review pass before calling Phase 4 done. Until then, build to these rules, which are the ones that skill will check:

- Every clickable control is at least 24 px tall on desktop; primary buttons 32 px or more.
- Body text contrast at least 4.5:1 against its background in both light and dark palettes. Help text is not allowed to drop below that to look subtle.
- No information carried by color alone. Status words accompany every colored indicator. Failed rows get an icon and the word, not just red.
- Sentence case for every label, button, and heading. Never Title Case, never ALL CAPS.
- One primary action per screen. Only "Blur faces" gets the accent color.
- Loading, empty, error, and success states all designed, not just the happy path. The empty state is the drop zone with its instruction. The error state shows what failed and what to do.
- Destructive or irreversible actions confirm first. Stop while running confirms. Replace existing files is an explicit opt-in.
- The window resizes. The drop zone and the progress list grow; the buttons do not.
- Every control reachable by keyboard with a visible focus ring. Every icon-only control has an accessible name.
- Spacing on an 8 px grid. Section spacing 24 px, control spacing 8 px, window margin 16 px.

### Wording rules (from the humanizer and asd-ste100 skills)

All user-facing strings live in `ui/strings.py`. Before Phase 4 verify, run the `asd-ste100` skill in Strict mode over that file and the `humanizer` skill over the README. Until then, write to these rules:

- Active voice. "The app writes files to this folder", not "Files are written to this folder".
- One instruction per sentence. At most 20 words per sentence.
- One word for one thing across the whole app. The input is always "video" or "folder". The result is always "blurred copy". The destination is always "output folder". Do not rotate between "file", "clip", "footage".
- No phrasal verbs where a single verb exists: "start", not "kick off"; "choose", not "pick out".
- No marketing words: no "seamless", "powerful", "smart", "AI-powered".
- No em dashes or en dashes anywhere in the UI or docs.
- No emoji in UI text.
- Error messages say what happened and what to do next, in that order. "Could not read this video. Check that the file is not open in another program."
- Do not upgrade a hedge to a fact. If detection may miss a face, the help text says "may miss".

Reference strings, already checked against the rules:

| Key | Text |
|---|---|
| drop_zone_empty | Drop a video or a folder here |
| choose_files | Choose files |
| choose_folder | Choose a folder |
| output_label | Output folder |
| choose_output | Choose output folder |
| start | Blur faces |
| stop | Stop |
| engine_standard | Standard |
| engine_standard_help | One detector. Faster. May miss small or turned faces. |
| engine_max | Maximum coverage |
| engine_max_help | Two detectors. About twice as slow. Finds more faces. |
| stride_help | Detect faces on every Nth frame. 1 is safest. Higher is faster. |
| replace_existing | Replace existing files |
| status_waiting | Waiting |
| status_detecting | Detecting |
| status_writing | Writing |
| status_done | Done |
| status_failed | Failed |
| status_stopped | Stopped |
| status_skipped | Already done |
| stop_confirm | Stop now? The app deletes unfinished files. |
| summary | {done} videos done. {failed} failed. |
| open_output | Open output folder |
| err_unreadable | Could not read this video. Check that the file is not open in another program. |
| err_no_space | Not enough disk space in the output folder. |
| err_ffmpeg | Could not write the blurred copy. See the log file for details. |

## 8. Phases with verify steps

Each phase is a pull request. Each verify step is a command or a test that passes, not a judgment call.

### Phase 0: environment and real-footage check

Set up the repo, `pyproject.toml`, virtual environment, pinned dependencies: `opencv-python-headless`, `numpy`, `imageio-ffmpeg`, `onnxruntime`, `PySide6`, `pytest`. Commit both ONNX models. Ask the product owner for three real Ego camera files (short, different lighting) and run `ffprobe` on them: codec, resolution, fps, VFR or CFR, rotation tag, audio codec.

Verify: `pytest tests/test_models.py` passes (sha256 of both models). `ffprobe` output for the three files is recorded in `docs/sample_footage.md`.

### Phase 1: core library from the prototype

Split `blurfaces_prototype.py` into the `faceblur/` modules in Section 3. Add the CenterFace backend and the ensemble. Add the progress callback and the cancel event. Add the sidecar JSON. Keep the detection math identical to the prototype unless a test shows a reason to change it.

Verify: `pytest tests/` passes with these tests. Detector finds at least 10 faces in `messi5.jpg` at conf 0.25 with det_sizes 640,1280. Propagation test: a synthetic sequence with a detection on frames 10 and 14 yields masks on frames 4 through 20. Redaction test: after `blur` mode, the mean absolute pixel difference inside the mask is above 20 and outside the mask is exactly 0. Audio test: a source with an audio track produces an output whose `ffprobe` shows one audio stream with the same codec. Cancel test: setting the event mid-video stops within 2 seconds and leaves no output file.

### Phase 2: CLI

Build `cli.py` over the library. Add `--workers` with `multiprocessing`.

Verify: run on a folder of 4 short videos with `--workers 2`. Four outputs exist, four sidecars exist, exit code 0. Corrupt one input, run again: three outputs, one Failed line, exit code 1.

### Phase 3: recall and speed baseline on Ego footage

Use the three real files from Phase 0. Extract every 30th frame from each source, and label every face by hand in those frames (a rectangle per face, in a CSV; about 100 to 300 rectangles total). Run the pipeline with defaults, then with `engine both`, then with `det_sizes 640,1280,1920`. For each run, count labeled faces whose rectangle is at least 80 percent covered by the output mask. Report recall per setting and seconds per source minute.

Verify: `docs/recall_report.md` exists with the table. The product owner picks the default setting from it. If default recall is under 99 percent on the labeled set, do not proceed to the UI phase; tune first (lower conf, add det size, raise persist, switch engine).

### Phase 4: UI

Build `ui/app.py` to Section 7. Wire progress and cancel to the library. Then run the `apple-design` skill review on screenshots of every state (empty, files chosen, running, stopped, done with failures, dark mode). Fix every Critical and High item. Then run `asd-ste100` Strict over `ui/strings.py` and `humanizer` over `README.md`.

Verify: a manual script in `tests/manual_ui.md` with these checks, each ticked. Drop a folder: count appears. Drop three files: count appears. Choose output folder: path appears, Start enables. Start: rows progress, window stays responsive when dragged. Stop: confirm dialog, partial files gone. Re-run: existing outputs show Already done. Dark mode: contrast holds. Tab through every control: focus ring visible on each. Design review report has zero open Critical or High items.

### Phase 5: packaging

PyInstaller spec, one-folder build, models and ffmpeg bundled. Build on a clean Windows VM or a machine without Python on PATH.

Verify: on a machine with no Python installed, unzip, double-click, process a folder of three videos, outputs play in Windows Media Player with audio. Build size recorded in README.

### Phase 6: speed (only if Phase 3 numbers are too slow for the owner)

Options in order of effort: raise `--workers` to CPU count; `stride 2` with `persist 8`; DirectML through `onnxruntime-directml` for CenterFace (already supported by that model); DirectML for YuNet (requires reimplementing YuNet post-processing in numpy). Measure after each step on the Phase 3 files. Re-run the Phase 3 recall check after every change that touches detection.

Verify: `docs/recall_report.md` updated with the new row. Recall did not drop below the Phase 3 accepted value.

## 9. Test data

Do not commit real Ego footage to the repo. Keep it in a local folder listed in `.gitignore`. Commit only: `messi5.jpg` and `lena.jpg` from the OpenCV samples (public), a generated synthetic video from the prototype's test script, and the hand-labeled CSV from Phase 3 (rectangles only, no frames).

## 10. Risks and open questions

Recall ceiling. No detector reaches 100 percent. Faces at extreme angles, faces smaller than about 10 px, faces behind glass or in mirrors, and faces in reflections will be missed sometimes. The propagation step lowers the miss rate but cannot cover a face that the detector never fires on across its whole appearance. The Phase 3 measurement tells the owner the real number. Say the number in the README rather than claiming full coverage.

False positives on Ego footage. Hands, posters, and screens showing faces trigger the detector. With the "never miss" setting this is accepted, but a poster face blurred on every frame for an hour may surprise the owner. Show an example in the Phase 3 report.

Speed on long recordings. At 5 frames per second on one core, one hour of 30 fps footage takes 6 hours on one core. With 8 workers on 8 files that is still 6 hours of wall time for 8 hours of footage. State the throughput in the UI estimate so the user plans around it. Phase 6 exists for this reason.

Model provenance. Both models are downloaded once and committed with their sha256. The test in Phase 0 catches a swapped file. If the owner's security policy requires it, note the source commit of each model in `models/README.md`.

Voice PII. Audio passes through untouched. The scope decision is faces only, so this is correct, but a compliance reviewer will ask. Put one line in the README saying audio is not modified.

Open: are Ego camera files CFR or VFR (Phase 0 answers this). Open: does the owner want the output folder to mirror subfolders when input is recursive (default: flat, with a `--recursive` flag that mirrors structure).

## 11. Instructions for Claude Code

Read `blurfaces_prototype.py` first. It works. Refactor it, do not rewrite it from memory.

Invoke the `apple-design` skill for the Phase 4 review. Invoke `asd-ste100` (Strict mode) over `ui/strings.py` and `humanizer` over `README.md` before Phase 4 verify. If those skills are not installed in your environment, apply the rule lists in Section 7 by hand and say so in the PR.

Do not add features beyond this plan. In particular: no license plate detection, no body blur, no cloud upload, no settings screen, no telemetry. If a phase seems to need something not listed, stop and ask.

Pin every dependency. Record the exact versions that passed Phase 5 in the README.

Write the README in the same wording rules as the UI: sentence case headings, no em dashes, active voice, one word per thing.
