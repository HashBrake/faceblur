# STATE — handover notes

Last updated: 2026-09-05. This file says where the project stands, what was
verified, and what a new person or session should do next. The README explains
how to use the tool; `docs/report.md` holds the measurements.

## What this is

FaceBlur: video in, video with faces blurred out. Windows desktop window plus a
command line. No labels, no human step anywhere in the pipeline or in its
evaluation. Built for Ego camera footage (1600x1300, ~30 fps, h264, no audio)
of card collectors, where the goal is: destroy the minimum region that hides a
person's identity and leave hands, cards, screens and everything else untouched.

## Where things stand

- Phases 0-5 of `FACEBLUR_BUILD_PLAN.md` are done. Phase 6 (installer, code
  signing) was not asked for and is not done.
- The precision rebuild after the first build's "blurs hands and whole screens"
  problem is done and measured (`docs/precision_audit.md`, `docs/precision_report.md`).
- GPU path (DirectML on any DX12 card, CUDA on cloud), phased parallel runner,
  copied clean stretches, NVENC encoding, entry-frame coverage: done.
- Tests: `tests/`, 741 passing (`.venv\Scripts\python.exe -m pytest tests`).
- Packaged app: `dist\FaceBlur\` builds from `build\faceblur.spec` (about 413 MB).
- Desktop shortcut `FaceBlur.lnk` on this PC launches `.venv\Scripts\pythonw.exe -m ui.app`.

## Last thing fixed

A join bug: stretches copied from the source (no B-frames) and NVENC pieces
(3-frame reorder delay) were placed a few frames apart by ffmpeg's concat
demuxer, so the joined file came out 0.115 s short and failed its own check.
On top of that the automatic fallback (encode everything) succeeded but the
record kept the first attempt's error, so the run reported "failed" with a good
file on disk and no audit JSON. Both fixed:

- encoded pieces carry no B-frames (`-bf 0`, both NVENC and x264);
- clean stretches are only copied when the source's own reorder delay is zero
  (`segments.reorder_delay`), else the video is encoded whole;
- a successful retry clears the error and writes the sidecar.

Cost of `-bf 0`: larger output files at the same quality. Size figures are in
`docs/report.md`.

## Sample footage

`footage/` holds four Ego files (264 s total). They and every output folder are
gitignored. Never commit them. The evaluation caches under `eval/cache/` are
derived from them and are gitignored too.

## How to run

```
.venv\Scripts\python.exe -m ui.app                        # window
.venv\Scripts\python.exe cli.py footage -o footage_blurred --workers 10
.venv\Scripts\python.exe -m pytest tests                  # tests
```

## Known limits and open items

- Faces under about 32 px are covered about half the time; motion blurred
  faces about three quarters. Larger models or a higher detection size raise
  recall at a cost in speed. `eval/sweep.py` is the place to try them; it picks
  settings without a human.
- Detection is the cost: about 100 ms per 1600x1300 frame on an RTX 3070
  through DirectML; 1.4 s of processing per second of video over the four
  samples with a 10-worker pool, 2.7 s on the busiest file. A CUDA build on a data-centre GPU should cut this several fold; not measured.
- NVDEC decoding was measured no faster than CPU decoding here and stays off
  (`--hwaccel cuda` turns it on).
- The window and the CLI use the same pool and the same defaults; the CLI
  default is 10 workers.
- No installer, no code signing, no cloud runner. `faceblur.batch.run_video`
  takes a `submit` callable so a cloud runner can map jobs onto anything.
- `gh` is not installed on this PC. Pushing needs the empty GitHub repo to
  exist first; git credential manager handles the login in the browser.

## Files that matter

| Path | What |
|---|---|
| `faceblur/settings.py` | every setting and its default |
| `faceblur/detect.py` | YuNet and CenterFace through onnxruntime, crop confirmation |
| `faceblur/track.py` | tracker: confirmation, continuation, motion tails |
| `faceblur/redact.py` | ellipse mask and pixel destruction |
| `faceblur/segments.py` | keyframe cuts, copies, encodes, concat, verify, reorder delay |
| `faceblur/batch.py` | the phased runner (`run_video`) |
| `faceblur/pipeline.py` | audit record, single-process path |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/` | label-free evaluation and the parameter sweep |
| `docs/report.md` | quality, accuracy, speed, hardware, assumptions, done and not done |
