# STATE — handover notes

Last updated: 2026-09-08. This file says where the project stands, what was
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
- The missed-faces pass of 2026-09-07 (section 10 of `docs/report.md`): done
  and measured. Faces at the frame edge, faces turned down, faces during a
  fast pan. Hands still 0.00 percent.
- The third pass (2026-09-08, section 11 of `docs/report.md`): faces lost
  while the wearer hits a ball (track stitching across a smear, gap filling
  on the smear's own boxes, blur-sized masks) and the wearer's hand kept out
  by track-level rules (established and sure tracks only get continuation,
  tails and backward reach). Hands 0.00 / 0.00 / 0.09 percent on the three
  files with hand oracles.
- The fourth pass (2026-09-08, section 12 of `docs/report.md`): identity
  measured directly with a face recogniser instead of by proxy
  (`eval/reid.py`, `models/sface.onnx`, evaluation only). It found that the
  evaluation caches were stale, that the shipped mask strength is right and
  the next setting up would not be, and that scanning at more pixels is not
  worth its cost. Caches and pseudo-labels now carry a fingerprint of what
  they were built from and rebuild themselves when it moves.
- The gate of 2026-09-08 (section 13 of `docs/report.md`): `faceblur/verify.py`
  detects on the finished copy and reports faces sitting on pixels the run
  never changed. `--check-output` records them in the audit record;
  `--quarantine` moves such a copy into a `quarantine` folder instead of
  shipping it. It is the only check in the project that can see a miss. On
  the four sample files it finds 127 boxes over 120 frames of 7926: real
  missed faces on the washroom files, the wearer's hand on the bat on the
  table tennis one. With the gate on, all four would be held back.
- Tests: `tests/`, 839 passing (`.venv\Scripts\python.exe -m pytest tests`).
- Packaged app: `dist\FaceBlur\` builds from `build\faceblur.spec`. Not
  rebuilt after 2026-09-07; the spec includes the third model.
- Desktop shortcut `FaceBlur.lnk` on this PC launches `.venv\Scripts\pythonw.exe -m ui.app`,
  so it always runs the current code.

## What changed on 2026-09-07 (missed faces)

The user reported faces still showing, most in `004100`, when the camera moves
fast and when people enter the frame. A read-only diagnostic (`eval/misses.py`
is its permanent form) found four groups of misses and one thing not to do:
raising sensitivity, because YuNet scores the wearer's own hand on the mop at
0.74-0.82 and a wall sign at 0.80; only CenterFace's confirmation keeps those
out. What was built instead:

1. Confirmation crops are square and reflect-padded past the frame edge, so a
   face half out of the picture is confirmed in context.
2. CenterFace sees each crop, its mirror, and a tighter crop; the extra views
   count only at 0.4 or more (`verify_conf_view`), the plain view at 0.3. With
   "the best view counts" at 0.3 the mirror view let hands through (1.45
   percent of hand pixels on `004310`).
3. A third detector family, UltraFace (MIT, `models/ultraface.onnx`), breaks
   the tie when CenterFace scores 0.25-0.3. Below 0.25 it is never asked:
   UltraFace fires on hands too, and asking it from 0.1 up cost 5.5 percent of
   hand pixels.
4. The tracker takes the camera's own shift out of its prediction
   (`faceblur/motion.py`, phase correlation, ~1 ms a frame), links by centre
   distance when a small fast face has no overlap, allows a ten-frame gap while
   the camera moves fast, and extends the entry tail of a moving face to twelve
   frames, until the extrapolated box has left the picture. Tail growth went
   from 5 to 3 percent a frame.
5. A confirmed track may follow a face up to 35 percent of the frame as a
   person walks up to the camera; such a box never starts a track.
6. Crops are cut only for YuNet boxes at `min(conf, 0.5)` and up: a box below
   `conf` can never be confirmed, and cutting crops from 0.2 up doubled the
   confirmation time on busy footage.
7. Worker processes on a GPU are capped at four (`faceblur.detect.GPU_WORKERS`):
   each holds about 850 MB of graphs; six reached 7.3 GB of the 8 and DirectML paged.

Measurement additions: `eval/misses.py` (unconfirmed stretches, also written to
every audit record as `unconfirmed_runs`), synthetic "edge" and "pan" sequences
with an entry-delay number, the camera shift in the eval cache (version 2),
and an off-face gate that counts detector boxes alone (0.2 percent) beside the
total (0.5 percent, up from 0.3 because the longer entry tails are off-face by
construction).

## Sample footage

`footage/` holds four Ego files (264 s total). They and every output folder are
gitignored. Never commit them. The evaluation caches under `eval/cache/` are
derived from them and are gitignored too. The caches are at version 3; an older
cache is rebuilt on first use (about a minute a video with four workers).

## How to run

```
.venv\Scripts\python.exe -m ui.app                        # window
.venv\Scripts\python.exe cli.py footage -o footage_blurred
.venv\Scripts\python.exe -m pytest tests                  # tests
.venv\Scripts\python.exe -m eval.misses VIDEO --images DIR   # where faces may still be missed
.venv\Scripts\python.exe -m eval.reid VIDEO BLURRED         # is anybody still identifiable
.venv\Scripts\python.exe -m faceblur.verify VIDEO BLURRED --workers 4   # what the copy still shows
```

## Known limits and open items

- The exposure proxy (`exposed_40`, `exposed_24_40` in `eval/measure.py`)
  brackets pairs of different faces too; read it as a difference between
  settings. Section 12.3 re-measures it on rebuilt evidence: on `004100` it
  reads 91 frames where section 11 reported 55. A hand detector in the pipeline is the one thing that would
  make the hand rules unnecessary; MediaPipe's palm model needs converting
  to ONNX to run in the main environment.
- Frames where nothing detects anything inside a smear get the interpolated
  mask on the straight line between sightings, which can be far from the
  smear. Those frames are smears; whether they count as exposure is a
  judgement the harness cannot make.

- Faces under about 32 px are covered about half the time; motion blurred
  faces about three quarters. A larger detector would raise this at a cost in
  speed; `eval/sweep.py` is where to try one.
- The remaining unconfirmed stretches on `004100` are the wearer's hand on the
  mop and a wall sign; on `004310` twelve stretches, mostly hands. They are
  listed in each audit record so a batch can be spot checked where it matters.
- Detection costs about 110-130 ms per 1600x1300 frame on an RTX 3070 through
  DirectML with the new confirmation; see `docs/report.md` section 10 for the
  batch time.
- The sweep gates were loosened on the total off-face number (0.3 to 0.5
  percent) for the reason above. If tails ever need to be cheaper, the
  detector-only gate is the one that must not move.
- No installer, no code signing, no cloud runner. `faceblur.batch.run_video`
  takes a `submit` callable so a cloud runner can map jobs onto anything.
- The output-side check uses the same detectors as the pipeline, so it
  reports the wearer's hand as a face too. The track-level rules that keep
  hands out of the mask have no counterpart there, and giving it one is the
  obvious next piece of work on it.
- The recogniser bounds one thing only: what SFace can do at these distances.
  It works on `003939`, where people are at conversational distance, and is
  near useless on faces across a room. A better model, or a person who knows
  the room, is outside what any number here covers.
- Changing `faceblur/detect.py` at all, comments included, invalidates the raw
  caches by design: they carry a hash of that file. A rebuild is about a
  minute a video on four workers.
- `gh` is not installed on this PC; git credential manager handles the push.

## Files that matter

| Path | What |
|---|---|
| `faceblur/settings.py` | every setting and its default, with the reason |
| `faceblur/detect.py` | YuNet, CenterFace, UltraFace through onnxruntime; crops, views, confirmation |
| `faceblur/motion.py` | camera shift between frames |
| `faceblur/track.py` | tracker: confirmation, camera-aware prediction, continuation, tails |
| `faceblur/redact.py` | ellipse mask and pixel destruction |
| `faceblur/segments.py` | keyframe cuts, copies, encodes, concat, verify, reorder delay |
| `faceblur/batch.py` | the phased runner (`run_video`), `unconfirmed_runs` |
| `faceblur/pipeline.py` | audit record, single-process path |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/` | label-free evaluation, the sweep, the misses yardstick |
| `faceblur/verify.py` | the check that reads the output: faces still in the copy, and the quarantine gate |
| `eval/reid.py` | the recogniser: threshold on this footage, what the mask does, what is left in the copy |
| `models/sface.onnx` | face recogniser, evaluation only, never in the build |
| `docs/report.md` | quality, accuracy, speed, hardware, assumptions, done and not done |
