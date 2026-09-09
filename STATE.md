# STATE — handover notes

Last updated: 2026-09-09. This file says where the project stands, what was
verified, and what a new person or session should do next. The README explains
how to use the tool; `docs/report.md` holds the measurements, one section per
pass, and is the place to look before changing anything measured.

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
- Pass two, 2026-09-07, section 10: faces at the frame edge, faces turned down,
  faces during a fast pan. Hands still 0.00 percent.
- Pass three, 2026-09-08, section 11: faces lost while the wearer hits a ball
  (track stitching across a smear, gap filling on the smear's own boxes,
  blur-sized masks), and the wearer's hand kept out by track-level rules.
- Pass four, 2026-09-08, section 12: identity measured directly with a face
  recogniser instead of by proxy. It also found that the evaluation caches
  were stale, so every number since 2026-09-07 had described a build that no
  longer existed.
- Pass five, 2026-09-08 to 09, section 13: `faceblur/verify.py`, the check
  that reads the output instead of the source, and the `--quarantine` gate.
- Tests: `tests/`, 839 passing (`.venv\Scripts\python.exe -m pytest tests`).
- Everything is committed and pushed: `HashBrake/faceblur`, `main` at 55687c5.
- Packaged app: `dist\FaceBlur\` builds from `build\faceblur.spec`. Not
  rebuilt since 2026-09-07. The spec ships three detectors and their licences
  and deliberately does not ship the recogniser; `tests/test_models.py` fails
  if either changes.
- Desktop shortcut `FaceBlur.lnk` on this PC launches `.venv\Scripts\pythonw.exe -m ui.app`,
  so it always runs the current code.

## The numbers, and what to say about them

Measured on the four sample files with the shipped defaults, on rebuilt
evidence (see "the caches were stale" below). Output folder `footage_blurred_v8`,
507 s for 264 s of video.

| Measure | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| Faces two detectors agree on, covered | — | 100.0 % | 99.7 % | 99.7 % |
| Faces of known position, pasted in, covered | — | 87.8 % | 78.7 % | 86.2 % |
| Masked pixels where no detector sees a face | — | 0.18 % | 0.77 % | 0.43 % |
| Hand pixels touched | — | 0.00 % | 0.00 % | 0.09 % |
| Confirmed faces kept covered while visible | — | 99.0 % | 99.5 % | 93.5 % |
| Face-shaped boxes still in the copy, untouched | 8 | 8 | 17 | 94 |

(`003939` has no MediaPipe oracle, so it has no consensus or hand numbers.)

If someone asks how good it is in one percentage, there isn't one, and the
honest answer is three numbers: **99.7 to 100 percent** of the faces the
detectors agree on are covered; **79 to 88 percent** of faces planted at known
positions are covered, which is the hard test and the one to quote — 83
percent at 64 px and over, 65 percent under 32 px; and hands are touched
**0.00 to 0.09 percent**. Per face-sighting, not per person.

On the legal side, nothing sets a percentage. Both GDPR and Vietnam's Decree
13 ask whether a person can be identified by means reasonably likely to be
used, and the standard is a **human** who may already know them, not a
recogniser. Faces alone will not make this footage anonymous — clothing,
tattoos, gait, interiors and timestamps remain — so the defensible posture is
to treat the output as pseudonymised personal data and let the measurements
support "appropriate technical measures", not "no longer personal data". The
audit records and `docs/report.md` are what an auditor actually asks for.

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
   frames, until the extrapolated box has left the picture.
5. A confirmed track may follow a face up to 35 percent of the frame as a
   person walks up to the camera; such a box never starts a track.
6. Crops are cut only for YuNet boxes at `min(conf, 0.5)` and up: a box below
   `conf` can never be confirmed, and cutting crops from 0.2 up doubled the
   confirmation time on busy footage.
7. Worker processes on a GPU are capped at four (`faceblur.detect.GPU_WORKERS`):
   each holds about 850 MB of graphs; six reached 7.3 GB of the 8 and DirectML paged.

Pass three then added track stitching across a smear, gap filling on the
smear's own boxes, and the rule that only an established, sure track gets
continuation, tails or backward reach — which is what keeps the wearer's hand
out. Section 11 has the tables.

## What changed on 2026-09-08 (identity, and the evidence it rests on)

Everything until then asked whether a mask covered a box. `eval/reid.py` asks
whether anybody is still identifiable, with SFace (`models/sface.onnx`,
Apache 2.0, evaluation only, never in the build). Three findings, in order of
how much they matter to a next person:

1. **The caches were stale.** The raw candidate caches under `eval/cache/`
   had been built halfway through the 2026-09-07 pass, before that pass
   settled which boxes get a confirmation crop. Nothing said so. Caches and
   synthetic sets now carry a fingerprint of the code and labels they came
   from and rebuild when it moves (`eval/common.py`, `eval/synthetic.py`,
   `tests/test_eval_cache.py`). Re-measured, recall reads better than section
   11 claimed and the exposure proxy on `004100` reads worse: 91 frames where
   section 11 reported 55.
2. **The shipped mask strength is right, and the next setting up is not.**
   `strength` is blocks across a face, so 12 keeps four times the detail of 6.
   On `003939`, the one file where the recogniser genuinely works, 6 blocks
   defeat it on all sixty of the largest faces and 12 blocks leave ten
   recognisable. 4 blocks and solid are clean everywhere.
3. **Chasing smaller faces is not worth it.** A 2560 pass costs 65 percent
   more time for 3 percent more boxes; 2x2 tiling costs 103 percent more and
   puts a box on the wearer's hand. Under 32 px the recogniser cannot identify
   anybody in the untouched source either. The tiling code was written,
   measured and removed; section 12.4 is what remains of it.

Read the leak numbers with their controls: a probe whose crop is mostly
unchanged room, or whose face the recogniser cannot match to the same person
elsewhere in the source, decides nothing and is reported separately.

## What changed on 2026-09-08 to 09 (the check that reads the output)

Every other measurement reads the source and inherits the pipeline's own
blindness: a miss is invisible to the thing that missed it.
`faceblur/verify.py` detects on the finished copy and asks of each box whether
anything happened there — comparing where a mask would land, at the peak
rather than the average, with the encoder's own noise off both sides.

- `--check-output` writes what it finds into the audit record.
- `--quarantine` moves a copy that still shows a face of 24 px or more into a
  `quarantine` folder beside the output instead of shipping it. The file is
  kept, since it is the only blurred copy of that video, and its record names
  the frames.
- `python -m faceblur.verify SOURCE COPY --workers 4` runs the same check on
  copies already written; exit code 1 when it finds anything.

It found real misses immediately, including a 98 px face at the frame edge and
a 40 px face in a washroom mirror that the source-side detectors do not find
at all. It costs about the same as the detection pass (32 s against 37 s on a
984 frame file, four workers).

**With the gate on, all four sample files would be held back.** Some of that
is real misses and some is the check inheriting what the detectors get wrong:
it has no counterpart to the track-level rules that keep hands out of the
mask, so on the table tennis file most of its finds are the wearer's hand on
the bat. Section 13.4 goes through them one by one.

## Sample footage and outputs

`footage/` holds four Ego files (264 s total). They, every output folder and
`eval/cache/` are gitignored. Never commit them. `footage_blurred_v8` is the
current output set, made with the shipped defaults on 2026-09-08; the older
`footage_blurred*` folders are from earlier builds and can be deleted (about
3 GB). Rebuilding an evaluation cache takes about a minute a video on four
workers, and happens on its own when the fingerprint no longer matches.

## How to run

```
.venv\Scripts\python.exe -m ui.app                        # window
.venv\Scripts\python.exe cli.py footage -o footage_blurred_v9
.venv\Scripts\python.exe cli.py footage -o out --quarantine    # with the gate
.venv\Scripts\python.exe -m pytest tests                  # tests
.venv\Scripts\python.exe -m eval.misses VIDEO --images DIR      # where faces may still be missed
.venv\Scripts\python.exe -m eval.reid VIDEO BLURRED             # is anybody still identifiable
.venv\Scripts\python.exe -m faceblur.verify VIDEO BLURRED --workers 4   # what the copy still shows
.venv\Scripts\python.exe -m eval.sweep VIDEO                    # choose settings, write the report
```

The evaluation needs `.venv-eval` for MediaPipe (the third detector and the
hand regions); see the README. `eval.reid` and `faceblur.verify` need only the
main environment.

## Known limits and open items

- **The output-side check has no hand rule.** It uses the same detectors as
  the pipeline, which call the wearer's hand a face, and the track-level rules
  that keep hands out of the mask have no counterpart there. Giving it one is
  the obvious next piece of work, and it is what stands between the gate and
  being usable unattended on the table tennis footage.
- **A hand detector in the pipeline** is the one thing that would make the
  hand rules unnecessary altogether. MediaPipe's palm model needs converting
  to ONNX to run in the main environment.
- The exposure proxy (`exposed_40`, `exposed_24_40` in `eval/measure.py`)
  brackets pairs of different faces too; read it as a difference between
  settings, not as a count of exposed faces.
- Frames where nothing detects anything inside a smear get the interpolated
  mask on the straight line between sightings, which can be far from the
  smear. Whether those count as exposure is a judgement the harness cannot
  make.
- Faces under 32 px and motion blurred faces are each covered about two
  thirds of the time, against 83 percent for faces of 64 px and over. Section
  12.4 says why raising that with a bigger scan was rejected.
- The recogniser bounds one thing only: what SFace can do at these distances.
  It works on `003939`, where people are at conversational distance, and is
  near useless on faces across a room. A better model, or a person who knows
  the room, is outside what any number here covers.
- Detection costs about 110-130 ms per 1600x1300 frame on an RTX 3070 through
  DirectML. The output-side check costs the same again when it is on.
- Changing `faceblur/detect.py` at all, comments included, invalidates the raw
  caches by design: they carry a hash of that file.
- The sweep gates sit at 0.7 percent off-face total and 0.3 percent from
  detector boxes. If tails ever need to be cheaper, the detector-only gate is
  the one that must not move.
- No installer, no code signing, no cloud runner. `faceblur.batch.run_video`
  takes a `submit` callable so a cloud runner can map jobs onto anything.
- `gh` is not installed on this PC; git credential manager handles the push.

## If the scope grows: PII beyond faces

Asked on 2026-09-09, assessed, not built. The short version:

- The architecture carries over. A new PII class is a new detector family;
  the phased runner, segment encoder, tracker, audit record and the
  `--quarantine` gate do not care what they are masking. `redact.py` would
  need a polygon path next to the ellipse one, since text is not oval.
- Cheapest wins first: strip container metadata and any GPS, and decide the
  audio policy. Audio is copied through untouched today and carries names and
  voiceprints; for a dataset, stripping it is the honest default. Hours of
  work, not days.
- Text is the real work, roughly two to three weeks, and most of that is the
  label-free evaluation rather than the detector — the same split as the face
  work. A DBNet or PP-OCR detector (Apache 2.0, ONNX) at two scales about
  doubles the detect phase.
- The hard question is not technical: on card-collector footage, text is often
  the signal rather than the noise. Blanket text masking is easy and may
  destroy what the dataset is for, so phase one is a written policy on which
  text is personal and which is product.
- Selective redaction (OCR, then classify names, phones, addresses) has a much
  lower ceiling: OCR on this footage works on text over roughly 20-25 px and
  sharp, and when it fails you do not know what you left. The safe fallback
  collapses back into blanket masking.
- Screens as objects are about a week (beware AGPL on most YOLO derivatives);
  plates a few days and only if outdoor footage matters.

## Files that matter

| Path | What |
|---|---|
| `faceblur/settings.py` | every setting and its default, with the reason |
| `faceblur/detect.py` | YuNet, CenterFace, UltraFace through onnxruntime; crops, views, confirmation |
| `faceblur/motion.py` | camera shift between frames |
| `faceblur/track.py` | tracker: confirmation, camera-aware prediction, continuation, tails |
| `faceblur/redact.py` | ellipse mask and pixel destruction |
| `faceblur/segments.py` | keyframe cuts, copies, encodes, concat, verify, reorder delay |
| `faceblur/batch.py` | the phased runner (`run_video`), `unconfirmed_runs`, quarantine |
| `faceblur/pipeline.py` | audit record, single-process path |
| `faceblur/verify.py` | the check that reads the output, and the gate |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/` | label-free evaluation, the sweep, the misses yardstick |
| `eval/common.py` | the raw cache and its fingerprint |
| `eval/reid.py` | the recogniser: threshold on this footage, what the mask does, what is left in the copy |
| `models/sface.onnx` | face recogniser, evaluation only, never in the build |
| `docs/report.md` | quality, accuracy, speed, hardware, assumptions, done and not done |
