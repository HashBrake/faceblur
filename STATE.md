# STATE — handover notes

Last updated: 2026-09-09, second pass that day. This file says where the
project stands, what was verified, and what a new person or session should do
next. The README explains
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
- Pass six, 2026-09-09, section 14: the hand rule (`faceblur/hands.py`), and
  the eye pass over all 108 boxes the check reports that says what they
  actually are. It corrects section 13.4: the check finds real faces two and
  a half times as often as hands, so the hand rule was not what stood between
  the gate and unattended use.
- Tests: `tests/`, 874 passing (`.venv\Scripts\python.exe -m pytest tests`).
- Everything is committed and pushed: `HashBrake/faceblur`, `main` at 55687c5.
- Packaged app: `dist\FaceBlur\` builds from `build\faceblur.spec`. Not
  rebuilt since 2026-09-07, and it now has two more models to carry, so it
  needs rebuilding before it is handed to anyone. The spec ships three face
  detectors, the two hand models and their licences, and deliberately does not
  ship the recogniser; `tests/test_models.py` fails if any of that changes.
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

**With the gate on, all four sample files would be held back**, and after the
2026-09-09 pass it is clear that this is mostly for the right reason. All 108
boxes the check reports were looked at by eye: 52 are faces the run really did
miss, 20 are the wearer's hand, 30 are neither (a television, a picture on a
wall, a flat wall, a motion smear) and 6 could not be called. Sections 13.4
and 14.1 go through them.

## What changed on 2026-09-09 (the hand rule, and what the check really finds)

`faceblur/hands.py` gives the output-side check the thing it was missing: a
way to tell the wearer's hand from a face, which the pipeline does with
track-level rules that a one-frame check cannot use. Two MediaPipe models,
Apache 2.0, converted from its wheel by `models/make_hands.py` and committed:
the palm detector proposes, the hand-landmark model confirms. A flagged box a
confirmed hand covers stays in the record marked with the coverage, and is
left out of every count the gate reads. It sets aside 13 of the 20 hands and
none of the 52 faces, and it costs nothing measurable: 338 s against 341 s
with it off, on the busiest sample file. `--no-hand-rule` turns it off.

Two things a next person should take from that pass, in order:

1. **The check is mostly finding real faces, not hands.** Section 13.4 named
   the largest finds, which were hands, and left the impression that the file
   was full of them. It is not: 52 faces to 20 hands over the four files, 34
   to 16 on the table tennis file alone. The hand rule was called the thing
   standing between the gate and unattended use. It is not. What stands there
   is that these copies still show around fifty faces nothing masked, and the
   only way past that is to miss fewer of them.
2. **Two models, because one is loose.** The palm detector on its own put 138
   boxes on the crops around those 108 finds and the landmark model rejects 97
   of them; two crops in five carry one it rejects. Since a palm box implies a
   region 2.6 times its size, an invented one covers whatever face is beside
   it — which is exactly what happened on `004100` frame 450. Over 1050
   combinations of window set, palm floor, region and coverage, the palm
   detector alone loses no face in 256 of them and the best of those reaches
   11 of 20 hands, with only 4 combinations there; with the confirmation, all
   1050 lose no face and 24 of them reach 13. The point is not the two extra
   hands, it is that the geometry stops mattering.

The settings sit on a plateau, on purpose. `hand_presence` is the one that
decides; at 0.7 the rule loses no face across a palm floor of 0.3 to 0.7, a
coverage of 0.3 to 0.7 and a region of 1.6 to 2.6 times the palm box, and
sets aside 12 or 13 hands over almost all of that. Section 14.2 has the
table, and says what the same rule looks like without the second model: a
needle, where growing the region one step loses seven faces.

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
.venv\Scripts\python.exe -m faceblur.verify VIDEO BLURRED --no-hand-rule  # ... counting hands as faces
.venv\Scripts\python.exe -m eval.sweep VIDEO                    # choose settings, write the report
```

The evaluation needs `.venv-eval` for MediaPipe (the third detector and the
hand regions); see the README. `eval.reid` and `faceblur.verify` need only the
main environment.

## Known limits and open items

- **The gate still holds all four sample files, and should.** They show
  faces. Missing fewer of them is the work that would change that, and section
  12.4 already says the cheap ways of doing it (a bigger scan, tiling) were
  measured and rejected. What has not been tried is running the check's own
  detectors over the copy at a second scale, or letting a find in the copy
  send the pipeline back to that stretch of the source with the thresholds
  lowered, which is a targeted second chance rather than a blanket one.
- **Seven of the twenty hands still read as faces**, six on the table tennis
  file: a fist gripping a bat, side on, in motion, where neither window shows
  the palm detector enough hand. Reaching further costs faces, which is the
  wrong trade for a gate.
- **The 30 finds that are neither a face nor a hand are untouched** — a
  television, a picture on a wall, a flat wall, a motion smear. They hold
  copies back exactly as before. Masking a screen is a policy question before
  it is a detection one.
- **The hand models are now in the pipeline's environment**, which makes the
  older idea of using them to replace the tracker's hand rules cheap to try.
  It would mean re-tuning and re-measuring everything the track-level rules
  touch, so it was deliberately left alone: every number in this file and in
  `docs/report.md` was measured with those rules in place.
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
| `faceblur/hands.py` | palm detector and landmark model: is this box a hand |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/` | label-free evaluation, the sweep, the misses yardstick |
| `eval/common.py` | the raw cache and its fingerprint |
| `eval/reid.py` | the recogniser: threshold on this footage, what the mask does, what is left in the copy |
| `models/sface.onnx` | face recogniser, evaluation only, never in the build |
| `docs/report.md` | quality, accuracy, speed, hardware, assumptions, done and not done |
