# STATE — handover notes

Last updated: 2026-09-10, after work package R1. This file says where the
project stands, what was verified, and what a new person or session should do
next. The README explains how to use the tool; `docs/report.md` holds the
measurements, one section per pass, and is the place to look before changing
anything measured.

## What this is, and what it is becoming

**Scope changed on 2026-09-10.** The owner asked for sensitive information
beyond faces. It is now three kinds, each a switch of its own: **faces**
(built, sections 3 and 10 to 14), **screens** as objects (built the same day,
section 15), **personal text** (not built). Audio and container metadata were
offered at the same time and not chosen. The decision and the two rules it
came with are in `FACEBLUR_BUILD_PLAN.md` under "Scope change"; the one that
shapes the work still to do is that **cards are left alone**, because on this
footage card names, prices and grading labels are the point of the dataset,
so only personal text is ever masked.

Faces are on by default and screens are not. Widening what the tool destroys
is a decision on the day, not a side effect of an upgrade.

Most of what follows is about faces, which is where almost all the
measurement is. Section 15 and the 2026-09-10 note below are the screens.

FaceBlur: video in, video with the sensitive things destroyed. Windows
desktop window plus a command line. No labels and no human step anywhere in
the pipeline, and none in any measurement that gets re-run — with two
exceptions, both recorded where they are used and both the same shape of
problem. Deciding whether a rule that sets a detection aside is safe means
knowing what those detections actually are, and nothing in this repository
knows: section 14.1, the 108 boxes the output-side check reported, and
section 15.2, the 85 screen runs. Each was labelled by eye, once, by one
person. Nothing the pipeline does depends on a label. Built for Ego camera footage (1600x1300, ~30 fps, h264,
no audio) of card collectors, where the goal is: destroy the minimum region
that hides a person's identity and leave hands, cards, screens and everything
else untouched.

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
- Pass seven, 2026-09-10, section 15: the scope change, the kind registry
  that makes each class a switch of its own (`faceblur/classes.py`), and the
  screen class (`faceblur/screens.py`). The screen detector failed its first
  run over real footage and the two rules that fixed it are the interesting
  part; see the note below.
- Pass six, 2026-09-09, section 14: the hand rule (`faceblur/hands.py`), and
  the eye pass over all 108 boxes the check reports that says what they
  actually are. It corrects section 13.4: the check finds real faces two and
  a half times as often as hands, so the hand rule was not what stood between
  the gate and unattended use.
- Tests: `tests/`, 999 passing (`.venv\Scripts\python.exe -m pytest tests`).
- Everything is committed and pushed to `HashBrake/faceblur`, `main`, and the
  working tree is clean. The last change of substance is e2663c0, the hand
  rule; commits after it are these notes catching up.
- Packaged app: `dist\FaceBlur\` builds from `build\faceblur.spec`, rebuilt on
  2026-09-10 in work package R1. It carries three face detectors, two hand
  models, the screen detector and their licences, and deliberately not the
  recogniser. `FaceBlur.exe` with arguments is now the command line as well as
  the window, so the packaged build can be checked without a person clicking.
  `tests/test_models.py` fails if any `.onnx` in `models/` is neither in the
  spec nor listed in `NOT_SHIPPED` with a reason.
- Desktop shortcut `FaceBlur.lnk` on this PC launches `.venv\Scripts\pythonw.exe -m ui.app`,
  so it always runs the current code.

## The build plan being executed, and the packages done so far

`FACEBLUR_BUILD_PLAN_V2.md` is the spec this session and the ones after it are
executing. It was written on 2026-09-10 from an audit of the v1 plan against
what was actually built, and where the two disagree v2 wins. It runs the work
in this order: R1, S1, the per kind mask accounting of its section 4.3, E1,
F0 measure, F1, T1, T2, F0 veto, T3, T4 if there is time, then M1. About four
weeks. Each package ends with its verify step passing, its report section
written and one commit.

| Package | What | State |
|---|---|---|
| R1 | Housekeeping: documents and the packaged app say what the code does | Done 2026-09-10, report section 9.1 |
| S1 | Screens in the output side check and the gate | Next |
| 4.3 | Per kind mask accounting | Not started |
| E1 | Object oracle for evaluation | Not started |
| F0 | Faces printed on cards | Not started |
| F1 | Second chance for missed faces | Not started |
| T1 to T4 | Personal text | Not started |
| M1 | Metadata line in the record | Not started |

### Decisions taken by default

None yet. The three the spec leaves to the owner are D1 (faces printed on
cards), D2 (which surfaces text is left on) and D3 (screens off by default).
Each will be built to the spec's proposed default, made a setting, and listed
here when its package lands, so that any of them can be reversed with one
setting.

### R1, done 2026-09-10

The documents, the model list and the packaged app were all behind the code.
What changed:

- `README.md` describes three kinds rather than one, with a table saying which
  are built and which are on by default, a section on how a screen is decided
  and the four caveats that come with it, the full model list including
  YOLOX-tiny and the two hand models, and a command line block that matches
  `cli.py`.
- `docs/report.md` section 7 no longer says text and screens are both out of
  scope; section 2 lists every model and which of them ship; section 9.1 is
  new and says what the packaged app carries and how that is now checked.
- The scope change table in `FACEBLUR_BUILD_PLAN.md` says screens are built.
- `build\faceblur.spec` lost `centerface.onnx`, which is 7 MB of a static
  graph nothing loads at run time, and gained `cli` as a hidden import.
- `faceblur_app.py`: `FaceBlur.exe` with arguments runs the command line, so
  the packaged build has a verify step that is a command and not a person.
- `tests/test_models.py` gained two tests: every `.onnx` in `models/` is
  either in the spec or in `NOT_SHIPPED` with the reason the build does not
  need it, and every one has a pinned sha256. 999 tests pass.
- `dist\FaceBlur` was rebuilt. It was five days old and carried neither the
  hand models nor the screen model, so a person handed that folder had a
  screens checkbox that could not work.

The verify run was `dist\FaceBlur\FaceBlur.exe footage -o footage_blurred_r1
--mask face,screen --workers 4`. It wrote four copies in 556 s, every record
carrying `screens` above zero, and `005035` reproduced section 15.4 to the
decimal: 0.99 percent of the average frame, 13.3 percent at worst, 40 frames
over budget, 848 frames carrying a screen mask. The packaged build and the
source build agree.

What a next person should not assume from this package: nothing here measured
anything new about faces or screens. R1 moved documents and the build to where
the code already was.

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
| Faces the copy still shows, on untouched pixels | 7 | 8 | 15 | 64 |
| ... and hands the rule set aside, not counted above | 1 | 0 | 2 | 11 |

(`003939` has no MediaPipe oracle, so it has no consensus or hand numbers.)

The last two rows were re-measured on 2026-09-09 with the hand rule on.
Section 13.4 reports 94 boxes on `005035` where two runs of the current code
report 75; the other three files reproduce exactly. Nothing was found to
explain it and the 2026-09-08 run cannot be repeated, so the section keeps its
number, this table carries the one the code now gives, and a third party
should trust the second. Whatever it was, it does not change the finding:
that file still shows dozens of faces.

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

## What changed on 2026-09-10 (the scope, and screens)

Two things, and the second is a lesson rather than a feature.

**Each kind is a switch.** `faceblur/classes.py` is a registry: a kind carries
its name, its label, the shape of its mask and whether a detector exists
behind it. The window's checkboxes, the `--mask` flag and its help text are
all derived from that last field, so a switch cannot promise more than the
build can do, and a kind that lands needs one line changed. `Detection` now
carries a `kind`, and `redact.py` has a polygon path beside the ellipse: an
ellipse over a line of text spills onto the page in the middle and misses the
first and last characters. Blocks for a quad are sized by its **short** side,
because sizing a 400 px line of text by its length gives one block across it.

**The screen class failed its first run, and the fix is worth reading before
adding a fourth kind.** YOLOX-tiny, Apache 2.0, decode verified against a
frame by eye. Run end to end on `005035` it destroyed 2.84 percent of the
average frame and 42.7 percent of the worst, and put 200 frames over the mask
budget where faces alone put none.

The cause was not the threshold. A COCO detector calls any large flat
rectangle a television, and this footage is made of them: **the blue table
tennis table is detected as a laptop, a television and a phone by turns**.
Washroom mirrors and glass walls make up the rest. The score is no help at
all: the table reads 0.88 over a third of the frame, the real television
across the hall reads 0.85 over one percent. A bigger model is no help
either, and is worse: YOLOX-s misses the real television outright.

All 85 detection runs across the four files were looked at, one at a time: 54
real screens, 29 not, 2 that could not be called. Two rules came out of that,
and the reason both are needed is the useful part:

1. **Seen in three frames before anything is masked** (`screen_min_run`). 14
   of the 29 false runs last a single frame and not one real screen does. A
   table looks like a laptop from some angles and not others; a monitor on a
   wall stays a monitor. Same idea as `established_after` on the face side.
2. **A cap on the box at 12 percent of the frame** (`screen_max_area`).

A cap on its own has to sit at 4 percent to reach 91 percent precision, and
at 4 percent it throws away a real monitor covering 10.6 percent of the
frame. That is the worst screen to lose: one that large and that close is the
one most likely to be showing something readable. With persistence carrying
the precision the cap can sit three times looser at the same 91 percent, and
that monitor is kept. Going to four frames buys two points and costs fifteen
more real screens, because most sightings of the wall television are bursts
of three or four frames as the camera swings past.

End to end on `005035`, the file that showed the problem:

| | mean | p95 | worst | over budget |
|---|---|---|---|---|
| faces only | 0.69 % | 1.90 % | 8.3 % | 9 |
| plus screens, first try | 2.84 % | 15.77 % | 42.7 % | 357 |
| plus screens, cap only | 1.05 % | 2.57 % | 8.3 % | 17 |
| plus screens, shipped rule | 0.99 % | 2.56 % | 13.3 % | 40 |

Persistence lowered the masked frames from 1326 to 848 and kept the large
monitor: it drops flickers, not screens. The worst frame and the extra
over-budget frames are that monitor, correctly masked.

What a next person should not assume about screens:

- **There is no recall number.** There is no oracle for screens on this
  footage, so unlike faces there is precision against one person's eye over
  85 runs and nothing else. A screen no detector ever finds is not counted.
- **9 percent of what it masks is not a screen**, in small patches: 2.4
  percent of a frame of washroom wall for 10 frames, 0.4 percent for 27.
- **The cap is tuned on one venue.** The table tennis hall is what makes it
  necessary and what sets its value. A room with a large monitor close to the
  camera and no large flat furniture would want a looser one, and there is no
  footage here to set it on.
- **The output-side check does not cover screens.** `verify.py` reads the
  copy for faces only. A screen the run missed is not reported and does not
  hold a copy back.

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
.venv\Scripts\python.exe cli.py footage -o out --mask face,screen   # screens too
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

- **Screens are measured for precision only, and only on this venue.** The
  four caveats are in the 2026-09-10 note above. The first thing to build for
  them is coverage in `verify.py`, so that a missed screen holds a copy back
  the way a missed face does.
- **Personal text is the piece still owed.** Two to three weeks, most of it
  the label-free evaluation rather than the detector, and the rule that cards
  are left alone is what makes it hard: OCR on this footage works on text
  over roughly 20 to 25 px and sharp, and where it fails you do not know what
  was left. Expect the text gate to hold copies back far more often than the
  face gate does, and say so before anyone runs a batch.
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
| `faceblur/classes.py` | the kinds of thing that can be masked, and which are ready |
| `faceblur/screens.py` | YOLOX, the size cap and the persistence rule |
| `faceblur/verify.py` | the check that reads the output, and the gate |
| `faceblur/hands.py` | palm detector and landmark model: is this box a hand |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/` | label-free evaluation, the sweep, the misses yardstick |
| `eval/common.py` | the raw cache and its fingerprint |
| `eval/reid.py` | the recogniser: threshold on this footage, what the mask does, what is left in the copy |
| `models/sface.onnx` | face recogniser, evaluation only, never in the build |
| `docs/report.md` | quality, accuracy, speed, hardware, assumptions, done and not done |
