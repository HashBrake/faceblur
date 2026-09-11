# STATE, handover notes

Last updated 2026-09-11, after work package R2 of `FACEBLUR_BUILD_PLAN_V2.md`.
Packages R1, S1, 4.3, E1 and R2 are done, committed and pushed. **H1, the
handled zone, is next and has not been started.** It is the critical path:
F2, F1, T3 and S2 all measure with the zone on.

This file says where the project stands and what to do next. `README.md` says
how to use the tool. `docs/report.md` holds the measurements, one section per
pass, and is the place to look before changing anything measured.
`FACEBLUR_BUILD_PLAN_V2.md` is the spec being executed; where it and the code
disagree the code wins and this file says so.

Read "The rule" and "Where things stand" first. Everything else is reference.

## The rule

The owner, 2026-09-11: *remove all information not pertinent to the task the
collector is doing. Anything the collector is interacting with or handling is
not blurred at all. Everything else that carries information (text, screens,
faces) is blurred.*

This is the spine of the project now and it replaced three earlier rules: the
text policy by surface, "cards are left alone", and "hands untouched, minimum
region everywhere". Two consequences are easy to miss:

- **The precision constraint moved from hands to the handled zone.** It used
  to be that the wearer's hand must never be masked, and that constraint is
  what rejected every setting that would have found more faces. Once the zone
  exists and is subtracted from every mask, the hand cannot be masked by any
  setting, and those settings can be re-measured. That is package F2, and
  nothing may be loosened before H1 lands.
- **Over masking outside the zone stopped being a failure.** A wall sign the
  collector reads is text, and text outside the zone is to be masked. YuNet
  scoring a sign at 0.80 was a false positive under the old rule and is a
  correct find under this one. What still constrains masking outside the zone
  is the per frame mask budget, so the environment stays usable, not privacy.

Inside the zone, precision is absolute: nothing is ever masked there. Outside
it, a miss is the leak and over masking of non informational things is the
cheaper error.

## What this is

FaceBlur: video in, video with the sensitive things destroyed. A Windows
window plus a command line, running on the owner's PC. Video never leaves the
machine, no cloud, no telemetry.

Three kinds, each a switch of its own (`faceblur/classes.py`): **faces**
(built and measured), **screens** as objects (built and measured), **personal
text** (not built). Plates and bodies are out of scope. Audio and container
metadata were offered on 2026-09-10 and not chosen.

Faces are on by default today. Once text is ready, all three go on by default:
that is package D and the owner's D3.

No labels and no human step anywhere in the pipeline, and none in any
measurement that gets re-run. A by eye audit is allowed once, only to check
that a rule which sets a detection aside is safe, and **its labels are
committed under `docs/audits/`**. Two earlier audits did not commit their
labels and cannot be re-scored; see "should not assume".

## The footage

`footage/` holds four Ego camera files, 1600x1300, about 30 fps, h264, no
audio, 264 s in total. They, every output folder and `eval/cache/` are
gitignored. Never commit them.

**They are not card collector footage.** They are a washroom being cleaned, a
corridor being mopped, a canteen and a table tennis hall, and the wearer
appears to be a facilities worker. Measured, not guessed: the OWLv2 oracle
asked about trading cards, graded slabs and binder pages over 1587 frames
spread across all four files reports zero at any score of 0.2 or above,
against 3213 televisions on one file alone (report section 16.2.1).

Everything else in this file is consistent with that once you look for it: the
wearer's hand on a mop, the face in a washroom mirror, the blue table tennis
table that reads as a laptop.

The card collector dataset is still what the tool is for. No number in this
repository has ever been measured on a card.

## Where things stand

Everything is committed and pushed to `HashBrake/faceblur` `main`; the working
tree is clean. Tests: `tests/`, 1046 passing, about four minutes
(`.venv\Scripts\python.exe -m pytest tests`).

| Package | What | State |
|---|---|---|
| R1 | Documents and the packaged app say what the code does | Done, report 9.1 |
| S1 | Screens in the output side check and the gate | Done, report 16.1 |
| 4.3 | Per kind mask accounting | Done, report 16.3 |
| E1 | Object oracle for evaluation | Done, report 16.2 |
| R2 | Fixes from the review of that pass | Done, report 16.6 |
| **H1** | **The handled zone** | **Next. Critical path** |
| G1 | The gate in the window | Not started |
| F2 | Recall outside the zone | Not started, needs H1 |
| F1 | Second chance for missed faces | Not started, needs R2 and H1 |
| T1 | Text detector | Not started |
| T3 | Text policy, gate, ready | Not started, needs T1 and H1 |
| S2 | Screens outside the zone | Not started, needs H1 |
| D | Defaults: all three kinds on | Not started, needs T3 |
| T4, M1 | Identifiers, metadata line | Optional |
| ~~F0~~, ~~T2~~ | Faces on cards, card veto | **Dropped by the rule, not deferred. Do not build them** |

### What to do next

**Start H1.** A per frame region nothing is ever masked in, built from the
wearer's hands with the palm and landmark models that already ship, applied to
every kind at the last step in `redact.redact`, and checked on the copy. The
spec has the geometry, the memory rule and the settings. Four things to know
before starting:

- There is **no `docs/audits/` folder yet**. H1's by eye audit of the zone on
  100 sampled frames is the first time the "commit the labels" rule is
  exercised, and it creates that folder. The format is CSV of frame numbers
  and boxes, never pixels.
- `faceblur/hands.py` already has the palm detector, the landmark model and a
  `hand_cover` helper, built for the output side check. H1 needs the hands
  themselves rather than a yes or no about one box, so read what is there
  before writing `faceblur/zone.py`.
- The zone is **not** a kind and not a `Detection`. It is a per frame list of
  polygons carried beside `per_frame` and applied as `alpha *= 1 - zone_alpha`
  after `build_alpha`. Keep it that way: the audit can then report both what
  would have been masked and what was.
- `zone=False` must reproduce today's alpha exactly. That is the test that
  says the zone is subtractive and nothing else changed.

**Do not lower any face threshold before H1 lands.** Until the zone exists,
section 10's finding stands: those settings mask the wearer's hand. After H1
they are package F2 and they are measured.

**Do not build F0 or T2.** There is nothing to measure and the rule answers
both by construction.

## Decisions taken by default

Judgement calls made without asking, each reversible, each recorded where it
lives:

- **R2: the oracle's weight hash is a constant in `eval/oracle_owl.py`, with a
  test tying it to `models/README.md`.** The spec said to verify against the
  value in the README. Parsing markdown at run time to decide whether to trust
  a model is worse than one constant plus a test that the two agree, so the
  check reads the constant and `test_the_pinned_weights_are_the_ones_the_documentation_names`
  fails if the README ever drifts from it.
- **R2: a screen over the size cap stays invisible to the check.**
  `residual_job` calls the same `screens.detect` the pipeline calls, so the 12
  percent cap applies on the copy side too. That is scope rather than a miss,
  the spec asked for the behaviour to be kept, and it is now written down in
  report section 16.4 and the README rather than left to be rediscovered.

Nothing else has been decided by default. The owner's D1, D2 and D3 are
stated in the spec and are not open.

## The numbers

Faces, on copies made with faces alone, shipped defaults, output folder
`footage_blurred_v8`, 507 s for 264 s of video:

| Measure | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| Faces two detectors agree on, covered | n/a | 100.0 % | 99.7 % | 99.7 % |
| Faces of known position, pasted in, covered | n/a | 87.8 % | 78.7 % | 86.2 % |
| Masked pixels where no detector sees a face | n/a | 0.18 % | 0.77 % | 0.43 % |
| Hand pixels touched | n/a | 0.00 % | 0.00 % | 0.09 % |
| Confirmed faces kept covered while visible | n/a | 99.0 % | 99.5 % | 93.5 % |
| Faces the copy still shows, on untouched pixels | 7 | 8 | 15 | 64 |
| ... and hands the rule set aside, not counted above | 1 | 0 | 2 | 11 |

(`003939` has no MediaPipe oracle, so it has no consensus or hand numbers.)

With screens on as well, measured 2026-09-10 to 11, output folders
`footage_blurred_r1` and `footage_blurred_43`:

| Measure | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| Frame destroyed by the face mask, mean | 0.85 % | 0.51 % | 2.31 % | 0.69 % |
| Frame destroyed by the screen mask, mean | 0.05 % | 0.43 % | 0.01 % | 0.31 % |
| Frames over the 5 percent budget: union | 0 | 6 | 24 | 40 |
| ... of which the face mask alone | 0 | 6 | 23 | 9 |
| Screens the copy still shows | 11 | 14 | 10 | 129 |
| Runs of those long enough to hold the copy back | 1 | 3 | 0 | 7 |
| Held back for | face, screen | face, screen | face | face, screen |
| Screen recall against the oracle | 0 % | 2 % | 0 % | 23 % |

If someone asks how good it is in one percentage, there is not one, and the
honest answer is three numbers: **99.7 to 100 percent** of the faces the
detectors agree on are covered; **79 to 88 percent** of faces planted at known
positions are covered, which is the hard test and the one to quote, at 83
percent for 64 px and over and 65 percent under 32 px; and hands are touched
**0.00 to 0.09 percent**. Per face sighting, not per person. Screens have no
comparable figure: 0 to 23 percent recall against one model's opinion.

On the legal side nothing sets a percentage. Both GDPR and Vietnam's Decree 13
ask whether a person can be identified by means reasonably likely to be used,
and the standard is a **human** who may already know them, not a recogniser.
Faces alone will not make this footage anonymous, because clothing, tattoos,
gait, interiors and timestamps remain. The defensible posture is to treat the
output as pseudonymised personal data and let the measurements support
"appropriate technical measures", not "no longer personal data". The audit
records and `docs/report.md` are what an auditor actually asks for.

## What a next person should not assume

**About the screens.**

- **The screen class misses most screens and holds most copies back for the
  ones it finds.** Recall against the oracle is 0 to 23 percent; three of four
  files are held back for a screen. It is fit for best effort masking with the
  gate off, not for unattended use, until S2. Report sections 16.1 and 16.2.
- **The size cap and the persistence rule are not what costs the recall.** The
  cap loses nothing the oracle calls a screen and persistence loses nine
  sightings on one file. `screen_conf` 0.5 is what loses the rest, and section
  15.1 measured what lowering it costs. S2 is where that trade gets decided.
- **A screen over the cap is invisible to the check too**, by design.
- **`screen_max_area` does not bound what is destroyed.** The cap is checked
  on the detector's box and `screen_pad` grows it afterwards, so a box at the
  12 percent cap is masked at up to about 13.4 percent of the frame.
- **The screen rules are tuned on one venue.** The table tennis hall is what
  makes the cap necessary and what sets its value.

**About the evidence.**

- **The by eye labels behind report sections 14.1 and 15.2 were never
  committed.** Only the counts survive. So the oracle's screen precision of 56
  to 68 percent and the eye's 91 percent cannot be reconciled by anybody, now
  or later, and neither audit can be checked or extended. H1's audit must not
  repeat this: commit the labels under `docs/audits/`.
- **The two face count tables above differ, and that is the copies, not the
  check.** Report section 16.1 gives 7, 9, 16 and 67 where the first table
  gives 7, 8, 15 and 64. The first reads copies made with faces alone, the
  second copies made with faces and screens, so a face detector is not looking
  at the same pixels.
- **Report section 13.4 reports 94 boxes on `005035` where the current code
  reports 75.** The other three files reproduce exactly. Nothing explained it,
  the 2026-09-08 run cannot be repeated, and the finding does not change:
  that file still shows dozens of faces.
- **The caches went stale once and nothing said so** (2026-09-08), and every
  number for a week described a build that no longer existed. Every cache now
  carries a fingerprint of the code that wrote it. Changing
  `faceblur/detect.py` at all, comments included, invalidates the raw caches
  by design.
- **The recogniser bounds one thing only**: what SFace can do at these
  distances. It works on `003939`, where people are at conversational
  distance, and is near useless across a room.
- **The exposure proxy brackets pairs of different faces too.** Read
  `exposed_40` and `exposed_24_40` as a difference between settings, not as a
  count of exposed faces.

**About what is not built.**

- **The window cannot run the check or the gate at all.** `ui/app.py` never
  sets `check_output` or `quarantine`, so no run started from the window is
  checked or held back, and the person most likely to need the gate is the one
  using the window. That is package G1.
- **The gate holds all four sample files, and should.** They show faces.
  Package F1 is the work that would change that.
- **Seven of the twenty hands still read as faces to the output side check**,
  six on the table tennis file: a fist gripping a bat, side on, in motion. H1
  should make this moot for masking, but the check's own hand rule stays as
  belt and braces.
- **No installer, no code signing, no cloud runner.** `batch.run_video` takes a
  `submit` callable so a cloud runner can be written without touching the
  pipeline.

## How to run

```
.venv\Scripts\python.exe -m ui.app                              # window
.venv\Scripts\python.exe cli.py footage -o out                  # faces
.venv\Scripts\python.exe cli.py footage -o out --mask face,screen
.venv\Scripts\python.exe cli.py footage -o out --quarantine     # with the gate
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m faceblur.verify VIDEO BLURRED --workers 4 --mask face,screen
.venv\Scripts\python.exe -m eval.misses VIDEO --images DIR      # where faces may still be missed
.venv\Scripts\python.exe -m eval.reid VIDEO BLURRED             # is anybody still identifiable
.venv\Scripts\python.exe -m eval.sweep VIDEO                    # choose settings, write the report
.venv\Scripts\python.exe -m eval.screens VIDEO                  # the screen class against the oracle
.venv-oracle\Scripts\python.exe eval\oracle_owl.py VIDEO --stride 5
dist\FaceBlur\FaceBlur.exe footage -o out --mask face,screen    # packaged, no Python needed
```

Three environments. `.venv` runs everything that ships. `.venv-eval` has
MediaPipe, for the third face detector and the hand regions. `.venv-oracle`
has torch and OWLv2. Three rather than two because MediaPipe pins numpy below
2 and every face number here was measured through MediaPipe; loosening that
pin to save disk would put the face side's evidence at risk.

`faceblur.verify` exits 1 when the gate would hold the copy back and 0
otherwise, and prints everything it found either way.

`eval.oracle_owl` takes about 4.2 s a frame on the processor, so the four
files at stride 5 are about two hours. It resumes after a kill, checks the
sha256 of its weights before it starts, and its cache carries a fingerprint of
its own source.

Detection costs about 110 to 130 ms per frame on the RTX 3070 through
DirectML. The output side check costs about the same again. Do not raise
`detect.GPU_WORKERS` above four: each worker holds about 850 MB of graphs and
six reached 7.3 GB of the 8 and DirectML paged.

The desktop shortcut `FaceBlur.lnk` on this PC launches
`.venv\Scripts\pythonw.exe -m ui.app`, so it always runs the current code.
`gh` is not installed; git credential manager handles the push.

## How it got here

One line per pass. The report has each one in full and is the place to read
before changing anything a pass measured.

| When | What | Report |
|---|---|---|
| 2026-09-05 to 06 | First build, then the precision rebuild that followed it destroying 33 percent of every frame and 82 percent of hand pixels | `docs/precision_audit.md`, 3 to 9 |
| 2026-09-07 | Missed faces: confirmation crops, three detector families, camera aware tracking. Found that raising sensitivity masks the wearer's hand on a mop | 10 |
| 2026-09-08 | Faces lost during a hit: track stitching across a smear. Then identity measured directly with a recogniser, which also found the caches were stale | 11, 12 |
| 2026-09-08 to 09 | `faceblur/verify.py`, the check that reads the copy instead of the source, and the `--quarantine` gate | 13 |
| 2026-09-09 | The hand rule: two MediaPipe models tell the wearer's hand from a face in the check. The eye pass over 108 boxes corrected 13.4 | 14 |
| 2026-09-10 | Scope to three kinds, the kind registry, and the screen class. The detector failed its first run and two rules fixed it | 15 |
| 2026-09-10 to 11 | R1, S1, 4.3, E1: documents and build caught up, screens in the gate, per kind accounting, the object oracle | 16.1 to 16.3 |
| 2026-09-11 | The owner restated the goal as one rule. The spec was rewritten around it; F0 and T2 dropped | spec section 3 |
| 2026-09-11 | R2: review fixes, exit code follows the gate, rows carry their box, imports checked by `ast` | 16.6 |

## Open questions for the owner

- **Card footage.** The rule no longer depends on cards, but nothing about
  card text or printed faces can be measured until there is footage with cards
  in it. One file of a collector at work would do.
- **A screen someone else holds up to the wearer**, a phone shown across a
  table. By the rule that is handled and protected; by intent it may not be.
  The default protects it.
- **A second venue.** Every screen and text setting will be tuned on the table
  tennis hall and the canteen otherwise.
- **Disk.** The machine sits near full. The owner frees space; this session
  does not delete `footage_blurred*` folders or environments.

## Files that matter

| Path | What |
|---|---|
| `FACEBLUR_BUILD_PLAN_V2.md` | the spec being executed; where it and the code disagree, the code wins |
| `faceblur/settings.py` | every setting and its default, with the reason |
| `faceblur/detect.py` | YuNet, CenterFace, UltraFace; crops, views, confirmation |
| `faceblur/motion.py` | camera shift between frames |
| `faceblur/track.py` | tracker: confirmation, camera aware prediction, continuation, tails |
| `faceblur/redact.py` | mask shapes, pixel destruction, per kind shares. H1 adds the zone here |
| `faceblur/segments.py` | keyframe cuts, copies, encodes, concat, verify |
| `faceblur/batch.py` | the phased runner `run_video`, the worker side model caches |
| `faceblur/pipeline.py` | the audit record, single process path |
| `faceblur/classes.py` | the kinds, and which are ready |
| `faceblur/screens.py` | YOLOX, the size cap, the persistence rule |
| `faceblur/verify.py` | the check that reads the copy, `held_for`, the gate |
| `faceblur/hands.py` | palm detector and landmark model. H1 builds on this |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/common.py` | the raw cache and its fingerprint |
| `eval/measure.py`, `eval/sweep.py` | the label free harness and the gates |
| `eval/reid.py` | the recogniser: what the mask does, what is left in the copy |
| `eval/oracle_owl.py` | the object oracle: OWLv2, pinned, never ships |
| `eval/screens.py` | the screen class scored against the oracle |
| `models/README.md` | every model: source, sha256, input layout, output layout |
| `docs/report.md` | every measurement, one section per pass |
| `docs/audits/` | by eye labels. Does not exist yet; H1 creates it |
