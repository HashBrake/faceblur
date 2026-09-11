# STATE, handover notes

Last updated 2026-09-11, after work package H2 of `FACEBLUR_BUILD_PLAN_V2.md`.
Packages R1, S1, 4.3, E1, R2, H1 (the handled zone) and **H2 (its follow ups)**
are done, committed and pushed. **G1, the gate in the window, is next.** F2,
F1, T3 and S2 all have the zone to measure with.

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

H1 built the zone and found that "inside the zone" needs two definitions, not
one. What the wearer **reaches over** is what they are holding, so text and
screens there are part of the task. A face inside somebody's reach is usually
**their own face**, so a face is protected only where a hand is actually on
it. Before that split the zone was protecting bystanders' faces from
redaction, measured on 20 of 27 labelled cases. Report section 17.5.

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
tree is clean. Tests: `tests/`, 1084 passing, about four minutes
(`.venv\Scripts\python.exe -m pytest tests`).

| Package | What | State |
|---|---|---|
| R1 | Documents and the packaged app say what the code does | Done, report 9.1 |
| S1 | Screens in the output side check and the gate | Done, report 16.1 |
| 4.3 | Per kind mask accounting | Done, report 16.3 |
| E1 | Object oracle for evaluation | Done, report 16.2 |
| R2 | Fixes from the review of that pass | Done, report 16.6 |
| H1 | The handled zone | Done, report 17 |
| H2 | Zone follow ups | Done, report 17.3 and 17.7 |
| **G1** | **The gate in the window** | **Next** |
| F2 | Recall outside the zone | Not started. Needs its two harness prerequisites |
| F1 | Second chance for missed faces | Not started |
| T1 | Text detector | Not started |
| T3 | Text policy, gate, ready | Not started, needs T1 and the reach gate the spec now asks for |
| S2 | Screens outside the zone | Not started |
| D | Defaults: all three kinds on | Not started, needs T3 |
| T4, M1 | Identifiers, metadata line | Optional |
| ~~F0~~, ~~T2~~ | Faces on cards, card veto | **Dropped by the rule, not deferred. Do not build them** |

### What to do next

**Start G1**, the gate in the window. A day or two, and it closes the oldest
gap in the project: `ui/app.py` never sets `check_output` or `quarantine`, so
nobody who uses the window gets the check or the gate, and the window is the
workflow this tool was built around. Every hook the spec names was confirmed
to exist on 2026-09-11:

- `ui/app.py` `_build_advanced` is where the checkbox goes, beside
  `replace_check`, which is the pattern to copy.
- `run_video` already emits a `checking` progress stage, and `ui/strings.py`
  already has `STATUS_CHECKING` and its mark. Nothing new is needed in the
  worker to make the status word appear.
- The record carries `held_back_for`, so the summary line can name the kinds a
  copy was held back for rather than just counting them.
- `tests/manual_ui.md` exists and is the file to update beside
  `tests/test_ui.py`.

Say in the help text that the check costs a second detection pass, because it
roughly doubles a run and the person ticking it should know before they start
a batch overnight.

**F2 is the point of H1.** Every setting section 10
rejected was rejected for masking the wearer's hand, and the zone subtracts
the hand from the mask whatever the threshold says. Re-measure `conf` 0.4 to
0.6, `engine both`, `verify` off, the mirror view at 0.3, `third_conf` down to
0.1 and a 2560 scan, with the zone on. Two things to fix in the harness first:

- `eval/measure.py` does not know the zone exists, so a sweep measures masks
  the pipeline would not apply. F2 has to apply the zone there or its numbers
  are about a build that does not ship.
- **`hand_damage` has changed meaning** and is no longer expected to be zero.
  MediaPipe's hulls are every hand in the frame and the rule protects only the
  wearer's, so a mask on a bystander's hand is permitted now. Score hulls that
  intersect the zone separately from those that do not before leaning on that
  gate. Section 17.4.

**Do not lower any face threshold outside F2**, and do not build F0 or T2.

## Decisions taken by default

Judgement calls made without asking, each reversible, each recorded where it
lives:

- **H1: a face is protected only where a hand is on it** (`zone_face_needs_hand`,
  on), at 1.6 times the palm box (`zone_face_scale`). The audit found the zone
  protecting bystanders' faces from redaction in 20 of 27 labelled cases, and
  this is the fix. **It reverses the owner's D1**: a printed face on a card
  held in the hand is now masked, because the card sits in the reach and not
  under the hand. That cannot be tested here, because there are no cards in
  this footage, and the trade was taken deliberately: masking a mass produced
  photograph on a card is a smaller harm than leaving a stranger's real face
  in a copy. `zone_face_needs_hand=False` gives D1 back. Report section 17.5.
  **Ratified 2026-09-11**: the owner delegated the call and it stands, pending
  H2 item 2, which is the only thing that may reopen it.
- **H1: the precision gate tests the hand, not the reach.** Once a face may be
  masked inside the reach on purpose, a gate measuring the reach reports the
  rule working as the rule broken. The gate measures where the promise is
  absolute. What the reach prevented is reported by the pipeline instead, as
  `masked_in_zone_prevented`. Section 17.6.
- **H1: `zone_stride` stays 1.** Stride 2 saves 18 percent of the detect phase
  and halves the hand evidence. That is a bad trade for the one promise that
  is absolute.
- **H2: `zone_orientation` was measured and not shipped.** The build plan asked
  for the wrist to knuckle direction as a third "whose hands" condition. Scored
  against the 179 committed labels the two distributions lie on top of each
  other: the wearer's hands run 1 to 96 degrees from straight up and a
  bystander's 7 to 105. At the proposed 100 degrees it drops 1 bystander of 20
  and nothing else; at 60 it drops 6 and costs 18 of the wearer's own hands.
  That refutes the hypothesis rather than the threshold, so no setting was
  added: a knob whose only honest default is "no effect" is worse than a note
  saying it was tried. `zone_face_needs_hand` is unchanged and the re-sweep
  that was conditional on this table was not run. Report 17.7.
- **H2: `zone_top_frac` ships at 0.15, not the 0.2 the numbers allow.** Both
  leave hands per frame and zone share identical on all four files and both
  reproduce the check exactly; 0.2 saves 40 percent of the hand pass and 0.15
  saves 22. At 0.2 the topmost window starts at y=512, so no palm in the top 39
  percent of the frame can be found at all, against the top 20 percent at 0.15.
  Nothing on this footage holds anything up to look at it and the collector
  footage this tool is for is people doing exactly that, so the blind region
  was worth more than the extra 18 percent. Report 17.7 has the table.

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

With the handled zone on, from the H1 run of 2026-09-11:

| Measure | `003939` | `004100` | `004310` | `005035` |
|---|---|---|---|---|
| Frames with a zone | 2033/2033 | 982/984 | 938/938 | 3971/3971 |
| Zone share of the frame, mean | 8.98 % | 7.17 % | 10.12 % | 10.68 % |
| ... max | 36.41 % | 34.13 % | 17.66 % | 34.84 % |
| Hands per frame | 1.62 | 1.59 | 2.62 | 1.15 |
| Mask the zone took back | 0.036 % | 0.008 % | 0.015 % | 0.035 % |
| Frames over the zone gate | 0 | 0 | 0 | 0 |
| Worst zone frame | 0.10 % | 0.10 % | 0.50 % | 0.30 % |

Over all four files together, **"whose hands" is wrong 12 percent of the
time**: 20 of the 173 polygons that could be called, out of 179 on 100
uniformly sampled frames. Report 17.7 and
`docs/audits/zone_hands_*.csv`.

The zone takes almost nothing back because the track level hand rules were
already keeping masks off hands. What it changes is that the protection is
structural rather than three thresholds holding, which is what unblocks F2.

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

**About the zone.**

- **One of twenty audited bystander faces is still protected** after the fix,
  and **three of seven of the wearer's own hands are now masked** where the
  zone used to protect them. Both are the same hard case from opposite sides:
  a face and a hand close enough together that no region separates them.
- **"Whose hands" is wrong 12 percent of the time** on a uniform sample of 179
  polygons, 20 of the 173 that could be called. The 74 percent figure from H1
  came from the faces the zone protected, a set selected for being failures.
  Quote the 12. Report 17.7.
- **The 44 of 52 hand rule agreement in 17.4 is not from the copies that
  ship.** On the shipped copies the same comparison reads 13 of 21, because
  tightening the face region masks more of the wearer's hands and fewer
  survive in the copy to be compared at all. Both are in 17.4; do not quote
  one against the other.
- **29 of the 56 faces the zone set aside were never examined.** They are in
  `docs/audits/zone_faces_set_aside_2026-09-11.csv` marked as such.
- **The remembered zone drifts** under camera rotation or zoom, because
  `motion.estimate_shift` measures a translation only. The memory is three
  seconds for that reason and the drift is not separately measured.
- **The check cannot rebuild the remembered zone**, because memory crosses
  frame range boundaries and each range runs in its own worker. Only the live
  half is checked.
- **The zone costs about 45 ms a frame** since H2 cut the window grid from 30
  to 24, down from about 60. It still roughly doubles the detect phase. A
  further cut to 18 windows is measured and available at `zone_top_frac` 0.2,
  and was not taken; see decisions taken by default.
- **Do not re-tile the window grid to make it smaller.** Skip rows and leave
  the rest where they are. Laying a smaller grid over what is left moves every
  window, and what the palm model makes of a crop depends on where the hand
  sits in it, so the zone changes rather than shrinks: the first attempt moved
  hands per frame by 44 percent on one file while saving the same time.

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
.venv\Scripts\python.exe -m eval.zone VIDEO --frames 40         # zone share, hands per frame, cost
.venv\Scripts\python.exe -m eval.zone VIDEO --copy COPY         # masked pixels inside MediaPipe's hands
.venv\Scripts\python.exe -m eval.zone VIDEO --audit DIR         # render frames, write the label CSV
.venv\Scripts\python.exe cli.py footage -o out --no-zone        # measure what the zone costs
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

The handled zone runs on every run and costs about 60 ms a frame, roughly
doubling the detect phase: 966 s against 556 s over the four sample files.
`--no-zone` is for measuring that, never for a copy that ships. `eval.zone
--audit` writes its frames wherever you point it, which must be outside the
repository, and its labels to `docs/audits/`, which is committed.

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
| 2026-09-11 | H1: the handled zone. Three defects found by measurement, including the zone protecting bystanders' faces | 17 |
| 2026-09-11 | H2: the gate threshold fixed, the uniform sample labelled, the orientation rule refuted, the window grid cut | 17.3, 17.7 |

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
| `faceblur/hands.py` | palm detector and landmark model, for the output side check |
| `faceblur/zone.py` | the handled zone: whose hands, how far, how long remembered |
| `eval/zone.py` | zone share, cost, hand damage on a copy, the audit renderer |
| `cli.py`, `ui/app.py` | the two front ends |
| `eval/common.py` | the raw cache and its fingerprint |
| `eval/measure.py`, `eval/sweep.py` | the label free harness and the gates |
| `eval/reid.py` | the recogniser: what the mask does, what is left in the copy |
| `eval/oracle_owl.py` | the object oracle: OWLv2, pinned, never ships |
| `eval/screens.py` | the screen class scored against the oracle |
| `models/README.md` | every model: source, sha256, input layout, output layout |
| `docs/report.md` | every measurement, one section per pass |
| `docs/audits/` | by eye labels, committed. H1's are the first that can be re-scored |
