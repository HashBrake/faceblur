# FaceBlur build plan v2: implementation spec

Handoff document for the next Claude Code session. First written 2026-09-10
from an audit of `FACEBLUR_BUILD_PLAN.md` (v1) against the code. **Rewritten
2026-09-11** after the owner restated the goal as one rule (section 3) and
after work packages R1, S1, 4.3 and E1 landed. Read this file, then
`STATE.md`, then sections 13 to 16 of `docs/report.md`, before writing code.
v1 stays in the repo as history; where the two disagree, this file wins.
Where this file and the code disagree, check the code and say so in
`STATE.md`.

No code was written for this spec. Every design names the file and function
it hooks into so that the next session can check the hook still exists.

## 0. Where things stand and what changed

Done and pushed (commits `def0abd` to `18e6b82`, report sections 16 and 17):
the documents and packaged app caught up with the code (R1); the output side
check reads the copy for screens and can hold a copy back for one (S1); the
audit record splits the masked share by kind (4.3); an open vocabulary oracle
gives screens their first recall number, 0 to 23 percent (E1); the fixes from
the review of that pass (R2); and **the handled zone (H1)**, on every frame
of every sample file, subtracted from every mask, with a precision gate in
the check and the first audit in this project whose labels are committed.
All 1084 tests pass.

**Status on 2026-09-11, after H1.** Building the zone found that "inside the
zone" needs two definitions. What the wearer reaches over is what they hold,
so text and screens there are protected. A face inside somebody's reach is
usually their own face: the audit found the zone protecting bystanders' faces
in 20 of 27 labelled cases, so a face is now protected only where a hand is
on it (`zone_face_needs_hand`, at `zone_face_scale` 1.6 palm boxes, chosen by
sweeping against the labels). That reverses D1 for printed faces on handled
cards. **The owner delegated the call on 2026-09-11 and it stands**: a real
face outranks a printed one, and there is no card footage to lose anything on
until there is. Package H2 measures a "whose hands" rule that may give D1
back for free.

Two findings from that pass change the plan:

- **The four sample files contain no cards.** They are a washroom, a
  corridor, a canteen and a table tennis hall. Nothing about cards can be
  measured until the owner supplies card footage.
- **The owner restated the goal on 2026-09-11** as one rule, in section 3.
  It replaces the by surface text policy, the "cards are left alone" rule and
  the three owner decisions of the first version of this file. It also moves
  the precision constraint from "hands" to "what the wearer is handling",
  which reopens recall options that were rejected because they masked the
  wearer's hand.

The work that remains, in order: zone follow ups (H2), the gate in the
window (G1), recall outside the zone (F2), the second chance for missed faces
(F1), the text detector and policy (T1, T3), screens outside the zone (S2),
defaults (D), and two optional pieces (T4, M1). F0 and T2 are dropped;
section 3 says why. H2 comes before F2 because F2 loosens thresholds outside
the zone, and every error in "whose hands" gets larger when it does.

## 1. Standing decisions

Each one is the owner's stated instruction or a result measured and recorded
in `docs/report.md`.

| Question | Decision | Where recorded |
|---|---|---|
| Where it runs | The user's Windows PC. Video never leaves the machine. No cloud, no telemetry. | v1 section 1 |
| The rule | Keep what the wearer is handling, untouched. Destroy every other informational thing: faces, text, screens. Section 3. | Owner, 2026-09-11 |
| Kinds | Three, each a switch of its own: faces, personal text, screens. Plates and bodies out. Audio and container metadata declined on 2026-09-10. | v1 scope change, `faceblur/classes.py` |
| Default switches | Faces on today. Once text is ready, **face, text and screen are all on by default** (owner, 2026-09-11, D3 below). The handled zone is what makes a wide default safe. | package D |
| Precision against recall | Inside the handled zone precision is absolute: nothing is ever masked there. Outside it, a miss is the leak and over masking of non handled things is the cheaper error, subject to a per frame mask budget so the environment stays usable. This replaces "hands untouched, minimum region everywhere". The first build's failure (33 percent of every frame, 82 percent of hand pixels) was a failure inside the zone. | Owner 2026-09-11; `docs/precision_audit.md` |
| Human steps | None in the pipeline. None in any number that gets re-run. A one off audit by eye is allowed only to check that a rule which sets a detection aside is safe; it is done once, recorded with its counts and the cases it could not call, **its labels are committed beside the report**, and it is never the number quoted. The two earlier audits (sections 14.1, 15.2) did not commit their labels and cannot be re-scored. | `docs/report.md` preamble, section 16.2 |
| What the output is | Pseudonymised personal data, never "anonymous". The audit record and the report are the evidence of "appropriate technical measures". | `STATE.md` |
| Licences | Apache 2.0, MIT, BSD. No AGPL, no GPL, no research only licences in anything that ships. Evaluation only models never enter `build/faceblur.spec`; `tests/test_models.py` enforces it. | v1 section 1 |
| Dependencies | Every one pinned. Real footage, every output folder and `eval/cache/` gitignored. Model files committed with sha256 and licence text. Three environments: `.venv` ships, `.venv-eval` (MediaPipe), `.venv-oracle` (OWLv2). | `requirements*.txt` |
| Working agreement | The session runs autonomously, measures, records in `docs/report.md` and `STATE.md`, commits per package with the trailers, pushes to `HashBrake/faceblur` `main`. Nothing in this file waits on the owner. | Owner instruction 2026-09-04 |
| Disk | The owner frees space themselves. Do not delete `footage_blurred*` folders or environments. | Owner, 2026-09-11 |

## 2. Definitions

**Kind.** One of `face`, `text`, `screen` in `faceblur/classes.py`: name,
label, mask shape (`ELLIPSE` or `POLYGON`), `ready`. `Detection.kind` says
what a box is; `Detection.source` says which detector said so.

**Ready.** A kind may be set `ready=True` only when it has all four: a
detector inside `batch.detect_job`; coverage in `faceblur/verify.py`
(`KINDS_CHECKED`, enforced by `tests/test_classes.py`); precision measured on
the sample files in `docs/report.md`; caveats written. Faces and screens have
all four. Text has none.

**Handled zone.** The region of a frame the wearer is interacting with:
their hands, what is in or touching them, and anything they handled in the
last few seconds that has not moved. Nothing is ever masked inside it. Built
in H1 from the palm and landmark models that already ship.

**Live zone and remembered zone.** The live zone is around hands seen in this
frame. The remembered zone is where handled things were in the last
`zone_memory` seconds, moved with the camera. The live zone protects every
kind. The remembered zone protects text and screens only: a person who walks
into where a card was put down is still a person.

**Gate.** `--quarantine`. A copy the output side check finds a residual in,
or in which the check finds destroyed pixels inside the handled zone, is
moved to `quarantine\` beside the output with its record naming the frames.
The file is never deleted.

**Oracle.** A model used only for evaluation. MediaPipe is the face and hand
oracle (`.venv-eval`); OWLv2 is the object oracle (`.venv-oracle`). Neither
ships; neither output is ever a mask.

## 3. The governing rule

The owner, 2026-09-11: *remove all information not pertinent to the task the
collector is doing. Anything the collector is interacting with or handling
is not blurred at all. Everything else that carries information (text,
screens, faces) is blurred.* Three clarifications, same day:

- **D1. Handled** means objects in or touching the wearer's hands, plus a
  short memory of a few seconds for things put down that have not moved. A
  wall sign the collector reads is masked. Items laid out on the table are
  protected while being worked through.
- **D2. A handled screen is never blurred.** The collector's own phone or
  laptop in hand is part of the task, whatever it shows.
- **D3. Once text is built, face, text and screen are all on by default.**
  The default run is the privacy safe run.

What the rule resolves:

- A printed face on a card in hand is not blurred. The old F0 question is
  answered by construction and F0 is dropped. **Amended after H1:** a face is
  protected only where a hand is on it, so a printed face on a card in the
  reach is masked today. The owner accepted that on 2026-09-11 (delegated
  decision, section 0); `zone_face_needs_hand=False` gives D1 back, and H2
  measures whether a better "whose hands" rule makes the trade unnecessary.
- Text needs no card veto. Text outside the zone is masked whatever surface
  it is on; text inside is left. T2 is dropped. Card footage is still needed
  to measure how much card text sits inside the zone in practice, but the
  rule does not depend on it.
- The wearer's own hand can no longer be masked by any setting, because the
  zone is subtracted from every mask. The track level hand rules and the
  output side hand rule stay as belt and braces, but they are no longer what
  keeps hands safe, and the settings that were rejected for masking hands
  (section 10: lower `conf`, `--engine both`, `--no-verify`, the mirror view
  at 0.3) can be re-measured with the zone on. That is package F2.
- YuNet scoring a wall sign at 0.80 is no longer a false positive to defend
  against: a sign is text, and text outside the zone is to be masked.

What the rule does not decide, and this spec settles by default:

- **Whose hands.** Bystanders have hands. In egocentric footage the wearer's
  hands are the largest, lowest and most persistent; H1 uses size and
  position and measures how often a bystander's hand qualifies.
- **How far the zone reaches past the hand.** A card in hand extends one
  hand's width past the fingers; a phone the same. H1 starts at twice the
  hand rectangle and measures.
- **Bodies and clothing.** Out of scope as before. Text on clothing is text.
- **Large non informational objects outside the zone** (the table tennis
  table) are not to be masked; the mask budget and the screen size cap
  stay. The rule is about information, not about everything.

## 4. Architecture

### 4.1 The pipeline as it stands, with the zone added

```
batch.run_video(src, dst, settings, submit, ...)
  1. detect      detect_job per frame range, in workers
                   faces:   DetectorBank.detect_raw -> filter_candidates / weak_candidates
                   screens: ScreenDetector.detect            (if settings.wants("screen"))
                   hands:   zone.HandFinder.detect  (H1)     (always; the zone is not a kind)
                   text:    TextDetector.detect     (T1)     (if settings.wants("text"))
  2. track       Tracker.run(strong, weak, shifts, shape) -> per_frame, tracks
  2b. screens    screens.hold(found, settings, n)   -> per_frame[i] += screen boxes
  2c. text       text.hold(found, settings, n)      -> per_frame[i] += text quads   (T1)
  2d. zone       zone.build(hands, shifts, settings, n) -> zone_per_frame            (H1)
  3. segment     plan_segments / split_long: copy clean GOPs, encode masked ones
  4. write       encode_job: decode_range -> redact(frame, dets, settings, zone) -> encode
  5. join        concat + verify; fall back to encoding everything
  6. check       verify.check(...) -> residual_* per kind, zone_pixels_changed; held_for
  7. second      F1: seeds from step 6 -> steps 2 to 6 once more
```

`per_frame: list[list[Detection]]` is still the one structure every kind
lands in. The zone is not a `Detection` and not a kind: it is a per frame
list of convex polygons in source pixels, `zone_per_frame: list[list[tuple]]`,
carried beside `per_frame` into `encode_job` and applied inside `redact` as
`alpha *= 1 - zone_alpha`. A frame with an empty zone list costs nothing.

### 4.2 Where each remaining package hooks in

| Package | Detect | Hold / track | Redact | Check | Record | Settings |
|---|---|---|---|---|---|---|
| R2 review fixes | none | none | none | `verify.main` exit code follows `held_for`; face rows carry `w`, `h`, landmarks | none | none |
| H1 handled zone | `detect_job`: `zone.HandFinder` per frame (stride allowed) | `zone.build`: live and remembered polygons, camera shift applied | `redact.redact` takes `zone`; `build_alpha` unchanged; zone alpha subtracted | `residual_job`: rebuild the zone from the source frames, measure changed pixels inside it | `zone_share_mean`, `zone_frames`, `zone_pixels_changed`, `hands_per_frame` | `zone_*` |
| G1 gate in the window | none | none | none | none | none | `ui/app.py` sets `check_output`, `quarantine`; a "checking" stage |
| F2 recall outside the zone | settings only | settings only | none | none | none | new sweep grid in `eval/sweep.py` |
| F1 second chance | none | seeds into a second `Tracker.run` | none | runs once more | `second_chance_*` | `second_chance` |
| T1 text detector | `detect_job`: `TextDetector.detect` | `text.hold` | polygon path exists | none until T3 | `text_lines`, `text_frames` | `text_*` |
| T3 text policy, gate, ready | none | zone applied as for every kind | none | `residual_job`: text on the copy | `residual_text_*` | `text_gate_*` |
| S2 screens outside the zone | settings only | none | none | none | none | `screen_conf` re-measured |
| D defaults | none | none | none | none | none | `Settings.mask` default; `ui/app.py` `_build_mask`; `classes.py` |
| T4 identifiers | none | boxes for regex hits | none | none | `identifier_hits`, never text | `text_identifiers` |
| M1 metadata line | none | none | none | none | `source_audio`, `source_tags_present` | none |

### 4.3 The zone in the gate, and the sweep gates

The output side check gains a second question beside "what is still
visible": **was anything destroyed inside the handled zone?** `residual_job`
rebuilds the zone from the source frames (hands are unmasked in both, and the
source is sharper), compares source and copy inside it with the same encoder
noise floor `frame_noise` uses, and reports `zone_pixels_changed` per frame as
a share of the zone. Anything over `zone_gate_changed` (first guess 0.5
percent of zone pixels moved more than `FLAT_DIFF` past the noise) holds the
copy back, for `zone`. This is the precision gate. It replaces "hand pixels
touched" as the hard number, and it can be run by a third party on any copy.

`eval/sweep.py` `GATES` change with the rule. `hand_damage` stays, measured
with MediaPipe's independent hand regions, and its gate stays at 0.1 percent:
it is now the witness that the zone works rather than the thing that
constrains recall. `off_face_mean` and `off_face_detections_mean` become
`off_face_outside_zone_*`, measured over pixels outside the zone only, and
their gates loosen to 1.5 and 0.6 percent as a first guess that F2 sets by
measurement. A new gate `zone_damage` at 0.0 percent measures masked pixels
inside the pipeline's own zone, which must be zero by construction and is
measured anyway.

### 4.4 Per kind accounting, worker caches, static graphs, cache fingerprints

Unchanged from the first version of this file and now built: `redact.masked_shares`,
`AuditRecord.masked_*_by_kind`; `batch._bank`, `_screens`, `_hand_rule` as
the worker side caches (add `_zone`, `_text` the same way; `verify.residual_job`
imports them from `batch`); every ONNX model at fixed input shapes through
`detect.SessionPool`; every eval cache JSON with a fingerprint of the module
that wrote it. DirectML memory: four workers at about 850 MB of face graphs
each; the hand models add about 15 MB, YOLOX 80 MB, a PP-OCR mobile detector
about 60 MB. Do not raise `GPU_WORKERS` above four; measure the peak with
every kind on and record it.

## 5. Work packages

Each package ends with a verify step that is a command or a test, a report
section (17 onward), a `STATE.md` update and one commit. Order: R2, H1, G1,
F2, F1, T1, T3, S2, D, then T4 and M1 if time allows.

### R2. Fixes from the review of the last pass (1 day)

1. `verify.main` exits 1 on any residual screen box, including a flicker
   under the persistence rule. Exit code follows `held_for(report, settings)`
   and nothing else; the printed summary still lists every find.
2. `verify._row` for faces carries only `x`, `y`, `px`, `score`. F1 needs a
   full `Detection` per find. Add `w`, `h`, `landmarks` (rounded, as
   `Detection.to_dict` writes them) to every row of every kind.
3. `residual_job` runs `screens.detect(after)`, which applies the 12 percent
   size cap on the copy side, so a monitor over the cap the pipeline chose not
   to mask is never reported. Keep the behaviour (it is scope, not a miss) and
   say so in report section 16.4 and the README's screen caveats.
4. Two brittle tests. `test_the_sweep_measures_faces_and_never_sees_a_screen`
   asserts the substring `screens` is absent from the harness source;
   `test_the_oracle_never_runs_in_the_pipeline` greps `faceblur/*.py` for
   `torch`. Both fail on a comment. Parse imports with `ast` and check module
   names (`faceblur.screens`, `torch`, `transformers`) instead.
5. Report section 9 says 999 tests; the count is 1037. `STATE.md` "If the
   scope grows" still says screens are a week of work. Consolidate
   `STATE.md`: the S1 findings appear four times.
6. The README's screen caveats say in one sentence what sections 16.1 and
   16.2 mean together: the screen class misses most screens and holds most
   copies back for the ones it finds, so it is fit for best effort masking
   with the gate off, not for unattended use, until S2.
7. `eval/oracle_owl.run` verifies the sha256 of `model.safetensors` against
   the value in `models/README.md` before using it.

**Verify.** `pytest tests`; `python -m faceblur.verify` on an R1 copy with
`--mask face,screen` exits 0 when `held_for` is empty.

### H1. The handled zone (done 2026-09-11, report section 17)

Built as specified below with three deviations, each measured and recorded:
a face is protected only where a hand is on it (`zone_face_needs_hand`,
`zone_face_scale` 1.6) rather than inside the grown reach; the precision gate
tests the hand region, not the reach, and only the live half of the zone; and
`zone_stride` stays at 1 because 2 halved the hand evidence for 18 percent of
the detect phase. The design text is kept as the record of what was asked.

**Goal.** A per frame region nothing is ever masked in, built from the
wearer's hands, measured on the four files, applied to every kind, and
checked on the copy.

**Detector.** `faceblur/zone.py`, `HandFinder`, from `hands.PalmDetector`
and `hands.HandLandmarks` which already ship. The palm model takes 192 px;
the wearer's hands on this camera are 150 to 500 px, so run it over sliding
windows of `zone_window` (512) with half overlap across the frame, pool the
hands, suppress duplicates across windows at IoU 0.5 on the palm boxes, and
confirm each with the landmark model at `hand_presence` (0.7, the plateau
section 14.2 measured). Cost: about 12 palm inferences and a few landmark
inferences per frame; measure, and allow `zone_stride` (default 1, try 2)
because the zone has memory.

**Whose hands.** A hand qualifies as the wearer's when its palm box long side
is at least `zone_min_hand_px` (120) **or** its rectangle touches the lower
`zone_edge_frac` (0.25) of the frame. First guess. Measure on the four files
how many qualifying hands per frame there are and, with a by eye audit of
up to 100 sampled frames whose labels are committed to
`docs/audits/zone_hands_2026-09.csv`, how many are a bystander's.

**Geometry.** For each qualifying hand, the live polygon is the landmark
model's rotated rectangle (`Hand.quad`, already 2.6 times the palm) grown
about its centre by `zone_scale` (first guess 1.5, so about four palm widths
across), clipped to the frame. That is what covers a card, a phone or a bat
in the hand. Union of polygons per frame.

**Memory.** Every live polygon is remembered for `zone_memory` seconds
(3.0). A remembered polygon is moved each frame by that frame's camera shift
(`shifts` from `motion.estimate_shift`, already computed in `detect_job`) so
it stays on the object rather than on the pixels, and dropped when it leaves
the frame or expires. The remembered zone protects text and screens only, per
section 2. A live hand refreshes any remembered polygon it overlaps.

**Applying it.** `redact.redact(frame, dets, settings, zone=None)`: after
`build_alpha`, draw the zone polygons into a second canvas with the same
feather, and `alpha *= 1 - zone_alpha` for the kinds the zone protects on
this frame (all kinds for live polygons, text and screens for remembered
ones; so two canvases, or one per protection class). Nothing upstream
changes: detectors, tracker and holds run as now, and the zone is applied at
the last step so that the audit can report both what would have been masked
and what was. `masked_shares` measures after the zone is applied.

**Record.** `zone_share_mean`, `zone_share_max`, `zone_frames` (frames with a
non empty zone), `hands_per_frame_mean`, `masked_in_zone_prevented` (share of
the frame the zone removed from masks, mean), and in the check
`zone_pixels_changed_max`, `zone_frames_over`, and `zone` in `held_back_for`.

**Settings.** `zone: bool = True`, `zone_window`, `zone_stride`,
`zone_min_hand_px`, `zone_edge_frac`, `zone_scale`, `zone_memory`,
`zone_gate_changed`, `zone_device` (like `hand_device`). `--no-zone` on the
command line, off only for measurement. Every one validated, in `to_dict`,
with a comment naming the report section that measured it.

**Measure**, report section 17:

1. Zone share of the frame per file, mean and p95, and hands per frame.
2. The 20 hand finds of section 14.1 (by frame number, from the record of
   the 2026-09-09 run) fall inside the zone; the 52 real faces fall outside.
   These are the only labels that exist for this, and only their frame
   numbers survive; use them as a smoke test and say so.
3. `hand_damage` from the sweep with the zone on: expect 0.00 on every file.
4. Cost per frame with `zone_stride` 1 and 2.
5. A by eye audit of zone polygons on 100 sampled frames, labels committed:
   is the zone on the wearer's hands and what they hold, does it swallow a
   bystander, does the remembered zone drift off its object under camera
   motion.

**Tests.** Polygon union and clipping; memory expiry and camera shift
applied; a live hand refreshes memory; a face inside a live polygon is not
masked and inside a remembered one is; a screen inside a remembered polygon
is not masked; `zone=False` reproduces the current alpha exactly; the check
reports changed pixels inside a zone on a copy made with `--no-zone` and
none on a copy made with it.

### H2. Zone follow ups (2 days)

Four things the H1 review found, in order. The first is a shipping default
that holds back a clean copy and is not optional.

1. **Set `zone_gate_changed` from the distribution.** The shipped 0.5 percent
   sits on a real reading: `004310` reads 0.501 on one frame and is
   quarantined for `zone` by a thousandth, while the other files read 0.10 to
   0.30 and a copy made with `--no-zone` reads 17.7. Set it at 1.0 percent,
   re-run `004310` with `--check-output`, and record the four readings and
   the margin in section 17.3. A gate that fires on a good copy is a defect,
   not a caveat.
2. **Measure the wrist to knuckle direction as the "whose hands" rule.**
   `hands.rect_for` already computes the angle that puts the wrist to middle
   knuckle line upright. The wearer's hands point up the frame, away from the
   chest camera; a bystander facing the wearer has hands pointing down or
   toward it. Add `zone_orientation` (degrees from straight up within which a
   hand may be the wearer's; first guess 100) as a third condition beside
   size and edge, score it against the 27 labelled cases in
   `docs/audits/zone_faces_set_aside_2026-09-11.csv` and the labels from item
   3, and report the table. If it separates the 20 bystander cases from the
   7 wearer hands on its own, re-sweep `zone_face_needs_hand=False` with it
   on: that would give D1 back and stop masking the three wearer hands the
   1.6 palm region costs. If it does not, keep the current rule and say so.
   Do not flip `zone_face_needs_hand` without that table.
3. **Label the uniform sample.** `docs/audits/zone_hands_*.csv` holds 179
   rendered rows and no verdicts. Fill `verdict` (`wearer`, `bystander`,
   `not a hand`, `cannot tell`) and `note` for all 179 and commit. That is
   the only measurement of how often "whose hands" is wrong on frames nobody
   pre-selected, and 17.6 admits it does not exist. Then correct the sentence
   in 17.5 that says these labels were committed.
4. **Cut the window grid.** 30 windows of 512 px at half overlap cover the
   whole frame; the wearer's hands never reach the top third. Measure the
   zone on the four files with the top row of windows dropped
   (`zone_top_frac`, first guess 0.3): hands per frame, zone share, the 44 of
   52 hand rule agreement of 17.4, and ms per frame. Ship it only if the
   first three do not move.

**Verify.** `pytest tests`; the four files with `--check-output` held back
for `zone` on none of them; section 17 amended, not appended; `STATE.md`
"decisions taken by default" updated with the outcome of item 2 either way.

### R3. After H2 (1 hour)

H2 is done (report 17.3 and 17.7): the gate is set from the distribution at
1.0 percent, the 179 uniform audit rows are labelled (12 percent "whose
hands" error), the orientation rule was measured and refuted on this
footage, the window grid is cut to 24 with `zone_top_frac` 0.15. Two things
the review of it found:

1. The orientation table was produced by code that was not committed. Add
   `eval/zone.py --orientation`: read `docs/audits/zone_hands_*.csv`, recover
   the wrist to knuckle angle from each committed quad, print the two tables
   in 17.7, and cite the command there. Every number in the report has to
   come from a command a third party can re-run; H2 reaffirmed that rule in
   the same section it broke it.
2. The 20 bystander rows are all side on to the wearer (a counter worker
   reaching in, a man bending over, a man at a sink). The hypothesis was
   about a bystander facing the wearer, and none is in the sample. Narrow
   the sentence in 17.7 to what was measured and put the face to face case
   on the "re-measure when card footage arrives" list in `STATE.md`, beside
   D1 and `zone_top_frac`.

**Verify.** The command prints the tables in 17.7; `pytest tests`.

### G1. The gate in the window (1 to 2 days)

`ui/app.py` never sets `check_output` or `quarantine`, so no run started
from the window is checked or held back, and the window is the workflow v1
was built around. Add under "Advanced settings" a checkbox "Check each copy
and hold back any that still shows something" (on by default; the cost is a
second detection pass and the record says so), a "Checking" status word and
progress stage wired to the `checking` reports `run_video` already emits, and
a summary line that names how many copies were held back and for which
kinds, from `held_back_for`. Strings in `ui/strings.py` under the wording
rules; `tests/test_ui.py` and `tests/manual_ui.md` updated.

### F2. Recall outside the zone (3 days)

**Goal.** Re-measure the settings section 10 rejected for masking the
wearer's hand, now that the zone protects it, and move the defaults if the
numbers say to.

**Two prerequisites in the harness, found by H1 (report 17.4).**
`eval/measure.evaluate` builds masks from the raw cache and the tracker and
does not apply the zone, so a sweep today scores masks the pipeline would
not apply; build the zone from the cached hands (add them to the raw cache
with their own fingerprint) and subtract it exactly as `redact` does before
any number is read. And `hand_damage` now measures two things: MediaPipe's
hulls are every hand in the frame, the rule protects only the wearer's, so a
mask on a bystander's hand is permitted. Split it into `hand_damage_wearer`
(hulls that intersect the zone; gate 0.1 percent as before) and
`hand_damage_other` (reported, no gate) before leaning on it.

**Method.** `eval/sweep.py` gains a grid over: `conf` 0.4 to 0.6, `engine`
`yunet` and `both`, `verify` on and off, `confirm_tta` 2 with
`verify_conf_sure` 0.3, `third_conf` down to 0.1, `det_sizes` with 2560
added, with the zone on. The gates of section 4.3 apply. The score stays
"both kinds of recall, equal weight". Also run `eval/reid` on the winner.

**Expected shape.** Recall on pasted faces rises from 79 to 88 percent
toward the low 90s; off face masking outside the zone rises; hand damage
stays at zero because the zone is subtracted. If off face masking outside
the zone exceeds its gate, the answer is no, and the report says which
setting cost what.

**Do not** change `hand_presence`, `screen_min_run` or `strength`; they are
not what this package is about.

### F1. Second chance for missed faces (5 days)

Unchanged in design from the first version of this file: the output side
check's confirmed, hand filtered `missed` face rows become seeds for a second
`Tracker.run`, the file is written again, checked again, and the record
carries `residual_faces_first_pass`, `second_chance_seeds`,
`second_chance_tracks_added`, `second_chance_seconds`. No threshold lowered.
At most `second_chance` (1) extra rounds. Seeds require `check_stride == 1`
(refused in `validate` otherwise), inherit `kind="face"` and `source="copy"`,
and are dropped if they overlap any hand quad the rule returned or lie
inside the live zone. Write to `part_path(dst)` and `os.replace` after the
join verifies; test the crash path. Measure residual faces before and after,
`hand_damage`, `zone_damage`, off face outside the zone, and `eval/reid`
leaks; report section 18. Depends on R2 item 2.

### T1. Text detector (3 days)

**Model.** PP-OCRv4 mobile detection (PaddleOCR, Apache 2.0), converted from
PaddleOCR's own release with `paddle2onnx` in a throwaway environment; record
source URL, source sha256, converter version and output sha256 in
`models/README.md`; commit the ONNX and licence. DBNet from MMOCR (Apache
2.0) is the fallback, CRAFT (MIT) the third. No pre converted file from an
unknown repository.

**Inference.** Fixed shapes through `SessionPool`, two long sides starting at
960 and 1920, letterboxed with the model's own mean and std (check against a
frame by eye and record which normalisation lights the map up). Probability
map, `text_thresh` 0.3, contours, `cv2.minAreaRect`, unclip `text_unclip`
1.6, keep `text_box_thresh` 0.6. **Four point convex quads only**:
`build_alpha` uses `fillConvexPoly`. Floor at `text_min_px` (12, the short
side); a quad over `text_max_frac` (0.3) of the frame is a page, kept and
counted as `text_pages`. Merge scales by union then NMS on bounding boxes at
0.5.

**Grouping.** `text.hold` on the pattern of `screens.hold`: `text_min_run` 2,
`text_gap` 6, `text_tail` 3, linking by bounding box IoU at `text_link_iou`
0.3.

**Measure, before any policy**, report section 19.1: lines per frame, share
of the frame, cost at one and two scales, and the surface each line sits on
per the OWLv2 oracle (screen, document, badge, sign, none), plus the share
of lines inside the handled zone. The canteen's signage and packaging and
the shirt with text on it in the washroom file are the cases to look at.

**Not in T1.** Masking, the gate, OCR. `TEXT.ready` stays `False`; text runs
under `eval/text.py` only.

### T3. Text policy, gate, and ready (3 days)

**Policy.** Text outside the handled zone is masked, whatever the surface.
Text inside the live or remembered zone is left, by the same subtraction
every kind gets. Below `text_min_px` nothing is masked because nothing is
found. No card veto, no surface classes, no OCR.

**Gate.** `residual_job` runs the text detector on the copy, floors, applies
the zone, `applied` through `region_for`, `classify` as before, persistence
`text_gate_min_run` 2 in checked frames, `text_gate_min_px` 16.
`held_for` adds `text`. `KINDS_CHECKED` gains `text`.

**The other half of the promise.** The precision gate from H1 tests only the
hand region and only the live zone (report 17.6), so nothing yet checks that
text and screens inside the reach or the remembered zone were left alone.
Before `TEXT.ready` is set, the check must rebuild the reach on the source
frames (live polygons at `zone_scale`, which needs no memory) and report
text and screen pixels destroyed inside it as `reach_pixels_changed`, gated
like `zone_gate_changed`. The remembered half stays unchecked and section 19
says so.

**Ready.** `TEXT.ready = True`;
`tests/test_classes.py::test_which_kinds_this_build_can_actually_do` changes
with it. `build/faceblur.spec` and `tests/test_models.py` take the model.
The window's help text drops its "Not in this build yet" prefix on its own.

**Measure**, report sections 19.2 to 19.4: precision by surface against the
oracle, text share of the frame per file, cost, share of the frame masked in
total with all three kinds on, the caveats (no recall below the floor, one
venue, OCR not used and why, no card footage so the inside zone share of card
text is unmeasured).

### S2. Screens outside the zone (2 days)

Section 16.2: the shipped screen class recalls 0 to 23 percent of what the
oracle sees, the cap and persistence cost almost none of it, and on the
table tennis hall YOLOX scores something at 91 percent of the misses, almost
all under `screen_conf` 0.5. Under the rule, a missed screen outside the zone
is a leak and a masked table is a cost to the environment, not to privacy.
Re-run `eval/screens.py` at `screen_conf` 0.5, 0.4, 0.3, 0.25 with the zone
on, reporting recall and precision against the oracle, the masked share of
the frame and the frames over budget per file. Move the default only if
precision stays over 50 percent against the oracle and the union masked
share stays under 2 percent mean on every file; otherwise leave it and say
what it would cost. Handled screens are protected by the zone whatever the
floor.

### D. Defaults (half a day, after T3)

`Settings.mask` default becomes `("face", "text", "screen")`; `ui/app.py`
`_build_mask` ticks every ready kind; `classes.py` keeps listing order;
`README.md` and the CLI help say the default run masks everything outside
the handled zone and how to narrow it. `tests/test_classes.py::test_the_default_run_masks_faces_and_nothing_else`
changes to say the new default, on purpose. This package is the owner's D3
and needs no further instruction.

### T4. Identifiers by pattern (3 days, optional)

PP-OCRv4 recognition (Apache 2.0) on text quads over `text_ocr_min_px` (20)
outside the zone; regex for phone, email and postal shapes; masked even
where the detector's persistence rule would have waited. **The recognised
string is never written anywhere**; a test feeds a known string through a
synthetic frame and greps every file written. Counts by type only.

### M1. Metadata line (half a day)

`source_audio: bool` and `source_tags_present: list[str]` (names only, never
values) in the record from `video.probe`.

## 6. Cross cutting rules

- **Settings.** Prefix per kind or feature (`zone_`, `text_`, `screen_`),
  gates `*_gate_*`, validated with a message naming the field and bound,
  listed in `to_dict`, with a comment saying why the default is what it is
  and which report section measured it, or "first guess, section N will
  measure it".
- **Audit record.** New fields default to zero or empty so an old record
  reads back. Set aside items stay in a list marked with the reason.
- **Models.** File, licence text, `models/README.md` entry (URL, size,
  sha256, input layout, normalisation, output layout), `EXPECTED` in
  `tests/test_models.py`, spec `datas`. Decode checked against a real frame by
  eye and against a hand built output in a test.
- **Eval caches and audits.** JSON, fingerprinted, boxes only, under
  `eval/cache/`. By eye labels committed under `docs/audits/` as CSV of frame
  numbers and boxes, never pixels. Frames rendered for an audit go outside
  the repo or under a gitignored folder and are deleted after.
- **Tests.** Named as sentences saying what would go wrong. Import checks by
  `ast`, not by substring. Update the count in report section 9.
- **Docs.** Sentence case headings, active voice, no em dashes or en dashes,
  one word per thing.
- **Commits.** One per package, pushed to `HashBrake/faceblur` `main`, with
  the trailers the session is given.
- **Handover.** `STATE.md` at the end of every package: what changed,
  numbers, what a next person should not assume. Consolidate rather than
  append; a 900 line handover is not read.

## 7. Security and edge case risks

**Leaks**

1. **The zone swallows a bystander.** A bystander's hand qualifies as the
   wearer's, or the grown polygon reaches a bystander's face or badge.
   Mitigation: size and edge rules; the remembered zone protects text and
   screens only; `zone_scale` measured; audit with committed labels. Owner:
   H1.
2. **The remembered zone drifts off its object** under camera rotation or
   zoom, which `estimate_shift` (translation only) does not model, and
   uncovers or protects the wrong pixels. Mitigation: short memory; drop on
   leaving the frame; measured in the H1 audit; `zone_memory` is a setting.
3. **The zone protects a screen someone else is holding up to the wearer**
   (a phone shown across a table). By the rule that is handled; by intent it
   may not be. Report the case; no mitigation without a decision.
4. **Hands not found, zone empty, hand masked.** The old hand rules stay in
   the tracker and the check as belt and braces; `hand_damage` against the
   MediaPipe oracle stays a sweep gate; `zone_pixels_changed` in the check
   catches a masked hand on the copy.
5. **F2 loosens a threshold and something outside the zone that is not
   informational is masked** (the table, a wall). Mitigation: mask budget,
   `off_face_outside_zone` gates, the screen size cap; the report says what
   each setting cost.
6. **OCR fails silently.** No OCR in the primary mechanism; T4 additive.
7. **A masked region that is not destroyed.** Block count sized by the short
   side for quads; a test shows a 400 by 20 px line is unreadable at the
   shipped strength.
8. **Frame misalignment between source and copy.** The join verifies frame
   count and timestamps; `plan_jobs` uses the shorter of the two; a copy one
   frame short must fail loudly.
9. **The gate quarantines everything for a flicker.** Persistence in the
   gate for screens (done) and text (T3), measured before the condition is
   switched on.
10. **Second chance reintroduces a hand or the 30 "neither" finds.** Seeds
    pass the hand rule, the hand quad guard and the live zone; tracker rules
    unchanged; measured.

**Supply chain and data handling**

11. **Model files** pinned by sha256 in a test, licence file, recorded
    source, no runtime download. The oracle's safetensors hash is checked at
    load (R2 item 7).
12. **Licences.** PP-OCR, DBNet, OWLv2, YOLOX, MediaPipe are Apache 2.0;
    CRAFT MIT. YOLO-World, Ultralytics and most YOLOv5 to v8 derivatives are
    GPL or AGPL and must not be used even in evaluation code. Read the
    licence file in the release.
13. **Pickle.** Existing raw caches only; every new cache is JSON.
14. **Recognised text, tag values, pixels** never reach the record, a log or
    a cache. The record is shipped beside the copy and read by strangers.
15. **Quarantine keeps the only copy.** Second chance replaces atomically;
    a crash leaves the first pass copy and its record.
16. **GPU memory.** Four workers; measure the peak with every kind on; the
    text detector's second scale falls back to the CPU (`text_device`) if
    DirectML pages.

**Edge cases in the footage**

17. Non convex text contours: `minAreaRect` first; test a U shape.
18. Text at the frame edge and zero area quads: clip, drop before
    `region_for`.
19. A page filling the frame: masked whole, counted as `text_pages`.
20. Both hands out of frame while the wearer reads a document on the table:
    the document is protected by memory for `zone_memory` seconds and then
    masked. That is the rule as the owner stated it; report how often it
    happens.
21. Reflections: a phone screen in a mirror is outside the zone and masked
    if found; the washroom mirror face is a face. Report, no change.
22. Faces on screens: masked by both classes; union alpha handles overlap.
23. Vertical and rotated text: the quad's short side is the cap height; test
    a 90 degree line.
24. The wearer's own badge or clothing text: outside the zone unless a hand
    is on it; masked. A test frame with a badge at 30 px is masked.
25. Binder pages and stacks of cards in hand: inside the zone; the zone is
    the union of polygons, so cost is linear in hands, not in cards.

## 8. Measurement plan

| Number | Method | Section |
|---|---|---|
| Zone share, hands per frame, cost; the 20 hands inside and 52 faces outside; hand damage with the zone on; audit with committed labels | H1 | 17 |
| Recall and off face masking outside the zone across the re-opened settings; reid leaks on the winner | F2 | 17.5 |
| Residual faces before and after second chance; hand and zone damage; files still held back | F1 | 18 |
| Text lines per frame, share, cost, surface distribution, share inside the zone | T1 | 19.1 |
| Text precision by surface; total masked share with three kinds on | T3 | 19.2 to 19.4 |
| Screen recall and precision against the oracle across `screen_conf`, masked share, over budget | S2 | 16.5 |
| Identifier hits by type | T4 | 19.5 |
| GPU peak with every kind on | 4.4 | 4.1 update |

Every number is produced by a command under `eval/` or `faceblur.verify`
that a third party can re-run. By eye audits are recorded as such, their
labels committed, and are not the number quoted.

## 9. Sequence and estimates

| Order | Package | Estimate | Depends on |
|---|---|---|---|
| done | R2 review fixes | | |
| done | H1 handled zone | | |
| done | H2 zone follow ups | | |
| 1 | R3 after H2 | 1 hour | nothing |
| 2 | G1 gate in the window | 1 to 2 days | nothing |
| 3 | F2 recall outside the zone | 4 days, with its two harness prerequisites | nothing |
| 4 | F1 second chance | 5 days | nothing |
| 5 | T1 text detector | 3 days | nothing |
| 6 | T3 text policy, gate, ready | 4 days, with the reach gate | T1 |
| 7 | S2 screens outside the zone | 2 days | nothing |
| 8 | D defaults | half a day | T3 |
| 9 | T4 identifiers | 3 days | T3, optional |
| 10 | M1 metadata line | half a day | nothing |

About four weeks from here; three without T4. The order is fixed. A package
whose verify step cannot be made to pass after a real attempt is recorded in
`STATE.md` as blocked, with what was tried, and the next package that does
not depend on it starts; nothing waits on the owner.

## 10. What not to do

- Do not lower a face threshold or add a detector view **before H1 lands**:
  until the zone exists, section 10's finding stands and those settings mask
  the wearer's hand. After H1 they are F2, measured.
- Do not add a 2560 scan for faces outside F2; it was measured and rejected
  once (section 12.4) and only the zone changes the calculation.
- Do not change `faceblur/detect.py` casually: every edit, comments included,
  invalidates the raw eval caches by design.
- Do not move `hand_presence` off 0.7 without re-running the grid in
  section 14.2; H1 reuses that plateau.
- Do not tighten `screen_min_run` to 4.
- Do not ship a mode that only Gaussian blurs.
- Do not delete `footage_blurred*` folders or an environment; disk is the
  owner's.
- Do not build F0 or T2; they are dropped by the rule, not deferred.
- Do not produce a by eye audit without committing its labels, and do not
  write that labels are committed when the CSV holds none.
- Do not flip `zone_face_needs_hand` without the H2 item 2 table. It is the
  difference between a stranger's face left in a copy and a printed face on
  a card being masked, and only the labels can say which rule pays.
- Do not trust `hand_damage` as a single number after H1; it counts
  bystanders' hands, which the rule allows to be masked. Split it (F2).

## 11. Open questions for the owner

- **Card footage.** The rule no longer depends on cards, but no number about
  card text or printed faces can be measured until there is footage with
  cards in it. One file of a collector at work would do. It is also the only
  way to find out what the D1 reversal after H1 actually costs.
- **A screen someone else holds up to the wearer** (risk 3): handled, or
  not? The default protects it.
- **A second venue.** Every screen and text setting will be tuned on the
  table tennis hall and the canteen otherwise.
