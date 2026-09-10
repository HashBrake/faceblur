# FaceBlur build plan v2: implementation spec

Handoff document for the next Claude Code session. Written 2026-09-10 after an
audit of `FACEBLUR_BUILD_PLAN.md` (v1) against what was actually built. Read
this file, then `STATE.md`, then sections 13 to 15 of `docs/report.md`, before
writing code. v1 stays in the repo as history; where the two disagree, this
file wins.

No code was written for this spec. Every design below names the file and
function it hooks into so that the next session can check the hook still
exists before building on it.

## 0. What this spec is for

v1 describes a faces only build that masks rectangles at conf 0.25 and never
misses a face. The project that exists masks an eye line ellipse only when
three detector families agree, keeps hands at 0.00 percent, checks its own
output, holds a copy back when the check finds a face, and has a second kind
(screens) built this morning. v1 says "do not reopen" decisions that were
reopened by measurement in the first week. This spec records the decisions as
they now stand and specifies the work that remains: screens finished, faces
made fit for unattended use, and personal text built.

## 1. Standing decisions

These are current. Each one is either the owner's stated instruction or a
result that was measured and recorded in `docs/report.md`.

| Question | Decision | Where recorded |
|---|---|---|
| Where it runs | The user's Windows PC. Video never leaves the machine. No cloud, no telemetry. | v1 section 1 |
| What is masked | Three kinds, each a switch of its own: faces, personal text, screens. Plates and bodies out. Audio and container metadata declined on 2026-09-10. | v1 scope change, `faceblur/classes.py` |
| Default switches | Faces on. Nothing else on by default, ever, without the owner saying so on the day. | `STATE.md`, `ui/app.py` `_build_mask` |
| Precision against recall | Precision constraints are hard: hands untouched (sweep gate 0.1 percent), off face masking under 0.7 percent of the frame, mask budget 5 percent per frame. Recall is maximised inside them. v1's "never miss a face, accept over blurring" is withdrawn: the build that followed it destroyed 33 percent of every frame and 82 percent of hand pixels and was unusable. | `docs/precision_audit.md`, `eval/sweep.py` `GATES` |
| Cards are left alone | Card names, prices, grading labels and everything printed on a card or slab are the dataset. Only personal text is masked. | v1 scope change |
| Human steps | None in the pipeline. None in any number that gets re-run. A one off audit by eye is allowed only to check that a rule which sets a detection aside is safe, is done once, is recorded with its counts and the cases it could not call, and is never the number quoted. Done twice so far: 108 boxes (section 14.1), 85 screen runs (section 15.2). | `docs/report.md` preamble |
| What the output is | Pseudonymised personal data, never "anonymous". Clothing, gait, interiors, timestamps remain. The audit record and the report are the evidence of "appropriate technical measures". | `STATE.md` |
| Licences | Apache 2.0, MIT, BSD. No AGPL, no GPL, no research only licences in anything that ships. Evaluation only models may be looser but never enter `build/faceblur.spec`. | v1 section 1, `tests/test_models.py` |
| Dependencies | Every one pinned. Real footage and every output folder gitignored. Model files committed with sha256 and licence text. | `requirements.txt`, `.gitignore`, `models/README.md` |
| Working agreement | The session runs autonomously, makes judgement calls, measures them, records them in `docs/report.md` and `STATE.md`, and commits per work package with the required trailers. The owner's calls are the three decisions in section 3 and nothing else; take the proposed default, make it a setting, record it, carry on. | Owner instruction 2026-09-04 |

## 2. Definitions

**Kind.** One of `face`, `text`, `screen` in `faceblur/classes.py`. A kind
carries a name, a label, a mask shape (`ELLIPSE` or `POLYGON`) and `ready`.
`Detection.kind` says what a box is; `Detection.source` says which detector
said so. `redact.region_for` chooses the shape from the kind.

**Ready.** A kind may only be set `ready=True` when it has all four:

1. A detector that produces `Detection` rows with that kind, run inside
   `batch.detect_job`.
2. Coverage in `faceblur/verify.py`, so that a miss of that kind found in the
   finished copy is counted and can hold the copy back under `--quarantine`.
3. Precision measured on the four sample files and written into
   `docs/report.md`, with the by eye audit if one was needed.
4. Caveats written: what is not measured, what venue the settings were tuned
   on, what the class does not do.

Screens have 1, 3 and 4. Work package S1 gives them 2. Text has none yet.

**Surface.** For text, where the text sits: a card or slab, a screen, a
document or label or badge, a wall or sign, handwriting on anything. The text
policy in section 3 is written by surface, not by content, because OCR on
this footage fails silently below about 20 px and a failed read is a leak
nobody sees.

**Gate.** `--quarantine`. A copy the output side check finds a residual in is
moved to `quarantine\` beside the output with its record naming the frames.
The file is never deleted.

**Oracle.** A model used only in `.venv-eval` to produce pseudo labels for a
measurement. It never ships and its output is never a mask. MediaPipe is the
face oracle today. Work package E1 adds one for objects.

## 3. Decisions for the owner, with the default to proceed on

The next session does not wait on these. It builds the default, makes it a
setting, records the choice in `STATE.md` under a heading "Decisions taken by
default, 2026-09-xx", and lists them at the top of its handover so the owner
can reverse any of them with one setting.

**D1. Faces printed on cards.** Sports cards carry a player's face. A card
held at arm's length on this camera is about 120 px wide, so a printed face is
30 to 50 px, inside the range the detectors cover about 80 percent of the
time, and CenterFace confirms printed faces (section 13.4 lists "a picture on
a wall" among its finds). Nothing today tells a printed face from a live one.
Proposed default: **measure first (F0), then leave printed faces on cards
alone** with a card region veto, because the owner's rule is that cards are
the dataset and a mass produced photo of a public figure on a card is not the
collector's identity. The setting is `face_on_card` with values `mask` and
`leave`; default `leave` once F0 has a number, `mask` until then.

**D2. Text policy by surface.** Proposed default when `--mask text` is on:

- Masked: every text line the detector finds, wherever it is, **except** on a
  card or slab. That includes text on screens (a readable message is personal
  text whether or not the screen switch is on), documents, labels, badges,
  packaging, whiteboards, walls and signs, and all handwriting.
- Left: text whose quad lies inside a card or slab region (T2), and text
  below `text_min_px` (default 12 px cap height), which is below what any
  reader could use and below where the detector is reliable.
- Over masking of shop signs and product packaging is accepted and measured.
  A signage exemption is not built unless the number in T1 says the cost is
  large; it would be a distilled surface class in T2, not OCR.

The setting is `text_leave_on` (tuple, default `("card", "slab")`) so the
list can grow to `signage` without code.

**D3. Screens by default.** Proposed: stay off. A kind goes on by default
only by the owner's instruction on the day.

## 4. Architecture

### 4.1 The pipeline as it stands

```
batch.run_video(src, dst, settings, submit, ...)
  1. detect      detect_job per frame range, in workers
                   faces:   DetectorBank.detect_raw -> filter_candidates / weak_candidates
                   screens: ScreenDetector.detect          (if settings.wants("screen"))
  2. track       Tracker.run(strong, weak, shifts, shape) -> per_frame, tracks
  2b. screens    screens.hold(found, settings, n) -> per_frame[i] += screen boxes
  3. segment     plan_segments / split_long: copy clean GOPs, encode masked ones
  4. write       encode_job: decode_range -> redact -> SegmentEncoder; copy_job
  5. join        concat + verify; fall back to encoding everything
  6. check       verify.check(src, dst, ...) -> residual_*; quarantine_output
```

`per_frame: list[list[Detection]]` is the one structure every kind lands in.
Anything that masks must put `Detection` rows with the right `kind` into it
before step 3. Anything that measures the copy must run inside
`verify.residual_job`, which already decodes source and copy in lock step.

### 4.2 Where each work package hooks in

| Package | Detect | Track / hold | Redact | Check | Record | Settings |
|---|---|---|---|---|---|---|
| S1 screens in the gate | none | none | none | `residual_job`: screen detector on the copy, `applied()` generalised to `region_for` | `residual_screens`, `residual_screen_max_frac`, `residual_screen_runs` | `screen_gate_min_px`, `screen_gate_min_run` |
| E1 object oracle | none (eval only) | none | none | none | none | none in `Settings`; `eval/oracle_owl.py` has its own |
| F0 faces on cards | none | `Tracker.run` output filtered by card regions before step 3 | none | card regions also veto residual faces so the gate does not hold a copy back for a card | `faces_on_cards_left` | `face_on_card` |
| F1 second chance | none | copy side finds seeded into a second `Tracker.run` | none | runs once more after re-write | `second_chance_seeds`, `residual_faces_first_pass` | `second_chance`, `second_chance_min_run` |
| T1 text detector | `detect_job`: `TextDetector.detect` | `text.hold` like `screens.hold` | polygon path, already there | none until T3 | `text_lines`, `text_frames` | `text_*` |
| T2 card veto | `detect_job`: `CardDetector.detect` or geometry | veto applied in `text.hold` and in F0 | none | veto applied to residual text | `cards_seen`, `text_left_on_cards` | `text_leave_on`, `card_*` |
| T3 text in the gate | none | none | none | `residual_job`: text detector on the copy | `residual_text`, `residual_text_max_px` | `text_gate_min_px` |
| T4 identifiers | none | adds boxes for regex hits on non veto surfaces | none | none | `identifier_hits` by type, never the text | `text_identifiers` |
| M1 metadata line | none | none | none | none | `source_audio`, `source_tags_present` | none |

### 4.3 Per kind mask accounting

`encode_job` returns one `masked` share per frame from the union alpha. Once
two kinds mask at once, the face gates in `eval/sweep.py` and the
`frames_over_budget` count in the record are diluted by screen and text
masks, which are large by design. Change `redact.redact` to return, beside
the union alpha, the share of the frame each kind's regions cover (compute
`build_alpha` per kind on the same canvas size; it is cheap next to the
blend), and carry `masked_by_kind: dict[str, list[float]]` through
`encode_job` into `AuditRecord.masked_mean_by_kind`, `masked_max_by_kind`,
`frames_over_budget_by_kind`. The existing single numbers stay and keep their
meaning as the union. The sweep gates read the face share only.

### 4.4 Worker side model caches

`batch._bank`, `batch._screens`, `batch._hand_rule` each hold one model set
per worker keyed by the settings that shape it. Add `_text` and `_cards` the
same way. `verify.residual_job` imports these caches from `batch`, so the
check reuses the worker's models rather than loading its own; keep that.

DirectML memory: four workers hold about 850 MB of face graphs each on an
8 GB card. YOLOX tiny adds about 80 MB per worker; a PP-OCR mobile text
detector at two fixed sizes about 60 MB. Do not raise `GPU_WORKERS` above four.
Measure with `nvidia-smi` during a four worker run of `--mask face,text,screen`
and record the peak in the report.

### 4.5 Static graphs

DirectML mis-computes dynamic shape graphs in this repo's experience
(`docs/report.md` section 6). Every new ONNX model runs at fixed input shapes
through `detect.SessionPool`, one session per shape. Text detection therefore
runs at fixed long sides (two of them, chosen in T1), letterboxed the way
`screens.letterbox` does, and never at the source resolution.

### 4.6 Evaluation cache fingerprint

`eval/common.fingerprint` hashes `faceblur/detect.py` and `RAW_SETTINGS`.
The raw face cache does not depend on text or screen code, so it does not
need to move. New caches (oracle pseudo labels, text runs, card runs) each
carry their own fingerprint: sha256 of the module that produced them plus the
settings that shape them, and a version number. Store them as JSON, not
pickle; `eval/cache/` is gitignored either way.

## 5. Work packages

Order matters. R1 and S1 are debt on a kind already called ready. E1 comes
before any text or card work because F0, T2 and the screen recall number all
depend on it. F1 is the one package that changes whether the tool can run
unattended, which is worth more than a third kind, so it comes before text.

Each package ends with a verify step that is a command or a test, a report
section, a `STATE.md` update and one commit with the trailers.

### R1. Housekeeping (1 day)

**Goal.** Documents and the packaged app say what the code does.

**Scope.**

- `README.md`: lines 15 and 175 to 179 say faces are the only kind. Rewrite
  "What it does and does not do" for three kinds with screens built and text
  not; add a "Screens" subsection with the four caveats from section 15.4;
  add `yolox_tiny.onnx` to the detectors table; update the CLI usage block to
  match `cli.py`.
- `docs/report.md` section 7: "No number plate, text, or body redaction"
  becomes plates and bodies only, with a pointer to section 15.
- `FACEBLUR_BUILD_PLAN.md` scope change table: screens row to "Built, section
  15".
- Rebuild `dist\FaceBlur` with `build\faceblur.spec`. The current one is from
  2026-09-05 and lacks the hand and screen models.
- `tests/test_models.py`: a test that every `.onnx` in `models/` except
  `sface.onnx` appears in the spec, so the next model cannot be forgotten.

**Verify.** `pytest tests`. On a machine or account without Python on PATH,
`dist\FaceBlur\FaceBlur.exe` runs a batch of the four samples with screens
ticked and writes four copies with records that carry `screens > 0`.

### S1. Screens in the gate (2 days)

**Goal.** A screen the run missed, found in the finished copy, is counted and
can hold the copy back. Condition 2 of "ready".

**Design.**

- `verify.applied()` calls `ellipse_for` and reads the ellipse bounds. Change
  it to take the region from `redact.region_for(det, settings)` and use its
  `bounds()`. The control `redact(before, [det], settings)` already handles a
  quad through `shape_of`. Nothing else in `applied` is face specific.
- `residual_job`: when `settings.wants("screen")`, also run
  `batch._screens(settings).detect(after)` on the copy frame. For each screen
  box, `applied` and `classify` exactly as for faces. The size floor for
  screens is `screen_gate_min_px` (default 48, long side) rather than
  `quarantine_min_px`, because a 24 px screen shows nothing readable.
- **Persistence in the gate.** The detector calls a table a television for a
  frame. Section 15.2: 14 of 29 false runs last a single frame. A gate that
  quarantined every copy for a flicker would hold every file back. So the
  check collects screen finds per frame across the whole file (they already
  come back per job and are merged in `summarise`), groups them with
  `screens.runs_in`, and counts only runs of at least `screen_gate_min_run`
  checked frames (default 3, and the run length is in checked frames, so with
  `check_stride 2` that is six source frames; say so in the setting's
  comment).
- A masked screen still reads as a screen to YOLOX (a pixelated rectangle is
  a flat rectangle). That is what `classify` is for: `changed / would` near
  one means the mask ran. The peak percentile rule from faces applies
  unchanged; test it on a synthetic masked quad.
- Hands do not apply to screens. The hand rule is not asked.
- `summarise` adds `residual_screens`, `residual_screen_frames`,
  `residual_screen_runs`, `residual_screen_max_px`, and rows in
  `residual_list` carry `"kind": "screen"`. Face rows carry `"kind": "face"`
  so a reader of an old record sees the field is new.
- `batch.run_video` step 6: `held_back` is true if the face condition holds
  **or** (`settings.wants("screen")` and `residual_screen_max_px >=
  screen_gate_min_px` and at least one run of length `screen_gate_min_run`).
  `cli.check_line` and `ui/strings.py` say which kind held it back.
- `describe()` in `verify.py` and `python -m faceblur.verify` gain
  `--mask face,screen` so the standalone check knows what to look for; the
  default stays `face`.

**Settings.** `screen_gate_min_px: int = 48`, `screen_gate_min_run: int = 3`,
validated like their neighbours, in `to_dict`.

**Tests.** `tests/test_verify.py`: a quad region reads masked after
`redact` and missed on an untouched copy; a single frame screen find does
not hold a copy back, three consecutive do; the record fields round trip. A
test in `tests/test_classes.py` that every ready kind is named in
`verify.KINDS_CHECKED` (a tuple the module exports), so condition 2 is
enforced by a test rather than by memory.

**Measure.** Run `--mask face,screen --check-output` on the four samples.
Report section 16.1: screens the check finds on the copy, by file, and how
many are the two frame monitor section 15.4 says the persistence rule loses.
Expect the four files to be held back for faces anyway; say whether any would
be held back for a screen alone.

### E1. Object oracle, evaluation only (3 days)

**Goal.** A witness for screens, cards, slabs and text surfaces that is
independent of the shipped detectors, so that screens get a recall number,
F0 gets card regions, and T1/T2 get surface labels. Same role MediaPipe
plays for faces.

**Model.** OWLv2 (Google, Apache 2.0) through `transformers` in `.venv-eval`.
Pin the model revision hash in `requirements-eval.txt` comments and in
`eval/oracle_owl.py`; cache the weights under a gitignored folder; never
download at pipeline run time; never ship. If `transformers` plus a CPU
`torch` cannot be pinned cleanly beside MediaPipe's numpy 1.26 pin, make a
third environment `.venv-oracle` rather than loosening a pin. GroundingDINO
is the fallback (Apache 2.0, heavier).

**Prompts.** `television`, `computer monitor`, `laptop`, `mobile phone`,
`trading card`, `graded card slab`, `binder page of trading cards`, `name
badge`, `paper document`, `handwritten note`, `sign`. Run at stride 5 over
the four samples on the CPU; expect about an hour per file; cache results as
JSON keyed by frame with boxes, label and score, never crops.

**Outputs.**

- `eval/oracle_owl.py VIDEO --stride N` writes `eval/cache/<stem>.owl.json`.
- `eval/screens.py VIDEO` measures the shipped screen class against the
  oracle: recall of oracle screens over 0.4 percent of the frame (the size
  the real television reads at, section 15.2) by the shipped rule, and
  precision of the shipped rule against the oracle. Also re-scores the 85 by
  eye runs against the oracle and reports agreement, so the reader knows how
  far to trust either witness.
- Card regions per frame for F0 and T2.

**Verify.** `docs/report.md` section 16.2 has the screen recall number and
the oracle agreement number. The oracle's licence and revision are in
`models/README.md` under an "Evaluation only" heading with `sface.onnx`.

**Risk to name in the report.** The oracle is one model's opinion. Its recall
number for screens is a bound, not a truth, the same caveat section 12.5
gives for the recogniser.

### F0. Faces on cards (1 day after E1, plus the owner's call)

**Goal.** Know how often the face class masks a printed face on a card, and
give the owner the switch.

**Measure.** For each face track from `pipeline.plan` (through the eval raw
cache), the share of its detections whose ellipse centre lies inside an
oracle `trading card` or `graded card slab` box. A track with more than half
its centres inside cards is a printed face. Report section 17: tracks, frames
and masked pixels that are printed faces, per file, and three frame numbers
per file for the owner to look at if they wish.

**Design.** `face_on_card: str = "mask"` until the number exists, then
`"leave"` as D1 says. When `leave`, a card veto runs between step 2 and
step 3 in `run_video`: drop every face `Detection` from `per_frame` whose
ellipse centre lies inside a card region of that frame, where card regions
come from T2's shipped card detector. Until T2 exists, F0 only measures; the
setting is added but only `mask` is accepted by `validate`, with a message
that says the veto is not built.

**Edge cases.** A live face behind a card held up to the camera: the ellipse
centre is on the face, not on the card, unless the card covers the face, in
which case there is no face to mask. A card in a binder page filling the
frame: card regions cover most of the frame; a live face reflected in a
sleeve is below 24 px and outside the gate anyway. A card with a real
person's photo who is present in the room (a custom card): out of scope,
say so.

### F1. Second chance for missed faces (1 week)

**Goal.** The gate holds all four sample files back for about fifty real
faces. Lower that without raising thresholds globally, which section 10
showed masks hands and a wall sign.

**The idea.** The output side check already runs the full confirmed detector
bank on every checked frame of the copy and, through the hand rule, sets the
wearer's hands aside. A find classified `missed` and not a hand is a
confirmed face at known coordinates in a frame the pipeline never masked.
The copy and the source share geometry frame for frame (verified by the
join). So the finds are evidence the tracker never saw, and the cheapest
second chance is to give it to the tracker and write the file again.

**Design.**

1. Run steps 1 to 6 as now. If `settings.second_chance == 0` or the check
   found no `missed` face rows, stop.
2. Build `seeds: list[list[Detection]]` of length `n`, one row per checked
   frame, from the `missed` rows that are not hands, with the copy side
   detection's box, landmarks and score (extend `residual_list` rows with
   what `Detection.to_dict` carries, or return the `Detection` objects
   alongside; the rows are what the record keeps).
3. Merge seeds into `strong` from step 1 (append per frame; `nms_detections`
   at `settings.nms_detect` to fold duplicates) and run `tracker_for(settings)
   .run(strong, weak, shifts, shape)` again. The tracker's own rules apply
   unchanged: a seed still needs `min_track` sightings, the sure track rule
   still gates continuation, plausibility gates still apply. This is what
   keeps the 30 "neither" finds (a television, a picture, a wall) out: they
   have to persist and confirm like anything else. Do not lower any
   threshold for the second pass.
4. Re-run steps 2b to 5 on the new `per_frame`. Clean stretches are copied
   again, masked stretches re-encoded. Write to a temporary name and
   `os.replace` over `dst` only after the join verifies, as the first pass
   does.
5. Run step 6 once more. The record carries `residual_faces_first_pass`,
   `second_chance_seeds`, `second_chance_tracks_added`, and the final
   `residual_*`. Quarantine decides on the final numbers.
6. At most `second_chance` extra rounds (default 1). No third round by
   default; measure whether a second buys anything and say.

**Constraints.**

- Seeds only from `check_stride == 1` runs. With a stride the tracker's gap
  rule may not bridge the seeds; refuse the combination in `validate` rather
  than silently doing less.
- Seeds inherit `kind="face"` and `source="copy"` so the record can tell them
  apart and the sweep can count them.
- The hand rule runs before seeding, as now. Add a second guard: a seed
  whose box overlaps a hand quad the rule returned, at any coverage, is not
  seeded even if it was under `hand_cover`. Hands are the thing this must not
  reintroduce.
- Cost is one more write and one more check on files with finds. On the
  samples that is every file. Record `second_chance_seconds`.

**Measure.** The label free harness plus the check: for the four files,
`residual_faces` before and after, hand pixels touched (`eval/measure`
`hand_damage`), off face masking, and the recogniser leak count from
`eval/reid`. Report section 18. The result the owner cares about is the
number of sample files the gate would still hold back, and why.

**Tests.** A synthetic video with a face the source side detectors are made
to miss (paste a face only into the copy side test, or mock `bank.detect`
on the copy to return one box for three frames): after the second chance the
copy is masked there. A single frame seed does not produce a mask. A seed
overlapping a hand quad is dropped. `second_chance=0` changes nothing.

### T1. Text detector (3 days)

**Goal.** Find text lines as rotated quads, measure what that costs and what
it fires on. No masking policy yet beyond the size floor.

**Model.** PP-OCRv4 mobile detection model (PaddleOCR, Apache 2.0). Convert
from PaddleOCR's own release with `paddle2onnx` in a throwaway environment;
record the source URL, the source tar's sha256, the converter version and
the output sha256 in `models/README.md`; commit `models/ppocr_det.onnx` and
the licence text. If conversion is not reproducible in a day, DBNet from
MMOCR (Apache 2.0) is the fallback. CRAFT (MIT) is the third. Do not take a
pre converted file from an unknown repository.

**Inference.** Fixed input shapes through `SessionPool`: two long sides,
starting at 960 and 1920, letterboxed top left on the model's own mean and
std (not raw, unlike YOLOX; check against a frame by eye as section 15.1
did, and record which normalisation made the probability map light up).
Output is a probability map; threshold at `text_thresh` (0.3), find contours,
`cv2.minAreaRect` per contour, unclip by `text_unclip` (1.6), keep boxes whose
mean probability is over `text_box_thresh` (0.6). **Emit four point convex
quads only**: `redact.build_alpha` uses `fillConvexPoly`, and a curved or
concave contour would fill wrong. Filter by cap height (`text_min_px`, 12)
measured as the quad's short side, and by `text_max_frac` (a quad over 30
percent of the frame is a page, not a line; keep it, but flag it in the
record as `text_pages`).

**Grouping.** `text.hold` on the pattern of `screens.hold`: runs by same
kind and overlap, `text_min_run` (start at 2; text is small and the camera
shakes, so 3 may cost real lines), `text_gap` 6, `text_tail` 3. Overlap for
rotated quads: axis aligned IoU of the bounding boxes at `text_link_iou`
(0.3) is enough for linking; do not build polygon IoU unless the measurement
says linking is what fails.

**Merge across scales.** The 960 pass finds large text, the 1920 pass small.
Union, then NMS on bounding boxes at 0.5, keeping the higher scoring quad.

**Measure, before any policy.** On the four samples, by file: lines found per
frame, share of the frame their quads cover, time per frame at one and two
scales on the GPU, and the surface each line sits on according to the E1
oracle (card, slab, screen, document, badge, sign, none). The card column is
the number T2 is decided on. Report section 19.1. A by eye audit of up to 100
runs is allowed here under the standing rule if the oracle's surface labels
disagree with what the detector finds; record it like sections 14.1 and 15.2.

**Not in T1.** Masking with a policy, the gate, OCR. `TEXT.ready` stays
`False`; the CLI refuses `--mask text` with the existing message. Text runs
under an eval flag only (`eval/text.py`).

### T2. Card and slab veto (2 to 5 days, decided by T1's numbers)

**Goal.** Text on a card or slab is left alone. Condition for D2.

**Decide by measurement, in this order.**

1. **Size floor alone.** If T1 finds that text on cards is nearly all below
   `text_min_px` at the distances in the footage (card names print at about
   8 to 15 px at arm's length on this camera) and personal text that matters
   is above it, the floor is the veto. Ship that, measure the leak of card
   text over the floor, and stop if it is under 2 percent of card text lines.
2. **Geometric veto.** Cards are 63 by 88 mm, aspect 0.716; slabs about
   0.74. A quad that lies inside a rectangle of that aspect found by edge
   detection around the quad (Canny, `approxPolyDP`, four corners, aspect
   within 0.05, area under `card_max_frac`) is on a card. Cheap, no model, no
   training. Fails on occluded and binder cards; measure how often.
3. **Distilled card detector.** Train YOLOX tiny (Apache 2.0 training code)
   on frames pseudo labelled by the E1 oracle for `card`, `slab`, `binder
   page`, in a throwaway training environment, and ship the ONNX. No human
   labels, consistent with the standing rule; quality bounded by the oracle,
   say so. Only if 1 and 2 leak more than the number set in step 1.

Whichever ships, the veto is `text.veto(quads, cards, settings)`: a text quad
is left when at least `text_veto_cover` (0.8) of its area lies inside a card
region. **A partial overlap is masked**, because a note lying across a card
is a note.

**Also used by F0.** The same card regions veto faces when `face_on_card ==
"leave"`.

**Record.** `cards_seen`, `text_left_on_cards`, `faces_left_on_cards`.

### T3. Text in the gate, and ready (2 days)

**Goal.** Conditions 2 to 4 for text, then `TEXT.ready = True`.

- `residual_job`: run the text detector on the copy, apply the same size
  floor and card veto, `applied` through `region_for`, `classify` as before.
  Persistence in the gate as S1 does, `text_gate_min_run` 2.
- `held_back` adds the text condition on `residual_text_max_px >=
  text_gate_min_px` (default 16, the smallest cap height a reader could use).
- `TEXT.ready = True`; `tests/test_classes.py::test_which_kinds_this_build_can_actually_do`
  changes with it, on purpose; `ui/strings.py` `MASK_TEXT_HELP` already reads
  in the present tense and the window drops the "Not in this build yet"
  prefix on its own.
- Per kind mask accounting from 4.3 must be in before this, or text will
  push every file over the 5 percent budget for the face gates.
- `build/faceblur.spec` and `tests/test_models.py` take the model.

**Measure.** Report section 19.2 to 19.4: precision by surface against the
oracle, share of the frame masked by text per file, cost, the leak of card
text, and the caveats: no recall for text below the detector's floor, tuned
on one venue, OCR not used and why.

### T4. Identifiers by pattern (3 days, optional)

**Goal.** Phone numbers, email addresses and postcodes are masked even on a
surface the policy leaves, when they are large enough to read.

- PP-OCRv4 recognition model (Apache 2.0) on text quads over `text_ocr_min_px`
  (20), only on non card surfaces or, if D2 is amended, on all.
- Patterns: international and local phone formats, email, Vietnamese and
  general postal number shapes. No name lists; a name is not detectable by
  pattern and a wrong list is a false promise.
- **The recognised string is never written anywhere**: not to the record,
  not to a log, not to an eval cache. The record carries counts by type and
  the frame and box. A test asserts that no field of the record or of any
  eval output contains the recognised text (feed a known string through a
  synthetic frame and grep every file written).

### M1. Metadata line (half a day)

Container metadata was declined. Add to the record `source_audio: bool` and
`source_tags_present: list[str]` (tag names only, never values: a GPS value
is itself personal data) from `video.probe`. The README line "copies the
audio track untouched" gains "the four sample files carry none".

## 6. Cross cutting rules for every package

- **Settings.** Prefix per kind (`screen_`, `text_`, `card_`), gate settings
  `*_gate_*`, every new field validated in `Settings.validate` with a message
  that names the field and the bound, listed in `to_dict`, with a comment
  that says why the default is what it is and which report section measured
  it. A default with no measurement says "first guess, section N will
  measure it".
- **Audit record.** New fields default to zero or empty so an old record reads
  back. Every count a decision hangs on leaves set aside items out and keeps
  them in a list marked with the reason, the way hands are.
- **Models.** Commit the file, its licence text and a `models/README.md`
  entry with URL, size, sha256, input layout, normalisation and output
  layout. Add to `tests/test_models.py::EXPECTED` and to the spec's `datas`.
  Verify the decode against a real frame by eye and against a hand built
  output in a test, as `tests/test_screens.py` does.
- **Eval caches.** JSON, fingerprinted, under `eval/cache/`, boxes only. No
  crops, no frames, no recognised text on disk anywhere under the repo. Frames
  written for a by eye audit go to a folder outside the repo or under
  `footage_*`, which is gitignored, and are deleted after.
- **Tests.** Named as sentences that say what would go wrong. Pure logic tests
  need no model; model tests skip when the file is missing, as today. Update
  the count in `docs/report.md` section 9.
- **Docs.** Sentence case headings, active voice, no em dashes or en dashes,
  one word per thing (video, blurred copy, output folder, mask, kind).
- **Commits.** One per package, message says what and why, trailers as the
  session is told. Push to `HashBrake/faceblur` `main`.
- **Handover.** Update `STATE.md` at the end of every package: what changed,
  numbers, decisions taken by default, what a next person should not assume.

## 7. Security and edge case risks

Numbered so the report can cite them. Each names the mitigation and the
package that owns it.

**Leaks (the thing this tool exists to prevent)**

1. **OCR fails silently on small or blurred text.** Mitigation: policy by
   surface, no OCR in the primary mechanism; T4 is additive only. Owner: D2,
   T1 to T3.
2. **A card veto swallows personal text on or near a card.** A sticky note on
   a card, a phone on a binder page, a badge beside a slab. Mitigation: veto
   needs `text_veto_cover` 0.8 of the quad inside the card region; partial
   overlap masks; card regions capped at `card_max_frac`. Measured in T2.
3. **The distilled card detector calls a document a card.** Mitigation:
   aspect check on the region as a second condition even when a model found
   it; leak of oracle documents measured in T2.
4. **Second chance reintroduces hands.** Mitigation: no threshold lowered;
   seeds pass the hand rule and a hand quad overlap guard; tracker rules
   unchanged; `hand_damage` measured before and after. Owner: F1.
5. **Second chance masks the 30 "neither" finds** (television, picture,
   wall). Mitigation: seeds still need `min_track` and plausibility; off face
   masking measured. Accepted cost if it stays under the sweep gates; say so.
6. **A masked region that is not destroyed.** Pixelate and blur modes both
   shrink then upsample, so no mode is a plain Gaussian. For a quad the block
   count is sized by the short side (`QuadRegion.block_scale`); a test must
   show a 400 px by 20 px line of text is unreadable after `redact` at the
   shipped strength, measured by the recogniser's text counterpart if T4
   ships, else by block size against cap height.
7. **A copy with two kinds on passes the face gate because the face mask
   share is diluted by screen masks.** Mitigation: per kind accounting, 4.3.
8. **Frame misalignment between source and copy** would make `applied` read
   the wrong pixels. The join verifies frame count and timestamps; `plan_jobs`
   uses `min(len(times), len(out_times))`. Keep both; a test with a copy one
   frame short must fail the check loudly, not read shifted.
9. **The gate quarantines everything for a flicker** once screens and text
   are in it. Mitigation: persistence in the gate (S1, T3), measured on the
   four files before the gate condition is switched on.
10. **Stride interactions.** `screen_min_run` and `text_min_run` count hits,
    which at `stride 2` are two source frames apart; the gate's run lengths
    are in checked frames. Settings comments say so; `validate` refuses
    `second_chance` with `check_stride > 1`.

**Supply chain and data handling**

11. **Model files.** Every shipped model has a pinned sha256 in a test, a
    licence file, and a recorded source. No runtime download anywhere. The
    oracle downloads once in `.venv-eval` with a pinned revision and is never
    in the spec; `tests/test_models.py` asserts `sface`, `owl` and `torch` are
    absent from the spec text.
12. **Licences.** PP-OCR, DBNet, OWLv2, YOLOX are Apache 2.0; CRAFT MIT.
    YOLO-World, Ultralytics and most YOLOv5 to v8 derivatives are GPL or AGPL
    and must not be used even for evaluation code that lands in the repo.
    Check the licence file in the release, not the README.
13. **Pickle.** `eval/common.py` caches are pickle, loaded from a gitignored
    local folder; acceptable, but every new cache is JSON. Never unpickle
    anything that did not come from this machine.
14. **Recognised text is personal data.** T4 never persists it; a test proves
    it. Eval outputs that show frames (`eval.misses --images`) already go to a
    user named folder; the same for any text or card visualisation.
15. **The audit record is shipped beside the copy** and will be read by
    people who did not run the tool. It carries box coordinates and counts;
    it must never carry pixels, strings from OCR, tag values, or absolute
    paths beyond the file names it carries today.
16. **Quarantine keeps the only copy.** Nothing deletes a quarantined file.
    Second chance writes to `part_path(dst)` and replaces atomically; a crash
    mid rewrite leaves the first pass copy in place with its record, not a
    half written file. Test the crash path with a raised `VideoError` in the
    second write.
17. **Model memory on the GPU.** Four workers plus three more graphs each.
    Measure peak; if DirectML pages, the text detector's second scale runs on
    the CPU (`text_device` like `hand_device`), and the report says so.

**Edge cases in the footage**

18. **Non convex text contours.** `minAreaRect` before masking; a test feeds a
    U shaped contour and asserts a convex quad.
19. **Text at the frame edge.** Quads clipped to the frame in `_pad` style;
    `grow_quad` about the centre cannot fold a thin quad; zero area quads
    dropped before `region_for`.
20. **A page filling the frame.** Masked whole; flagged as `text_pages`;
    accepted by D2 and counted against the text budget only.
21. **Binder pages.** Dozens of cards; the card veto must handle many small
    regions per frame without quadratic cost (grid index by cell, or bounding
    box prefilter). Measure time on a binder heavy stretch if the samples
    have one; if not, say there is no footage to set it on.
22. **Reflections.** A phone screen reflected in glass is a screen the oracle
    may or may not call one. Out of scope for the number; say so.
23. **Faces on screens.** A face on a television is masked by the face class
    today (section 13.4). With screens on it is masked twice; the union alpha
    handles overlap. With screens off it stays as it is. No change.
24. **Vertical and rotated text.** PP-OCR det finds rotated lines; the quad's
    short side is the cap height either way. A test with a 90 degree line.
25. **The wearer's own badge or clothing text.** Personal, masked under D2.
    A test frame with a badge at 30 px cap height is masked.
26. **Card text under the size floor that is nevertheless readable in the
    source** (a close up). The floor is a leak in the other direction: it
    leaves card text alone by design and personal text under 12 px by
    consequence. Report it as a caveat with the count from T1.

## 8. Measurement plan

| Number | Method | Section |
|---|---|---|
| Screens the check finds in the copy, held back for a screen alone | S1 on four files | 16.1 |
| Screen recall against the oracle, oracle agreement with the 85 runs | E1 | 16.2 |
| Printed faces on cards: tracks, frames, pixels | F0 | 17 |
| Residual faces before and after second chance; hand damage; off face; reid leaks; files still held back | F1 | 18 |
| Text lines per frame, share of frame, cost, surface distribution | T1 | 19.1 |
| Text precision by surface, card text leak, over masking of signage | T2, T3 | 19.2 to 19.4 |
| Identifier hits by type, never the text | T4 | 19.5 |
| GPU peak memory with all kinds on, four workers | 4.4 | 4.1 update |

Every number is produced by a command under `eval/` or `faceblur.verify`
that a third party can re-run; the by eye audits allowed by section 1 are
recorded as such and are not the number quoted.

## 9. Sequence and estimates

| Order | Package | Estimate | Depends on |
|---|---|---|---|
| 1 | R1 housekeeping | 1 day | nothing |
| 2 | S1 screens in the gate | 2 days | R1 |
| 3 | 4.3 per kind accounting | 1 day | nothing; before T3 |
| 4 | E1 oracle | 3 days | nothing |
| 5 | F0 faces on cards, measure | 1 day | E1 |
| 6 | F1 second chance | 5 days | S1 |
| 7 | T1 text detector, measure | 3 days | E1 |
| 8 | T2 card veto | 2 to 5 days | T1 |
| 9 | F0 veto | 1 day | T2 |
| 10 | T3 text in the gate, ready | 2 days | T2, 4.3 |
| 11 | T4 identifiers | 3 days | T3, optional |
| 12 | M1 metadata line | half a day | nothing |

About four weeks for everything, three without T4. F1 can run in parallel
with E1 to T1 if two sessions are used, since it touches `batch.py` and
`verify.py` while the text work touches new modules; merge S1 first either
way.

## 10. What not to do

From `STATE.md` and the report, so the next session does not re-learn them:

- Do not raise face sensitivity or lower `conf`: YuNet scores a hand on a mop
  at 0.74 to 0.82 and a wall sign at 0.80.
- Do not add a 2560 scan or tiling for faces; measured and rejected, section
  12.4.
- Do not change `faceblur/detect.py` casually: every edit, comments included,
  invalidates the raw eval caches by design.
- Do not move `hand_presence` off 0.7 without re-running the 1050 point grid
  in section 14.2.
- Do not use the hand models to replace the tracker's hand rules without
  re-measuring everything the rules touch.
- Do not tighten `screen_min_run` to 4: two points of precision for fifteen
  real screens.
- Do not ship a mode that only Gaussian blurs.
- Do not put a kind on by default.

## 11. Open questions for the owner, beyond D1 to D3

- Does the footage contain sports cards with real faces, or only game cards?
  Changes the weight of D1.
- Is there footage from a second venue? The screen cap and every text number
  will be tuned on the table tennis hall and the canteen otherwise.
- Should a copy the gate holds back for text alone be delivered with the text
  masked at a lower floor instead of quarantined? Not built; a policy choice.
