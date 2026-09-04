# Precision audit of the first build

The first build blurred every face and also the wearer's hands, whole counters,
and in places most of the frame. The footage is training data, so every false
blur destroys signal. This audit found why. `docs/precision_report.md` holds the
measurements of the rebuild.

## What the first build destroyed

Diff of every source frame against its blurred copy, four videos, 7926 frames:
15 percent of frames had more than a quarter of the picture destroyed.

On the 42 hand labelled frames from Phase 3:

| Stage | Share of frame masked |
|---|---|
| Labelled faces only | 1.0% |
| Raw detections at conf 0.25 | 2.9% |
| plus 30 percent padding | 6.8% |
| plus propagation, 6 frames each way, 6 percent growth per frame | 17.5% |
| plus both, the shipped build | 33.2%, max 92.6% |

89.6 percent of masked pixels lay more than 40 px from any labelled face.

## Root causes, ranked

1. **Propagation and padding multiplied every mistake eleven times.** A box
   became 13 frames of mask that grew as it spread, then padding grew it again,
   then NMS at 0.6 merged neighbours into bands. 14 percent of boxes appeared in
   one frame only.
2. **The threshold was far too low, and the big boxes were the wrong ones.**
   Boxes scoring 0.25 to 0.40 were 62 percent correct by count but 12 percent
   correct by area. Boxes over 160 px were 23 to 33 percent correct. The largest
   real face was 150 px; every worst frame held a 300 to 800 px box at score
   0.27 to 0.54 on a wall, a mirror, a blurred body, or the wearer's arm. Boxes
   scoring 0.8 or more were 100 percent correct.
3. **Rectangles with 30 percent padding were six times the face.** An ellipse
   from the five landmarks covers the identity region at a median 17 percent of
   the padded rectangle's area.
4. **Union everywhere.** Multi-scale union and the "both" engine were chosen for
   recall. Each addition added false boxes. Requiring the two detectors to agree
   gave 95 percent precision against 77 percent.
5. **No sanity checks.** No size cap, no persistence requirement, no mask budget.
6. **The measurement was one sided.** Phase 3 measured recall only. The plan's
   99 percent gate rewarded over blurring. Nothing measured precision.
7. **Whole frame re-encode at crf 20** degraded every pixel of training data.

## What the rebuild changed

| Area | First build | Rebuild |
|---|---|---|
| Detection sizes | 640 and 1280 | 1280 and 1920 |
| Threshold | 0.25 | swept, gate driven |
| Second detector | union for recall | confirms each box |
| Size cap | none | 15 percent of the frame's long side |
| Temporal | copy 6 frames each way with growth | tracker, 3 confirmed sightings minimum, continuation on YuNet alone once confirmed, gaps interpolated, 2 frame tails, no growth |
| Mask | rectangle, 30 percent pad | ellipse on the eye line, feathered |
| Pixel destruction | whole frame downsample | per face, blocks scale with face size |
| Encode | crf 20 medium | crf 12 fast |
| Quality report | none | masked share per frame, flagged frames, tracks |
| Evaluation | hand labels, recall only | no labels: synthetic recall, detector consensus, hand regions, mask budget |
