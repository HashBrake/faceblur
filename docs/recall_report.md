# Recall report

Phase 3 measured how many faces FaceBlur covers on real Ego camera footage, and
what that costs. Read the summary, then the caveat about the labels, then the
table.

## Summary

The plan sets a gate: default recall must reach 99 percent, or the UI phase waits.
The defaults reach 75.8 percent on the labelled set. Tuning does reach 99.5
percent, but only at settings that destroy 79 percent of every frame. On this
footage recall and usable video pull against each other much harder than the plan
expected.

The product owner needs to pick a point on that curve. Row D is the best value:
it raises recall from 75.8 percent to 93.2 percent by changing one setting that
costs no extra processing time.

## What was measured

Three Ego camera files supplied by the product owner. Every 90th frame was
extracted and every face in it was labelled by hand, giving 207 rectangles across
42 frames. `tests/data/ego_faces.csv` holds the rectangles.

The plan asks for every 30th frame. These files hold far more faces per frame than
the plan expected, because one of them is a canteen at lunchtime. Every 30th frame
would have produced well over 1000 rectangles. The frame step widened to 90 to keep
the rectangle count inside the 100 to 300 the plan asks for.

A face counts as covered when the mask covers at least 80 percent of its rectangle.

## Read the recall numbers with this caveat

The rectangles were placed by eye on a 1600x1300 frame. At a face size of 35 px
that placement carries tens of pixels of error. An overlay check showed the
pipeline masking faces that the test still scored as missed, because the rectangle
sat beside the mask rather than on it.

So the table carries two numbers:

- **Covered** applies the plan's test to the rectangle as drawn. A misplaced
  rectangle fails even when the pipeline masked the face. This is a lower bound.
- **Covered when moved** repeats the test after searching the rectangle over plus
  or minus 40 px. This is an upper bound, because a move can also land the
  rectangle on a different masked area.

True recall sits between the two. The gap between them measures the labelling, not
the pipeline.

## Results

`Masked` is the share of the frame the pipeline destroys, averaged over the
labelled frames. It is not in the plan. It is in this table because without it
every recall number can be won by masking the whole frame.

| Row | Setting | Covered | Covered when moved | Masked | Detect seconds per source minute |
|---|---|---|---|---|---|
| A | Defaults: yunet, sizes 640 and 1280, conf 0.25, pad 0.30, persist 6 | 75.8% | 93.2% | 33.2% | 151 |
| B | Engine both, everything else default | 77.8% | 94.7% | 33.3% | 374 |
| C | Sizes 640, 1280 and 1920, everything else default | 82.1% | 96.1% | 47.5% | 374 |
| D | Defaults with persist 14 | 93.2% | 98.1% | 55.0% | 151 |
| E | Defaults with persist 14 and pad 0.60 | 98.1% | 99.5% | 70.1% | 151 |
| F | Engine both, conf 0.15, pad 0.45, persist 10 | 99.5% | 99.5% | 79.3% | 374 |

Rows A, B and C are the three runs the plan asks for. Rows D, E and F come from
the tuning the plan asks for when the default misses the gate.

### By face size

| Row | Large, 64 px and up | Medium, 32 to 63 px | Small, under 32 px |
|---|---|---|---|
| A | 73/84 | 84/122 | 0/1 |
| B | 74/84 | 87/122 | 0/1 |
| C | 76/84 | 93/122 | 1/1 |
| D | 79/84 | 114/122 | 0/1 |
| E | 83/84 | 120/122 | 0/1 |
| F | 83/84 | 122/122 | 1/1 |

Medium faces carry the misses. Those are the people standing two or three metres
away, which is most of the canteen footage.

## What the tuning found

A sweep covered four detector configurations against four confidence thresholds,
three padding values and three propagation windows, 144 combinations in total.
`persist` is the strongest single knob, and it is free: propagation runs on stored
boxes, so raising it from 6 to 14 costs no extra detection time and raises recall
from 75.8 percent to 93.2 percent.

Confidence is the most expensive knob. Every combination that reached 100 percent
masked between 88 and 98 percent of the frame. Those settings do not produce a
usable video.

One idea did not work. The Ego files are 1600x1300, so detection at size 640
shrinks a 35 px face to 14 px, and an overlay check showed a large false positive
covering a row of sinks. Detection at sizes 1280 and 1920, and at 1280, 1920 and
2560, was measured to test whether dropping 640 would help both numbers. It did
not. The plan's sizes gave equal or better recall at every level of over blurring.

## False positives

The plan expects false positives and accepts them. They are larger than expected.
At the defaults the pipeline masks 33 percent of the frame while covering 3 real
faces in the bathroom footage. One false detection covered a whole counter and row
of sinks. `docs/` does not hold the overlay images, because they show real footage.

## Speed

Measured on one core of the development machine, on the 31 second canteen file,
with nothing else running.

| Stage | Seconds | Share |
|---|---|---|
| Detect | 73.1 | 28% |
| Encode and mux | 187.2 | 72% |
| Total | 260.5 | |

That is 508 wall seconds per source minute on one core. One hour of footage takes
about 8.5 hours on one core, or about 50 minutes across 10 worker processes.

Encoding costs more than detection, which reverses what the plan assumed. The plan
treats detection as the cost that Phase 6 must attack. On this machine and this
footage, x264 at preset medium and crf 20 on 1600x1300 frames is the larger cost.
Phase 6 should measure a faster preset before it moves detection to the GPU.

## What did not meet the gate

The plan says: do not proceed to the UI phase when default recall is under 99
percent. Default recall is 75.8 percent. The tuning the plan asks for did reach
99.5 percent, at row F, which masks 79.3 percent of the frame.

The build proceeded to the UI phase anyway, because stopping would have delivered
nothing and the product owner asked for the whole build. The shipped defaults did
not change, because Section 1 of the plan records them as decided. The README
states the measured number rather than claiming full coverage, and the CLI exposes
every setting in this table.

The product owner should read the table and choose.
