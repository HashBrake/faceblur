# The scoped second chance on 004310, 2026-10-16 (T17, T19)

Two diagnostic runs on this PC of
`ego_AZER76400FE_19700101_004310_camera_left_part0001.mp4` (938 frames), at
4 workers with `--quarantine --second-chance 1` and every other setting at
its default (`second_chance_scope` "changed", `second_chance_margin_seconds`
0.5). Their outputs and records are under the gitignored `bench_out/`; this
note holds the measured numbers so that report section 26.4 and DECISIONS
D30 and D31 have a committed source.

The question: why the scoped round re-encoded all 938 frames of this file
on both machines (`docs/bench/speed_2026-10-16.json`,
`docs/bench/speed_2026-10-16_5090.json`). Either the seeds are spread so
that no clean stretch is left to copy, or the re-track moves boxes by
fractions of a pixel and the mask comparison reads every frame as changed.

## Run A (T17): why the round wrote the whole file

| | |
|---|---|
| wall | 220.7 s |
| `second_chance_scope_fallback` | `no_copy_segment` |
| seeds / kept | 41 / 3 |
| frames re-encoded | 938 of 938 |
| residual faces, first check → after the round | 41 → 7 |
| reorder delay | 0 |
| keyframes in the first copy | every 30 frames, 32 in 938 |
| `min_copy_seconds` | 2.0 s (60 frames) |
| margin | 0.5 s (15 frames) |

The copy could be cut at any second, and the encoder reorders nothing, so
neither of those forced the whole file. After the changed frames are
widened by the margin and clean stretches shorter than `min_copy_seconds`
are folded into their neighbours, no segment is left to copy.

## Run B (T19): what the re-track changed

The mask key was wrapped in the process to count the changed frames and
measure each change. 213.3 s wall.

| | |
|---|---|
| frames changed, before the margin | 600 of 938 |
| of those, a box gained or lost | 541 |
| of those, a box moved by 2 px or more | 59 |
| of those, a box moved by less than 2 px | 0 |
| changed runs | 23 |
| longest clean stretch | 81 frames |

The changed runs, as inclusive frame ranges:

(43, 55), (58, 77), (80, 99), (102, 170), (173, 200), (233, 282),
(284, 285), (288, 289), (293, 304), (314, 316), (318, 325), (362, 362),
(384, 384), (410, 413), (415, 427), (429, 430), (436, 437), (485, 545),
(561, 562), (565, 750), (753, 791), (794, 854), (856, 856).

## Conclusion

The seeds are spread over the file, and the re-track is not jitter: no
frame changed by less than 2 px, and nine in ten changed frames gained or
lost a box. The 41 seeds re-anchor the tracker over most of the file, which
is the second chance doing its job, and the scoped round correctly falls
back to the whole file here. Nothing is tuned on this (DECISIONS D31).
