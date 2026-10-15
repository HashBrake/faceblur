# Hot storage size-vs-VMAF curve: RTX 3070, local PC, NVENC HEVC (ffmpeg 7.1 gyan.dev via imageio-ffmpeg); AV1 via libaom-av1 software, smallest file only (this GPU has no AV1 NVENC)

- Measured: 2026-10-15T23:15:11+00:00
- VMAF floor (worst file): 95
- Size target (mean, share of H.264): 0.70
- HEVC encoder: nvenc
- AV1 encoder: software

## Inputs

| file | size (MB) | frames | duration (s) |
|---|---|---|---|
| ego_AZER76400FE_19700101_003939_camera_left_part0001_blurred.mp4 | 207.7 | 2033 | 67.7 |
| ego_AZER76400FE_19700101_004100_camera_left_part0001_blurred.mp4 | 105.7 | 984 | 32.8 |
| ego_AZER76400FE_19700101_004310_camera_left_part0001_blurred.mp4 | 128.9 | 938 | 31.2 |
| ego_AZER76400FE_19700101_005035_camera_left_part0001_blurred.mp4 | 484.6 | 3971 | 132.3 |

## HEVC

| quality | files | mean size ratio | min..max ratio | min VMAF | mean VMAF | encode fps |
|---|---|---|---|---|---|---|
| 20 | 4 | 0.54 | 0.49..0.58 | 99.1 | 99.5 | 152.1 |
| 22 | 4 | 0.42 | 0.38..0.44 | 98.8 | 99.3 | 159.2 |
| 24 | 4 | 0.32 | 0.29..0.34 | 98.2 | 98.8 | 159.9 |
| 26 | 4 | 0.25 | 0.22..0.26 | 97.4 | 98.2 | 160.3 |
| 28 | 4 | 0.19 | 0.17..0.20 | 96.2 | 97.1 | 160.7 |
| 30 | 4 | 0.15 | 0.13..0.15 | 94.5 | 95.6 | 160.7 |
| 32 | 4 | 0.11 | 0.10..0.12 | 92.2 | 93.4 | 160.5 |
| 34 | 4 | 0.09 | 0.08..0.10 | 89.2 | 90.6 | 160.6 |

Chosen HEVC quality: cq 28 — min VMAF 96.2, mean size 0.19 of H.264 (target 0.70 met)

## AV1

| quality | files | mean size ratio | min..max ratio | min VMAF | mean VMAF | encode fps |
|---|---|---|---|---|---|---|
| 24 | 1 | 0.13 | 0.13..0.13 | 97.0 | 97.0 | 1.7 |
| 30 | 1 | 0.08 | 0.08..0.08 | 94.4 | 94.4 | 2.0 |
| 36 | 1 | 0.05 | 0.05..0.05 | 90.2 | 90.2 | 2.3 |

Chosen AV1 quality: crf 24 — min VMAF 97.0, mean size 0.13 of H.264 (target 0.70 met)
