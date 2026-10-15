# Hot storage size-vs-VMAF curve: RTX 5090 (GreenNode g7-standard-24x64-1rtx5090, driver 580.159.04), NVENC HEVC and NVENC AV1, ffmpeg n8.1.3 BtbN static build, encoder commit c82bd26; AV1 grid extended to cq 44

- Measured: 2026-10-15T22:22:55+00:00
- VMAF floor (worst file): 95
- Size target (mean, share of H.264): 0.70
- HEVC encoder: nvenc
- AV1 encoder: nvenc

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
| 20 | 4 | 0.53 | 0.48..0.57 | 99.2 | 99.6 | 188.6 |
| 22 | 4 | 0.41 | 0.37..0.44 | 98.8 | 99.3 | 191.1 |
| 24 | 4 | 0.31 | 0.29..0.33 | 98.3 | 98.9 | 191.1 |
| 26 | 4 | 0.24 | 0.22..0.25 | 97.6 | 98.3 | 188.9 |
| 28 | 4 | 0.18 | 0.17..0.19 | 96.5 | 97.3 | 188.9 |
| 30 | 4 | 0.14 | 0.13..0.15 | 95.0 | 96.0 | 189.0 |
| 32 | 4 | 0.11 | 0.10..0.12 | 92.8 | 94.1 | 189.0 |
| 34 | 4 | 0.08 | 0.07..0.09 | 90.0 | 91.4 | 188.2 |

Chosen HEVC quality: cq 28 — min VMAF 96.5, mean size 0.18 of H.264 (target 0.70 met)

## AV1

| quality | files | mean size ratio | min..max ratio | min VMAF | mean VMAF | encode fps |
|---|---|---|---|---|---|---|
| 20 | 4 | 0.68 | 0.66..0.70 | 99.7 | 99.9 | 352.0 |
| 22 | 4 | 0.55 | 0.53..0.57 | 99.6 | 99.8 | 354.0 |
| 24 | 4 | 0.45 | 0.43..0.47 | 99.5 | 99.7 | 354.0 |
| 26 | 4 | 0.37 | 0.36..0.39 | 99.3 | 99.6 | 354.2 |
| 28 | 4 | 0.30 | 0.29..0.32 | 99.1 | 99.4 | 354.2 |
| 30 | 4 | 0.24 | 0.23..0.27 | 98.8 | 99.2 | 362.1 |
| 32 | 4 | 0.20 | 0.19..0.22 | 98.4 | 98.8 | 362.5 |
| 34 | 4 | 0.16 | 0.15..0.19 | 97.8 | 98.3 | 362.0 |
| 36 | 4 | 0.14 | 0.13..0.16 | 96.9 | 97.6 | 361.0 |
| 38 | 4 | 0.11 | 0.10..0.13 | 95.7 | 96.7 | 362.4 |
| 40 | 4 | 0.09 | 0.09..0.11 | 94.3 | 95.5 | 362.5 |
| 42 | 4 | 0.08 | 0.07..0.09 | 92.6 | 94.0 | 361.9 |
| 44 | 4 | 0.07 | 0.06..0.08 | 90.6 | 92.1 | 362.5 |

Chosen AV1 quality: cq 38 — min VMAF 95.7, mean size 0.11 of H.264 (target 0.70 met)
