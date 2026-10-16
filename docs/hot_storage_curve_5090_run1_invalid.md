# Hot storage size-vs-VMAF curve: RTX 5090 (GreenNode g7-standard-24x64-1rtx5090, driver 580.159.04), NVENC HEVC and NVENC AV1, ffmpeg n8.1.3 BtbN static build

> **Invalid run, kept for the record.** Measured with encoder commit 3b59cc9, which wrote exact 30 fps timestamps instead of passing the camera's 30.0085 fps through; the 132 s file drifted against its source and scored VMAF ≈ 87.9 at every quality. Fixed in T05a (c82bd26); the valid curve is `hot_storage_curve_5090.md`.

- Measured: 2026-10-15T21:13:21+00:00
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
| 20 (failed a frame or duration check) | 4 | 0.35 | 0.29..0.39 | 87.8 | 95.9 | 188.4 |
| 22 (failed a frame or duration check) | 4 | 0.35 | 0.29..0.39 | 87.8 | 95.9 | 188.9 |
| 24 (failed a frame or duration check) | 4 | 0.31 | 0.28..0.33 | 87.8 | 95.9 | 191.1 |
| 26 (failed a frame or duration check) | 4 | 0.24 | 0.22..0.25 | 87.6 | 95.3 | 189.0 |
| 28 (failed a frame or duration check) | 4 | 0.18 | 0.17..0.19 | 87.3 | 94.3 | 188.9 |
| 30 (failed a frame or duration check) | 4 | 0.14 | 0.13..0.15 | 86.6 | 93.0 | 188.9 |
| 32 (failed a frame or duration check) | 4 | 0.11 | 0.10..0.12 | 85.3 | 91.2 | 188.7 |
| 34 (failed a frame or duration check) | 4 | 0.08 | 0.07..0.09 | 83.1 | 88.6 | 188.3 |

Chosen HEVC quality: no quality reaches 95; best achievable is cq 22 — min VMAF 87.8, mean size 0.35 of H.264 (target 0.70 met)

## AV1

| quality | files | mean size ratio | min..max ratio | min VMAF | mean VMAF | encode fps |
|---|---|---|---|---|---|---|
| 20 (failed a frame or duration check) | 4 | 0.68 | 0.66..0.70 | 87.9 | 96.9 | 351.8 |
| 22 (failed a frame or duration check) | 4 | 0.55 | 0.54..0.57 | 87.9 | 96.8 | 354.2 |
| 24 (failed a frame or duration check) | 4 | 0.42 | 0.37..0.45 | 87.9 | 96.7 | 353.8 |
| 26 (failed a frame or duration check) | 4 | 0.37 | 0.35..0.39 | 87.9 | 96.6 | 353.7 |
| 28 (failed a frame or duration check) | 4 | 0.30 | 0.29..0.32 | 87.9 | 96.4 | 354.2 |
| 30 (failed a frame or duration check) | 4 | 0.24 | 0.23..0.27 | 87.9 | 96.2 | 353.7 |
| 32 (failed a frame or duration check) | 4 | 0.20 | 0.19..0.22 | 87.8 | 95.8 | 353.9 |
| 34 (failed a frame or duration check) | 4 | 0.16 | 0.15..0.19 | 87.7 | 95.3 | 362.2 |

Chosen AV1 quality: no quality reaches 95; best achievable is cq 20 — min VMAF 87.9, mean size 0.68 of H.264 (target 0.70 met)
