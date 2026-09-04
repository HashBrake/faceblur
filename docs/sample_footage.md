# Sample footage

Phase 0 record of the Ego camera files the product owner supplied.
The files stay in `footage/`, which git ignores. This file records what is in them.

Probed with the ffmpeg that `imageio-ffmpeg` bundles (`ffmpeg-win-x86_64-v7.1.exe`).
`ffprobe` is not installed on this machine and `imageio-ffmpeg` does not ship it,
so stream facts come from `ffmpeg -i` and frame timing comes from the
`mkvtimestamp_v2` muxer with `-c copy`, which reads every video packet timestamp
without decoding the file.

## Summary

| File | Size | Video codec | Resolution | fps | Rate | Rotation | Audio codec | Video frames |
|---|---|---|---|---|---|---|---|---|
| ego_AZER76400FE_19700101_003939_camera_left_part0001.mp4 | 104.9 MB | h264 | 1600x1300 | 30.51 | near-CFR | none | none | 2033 |
| ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4 | 56.3 MB | h264 | 1600x1300 | 30.51 | near-CFR | none | none | 984 |
| ego_AZER76400FE_19700101_004310_camera_left_part0001.mp4 | 91.5 MB | h264 | 1600x1300 | 30.51 | near-CFR | none | none | 938 |
| ego_AZER76400FE_19700101_005035_camera_left_part0001.mp4 | 297.8 MB | h264 | 1600x1300 | 30.51 | near-CFR | none | none | 3971 |

## Per file detail

### ego_AZER76400FE_19700101_003939_camera_left_part0001.mp4

```
Duration: 00:01:07.74, start: 0.000000, bitrate: 12992 kb/s
Stream #0:0[0x1](und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(progressive), 1600x1300, 13622 kb/s, 30.51 fps, 30 tbr, 90k tbn (default)
(no audio stream)
(no rotation metadata)
```

Frame timing: median delta 33.00 ms, min 17.00, max 34.00, spread 17.00 ms, 1/2032 deltas off median by more than 1 ms

### ego_AZER76400FE_19700101_004100_camera_left_part0001.mp4

```
Duration: 00:00:32.78, start: 0.000000, bitrate: 14413 kb/s
Stream #0:0[0x1](und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(progressive), 1600x1300, 14730 kb/s, 30.51 fps, 30 tbr, 90k tbn (default)
(no audio stream)
(no rotation metadata)
```

Frame timing: median delta 33.00 ms, min 17.00, max 34.00, spread 17.00 ms, 1/983 deltas off median by more than 1 ms

### ego_AZER76400FE_19700101_004310_camera_left_part0001.mp4

```
Duration: 00:00:31.24, start: 0.000000, bitrate: 24559 kb/s
Stream #0:0[0x1](und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(progressive), 1600x1300, 25021 kb/s, 30.51 fps, 30 tbr, 90k tbn (default)
(no audio stream)
(no rotation metadata)
```

Frame timing: median delta 33.00 ms, min 17.00, max 34.00, spread 17.00 ms, 1/937 deltas off median by more than 1 ms

### ego_AZER76400FE_19700101_005035_camera_left_part0001.mp4

```
Duration: 00:02:12.33, start: 0.000000, bitrate: 18877 kb/s
Stream #0:0[0x1](und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(progressive), 1600x1300, 19351 kb/s, 30.51 fps, 30 tbr, 90k tbn (default)
(no audio stream)
(no rotation metadata)
```

Frame timing: median delta 33.00 ms, min 17.00, max 34.00, spread 17.00 ms, 1/3970 deltas off median by more than 1 ms

