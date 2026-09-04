# FaceBlur report

What was built, how well it works, how fast it runs, on what, and what was
left out on purpose. Every number here was produced by the code in this
repository with no human labels and no human judgement in the loop. Dated
2026-09-05.

## 1. The task

Video in, video out. Every face in the output is destroyed inside the smallest
region that hides identity: an ellipse over the eyes, nose and mouth. Nothing
else changes: hands, cards, screens, bodies, hair and background keep their
original pixels, so the output stays usable as training data. No labelling,
no review step, no parameter a person has to pick per video.

The footage is from an Ego wearable camera pointed at card collectors:
1600x1300, about 30 fps, H.264, no audio, near constant frame rate. Four
sample files, 264 seconds in total, sit in `footage/` (gitignored).

## 2. Hardware and software the numbers come from

| | |
|---|---|
| CPU | Intel Core i7-12700K, 12 cores / 20 threads |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 3070, 8 GB, driver 595.79 (DirectML, NVENC) |
| OS | Windows 11 Enterprise 10.0.26200 |
| Python | 3.12.8 |
| Inference | onnxruntime-directml 1.24.4, onnx 1.17.0 |
| Video | ffmpeg 7.1 from imageio-ffmpeg 0.6.0, h264_nvenc for encoding |
| Detectors | YuNet (opencv_zoo, Apache 2.0), CenterFace (deface 1.5.0, MIT) |

Exact package versions are pinned in `requirements.txt`; model hashes are in
`models/README.md` and in every audit record.

## 3. Quality and accuracy

### 3.1 How it is measured without labels

There are no ground-truth labels for this footage and the process must not
need any, so `eval/` measures five things automatically:

- **Consensus recall.** Three unrelated detectors (YuNet, CenterFace,
  MediaPipe BlazeFace) run over the video; a face any two agree on is a
  pseudo-label. Recall is the share of pseudo-faces whose identity ellipse is
  inside the mask.
- **Synthetic recall.** Face crops of known position are pasted into real
  frames of the same video at various sizes, sharp and motion blurred. The
  truth is exact. This catches the misses consensus cannot see.
- **Off-face masking.** Masked pixels where no detector sees a face. The
  "strict" variant counts only consensus faces as face, an upper bound.
- **Hand damage.** MediaPipe Hands regions; the share of hand pixels touched.
- **Continuity.** Once a face is confirmed, how often it stays covered while
  any detector still sees it.

`eval/sweep.py` tries a grid of settings and keeps only those that pass the
gates (off-face at most 0.3 % mean and 3 % on any frame, hands at most 0.1 %),
then picks the best mean of consensus and synthetic recall, with continuity as
the tie-break. The chosen defaults are the ones in `faceblur/settings.py`.
Full tables: `docs/precision_report.md`.

### 3.2 Results on the sample footage

Measured on `004310` (938 frames, the busiest file, 215 tracks). Where two
numbers are given they are the sweep's measurement and the README's earlier
measurement of the same build.

| Measure | First build | Shipped build |
|---|---|---|
| Share of the frame destroyed, mean | 33 % | 1.5–1.7 % |
| Share of the frame destroyed, worst frame | 93 % | 3.5–3.8 % |
| Masked pixels where no detector sees a face, mean | 29.7 % | 0.16–0.30 % |
| Hand pixels touched | 82 % | 0.00 % |
| Faces the detectors agree on, covered | 100 % | 98.5–99.0 % |
| Confirmed faces kept covered while still visible | – | 99.0–99.6 % |

Synthetic recall, faces of known position pasted into real frames:

| Faces | Covered |
|---|---|
| 64 px and larger, sharp | 89 % |
| 64 px and larger, motion blurred | 73 % |
| 32 to 63 px | 94 % |
| under 32 px | 46 % |
| all | 83 % |

Mask area over identity-ellipse area: 2.0. The destroyed region is about
twice the area of the minimal ellipse; that margin keeps a moving face covered
and gives the soft edge.

Across the four sample files, from the audit records: 0.25–1.7 % of each frame
destroyed on average, at most 4.8 % on any frame, zero frames over the 5 %
budget.

### 3.3 What the misses are

Small faces (under 32 px on a 1600 px frame: someone across the room) and
strongly motion blurred faces. The detectors themselves find a 24 px face
about half the time at any threshold; that is a model limit, not a threshold
choice. The first build reached 100 % on the agreed faces by masking a third of
every frame. This build trades a few points of recall on the smallest faces
for leaving everything else intact, which was the stated goal.

### 3.4 Entry frames

Faces entering the field of view used to show for one or two frames before
the mask started. Now a confirmed track is extended backwards through weaker
sightings, and its mask reaches six frames before the first sighting and two
after the last, following the face's motion and growing 5 % per frame. The
continuity metric and `tests/test_entry.py` cover this.

## 4. Speed

### 4.1 Per frame

| Path | Detection per 1600x1300 frame |
|---|---|
| CPU, first precision build | 700 ms |
| CPU, shipped build | 350 ms |
| GPU (RTX 3070, DirectML), shipped build, one process | 100 ms |

Detection dominates. Decoding costs about 3.5 ms per frame, masking and NVENC
encoding about 10 ms per frame per worker, copying clean stretches nothing.

### 4.2 The four sample files, end to end

Same defaults, same machine. "Window" is the desktop UI on its pool; "CLI" is
`cli.py footage -o ... --workers 10` in one process pool. Wall time per video
includes probing, detection, tracking, writing, joining and verification.

| File | Video | Source | Window wall | CLI wall | CLI detect | CLI write | Copied | Output | Ratio (CLI) |
|---|---|---|---|---|---|---|---|---|---|
| `003939` | 68 s, 2033 frames | 110 MB | 75 s | 69 s | 43 s | 25 s | 150 frames | 204 MB | 1.01 : 1 |
| `004100` | 33 s, 984 frames | 59 MB | no record (see note) | 41 s | 31 s | 10 s | 120 frames | 103 MB | 1.26 : 1 |
| `004310` | 31 s, 938 frames | 96 MB | 77 s | 83 s | 66 s | 17 s | 0 frames | 129 MB | 2.66 : 1 |
| `005035` | 132 s, 3971 frames | 312 MB | 182 s | 173 s | 134 s | 39 s | 60 frames | 482 MB | 1.31 : 1 |
| **all four** | **264 s** | | **333 s + 1 file** | **367 s batch** | | | | | **1.39 : 1** |

Window and CLI run the same pool and the same code, so the per-video times
match within noise. The window run's `004100` produced a good file but no
audit record because of the bookkeeping bug fixed on 2026-09-05 (STATE.md);
its time was not recorded. The CLI batch total (367 s, 6 min 8 s wall)
includes starting the 10 worker processes and loading the models once.

Output size: the outputs are larger than the sources because they are
encoded at a higher quality (NVENC cq 16) than the camera used, and because
the encoded pieces carry no B-frames (+5.5 % against the same encode with
B-frames, measured on `003939`: 203.6 MB against 193.0 MB).

### 4.3 What would make it faster

- A CUDA build on a data-centre GPU: the detection graph is the same ONNX,
  onnxruntime-gpu replaces onnxruntime-directml. Not measured here.
- More worker processes only help until the GPU saturates; 10 workers on an
  RTX 3070 already share one card.
- `--stride 2` halves detection. The sweep measures what that costs in recall
  on the footage and picks it only if it stays inside the gates; on this
  footage it did not.
- NVDEC decoding (`--hwaccel cuda`) was measured no faster than CPU decoding
  here and stays off.

## 5. Assumptions

- Footage like the samples: a wearable camera, faces 20–300 px, near constant
  frame rate, H.264 in MP4. Other containers and codecs are read by ffmpeg but
  were not tested.
- No audio in the samples. When audio exists it is copied through untouched.
- A face needs to be seen twice by YuNet and confirmed once by CenterFace
  before any pixel is touched. A face that appears for one frame only is not
  masked; on this footage that is the trade that keeps hands untouched.
- The identity region is the eyes-nose-mouth ellipse. Hair, ears and the
  outline of the head are left. This is deliberate and is what "bare minimum"
  was taken to mean.
- Pixelation (six blocks across the face, from a shrunk copy) destroys
  identity. A plain Gaussian blur is invertible in principle and is not offered
  on its own.
- Windows with a DirectX 12 GPU or a CPU. Linux and macOS are not tested.
- The evaluation's pseudo-labels are only as good as the union of three
  detectors. A face all three miss is invisible to the measurement. The
  synthetic test is there to bound that.

## 6. Done on purpose

- **Two architectures must agree** before a mask is drawn: YuNet finds,
  CenterFace confirms on a crop. One detector at a low threshold is what
  blurred hands in the first build.
- **Size and shape gates**: at most 15 % of the long side, roughly square. A
  whole screen cannot be a face.
- **Tracker** with confirmation (seen twice, close together), interpolation
  over gaps up to five frames, backward continuation, motion tails.
- **Ellipse mask rotated to the eye line**, feathered, with a per-face block
  size so a small face is destroyed as thoroughly as a large one.
- **Only the region inside the ellipse changes**; the rest of the frame is the
  decoded source pixel. Stretches with no mask are stream-copied, byte for
  byte.
- **Automatic parameter selection** by sweep with gates: no person picks a
  threshold.
- **Audit record per output** with how much of each frame was destroyed,
  flagged frames, every setting, model hashes, compute provider and timing.
- **Verified joins**: frame count, forward timestamps, length against the
  source, clean demux; a failed join falls back to encoding the whole file.
- **Encoded pieces carry no B-frames** so they join exactly with copied source
  pieces; copies are only made from sources with no reordering (see 8).
- **Every dependency pinned**, real footage and every output folder gitignored.
- **GPU through DirectML**, so any DX12 card works without a CUDA install;
  static ONNX graphs are built per input size because DirectML mis-computes
  the dynamic ones.

## 7. Not done on purpose

- **No installer, no code signing** (Phase 6 of the plan was not requested).
- **No cloud runner.** `faceblur.batch.run_video` takes a `submit` callable and
  detection runs on independent frame ranges, so one can be written without
  changing the pipeline. Not built.
- **No larger detector.** RetinaFace, SCRFD or YOLO-face variants would raise
  small-face recall at 3–10x the compute. The evaluation harness is where to
  test one; the defaults were not changed without measurement.
- **No number plate, text, or body redaction.**
- **No manual review UI.** The requirement was no human intervention; the
  audit record is the substitute.
- **NVDEC** off by default, measured no faster here.
- **`--no-verify` and `--engine both`** exist for recall at any cost, and are
  documented as blurring hands (7.7 % of hand pixels on this footage).

## 8. Known limits and open issues

- Faces under 32 px: about half covered. Motion blurred faces: about three
  quarters.
- A source with B-frames is encoded whole rather than partly copied. Still
  correct, slower for videos with few faces.
- `-bf 0` on the encoder makes output files larger at the same quality; see
  the size column in 4.2.
- Outputs are H.264 at crf 12 / NVENC cq 16 and are larger than the sources
  (13 Mb/s in), because the camera encoded at a lower quality than the setting
  chosen to keep training data intact.
- The window runs videos one after another; a queue of many short videos
  would go faster with videos in parallel. Not needed for the sample sizes.

## 9. Reproduce

```
.venv\Scripts\python.exe -m pytest tests                      # 741 tests
.venv\Scripts\python.exe cli.py footage -o footage_blurred --workers 10
.venv-eval\Scripts\python.exe eval\oracle_mediapipe.py VIDEO --stride 5
.venv\Scripts\python.exe -m eval.consensus VIDEO --stride 10
.venv\Scripts\python.exe -m eval.sweep VIDEO                  # writes docs/precision_report.md
```
