# Privacy desensitization module — scope expansion research (2026-11-05)

Input for the autonomous build of the expanded module. Sources: the two partner
PDFs in `new files/` (pipeline collaboration overview, DF1/DF2 HDF5 data
layout), the other developer's message about the five-container chain, and
four web research passes run on 2026-11-05. Numbers below are "as found", not
independently verified; items marked [?] are unconfirmed.

## 1. What the partner pipeline needs from this module

The post-processing chain is five Docker containers, each reading one
recording as HDF5 and writing its result beside it:

1. camera folder -> HDF5 (partner)
2. **privacy desensitization: blur, strip audio** (this module, our team)
3. sub-task segmentation (VNG)
4. captioning (VNG)
5. LeRobot v3 dataset (partner)

Stage 3 expects `episode.h5`: "the blurred, silent recording". Interface table
in the PDF: input HDF5 (video + IMU + SLAM); output HDF5 "preserved raw video,
generated new video". Sub-items in the responsibility tree under Privacy
Desensitization: face detection & blurring, licence plate detection,
screen/text detection, voice muting, audit logging.

HDF5 layout (DF2, the newer one): `dataset/observation/image/<camera>/` with
cameras `rgb_ego_center_left|right`, `rgb_ego_side_left|right`,
`ir_ego_left|right`, `rgb_wrist_left|right` (+ `_up`/`_down` variants planned);
`observation/timeline/`, `observation/audio/` (optional), `observation/state/`
(body, left_hand, right_hand), `env/objects`, `accessories/imu/...`.
Each camera group holds video, timestamps and intrinsics/extrinsics.
The partner has not yet said whether video is stored as encoded bytes or
decoded frames; the sample recording will settle it.

## 2. Decisions taken for the build (owner delegated them)

- **D-A Two outputs, not one shared file.** The module leaves the input HDF5
  untouched (it is the restricted raw master) and writes `episode.h5` beside
  it: de-identified video for every camera, no audio, IMU/SLAM/state/timeline
  copied through, plus the audit record. A `--spec-layout` flag additionally
  writes the single-file form from the PDF (a copy of the input with the
  de-identified video added under a sibling group) for integration tests with
  VNG. A test asserts the release file holds no raw video, no audio and no
  external links.
- **D-B Audio: strip from the release.** Silero VAD (MIT) speech segments go
  into the audit record so the Language team knows what was removed. Raw
  audio stays only in the untouched input. No wearer/bystander split: generic
  diarization scores ~47 % DER on Ego4D, too wrong to rely on.
- **D-C Obfuscation strength.** Gaussian blur is reversible by current
  diffusion attacks (Revelio 2025, Privacy Blur Dec 2025); pixelation is
  safe only at very coarse blocks. Keep the current six-blocks-across-a-face
  pixelation only if a re-ID check after a deblur attempt still fails;
  otherwise solid fill. Plates and text: solid fill.
- **D-D Test data.** Build a converter that packs the four `footage/` sample
  mp4s into a DF2-style HDF5 fixture (encoded H.264 packets + per-frame
  timestamps). The group paths live in one config module so the layout can
  be swapped when the partner's sample and schema checker arrive.
- **D-E Local only.** Everything runs on this PC (DirectML, Docker Desktop
  with the local GPU). Cloud GPU runs are BLOCKED tasks for the owner.
- **D-F Licences.** Ship nothing AGPL/GPL (Ultralytics YOLOv8/11/12, YOLOE,
  YOLO-World) and nothing non-commercial (SCRFD weights, DiariZen weights,
  SAM 3 until its licence is read). Every new weight file gets a
  `models/<name>.LICENSE.txt` like the existing ones.

## 3. Findings per sub-item

### Faces (existing: YuNet + CenterFace + UltraFace consensus, SFace re-ID check)
- Weak spot is known: 65 % recall on <32 px and motion-blurred faces.
- Cheapest gains: bidirectional gap fill in the existing IoU tracker; a head
  box fallback where no face fires (pose- or CrowdHuman-derived head,
  arXiv 2301.04243 idea) for back-of-head and far people.
- Extra voter candidates: EgoBlur face (Meta, arXiv 2308.13093, weights
  Apache 2.0 on HF `projectaria/EgoBlur`, TorchScript; ONNX export untested).
  SCRFD only with a commercial licence. No 2025-26 WIDER-Hard SOTA found.
- IR cameras: no NIR face benchmark found; expect to fine-tune on own IR
  frames with pseudo-labels from the aligned RGB view.
- Generative replacement (DeepPrivacy2, diffusion swaps): heavy, licence
  risk, fails on tiny faces. Not for this pipeline.
- Evaluation: keep the label-free harness; add a few hundred hand-labelled
  hard frames per camera type, recall by size bucket; run re-ID after an
  attempted deblur, not only on the mask.

### Licence plates (new)
- EgoBlur plate model (same release, Apache 2.0, egocentric training data).
  Best base; check motorbike plates.
- Vietnamese data: Roboflow "vietnamese-motobike-plate" and
  "vietnamese-car-license-plate" sets (licence unstated), MAPR 2018 bike set
  (3k images, link dead). Chinese: CCPD (ECCV 2018, 250k+, MIT [?]).
- Train with a permissive stack (YOLOX, RT-DETR, Detectron2), never
  Ultralytics.

### Screens (existing YOLOX-tiny COCO tv/laptop/phone; eval has an OWL oracle)
- Open-vocabulary options with safe licences: OWLv2 (Apache 2.0), Florence-2
  (MIT). Grounding DINO 1.5 is API-only; YOLOE AGPL; YOLO-World GPL; SAM 3
  custom licence, cloud-only candidate at best. No dedicated display dataset
  found; EgoLife/EgoMonth mask screens with big models plus human review.

### Text (existing PP-OCRv3 DB detector, masking not wired)
- PP-OCRv5 (PaddleOCR 3.0, arXiv 2507.05595, Apache 2.0) is the current
  detector; Vietnamese via the Latin model [?].
- PII classification after OCR (GLiNER-PII, Presidio) is English-centric and
  licence-inconsistent; nothing good for Vietnamese/Chinese. Policy stays
  "mask all text outside the handled zone", no reading.

### Documents, ID cards, badges
- No permissive off-the-shelf detector. MIDV-500/2020 are not VN/CN ids; the
  72k-image 2024 set is CC-BY-NC (evaluation only). Route: open-vocab
  prompts + the text detector.

### Voice
- Silero VAD: MIT, ~2 MB, <1 ms per 30 ms chunk on CPU, multilingual.
- Diarization: pyannote community-1 (CC-BY-4.0, gated) 46.8 % DER on Ego4D;
  Sortformer 4spk-v2 (CC-BY-4.0) English-heavy; DiariZen weights CC-BY-NC.
- ASR if ever needed: Qwen3-ASR 1.7B (Apache 2.0, VN+ZH+EN), PhoWhisper
  (Apache 2.0, Vietnamese).
- Speaker anonymization (VoicePrivacy 2024) is leaky and English-tested.
- No egocentric dataset found that mutes bystanders automatically; they rely
  on consent. Hence D-B.

### HDF5, Docker, audit
- Store de-identified video as encoded HEVC/H.264 packets (vlen uint8 per
  packet or GOP) + int64 pts + keyframe index; LeRobot v3 wants mp4 anyway so
  stage 5 can copy bytes. Short GOPs (~1 s). Write once, close; no SWMR.
- HDF5 has no usable encryption filter; protect the raw master at the
  storage layer. Hence D-A.
- Audit: append-only JSONL with a hash chain; record sha256 of input, output,
  every model, thresholds, image digest; per-frame manifest with counts/boxes
  per class (face, plate, screen, text) and VAD segments; optional PROV-O
  terms. C2PA not needed internally.
- Docker: `nvidia/cuda:12.x-cudnn-runtime` base, onnxruntime-gpu pinned
  (1.21–1.26 built on CUDA 12.8/cuDNN 9), weights baked in with sha256 check
  at start, `HF_HUB_OFFLINE=1`, non-root, `/in` read-only, HEALTHCHECK runs a
  one-frame inference, interface
  `docker run --gpus all -v in:/in:ro -v out:/out <image>`.
- Contract tests: JSON Schema for the audit record and a layout descriptor
  for the HDF5; validate real output with `jsonschema` in CI.

### Legal (context only, not a build task)
- Vietnam: Law 91/2025/QH15 in force 2026-01-01; Decree 356/2025/ND-CP
  replaced Decree 13/2023. Biometric data is sensitive; de-identified data
  is treated as not personal [?].
- China: facial recognition security measures (2025-06-01), GB/T 42460-2023
  de-identification effectiveness guide.
- EU: EDPB draft anonymisation guidelines 02/2026; masked egocentric video is
  pseudonymised, not anonymous (gait, voice, interiors remain). Treat the
  release as personal data; the audit trail is what an auditor will read.
