# Models

Three ONNX face detectors ship with this repo, and one face recogniser that
only the evaluation harness uses. All four are committed so that a build never
downloads a model. `tests/test_models.py` checks the sha256 of each file.

## yunet.onnx

| Field | Value |
|---|---|
| Original name | face_detection_yunet_2023mar.onnx |
| Source | opencv_zoo, models/face_detection_yunet |
| URL | https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx |
| Licence | Apache 2.0 |
| Size | 232589 bytes |
| sha256 | 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 |

The GitHub raw URL returns a Git LFS pointer. Fetch the file from the LFS media
endpoint above. This repo runs the model through `cv2.FaceDetectorYN`.

## centerface.onnx

| Field | Value |
|---|---|
| Original name | centerface.onnx |
| Source | the deface package, version 1.5.0, file deface/centerface.onnx |
| URL | https://pypi.org/project/deface/1.5.0/ |
| Licence | MIT, see centerface.LICENSE.txt |
| Size | 7304518 bytes |
| sha256 | 09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc |

Taken from the deface wheel on 2026-09-04. The build plan calls this file
`centerface.bnmerged.onnx`. The wheel names it `centerface.onnx`. It is the same
model. This repo runs it through `onnxruntime` and does not import deface.

## yunet_dynamic.onnx and centerface_dynamic.onnx

Derived copies with dynamic input and output axes, made once, offline, by
`models/make_dynamic.py`. The originals declare fixed shapes (YuNet 640x640,
CenterFace 10x3x32x32) that onnxruntime enforces. OpenCV ignores the declared
shape, which is why the first build could use the original YuNet file. This
build runs both models through onnxruntime so they can use the GPU.

| File | sha256 | Size |
|---|---|---|
| yunet_dynamic.onnx | c0aa2a665abc3daba84ab666ee6b15352852128c84c5267897ad18e590ac466f | 232622 |
| centerface_dynamic.onnx | e50c58b64599f94a343ac1dc58fa4902635c2e1939b316afb39142c39c99b22e | 7304533 |

## ultraface.onnx

| Field | Value |
|---|---|
| Original name | version-RFB-320.onnx |
| Source | Linzaer, Ultra-Light-Fast-Generic-Face-Detector-1MB, models/onnx |
| URL | https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx |
| Licence | MIT, see ultraface.LICENSE.txt |
| Size | 1270727 bytes |
| sha256 | 34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017 |

Taken on 2026-09-07. The third detector family. It is asked only about a
candidate that CenterFace is unsure of, on the same crop. Input 320x240 RGB,
`(x - 127) / 128`; outputs `scores` [N, 4420, 2] and `boxes` [N, 4420, 4] in
normalised x1, y1, x2, y2. No landmarks.

`ultraface_dynamic.onnx` is the derived copy from `models/make_dynamic.py`:
batch 1 becomes N, the input keeps 320x240 (the anchor layout is built for
it), and the weights the opset 9 export listed as graph inputs are moved out.

| File | sha256 | Size |
|---|---|---|
| ultraface_dynamic.onnx | 043540a05268e7b6392a949ec708a707da66151a32bcb5dbb6f9e685bc393051 | 1259728 |

## sface.onnx

| Field | Value |
|---|---|
| Original name | face_recognition_sface_2021dec.onnx |
| Source | opencv_zoo, models/face_recognition_sface |
| URL | https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx |
| Licence | Apache 2.0, see sface.LICENSE.txt |
| Size | 38696353 bytes |
| sha256 | 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79 |

A MobileFaceNet trained with the SFace loss, taken on 2026-09-08. It embeds an
aligned 112x112 face into 128 numbers; two embeddings of one person point the
same way. `eval/reid.py` uses it to ask whether the blurred copy still
identifies anybody, and it needs the 5 landmarks YuNet already produces, which
is why this model rather than another: they are made to work together and both
are Apache 2.0.

**Not part of the app.** It is evaluation only, and the packaged build does not
carry it. A tool that redacts faces has no business shipping a face recogniser.

## palm_detection.onnx and hand_landmark.onnx

| Field | Value |
|---|---|
| Original names | palm_detection_full.tflite, hand_landmark_full.tflite |
| Source | the mediapipe package, version 0.10.21, `mediapipe/modules/` |
| URL | https://pypi.org/project/mediapipe/0.10.21/ |
| Licence | Apache 2.0, see palm_detection.LICENSE.txt and hand_landmark.LICENSE.txt |

| File | sha256 | Size |
|---|---|---|
| palm_detection.onnx | 03440370cb546a4fdcff45524300a5a8293afc84cfaf44c0b112d209386ffa30 | 4605970 |
| hand_landmark.onnx | 00e6f22f25156220974589e012562ccce84b3d4b043690ac6085e8701261a9df | 10914627 |

Taken on 2026-09-09 and converted by `models/make_hands.py`, which needs
TensorFlow and tf2onnx in a throwaway environment and is run once, offline.
The conversion is faithful: on random input the ONNX and the tflite agree to
1.5e-4 on every raw output.

They are what lets the output-side check tell the wearer's own hand from a
face the run missed. MediaPipe itself only runs in `.venv-eval`; the check
runs in the main environment, which has onnxruntime and nothing else, so the
models are converted rather than imported. The pipeline does not use them:
they load only when `--check-output` or `--quarantine` is on.

`palm_detection.onnx` takes one 192x192 RGB image in [0, 1], NHWC, and returns
`boxes` [1, 2016, 18] and `scores` [1, 2016, 1] against MediaPipe's own anchor
layout, which `faceblur/hands.py` reproduces. `hand_landmark.onnx` takes one
224x224 crop of the rectangle a palm box implies and returns `landmarks`,
`presence`, `handedness` and `world`; only `presence` is used, and it is
already a probability — a sigmoid over it would compress black (0.007) and a
hand (0.89) into 0.50 and 0.71 and leave nothing to threshold.

## yolox_tiny.onnx

| Field | Value |
|---|---|
| Original name | yolox_tiny.onnx |
| Source | Megvii, YOLOX, release 0.1.1rc0 |
| URL | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx |
| Licence | Apache 2.0, see yolox.LICENSE.txt |
| Size | 20219662 bytes |
| sha256 | 427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7 |

Taken on 2026-09-10, for the screen class. Licence came first in the choice:
most YOLO derivatives in common use are AGPL and this project cannot ship
them. Of the permissive detectors that remained, this is the smallest whose
decode matches one already in the repo, so `faceblur/screens.py` could be
checked against a frame rather than trusted.

Input `images` [1, 3, 416, 416], BGR, 0 to 255, NCHW, **no mean and no
standard deviation**: normalising it the ImageNet way drops the best score on
a frame holding a television from 0.85 to 0.004. The frame is letterboxed to
the top left on grey (114), as YOLOX's own preprocessing does. Output
`output` [1, 3549, 85] is undecoded: four box numbers, an objectness, then 80
class scores, against the anchor free grid of strides 8, 16 and 32 that gives
52x52 + 26x26 + 13x13 = 3549 rows.

It is a COCO detector and this project wants three of its eighty classes: tv
(62), laptop (63) and cell phone (67). `screen_labels` chooses among those
three and nothing else in the code hard-codes them.
