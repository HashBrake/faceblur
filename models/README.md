# Models

Three ONNX face detectors ship with this repo. Both are committed so that a build
never downloads a model. `tests/test_models.py` checks the sha256 of each file.

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
