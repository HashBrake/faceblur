# Models

Two ONNX face detectors ship with this repo. Both are committed so that a build
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
