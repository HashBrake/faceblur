"""Fetch the licence plate detector and derive its dynamic copy. Run once.

The model is LPD-YuNet from opencv_zoo, models/license_plate_detection_yunet,
the same zoo `yunet.onnx` and the PP-OCR text model come from, Apache 2.0.
It is fetched from GitHub's LFS media endpoint (the raw URL returns a Git LFS
pointer, as for `yunet.onnx`), pinned to one zoo commit and one sha256, and
written unchanged as `lpd_yunet.onnx`.

That file declares a fixed input of [1, 3, 240, 320], which onnxruntime
enforces. 320x240 is too small for this model: on `tests/data/plate_cc0.jpg`
the plate shrinks to about 57x20 pixels, the score falls from 0.999 to 0.949
and two of the four corners collapse onto the middle of the plate. The zoo's
own demo never runs it at 320x240 either; it calls `setInputSize([w, h])` with
the image's own size, which OpenCV's DNN module allows because it ignores the
declared shape. The graph is fully convolutional (Conv, Relu, MaxPool,
Reshape to [-1, k], Softmax, Concat), so the derived `lpd_yunet_dynamic.onnx`
only frees the input's H and W and the outputs' row count, the way
`make_dynamic.py` does for the face models, and drops the stale shapes.

Needs only the standard library to fetch and the onnx package, which `.venv`
has, to derive:

    .venv/Scripts/python.exe models/make_plates.py
"""
import hashlib
import urllib.request
from pathlib import Path

import onnx
from onnx.tools.update_model_dims import update_inputs_outputs_dims

HERE = Path(__file__).resolve().parent
SRC = HERE / "lpd_yunet.onnx"
DST = HERE / "lpd_yunet_dynamic.onnx"

COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"   # opencv/opencv_zoo main, 2026-11-05
URL = (f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{COMMIT}/models/"
       "license_plate_detection_yunet/license_plate_detection_lpd_yunet_2023mar.onnx")
SHA256 = "6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7"
SIZE = 4146213


def fetch() -> bytes:
    request = urllib.request.Request(URL, headers={"User-Agent": "faceblur-make-plates"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def derive() -> None:
    """Free H, W and the row count N; everything else stays as the zoo made it."""
    model = onnx.load(str(SRC))
    # The exporter recorded every intermediate shape at 320x240. Left in, they
    # make onnxruntime warn on each run at any other size.
    del model.graph.value_info[:]
    model = update_inputs_outputs_dims(
        model,
        {"input": [1, 3, "H", "W"]},
        {"loc": ["N", 14], "conf": ["N", 2], "iou": ["N", 1]},
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(DST))


def main() -> None:
    data = fetch()
    digest = hashlib.sha256(data).hexdigest()
    # A commit is immutable, so a different hash means a cut short or altered
    # download, never a newer model.
    if len(data) != SIZE or digest != SHA256:
        raise SystemExit(f"{URL} gave {len(data)} bytes, sha256 {digest}; "
                         f"expected {SIZE} bytes, sha256 {SHA256}")
    SRC.write_bytes(data)
    onnx.checker.check_model(str(SRC))
    derive()
    for path in (SRC, DST):
        blob = path.read_bytes()
        print(f"{path.name}  {len(blob)} bytes  sha256 {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()
