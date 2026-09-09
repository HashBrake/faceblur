"""Derive the two hand models from the MediaPipe wheel. Run once, offline.

MediaPipe ships them as TensorFlow Lite, and MediaPipe itself only runs in
`.venv-eval`. The output-side check runs in the main environment, which has
onnxruntime and nothing else, so both models are converted once here and the
results committed. Nothing in the app imports TensorFlow.

The conversion needs TensorFlow and tf2onnx, neither of which is a dependency
of this project. Build them a throwaway environment:

    py -m venv convert
    convert\Scripts\python.exe -m pip install tensorflow-cpu tf2onnx onnx
    convert\Scripts\python.exe models/make_hands.py \
        --mediapipe .venv-eval/Lib/site-packages/mediapipe

What the rewrite does after tf2onnx has run:

- gives the input and every output a name that says what it carries, so
  `faceblur/hands.py` needs no knowledge of what the converter called them.
  The two [1, 1] outputs of the landmark model cannot be told apart by shape,
  so they are taken in the order the tflite file lists them, which is the
  order MediaPipe's own graph splits them in: landmarks, presence,
  handedness, world landmarks,
- fixes every dimension. Both graphs are batch 1 at their own square size and
  there is no reason to want another: the check asks about one crop at a
  time, and a fixed graph is what DirectML wants anyway,
- strips the value_info the converter left behind, which describes shapes
  that no longer exist after the rename.

Both inputs stay NHWC, as the tflite models have them. `faceblur/hands.py`
feeds them that way rather than paying for a transpose the graph would only
undo.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent

# Each model: the tflite under the mediapipe package, the file to write, and
# what to call the input and the outputs. Outputs are named by shape where the
# shape is unique, and otherwise by their position in the tflite file.
MODELS = {
    "palm_detection.onnx": {
        "tflite": "modules/palm_detection/palm_detection_full.tflite",
        "outputs": ["boxes", "scores"],          # [1, 2016, 18], [1, 2016, 1]
    },
    "hand_landmark.onnx": {
        "tflite": "modules/hand_landmark/hand_landmark_full.tflite",
        "outputs": ["landmarks", "presence", "handedness", "world"],
    },
}


def rename(model: onnx.ModelProto, old: str, new: str) -> None:
    """Rename one tensor everywhere it is mentioned."""
    if old == new:
        return
    for value in list(model.graph.input) + list(model.graph.output):
        if value.name == old:
            value.name = new
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new


def tidy(src: Path, dst: Path, names: list[str]) -> None:
    """Rename the input and outputs, drop stale shapes, and check the result.

    tf2onnx writes the outputs in the order the tflite file lists them, which
    is the order `names` is in.
    """
    model = onnx.load(str(src))
    if len(model.graph.input) != 1:
        raise SystemExit(f"expected one input, got {[i.name for i in model.graph.input]}")
    if len(model.graph.output) != len(names):
        raise SystemExit(f"expected {len(names)} outputs, got "
                         f"{[o.name for o in model.graph.output]}")
    rename(model, model.graph.input[0].name, "input")
    for value, name in zip(list(model.graph.output), names):
        rename(model, value.name, name)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    shapes = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim]
              for o in model.graph.output}
    data = dst.read_bytes()
    print(f"{dst.name}  {hashlib.sha256(data).hexdigest()}  {len(data)} bytes  {shapes}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mediapipe", required=True, help="the installed mediapipe package")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()
    package = Path(args.mediapipe)

    for out_name, spec in MODELS.items():
        tflite = package / spec["tflite"]
        if not tflite.is_file():
            raise SystemExit(f"no such file: {tflite}")
        dst = HERE / out_name
        raw = dst.with_suffix(".raw.onnx")
        subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--opset", str(args.opset),
                        "--tflite", str(tflite), "--output", str(raw)], check=True)
        tidy(raw, dst, spec["outputs"])
        raw.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
