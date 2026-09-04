"""Derive the dynamic-axis copies of both models. Run once, offline.

The CenterFace model in the deface wheel declares a fixed input shape of
[10, 3, 32, 32]. onnxruntime honours that, so the model cannot see a frame at
any other size. deface works around this at load time with the onnx package.
This repo does the rewrite once and commits the result, so that running the
detector needs onnxruntime only.

Needs the onnx package, which is not a dependency of this project:

    .venv/Scripts/python.exe -m pip install onnx==1.20.0
    .venv/Scripts/python.exe models/make_dynamic.py
    .venv/Scripts/python.exe -m pip uninstall -y onnx
"""
import hashlib
from pathlib import Path

import onnx
from onnx.tools.update_model_dims import update_inputs_outputs_dims

HERE = Path(__file__).resolve().parent

# Names and meanings come from deface 1.5.0, deface/centerface.py.
INPUT_DIMS = {"input.1": ["B", 3, "H", "W"]}
OUTPUT_DIMS = {
    "537": ["B", 1, "h", "w"],   # heatmap
    "538": ["B", 2, "h", "w"],   # scale
    "539": ["B", 2, "h", "w"],   # offset
    "540": ["B", 10, "h", "w"],  # landmarks
}


def derive(src, dst, in_dims, out_dims):
    model = onnx.load(str(src))
    ins = {n.name: [d.dim_value for d in n.type.tensor_type.shape.dim] for n in model.graph.input}
    outs = {n.name: [d.dim_value for d in n.type.tensor_type.shape.dim] for n in model.graph.output}
    ins.update(in_dims)
    outs.update(out_dims)
    onnx.save(update_inputs_outputs_dims(model, ins, outs), str(dst))
    for p in (src, dst):
        data = p.read_bytes()
        print(f"{p.name}  {hashlib.sha256(data).hexdigest()}  {len(data)} bytes")


def main():
    derive(HERE / "centerface.onnx", HERE / "centerface_dynamic.onnx", INPUT_DIMS, OUTPUT_DIMS)
    # YuNet: input [1, 3, 640, 640] becomes [N, 3, H, W]; every output keeps its
    # last axis and gets a free anchor count per stride.
    yunet_out = {}
    for name, last in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
        for s in (8, 16, 32):
            yunet_out[f"{name}_{s}"] = ["N", f"A{s}", last]
    derive(HERE / "yunet.onnx", HERE / "yunet_dynamic.onnx",
           {"input": ["N", 3, "H", "W"]}, yunet_out)


if __name__ == "__main__":
    main()
