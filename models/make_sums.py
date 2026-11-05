"""Write models/SHA256SUMS: one `<sha256>  models/<file>` line per model, sha256sum's format.

The privacy image runs `sha256sum -c models/SHA256SUMS` before it touches any
input, so a weight file that changed after the build stops the container
instead of masking with a model nobody reviewed. sha256sum reads the names
from the folder it runs in, the repo root (/app in the image), not from the
sums file's own folder, hence the models/ prefix. Run it again after adding
or replacing a model:

    python models/make_sums.py
"""
import hashlib
from pathlib import Path

MODELS = Path(__file__).resolve().parent


def main() -> None:
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  models/{p.name}\n"
             for p in sorted(MODELS.glob("*.onnx"))]
    # LF only: sha256sum on Linux reads a CR as part of the file name.
    (MODELS / "SHA256SUMS").write_bytes("".join(lines).encode("ascii"))


if __name__ == "__main__":
    main()
