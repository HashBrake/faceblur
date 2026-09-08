# PyInstaller spec for FaceBlur. One folder build.
#
# Build it from the repo root:
#
#     .venv\Scripts\python.exe -m PyInstaller build\faceblur.spec --noconfirm
#
# The result is dist\FaceBlur\. Zip that folder. The user unpacks it and runs
# FaceBlur.exe. No Python is needed on the target machine.
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(SPECPATH).resolve().parent

# The three detectors, and the licences that travel with them. The face
# recogniser in models/ is evaluation only and stays out of the build.
datas = [
    (str(ROOT / "models" / "yunet.onnx"), "models"),
    (str(ROOT / "models" / "yunet_dynamic.onnx"), "models"),
    (str(ROOT / "models" / "centerface.onnx"), "models"),
    (str(ROOT / "models" / "centerface_dynamic.onnx"), "models"),
    (str(ROOT / "models" / "ultraface_dynamic.onnx"), "models"),
    (str(ROOT / "models" / "centerface.LICENSE.txt"), "models"),
    (str(ROOT / "models" / "ultraface.LICENSE.txt"), "models"),
    (str(ROOT / "models" / "README.md"), "models"),
]

# The static ffmpeg that imageio-ffmpeg carries. The user does not install
# ffmpeg, so it has to travel inside the build.
ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
datas.append((str(ffmpeg), "imageio_ffmpeg/binaries"))

hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._pybind_state",
    # The GPU path builds static graphs per input size with onnx.
    "onnx",
    "onnx.tools.update_model_dims",
    "onnx.shape_inference",
    "faceblur.batch",
    "faceblur.segments",
    # Started in a worker process by name, so the analysis cannot see them.
    "ui.worker",
    "faceblur.detect",
    "faceblur.pipeline",
    "faceblur.redact",
    "faceblur.track",
    "faceblur.video",
]

# Nothing here is used at run time, and each one adds tens of megabytes.
excludes = [
    "tkinter", "matplotlib", "scipy", "pandas", "IPython", "jupyter",
    "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia", "PySide6.QtQuick", "PySide6.QtQml",
    "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtDesigner",
]

a = Analysis(
    [str(ROOT / "faceblur_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceBlur",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # A windowed build. The worker processes are started by this same
    # executable and never show a console.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FaceBlur",
)
