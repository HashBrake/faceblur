"""Time the detection pass alone, in one process, on a range of frames.

    python cloud/bench_detect.py VIDEO [FRAMES] [cuda option=value ...]

Says how many milliseconds a frame the detectors and the hand pass cost on
this machine, which is the number that decides how many cards a run needs.
Options after the frame count go to the CUDA provider, to compare settings.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faceblur.detect as detect  # noqa: E402
from faceblur import hands  # noqa: E402
from faceblur.batch import detect_job  # noqa: E402
from faceblur.segments import frame_times  # noqa: E402
from faceblur.settings import Settings  # noqa: E402
from faceblur.video import probe  # noqa: E402

src = Path(sys.argv[1])
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
opts = dict(a.split("=", 1) for a in sys.argv[3:])
if opts:
    plain = detect.providers_for

    def with_opts(device):
        return [("CUDAExecutionProvider", opts) if p == "CUDAExecutionProvider" else p
                for p in plain(device)]
    detect.providers_for = with_opts
    hands.providers_for = with_opts

info, times = probe(src), frame_times(src)
s = Settings()
detect_job({"src": str(src), "info": info, "times": times, "settings": s, "start": 0, "end": 30})
t = time.time()
detect_job({"src": str(src), "info": info, "times": times, "settings": s, "start": 30,
            "end": 30 + frames})
ms = (time.time() - t) / frames * 1000
print(f"{ms:.1f} ms a frame over {frames} frames, options {opts or 'default'}")
