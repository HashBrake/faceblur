"""FaceBlur core library. Blur every confirmed face in a video, on this PC.

The library holds no UI code and prints nothing. Progress goes through a
callback. Cancellation goes through a threading.Event.
"""
from .detect import Detection, DetectorBank, filter_candidates
from .pipeline import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_STOPPED,
    AuditRecord,
    Plan,
    output_path,
    plan,
    process_video,
    sidecar_path,
)
from .settings import VIDEO_EXT, Settings, SettingsError, parse_det_sizes
from .track import Tracker

__version__ = "0.2.0"

__all__ = [
    "AuditRecord", "Detection", "DetectorBank", "Plan", "Settings", "SettingsError",
    "Tracker", "VIDEO_EXT", "STATUS_DONE", "STATUS_FAILED", "STATUS_SKIPPED",
    "STATUS_STOPPED", "filter_candidates", "output_path", "parse_det_sizes",
    "plan", "process_video", "sidecar_path", "__version__",
]
