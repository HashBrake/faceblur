"""FaceBlur core library. Blur every face in a video, on this PC.

The library holds no UI code and prints nothing. Progress goes through a
callback. Cancellation goes through a threading.Event.
"""
from .pipeline import (
    AuditRecord,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_STOPPED,
    output_path,
    process_video,
    sidecar_path,
)
from .settings import VIDEO_EXT, Settings, SettingsError, parse_det_sizes

__version__ = "0.1.0"

__all__ = [
    "AuditRecord", "Settings", "SettingsError", "VIDEO_EXT",
    "STATUS_DONE", "STATUS_FAILED", "STATUS_SKIPPED", "STATUS_STOPPED",
    "output_path", "parse_det_sizes", "process_video", "sidecar_path",
    "__version__",
]
