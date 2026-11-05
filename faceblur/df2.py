"""Where things live in a DF2 file: every group path and dataset name, in one place.

The partner exchanges recordings as HDF5 in a layout they call DF2. Their PDF
gives the group tree but not what sits inside a camera group, so the names
below for the video datasets and attrs are this project's choice. Nothing else
in the code spells a path out: when the partner's real sample arrives and the
layout differs, this is the only file that changes.

Video is kept as the encoded H.264 packets exactly as demuxed from the camera
mp4, one variable-length uint8 element per packet. That keeps the file the size
of the mp4 and lets the original be remuxed bit for bit, rather than storing
decoded frames that would be a hundred times larger.
"""
from __future__ import annotations

ROOT = "dataset"

META_GROUP = f"{ROOT}/meta"
OBSERVATION_GROUP = f"{ROOT}/observation"
TIMELINE_GROUP = f"{OBSERVATION_GROUP}/timeline"
IMAGE_GROUP = f"{OBSERVATION_GROUP}/image"
AUDIO_GROUP = f"{OBSERVATION_GROUP}/audio"
STATE_GROUP = f"{OBSERVATION_GROUP}/state"
ENV_GROUP = f"{OBSERVATION_GROUP}/env"
ACCESSORIES_GROUP = f"{ROOT}/accessories"

# The release form built later puts the de-identified cameras beside the raw
# ones under this sibling, so a reader of the spec layout finds both.
RELEASE_IMAGE_GROUP = f"{OBSERVATION_GROUP}/image_deidentified"

# In the partner's order.
CAMERAS = (
    "rgb_wrist_left", "rgb_wrist_right",
    "rgb_wrist_left_up", "rgb_wrist_left_down",
    "rgb_wrist_right_up", "rgb_wrist_right_down",
    "rgb_ego_center_left", "rgb_ego_center_right",
    "rgb_ego_side_left", "rgb_ego_side_right",
    "ir_ego_left", "ir_ego_right",
)

# Names `pack` gives its mp4s when none are named: the camera files on hand are
# the head-worn ones, so CAMERAS from the ego centre pair onwards, then the rest.
PACK_ORDER = CAMERAS[CAMERAS.index("rgb_ego_center_left"):] + CAMERAS[:CAMERAS.index("rgb_ego_center_left")]

# Every leaf group of the tree that is neither video nor audio. The module
# never interprets these; it copies them to the output untouched.
PASSTHROUGH_GROUPS = (
    f"{META_GROUP}/task/sop",
    f"{META_GROUP}/task/illustrations",
    f"{META_GROUP}/scene",
    f"{META_GROUP}/annotation/language/self_speech",
    f"{META_GROUP}/annotation/language/external_speech",
    f"{META_GROUP}/annotation/language/prompt",
    f"{STATE_GROUP}/body/root",
    f"{STATE_GROUP}/body/joints",
    f"{STATE_GROUP}/body/motion",
    f"{STATE_GROUP}/left_hand/handpose",
    f"{STATE_GROUP}/left_hand/joints",
    f"{STATE_GROUP}/left_hand/tactile",
    f"{STATE_GROUP}/right_hand/handpose",
    f"{STATE_GROUP}/right_hand/joints",
    f"{STATE_GROUP}/right_hand/tactile",
    f"{ENV_GROUP}/objects/obj_1",
    f"{ENV_GROUP}/environment_map",
    f"{ACCESSORIES_GROUP}/imu/imu_ego",
    f"{ACCESSORIES_GROUP}/imu/imu_wrist_left",
    f"{ACCESSORIES_GROUP}/imu/imu_wrist_right",
    f"{ACCESSORIES_GROUP}/emg/emg_left",
    f"{ACCESSORIES_GROUP}/emg/emg_right",
    f"{ACCESSORIES_GROUP}/mocap",
)

# Datasets inside a camera group.
VIDEO = "video"                  # vlen uint8, one encoded packet per element
PTS = "pts"                      # int64, in the stream time base
DTS = "dts"                      # int64, in the stream time base
KEYFRAME = "keyframe"            # bool
TIMESTAMPS_NS = "timestamps_ns"  # int64, pts in nanoseconds, one per packet
INTRINSICS = "intrinsics"        # 3x3 float64
EXTRINSICS = "extrinsics"        # 4x4 float64

# Attrs on a camera group.
ATTR_CODEC = "codec"
ATTR_WIDTH = "width"
ATTR_HEIGHT = "height"
ATTR_FPS = "fps"
ATTR_TIME_BASE_NUM = "time_base_num"
ATTR_TIME_BASE_DEN = "time_base_den"
ATTR_PIX_FMT = "pix_fmt"
ATTR_EXTRADATA = "extradata"     # uint8 array, the avcC box for H.264
ATTR_FRAME_COUNT = "frame_count"

# The audio dataset and its attrs.
AUDIO_PCM = "pcm"                # int16 [samples, channels]
ATTR_SAMPLE_RATE = "sample_rate"
ATTR_CHANNELS = "channels"

# The timeline dataset.
TIMELINE_TIMESTAMPS_NS = "timestamps_ns"


def camera_path(name: str) -> str:
    """The group path of one camera."""
    return f"{IMAGE_GROUP}/{name}"
