"""Read and write the pieces of a DF2 HDF5 file: camera packets, audio, the rest.

A camera is stored as the encoded packets of its mp4, not as frames (see
df2.py). `write_video_group` demuxes an mp4 into a camera group and
`read_video_group` remuxes the group back into an mp4 that OpenCV decodes to the
same frames, because no packet was decoded or re-encoded on the way.

The remux is the delicate part. PyAV's `add_stream` opens an encoder when the
file header is written, and libx264 then replaces the stored extradata with its
own, so the decoder gets SPS/PPS that do not match the packets and every frame
fails. `add_mux_stream` opens no encoder but has no way to set extradata, and
the stored packets are length-prefixed (avcC), so PyAV cannot find the
parameter sets in-band either. What works is `add_stream_from_template`: its
codec context counts as initialised and is never opened, so the stored avcC set
on it reaches the mp4 header unchanged. The template is a raw H.264 stream made
in memory from the SPS and PPS inside that same avcC.

The other helpers serve the release checks built later: copying every group
that is not video or audio, and proving a file has no links out and no raw
video outside the group that is allowed to hold it.
"""
from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np

from . import df2


def camera_groups(f: h5py.File) -> list[str]:
    """Camera names under IMAGE_GROUP that hold video, known names first in CAMERAS order."""
    if df2.IMAGE_GROUP not in f:
        return []
    image = f[df2.IMAGE_GROUP]
    present = [name for name, obj in image.items()
               if isinstance(obj, h5py.Group) and df2.VIDEO in obj]
    known = [name for name in df2.CAMERAS if name in present]
    return known + sorted(name for name in present if name not in df2.CAMERAS)


def write_video_group(group: h5py.Group, mp4: Path) -> int:
    """Store the first video stream of `mp4` in `group` as encoded packets. Returns the packet count."""
    with av.open(str(mp4)) as container:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        packets, pts, dts, keyframe = [], [], [], []
        for packet in container.demux(stream):
            # The demuxer ends with an empty flush packet; it is not a frame.
            if packet.size == 0 or packet.pts is None:
                continue
            packets.append(np.frombuffer(bytes(packet), dtype=np.uint8))
            pts.append(packet.pts)
            dts.append(packet.dts if packet.dts is not None else packet.pts)
            keyframe.append(packet.is_keyframe)
        time_base = stream.time_base
        rate = stream.average_rate or stream.guessed_rate
        codec, width, height, pix_fmt = ctx.name, ctx.width, ctx.height, ctx.pix_fmt
        extradata = bytes(ctx.extradata or b"")

    # No gzip: the packets are already compressed and would only cost time.
    video = group.create_dataset(df2.VIDEO, (len(packets),), dtype=h5py.vlen_dtype(np.uint8))
    for i, data in enumerate(packets):
        video[i] = data
    pts_arr = np.asarray(pts, dtype=np.int64)
    group.create_dataset(df2.PTS, data=pts_arr)
    group.create_dataset(df2.DTS, data=np.asarray(dts, dtype=np.int64))
    group.create_dataset(df2.KEYFRAME, data=np.asarray(keyframe, dtype=bool))
    # Integer arithmetic so a 90 kHz clock lands on the exact nanosecond.
    ns = pts_arr * time_base.numerator * 1_000_000_000 // time_base.denominator
    group.create_dataset(df2.TIMESTAMPS_NS, data=ns.astype(np.int64))
    # Placeholders until the partner's calibration arrives.
    group.create_dataset(df2.INTRINSICS, data=np.eye(3, dtype=np.float64))
    group.create_dataset(df2.EXTRINSICS, data=np.eye(4, dtype=np.float64))

    group.attrs[df2.ATTR_CODEC] = codec
    group.attrs[df2.ATTR_WIDTH] = width
    group.attrs[df2.ATTR_HEIGHT] = height
    group.attrs[df2.ATTR_FPS] = float(rate)
    group.attrs[df2.ATTR_TIME_BASE_NUM] = time_base.numerator
    group.attrs[df2.ATTR_TIME_BASE_DEN] = time_base.denominator
    group.attrs[df2.ATTR_PIX_FMT] = pix_fmt
    group.attrs[df2.ATTR_EXTRADATA] = np.frombuffer(extradata, dtype=np.uint8)
    # The camera files report no frame count, and there is one packet per frame.
    group.attrs[df2.ATTR_FRAME_COUNT] = len(packets)
    return len(packets)


def _avcc_parameter_sets(avcc: bytes) -> bytes:
    """The SPS and PPS inside an avcC box, as an Annex B byte stream."""
    nals, i = [], 5
    for count_mask in (0x1F, 0xFF):  # SPS count is the low 5 bits, PPS count a whole byte
        count = avcc[i] & count_mask
        i += 1
        for _ in range(count):
            length = int.from_bytes(avcc[i:i + 2], "big")
            nals.append(avcc[i + 2:i + 2 + length])
            i += 2 + length
    return b"".join(b"\x00\x00\x00\x01" + nal for nal in nals)


def read_video_group(group: h5py.Group, mp4: Path) -> None:
    """Remux the packets of a camera group into `mp4`, with no decode or re-encode."""
    attrs = group.attrs
    codec = str(attrs[df2.ATTR_CODEC])
    if codec != "h264":
        raise ValueError(f"only h264 camera groups can be remuxed, this one is {codec}")
    extradata = bytes(np.asarray(attrs[df2.ATTR_EXTRADATA], dtype=np.uint8))
    time_base = Fraction(int(attrs[df2.ATTR_TIME_BASE_NUM]), int(attrs[df2.ATTR_TIME_BASE_DEN]))

    # See the module docstring for why the stream comes from a template.
    template = av.open(io.BytesIO(_avcc_parameter_sets(extradata)), format="h264")
    try:
        with av.open(str(mp4), "w") as out:
            stream = out.add_stream_from_template(template.streams.video[0])
            ctx = stream.codec_context
            ctx.width = int(attrs[df2.ATTR_WIDTH])
            ctx.height = int(attrs[df2.ATTR_HEIGHT])
            ctx.pix_fmt = str(attrs[df2.ATTR_PIX_FMT])
            ctx.extradata = extradata
            ctx.framerate = Fraction(float(attrs[df2.ATTR_FPS])).limit_denominator(1_000_000)
            stream.time_base = time_base

            pts, dts, keyframe = group[df2.PTS][:], group[df2.DTS][:], group[df2.KEYFRAME][:]
            video = group[df2.VIDEO]
            for i in range(len(video)):
                packet = av.Packet(video[i].tobytes())
                packet.pts = int(pts[i])
                packet.dts = int(dts[i])
                packet.time_base = time_base
                packet.stream = stream
                # Without the flag the mp4 has no sync samples and players cannot seek.
                packet.is_keyframe = bool(keyframe[i])
                out.mux(packet)
    finally:
        template.close()


def copy_passthrough(src_file: h5py.File, dst_file: h5py.File,
                     skip: tuple[str, ...] = (df2.IMAGE_GROUP, df2.AUDIO_GROUP)) -> None:
    """Copy everything in `src_file` except the `skip` paths, with root attrs."""
    for key, value in src_file.attrs.items():
        dst_file.attrs[key] = value
    _copy_members(src_file, dst_file, skip)


def _member_path(group: h5py.Group, name: str) -> str:
    """`a/b/name` with no leading slash, the form df2 paths are written in."""
    return f"{group.name.rstrip('/')}/{name}".lstrip("/")


def _copy_members(src: h5py.Group, dst: h5py.Group, skip: tuple[str, ...]) -> None:
    for name in src:
        path = _member_path(src, name)
        if path in skip:
            continue
        obj = src[name]
        # A group holding a skipped path is rebuilt member by member; any other
        # member is copied whole, which keeps dtype, shape, values and attrs.
        if isinstance(obj, h5py.Group) and any(s.startswith(path + "/") for s in skip):
            child = dst.require_group(name)
            for key, value in obj.attrs.items():
                child.attrs[key] = value
            _copy_members(obj, child, skip)
        else:
            src.copy(obj, dst, name=name)


def external_links(f: h5py.File) -> list[str]:
    """Paths of every soft or external link in the file. A release must have none."""
    found: list[str] = []

    def walk(group: h5py.Group) -> None:
        for name in group:
            link = group.get(name, getlink=True)
            if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                found.append(_member_path(group, name))
            elif isinstance(group[name], h5py.Group):
                walk(group[name])

    walk(f)
    return found


def has_audio(f: h5py.File) -> bool:
    return f"{df2.AUDIO_GROUP}/{df2.AUDIO_PCM}" in f


def has_video_outside(f: h5py.File, allowed_group: str) -> list[str]:
    """Paths of datasets named VIDEO that are not under `allowed_group`."""
    found: list[str] = []
    prefix = allowed_group.strip("/") + "/"

    def visit(path: str, obj) -> None:
        if (isinstance(obj, h5py.Dataset) and path.rsplit("/", 1)[-1] == df2.VIDEO
                and not path.startswith(prefix)):
            found.append(path)

    f.visititems(visit)
    return found


def write_audio_group(f: h5py.File, wav: Path) -> None:
    """Decode a sound file to int16 PCM [samples, channels] in AUDIO_GROUP."""
    with av.open(str(wav)) as container:
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels
        # Packed s16 gives interleaved samples, so a reshape makes one row per instant.
        resampler = av.AudioResampler(format="s16", layout=stream.codec_context.layout, rate=rate)
        chunks = []
        for frame in container.decode(stream):
            chunks += [out.to_ndarray().reshape(-1, channels) for out in resampler.resample(frame)]
        chunks += [out.to_ndarray().reshape(-1, channels) for out in resampler.resample(None)]
    pcm = np.concatenate(chunks).astype(np.int16) if chunks else np.zeros((0, channels), np.int16)
    dataset = f.require_group(df2.AUDIO_GROUP).create_dataset(df2.AUDIO_PCM, data=pcm)
    dataset.attrs[df2.ATTR_SAMPLE_RATE] = rate
    dataset.attrs[df2.ATTR_CHANNELS] = channels
