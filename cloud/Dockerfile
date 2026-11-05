# FaceBlur for a Linux machine with NVIDIA cards. Built and tested on the
# owner's PC through Docker Desktop before any paid instance exists.
#
# Two images from one file, sharing every layer up to the code:
#
#   privacy (the default target): the de-identification module on its own.
#   Not root, no network needed, weights checked against models/SHA256SUMS at
#   every start, and one command: every DF2 file in /in, de-identified into /out.
#
#     docker build -f cloud/Dockerfile -t faceblur-privacy .
#     docker run --gpus all -v <in>:/in:ro -v <out>:/out faceblur-privacy
#
#   notebook: the GreenNode notebook image, JupyterLab and sshd as root, the way
#   the cloud runs have used it.
#
#     docker build -f cloud/Dockerfile --target notebook -t faceblur-cuda .
#
# cuDNN 9 and CUDA 12 are what onnxruntime-gpu 1.24 is built against.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base

# "video" exposes the card's encoder to ffmpeg, which "compute" alone does not.
# NVIDIA_DISABLE_REQUIRE: the base image refuses to start on a host driver
# older than CUDA 12.8 asks for. CUDA 12 runs on any 12.x driver through minor
# version compatibility, the host's driver is not ours to choose, and the
# smoke test proves CUDA works instead.
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    NVIDIA_DISABLE_REQUIRE=1 \
    PYTHONUNBUFFERED=1

# Ubuntu's ffmpeg is built with NVENC; the static one imageio-ffmpeg ships for
# Linux is not, and the encoder would fall back to libx264 on the CPU.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /venv
COPY cloud/requirements-cuda.txt /tmp/requirements-cuda.txt
RUN /venv/bin/pip install --no-cache-dir -r /tmp/requirements-cuda.txt

ENV PATH=/venv/bin:$PATH \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

WORKDIR /app
COPY models models
COPY faceblur faceblur
COPY cli.py pyproject.toml ./
COPY tests tests
COPY cloud cloud


FROM base AS notebook

RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server unzip curl \
 && rm -rf /var/lib/apt/lists/*

# rclone pinned from its own release, not Ubuntu's 1.60: newer S3 handling, and
# `rclone serve s3` lets the local test stand in for vStorage.
ARG RCLONE=v1.75.1
ARG RCLONE_SHA256=982b5aa772841168f8e380f139e9e787b2a105403e32b94da8676a0e1c0a13ab
RUN curl -fsSLo /tmp/rclone.zip https://downloads.rclone.org/$RCLONE/rclone-$RCLONE-linux-amd64.zip  && echo "$RCLONE_SHA256  /tmp/rclone.zip" | sha256sum -c -  && unzip -q -j /tmp/rclone.zip '*/rclone' -d /usr/local/bin && rm /tmp/rclone.zip

# GreenNode's notebook instances run a container image and reach it through
# JupyterLab on 8888, so the image carries one. It is only the way in: the
# pipeline does not import it. Its own dependencies are left to pip.
RUN /venv/bin/pip install --no-cache-dir jupyterlab==4.6.4

# A notebook host starts the container with its own command, and an SSH login
# shell does not see ENV, so login shells get the same settings.
RUN printf '%s\n' 'export PATH=/venv/bin:$PATH' 'export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg' \
        'export NVIDIA_DRIVER_CAPABILITIES=compute,video,utility' 'cd /app 2>/dev/null' \
        > /etc/profile.d/faceblur.sh

# Default: an SSH server if the platform hands in a key, and JupyterLab. A
# platform that starts the container with its own command replaces this.
EXPOSE 8888 22
CMD ["bash", "/app/cloud/entry.sh"]


# Last, so a plain `docker build` makes this one.
FROM base AS privacy

# Every model is in the image, so nothing may reach for a hub at run time: a
# library that tries fails at once instead of fetching an unreviewed file.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    PYTHONUNBUFFERED=1

# A fixed uid, so a host can give it write access to the output folder. It
# owns /out and nothing else: the code and weights stay root's and read-only.
RUN useradd --create-home --uid 10001 faceblur && mkdir -p /in /out && chown faceblur /out
COPY --chown=faceblur cloud/privacy_entry.sh /app/cloud/privacy_entry.sh
RUN chmod +x /app/cloud/privacy_entry.sh

USER faceblur
VOLUME ["/in", "/out"]

# One frame through every masking model: a broken provider or a missing weight
# shows as unhealthy instead of an hour into a run.
HEALTHCHECK --interval=5m --timeout=120s --start-period=60s --retries=1 \
    CMD ["python", "-m", "faceblur.h5", "selfcheck"]
CMD ["bash", "/app/cloud/privacy_entry.sh"]
