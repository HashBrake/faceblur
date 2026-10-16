#!/usr/bin/env bash
# Start this machine's share of the run on a plain GreenNode vServer GPU VM.
#
# The RTX 5090 is offered only as a vServer VM, not as a notebook, so the
# image runs under Docker on an Ubuntu host with the NVIDIA driver installed.
# Run from this PC with cloud/vm_bootstrap.sh IP SHARD STEP..., which copies
# the credentials and this script in over SSH (port 234, user stackops, the
# RSA key faceblur_greennode_rsa) and runs the steps under sudo.
#
# `setup` installs Docker and the NVIDIA container toolkit, pulls the image
# and checks that the card runs CUDA and the encoder inside it. `calibrate`
# times the detection pass at several process counts on the card. `run`
# starts stream_run.py in a detached container that survives a dropped SSH
# session; a restart skips what is finished. It takes SHARD=k/n, or QUEUE=1
# with no SHARD, and passes LEASE, SELF_DELETE and GN_SERVER_NAME through.
# stream_run.py is mounted over the image's copy, which predates the SHARD
# and disk room commits, and so is gn_api.py when it was copied in.
set -euo pipefail
set -a; . /root/faceblur.env; set +a
IMAGE=${IMAGE:-vcr.vngcloud.vn/110156-faceblur/faceblur-cuda:3}
WORK=${WORK:-/var/faceblur}
say() { echo "[$(date '+%F %T')] $*"; }
mkdir -p "$WORK"

case "${1:-}" in
setup)
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
        || { say "no NVIDIA driver on this host"; exit 1; }
    if ! command -v docker >/dev/null; then
        say "installing docker"
        curl -fsSL https://get.docker.com | sh
    fi
    if ! docker info 2>/dev/null | grep -qi nvidia; then
        say "installing the NVIDIA container toolkit"
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update -q && apt-get install -y -q nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
    fi
    echo "$VCR_SECRET" | docker login "${VCR_REGISTRY:-vcr.vngcloud.vn}" -u "$VCR_USER" --password-stdin >/dev/null
    docker pull -q "$IMAGE"
    say "checking CUDA and the encoder in the image"
    docker run --rm --gpus all --entrypoint bash "$IMAGE" -c "
        ffmpeg -v error -f lavfi -i color=size=256x256:rate=1 -frames:v 1 -c:v h264_nvenc -f null - && echo 'nvenc ok'
        python -c \"import onnxruntime as o; s=o.InferenceSession('/app/models/yunet.onnx', providers=['CUDAExecutionProvider']); print('onnxruntime on', s.get_providers()[0]); assert s.get_providers()[0]=='CUDAExecutionProvider'\""
    say "setup done; free on $WORK: $(df -h --output=avail "$WORK" | tail -1)"
    ;;
calibrate)
    # The smallest video over 200 MB in the bucket, 900 frames, N processes at once.
    docker run --rm --env-file /root/faceblur.env -v "$WORK:/work" --entrypoint bash "$IMAGE" -c '
        export RCLONE_CONFIG_VS_TYPE=s3 RCLONE_CONFIG_VS_PROVIDER=Other RCLONE_CONFIG_VS_ENDPOINT=$S3_ENDPOINT RCLONE_CONFIG_VS_ACCESS_KEY_ID=$S3_ACCESS_KEY RCLONE_CONFIG_VS_SECRET_ACCESS_KEY=$S3_SECRET_KEY
        [ -f /work/calib.mp4 ] || { f=$(rclone lsf -R --format sp --include "*.mp4" vs:$BUCKET/in | sort -n | awk -F";" "\$1>2e8{print \$2; exit}"); echo "sample: $f"; rclone copyto "vs:$BUCKET/in/$f" /work/calib.mp4; }'
    for n in ${CALIBRATE:-4 6 8 10 12}; do
        t0=$(date +%s)
        for i in $(seq "$n"); do
            docker run --rm --gpus all -v "$WORK:/work" --entrypoint python "$IMAGE" \
                cloud/bench_detect.py /work/calib.mp4 900 > "$WORK/calib_${n}_${i}.log" 2>&1 &
        done
        wait
        secs=$(( $(date +%s) - t0 ))
        echo "$n processes: $(python3 -c "print(round($n * 900 / $secs, 1))") frames a second ($secs s)" | tee -a "$WORK/calibration.txt"
    done
    ;;
run)
    if [ "${QUEUE:-0}" != "1" ]; then : "${SHARD:?SHARD=k/n, k from 0, or QUEUE=1}"; fi
    # The image's gn_api.py predates the library the self-delete hook loads.
    mounts=()
    if [ -f /root/gn_api.py ]; then mounts=(-v /root/gn_api.py:/app/cloud/gn_api.py:ro); fi
    docker rm -f faceblur 2>/dev/null || true
    docker run -d --name faceblur --restart on-failure:3 --gpus all --ipc=host \
        --env-file /root/faceblur.env -e SHARD="${SHARD:-0/1}" -e WORKERS="${WORKERS:-5}" -e LANES="${LANES:-1}" \
        -e RUN_ARGS="${RUN_ARGS:---quarantine --second-chance 1}" -e WORK=/work \
        -e QUEUE="${QUEUE:-0}" -e LEASE="${LEASE:-1800}" \
        -e SELF_DELETE="${SELF_DELETE:-0}" -e GN_SERVER_NAME="${GN_SERVER_NAME:-}" \
        -v "$WORK:/work" -v /root/stream_run.py:/app/cloud/stream_run.py:ro "${mounts[@]}" \
        --entrypoint python "$IMAGE" cloud/stream_run.py
    if [ "${QUEUE:-0}" = "1" ]; then mode="queue mode"; else mode="share $SHARD"; fi
    say "started $mode; log: tail -f $WORK/stream_run.log"
    ;;
*)
    echo "usage: vm_run.sh setup | calibrate | run"; exit 2 ;;
esac
