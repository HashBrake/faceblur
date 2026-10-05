#!/usr/bin/env bash
# Everything the paid instance does, in order, unattended:
#
#   1. check the cards and the disk
#   2. install Docker, the NVIDIA container toolkit and rclone if missing
#   3. load the image and the footage from object storage
#   4. one short smoke test on card 0: CUDA in use, the encoder working
#   5. the run, every card at once, results copied to storage every 30 minutes
#   6. a final copy of the results, a DONE marker in the bucket, power off
#
# Configuration comes from /root/faceblur.env (see cloud/run.env.example).
# Start it detached so a dropped SSH session does not stop it:
#
#   nohup bash instance_run.sh > /root/faceblur_setup.log 2>&1 &       steps 1 to 4, then stop
#   nohup bash instance_run.sh run > /root/faceblur_run.log 2>&1 &     the rest, once the
#                                                                      calibration is approved
#
# Run it again after anything stops it: every step skips what is already done.
set -euo pipefail
source /root/faceblur.env

: "${S3_ENDPOINT:?}" "${S3_ACCESS_KEY:?}" "${S3_SECRET_KEY:?}" "${BUCKET:?}"
DATA=${DATA:-/data}
WORKERS=${WORKERS:-6}
RUN_ARGS=${RUN_ARGS:---quarantine}
POWEROFF=${POWEROFF:-1}

export RCLONE_CONFIG_VS_TYPE=s3 RCLONE_CONFIG_VS_PROVIDER=Other \
       RCLONE_CONFIG_VS_ENDPOINT="$S3_ENDPOINT" \
       RCLONE_CONFIG_VS_ACCESS_KEY_ID="$S3_ACCESS_KEY" \
       RCLONE_CONFIG_VS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"

say() { echo "[$(date '+%F %T')] $*"; }

# ---- 1. cards and disk
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
say "cards: $GPUS"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
mkdir -p "$DATA/in" "$DATA/out"
say "disk free on $DATA: $(df -h --output=avail "$DATA" | tail -1)"

# ---- 2. tools
if ! command -v docker >/dev/null; then
    say "installing docker"
    curl -fsSL https://get.docker.com | sh
fi
if ! docker info 2>/dev/null | grep -q nvidia; then
    say "installing the NVIDIA container toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -q && apt-get install -y -q nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi
if ! command -v rclone >/dev/null; then
    say "installing rclone"
    curl -fsSL https://rclone.org/install.sh | bash
fi

# ---- 3. image and footage
if ! docker image inspect faceblur-cuda >/dev/null 2>&1; then
    say "loading the image"
    rclone copy "vs:$BUCKET/image/faceblur-cuda.tar.gz" "$DATA/image" --progress=false
    gunzip -c "$DATA/image/faceblur-cuda.tar.gz" | docker load
fi
say "copying footage from storage"
rclone copy "vs:$BUCKET/in" "$DATA/in" --transfers 32 --checkers 32 --stats 5m --stats-one-line
say "footage: $(find "$DATA/in" -type f | wc -l) files, $(du -sh "$DATA/in" | cut -f1)"

DOCKER=(docker run --rm --gpus all --ipc=host -v "$DATA:/data" faceblur-cuda)

# ---- 4. smoke test: one sample file, card 0, check the record says CUDA and NVENC
if [ ! -f "$DATA/smoke/ok" ]; then
    say "smoke test"
    # The smallest real video, so the test takes minutes, not an hour.
    first=$(find "$DATA/in" -type f -iname '*.mp4' -size +20M -printf '%s %p\n' | sort -n | head -1 | cut -d' ' -f2-)
    rm -rf "$DATA/smoke"; mkdir -p "$DATA/smoke"
    "${DOCKER[@]}" bash -c "ffmpeg -v error -f lavfi -i color=size=256x256:rate=1 -frames:v 1 -c:v h264_nvenc -f null - \
        && echo nvenc ok \
        && python -c 'import onnxruntime as o; p=o.get_available_providers(); print(p); assert \"CUDAExecutionProvider\" in p' \
        && timeout 900 python cli.py '${first/$DATA//data}' -o /data/smoke --no-progress --workers $WORKERS $RUN_ARGS"
    rec=$(find "$DATA/smoke" -name '*.mp4.json' | head -1)
    # Every model on the card: one that fell back to the CPU would run the
    # whole job at a fraction of the speed being paid for.
    python3 -c "import json,sys; c=json.load(open('$rec'))['compute']; print(c); \
sys.exit(0 if c and all(v == 'CUDAExecutionProvider' for v in c.values()) else 1)" \
        || { say "smoke test: a model is not on CUDA, stopping"; exit 1; }
    touch "$DATA/smoke/ok"
    say "smoke test passed"
fi

# ---- 4b. calibration: every card busy at once, at several worker counts.
# Detection throughput with the whole machine loaded is what decides the
# run's length, and it cannot be measured anywhere but here: 48 cores shared
# by 8 cards may run out before the cards do.
if [ ! -f "$DATA/calibration.txt" ]; then
    first=$(find "$DATA/in" -type f -iname '*.mp4' -size +500M | head -1)
    for w in ${CALIBRATE_WORKERS:-3 5 7}; do
        say "calibration: $w processes on each of $GPUS cards"
        t0=$(date +%s)
        for g in $(seq 0 $((GPUS - 1))); do
            for k in $(seq 1 "$w"); do
                docker run --rm --gpus "device=$g" -v "$DATA:/data" faceblur-cuda \
                    python cloud/bench_detect.py "${first/$DATA//data}" 900 > /dev/null 2>&1 &
            done
        done
        wait
        secs=$(( $(date +%s) - t0 ))
        fps=$(python3 -c "print(round($GPUS * $w * 900 / $secs, 1))")
        echo "workers $w: $fps frames a second over the machine ($secs s)" | tee -a "$DATA/calibration.txt"
    done
fi
cat "$DATA/calibration.txt"
if [ "${1:-}" != run ]; then
    say "setup and calibration done. Start the run with: bash instance_run.sh run"
    exit 0
fi

# ---- 5. the run, with results copied out while it goes
( while sleep 1800; do
      rclone copy "$DATA/out" "vs:$BUCKET/out" --exclude "_shards/**" --exclude "*.part" --exclude "*.unchecked" --exclude "faceblur_*/**" --transfers 16 --stats-log-level DEBUG || true
  done ) &
SYNC=$!
say "run: $GPUS cards, $WORKERS workers each, $RUN_ARGS"
set +e
"${DOCKER[@]}" python cloud/run_shards.py /data/in /data/out --gpus "$GPUS" -- \
    --workers "$WORKERS" $RUN_ARGS
status=$?
set -e
kill $SYNC 2>/dev/null || true
say "run ended with status $status"

# ---- 6. results out, marker, power off
rclone copy "$DATA/out" "vs:$BUCKET/out" --exclude "_shards/**" --exclude "*.part" --exclude "*.unchecked" --exclude "faceblur_*/**" --transfers 32 --stats 5m --stats-one-line
cp /root/faceblur_run.log "$DATA/out/_logs/instance_run.log" 2>/dev/null || true
rclone copy "$DATA/out/_logs" "vs:$BUCKET/out/_logs"
echo "status $status $(date '+%F %T')" | rclone rcat "vs:$BUCKET/out/DONE"
say "results are in vs:$BUCKET/out"
if [ "$POWEROFF" = 1 ]; then
    say "powering off. Delete the instance in the GreenNode portal to stop paying for it."
    poweroff
fi
