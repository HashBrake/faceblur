#!/usr/bin/env bash
# Start the whole run on a GreenNode notebook instance running faceblur-cuda.
#
# The notebook host starts the container with its own command, so this is run
# over SSH once the instance is Active, after the credentials are copied in:
#
#   cat cloud/greennode.env | ssh -p PORT root@HOST 'cat > /root/faceblur.env && chmod 600 /root/faceblur.env'
#   ssh -p PORT root@HOST 'bash -l /app/cloud/notebook_run.sh'
#
# It returns at once; the run goes on under nohup. Progress is in
# /var/faceblur/stream_run.log here and in out/_logs/ in the bucket.
#
# Settings of the run decided on 2026-10-06: the check, quarantine and one
# second chance round; five workers a card, measured best on a 4090 with six
# cores. The work folder is on the root disk: block storage over 20 GB fails to
# provision on this account (2026-10-07), and stream_run.py waits for room.
set -euo pipefail
set -a; . /root/faceblur.env; set +a
export WORK=${WORK:-/var/faceblur} WORKERS=${WORKERS:-5}
export RUN_ARGS=${RUN_ARGS:---quarantine --second-chance 1}
mkdir -p "$WORK"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
cd /app
nohup python cloud/stream_run.py > "$WORK/nohup.out" 2>&1 &
echo "started, pid $!; log: tail -f $WORK/stream_run.log"
