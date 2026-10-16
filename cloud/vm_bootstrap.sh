#!/usr/bin/env bash
# From this PC: put one fresh vServer GPU VM to work on its share of the run.
#
#   bash cloud/vm_bootstrap.sh IP SHARD [setup|calibrate|run]...
#   bash cloud/vm_bootstrap.sh 116.118.92.82 0/8 setup run
#
# GreenNode's Ubuntu GPU image listens for SSH on port 234 and logs in as
# `stackops`, with sudo; the console takes only RSA keys. The VM must be made
# with "Skip password change at first login" ticked, or the first login asks
# for a password nobody has. Copies the credentials, the runner and the
# current stream_run.py, then runs the steps of vm_run.sh in order.
set -euo pipefail
IP=$1 SHARD=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
SSH=(ssh -p 234 -i "$HOME/.ssh/faceblur_greennode_rsa" -o StrictHostKeyChecking=accept-new
     -o ConnectTimeout=15 -o ServerAliveInterval=30 -o BatchMode=yes "stackops@$IP")
say() { echo "[$(date '+%T') $IP $SHARD] $*"; }

# GreenNode hands a new VM the floating IP of a deleted one, and ssh then
# refuses the changed host key; that showed only as "no SSH" (2026-10-16).
ssh-keygen -R "[$IP]:234" >/dev/null 2>&1 || true
for i in $(seq 1 40); do
    "${SSH[@]}" true 2>/dev/null && break
    [ "$i" = 40 ] && { say "no SSH after 10 minutes"; exit 1; }
    sleep 15
done
say "SSH up"
# The GreenNode API keys (GN_*) stay on this PC, unless the run is to delete
# its own VM when done: SELF_DELETE=1 GN_SERVER_NAME=<vm name>. The owner
# chose that on 2026-10-16 with the full-access account, knowing a VM that
# holds the keys can manage every server in the project; so it is opt-in.
if [ "${SELF_DELETE:-0}" = 1 ]; then
    : "${GN_SERVER_NAME:?SELF_DELETE=1 needs GN_SERVER_NAME, the name of this VM}"
    cat "$HERE/greennode.env" | "${SSH[@]}" 'sudo mkdir -p /root && sudo tee /root/faceblur.env >/dev/null && sudo chmod 600 /root/faceblur.env'
    "${SSH[@]}" 'sudo tee /root/gn_api.py >/dev/null' < "$HERE/gn_api.py"
else
    grep -v "^GN_" "$HERE/greennode.env" | "${SSH[@]}" 'sudo mkdir -p /root && sudo tee /root/faceblur.env >/dev/null && sudo chmod 600 /root/faceblur.env'
fi
"${SSH[@]}" 'sudo tee /root/stream_run.py >/dev/null' < "$HERE/stream_run.py"
"${SSH[@]}" 'sudo tee /root/vm_run.sh >/dev/null' < "$HERE/vm_run.sh"
# The patch pass runs this checkout's faceblur package and patch_run.py, not
# the image's older copies (see `patch` in vm_run.sh).
"${SSH[@]}" 'sudo tee /root/patch_run.py >/dev/null' < "$HERE/patch_run.py"
tar -C "$HERE/.." --exclude=__pycache__ -cf - faceblur     | "${SSH[@]}" 'sudo rm -rf /root/faceblur_pkg && sudo mkdir -p /root/faceblur_pkg && sudo tar -C /root/faceblur_pkg -xf -'
for step in "$@"; do
    say "$step"
    "${SSH[@]}" "sudo SHARD=$SHARD WORKERS=${WORKERS:-8} LANES=${LANES:-1} QUEUE=${QUEUE:-} LEASE=${LEASE:-1800} SELF_DELETE=${SELF_DELETE:-0} GN_SERVER_NAME=${GN_SERVER_NAME:-} bash /root/vm_run.sh $step" 2>&1 | sed "s/^/  /"
done
say "done: $*"
