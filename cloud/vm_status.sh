#!/usr/bin/env bash
# One line per VM of the run: slice, progress line, GPU busy, load.
#   bash cloud/vm_status.sh            (the fleet of 2026-10-14)
#   bash cloud/vm_status.sh IP:k/n ...
FLEET=${*:-116.118.92.106:0/8 116.118.92.84:2/8 116.118.92.109:4/8 116.118.92.113:5/8 116.118.92.111:6/8 116.118.92.114:7/8}
for spec in $FLEET; do
    ip=${spec%%:*}; shard=${spec#*:}
    ( out=$(ssh -p 234 -i "$HOME/.ssh/faceblur_greennode_rsa" -o BatchMode=yes -o ConnectTimeout=10 \
            stackops@"$ip" 'sudo grep -E "progress|finished:|STOPPED|failed|no card" /var/faceblur/stream_run.log | tail -1;
                            nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader;
                            cut -d" " -f1 /proc/loadavg; sudo docker ps --format "{{.Status}}" | head -1' 2>/dev/null | tr '\n' '|')
      echo "$shard $ip | ${out:-UNREACHABLE}" ) &
done | sort
wait
