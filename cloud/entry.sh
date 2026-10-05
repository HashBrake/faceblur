#!/usr/bin/env bash
# What the image runs when nothing else is asked of it, on a notebook host.
mkdir -p /run/sshd /root/.ssh
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys 2>/dev/null
/usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password || true
exec jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --ServerApp.root_dir=/
