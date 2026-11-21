#!/usr/bin/env bash
# Container start: an SSH server, then the Browser Lab in the background (see mcts-lab).
# RunPod passes the account's SSH public key in PUBLIC_KEY. MCTS_AUTOSTART=0 skips the lab.
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    install -d -m 700 /root/.ssh
    touch /root/.ssh/authorized_keys
    grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys || echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# SSH sessions start with a clean environment; give them the container's.
printenv | grep -E '^(PATH|LD_LIBRARY_PATH|NVIDIA_[A-Z_]+|CUDA_[A-Z_]+|MCTS_[A-Z_]+|RUNPOD_[A-Z_]+)=' > /etc/environment || true

ssh-keygen -A
mkdir -p /run/sshd
/usr/sbin/sshd

if [[ "${MCTS_AUTOSTART:-1}" == "1" ]]; then
    mcts-lab
fi
exec sleep infinity
