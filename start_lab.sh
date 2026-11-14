#!/usr/bin/env bash
# ==============================================================================
# Pure-Rust AlphaZero Live Browser Lab, Multi-Match Arena & Play Inspector
# ==============================================================================
# Usage:
#   ./start_lab.sh               # Start Browser Lab in IDLE state (Tic-Tac-Toe & Ultimate TTT)
#   ./start_lab.sh --metal       # Enable Apple Silicon GPU (Metal) acceleration
#   ./start_lab.sh --cuda        # Enable NVIDIA GPU (CUDA) acceleration
#
# Environment Variables:
#   MCTS_BIND_ADDR=127.0.0.1:8080 ./start_lab.sh
#   MCTS_OPEN_BROWSER=1 ./start_lab.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIND_ADDR="${MCTS_BIND_ADDR:-127.0.0.1:8080}"
export MCTS_BIND_ADDR="$BIND_ADDR"

CARGO_FEATURES=()
APP_ARGS=("--serve")

for arg in "$@"; do
    case "$arg" in
        --lab-only)
            APP_ARGS=("--lab-only")
            ;;
        --metal)
            CARGO_FEATURES=("--features" "metal")
            ;;
        --cuda)
            CARGO_FEATURES=("--features" "cuda")
            ;;
        *)
            APP_ARGS+=("$arg")
            ;;
    esac
done

echo "================================================================"
echo " Starting Pure-Rust AlphaZero Browser Lab"
echo " URL:  http://${BIND_ADDR}"
echo " Mode: ${APP_ARGS[*]}"
echo "================================================================"

if command -v pgrep >/dev/null 2>&1 && pgrep -f "target/.*/examples/train_alphazero" >/dev/null 2>&1; then
    echo "Stopping previous train_alphazero instance..."
    pkill -9 -f "target/.*/examples/train_alphazero" 2>/dev/null || true
    sleep 0.3
fi

if [[ "${MCTS_OPEN_BROWSER:-0}" == "1" ]] && command -v open >/dev/null 2>&1; then
    (sleep 1.5 && open "http://${BIND_ADDR}") &
fi

exec cargo run --release --offline ${CARGO_FEATURES[@]+"${CARGO_FEATURES[@]}"} --example train_alphazero -- "${APP_ARGS[@]}"
