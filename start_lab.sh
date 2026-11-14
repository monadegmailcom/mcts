#!/usr/bin/env bash
# ==============================================================================
# Pure-Rust AlphaZero Live Browser Lab, Multi-Match Arena & Play Inspector
# ==============================================================================
# Usage:
#   ./start_lab.sh               # Train Tic-Tac-Toe (~2s) & keep Browser Lab live
#   ./start_lab.sh --lab-only    # Skip training, load ttt_alphazero.safetensors & serve
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

if [[ "${MCTS_OPEN_BROWSER:-0}" == "1" ]] && command -v open >/dev/null 2>&1; then
    (sleep 1.5 && open "http://${BIND_ADDR}") &
fi

exec cargo run --release --offline "${CARGO_FEATURES[@]}" --example train_alphazero -- "${APP_ARGS[@]}"
