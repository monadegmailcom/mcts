#!/usr/bin/env bash
# ==============================================================================
# Pure-Rust AlphaZero Live Browser Lab, Multi-Match Arena & Play Inspector
# ==============================================================================
# Usage:
#   ./start_lab.sh               # Start Browser Lab in IDLE state (Tic-Tac-Toe & Ultimate TTT)
#   ./start_lab.sh --metal       # Enable Apple Silicon GPU (Metal) acceleration
#   ./start_lab.sh --cuda        # Enable NVIDIA GPU (CUDA) acceleration
#   ./start_lab.sh --metal --tch # Also build the libtorch backend (selectable in the lab)
#
# Environment Variables:
#   MCTS_BIND_ADDR=127.0.0.1:8080 ./start_lab.sh
#   MCTS_OPEN_BROWSER=1 ./start_lab.sh
#   LIBTORCH=/opt/libtorch ./start_lab.sh --tch          # standalone libtorch 2.9 (Docker/Linux)
#   MCTS_TORCH_PYTHON=/path/venv/bin/python ./start_lab.sh --tch   # use a PyTorch 2.9 install
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIND_ADDR="${MCTS_BIND_ADDR:-127.0.0.1:8080}"
export MCTS_BIND_ADDR="$BIND_ADDR"

FEATURES=()
APP_ARGS=("--serve")
USE_TCH=0

for arg in "$@"; do
    case "$arg" in
        --lab-only)
            APP_ARGS=("--lab-only")
            ;;
        --metal)
            FEATURES+=("metal")
            ;;
        --cuda)
            FEATURES+=("cuda")
            ;;
        --tch)
            FEATURES+=("tch")
            USE_TCH=1
            ;;
        *)
            APP_ARGS+=("$arg")
            ;;
    esac
done

# libtorch for the `tch` feature: a standalone LIBTORCH, or the torch package of a Python install.
if [[ "$USE_TCH" == "1" && -z "${LIBTORCH:-}" ]]; then
    TORCH_PY="${MCTS_TORCH_PYTHON:-python3}"
    if ! "$TORCH_PY" -c "import torch" >/dev/null 2>&1; then
        echo "--tch needs libtorch: set LIBTORCH=/path/to/libtorch, or MCTS_TORCH_PYTHON to a python with torch 2.9 installed." >&2
        exit 1
    fi
    export LIBTORCH_USE_PYTORCH=1
    # tch 0.22 checks for torch 2.9.0 exactly; any 2.9.x patch release is ABI compatible.
    export LIBTORCH_BYPASS_VERSION_CHECK=1
    export PATH="$(dirname "$TORCH_PY"):$PATH"
    echo "libtorch from: $("$TORCH_PY" -c 'import torch, os; print(torch.__version__, os.path.dirname(torch.__file__))')"
fi

CARGO_FEATURES=()
if [[ ${#FEATURES[@]} -gt 0 ]]; then
    CARGO_FEATURES=("--features" "$(IFS=,; echo "${FEATURES[*]}")")
fi

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
