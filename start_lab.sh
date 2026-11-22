#!/usr/bin/env bash
# ==============================================================================
# Pure-Rust AlphaZero Live Browser Lab, Multi-Match Arena & Play
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
#
# libtorch for --tch (a C++ library; no Python involved):
#   ./start_lab.sh --tch                                # ./libtorch, downloaded on first use
#   LIBTORCH=/opt/libtorch ./start_lab.sh --tch         # a libtorch 2.9 installed elsewhere
#   MCTS_LIBTORCH_VARIANT=cu126 ./start_lab.sh --cuda --tch   # Linux: CUDA build (default cu128)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIND_ADDR="${MCTS_BIND_ADDR:-127.0.0.1:8080}"
export MCTS_BIND_ADDR="$BIND_ADDR"

FEATURES=()
APP_ARGS=("--serve")
USE_TCH=0
USE_CUDA=0

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
            USE_CUDA=1
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

# libtorch release for --tch; must match the tch crate in Cargo.toml (tch 0.22 <-> libtorch 2.9).
LIBTORCH_VERSION=2.9.1

# Official libtorch download for this platform.
libtorch_url() {
    local base="https://download.pytorch.org/libtorch"
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) echo "$base/cpu/libtorch-macos-arm64-$LIBTORCH_VERSION.zip" ;;
        Linux-x86_64)
            local variant="cpu"
            [[ "$USE_CUDA" == "1" ]] && variant="${MCTS_LIBTORCH_VARIANT:-cu128}"
            echo "$base/$variant/libtorch-shared-with-deps-$LIBTORCH_VERSION%2B$variant.zip" ;;
        *) return 1 ;;
    esac
}

# ./libtorch, downloaded on first use and again when the wanted release changes.
ensure_libtorch() {
    local dir="$SCRIPT_DIR/libtorch" url tmp
    if ! url="$(libtorch_url)"; then
        echo "--tch: no libtorch download for $(uname -sm); set LIBTORCH=/path/to/libtorch (version $LIBTORCH_VERSION)." >&2
        exit 1
    fi
    [[ "$(cat "$dir/.download-url" 2>/dev/null)" == "$url" ]] && return
    echo "Downloading libtorch $LIBTORCH_VERSION (one-time): $url" >&2
    tmp="$(mktemp -d)"
    curl -fL --progress-bar -o "$tmp/libtorch.zip" "$url"
    unzip -q "$tmp/libtorch.zip" -d "$tmp"
    rm -rf "$dir"
    mv "$tmp/libtorch" "$dir"
    echo "$url" > "$dir/.download-url"
    rm -rf "$tmp"
}

if [[ "$USE_TCH" == "1" ]]; then
    if [[ -z "${LIBTORCH:-}" ]]; then
        ensure_libtorch
        export LIBTORCH="$SCRIPT_DIR/libtorch"
    fi
    # tch 0.22 checks for libtorch 2.9.0 exactly; any 2.9.x patch release is ABI compatible.
    export LIBTORCH_BYPASS_VERSION_CHECK=1
    echo "libtorch: $LIBTORCH ($(cat "$LIBTORCH/build-version" 2>/dev/null || echo 'unknown version'))"
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

# Stop a lab still holding our port from an earlier start; labs on other ports keep running.
PORT="${BIND_ADDR##*:}"
if command -v lsof >/dev/null 2>&1; then
    for pid in $(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null); do
        if [[ "$(ps -o comm= -p "$pid")" == *train_alphazero ]]; then
            echo "Stopping previous train_alphazero instance on port $PORT..."
            kill -9 "$pid"
            sleep 0.3
        fi
    done
fi

if [[ "${MCTS_OPEN_BROWSER:-0}" == "1" ]] && command -v open >/dev/null 2>&1; then
    (sleep 1.5 && open "http://${BIND_ADDR}") &
fi

exec cargo run --release --offline ${CARGO_FEATURES[@]+"${CARGO_FEATURES[@]}"} --example train_alphazero -- "${APP_ARGS[@]}"
