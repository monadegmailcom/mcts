#!/usr/bin/env bash
# Backend comparison on a CUDA machine (e.g. a RunPod pod from a pytorch/pytorch *-devel image
# with Rust installed). Runs the same networks through candle, PyTorch and tch-rs (libtorch).
#   bash experiments/bench_cuda.sh            # all three
# Output lines: infer = batched forward incl. upload/readback (pos/s), train = fwd+bwd+AdamW (ms).
set -euo pipefail
cd "$(dirname "$0")/.."
SECS="${SECS:-2}"

echo "== candle (cuda)";        cargo run --release --features cuda  --example bench_nn -- devices=gpu secs="$SECS"
echo "== candle (cuda+cudnn)";  cargo run --release --features cudnn --example bench_nn -- devices=gpu secs="$SECS" || echo "(cudnn build failed: cuDNN missing?)"
echo "== pytorch";              python experiments/torch_bench.py devices=cuda secs="$SECS" bn=1 compile=1
echo "== tch-rs (libtorch)"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__) + "/lib")')"
( cd experiments/tch_bench && LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 cargo build --release \
  && LD_LIBRARY_PATH="$TORCH_LIB" ./target/release/tch_bench devices=cuda secs="$SECS" )
