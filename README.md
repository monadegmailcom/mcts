# `mcts` — Generic N-Player MCTS, AlphaZero & Minimax Engine in Rust

A ground-up Rust evolution of [`beat-it`](https://github.com/monadegmailcom/beat-it), designed for mathematical search purity, cache-line density, and compile-time generality across **1-player puzzles**, **2-player zero-sum games**, and **N-player games**.

## Core Architectural Principles

1. **Zero Virtual Loss — Pure Sequential Tree Search**
   - Every MCTS / AlphaZero tree is strictly single-threaded: zero atomics, zero per-node `shared_mutex`, and **100% informed PUCT selections**.
   - Parallelism scales cleanly at the outer levels: **Game Parallelism** for self-play batch saturation and **Root Parallelism** for tournament time controls.
2. **Generalized to `const N: usize` Players**
   - `GameStatus::Terminal([f32; N])` and acting-player-perspective `value_sum` handle `N = 1` single-player solvers (e.g. rotation puzzles / Rubik's Cube), `N = 2` zero-sum games (Tic-Tac-Toe, Nim, Ultimate TTT), and `N > 2` multi-player games without sign-flipping assumptions.
3. **Unified Playout MCTS & AlphaZero Engine (`src/mcts.rs`)**
   - Parameterized over `SelectionPolicy` (`Uct` vs `Puct`) and `LeafEvaluator` (`PlayoutEvaluator` vs neural-network prior + value evaluation).
4. **20-Byte Nodes via Root-to-Leaf State Reconstruction & Generational Arena (`src/tree.rs`)**
   - Reconstructs `Game` on the stack during root-to-leaf descent (`curr_game.apply(mv)`), eliminating `PreNode`/`FixNode`, virtual dispatch, parent pointers, and `State` storage from tree nodes.
   - Retains `GenerationalTree` ping-pong subtree compaction (`O(1)` arena reset, contiguous child slices `&[Node]`, zero `unsafe` code).

## Quick Start

```bash
# Run unit tests
cargo test

# Run linter with strict warnings
cargo clippy --all-targets -- -D warnings

# Run release throughput & memory footprint benchmark
cargo run --release --example bench_mcts
```

## Adding a Game

Everything game-specific lives in the game's module (`src/games/*.rs`):

- `GameRules<N>`: rules, moves, terminal status.
- `AlphaZeroGame<G, P, N>`: the three required items (`SENTINEL_MOVE`, `move_to_policy_index`,
  `encode_state`) plus optional overrides with generic defaults: `net_config` (architecture;
  default residual MLP), `symmetries` (data augmentation; default none), `hyperparams`
  (self-play / training / evaluation defaults) and `net_presets` (architectures offered in the lab's
  Experiments tab).

`src/nn.rs`, `src/nn_tch.rs`, `src/training.rs` and the self-play engine are game-independent.

## Neural-Network Backends

Two interchangeable backends run the same architectures (`ResNet2d`, `ResMlp`, `BlockNet`) with
identical parameter names, so `.safetensors` checkpoints load in either one:

| Backend | Build | Devices |
|---|---|---|
| candle (pure Rust, default) | always | CPU, Apple Metal (`--features metal`), CUDA (`--features cuda`, `cudnn`) |
| libtorch via tch-rs | `--features tch` | CPU, Apple MPS, CUDA |

The backend is chosen at runtime (`NnDevice { backend, gpu }`, the lab's System panel, or
`--backend=candle|tch`). Building with `tch` needs libtorch **2.9**, the C++ library behind
PyTorch (no Python involved). `start_lab.sh --tch` downloads the official build for the platform
into `./libtorch` on first use (version pinned in the script); `LIBTORCH` points it elsewhere:

```bash
./start_lab.sh --metal --tch                      # macOS arm64: downloads ./libtorch
./start_lab.sh --cuda --tch                       # Linux: CUDA build (MCTS_LIBTORCH_VARIANT, default cu128)
LIBTORCH=/opt/libtorch ./start_lab.sh --cuda --tch  # Docker: a libtorch baked into the image
```

For NVIDIA GPUs in the cloud, [RUNPOD.md](RUNPOD.md) describes the Docker image (built by GitHub
Actions) and running the lab on RunPod with SSH access.

Plain cargo commands with `tch` need the same libtorch and the version-check bypass (tch 0.22
asks for exactly 2.9.0; 2.9.x patch releases are compatible):

```bash
LIBTORCH=$PWD/libtorch LIBTORCH_BYPASS_VERSION_CHECK=1 cargo test --release --features metal,tch
```

Throughput comparison: `cargo run --release --features metal,tch --example bench_nn`
(and `experiments/bench_cuda.sh` on a CUDA machine).

## Rubik's cube round game

`examples/train_rubiks.rs` trains a cube solver by playing rounds of rising difficulty: round `n`
is a cube scrambled with `n` quarter turns, solving it within `2n + 10` moves (at most 100) leads to
round `n + 1` (up to 26), failing ends the game, and a game scores `n + 1/r` for its last solved round (solved in `r` moves). Each
round is searched and labelled on its own with a discounted value (`γ^(moves to solved)`), so the
network learns how far a cube is from solved; the game moves self-play to deeper scrambles as the
network improves.

```bash
LIBTORCH=$PWD/libtorch LIBTORCH_BYPASS_VERSION_CHECK=1 \
  cargo run --release --features metal,tch --example train_rubiks -- minutes=60
```

Options are `key=value` (see the file header); weights go to `checkpoints/rubiks_rounds.safetensors`
after every iteration and `init=rubiks_rounds.safetensors` continues a run.

## Browser Lab

`./start_lab.sh [--metal|--cuda] [--tch]` serves the lab at <http://127.0.0.1:8080>: live training
with all hyperparameters as knobs (one chart per training metric, plus live CPU/GPU utilization per
phase), an arena for matches between engines and checkpoints, and an
**Experiments** tab that queues training runs from scratch (architecture, backend, device,
hyperparameters, training-time budget) and plots their strength against fixed opponents per
training minute and per self-play game. Results persist in `lab_experiments.json`. Checkpoints are kept in `checkpoints/` (or `$MCTS_CHECKPOINT_DIR`, e.g. a mounted
volume in a container) and named `<game>_<name>.safetensors`. Each records its network architecture in the safetensors
metadata, so loading needs no architecture choice (older files are matched against the game's presets).

Evaluation opponents are configurable on the training page and per experiment: random, classical
MCTS (simulations, exploration constant), Minimax (depth) or a saved AlphaZero checkpoint
(file, simulations). Training can continue from a saved network: **Load → Live** in
the arena tab (training page), or *Start From* in an experiment. **Load → Live** gives the live
network the checkpoint's architecture (shown in the *Network* badge), and the lab starts with the
live network it last trained; *Start From* needs the experiment's architecture. A checkpoint carries
weights only (no replay buffer or optimizer state).

The **Play** tab plays single games of Tic-Tac-Toe or Ultimate TTT. Each side is a human or an
engine (random, Minimax, MCTS, AlphaZero with the live network or a checkpoint and any number of
root-parallel trees, which share one network batch per step), so it covers
engine vs. engine and playing against an engine. When an engine is to move it thinks first and a heat
map shows what it thinks of every candidate move (⏭ plays it, ▶ keeps playing): share of the search, network prior or win rate for MCTS/AlphaZero,
and the score of every move for Minimax. A win-probability graph and the move list let you review the game.
New games implement `play::PlayableGame` (start position, Minimax heuristic, evaluation openings,
cell owners). The evaluation openings are shared by the training evaluation, experiments and arena matches.
