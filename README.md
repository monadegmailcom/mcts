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
`--backend=candle|tch`). Building with `tch` needs libtorch / PyTorch **2.9**:

```bash
# Dev machine with a PyTorch 2.9 install
MCTS_TORCH_PYTHON=/path/to/venv/bin/python ./start_lab.sh --metal --tch
# Linux / Docker with a standalone libtorch
LIBTORCH=/opt/libtorch ./start_lab.sh --cuda --tch
```

Throughput comparison: `cargo run --release --features metal,tch --example bench_nn`
(and `experiments/bench_cuda.sh` on a CUDA machine).

## Browser Lab

`./start_lab.sh [--metal|--cuda] [--tch]` serves the lab at <http://127.0.0.1:8080>: live training
with all hyperparameters as knobs, an arena for matches between engines and checkpoints, and an
**Experiments** tab that queues training runs from scratch (architecture, backend, device,
hyperparameters, training-time budget) and plots their strength against fixed opponents per
training minute and per self-play game. Results persist in `lab_experiments.json`. Checkpoints are kept in `checkpoints/` (or `$MCTS_CHECKPOINT_DIR`, e.g. a mounted
volume in a container) and named `<game>_<name>.safetensors`.

Evaluation opponents are configurable on the training page and per experiment: random, classical
MCTS (simulations, exploration constant), Minimax (depth) or a saved AlphaZero checkpoint
(file, architecture, simulations, root-parallel trees). Training can continue from a saved network: **Load → Live** in
the arena tab (training page), or *Start From* in an experiment. A checkpoint only loads into a
network of the same architecture; it carries weights only (no replay buffer or optimizer state).

The **Play** tab plays single games of Tic-Tac-Toe or Ultimate TTT. Each side is a human or an
engine (random, Minimax, MCTS, AlphaZero with the live network or a checkpoint and any number of
root-parallel trees, which share one network batch per step), so it covers
engine vs. engine, playing against an engine, and an analysis board. A heat map shows the searching
engine's view of every candidate move (share of the search, network prior, or win rate); on a human's
turn an analyst engine provides it. A win-probability graph and the move list let you review the game.
New games implement `play::PlayableGame` (start position, Minimax heuristic, cell owners).
