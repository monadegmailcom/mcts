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
