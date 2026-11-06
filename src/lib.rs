//! `mcts` — High-performance generic `N`-player MCTS, AlphaZero, and Minimax engine.
//!
//! Designed as a clean, idiomatic Rust evolution of the C++ `beat-it` engine:
//! - Generalized to `const N: usize` players (`N = 1` puzzles, `N = 2` zero-sum games, `N > 2` multi-player).
//! - Unified single-threaded MCTS engine (`mcts`) shared between playout-based UCT and AlphaZero PUCT.
//! - Zero-`unsafe` two-generation compact arena (`tree`) with 20-byte nodes and root-to-leaf state reconstruction.

pub mod game;
pub mod games;
pub mod mcts;
pub mod player;
pub mod statistics;
pub mod tree;
