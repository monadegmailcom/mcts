//! End-to-End Pure-Rust AlphaZero Training & Evaluation Showcase.
//!
//! Demonstrates:
//! 1. **Two-Player (`N = 2`) Tic-Tac-Toe AlphaZero Training**:
//!    Trains an [`AlphaZeroNet`] from random initialization via Game-Parallel Self-Play
//!    ([`run_self_play`]), [`ReplayBuffer`], and [`AlphaZeroTrainer`], evaluating after each
//!    iteration against `RandomPlayer`, `Minimax(depth=2)`, and `Minimax(depth=5)` using
//!    batched [`run_multi_match`].
//! 2. **Single-Player (`N = 1`) 3x3x3 Rubik's Cube Curriculum AlphaZero Training**:
//!    Trains an [`AlphaZeroNet`] on progressively deeper cube scrambles (`depth = 1 -> 2 -> 3`).
//!
//! Run with:
//! - CPU: `cargo run --release --example train_alphazero`
//! - Apple Silicon GPU: `cargo run --release --features metal --example train_alphazero`
//! - NVIDIA Multi-GPU (RunPod): `cargo run --release --features cuda --example train_alphazero`

use std::time::Instant;

use rand::prelude::{SeedableRng, SmallRng};

use mcts::alphazero::{
    AlphaZeroConfig, AlphaZeroPlayer, SelfPlayConfig, run_self_play, run_self_play_with_initializer,
};
use mcts::game::PlayerIndex;
use mcts::games::rubiks_cube::{RubiksGame, RubiksState};
use mcts::games::tic_tac_toe::{EMPTY_STATE, TicTacToe, TttGame, TttHeuristic};
use mcts::matchup::{RandomPlayer, run_multi_match};
use mcts::minimax::MinimaxPlayer;
use mcts::nn::{AlphaZeroNetConfig, default_device, discover_devices};
use mcts::player::Player;
use mcts::training::{AlphaZeroTrainer, ReplayBuffer};

fn main() -> candle_core::Result<()> {
    let train_device = default_device();
    let infer_devices = discover_devices(8);
    println!("=== Pure-Rust AlphaZero Training Pipeline ===");
    println!("Training device:  {train_device:?}");
    println!("Inference devices ({} workers): {infer_devices:?}", infer_devices.len());
    println!();

    // -------------------------------------------------------------------------
    // Part 1: Train AlphaZero on Tic-Tac-Toe (G = 27, P = 9, N = 2)
    // -------------------------------------------------------------------------
    println!("=== Part 1: Training AlphaZero on Tic-Tac-Toe (G=27, P=9, N=2) ===");
    let mut ttt_trainer = AlphaZeroTrainer::<27, 9, 2>::new(
        AlphaZeroNetConfig::tic_tac_toe_mlp(),
        2e-3,
        1e-4,
        train_device.clone(),
    )?;
    let ttt_service = ttt_trainer.build_inference_service(64, &infer_devices)?;
    let mut ttt_buffer = ReplayBuffer::<27, 9, 2>::new(16_384);
    let ttt_root = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
    let mut rng = SmallRng::seed_from_u64(2026);

    let iterations = 12usize;
    for iter in 1..=iterations {
        let iter_start = Instant::now();
        let sp_cfg = SelfPlayConfig::new(128, 64, 1000 + (iter as u64) * 97)
            .with_workers(4)
            .with_opening_moves(4)
            .with_dirichlet(0.3, 0.25);

        let sp_res = run_self_play(&ttt_root, &ttt_service, &sp_cfg);
        ttt_buffer.extend(&sp_res.positions);

        let metrics = ttt_trainer.train_epoch(&ttt_buffer, &ttt_service, 64, 60, &mut rng)?;

        let f_az = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(AlphaZeroPlayer::new(
                *g,
                &ttt_service,
                AlphaZeroConfig::new(1, 100, seed).with_dirichlet(0.3, 0.0),
            ))
        };
        let f_rnd = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(RandomPlayer::new(*g, seed))
        };
        let f_mm2 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 2, seed))
        };
        let f_mm5 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 5, seed))
        };

        let vs_rnd = run_multi_match(&ttt_root, [&f_az, &f_rnd], 40, 2, 500 + iter as u64);
        let vs_mm2 = run_multi_match(&ttt_root, [&f_az, &f_mm2], 40, 2, 600 + iter as u64);
        let vs_mm5 = run_multi_match(&ttt_root, [&f_az, &f_mm5], 40, 2, 700 + iter as u64);

        println!(
            "Iter {iter:2}/{iterations} ({:5.1} ms) | Buf: {:5} | Loss: {:.4} (pol {:.4}, val {:.4}) | vs Random: {:2}W-{:2}D-{:2}L | vs MM(d=2): {:2}W-{:2}D-{:2}L | vs MM(d=5): {:2}W-{:2}D-{:2}L",
            iter_start.elapsed().as_secs_f64() * 1000.0,
            ttt_buffer.len(),
            metrics.total_loss,
            metrics.policy_loss,
            metrics.value_loss,
            vs_rnd.wins[0],
            vs_rnd.draws,
            vs_rnd.wins[1],
            vs_mm2.wins[0],
            vs_mm2.draws,
            vs_mm2.wins[1],
            vs_mm5.wins[0],
            vs_mm5.draws,
            vs_mm5.wins[1],
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Part 2: Train N = 1 AlphaZero on 3x3x3 Rubik's Cube Curriculum (G = 324, P = 12, N = 1)
    // -------------------------------------------------------------------------
    println!("=== Part 2: N=1 3x3x3 Rubik's Cube Curriculum AlphaZero (G=324, P=12, N=1) ===");
    let mut cube_trainer = AlphaZeroTrainer::<324, 12, 1>::new(
        AlphaZeroNetConfig::ResMlp {
            hidden_dim: 128,
            num_res_blocks: 2,
            head_hidden: 64,
        },
        2e-3,
        1e-4,
        train_device,
    )?;
    let cube_service = cube_trainer.build_inference_service(64, &infer_devices)?;
    let mut cube_buffer = ReplayBuffer::<324, 12, 1>::new(16_384);

    for depth in 1..=3u8 {
        for epoch in 1..=3usize {
            let start = Instant::now();
            let sp_cfg = SelfPlayConfig::new(48, 120, ((depth as usize) * 100 + epoch) as u64)
                .with_workers(4)
                .with_opening_moves(0)
                .with_dirichlet(0.3, 0.15);

            let res = run_self_play_with_initializer(&cube_service, &sp_cfg, |_idx, r| {
                RubiksGame::new(PlayerIndex::P0, RubiksState::scrambled(depth, depth + 1, r))
            });
            cube_buffer.extend(&res.positions);
            let metrics = cube_trainer.train_epoch(&cube_buffer, &cube_service, 64, 40, &mut rng)?;

            println!(
                "Scramble depth {depth} (epoch {epoch}/3, {:5.1} ms) | Solved: {:2}/{:2} | Loss: {:.4} (pol {:.4}, val {:.4})",
                start.elapsed().as_secs_f64() * 1000.0,
                res.wins[0],
                res.total_games,
                metrics.total_loss,
                metrics.policy_loss,
                metrics.value_loss,
            );
        }
    }

    Ok(())
}
