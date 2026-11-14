//! End-to-End Pure-Rust AlphaZero Training, Live Browser Lab, & Evaluation Showcase.
//!
//! Demonstrates:
//! 1. **Live Browser Lab & Telemetry Server (`http://127.0.0.1:8080`)**:
//!    Streams the 5 Golden AlphaZero Metrics (Arena W/D/L, Policy KL & Root Entropy, Value MSE,
//!    and System Queue/Batch Fill), supports live hyperparameter knob tuning, and lets you play
//!    interactively against the live-trained model while inspecting raw NN priors vs. MCTS visits.
//! 2. **Two-Player (`N = 2`) Tic-Tac-Toe AlphaZero Training**:
//!    Trains an [`AlphaZeroNet`] from random initialization to optimal play (`0` losses against
//!    `Minimax(depth=5)`) and saves `ttt_alphazero.safetensors`.
//! 3. **Single-Player (`N = 1`) 3x3x3 Rubik's Cube Curriculum AlphaZero Training**:
//!    Trains an [`AlphaZeroNet`] on progressively deeper cube scrambles (`depth = 1 -> 2 -> 3`).
//!
//! Run with:
//! - Quick start script: `./start_lab.sh` (or `./start_lab.sh --lab-only`)
//! - CPU (train + keep lab open): `cargo run --release --example train_alphazero -- --serve`
//! - Instant Lab only (no training): `cargo run --release --example train_alphazero -- --lab-only`
//! - Apple Silicon GPU: `cargo run --release --features metal --example train_alphazero -- --serve`
//! - NVIDIA Multi-GPU (RunPod): `cargo run --release --features cuda --example train_alphazero -- --serve`

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use mcts::telemetry::{IterationTelemetry, LiveKnobs, TelemetryHub};
use mcts::training::{AlphaZeroTrainer, ReplayBuffer};

fn main() -> candle_core::Result<()> {
    let lab_only = std::env::args().any(|a| a == "--lab-only");
    let keep_serving = lab_only || std::env::args().any(|a| a == "--serve");
    let bind_addr = std::env::var("MCTS_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let train_device = default_device();
    let infer_devices = discover_devices(8);
    let device_label = format!("{train_device:?} ({} worker(s))", infer_devices.len());

    println!("=== Pure-Rust AlphaZero Training Pipeline & Live Lab ===");
    println!("Training device:  {train_device:?}");
    println!(
        "Inference devices ({} workers): {infer_devices:?}",
        infer_devices.len()
    );

    // -------------------------------------------------------------------------
    // Part 1: Train AlphaZero on Tic-Tac-Toe (G = 27, P = 9, N = 2)
    // -------------------------------------------------------------------------
    let iterations = 15usize;
    let initial_knobs = LiveKnobs {
        paused: false,
        games_per_iteration: 128,
        simulations_per_move: 80,
        train_steps: 24,
        batch_size: 64,
        learning_rate: 1.5e-3,
        dirichlet_epsilon: 0.25,
        c_init: 1.25,
    };

    let hub = TelemetryHub::new(device_label, iterations, initial_knobs);
    match hub.spawn_http_server(&bind_addr) {
        Ok(addr) => println!("🌐 Live Browser Lab & Play Inspector listening on: http://{addr}"),
        Err(err) => println!("⚠️ Could not bind Browser Lab on {bind_addr}: {err}"),
    }
    println!();

    let ttt_net_cfg = AlphaZeroNetConfig::tic_tac_toe_mlp();
    let mut ttt_trainer = AlphaZeroTrainer::<27, 9, 2>::new(
        ttt_net_cfg,
        initial_knobs.learning_rate,
        1e-4,
        train_device.clone(),
    )?;
    let ckpt_path = "ttt_alphazero.safetensors";
    if lab_only && Path::new(ckpt_path).exists() {
        ttt_trainer.load_checkpoint(ckpt_path)?;
        println!("📦 Loaded existing checkpoint into live model: {ckpt_path}");
    }
    let max_batch_size = 64usize;
    let ttt_service = Arc::new(ttt_trainer.build_inference_service(max_batch_size, &infer_devices)?);
    hub.attach_ttt_service(Arc::clone(&ttt_service), ttt_net_cfg, infer_devices.clone());

    if !lab_only {
        println!("=== Part 1: Training AlphaZero on Tic-Tac-Toe (G=27, P=9, N=2) ===");
        let buffer_capacity = 16_384usize;
        let mut ttt_buffer = ReplayBuffer::<27, 9, 2>::new(buffer_capacity);
        let ttt_root = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let mut rng = SmallRng::seed_from_u64(2026);

        println!(
            "{:<4} | {:>6} | {:>7} | {:>7} | {:>7} | {:>6} | {:>11} | {:>11} | {:>11} | {:>11} | {:>8}",
            "Iter",
            "Loss",
            "Pol KL",
            "Val MSE",
            "Root H",
            "Moves",
            "vs Random",
            "vs MM(d=2)",
            "vs MM(d=5)",
            "Batch(Ded%)",
            "Sims/sec"
        );
        println!("{}", "-".repeat(120));

        for iter in 1..=iterations {
            hub.wait_if_paused();
            let knobs = hub.knobs();
            ttt_trainer.set_learning_rate(knobs.learning_rate);
            ttt_service.reset_stats();

            // 1. Self-Play Phase
            hub.set_phase(iter, "Self-Play");
            let sp_start = Instant::now();
            let sp_cfg = SelfPlayConfig::new(
                knobs.games_per_iteration,
                knobs.simulations_per_move,
                1000 + (iter as u64) * 97,
            )
            .with_workers(4)
            .with_opening_moves(4)
            .with_puct(19_652.0, knobs.c_init)
            .with_dirichlet(0.3, knobs.dirichlet_epsilon);

            let sp_res = run_self_play(&ttt_root, &ttt_service, &sp_cfg);
            let self_play_ms = (sp_start.elapsed().as_secs_f64() * 1000.0) as f32;
            ttt_buffer.extend(&sp_res.positions);

            let batch_stats = ttt_service.batch_size_stats();
            let uniq_batch_stats = ttt_service.unique_batch_size_stats();
            let dedup_saved_pct = ttt_service.dedup_savings_ratio() * 100.0;
            let infer_stats = ttt_service.inference_time_stats();
            let approx_sims = (sp_res.positions.len() * knobs.simulations_per_move) as f32;
            let simulations_per_sec = approx_sims / (self_play_ms / 1000.0).max(1e-6);

            // 2. SGD Training Phase
            hub.set_phase(iter, "Training SGD");
            let train_start = Instant::now();
            let metrics = ttt_trainer.train_epoch(
                &ttt_buffer,
                &ttt_service,
                knobs.batch_size,
                knobs.train_steps,
                &mut rng,
            )?;
            let train_ms = (train_start.elapsed().as_secs_f64() * 1000.0) as f32;

            // 3. Arena Evaluation Phase
            hub.set_phase(iter, "Arena Evaluation");
            let eval_start = Instant::now();
            let svc_ref = ttt_service.as_ref();
            let f_az = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
                Box::new(AlphaZeroPlayer::new(
                    *g,
                    svc_ref,
                    AlphaZeroConfig::new(1, 96, seed)
                        .with_puct(19_652.0, knobs.c_init)
                        .with_dirichlet(0.3, 0.0),
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
            let eval_ms = (eval_start.elapsed().as_secs_f64() * 1000.0) as f32;

            let rec = IterationTelemetry {
                iteration: iter,
                total_loss: metrics.total_loss,
                policy_loss: metrics.policy_loss,
                policy_kl: metrics.policy_kl,
                value_loss: metrics.value_loss,
                root_entropy_mean: sp_res.root_entropy_stats.mean(),
                game_moves_mean: sp_res.game_moves_stats.mean(),
                self_play_wins: sp_res.wins,
                self_play_draws: sp_res.draws,
                vs_random: [vs_rnd.wins[0], vs_rnd.draws, vs_rnd.wins[1]],
                vs_mm2: [vs_mm2.wins[0], vs_mm2.draws, vs_mm2.wins[1]],
                vs_mm5: [vs_mm5.wins[0], vs_mm5.draws, vs_mm5.wins[1]],
                buffer_len: ttt_buffer.len(),
                buffer_capacity,
                batch_size_mean: batch_stats.mean(),
                batch_size_std: batch_stats.stddev(),
                batch_size_max: batch_stats.max(),
                unique_batch_mean: uniq_batch_stats.mean(),
                dedup_saved_pct,
                max_batch_size,
                inference_us_mean: infer_stats.mean(),
                simulations_per_sec,
                self_play_ms,
                train_ms,
                eval_ms,
            };

            println!(
                "{iter:>4} | {:>6.3} | {:>7.4} | {:>7.4} | {:>7.3} | {:>6.1} | {:>2}W-{:>2}D-{:>2}L | {:>2}W-{:>2}D-{:>2}L | {:>2}W-{:>2}D-{:>2}L | {:>4.1}({:>4.1}%) | {:>8.0}",
                rec.total_loss,
                rec.policy_kl,
                rec.value_loss,
                rec.root_entropy_mean,
                rec.game_moves_mean,
                rec.vs_random[0],
                rec.vs_random[1],
                rec.vs_random[2],
                rec.vs_mm2[0],
                rec.vs_mm2[1],
                rec.vs_mm2[2],
                rec.vs_mm5[0],
                rec.vs_mm5[1],
                rec.vs_mm5[2],
                rec.batch_size_mean,
                rec.dedup_saved_pct,
                rec.simulations_per_sec,
            );

            hub.record_iteration(rec);
        }

        ttt_trainer.save_checkpoint(ckpt_path)?;
        println!("💾 Saved trained Tic-Tac-Toe checkpoint to: {ckpt_path}");
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
                let metrics =
                    cube_trainer.train_epoch(&cube_buffer, &cube_service, 64, 40, &mut rng)?;

                println!(
                    "Scramble depth {depth} (epoch {epoch}/3, {:5.1} ms) | Solved: {:2}/{:2} | Loss: {:.4} (KL {:.4}, val {:.4})",
                    start.elapsed().as_secs_f64() * 1000.0,
                    res.wins[0],
                    res.total_games,
                    metrics.total_loss,
                    metrics.policy_kl,
                    metrics.value_loss,
                );
            }
        }
    }

    hub.set_phase(iterations, "Completed — Interactive Play Ready");
    if keep_serving {
        println!();
        println!("🚀 Browser Lab, Arena & Interactive Play live at: http://{bind_addr}");
        println!("   Press Ctrl+C to exit.");
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }

    Ok(())
}
