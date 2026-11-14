//! End-to-End Pure-Rust AlphaZero Training, Live Browser Lab, & Evaluation Showcase.
//!
//! Demonstrates:
//! 1. **Interactive Live Browser Lab & Multi-Engine Arena (`http://127.0.0.1:8080`)**:
//!    - Starts in **Idle / Paused** state when launched with `--serve` (or `./start_lab.sh`) so you
//!      have full Start / Pause / Resume / Step (+1 Iter) / Reset control from the browser.
//!    - Supports switching live between **3x3 Tic-Tac-Toe (`G=27, P=9, N=2`)** and
//!      **9x9 Ultimate Tic-Tac-Toe (`G=324, P=81, N=2`)**.
//!    - Automatically saves `<game>_iter_NNN.safetensors` checkpoints every `N` iterations so you
//!      can pit networks from different training stages against each other or against MCTS,
//!      Tree-Minimax, and Minimax at calibrated equal thinking time per move.
//! 2. **Automated CLI Showcase (when run without `--serve`)**:
//!    - Trains Tic-Tac-Toe (`N = 2`) and 3x3x3 Rubik's Cube Curriculum (`N = 1`).
//!
//! Run with:
//! - Quick start script (starts idle in Browser Lab): `./start_lab.sh`
//! - CPU Interactive Lab: `cargo run --release --example train_alphazero -- --serve`
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
use mcts::games::ultimate_ttt::{
    EMPTY_STATE as UTTT_EMPTY_STATE, UltimateTicTacToe, UtttGame, UtttHeuristic,
};
use mcts::inference::InferenceService;
use mcts::matchup::{RandomPlayer, run_multi_match};
use mcts::minimax::MinimaxPlayer;
use mcts::nn::{AlphaZeroNetConfig, CandleBatchEvaluator, default_device, discover_devices};
use mcts::player::Player;
use mcts::telemetry::{IterationTelemetry, LabGame, LiveKnobs, TelemetryHub, TrainingCommand};
use mcts::training::{AlphaZeroTrainer, ReplayBuffer};

#[allow(clippy::too_many_arguments)]
fn run_ttt_iteration(
    iter: usize,
    hub: &TelemetryHub,
    trainer: &mut AlphaZeroTrainer<27, 9, 2>,
    service: &Arc<InferenceService<CandleBatchEvaluator<27, 9, 2>, 27, 9, 2>>,
    buffer: &mut ReplayBuffer<27, 9, 2>,
    max_batch_size: usize,
    rng: &mut SmallRng,
) -> candle_core::Result<()> {
    let knobs = hub.knobs_for(LabGame::Ttt);
    trainer.set_learning_rate(knobs.learning_rate);
    service.reset_stats();

    let root = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

    // 1. Self-Play Phase
    hub.set_phase_for(
        LabGame::Ttt,
        iter,
        format!(
            "Iter #{iter}: Self-Play ({} games × {} sims)...",
            knobs.games_per_iteration, knobs.simulations_per_move
        ),
    );
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

    let sp_res = run_self_play(&root, service, &sp_cfg);
    let self_play_ms = (sp_start.elapsed().as_secs_f64() * 1000.0) as f32;
    buffer.extend(&sp_res.positions);

    let batch_stats = service.batch_size_stats();
    let uniq_batch_stats = service.unique_batch_size_stats();
    let dedup_saved_pct = service.dedup_savings_ratio() * 100.0;
    let infer_stats = service.inference_time_stats();
    let approx_sims = (sp_res.positions.len() * knobs.simulations_per_move) as f32;
    let simulations_per_sec = approx_sims / (self_play_ms / 1000.0).max(1e-6);

    // 2. SGD Training Phase
    hub.set_phase_for(
        LabGame::Ttt,
        iter,
        format!("Iter #{iter}: SGD Training ({} steps)...", knobs.train_steps),
    );
    let train_start = Instant::now();
    let metrics = trainer.train_epoch(
        buffer,
        service,
        knobs.batch_size,
        knobs.train_steps,
        rng,
    )?;
    let train_ms = (train_start.elapsed().as_secs_f64() * 1000.0) as f32;

    // 3. Arena Evaluation Phase
    hub.set_phase_for(
        LabGame::Ttt,
        iter,
        format!("Iter #{iter}: Arena Evaluation (vs Random, MM2, MM5)..."),
    );
    let eval_start = Instant::now();
    let svc_ref = service.as_ref();
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

    let vs_rnd = run_multi_match(&root, [&f_az, &f_rnd], 40, 2, 500 + iter as u64);
    let vs_mm2 = run_multi_match(&root, [&f_az, &f_mm2], 40, 2, 600 + iter as u64);
    let vs_mm5 = run_multi_match(&root, [&f_az, &f_mm5], 40, 2, 700 + iter as u64);
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
        buffer_len: buffer.len(),
        buffer_capacity: buffer.capacity(),
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
        "[TTT  #{iter:>3}] Loss {:>5.3} | KL {:>6.4} | MSE {:>6.4} | vs Rnd {:>2}W-{:>2}D-{:>2}L | vs MM2 {:>2}W-{:>2}D-{:>2}L | vs MM5 {:>2}W-{:>2}D-{:>2}L | Dedup {:>4.1}% | {:>7.0} sims/s",
        rec.total_loss,
        rec.policy_kl,
        rec.value_loss,
        rec.vs_random[0],
        rec.vs_random[1],
        rec.vs_random[2],
        rec.vs_mm2[0],
        rec.vs_mm2[1],
        rec.vs_mm2[2],
        rec.vs_mm5[0],
        rec.vs_mm5[1],
        rec.vs_mm5[2],
        rec.dedup_saved_pct,
        rec.simulations_per_sec,
    );

    trainer.save_checkpoint("ttt_alphazero.safetensors")?;
    let auto_every = hub.auto_save_every_for(LabGame::Ttt);
    if auto_every > 0 && iter.is_multiple_of(auto_every) {
        let snap = format!("ttt_iter_{iter:03}.safetensors");
        trainer.save_checkpoint(&snap)?;
        println!("📸 Auto-saved TTT checkpoint: {snap}");
    }

    hub.record_iteration_for(LabGame::Ttt, rec);
    let paused_now = hub.knobs_for(LabGame::Ttt).paused;
    hub.set_phase_for(
        LabGame::Ttt,
        iter,
        if paused_now {
            format!("Paused after Iter #{iter}")
        } else {
            format!("Completed Iter #{iter}")
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_uttt_iteration(
    iter: usize,
    hub: &TelemetryHub,
    trainer: &mut AlphaZeroTrainer<324, 81, 2>,
    service: &Arc<InferenceService<CandleBatchEvaluator<324, 81, 2>, 324, 81, 2>>,
    buffer: &mut ReplayBuffer<324, 81, 2>,
    max_batch_size: usize,
    rng: &mut SmallRng,
) -> candle_core::Result<()> {
    let knobs = hub.knobs_for(LabGame::Uttt);
    trainer.set_learning_rate(knobs.learning_rate);
    service.reset_stats();

    let root = UtttGame::new(PlayerIndex::PLAYER1, UTTT_EMPTY_STATE);
    let h = UtttHeuristic::default();

    // 1. Self-Play Phase
    println!(
        "⏳ [UTTT #{iter:>3}] Self-Play ({} games × {} sims/move)...",
        knobs.games_per_iteration, knobs.simulations_per_move
    );
    hub.set_phase_for(
        LabGame::Uttt,
        iter,
        format!(
            "Iter #{iter}: Self-Play ({} games × {} sims)...",
            knobs.games_per_iteration, knobs.simulations_per_move
        ),
    );
    let sp_start = Instant::now();
    let sp_cfg = SelfPlayConfig::new(
        knobs.games_per_iteration,
        knobs.simulations_per_move,
        5000 + (iter as u64) * 131,
    )
    .with_workers(4)
    .with_opening_moves(6)
    .with_puct(19_652.0, knobs.c_init)
    .with_dirichlet(0.3, knobs.dirichlet_epsilon);

    let sp_res = run_self_play(&root, service, &sp_cfg);
    let self_play_ms = (sp_start.elapsed().as_secs_f64() * 1000.0) as f32;
    buffer.extend(&sp_res.positions);

    let batch_stats = service.batch_size_stats();
    let uniq_batch_stats = service.unique_batch_size_stats();
    let dedup_saved_pct = service.dedup_savings_ratio() * 100.0;
    let infer_stats = service.inference_time_stats();
    let approx_sims = (sp_res.positions.len() * knobs.simulations_per_move) as f32;
    let simulations_per_sec = approx_sims / (self_play_ms / 1000.0).max(1e-6);

    // 2. SGD Training Phase
    hub.set_phase_for(
        LabGame::Uttt,
        iter,
        format!(
            "Iter #{iter}: SGD Training ({} steps, buffer={})...",
            knobs.train_steps,
            buffer.len()
        ),
    );
    let train_start = Instant::now();
    let metrics = trainer.train_epoch(
        buffer,
        service,
        knobs.batch_size,
        knobs.train_steps,
        rng,
    )?;
    let train_ms = (train_start.elapsed().as_secs_f64() * 1000.0) as f32;

    // 3. Arena Evaluation Phase (vs Random, Minimax d=1, Minimax d=2)
    let eval_start = Instant::now();
    let svc_ref = service.as_ref();
    let eval_sims = knobs.simulations_per_move.clamp(20, 64);
    let f_az = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
        Box::new(AlphaZeroPlayer::new(
            *g,
            svc_ref,
            AlphaZeroConfig::new(1, eval_sims, seed)
                .with_puct(19_652.0, knobs.c_init)
                .with_dirichlet(0.3, 0.0),
        ))
    };
    let f_rnd = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
        Box::new(RandomPlayer::new(*g, seed))
    };
    let f_mm1 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
        Box::new(MinimaxPlayer::new(*g, h, 1, seed))
    };
    let f_mm2 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
        Box::new(MinimaxPlayer::new(*g, h, 2, seed))
    };

    hub.set_phase_for(
        LabGame::Uttt,
        iter,
        format!("Iter #{iter}: Arena Eval (1/3 vs Random)..."),
    );
    let vs_rnd = run_multi_match(&root, [&f_az, &f_rnd], 12, 4, 2500 + iter as u64);

    hub.set_phase_for(
        LabGame::Uttt,
        iter,
        format!("Iter #{iter}: Arena Eval (2/3 vs Minimax d=1)..."),
    );
    let vs_mm1 = run_multi_match(&root, [&f_az, &f_mm1], 12, 4, 2600 + iter as u64);

    hub.set_phase_for(
        LabGame::Uttt,
        iter,
        format!("Iter #{iter}: Arena Eval (3/3 vs Minimax d=2)..."),
    );
    let vs_mm2 = run_multi_match(&root, [&f_az, &f_mm2], 12, 4, 2700 + iter as u64);
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
        vs_mm2: [vs_mm1.wins[0], vs_mm1.draws, vs_mm1.wins[1]],
        vs_mm5: [vs_mm2.wins[0], vs_mm2.draws, vs_mm2.wins[1]],
        buffer_len: buffer.len(),
        buffer_capacity: buffer.capacity(),
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
        "[UTTT #{iter:>3}] Loss {:>5.3} | KL {:>6.4} | MSE {:>6.4} | vs Rnd {:>2}W-{:>2}D-{:>2}L | vs MM1 {:>2}W-{:>2}D-{:>2}L | vs MM2 {:>2}W-{:>2}D-{:>2}L | Dedup {:>4.1}% | {:>7.0} sims/s",
        rec.total_loss,
        rec.policy_kl,
        rec.value_loss,
        rec.vs_random[0],
        rec.vs_random[1],
        rec.vs_random[2],
        rec.vs_mm2[0],
        rec.vs_mm2[1],
        rec.vs_mm2[2],
        rec.vs_mm5[0],
        rec.vs_mm5[1],
        rec.vs_mm5[2],
        rec.dedup_saved_pct,
        rec.simulations_per_sec,
    );

    trainer.save_checkpoint("uttt_alphazero.safetensors")?;
    let auto_every = hub.auto_save_every_for(LabGame::Uttt);
    if auto_every > 0 && iter.is_multiple_of(auto_every) {
        let snap = format!("uttt_iter_{iter:03}.safetensors");
        trainer.save_checkpoint(&snap)?;
        println!("📸 Auto-saved UTTT checkpoint: {snap}");
    }

    hub.record_iteration_for(LabGame::Uttt, rec);
    let paused_now = hub.knobs_for(LabGame::Uttt).paused;
    hub.set_phase_for(
        LabGame::Uttt,
        iter,
        if paused_now {
            format!("Paused after Iter #{iter}")
        } else {
            format!("Completed Iter #{iter}")
        },
    );
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let interactive_lab =
        std::env::args().any(|a| a == "--serve" || a == "--lab-only");
    let bind_addr =
        std::env::var("MCTS_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let train_device = default_device();
    let infer_devices = discover_devices(8);
    let device_label = format!("{train_device:?} ({} worker(s))", infer_devices.len());

    println!("=== Pure-Rust AlphaZero Training Pipeline & Live Lab ===");
    println!("Training device:  {train_device:?}");
    println!(
        "Inference devices ({} workers): {infer_devices:?}",
        infer_devices.len()
    );

    let iterations = 15usize;
    let mut initial_knobs = LiveKnobs::default_ttt();
    // In interactive Browser Lab mode (`--serve` / `./start_lab.sh`), start IDLE (`paused = true`)
    // so training only runs when the user clicks Start / Resume / Step in the Browser Lab.
    initial_knobs.paused = interactive_lab;

    let hub = TelemetryHub::new(device_label, iterations, initial_knobs);
    match hub.spawn_http_server(&bind_addr) {
        Ok(addr) => println!("🌐 Live Browser Lab & Multi-Match Arena listening on: http://{addr}"),
        Err(err) => {
            println!("⚠️ Could not bind Browser Lab on {bind_addr}: {err}");
            if interactive_lab {
                println!(
                    "❌ Another process is already listening on {bind_addr}. Stop the old process (or run `./start_lab.sh` which cleans it up automatically) and try again."
                );
                return Ok(());
            }
        }
    }
    println!();

    let max_batch_size = 64usize;

    // 1. Initialize Tic-Tac-Toe (`G=27, P=9, N=2`) Trainer & Live Inference Service
    let ttt_net_cfg = AlphaZeroNetConfig::tic_tac_toe_mlp();
    let mut ttt_trainer = AlphaZeroTrainer::<27, 9, 2>::new(
        ttt_net_cfg,
        initial_knobs.learning_rate,
        1e-4,
        train_device.clone(),
    )?;
    if interactive_lab && Path::new("ttt_alphazero.safetensors").exists() {
        ttt_trainer.load_checkpoint("ttt_alphazero.safetensors")?;
        println!("📦 Loaded existing TTT checkpoint into live model: ttt_alphazero.safetensors");
    }
    let ttt_service = Arc::new(ttt_trainer.build_inference_service(max_batch_size, &infer_devices)?);
    hub.attach_ttt_service(Arc::clone(&ttt_service), ttt_net_cfg, infer_devices.clone());
    let mut ttt_buffer = ReplayBuffer::<27, 9, 2>::new(16_384);

    // 2. Initialize Ultimate Tic-Tac-Toe (`G=324, P=81, N=2`) Trainer & Live Inference Service
    let uttt_net_cfg = AlphaZeroNetConfig::ultimate_ttt_mlp();
    let mut uttt_trainer = AlphaZeroTrainer::<324, 81, 2>::new(
        uttt_net_cfg,
        2.0e-3,
        1e-4,
        train_device.clone(),
    )?;
    if interactive_lab && Path::new("uttt_alphazero.safetensors").exists() {
        uttt_trainer.load_checkpoint("uttt_alphazero.safetensors")?;
        println!("📦 Loaded existing UTTT checkpoint into live model: uttt_alphazero.safetensors");
    }
    let uttt_service =
        Arc::new(uttt_trainer.build_inference_service(max_batch_size, &infer_devices)?);
    hub.attach_uttt_service(
        Arc::clone(&uttt_service),
        uttt_net_cfg,
        infer_devices.clone(),
    );
    let mut uttt_buffer = ReplayBuffer::<324, 81, 2>::new(32_768);

    let mut rng = SmallRng::seed_from_u64(2026);

    if interactive_lab {
        println!("⏸  Browser Lab started in IDLE state at http://{bind_addr}");
        println!("   • Open http://{bind_addr} to choose Tic-Tac-Toe (3x3) or Ultimate Tic-Tac-Toe (9x9)");
        println!("   • Use ▶ Start/Resume, ⏸ Pause, ⏭ Step (+1 Iter), or 🔄 Reset in the Browser Lab");
        println!("   • Press Ctrl+C in this terminal to stop the server.");
        println!();

        loop {
            match hub.poll_training_command() {
                TrainingCommand::Idle => {
                    std::thread::sleep(Duration::from_millis(80));
                }
                TrainingCommand::Reset(LabGame::Ttt) => {
                    let lr = hub.knobs_for(LabGame::Ttt).learning_rate;
                    ttt_trainer.reset(&ttt_service, lr, 1e-4)?;
                    ttt_buffer.clear();
                    println!("🔄 Reset Tic-Tac-Toe model to fresh random weights and cleared replay buffer.");
                }
                TrainingCommand::Reset(LabGame::Uttt) => {
                    let lr = hub.knobs_for(LabGame::Uttt).learning_rate;
                    uttt_trainer.reset(&uttt_service, lr, 1e-4)?;
                    uttt_buffer.clear();
                    println!("🔄 Reset Ultimate Tic-Tac-Toe model to fresh random weights and cleared replay buffer.");
                }
                TrainingCommand::RunIteration {
                    game: LabGame::Ttt,
                    iteration,
                } => {
                    run_ttt_iteration(
                        iteration,
                        &hub,
                        &mut ttt_trainer,
                        &ttt_service,
                        &mut ttt_buffer,
                        max_batch_size,
                        &mut rng,
                    )?;
                }
                TrainingCommand::RunIteration {
                    game: LabGame::Uttt,
                    iteration,
                } => {
                    run_uttt_iteration(
                        iteration,
                        &hub,
                        &mut uttt_trainer,
                        &uttt_service,
                        &mut uttt_buffer,
                        max_batch_size,
                        &mut rng,
                    )?;
                }
            }
        }
    }

    // Non-interactive CLI batch showcase (`cargo run --release --example train_alphazero`)
    println!("=== Part 1: Training AlphaZero on Tic-Tac-Toe (G=27, P=9, N=2) ===");
    for iter in 1..=iterations {
        run_ttt_iteration(
            iter,
            &hub,
            &mut ttt_trainer,
            &ttt_service,
            &mut ttt_buffer,
            max_batch_size,
            &mut rng,
        )?;
    }
    println!();

    // Part 2: Train N = 1 AlphaZero on 3x3x3 Rubik's Cube Curriculum (G = 324, P = 12, N = 1)
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

    Ok(())
}
