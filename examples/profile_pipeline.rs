//! Pure-Rust System Pipeline & Queue Auto-Tuner (`profile_pipeline`).
//!
//! Benchmarks the interaction between:
//! - `self_play_workers` (CPU threads stepping MCTS trees via `run_parallel_games`)
//! - `inference_workers` (background threads draining the lock-free `ArrayQueue` into batches)
//! - `max_batch_size` (maximum states grouped into a single `CandleBatchEvaluator` forward pass)
//!
//! Unlike ML hyperparameters (`learning_rate`, `c_init`, `dirichlet_alpha`), system pipeline
//! parameters only affect **throughput (`sims/sec`)** and **batch fill ratio**, so we can find
//! the optimal configuration for any machine (Mac M-series or RunPod GPU) in a few seconds!
//!
//! Run with:
//! - CPU: `cargo run --release --example profile_pipeline`
//! - Apple Silicon GPU: `cargo run --release --features metal --example profile_pipeline`
//! - NVIDIA GPU (RunPod): `cargo run --release --features cuda --example profile_pipeline`

use std::time::Instant;

use mcts::alphazero::{SelfPlayConfig, run_self_play};
use mcts::game::PlayerIndex;
use mcts::games::tic_tac_toe::{EMPTY_STATE, TttGame};
use mcts::nn::{AlphaZeroNetConfig, default_device};
use mcts::training::AlphaZeroTrainer;

#[derive(Clone, Debug)]
struct ProfileRow {
    sp_workers: usize,
    inf_workers: usize,
    max_batch: usize,
    elapsed_ms: f64,
    sims_per_sec: f64,
    games_per_sec: f64,
    batch_mean: f32,
    unique_batch_mean: f32,
    dedup_pct: f32,
    infer_us_per_state: f32,
}

fn main() -> candle_core::Result<()> {
    let device = default_device();
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("=== Pure-Rust AlphaZero System Pipeline Auto-Profiler ===");
    println!("Detected Logical CPUs: {logical_cpus}");
    println!("Active Candle Device:  {device:?}");
    println!();

    let trainer = AlphaZeroTrainer::<27, 9, 2>::new(
        AlphaZeroNetConfig::tic_tac_toe_mlp(),
        2e-3,
        1e-4,
        device.clone(),
    )?;
    let root_game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

    let games_per_trial = 64usize;
    let sims_per_move = 32usize;

    let sp_worker_choices = [1usize, 2, 4, 8];
    let inf_worker_choices = [1usize, 2];
    let batch_choices = [16usize, 32, 64, 128];

    println!(
        "{:<10} | {:<11} | {:<9} | {:>10} | {:>9} | {:>15} | {:>8} | {:>11}",
        "SP Workers",
        "Inf Workers",
        "Max Batch",
        "Sims / sec",
        "Games / s",
        "Batch Log→Uniq",
        "Dedup %",
        "NN µs/state"
    );
    println!("{}", "-".repeat(102));

    let mut results: Vec<ProfileRow> = Vec::new();

    for &inf_workers in &inf_worker_choices {
        let devices: Vec<_> = (0..inf_workers).map(|_| device.clone()).collect();
        for &max_batch in &batch_choices {
            let service = trainer.build_inference_service(max_batch, &devices)?;
            for &sp_workers in &sp_worker_choices {
                service.reset_stats();
                let cfg = SelfPlayConfig::new(games_per_trial, sims_per_move, 2026)
                    .with_workers(sp_workers)
                    .with_opening_moves(2);

                let start = Instant::now();
                let sp_res = run_self_play(&root_game, &service, &cfg);
                let elapsed = start.elapsed().as_secs_f64();

                let total_moves = sp_res.positions.len();
                let approx_sims = (total_moves * sims_per_move) as f64;
                let sims_per_sec = approx_sims / elapsed;
                let games_per_sec = (games_per_trial as f64) / elapsed;

                let b_stats = service.batch_size_stats();
                let u_stats = service.unique_batch_size_stats();
                let t_stats = service.inference_time_stats();
                let dedup_pct = service.dedup_savings_ratio() * 100.0;

                let row = ProfileRow {
                    sp_workers,
                    inf_workers,
                    max_batch,
                    elapsed_ms: elapsed * 1000.0,
                    sims_per_sec,
                    games_per_sec,
                    batch_mean: b_stats.mean(),
                    unique_batch_mean: u_stats.mean(),
                    dedup_pct,
                    infer_us_per_state: t_stats.mean(),
                };

                println!(
                    "{:<10} | {:<11} | {:<9} | {:>10.0} | {:>9.1} | {:>6.1} -> {:<5.1} | {:>7.1}% | {:>9.2} µs",
                    row.sp_workers,
                    row.inf_workers,
                    row.max_batch,
                    row.sims_per_sec,
                    row.games_per_sec,
                    row.batch_mean,
                    row.unique_batch_mean,
                    row.dedup_pct,
                    row.infer_us_per_state,
                );
                results.push(row);
            }
        }
    }

    results.sort_by(|a, b| b.sims_per_sec.total_cmp(&a.sims_per_sec));
    println!();
    println!("=== Top 3 Optimal Pipeline Configurations on This Machine ===");
    for (rank, r) in results.iter().take(3).enumerate() {
        println!(
            "#{rank_num}: sp_workers={sp}, inf_workers={inf}, max_batch_size={mb} -> {sps:.0} sims/sec ({gps:.1} games/sec, {ms:.1} ms, batch={bm:.1}->{um:.1} [dedup {dp:.1}%], {us:.2} µs/state)",
            rank_num = rank + 1,
            sp = r.sp_workers,
            inf = r.inf_workers,
            mb = r.max_batch,
            sps = r.sims_per_sec,
            gps = r.games_per_sec,
            ms = r.elapsed_ms,
            bm = r.batch_mean,
            um = r.unique_batch_mean,
            dp = r.dedup_pct,
            us = r.infer_us_per_state,
        );
    }

    Ok(())
}
