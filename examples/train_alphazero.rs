//! End-to-End Pure-Rust AlphaZero Training, Live Browser Lab, & Evaluation Showcase.
//!
//! Demonstrates:
//! 1. **Interactive Live Browser Lab & Multi-Engine Arena (`http://127.0.0.1:8080`)**:
//!    - Starts in **Idle / Paused** state when launched with `--serve` (or `./start_lab.sh`) so you
//!      have full Start / Pause / Resume / Step (+1 Iter) / Reset control from the browser.
//!    - Supports switching live between **3x3 Tic-Tac-Toe (`G=27, P=9, N=2`)** and
//!      **9x9 Ultimate Tic-Tac-Toe (`G=324, P=81, N=2`)**.
//!    - Automatically saves `checkpoints/<game>_iter_NNN.safetensors` every `N` iterations so you
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

use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::prelude::{SeedableRng, SmallRng};

use mcts::alphazero::{
    AlphaZeroConfig, AlphaZeroGame, AlphaZeroPlayer, SelfPlayConfig, run_self_play,
    run_self_play_with_initializer,
};
use mcts::game::{Game, PlayerIndex};
use mcts::games::rubiks_cube::{RubiksGame, RubiksState};
use mcts::games::tic_tac_toe::TicTacToe;
use mcts::games::ultimate_ttt::{INPUT_SIZE as UTTT_INPUT_SIZE, UltimateTicTacToe};
use mcts::inference::InferenceService;
use mcts::matchup::{RandomPlayer, run_multi_match_with_openings};
use mcts::elo::{EloPool, Pairing};
use mcts::experiments::{ExperimentPoint, ExperimentSpec, RunStatus, score_pct};
use mcts::hyperparams::AlphaZeroHyperparams;
use mcts::mcts::{MctsConfig, MctsPlayer, PlayoutEvaluator, Uct};
use mcts::minimax::MinimaxPlayer;
use mcts::backend::{Backend, NetEvaluator, NnDevice};
use mcts::checkpoints;
use mcts::nn::AlphaZeroNetConfig;
use mcts::player::Player;
use mcts::opponents::OpponentSpec;
use mcts::play::PlayableGame;
use mcts::telemetry::{
    ArenaResult, DeviceChoice, IterationTelemetry, LabGame, LiveKnobs, SystemSettings, TelemetryHub,
    TrainingCommand, default_eval_opponents,
};
use mcts::training::{AlphaZeroTrainer, ReplayBuffer};

/// What the Browser Lab needs on top of [`PlayableGame`] to train and evaluate a two-player game.
/// Start position and Minimax heuristic come from [`PlayableGame`]; network, symmetries and
/// hyperparameter defaults from the game's [`AlphaZeroGame`] implementation.
trait LabGameSpec<const G: usize, const P: usize>: PlayableGame<G, P> + Sync + 'static {
    const LAB: LabGame;
    /// Short id used in checkpoint file names and logs (`ttt`, `uttt`).
    const TAG: &'static str;
    /// Base seed for self-play and evaluation of this game.
    const SEED: u64;
}

impl LabGameSpec<27, 9> for TicTacToe {
    const LAB: LabGame = LabGame::Ttt;
    const TAG: &'static str = "ttt";
    const SEED: u64 = 1000;
}

impl LabGameSpec<{ UTTT_INPUT_SIZE }, 81> for UltimateTicTacToe {
    const LAB: LabGame = LabGame::Uttt;
    const TAG: &'static str = "uttt";
    const SEED: u64 = 5000;
}

/// One lab iteration for any [`LabGameSpec`]: self-play -> SGD -> arena evaluation -> checkpoint.
#[allow(clippy::too_many_arguments)]
fn run_iteration<S, const G: usize, const P: usize>(
    iter: usize,
    hub: &TelemetryHub,
    trainer: &mut AlphaZeroTrainer<G, P, 2>,
    service: &Arc<InferenceService<NetEvaluator<G, P, 2>, G, P, 2>>,
    buffer: &mut ReplayBuffer<G, P, 2>,
    max_batch_size: usize,
    rng: &mut SmallRng,
) -> candle_core::Result<()>
where
    S: LabGameSpec<G, P>,
    S::State: Sync,
{
    let game = S::LAB;
    let tag = S::TAG.to_uppercase();
    let knobs = hub.knobs_for(game);
    trainer.set_learning_rate(knobs.learning_rate);
    trainer.set_weight_decay(knobs.weight_decay);
    buffer.set_capacity(knobs.buffer_capacity);
    service.reset_stats();
    let root = S::start();

    // 1. Self-play
    let sp_phase = format!(
        "Iter #{iter}: Self-Play ({} games × {} sims)...",
        knobs.games_per_iteration, knobs.simulations_per_move
    );
    println!("⏳ [{tag:<4} #{iter:>3}] {sp_phase}");
    hub.set_phase_for(game, iter, sp_phase);
    let sp_start = Instant::now();
    let sp_res = run_self_play(&root, service, &knobs.self_play_config(S::SEED + iter as u64 * 131));
    let self_play_ms = (sp_start.elapsed().as_secs_f64() * 1000.0) as f32;
    buffer.extend(&sp_res.positions);

    let batch_stats = service.batch_size_stats();
    let uniq_batch_stats = service.unique_batch_size_stats();
    let dedup_saved_pct = service.dedup_savings_ratio() * 100.0;
    let infer_stats = service.inference_time_stats();
    let approx_sims = (sp_res.positions.len() * knobs.simulations_per_move) as f32;
    let simulations_per_sec = approx_sims / (self_play_ms / 1000.0).max(1e-6);

    // 2. SGD training
    hub.set_phase_for(
        game,
        iter,
        format!("Iter #{iter}: SGD Training ({} steps, buffer={})...", knobs.train_steps, buffer.len()),
    );
    let train_start = Instant::now();
    let metrics = trainer.train_epoch(buffer, service, knobs.batch_size, knobs.train_steps, rng)?;
    let train_ms = (train_start.elapsed().as_secs_f64() * 1000.0) as f32;

    // 3. Arena evaluation against the configured opponents (argmax AlphaZero, no noise)
    let eval_start = Instant::now();
    let openings = S::eval_openings(knobs.eval_games.div_ceil(2));
    let arena = play_opponents::<S, G, P>(
        service,
        &openings,
        knobs.eval_simulations(),
        &knobs,
        &hub.eval_opponents_for(game),
        hub.system_settings().train_device(),
        S::SEED + 500_000 + iter as u64 * 10,
        |p| hub.set_phase_for(game, iter, format!("Iter #{iter}: Arena {p}...")),
    );
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
        arena,
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

    let mut arena_txt: Vec<String> =
        rec.arena.iter().map(|a| format!("vs {} {}W-{}D-{}L", a.label, a.wdl[0], a.wdl[1], a.wdl[2])).collect();
    if let Some(r) = hub.elo_pool(game).and_then(|pool| pool.estimate(rec.arena.iter().map(|a| (&a.opponent, a.wdl)))) {
        arena_txt.push(format!("Elo {:.0} ± {:.0}", r.elo, r.stderr));
    }
    println!(
        "[{tag:<4} #{iter:>3}] Loss {:>5.3} | KL {:>6.4} | MSE {:>6.4} | {} | Dedup {:>4.1}% | {:>7.0} sims/s",
        rec.total_loss,
        rec.policy_kl,
        rec.value_loss,
        arena_txt.join(" | "),
        rec.dedup_saved_pct,
        rec.simulations_per_sec,
    );

    trainer.save_checkpoint(checkpoints::path(&format!("{}_alphazero.safetensors", S::TAG)))?;
    let auto_every = hub.auto_save_every_for(game);
    if auto_every > 0 && iter.is_multiple_of(auto_every) {
        let snap = format!("{}_iter_{iter:03}.safetensors", S::TAG);
        trainer.save_checkpoint(checkpoints::path(&snap))?;
        println!("📸 Auto-saved {tag} checkpoint: {snap}");
    }

    hub.record_iteration_for(game, rec);
    let paused_now = hub.knobs_for(game).paused;
    hub.set_phase_for(
        game,
        iter,
        if paused_now { format!("Paused after Iter #{iter}") } else { format!("Completed Iter #{iter}") },
    );
    Ok(())
}

type LabService<const G: usize, const P: usize> = InferenceService<NetEvaluator<G, P, 2>, G, P, 2>;

/// The networks of `opponents` (in order; `None` for engines without one), loaded on `device`.
/// An `Err` says why a checkpoint could not be loaded.
fn load_opponent_networks<S, const G: usize, const P: usize>(
    opponents: &[OpponentSpec],
    device: NnDevice,
) -> Vec<Result<Option<LabService<G, P>>, String>>
where
    S: LabGameSpec<G, P>,
{
    opponents
        .iter()
        .map(|o| match o {
            OpponentSpec::AlphaZero { checkpoint, .. } => {
                let path = checkpoints::path(checkpoint);
                let cfg = checkpoints::net_config::<S, G, P, 2>(&path)?;
                let ev = NetEvaluator::<G, P, 2>::standalone(&cfg, &[device], Some(&path)).map_err(|e| e.to_string())?;
                Ok(Some(InferenceService::new(ev, 256, 1)))
            }
            _ => Ok(None),
        })
        .collect()
}

/// A player for `spec` starting at `game`; `net` is the network of an AlphaZero opponent.
fn opponent_player<'a, S, const G: usize, const P: usize>(
    spec: &OpponentSpec,
    game: &Game<S, 2>,
    seed: u64,
    net: Option<&'a LabService<G, P>>,
) -> Box<dyn Player<S, 2> + 'a>
where
    S: LabGameSpec<G, P>,
{
    match spec {
        OpponentSpec::Random => Box::new(RandomPlayer::new(game.clone(), seed)),
        OpponentSpec::Mcts { sims, exploration } => Box::new(MctsPlayer::new(
            game.clone(),
            S::SENTINEL_MOVE,
            Uct::new(*exploration),
            PlayoutEvaluator::new(seed ^ 0x55),
            MctsConfig::new(*sims, seed).with_arena_capacity(sims * 16),
        )),
        OpponentSpec::Minimax { depth } => Box::new(MinimaxPlayer::new(game.clone(), S::heuristic(), *depth, seed)),
        OpponentSpec::AlphaZero { sims, .. } => Box::new(AlphaZeroPlayer::new(
            game.clone(),
            net.expect("AlphaZero opponents come with their network"),
            AlphaZeroConfig::new(1, *sims, seed).with_dirichlet(0.3, 0.0),
        )),
    }
}

/// Plays argmax AlphaZero (`eval_sims` per move, no noise) on `service` against each opponent from
/// the fixed `openings` (colors swapped). An AlphaZero opponent's checkpoint is loaded on `device`;
/// if that fails, the opponent is reported with 0 games and the error in its label.
#[allow(clippy::too_many_arguments)]
fn play_opponents<S, const G: usize, const P: usize>(
    service: &LabService<G, P>,
    openings: &[Game<S, 2>],
    eval_sims: usize,
    hp: &AlphaZeroHyperparams,
    opponents: &[OpponentSpec],
    device: NnDevice,
    seed: u64,
    mut phase: impl FnMut(String),
) -> Vec<ArenaResult>
where
    S: LabGameSpec<G, P>,
    S::State: Sync,
{
    let threads = hp.eval_thread_count();
    let nets = load_opponent_networks::<S, G, P>(opponents, device);
    let mut results = Vec::with_capacity(opponents.len());
    for (k, (opp, net)) in opponents.iter().zip(&nets).enumerate() {
        let label = opp.label();
        let net = match net {
            Ok(net) => net.as_ref(),
            Err(e) => {
                println!("⚠️  Opponent '{label}' unavailable: {e}");
                results.push(ArenaResult { opponent: opp.clone(), label: format!("{label} (⚠ {e})"), wdl: [0, 0, 0] });
                continue;
            }
        };
        phase(format!("eval {}/{} vs {label}", k + 1, opponents.len()));
        // Defined per opponent: both factories passed together must share one borrow lifetime.
        let f_az = |g: &Game<S, 2>, seed: u64| -> Box<dyn Player<S, 2> + '_> {
            Box::new(AlphaZeroPlayer::new(
                g.clone(),
                service,
                AlphaZeroConfig::new(1, eval_sims, seed)
                    .with_puct(hp.c_base, hp.c_init)
                    .with_fpu(hp.fpu)
                    .with_dirichlet(0.3, 0.0),
            ))
        };
        let f_opp = |g: &Game<S, 2>, seed: u64| opponent_player::<S, G, P>(opp, g, seed, net);
        let r = run_multi_match_with_openings(openings, [&f_az, &f_opp], threads, seed + 1 + k as u64);
        results.push(ArenaResult { opponent: opp.clone(), label, wdl: [r.wins[0], r.draws, r.wins[1]] });
    }
    results
}

/// Plays a round-robin between `anchors` (`games_per_pair` games per pairing from the evaluation
/// openings, colors swapped) and fits their Elo ratings.
fn calibrate_elo<S, const G: usize, const P: usize>(
    hub: &TelemetryHub,
    anchors: Vec<OpponentSpec>,
    games_per_pair: usize,
) -> Result<EloPool, String>
where
    S: LabGameSpec<G, P>,
    S::State: Sync,
{
    let nets = load_opponent_networks::<S, G, P>(&anchors, hub.system_settings().train_device())
        .into_iter()
        .zip(&anchors)
        .map(|(net, a)| net.map_err(|e| format!("{}: {e}", a.label())))
        .collect::<Result<Vec<_>, _>>()?;
    let openings = S::eval_openings(games_per_pair.div_ceil(2));
    let threads = hub.knobs_for(S::LAB).eval_thread_count();
    let pairs: Vec<(usize, usize)> = (0..anchors.len()).flat_map(|a| (a + 1..anchors.len()).map(move |b| (a, b))).collect();
    let mut pairings = Vec::with_capacity(pairs.len());
    for (k, &(a, b)) in pairs.iter().enumerate() {
        hub.set_elo_status(S::LAB, format!("Pairing {}/{}: {} vs {}", k + 1, pairs.len(), anchors[a].label(), anchors[b].label()));
        let f_a = |g: &Game<S, 2>, seed: u64| opponent_player::<S, G, P>(&anchors[a], g, seed, nets[a].as_ref());
        let f_b = |g: &Game<S, 2>, seed: u64| opponent_player::<S, G, P>(&anchors[b], g, seed, nets[b].as_ref());
        let r = run_multi_match_with_openings(&openings, [&f_a, &f_b], threads, S::SEED + 700_000 + k as u64 * 10);
        println!("⚖️  {} vs {}: {}W-{}D-{}L", anchors[a].label(), anchors[b].label(), r.wins[0], r.draws, r.wins[1]);
        pairings.push(Pairing { a, b, wdl: [r.wins[0], r.draws, r.wins[1]] });
    }
    Ok(EloPool::calibrate(anchors, pairings, openings.len() * 2))
}

/// Trains a fresh network for experiment `id` until its budget is used up (or it is stopped),
/// evaluating at iteration 0, every `eval_every` iterations and at the end.
fn run_experiment<S, const G: usize, const P: usize>(
    hub: &TelemetryHub,
    id: u64,
    spec: &ExperimentSpec,
) -> candle_core::Result<RunStatus>
where
    S: LabGameSpec<G, P>,
    S::State: Sync,
{
    let backend = Backend::parse(&spec.backend).filter(|b| b.is_available()).unwrap_or(Backend::Candle);
    let dev = if spec.device == "cpu" { NnDevice::cpu(backend) } else { NnDevice::best(backend) };
    let (net_name, cfg) = S::net_presets()
        .into_iter()
        .find(|(n, _)| *n == spec.net)
        .unwrap_or_else(|| S::net_presets()[0]);
    let hp = spec.hp;
    let mut trainer = AlphaZeroTrainer::<G, P, 2>::new(cfg, hp.learning_rate, hp.weight_decay, dev)?;
    if !spec.init_checkpoint.is_empty() {
        let path = checkpoints::path(&spec.init_checkpoint);
        let file_cfg = checkpoints::net_config::<S, G, P, 2>(&path).map_err(candle_core::Error::Msg)?;
        if file_cfg != cfg {
            return Err(candle_core::Error::Msg(format!(
                "'{}' holds a {} network, but this run trains a {net_name} network",
                spec.init_checkpoint,
                checkpoints::preset_name::<S, G, P, 2>(&file_cfg)
            )));
        }
        trainer.load_checkpoint(&path)?;
    }
    let infer_devices = dev.replicas(spec.infer_workers);
    let service = trainer.build_inference_service(spec.infer_batch, &infer_devices)?;
    let mut buffer = ReplayBuffer::<G, P, 2>::new(hp.buffer_capacity).with_symmetries(S::symmetries());
    let mut rng = SmallRng::seed_from_u64(S::SEED ^ (id * 7919));
    let opponents = if spec.opponents.is_empty() { default_eval_opponents(S::LAB) } else { spec.opponents.clone() };
    let start_from = if spec.init_checkpoint.is_empty() { String::new() } else { format!(" · from {}", spec.init_checkpoint) };
    let device = device_label(dev, &infer_devices, spec.infer_batch);
    hub.experiment_started(id, opponents.iter().map(OpponentSpec::label).collect(), format!("{device} · {net_name} net{start_from}"));
    println!("🧪 Experiment #{id} '{}' on {device} ({net_name} net, {} min budget)", spec.name, spec.budget_minutes);

    let openings = S::eval_openings(spec.eval_games.div_ceil(2));
    let start = Instant::now();
    let (mut train_secs, mut games, mut positions) = (0.0f64, 0usize, 0usize);
    let (mut value_loss, mut policy_kl, mut sims_per_sec) = (0.0f32, 0.0f32, 0.0f32);
    let evaluate = |iter: usize, train_secs: f64, games: usize, positions: usize, vl: f32, kl: f32, sps: f32| {
        let wdl: Vec<[usize; 3]> =
            play_opponents::<S, G, P>(&service, &openings, spec.eval_sims, &hp, &opponents, dev, 9_000 + iter as u64 * 10, |p| {
                hub.experiment_phase(id, format!("iter {iter}: {p}"));
            })
            .iter()
            .map(|a| a.wdl)
            .collect();
        let scores: Vec<f32> = wdl.iter().map(|&[w, d, l]| score_pct(w, d, w + d + l)).collect();
        println!("🧪 #{id} iter {iter:>3} | {:>5.1} train-min | {games:>6} games | scores {scores:?}", train_secs / 60.0);
        hub.experiment_point(id, ExperimentPoint {
            iteration: iter,
            games,
            positions,
            train_secs,
            wall_secs: start.elapsed().as_secs_f64(),
            scores,
            wdl,
            value_loss: vl,
            policy_kl: kl,
            sims_per_sec: sps,
        });
    };
    evaluate(0, 0.0, 0, 0, 0.0, 0.0, 0.0);

    let budget_secs = spec.budget_minutes * 60.0;
    let mut iter = 0usize;
    let mut last_eval = 0usize;
    let status = loop {
        if hub.experiment_should_stop(id) {
            break RunStatus::Stopped;
        }
        if train_secs >= budget_secs || (spec.max_iterations > 0 && iter >= spec.max_iterations) {
            break RunStatus::Done;
        }
        iter += 1;
        hub.experiment_phase(id, format!("iter {iter}: self-play + training ({:.1}/{} min)", train_secs / 60.0, spec.budget_minutes));
        let t = Instant::now();
        service.reset_stats();
        let sp = run_self_play(&S::start(), &service, &hp.self_play_config(S::SEED + 31 * id + iter as u64 * 131));
        let sp_secs = t.elapsed().as_secs_f64();
        buffer.extend(&sp.positions);
        let m = trainer.train_epoch(&buffer, &service, hp.batch_size, hp.train_steps, &mut rng)?;
        train_secs += t.elapsed().as_secs_f64();
        games += sp.total_games;
        positions += sp.positions.len();
        (value_loss, policy_kl) = (m.value_loss, m.policy_kl);
        sims_per_sec = (sp.positions.len() * hp.simulations_per_move) as f32 / sp_secs.max(1e-6) as f32;
        if iter.is_multiple_of(spec.eval_every) {
            evaluate(iter, train_secs, games, positions, value_loss, policy_kl, sims_per_sec);
            last_eval = iter;
        }
    };
    if last_eval != iter {
        evaluate(iter, train_secs, games, positions, value_loss, policy_kl, sims_per_sec);
    }
    let ckpt = format!("{}_exp{id}_{net_name}.safetensors", S::TAG);
    trainer.save_checkpoint(checkpoints::path(&ckpt))?;
    println!("🧪 Experiment #{id} finished: {status:?} (weights: {ckpt})");
    Ok(status)
}

fn device_label(train: NnDevice, infer: &[NnDevice], infer_batch: usize) -> String {
    format!("{train} · {} inference worker(s) · batch {infer_batch}", infer.len())
}

type LiveService<const G: usize, const P: usize, const N: usize> =
    Arc<InferenceService<NetEvaluator<G, P, N>, G, P, N>>;

/// Moves a live model onto new devices: copies the weights into a fresh trainer on `train_dev`
/// (the AdamW moment estimates restart) and builds a new inference service around it.
#[allow(clippy::too_many_arguments)]
fn rebuild_model<const G: usize, const P: usize, const N: usize>(
    tag: &str,
    trainer: &mut AlphaZeroTrainer<G, P, N>,
    service: &mut LiveService<G, P, N>,
    lr: f64,
    weight_decay: f64,
    train_dev: NnDevice,
    infer_devs: &[NnDevice],
    infer_batch: usize,
) -> candle_core::Result<()> {
    let tmp = std::env::temp_dir().join(format!("mcts_lab_{tag}_{}.safetensors", std::process::id()));
    trainer.save_checkpoint(&tmp)?;
    let mut fresh = AlphaZeroTrainer::<G, P, N>::new(*trainer.net_config(), lr, weight_decay, train_dev)?;
    fresh.load_checkpoint(&tmp)?;
    let _ = std::fs::remove_file(&tmp);
    *service = Arc::new(fresh.build_inference_service(infer_batch, infer_devs)?);
    *trainer = fresh;
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let interactive_lab =
        std::env::args().any(|a| a == "--serve" || a == "--lab-only");
    let bind_addr =
        std::env::var("MCTS_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    // Backend: `--backend=candle|tch` (or `MCTS_BACKEND`); default libtorch when compiled in.
    let backend = std::env::args()
        .find_map(|a| a.strip_prefix("--backend=").map(str::to_string))
        .or_else(|| std::env::var("MCTS_BACKEND").ok())
        .and_then(|b| Backend::parse(&b))
        .filter(|b| b.is_available())
        .unwrap_or(if Backend::Tch.is_available() { Backend::Tch } else { Backend::Candle });
    let mut system = SystemSettings {
        backend,
        device: DeviceChoice::Gpu,
        // A network call costs about the same for 16 or 256 positions (fixed per-call overhead), so
        // batches should be as large as the games in flight allow: 512 serves up to 1024 games.
        infer_batch: 512,
        infer_workers: 8,
    };
    let (train_device, infer_devices) = system.devices();
    system.infer_workers = infer_devices.len();
    let label = device_label(train_device, &infer_devices, system.infer_batch);

    println!("=== Pure-Rust AlphaZero Training Pipeline & Live Lab ===");
    println!("Training device:  {train_device}  (backends compiled in: {:?})", Backend::available());
    println!(
        "Inference devices: {} worker(s) on {}",
        infer_devices.len(),
        infer_devices[0]
    );

    let iterations = 15usize;
    let mut initial_knobs = LiveKnobs::default_ttt();
    // In interactive Browser Lab mode (`--serve` / `./start_lab.sh`), start IDLE (`paused = true`)
    // so training only runs when the user clicks Start / Resume / Step in the Browser Lab.
    initial_knobs.paused = interactive_lab;

    let hub = TelemetryHub::new(label, iterations, initial_knobs);
    if interactive_lab {
        hub.enable_experiment_store("lab_experiments.json");
        hub.enable_elo_pools();
    }
    hub.set_system_info(system);
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

    let mut max_batch_size = system.infer_batch;

    // 1. Initialize Tic-Tac-Toe (`G=27, P=9, N=2`) Trainer & Live Inference Service
    let ttt_net_cfg = <TicTacToe as AlphaZeroGame<27, 9, 2>>::net_config();
    let mut ttt_trainer = AlphaZeroTrainer::<27, 9, 2>::new(
        ttt_net_cfg,
        initial_knobs.learning_rate,
        1e-4,
        train_device,
    )?;
    let ttt_live_file = checkpoints::path("ttt_alphazero.safetensors");
    if interactive_lab && ttt_live_file.exists() {
        ttt_trainer.load_checkpoint(&ttt_live_file)?;
        println!("📦 Loaded existing TTT checkpoint into live model: {}", ttt_live_file.display());
    }
    let mut ttt_service = Arc::new(ttt_trainer.build_inference_service(max_batch_size, &infer_devices)?);
    hub.attach_ttt_service(Arc::clone(&ttt_service), ttt_net_cfg, infer_devices.clone());
    let mut ttt_buffer = ReplayBuffer::<27, 9, 2>::new(initial_knobs.buffer_capacity)
        .with_symmetries(<TicTacToe as AlphaZeroGame<27, 9, 2>>::symmetries());

    // 2. Initialize Ultimate Tic-Tac-Toe (`G=324, P=81, N=2`) Trainer & Live Inference Service
    let uttt_net_cfg = <UltimateTicTacToe as AlphaZeroGame<{ UTTT_INPUT_SIZE }, 81, 2>>::net_config();
    let mut uttt_trainer = AlphaZeroTrainer::<{ UTTT_INPUT_SIZE }, 81, 2>::new(
        uttt_net_cfg,
        2.0e-3,
        1e-4,
        train_device,
    )?;
    let uttt_live_file = checkpoints::path("uttt_alphazero.safetensors");
    if interactive_lab && uttt_live_file.exists() {
        type Uttt = UltimateTicTacToe;
        match checkpoints::net_config::<Uttt, { UTTT_INPUT_SIZE }, 81, 2>(&uttt_live_file) {
            Ok(cfg) if cfg == uttt_net_cfg => {
                uttt_trainer.load_checkpoint(&uttt_live_file)?;
                println!("📦 Loaded existing UTTT checkpoint into live model: {}", uttt_live_file.display());
            }
            found => println!(
                "⚠️  Not loading {} (starting fresh): it holds {}, the live UTTT network is a {} network",
                uttt_live_file.display(),
                found.map_or_else(|e| e, |c| format!("a {} network", checkpoints::preset_name::<Uttt, { UTTT_INPUT_SIZE }, 81, 2>(&c))),
                checkpoints::preset_name::<Uttt, { UTTT_INPUT_SIZE }, 81, 2>(&uttt_net_cfg)
            ),
        }
    }
    let mut uttt_service =
        Arc::new(uttt_trainer.build_inference_service(max_batch_size, &infer_devices)?);
    hub.attach_uttt_service(
        Arc::clone(&uttt_service),
        uttt_net_cfg,
        infer_devices.clone(),
    );
    // The value head generalizes only across *distinct games* (all ~50 positions of a game share
    // one outcome and nearly the same board, so a small window lets the net memorize which game a
    // position came from). The default 300k positions keep ~6k recent games (~0.8 GB); the 8
    // board symmetries multiply the effective data again. Resized live via the buffer knob.
    let mut uttt_buffer =
        ReplayBuffer::<{ UTTT_INPUT_SIZE }, 81, 2>::new(LiveKnobs::default_uttt().buffer_capacity)
            .with_symmetries(<UltimateTicTacToe as AlphaZeroGame<{ UTTT_INPUT_SIZE }, 81, 2>>::symmetries());

    let mut rng = SmallRng::seed_from_u64(2026);

    if interactive_lab {
        println!("⏸  Browser Lab started in IDLE state at http://{bind_addr}");
        println!("   • Open http://{bind_addr} to choose Tic-Tac-Toe (3x3) or Ultimate Tic-Tac-Toe (9x9)");
        println!("   • Use ▶/⏸ Start-Pause, ⏭ Step (+1 Iter), or 🔄 Reset in the Browser Lab");
        println!("   • Press Ctrl+C in this terminal to stop the server.");
        println!();

        loop {
            if let Some(new_sys) = hub.take_system_change() {
                let (train_dev, infer_devs) = new_sys.devices();
                let ttt_k = hub.knobs_for(LabGame::Ttt);
                rebuild_model("ttt", &mut ttt_trainer, &mut ttt_service, ttt_k.learning_rate, ttt_k.weight_decay, train_dev, &infer_devs, new_sys.infer_batch)?;
                hub.attach_ttt_service(Arc::clone(&ttt_service), ttt_net_cfg, infer_devs.clone());
                let uttt_k = hub.knobs_for(LabGame::Uttt);
                rebuild_model("uttt", &mut uttt_trainer, &mut uttt_service, uttt_k.learning_rate, uttt_k.weight_decay, train_dev, &infer_devs, new_sys.infer_batch)?;
                hub.attach_uttt_service(Arc::clone(&uttt_service), uttt_net_cfg, infer_devs.clone());
                max_batch_size = new_sys.infer_batch;
                // Report what is actually in effect (e.g. Metal always uses one worker).
                let applied = SystemSettings { infer_workers: infer_devs.len(), ..new_sys };
                hub.set_system_info(applied);
                let label = device_label(train_dev, &infer_devs, new_sys.infer_batch);
                println!("🖥️  Rebuilt both models on {label}");
                hub.set_device_label(label);
            }
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
                TrainingCommand::RunExperiment { id, spec } => {
                    let res = match spec.game.as_str() {
                        "ttt" => run_experiment::<TicTacToe, 27, 9>(&hub, id, &spec),
                        _ => run_experiment::<UltimateTicTacToe, { UTTT_INPUT_SIZE }, 81>(&hub, id, &spec),
                    };
                    let status = res.unwrap_or_else(|e| {
                        println!("❌ Experiment #{id} failed: {e}");
                        RunStatus::Failed(e.to_string())
                    });
                    hub.experiment_finished(id, status);
                }
                TrainingCommand::CalibrateElo { game, anchors, games_per_pair } => {
                    let result = match game {
                        LabGame::Ttt => calibrate_elo::<TicTacToe, 27, 9>(&hub, anchors, games_per_pair),
                        LabGame::Uttt => calibrate_elo::<UltimateTicTacToe, { UTTT_INPUT_SIZE }, 81>(&hub, anchors, games_per_pair),
                    };
                    hub.finish_elo_calibration(game, result);
                }
                TrainingCommand::RunIteration {
                    game: LabGame::Ttt,
                    iteration,
                } => {
                    run_iteration::<TicTacToe, 27, 9>(
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
                    run_iteration::<UltimateTicTacToe, { UTTT_INPUT_SIZE }, 81>(
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
        run_iteration::<TicTacToe, 27, 9>(
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
