//! Trains an AlphaZero network to solve the 3x3x3 Rubik's cube by playing the round game
//! ([`RoundGame`]): round `n` is a cube scrambled with `n` quarter turns; solving it within
//! `2n + 10` moves (at most 100) leads to round `n + 1`, failing ends the game. A game scores `n + 1/r` for its
//! last solved round `n`, solved in `r` moves.
//!
//! Every iteration plays self-play games (with root noise), trains on the new positions, and
//! evaluates the network in noise-free games, printing the average game score, how often each
//! round is solved and in how many moves. The weights are saved to
//! `checkpoints/<save>.safetensors` after every iteration, so a run can be continued with
//! `init=<save>.safetensors`.
//!
//! Run (Apple GPU via libtorch, see `start_lab.sh` for the libtorch setup):
//! `cargo run --release --features metal,tch --example train_rubiks -- minutes=60`
//!
//! Options (`key=value`, defaults from [`RubiksCube::hyperparams`] and [`RubiksSettings`]):
//! - run: `minutes=60` (training time, evaluation excluded), `iterations=0` (`0` = no cap),
//!   `eval_every=1`, `save=rubiks_rounds`, `init=` (checkpoint to continue from)
//! - game: `rounds=26`; round `n` allows `budget_per_turn × n + budget_extra` moves, at most
//!   `budget_max` (`2`, `10`, `100`); `first_round=1`: self-play starts there (evaluation always
//!   starts at round 1)
//! - self-play: `games=512`, `sims=200`, `discount=0.9`, `alpha=0.5`, `eps=0.25`
//! - training: `steps=500`, `batch=512`, `lr=0.001`, `buffer=500000`
//! - evaluation: `eval_games=128`, `eval_sims=400`
//! - compute: `backend=tch|candle`, `device=gpu|cpu`, `infer_batch=512`

use std::time::Instant;

use rand::prelude::{SeedableRng, SmallRng};

use mcts::alphazero::AlphaZeroGame;
use mcts::backend::{Backend, NnDevice};
use mcts::checkpoints;
use mcts::games::rubiks_cube::{INPUT_SIZE, POLICY_SIZE, RoundGame, RoundStats, RubiksCube, RubiksSettings};
use mcts::training::{AlphaZeroTrainer, ReplayBuffer};

/// Command-line options as `key=value` pairs.
struct Args(Vec<String>);

impl Args {
    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.0
            .iter()
            .find_map(|a| a.strip_prefix(&format!("{key}=")))
            .map(|v| v.parse().unwrap_or_else(|_| panic!("invalid value for {key}: {v}")))
            .unwrap_or(default)
    }
}

fn main() -> candle_core::Result<()> {
    let args = Args(std::env::args().skip(1).collect());
    let mut hp = RubiksCube::hyperparams();
    hp.games_per_iteration = args.get("games", hp.games_per_iteration);
    hp.simulations_per_move = args.get("sims", hp.simulations_per_move);
    hp.dirichlet_alpha = args.get("alpha", hp.dirichlet_alpha);
    hp.dirichlet_epsilon = args.get("eps", hp.dirichlet_epsilon);
    hp.train_steps = args.get("steps", hp.train_steps);
    hp.batch_size = args.get("batch", hp.batch_size);
    hp.learning_rate = args.get("lr", hp.learning_rate);
    hp.buffer_capacity = args.get("buffer", hp.buffer_capacity);
    hp.eval_games = args.get("eval_games", hp.eval_games);
    hp.eval_sims = args.get("eval_sims", hp.eval_sims);
    let defaults = RubiksSettings::default();
    let settings = RubiksSettings {
        discount: args.get("discount", defaults.discount),
        rounds: RoundGame {
            first_round: args.get("first_round", defaults.rounds.first_round),
            max_rounds: args.get("rounds", defaults.rounds.max_rounds),
            moves_per_turn: args.get("budget_per_turn", defaults.rounds.moves_per_turn),
            extra_moves: args.get("budget_extra", defaults.rounds.extra_moves),
            max_moves: args.get("budget_max", defaults.rounds.max_moves),
        },
    }
    .sanitized();
    let minutes: f64 = args.get("minutes", 60.0);
    let max_iterations: usize = args.get("iterations", 0);
    let eval_every: usize = args.get::<usize>("eval_every", 1).max(1);
    let save = checkpoints::sanitize_name(&args.get("save", "rubiks_rounds".to_string())).expect("valid save name");
    let init: String = args.get("init", String::new());
    let infer_batch: usize = args.get("infer_batch", 512);

    let default_backend = if Backend::Tch.is_available() { "tch" } else { "candle" };
    let backend = Backend::parse(&args.get("backend", default_backend.to_string()))
        .filter(|b| b.is_available())
        .expect("backend not available (tch needs --features tch)");
    let device = if args.get("device", "gpu".to_string()) == "cpu" { NnDevice::cpu(backend) } else { NnDevice::best(backend) };

    let net = RubiksCube::net_config();
    let mut trainer = AlphaZeroTrainer::<INPUT_SIZE, POLICY_SIZE, 1>::new(net, hp.learning_rate, hp.weight_decay, device)?;
    if !init.is_empty() {
        let path = checkpoints::path(&init);
        let file_net = checkpoints::net_config::<RubiksCube, INPUT_SIZE, POLICY_SIZE, 1>(&path).map_err(candle_core::Error::Msg)?;
        if file_net != net {
            return Err(candle_core::Error::Msg(format!("{init} holds a different network: {file_net:?}")));
        }
        trainer.load_checkpoint(&path)?;
        println!("📦 Continuing from {}", path.display());
    }
    let service = trainer.build_inference_service(infer_batch, &device.replicas(1))?;
    let mut buffer = ReplayBuffer::<INPUT_SIZE, POLICY_SIZE, 1>::new(hp.buffer_capacity);
    let mut rng = SmallRng::seed_from_u64(7);

    let r = settings.rounds;
    println!(
        "=== Rubik's cube round game: rounds {} to {} in self-play, round n allows {}n + {} moves (at most {}) ===",
        r.first_round, r.max_rounds, r.moves_per_turn, r.extra_moves, r.max_moves
    );
    println!("{device} · {net:?}");
    println!(
        "self-play {} games × {} sims (γ {}) · {} steps × {} (lr {}) · eval {} games × {} sims · {minutes} min",
        hp.games_per_iteration,
        hp.simulations_per_move,
        settings.discount,
        hp.train_steps,
        hp.batch_size,
        hp.learning_rate,
        hp.eval_games,
        hp.eval_simulations()
    );
    let evaluate = |iter: usize| {
        let t = Instant::now();
        let stats = RoundStats::new(&settings.eval_rounds(), &settings.evaluate(&service, &hp, 900_000 + iter as u64).episodes);
        println!(
            "   eval: score {:.2} ± {:.2} · best round {} · {:.0} s\n{}",
            stats.score,
            stats.score_stderr,
            stats.best_round,
            t.elapsed().as_secs_f64(),
            stats.table()
        );
    };
    evaluate(0);

    let mut train_secs = 0.0;
    let mut iter = 0usize;
    while train_secs < minutes * 60.0 && (max_iterations == 0 || iter < max_iterations) {
        iter += 1;
        let t = Instant::now();
        let sp = settings.self_play(&service, &hp, 1000 + iter as u64 * 131);
        let sp_secs = t.elapsed().as_secs_f64();
        buffer.extend(&sp.positions);
        let m = trainer.train_epoch(&buffer, &service, hp.batch_size, hp.train_steps, &mut rng)?;
        train_secs += t.elapsed().as_secs_f64();
        trainer.save_checkpoint(checkpoints::path(&save))?;

        let stats = RoundStats::new(&settings.rounds, &sp.episodes);
        println!(
            "iter {iter:>3} | {:>5.1} min | self-play: score {:.2}, best round {}, {} positions, {:.0}k sims/s | KL {:.3} · value loss {:.4} | buffer {}",
            train_secs / 60.0,
            stats.score,
            stats.best_round,
            sp.positions.len(),
            sp.positions.len() as f64 * hp.simulations_per_move as f64 / sp_secs / 1000.0,
            m.policy_kl,
            m.value_loss,
            buffer.len()
        );
        if iter.is_multiple_of(eval_every) {
            evaluate(iter);
        }
    }
    println!("💾 Weights: {}", checkpoints::path(&save).display());
    Ok(())
}
