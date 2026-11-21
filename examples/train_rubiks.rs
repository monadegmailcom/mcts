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
//!   `budget_max` (`2`, `10`, `100`)
//! - self-play: `games=512`, `sims=200`, `discount=0.9`, `alpha=0.5`, `eps=0.25`
//! - training: `steps=500`, `batch=512`, `lr=0.001`, `buffer=500000`
//! - evaluation: `eval_games=128`, `eval_sims=400`
//! - compute: `backend=tch|candle`, `device=gpu|cpu`, `infer_batch=512`

use std::time::Instant;

use rand::prelude::{SeedableRng, SmallRng};

use mcts::alphazero::{AlphaZeroGame, Episode, SelfPlayConfig, run_chained_self_play};
use mcts::backend::{Backend, NetEvaluator, NnDevice};
use mcts::checkpoints;
use mcts::games::rubiks_cube::{INPUT_SIZE, POLICY_SIZE, RoundGame, RubiksCube, RubiksSettings};
use mcts::inference::InferenceService;
use mcts::training::{AlphaZeroTrainer, ReplayBuffer};

type Service = InferenceService<NetEvaluator<INPUT_SIZE, POLICY_SIZE, 1>, INPUT_SIZE, POLICY_SIZE, 1>;

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

/// How the games of one self-play or evaluation run went, per round.
struct RoundStats {
    scores: Vec<f64>,
    /// `reached[n]` / `solved[n]`: games that played / solved round `n + 1`.
    reached: Vec<usize>,
    solved: Vec<usize>,
    /// Total moves of the solved rounds `n + 1`.
    solve_moves: Vec<usize>,
}

impl RoundStats {
    fn new(games: &[Vec<Episode<1>>], max_rounds: usize) -> Self {
        let mut stats = Self {
            scores: games.iter().map(|g| RoundGame::score(g)).collect(),
            reached: vec![0; max_rounds],
            solved: vec![0; max_rounds],
            solve_moves: vec![0; max_rounds],
        };
        for game in games {
            for (n, round) in game.iter().enumerate() {
                stats.reached[n] += 1;
                if round.payoffs[0] > 0.0 {
                    stats.solved[n] += 1;
                    stats.solve_moves[n] += round.moves;
                }
            }
        }
        stats
    }

    fn mean_score(&self) -> f64 {
        self.scores.iter().sum::<f64>() / self.scores.len().max(1) as f64
    }

    /// Standard error of the mean score.
    fn score_stderr(&self) -> f64 {
        let n = self.scores.len() as f64;
        let mean = self.mean_score();
        (self.scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0) / n.max(1.0)).sqrt()
    }

    fn best_round(&self) -> usize {
        self.solved.iter().rposition(|&s| s > 0).map_or(0, |n| n + 1)
    }

    /// One line per round that was reached: solve rate and average moves of the solves.
    fn table(&self) -> String {
        let mut out = String::from("      round | reached | solved | moves (avg)\n");
        for n in 0..self.reached.len() {
            if self.reached[n] == 0 {
                break;
            }
            let rate = 100.0 * self.solved[n] as f64 / self.reached[n] as f64;
            let moves = if self.solved[n] > 0 {
                format!("{:.1}", self.solve_moves[n] as f64 / self.solved[n] as f64)
            } else {
                "-".into()
            };
            out += &format!("      {:>5} | {:>7} | {rate:>5.0}% | {moves:>6}\n", n + 1, self.reached[n]);
        }
        out
    }
}

fn play(service: &Service, config: &SelfPlayConfig, game: RoundGame) -> mcts::alphazero::SelfPlayResult<INPUT_SIZE, POLICY_SIZE, 1> {
    run_chained_self_play(service, config, &|played, rng| game.next_round(played, rng))
}

fn main() -> candle_core::Result<()> {
    let args = Args(std::env::args().skip(1).collect());
    let hp = RubiksCube::hyperparams();
    let settings = RubiksSettings::default();
    let defaults = settings.rounds;
    let game = RoundGame {
        max_rounds: args.get("rounds", defaults.max_rounds),
        moves_per_turn: args.get("budget_per_turn", defaults.moves_per_turn),
        extra_moves: args.get("budget_extra", defaults.extra_moves),
        max_moves: args.get("budget_max", defaults.max_moves),
    };
    let minutes: f64 = args.get("minutes", 60.0);
    let max_iterations: usize = args.get("iterations", 0);
    let eval_every: usize = args.get::<usize>("eval_every", 1).max(1);
    let save = checkpoints::sanitize_name(&args.get("save", "rubiks_rounds".to_string())).expect("valid save name");
    let init: String = args.get("init", String::new());
    let games: usize = args.get("games", hp.games_per_iteration);
    let sims: usize = args.get("sims", hp.simulations_per_move);
    let discount: f32 = args.get("discount", settings.discount);
    let (steps, batch): (usize, usize) = (args.get("steps", hp.train_steps), args.get("batch", hp.batch_size));
    let lr: f64 = args.get("lr", hp.learning_rate);
    let buffer_capacity: usize = args.get("buffer", hp.buffer_capacity);
    let (eval_games, eval_sims): (usize, usize) = (args.get("eval_games", hp.eval_games), args.get("eval_sims", hp.eval_sims));
    let infer_batch: usize = args.get("infer_batch", 512);

    let default_backend = if Backend::Tch.is_available() { "tch" } else { "candle" };
    let backend = Backend::parse(&args.get("backend", default_backend.to_string()))
        .filter(|b| b.is_available())
        .expect("backend not available (tch needs --features tch)");
    let device = if args.get("device", "gpu".to_string()) == "cpu" { NnDevice::cpu(backend) } else { NnDevice::best(backend) };

    let net = RubiksCube::net_config();
    let mut trainer = AlphaZeroTrainer::<INPUT_SIZE, POLICY_SIZE, 1>::new(net, lr, hp.weight_decay, device)?;
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
    let mut buffer = ReplayBuffer::<INPUT_SIZE, POLICY_SIZE, 1>::new(buffer_capacity);
    let mut rng = SmallRng::seed_from_u64(7);

    let self_play = |iter: usize| {
        SelfPlayConfig::new(games, sims, 1000 + iter as u64 * 131)
            .with_workers(mcts::hyperparams::resolve_threads(0))
            .with_opening_moves(hp.opening_moves)
            .with_puct(hp.c_base, hp.c_init)
            .with_fpu(hp.fpu)
            .with_dirichlet(args.get("alpha", hp.dirichlet_alpha), args.get("eps", hp.dirichlet_epsilon))
            .with_discount(discount)
    };
    let evaluation = |iter: usize| {
        SelfPlayConfig::new(eval_games, eval_sims, 900_000 + iter as u64)
            .with_workers(mcts::hyperparams::resolve_threads(0))
            .with_opening_moves(0)
            .with_puct(hp.c_base, hp.c_init)
            .with_fpu(hp.fpu)
            .with_dirichlet(0.3, 0.0)
            .with_discount(discount)
    };

    println!(
        "=== Rubik's cube round game: up to {} rounds, round n allows {}n + {} moves (at most {}) ===",
        game.max_rounds, game.moves_per_turn, game.extra_moves, game.max_moves
    );
    println!("{device} · {net:?}");
    println!(
        "self-play {games} games × {sims} sims (γ {discount}) · {steps} steps × {batch} (lr {lr}) · eval {eval_games} games × {eval_sims} sims · {minutes} min"
    );
    let evaluate = |iter: usize| {
        let t = Instant::now();
        let stats = RoundStats::new(&play(&service, &evaluation(iter), game).episodes, game.max_rounds as usize);
        println!(
            "   eval: score {:.2} ± {:.2} · best round {} · {:.0} s\n{}",
            stats.mean_score(),
            stats.score_stderr(),
            stats.best_round(),
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
        let sp = play(&service, &self_play(iter), game);
        let sp_secs = t.elapsed().as_secs_f64();
        buffer.extend(&sp.positions);
        let m = trainer.train_epoch(&buffer, &service, batch, steps, &mut rng)?;
        train_secs += t.elapsed().as_secs_f64();
        trainer.save_checkpoint(checkpoints::path(&save))?;

        let stats = RoundStats::new(&sp.episodes, game.max_rounds as usize);
        println!(
            "iter {iter:>3} | {:>5.1} min | self-play: score {:.2}, best round {}, {} positions, {:.0}k sims/s | KL {:.3} · value loss {:.4} | buffer {}",
            train_secs / 60.0,
            stats.mean_score(),
            stats.best_round(),
            sp.positions.len(),
            sp.positions.len() as f64 * sims as f64 / sp_secs / 1000.0,
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
