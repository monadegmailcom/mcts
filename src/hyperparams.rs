//! Game-independent AlphaZero training hyperparameters.
//!
//! Every game starts from [`AlphaZeroHyperparams::GENERIC`] and may override it through
//! [`crate::alphazero::AlphaZeroGame::hyperparams`]; the Browser Lab exposes all fields as live
//! knobs.

use crate::alphazero::{SelfPlayConfig, ValueQSchedule};
use crate::mcts::Fpu;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AlphaZeroHyperparams {
    // --- Self-play ---
    pub games_per_iteration: usize,
    pub simulations_per_move: usize,
    /// Self-play CPU worker threads (`0` = auto: all cores but two).
    pub self_play_threads: usize,
    /// Number of opening moves per game sampled proportionally to visit counts (temperature 1);
    /// later moves play the most-visited move.
    pub opening_moves: usize,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub c_init: f32,
    pub c_base: f32,
    /// `Q` the search assumes for untried moves; JSON `{"absolute": 0.0}` or `{"relative": 0.25}`.
    pub fpu: Fpu,
    /// Weight `qw` of the root search value in the value target: `(1 - qw) * z + qw * q`
    /// (`z` = game outcome). `0` is pure AlphaZero.
    pub value_q_weight: f32,
    /// How `qw` varies over a game: `"constant"` or `"linear_to_end"` (full on the first move, `0` on the last).
    pub value_q_schedule: ValueQSchedule,
    // --- SGD training ---
    pub train_steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    /// Replay buffer capacity in positions (oldest positions are dropped first).
    pub buffer_capacity: usize,
    // --- Arena evaluation ---
    /// Games per arena opponent (each opening is played once per seat).
    pub eval_games: usize,
    /// AlphaZero simulations per move during evaluation (`0` = same as self-play).
    pub eval_sims: usize,
    /// Arena worker threads (`0` = auto: all cores but two).
    pub eval_threads: usize,
}

impl AlphaZeroHyperparams {
    /// Defaults for a game that does not override them.
    pub const GENERIC: Self = Self {
        games_per_iteration: 128,
        simulations_per_move: 64,
        self_play_threads: 0,
        opening_moves: 4,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        c_init: 1.25,
        c_base: 19_652.0,
        fpu: Fpu::Absolute(0.0),
        value_q_weight: 0.0,
        value_q_schedule: ValueQSchedule::Constant,
        train_steps: 100,
        batch_size: 128,
        learning_rate: 1e-3,
        weight_decay: 1e-4,
        buffer_capacity: 100_000,
        eval_games: 40,
        eval_sims: 0,
        eval_threads: 0,
    };

    /// Self-play configuration for one iteration (`seed` should differ per iteration).
    pub fn self_play_config(&self, seed: u64) -> SelfPlayConfig {
        SelfPlayConfig::new(self.games_per_iteration, self.simulations_per_move, seed)
            .with_workers(resolve_threads(self.self_play_threads))
            .with_opening_moves(self.opening_moves)
            .with_puct(self.c_base, self.c_init)
            .with_fpu(self.fpu)
            .with_dirichlet(self.dirichlet_alpha, self.dirichlet_epsilon)
            .with_value_q_weight(self.value_q_weight)
            .with_value_q_schedule(self.value_q_schedule)
    }

    /// AlphaZero simulations per move in the arena (`0` = the self-play budget the net is trained for).
    pub fn eval_simulations(&self) -> usize {
        if self.eval_sims > 0 { self.eval_sims } else { self.simulations_per_move.max(16) }
    }

    pub fn eval_thread_count(&self) -> usize {
        resolve_threads(self.eval_threads)
    }
}

impl Default for AlphaZeroHyperparams {
    fn default() -> Self {
        Self::GENERIC
    }
}

/// Worker thread count for a setting where `0` means auto: all cores but two (one for the
/// inference worker, one for everything else).
pub fn resolve_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2))
        .unwrap_or(4)
        .max(2)
}
