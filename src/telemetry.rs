//! Pure-Rust Live Training Telemetry Hub, Multi-Game Controller (`Tic-Tac-Toe` & `Ultimate Tic-Tac-Toe`),
//! NN Checkpoint Manager, Compute-Calibrated Multi-Engine Match Arena, and Embedded Browser Lab.
//!
//! Replaces external Python TensorBoard dependencies with a zero-dependency pure-Rust HTTP + JSON
//! server (`std::net::TcpListener`) that serves:
//! - `GET /`: The interactive **AlphaZero Browser Lab, Multi-Engine Arena & Play Inspector**.
//! - `GET /api/state`: Live JSON snapshot of the active game (`"ttt"` or `"uttt"`), training
//!   lifecycle status (`paused`, `phase`, `current_iteration`), per-game training history,
//!   tunable hyperparameter knobs, registered `.safetensors` models, hardware calibration report,
//!   and multi-match history.
//! - `POST /api/train_control`: Interactive training lifecycle control (`start`, `pause`, `step`,
//!   `reset`, `set_game`, `set_targets`).
//! - `POST /api/knobs`: Updates live training hyperparameter knobs for the active game.
//! - `POST /api/models`: Saves the live NN to `<game>_<name>.safetensors`, takes a 1-click
//!   iteration snapshot, loads a `.safetensors` checkpoint into the live model, or deletes a checkpoint.
//! - `POST /api/calibrate`: Benchmarks actual per-move thinking times (`µs/sim`, `µs/eval`, and
//!   per-depth Minimax `µs/move`) on the current hardware for fair equal-compute matches.
//! - `POST /api/match`: Runs a parallel role-swapped multi-match (`ttt` or `uttt`) between any two
//!   configurable engines (`alphazero`, `mcts`, `tree-minimax`, `minimax`, `random`) and reports
//!   both W/D/L and actual measured thinking time (`µs/move`) per engine.
//! - `GET /api/inspect_ttt`: Evaluates any Tic-Tac-Toe board position using any registered NN
//!   model, returning raw NN policy priors `P(s, a)`, value `V(s)`, MCTS root visits `N(s, a)`,
//!   action values `Q(s, a)`, and intra-batch deduplication statistics.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use candle_core::Device;

use crate::alphazero::{AlphaZeroConfig, AlphaZeroGame, AlphaZeroPlayer};
use crate::game::{GameStatus, PlayerIndex};
use crate::games::tic_tac_toe::{self, EMPTY_STATE, Symbol, TicTacToe, TttGame, TttHeuristic};
use crate::games::ultimate_ttt::{
    self, EMPTY_STATE as UTTT_EMPTY_STATE, UltimateTicTacToe, UtttGame, UtttHeuristic,
};
use crate::inference::{BatchEvaluator, InferenceService};
use crate::matchup::{RandomPlayer, run_multi_match};
use crate::mcts::{MctsConfig, MctsPlayer, PlayoutEvaluator, Uct};
use crate::minimax::{MinimaxBudget, MinimaxPlayer, TreeMinimaxPlayer};
use crate::nn::{AlphaZeroNetConfig, CandleBatchEvaluator};
use crate::player::Player;
use crate::root_parallel::{RootParallelBudget, RootParallelConfig, RootParallelPlayer};

/// Selectable game environment in the Browser Lab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabGame {
    /// Standard 3x3 Tic-Tac-Toe (`G = 27, P = 9, N = 2`).
    Ttt,
    /// 9x9 Ultimate Tic-Tac-Toe (`G = 324, P = 81, N = 2`).
    Uttt,
}

impl LabGame {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ttt => "ttt",
            Self::Uttt => "uttt",
        }
    }

    #[inline]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Ttt => "Tic-Tac-Toe (3x3)",
            Self::Uttt => "Ultimate Tic-Tac-Toe (9x9)",
        }
    }

    #[inline]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Ttt => "ttt_",
            Self::Uttt => "uttt_",
        }
    }

    #[inline]
    pub fn from_str_lossy(raw: &str) -> Self {
        if raw.trim().eq_ignore_ascii_case("uttt") {
            Self::Uttt
        } else {
            Self::Ttt
        }
    }
}

/// Action returned to the background training worker loop by [`TelemetryHub::poll_training_command`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingCommand {
    /// Training is currently paused; sleep briefly and poll again.
    Idle,
    /// Reset the trainer, live inference weights, replay buffer, and telemetry history for `LabGame`.
    Reset(LabGame),
    /// Execute one training iteration (`iteration` 1-based) for `LabGame`.
    RunIteration { game: LabGame, iteration: usize },
}

/// Per-iteration training, search, arena, and system pipeline telemetry record.
#[derive(Clone, Debug)]
pub struct IterationTelemetry {
    pub iteration: usize,
    pub total_loss: f32,
    pub policy_loss: f32,
    pub policy_kl: f32,
    pub value_loss: f32,
    pub root_entropy_mean: f32,
    pub game_moves_mean: f32,
    pub self_play_wins: [usize; 2],
    pub self_play_draws: usize,
    /// `[wins, draws, losses]` against baseline Opponent 1 (`RandomPlayer`).
    pub vs_random: [usize; 3],
    /// `[wins, draws, losses]` against baseline Opponent 2 (`Minimax(d=2)` for TTT, `Minimax(d=1)` for UTTT).
    pub vs_mm2: [usize; 3],
    /// `[wins, draws, losses]` against baseline Opponent 3 (`Minimax(d=5)` for TTT, `Minimax(d=2)` for UTTT).
    pub vs_mm5: [usize; 3],
    pub buffer_len: usize,
    pub buffer_capacity: usize,
    pub batch_size_mean: f32,
    pub batch_size_std: f32,
    pub batch_size_max: f32,
    pub unique_batch_mean: f32,
    pub dedup_saved_pct: f32,
    pub max_batch_size: usize,
    pub inference_us_mean: f32,
    pub simulations_per_sec: f32,
    pub self_play_ms: f32,
    pub train_ms: f32,
    pub eval_ms: f32,
}

impl IterationTelemetry {
    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"iteration\":{},",
                "\"total_loss\":{:.6},",
                "\"policy_loss\":{:.6},",
                "\"policy_kl\":{:.6},",
                "\"value_loss\":{:.6},",
                "\"root_entropy_mean\":{:.6},",
                "\"game_moves_mean\":{:.4},",
                "\"self_play_wins\":[{},{}],",
                "\"self_play_draws\":{},",
                "\"vs_random\":[{},{},{}],",
                "\"vs_mm2\":[{},{},{}],",
                "\"vs_mm5\":[{},{},{}],",
                "\"buffer_len\":{},",
                "\"buffer_capacity\":{},",
                "\"batch_size_mean\":{:.3},",
                "\"batch_size_std\":{:.3},",
                "\"batch_size_max\":{:.1},",
                "\"unique_batch_mean\":{:.3},",
                "\"dedup_saved_pct\":{:.2},",
                "\"max_batch_size\":{},",
                "\"inference_us_mean\":{:.3},",
                "\"simulations_per_sec\":{:.1},",
                "\"self_play_ms\":{:.2},",
                "\"train_ms\":{:.2},",
                "\"eval_ms\":{:.2}",
                "}}"
            ),
            self.iteration,
            finite_or_zero(self.total_loss),
            finite_or_zero(self.policy_loss),
            finite_or_zero(self.policy_kl),
            finite_or_zero(self.value_loss),
            finite_or_zero(self.root_entropy_mean),
            finite_or_zero(self.game_moves_mean),
            self.self_play_wins[0],
            self.self_play_wins[1],
            self.self_play_draws,
            self.vs_random[0],
            self.vs_random[1],
            self.vs_random[2],
            self.vs_mm2[0],
            self.vs_mm2[1],
            self.vs_mm2[2],
            self.vs_mm5[0],
            self.vs_mm5[1],
            self.vs_mm5[2],
            self.buffer_len,
            self.buffer_capacity,
            finite_or_zero(self.batch_size_mean),
            finite_or_zero(self.batch_size_std),
            finite_or_zero(self.batch_size_max),
            finite_or_zero(self.unique_batch_mean),
            finite_or_zero(self.dedup_saved_pct),
            self.max_batch_size,
            finite_or_zero(self.inference_us_mean),
            finite_or_zero(self.simulations_per_sec),
            finite_or_zero(self.self_play_ms),
            finite_or_zero(self.train_ms),
            finite_or_zero(self.eval_ms),
        )
    }
}

/// Live-tunable hyperparameters controllable from the Browser Lab UI during training.
#[derive(Clone, Copy, Debug)]
pub struct LiveKnobs {
    pub paused: bool,
    pub games_per_iteration: usize,
    pub simulations_per_move: usize,
    pub train_steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub dirichlet_epsilon: f32,
    pub c_init: f32,
}

impl Default for LiveKnobs {
    fn default() -> Self {
        Self::default_ttt()
    }
}

impl LiveKnobs {
    pub const fn default_ttt() -> Self {
        Self {
            paused: true,
            games_per_iteration: 128,
            simulations_per_move: 80,
            train_steps: 28,
            batch_size: 64,
            learning_rate: 1.5e-3,
            dirichlet_epsilon: 0.25,
            c_init: 1.25,
        }
    }

    pub const fn default_uttt() -> Self {
        Self {
            paused: true,
            games_per_iteration: 16,
            simulations_per_move: 40,
            train_steps: 32,
            batch_size: 64,
            learning_rate: 2.0e-3,
            dirichlet_epsilon: 0.25,
            c_init: 1.25,
        }
    }

    fn to_json(self) -> String {
        format!(
            concat!(
                "{{",
                "\"paused\":{},",
                "\"games_per_iteration\":{},",
                "\"simulations_per_move\":{},",
                "\"train_steps\":{},",
                "\"batch_size\":{},",
                "\"learning_rate\":{:.6},",
                "\"dirichlet_epsilon\":{:.4},",
                "\"c_init\":{:.4}",
                "}}"
            ),
            self.paused,
            self.games_per_iteration,
            self.simulations_per_move,
            self.train_steps,
            self.batch_size,
            self.learning_rate,
            self.dirichlet_epsilon,
            self.c_init,
        )
    }
}

/// Metadata for a neural network model available in the Browser Lab.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub kind: &'static str,
    pub game: &'static str,
    pub size_bytes: u64,
    pub description: String,
}

impl ModelInfo {
    fn to_json(&self) -> String {
        format!(
            "{{\"id\":\"{}\",\"kind\":\"{}\",\"game\":\"{}\",\"size_bytes\":{},\"description\":\"{}\"}}",
            escape_json(&self.id),
            self.kind,
            self.game,
            self.size_bytes,
            escape_json(&self.description),
        )
    }
}

/// Measured per-move latency (`µs/move`) for a specific recursive Minimax depth.
#[derive(Clone, Copy, Debug)]
pub struct MinimaxDepthTiming {
    pub depth: u32,
    pub us_per_move: f32,
}

impl MinimaxDepthTiming {
    fn to_json(self) -> String {
        format!(
            "{{\"depth\":{},\"us_per_move\":{:.2}}}",
            self.depth,
            finite_or_zero(self.us_per_move)
        )
    }
}

/// Empirical hardware timing calibration report for a game (`ttt` or `uttt`), mapping search
/// parameters (`sims`, `evals`, `depth`) to estimated per-move thinking time (`µs/move`).
#[derive(Clone, Debug)]
pub struct CalibrationReport {
    pub game: &'static str,
    pub alphazero_us_per_sim_1tree: f32,
    pub alphazero_us_per_sim_4tree: f32,
    pub mcts_us_per_sim_1tree: f32,
    pub mcts_us_per_sim_4tree: f32,
    pub tree_minimax_us_per_eval: f32,
    pub minimax_depths: Vec<MinimaxDepthTiming>,
}

impl CalibrationReport {
    fn to_json(&self) -> String {
        let depths_json = self
            .minimax_depths
            .iter()
            .copied()
            .map(MinimaxDepthTiming::to_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{",
                "\"game\":\"{}\",",
                "\"alphazero_us_per_sim_1tree\":{:.3},",
                "\"alphazero_us_per_sim_4tree\":{:.3},",
                "\"mcts_us_per_sim_1tree\":{:.3},",
                "\"mcts_us_per_sim_4tree\":{:.3},",
                "\"tree_minimax_us_per_eval\":{:.3},",
                "\"minimax_depths\":[{}]",
                "}}"
            ),
            self.game,
            finite_or_zero(self.alphazero_us_per_sim_1tree),
            finite_or_zero(self.alphazero_us_per_sim_4tree),
            finite_or_zero(self.mcts_us_per_sim_1tree),
            finite_or_zero(self.mcts_us_per_sim_4tree),
            finite_or_zero(self.tree_minimax_us_per_eval),
            depths_json,
        )
    }
}

/// Configuration for one side of a Multi-Match in the Browser Lab (`ttt` or `uttt`).
#[derive(Clone, Debug)]
pub struct TttEngineConfig {
    /// `"alphazero"`, `"mcts"`, `"tree-minimax"`, `"minimax"`, or `"random"`.
    pub engine: String,
    /// Minimax search depth (`0..=9`).
    pub depth: u32,
    /// Whether Minimax / TreeMinimax uses the game's heuristic (`true`) or `ZeroHeuristic` (`false`).
    pub use_heuristic: bool,
    /// TreeMinimax budget mode: `"depth"`, `"evals"`, or `"ms"`.
    pub budget_kind: String,
    /// TreeMinimax budget value (depth plies, max node evaluations, or max milliseconds).
    pub budget_val: usize,
    /// TreeMinimax arena node capacity.
    pub arena_cap: usize,
    /// MCTS / AlphaZero simulations per tree per move.
    pub sims: usize,
    /// MCTS / AlphaZero root-parallel trees `K` per move (`1` = single tree, `> 1` = root parallel).
    pub num_trees: usize,
    /// Classical MCTS UCT exploration constant `c`.
    pub c_uct: f32,
    /// Proportional-sampling opening plies (`0` = greedy argmax).
    pub opening_moves: usize,
    /// AlphaZero neural network model identifier (`"live"`, `"untrained"`, or `"<file>.safetensors"`).
    pub model: String,
    /// AlphaZero PUCT `c_init`.
    pub c_init: f32,
    /// AlphaZero PUCT `c_base`.
    pub c_base: f32,
    /// AlphaZero root Dirichlet `alpha`.
    pub dir_alpha: f32,
    /// AlphaZero root Dirichlet `epsilon` (`0.0` disables root noise).
    pub dir_eps: f32,
}

impl Default for TttEngineConfig {
    fn default() -> Self {
        Self {
            engine: "alphazero".to_string(),
            depth: 2,
            use_heuristic: true,
            budget_kind: "evals".to_string(),
            budget_val: 200,
            arena_cap: 16_384,
            sims: 80,
            num_trees: 1,
            c_uct: 1.414,
            opening_moves: 0,
            model: "live".to_string(),
            c_init: 1.25,
            c_base: 19_652.0,
            dir_alpha: 0.3,
            dir_eps: 0.0,
        }
    }
}

impl TttEngineConfig {
    pub fn summary_label(&self) -> String {
        match self.engine.as_str() {
            "minimax" => format!(
                "Minimax(d={}, H={})",
                self.depth,
                if self.use_heuristic { "on" } else { "off" }
            ),
            "tree-minimax" => format!(
                "TreeMinimax({}={}, H={})",
                self.budget_kind,
                self.budget_val,
                if self.use_heuristic { "on" } else { "off" }
            ),
            "mcts" => format!(
                "MCTS({}x{} sims, c={:.2}, open={})",
                self.num_trees.max(1),
                self.sims.max(1),
                self.c_uct,
                self.opening_moves
            ),
            "alphazero" => format!(
                "AlphaZero({}, {}x{} sims, c={:.2}, eps={:.2})",
                self.model,
                self.num_trees.max(1),
                self.sims.max(1),
                self.c_init,
                self.dir_eps
            ),
            _ => "Random".to_string(),
        }
    }
}

/// Completed Multi-Match telemetry record stored in the Browser Lab's match history.
#[derive(Clone, Debug)]
pub struct TttMatchRecord {
    pub match_id: usize,
    pub game: &'static str,
    pub engine_a: String,
    pub engine_b: String,
    pub num_games: usize,
    pub num_workers: usize,
    pub wins_a: usize,
    pub draws: usize,
    pub wins_b: usize,
    pub score_a_pct: f32,
    pub avg_moves: f32,
    pub duration_ms: f32,
    pub games_per_sec: f32,
    /// Total compute time per move (`µs/move`, including MCTS tree walk + amortized NN forward pass) for Engine A.
    pub a_us_per_move: f32,
    /// Total compute time per move (`µs/move`, including MCTS tree walk + amortized NN forward pass) for Engine B.
    pub b_us_per_move: f32,
    pub nn_logical_batch: f32,
    pub nn_unique_batch: f32,
    pub nn_dedup_saved_pct: f32,
    pub nn_us_per_state: f32,
}

impl TttMatchRecord {
    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"match_id\":{},",
                "\"game\":\"{}\",",
                "\"engine_a\":\"{}\",",
                "\"engine_b\":\"{}\",",
                "\"num_games\":{},",
                "\"num_workers\":{},",
                "\"wins_a\":{},",
                "\"draws\":{},",
                "\"wins_b\":{},",
                "\"score_a_pct\":{:.2},",
                "\"avg_moves\":{:.2},",
                "\"duration_ms\":{:.2},",
                "\"games_per_sec\":{:.1},",
                "\"a_us_per_move\":{:.2},",
                "\"b_us_per_move\":{:.2},",
                "\"nn_logical_batch\":{:.2},",
                "\"nn_unique_batch\":{:.2},",
                "\"nn_dedup_saved_pct\":{:.2},",
                "\"nn_us_per_state\":{:.2}",
                "}}"
            ),
            self.match_id,
            self.game,
            escape_json(&self.engine_a),
            escape_json(&self.engine_b),
            self.num_games,
            self.num_workers,
            self.wins_a,
            self.draws,
            self.wins_b,
            finite_or_zero(self.score_a_pct),
            finite_or_zero(self.avg_moves),
            finite_or_zero(self.duration_ms),
            finite_or_zero(self.games_per_sec),
            finite_or_zero(self.a_us_per_move),
            finite_or_zero(self.b_us_per_move),
            finite_or_zero(self.nn_logical_batch),
            finite_or_zero(self.nn_unique_batch),
            finite_or_zero(self.nn_dedup_saved_pct),
            finite_or_zero(self.nn_us_per_state),
        )
    }
}

/// Detailed neural network + MCTS inspection of a single Tic-Tac-Toe board state.
#[derive(Clone, Debug)]
pub struct TttPositionInspection {
    pub status: &'static str,
    pub legal_mask: [bool; 9],
    pub raw_nn_priors: [f32; 9],
    pub raw_nn_value: f32,
    pub mcts_visits: [u64; 9],
    pub mcts_q_values: [f32; 9],
    pub chosen_move: Option<usize>,
    pub nn_requests: u64,
    pub nn_unique_evals: u64,
}

impl TttPositionInspection {
    fn to_json(&self) -> String {
        let mask_str = self
            .legal_mask
            .iter()
            .map(|b| if *b { "true" } else { "false" })
            .collect::<Vec<_>>()
            .join(",");
        let priors_str = self
            .raw_nn_priors
            .iter()
            .map(|v| format!("{:.5}", finite_or_zero(*v)))
            .collect::<Vec<_>>()
            .join(",");
        let visits_str = self
            .mcts_visits
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let q_str = self
            .mcts_q_values
            .iter()
            .map(|v| format!("{:.5}", finite_or_zero(*v)))
            .collect::<Vec<_>>()
            .join(",");
        let chosen_str = match self.chosen_move {
            Some(mv) => mv.to_string(),
            None => "null".to_string(),
        };
        format!(
            concat!(
                "{{",
                "\"status\":\"{}\",",
                "\"legal_mask\":[{}],",
                "\"raw_nn_priors\":[{}],",
                "\"raw_nn_value\":{:.5},",
                "\"mcts_visits\":[{}],",
                "\"mcts_q_values\":[{}],",
                "\"chosen_move\":{},",
                "\"nn_requests\":{},",
                "\"nn_unique_evals\":{}",
                "}}"
            ),
            self.status,
            mask_str,
            priors_str,
            finite_or_zero(self.raw_nn_value),
            visits_str,
            q_str,
            chosen_str,
            self.nn_requests,
            self.nn_unique_evals,
        )
    }
}

/// Evaluates `game` using `service`, returning both the raw neural network policy/value output
/// and the Root-Parallel MCTS visit counts and Q-values after `total_simulations`.
pub fn inspect_ttt_position<E: BatchEvaluator<27, 9, 2> + 'static>(
    game: &TttGame,
    service: &InferenceService<E, 27, 9, 2>,
    total_simulations: usize,
) -> TttPositionInspection {
    inspect_ttt_position_configured(game, service, 8, total_simulations, 1.25, 0.0)
}

/// Evaluates `game` using `service` with explicit MCTS parameters `(num_trees, total_simulations, c_init, dirichlet_epsilon)`.
pub fn inspect_ttt_position_configured<E: BatchEvaluator<27, 9, 2> + 'static>(
    game: &TttGame,
    service: &InferenceService<E, 27, 9, 2>,
    num_trees: usize,
    total_simulations: usize,
    c_init: f32,
    dirichlet_epsilon: f32,
) -> TttPositionInspection {
    let mut legal_mask = [false; 9];
    for mv in game.valid_moves() {
        legal_mask[mv as usize] = true;
    }

    let status_label = match game.status() {
        GameStatus::Ongoing => "ongoing",
        GameStatus::Terminal(payoffs) => {
            if payoffs[0] > payoffs[1] {
                "p1_win"
            } else if payoffs[1] > payoffs[0] {
                "p2_win"
            } else {
                "draw"
            }
        }
    };

    let (reqs_before, uniq_before) = service.dedup_counts();

    let mut encoded = [0.0f32; 27];
    TicTacToe::encode_state(game, &mut encoded);
    let client = service.client(1);
    client.submit(0, encoded);
    client.notify_service();
    client.wait_for_response();
    let resp = client.try_pop().expect("inference response must arrive");

    let mut raw_nn_priors = [0.0f32; 9];
    let mut max_logit = f32::NEG_INFINITY;
    for (idx, &is_legal) in legal_mask.iter().enumerate() {
        if is_legal && resp.policy_logits[idx] > max_logit {
            max_logit = resp.policy_logits[idx];
        }
    }
    if max_logit.is_finite() {
        let mut sum = 0.0f32;
        for (idx, &is_legal) in legal_mask.iter().enumerate() {
            if is_legal {
                let p = (resp.policy_logits[idx] - max_logit).exp();
                raw_nn_priors[idx] = p;
                sum += p;
            }
        }
        if sum > 0.0 {
            for p in &mut raw_nn_priors {
                *p /= sum;
            }
        }
    }
    let raw_nn_value = resp.value[0];

    if status_label != "ongoing" {
        let (reqs_after, uniq_after) = service.dedup_counts();
        return TttPositionInspection {
            status: status_label,
            legal_mask,
            raw_nn_priors,
            raw_nn_value,
            mcts_visits: [0; 9],
            mcts_q_values: [0.0; 9],
            chosen_move: None,
            nn_requests: reqs_after.saturating_sub(reqs_before),
            nn_unique_evals: uniq_after.saturating_sub(uniq_before),
        };
    }

    let trees = num_trees.clamp(1, 64);
    let sims_per_tree = total_simulations.div_ceil(trees).max(2);
    let mut player = AlphaZeroPlayer::new(
        *game,
        service,
        AlphaZeroConfig::new(trees, sims_per_tree, 2026)
            .with_puct(19_652.0, c_init)
            .with_dirichlet(0.3, dirichlet_epsilon),
    );
    let agg = player.evaluate_root();
    let (reqs_after, uniq_after) = service.dedup_counts();

    let mut mcts_visits = [0u64; 9];
    let mut mcts_q_values = [0.0f32; 9];
    let mut best_mv = None;
    let mut best_visits = 0u64;
    for child in &agg {
        let idx = child.mv as usize;
        mcts_visits[idx] = child.visits;
        mcts_q_values[idx] = child.q_value() as f32;
        if child.visits > best_visits || best_mv.is_none() {
            best_visits = child.visits;
            best_mv = Some(idx);
        }
    }

    TttPositionInspection {
        status: status_label,
        legal_mask,
        raw_nn_priors,
        raw_nn_value,
        mcts_visits,
        mcts_q_values,
        chosen_move: best_mv,
        nn_requests: reqs_after.saturating_sub(reqs_before),
        nn_unique_evals: uniq_after.saturating_sub(uniq_before),
    }
}

type TttInspectorFn = Box<dyn Fn(&TttGame, usize) -> TttPositionInspection + Send + Sync>;

#[derive(Clone)]
struct ModelContext<const G: usize, const P: usize, const N: usize> {
    live_service: Arc<InferenceService<CandleBatchEvaluator<G, P, N>, G, P, N>>,
    net_config: AlphaZeroNetConfig,
    devices: Vec<Device>,
}

impl<const G: usize, const P: usize, const N: usize> ModelContext<G, P, N> {
    fn resolve_evaluator(&self, model_id: &str) -> Result<CandleBatchEvaluator<G, P, N>, String> {
        let id = model_id.trim();
        if id.is_empty() || id == "live" {
            return Ok(self.live_service.evaluator().clone());
        }
        if id == "untrained" {
            return CandleBatchEvaluator::standalone(&self.net_config, &self.devices, None::<&str>)
                .map_err(|e| e.to_string());
        }
        let safe_name = sanitize_safetensors_filename(id)
            .ok_or_else(|| format!("Invalid checkpoint filename: {id}"))?;
        if !Path::new(&safe_name).exists() {
            return Err(format!("Checkpoint file '{safe_name}' does not exist"));
        }
        CandleBatchEvaluator::standalone(&self.net_config, &self.devices, Some(&safe_name))
            .map_err(|e| format!("Failed to load '{safe_name}': {e}"))
    }
}

struct GameLabState {
    phase: String,
    current_iteration: usize,
    total_iterations: usize,
    step_requests: usize,
    pause_after_iteration: bool,
    reset_requested: bool,
    auto_save_every: usize,
    knobs: LiveKnobs,
    history: Vec<IterationTelemetry>,
    matches: Vec<TttMatchRecord>,
    calibration: Option<CalibrationReport>,
}

impl GameLabState {
    fn new(total_iterations: usize, knobs: LiveKnobs, auto_save_every: usize) -> Self {
        let initial_phase = if knobs.paused {
            "Idle — Ready (Click ▶ Start Training or ⏭ Step +1)".to_string()
        } else {
            "Initializing".to_string()
        };
        Self {
            phase: initial_phase,
            current_iteration: 0,
            total_iterations,
            step_requests: 0,
            pause_after_iteration: false,
            reset_requested: false,
            auto_save_every,
            knobs,
            history: Vec::with_capacity(total_iterations.max(64)),
            matches: Vec::new(),
            calibration: None,
        }
    }
}

struct HubInner {
    device_name: String,
    active_game: LabGame,
    ttt_state: GameLabState,
    uttt_state: GameLabState,
    inspector: Option<Arc<TttInspectorFn>>,
    ttt_context: Option<ModelContext<27, 9, 2>>,
    uttt_context: Option<ModelContext<324, 81, 2>>,
}

impl HubInner {
    fn state_for(&self, game: LabGame) -> &GameLabState {
        match game {
            LabGame::Ttt => &self.ttt_state,
            LabGame::Uttt => &self.uttt_state,
        }
    }

    fn state_for_mut(&mut self, game: LabGame) -> &mut GameLabState {
        match game {
            LabGame::Ttt => &mut self.ttt_state,
            LabGame::Uttt => &mut self.uttt_state,
        }
    }

    fn active_state(&self) -> &GameLabState {
        self.state_for(self.active_game)
    }
}

/// Thread-safe telemetry, model registry, multi-match arena, compute calibrator, and live control
/// hub shared between the training loop and the embedded HTTP Browser Lab server.
#[derive(Clone)]
pub struct TelemetryHub {
    inner: Arc<Mutex<HubInner>>,
}

impl TelemetryHub {
    pub fn new(device_name: impl Into<String>, total_iterations: usize, knobs: LiveKnobs) -> Self {
        let mut uttt_knobs = LiveKnobs::default_uttt();
        uttt_knobs.paused = knobs.paused;
        Self {
            inner: Arc::new(Mutex::new(HubInner {
                device_name: device_name.into(),
                active_game: LabGame::Ttt,
                ttt_state: GameLabState::new(total_iterations, knobs, 5),
                uttt_state: GameLabState::new(30, uttt_knobs, 5),
                inspector: None,
                ttt_context: None,
                uttt_context: None,
            })),
        }
    }

    #[inline]
    pub fn active_game(&self) -> LabGame {
        self.inner
            .lock()
            .expect("telemetry mutex poisoned")
            .active_game
    }

    pub fn set_active_game(&self, game: LabGame) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        if guard.active_game != game {
            let prev = guard.active_game;
            guard.state_for_mut(prev).knobs.paused = true;
            guard.state_for_mut(prev).step_requests = 0;
            guard.state_for_mut(prev).pause_after_iteration = false;
            guard.active_game = game;
        }
    }

    pub fn set_inspector<F>(&self, f: F)
    where
        F: Fn(&TttGame, usize) -> TttPositionInspection + Send + Sync + 'static,
    {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.inspector = Some(Arc::new(Box::new(f)));
    }

    /// Attaches the live Tic-Tac-Toe (`G=27, P=9, N=2`) [`InferenceService`] and [`AlphaZeroNetConfig`].
    pub fn attach_ttt_service(
        &self,
        live_service: Arc<InferenceService<CandleBatchEvaluator<27, 9, 2>, 27, 9, 2>>,
        net_config: AlphaZeroNetConfig,
        devices: Vec<Device>,
    ) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        let svc_for_inspector = Arc::clone(&live_service);
        guard.inspector = Some(Arc::new(Box::new(move |game, sims| {
            inspect_ttt_position(game, &svc_for_inspector, sims)
        })));
        guard.ttt_context = Some(ModelContext {
            live_service,
            net_config,
            devices,
        });
    }

    /// Attaches the live Ultimate Tic-Tac-Toe (`G=324, P=81, N=2`) [`InferenceService`] and [`AlphaZeroNetConfig`].
    pub fn attach_uttt_service(
        &self,
        live_service: Arc<InferenceService<CandleBatchEvaluator<324, 81, 2>, 324, 81, 2>>,
        net_config: AlphaZeroNetConfig,
        devices: Vec<Device>,
    ) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.uttt_context = Some(ModelContext {
            live_service,
            net_config,
            devices,
        });
    }

    pub fn set_phase(&self, iteration: usize, phase: impl Into<String>) {
        let game = self.active_game();
        self.set_phase_for(game, iteration, phase);
    }

    pub fn set_phase_for(&self, game: LabGame, iteration: usize, phase: impl Into<String>) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        let st = guard.state_for_mut(game);
        st.current_iteration = iteration;
        st.phase = phase.into();
    }

    pub fn record_iteration(&self, record: IterationTelemetry) {
        let game = self.active_game();
        self.record_iteration_for(game, record);
    }

    pub fn record_iteration_for(&self, game: LabGame, record: IterationTelemetry) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        let st = guard.state_for_mut(game);
        st.current_iteration = record.iteration;
        if st.pause_after_iteration {
            st.pause_after_iteration = false;
            if st.step_requests == 0 {
                st.knobs.paused = true;
            }
        }
        st.history.push(record);
    }

    pub fn knobs(&self) -> LiveKnobs {
        self.inner
            .lock()
            .expect("telemetry mutex poisoned")
            .active_state()
            .knobs
    }

    pub fn knobs_for(&self, game: LabGame) -> LiveKnobs {
        self.inner
            .lock()
            .expect("telemetry mutex poisoned")
            .state_for(game)
            .knobs
    }

    pub fn auto_save_every_for(&self, game: LabGame) -> usize {
        self.inner
            .lock()
            .expect("telemetry mutex poisoned")
            .state_for(game)
            .auto_save_every
    }

    /// Blocks while `knobs.paused` is `true`, polling every `100ms`.
    pub fn wait_if_paused(&self) {
        while self.knobs().paused {
            thread::sleep(Duration::from_millis(100));
        }
    }

    /// Polls the next training lifecycle command for the background training worker loop.
    pub fn poll_training_command(&self) -> TrainingCommand {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        for game in [LabGame::Ttt, LabGame::Uttt] {
            let st = guard.state_for_mut(game);
            if st.reset_requested {
                st.reset_requested = false;
                st.current_iteration = 0;
                st.step_requests = 0;
                st.pause_after_iteration = false;
                st.knobs.paused = true;
                st.history.clear();
                st.phase = "Reset Complete — Fresh Untrained Weights (Iter 0)".to_string();
                return TrainingCommand::Reset(game);
            }
        }

        let game = guard.active_game;
        let st = guard.state_for_mut(game);
        if st.step_requests > 0 {
            st.step_requests -= 1;
            st.knobs.paused = false;
            if st.step_requests == 0 {
                st.pause_after_iteration = true;
            }
            let next_iter = st.current_iteration + 1;
            if next_iter > st.total_iterations {
                st.total_iterations = next_iter;
            }
            return TrainingCommand::RunIteration {
                game,
                iteration: next_iter,
            };
        }

        if !st.knobs.paused {
            let next_iter = st.current_iteration + 1;
            if next_iter <= st.total_iterations {
                return TrainingCommand::RunIteration {
                    game,
                    iteration: next_iter,
                };
            } else {
                st.knobs.paused = true;
                st.phase = format!(
                    "Completed {} Iterations — Click ▶ Start/Resume or ⏭ Step (+1) for more",
                    st.current_iteration
                );
            }
        }

        TrainingCommand::Idle
    }

    pub fn discover_models(&self) -> Vec<ModelInfo> {
        let guard = self.inner.lock().expect("telemetry mutex poisoned");
        let game = guard.active_game;
        let has_live = match game {
            LabGame::Ttt => guard.ttt_context.is_some(),
            LabGame::Uttt => guard.uttt_context.is_some(),
        };
        let iter = guard.state_for(game).current_iteration;
        discover_game_models(game, has_live, iter)
    }

    /// Runs a fast empirical timing calibration for `game` (`Tic-Tac-Toe` or `Ultimate Tic-Tac-Toe`),
    /// caching and returning the resulting [`CalibrationReport`].
    pub fn calibrate_game(&self, game: LabGame) -> Result<CalibrationReport, String> {
        let report = match game {
            LabGame::Ttt => {
                let ttt_ctx = self
                    .inner
                    .lock()
                    .expect("telemetry mutex poisoned")
                    .ttt_context
                    .clone();
                calibrate_ttt(ttt_ctx.as_ref())?
            }
            LabGame::Uttt => {
                let uttt_ctx = self
                    .inner
                    .lock()
                    .expect("telemetry mutex poisoned")
                    .uttt_context
                    .clone();
                calibrate_uttt(uttt_ctx.as_ref())?
            }
        };

        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.state_for_mut(game).calibration = Some(report.clone());
        Ok(report)
    }

    /// Executes a configured Tic-Tac-Toe Multi-Match between `cfg_a` and `cfg_b`.
    pub fn run_ttt_match(
        &self,
        cfg_a: &TttEngineConfig,
        cfg_b: &TttEngineConfig,
        num_games: usize,
        num_workers: usize,
        max_batch_size: usize,
        seed: u64,
    ) -> Result<TttMatchRecord, String> {
        self.run_match_for_game(
            LabGame::Ttt,
            cfg_a,
            cfg_b,
            num_games,
            num_workers,
            max_batch_size,
            seed,
        )
    }

    /// Executes a configured Multi-Match for `game` (`LabGame::Ttt` or `LabGame::Uttt`) between
    /// `cfg_a` and `cfg_b` across `num_games` role-swapped games on `num_workers` threads.
    #[allow(clippy::too_many_arguments)]
    pub fn run_match_for_game(
        &self,
        game: LabGame,
        cfg_a: &TttEngineConfig,
        cfg_b: &TttEngineConfig,
        num_games: usize,
        num_workers: usize,
        max_batch_size: usize,
        seed: u64,
    ) -> Result<TttMatchRecord, String> {
        let games = num_games.clamp(1, 5000);
        let workers = num_workers.clamp(1, 32);
        let batch_cap = max_batch_size.clamp(1, 512);

        let a_is_az = cfg_a.engine == "alphazero";
        let b_is_az = cfg_b.engine == "alphazero";

        let record = match game {
            LabGame::Ttt => {
                let ctx_opt = self
                    .inner
                    .lock()
                    .expect("telemetry mutex poisoned")
                    .ttt_context
                    .clone();

                let mut svc_a = None;
                let mut svc_b = None;
                if a_is_az || b_is_az {
                    let ctx = ctx_opt.ok_or_else(|| {
                        "Tic-Tac-Toe AlphaZero model context is not attached".to_string()
                    })?;
                    if a_is_az {
                        let eval = ctx.resolve_evaluator(&cfg_a.model)?;
                        let n_rep = eval.num_replicas();
                        svc_a = Some(Arc::new(InferenceService::new(eval, batch_cap, n_rep)));
                    }
                    if b_is_az {
                        let eval = ctx.resolve_evaluator(&cfg_b.model)?;
                        let n_rep = eval.num_replicas();
                        svc_b = Some(Arc::new(InferenceService::new(eval, batch_cap, n_rep)));
                    }
                }

                let svc_a_ref = svc_a.as_deref();
                let svc_b_ref = svc_b.as_deref();
                let factory_a = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
                    build_ttt_player(cfg_a, svc_a_ref, g, s)
                };
                let factory_b = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
                    build_ttt_player(cfg_b, svc_b_ref, g, s)
                };

                let root_game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
                let start = Instant::now();
                let outcome =
                    run_multi_match(&root_game, [&factory_a, &factory_b], games, workers, seed);
                let duration_ms = (start.elapsed().as_secs_f64() * 1000.0) as f32;
                summarize_match_outcome(
                    LabGame::Ttt,
                    cfg_a,
                    cfg_b,
                    games,
                    workers,
                    duration_ms,
                    &outcome,
                    svc_a.as_deref(),
                    svc_b.as_deref(),
                )
            }
            LabGame::Uttt => {
                let ctx_opt = self
                    .inner
                    .lock()
                    .expect("telemetry mutex poisoned")
                    .uttt_context
                    .clone();

                let mut svc_a = None;
                let mut svc_b = None;
                if a_is_az || b_is_az {
                    let ctx = ctx_opt.ok_or_else(|| {
                        "Ultimate Tic-Tac-Toe AlphaZero model context is not attached".to_string()
                    })?;
                    if a_is_az {
                        let eval = ctx.resolve_evaluator(&cfg_a.model)?;
                        let n_rep = eval.num_replicas();
                        svc_a = Some(Arc::new(InferenceService::new(eval, batch_cap, n_rep)));
                    }
                    if b_is_az {
                        let eval = ctx.resolve_evaluator(&cfg_b.model)?;
                        let n_rep = eval.num_replicas();
                        svc_b = Some(Arc::new(InferenceService::new(eval, batch_cap, n_rep)));
                    }
                }

                let svc_a_ref = svc_a.as_deref();
                let svc_b_ref = svc_b.as_deref();
                let factory_a =
                    |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
                        build_uttt_player(cfg_a, svc_a_ref, g, s)
                    };
                let factory_b =
                    |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
                        build_uttt_player(cfg_b, svc_b_ref, g, s)
                    };

                let root_game = UtttGame::new(PlayerIndex::PLAYER1, UTTT_EMPTY_STATE);
                let start = Instant::now();
                let outcome =
                    run_multi_match(&root_game, [&factory_a, &factory_b], games, workers, seed);
                let duration_ms = (start.elapsed().as_secs_f64() * 1000.0) as f32;
                summarize_match_outcome(
                    LabGame::Uttt,
                    cfg_a,
                    cfg_b,
                    games,
                    workers,
                    duration_ms,
                    &outcome,
                    svc_a.as_deref(),
                    svc_b.as_deref(),
                )
            }
        };

        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        let st = guard.state_for_mut(game);
        let match_id = st.matches.len() + 1;
        let mut final_rec = record;
        final_rec.match_id = match_id;
        st.matches.push(final_rec.clone());
        Ok(final_rec)
    }

    pub fn to_json(&self) -> String {
        let guard = self.inner.lock().expect("telemetry mutex poisoned");
        let game = guard.active_game;
        let st = guard.state_for(game);
        let has_live = match game {
            LabGame::Ttt => guard.ttt_context.is_some(),
            LabGame::Uttt => guard.uttt_context.is_some(),
        };
        let models = discover_game_models(game, has_live, st.current_iteration);
        let models_json = models
            .iter()
            .map(ModelInfo::to_json)
            .collect::<Vec<_>>()
            .join(",");
        let ttt_models = discover_game_models(
            LabGame::Ttt,
            guard.ttt_context.is_some(),
            guard.ttt_state.current_iteration,
        );
        let ttt_models_json = ttt_models
            .iter()
            .map(ModelInfo::to_json)
            .collect::<Vec<_>>()
            .join(",");
        let history_json = st
            .history
            .iter()
            .map(IterationTelemetry::to_json)
            .collect::<Vec<_>>()
            .join(",");
        let matches_json = st
            .matches
            .iter()
            .map(TttMatchRecord::to_json)
            .collect::<Vec<_>>()
            .join(",");
        let cal_json = match &st.calibration {
            Some(c) => c.to_json(),
            None => "null".to_string(),
        };
        format!(
            concat!(
                "{{",
                "\"device_name\":\"{}\",",
                "\"active_game\":\"{}\",",
                "\"active_game_title\":\"{}\",",
                "\"phase\":\"{}\",",
                "\"current_iteration\":{},",
                "\"total_iterations\":{},",
                "\"auto_save_every\":{},",
                "\"knobs\":{},",
                "\"models\":[{}],",
                "\"ttt_models\":[{}],",
                "\"calibration\":{},",
                "\"history\":[{}],",
                "\"matches\":[{}]",
                "}}"
            ),
            escape_json(&guard.device_name),
            game.as_str(),
            escape_json(game.title()),
            escape_json(&st.phase),
            st.current_iteration,
            st.total_iterations,
            st.auto_save_every,
            st.knobs.to_json(),
            models_json,
            ttt_models_json,
            cal_json,
            history_json,
            matches_json,
        )
    }

    /// Spawns a background HTTP server thread listening on `addr` (e.g. `"127.0.0.1:8080"` or
    /// `"0.0.0.0:8080"` for RunPod).
    pub fn spawn_http_server(&self, addr: &str) -> std::io::Result<String> {
        let listener = TcpListener::bind(addr)?;
        let bound_addr = listener.local_addr()?.to_string();
        let hub = self.clone();

        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let hub_ref = hub.clone();
                thread::spawn(move || {
                    let _ = handle_http_connection(stream, &hub_ref);
                });
            }
        });

        Ok(bound_addr)
    }
}

#[allow(clippy::too_many_arguments)]
fn summarize_match_outcome<const G: usize, const P: usize>(
    game: LabGame,
    cfg_a: &TttEngineConfig,
    cfg_b: &TttEngineConfig,
    games: usize,
    workers: usize,
    duration_ms: f32,
    outcome: &crate::matchup::MultiMatchSummary<2>,
    svc_a: Option<&InferenceService<CandleBatchEvaluator<G, P, 2>, G, P, 2>>,
    svc_b: Option<&InferenceService<CandleBatchEvaluator<G, P, 2>, G, P, 2>>,
) -> TttMatchRecord {
    let games_per_sec = (games as f32) / (duration_ms / 1000.0).max(1e-6);

    let moves_a = outcome.choose_time_us[0].count().max(1) as f32;
    let moves_b = outcome.choose_time_us[1].count().max(1) as f32;
    let mut a_us_per_move = outcome.choose_time_us[0].mean();
    let mut b_us_per_move = outcome.choose_time_us[1].mean();

    let mut logical_batch = 0.0f32;
    let mut unique_batch = 0.0f32;
    let mut dedup_saved_pct = 0.0f32;
    let mut us_per_state = 0.0f32;

    if let Some(sa) = svc_a {
        let t_stat = sa.inference_time_stats();
        let (_, uniq_a) = sa.dedup_counts();
        a_us_per_move += ((uniq_a as f32) * t_stat.mean()) / moves_a;
    }
    if let Some(sb) = svc_b {
        let t_stat = sb.inference_time_stats();
        let (_, uniq_b) = sb.dedup_counts();
        b_us_per_move += ((uniq_b as f32) * t_stat.mean()) / moves_b;
    }

    if let Some(sa) = svc_a {
        let mut b_stat = sa.batch_size_stats();
        let mut u_stat = sa.unique_batch_size_stats();
        let mut t_stat = sa.inference_time_stats();
        let (mut tot_req, mut uniq_ev) = sa.dedup_counts();

        if let Some(sb) = svc_b {
            b_stat.join(&sb.batch_size_stats());
            u_stat.join(&sb.unique_batch_size_stats());
            t_stat.join(&sb.inference_time_stats());
            let (r2, u2) = sb.dedup_counts();
            tot_req += r2;
            uniq_ev += u2;
        }
        logical_batch = b_stat.mean();
        unique_batch = u_stat.mean();
        us_per_state = t_stat.mean();
        if tot_req > 0 && uniq_ev < tot_req {
            dedup_saved_pct = (1.0 - (uniq_ev as f32) / (tot_req as f32)) * 100.0;
        }
    } else if let Some(sb) = svc_b {
        logical_batch = sb.batch_size_stats().mean();
        unique_batch = sb.unique_batch_size_stats().mean();
        us_per_state = sb.inference_time_stats().mean();
        dedup_saved_pct = sb.dedup_savings_ratio() * 100.0;
    }

    let wins_a = outcome.wins[0];
    let wins_b = outcome.wins[1];
    let draws = outcome.draws;
    let score_a_pct = (((wins_a as f32) + 0.5 * (draws as f32)) / (games as f32).max(1.0)) * 100.0;
    let total_moves = outcome.choose_time_us[0].count() + outcome.choose_time_us[1].count();
    let avg_moves = (total_moves as f32) / (games as f32).max(1.0);

    TttMatchRecord {
        match_id: 0,
        game: game.as_str(),
        engine_a: cfg_a.summary_label(),
        engine_b: cfg_b.summary_label(),
        num_games: games,
        num_workers: workers,
        wins_a,
        draws,
        wins_b,
        score_a_pct,
        avg_moves,
        duration_ms,
        games_per_sec,
        a_us_per_move,
        b_us_per_move,
        nn_logical_batch: logical_batch,
        nn_unique_batch: unique_batch,
        nn_dedup_saved_pct: dedup_saved_pct,
        nn_us_per_state: us_per_state,
    }
}

fn rand_ttt<'a>(g: &TttGame, s: u64) -> Box<dyn Player<TicTacToe, 2> + 'a> {
    Box::new(RandomPlayer::new(*g, s))
}

fn rand_uttt<'a>(g: &UtttGame, s: u64) -> Box<dyn Player<UltimateTicTacToe, 2> + 'a> {
    Box::new(RandomPlayer::new(*g, s))
}

fn calibrate_ttt(ctx: Option<&ModelContext<27, 9, 2>>) -> Result<CalibrationReport, String> {
    let root = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

    let mut minimax_depths = Vec::with_capacity(6);
    for depth in 1..=6u32 {
        let f_mm = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, depth, s))
        };
        let f_r = |g: &TttGame, s: u64| rand_ttt(g, s);
        let res = run_multi_match(&root, [&f_mm, &f_r], 6, 1, 900 + depth as u64);
        minimax_depths.push(MinimaxDepthTiming {
            depth,
            us_per_move: res.choose_time_us[0].mean().max(0.5),
        });
    }

    let eval_budget = 200usize;
    let f_tm = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
        Box::new(TreeMinimaxPlayer::new(
            *g,
            tic_tac_toe::NO_MOVE,
            TttHeuristic,
            MinimaxBudget::MaxEvaluations(eval_budget),
            s,
            8192,
        ))
    };
    let f_r1 = |g: &TttGame, s: u64| rand_ttt(g, s);
    let tm_res = run_multi_match(&root, [&f_tm, &f_r1], 6, 1, 950);
    let tree_minimax_us_per_eval = (tm_res.choose_time_us[0].mean() / (eval_budget as f32)).max(0.05);

    let mcts_sims = 200usize;
    let f_mcts_1 = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
        Box::new(MctsPlayer::new(
            *g,
            tic_tac_toe::NO_MOVE,
            Uct::new(1.414),
            PlayoutEvaluator::new(s ^ 0x55),
            MctsConfig::new(mcts_sims, s).with_arena_capacity(4096),
        ))
    };
    let f_r2 = |g: &TttGame, s: u64| rand_ttt(g, s);
    let m1_res = run_multi_match(&root, [&f_mcts_1, &f_r2], 6, 1, 960);
    let mcts_us_per_sim_1tree = (m1_res.choose_time_us[0].mean() / (mcts_sims as f32)).max(0.05);

    let f_mcts_4 = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
        Box::new(RootParallelPlayer::new(
            *g,
            tic_tac_toe::NO_MOVE,
            Uct::new(1.414),
            4,
            RootParallelConfig::new(RootParallelBudget::SimulationsPerThread(50), s)
                .with_arena_capacity_per_thread(2048),
            |_t, es| PlayoutEvaluator::new(es),
        ))
    };
    let f_r3 = |g: &TttGame, s: u64| rand_ttt(g, s);
    let m4_res = run_multi_match(&root, [&f_mcts_4, &f_r3], 6, 1, 970);
    let mcts_us_per_sim_4tree = (m4_res.choose_time_us[0].mean() / 200.0).max(0.05);

    let (az_1, az_4) = if let Some(c) = ctx {
        let eval = c.resolve_evaluator("live")?;
        let svc = InferenceService::new(eval, 32, 1);
        let svc_ref = &svc;
        let az_sims = 64usize;
        let f_az1 = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(AlphaZeroPlayer::new(
                *g,
                svc_ref,
                AlphaZeroConfig::new(1, az_sims, s),
            ))
        };
        let f_r4 = |g: &TttGame, s: u64| rand_ttt(g, s);
        let r1 = run_multi_match(&root, [&f_az1, &f_r4], 8, 2, 980);
        let (_, uniq1) = svc.dedup_counts();
        let moves1 = r1.choose_time_us[0].count().max(1) as f32;
        let total_us_1 = r1.choose_time_us[0].mean()
            + ((uniq1 as f32) * svc.inference_time_stats().mean()) / moves1;

        svc.reset_stats();
        let f_az4 = |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(AlphaZeroPlayer::new(
                *g,
                svc_ref,
                AlphaZeroConfig::new(4, 24, s),
            ))
        };
        let f_r5 = |g: &TttGame, s: u64| rand_ttt(g, s);
        let r4 = run_multi_match(&root, [&f_az4, &f_r5], 8, 2, 990);
        let (_, uniq4) = svc.dedup_counts();
        let moves4 = r4.choose_time_us[0].count().max(1) as f32;
        let total_us_4 = r4.choose_time_us[0].mean()
            + ((uniq4 as f32) * svc.inference_time_stats().mean()) / moves4;

        (
            (total_us_1 / (az_sims as f32)).max(0.5),
            (total_us_4 / 96.0).max(0.5),
        )
    } else {
        (5.0, 3.5)
    };

    Ok(CalibrationReport {
        game: "ttt",
        alphazero_us_per_sim_1tree: az_1,
        alphazero_us_per_sim_4tree: az_4,
        mcts_us_per_sim_1tree,
        mcts_us_per_sim_4tree,
        tree_minimax_us_per_eval,
        minimax_depths,
    })
}

fn calibrate_uttt(ctx: Option<&ModelContext<324, 81, 2>>) -> Result<CalibrationReport, String> {
    let root = UtttGame::new(PlayerIndex::PLAYER1, UTTT_EMPTY_STATE);
    let h = UtttHeuristic::default();

    let mut minimax_depths = Vec::with_capacity(4);
    for depth in 1..=4u32 {
        let f_mm = |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
            Box::new(MinimaxPlayer::new(*g, h, depth, s))
        };
        let f_r = |g: &UtttGame, s: u64| rand_uttt(g, s);
        let games = if depth >= 4 { 2 } else { 4 };
        let res = run_multi_match(&root, [&f_mm, &f_r], games, 2, 1900 + depth as u64);
        minimax_depths.push(MinimaxDepthTiming {
            depth,
            us_per_move: res.choose_time_us[0].mean().max(1.0),
        });
    }

    let eval_budget = 250usize;
    let f_tm = |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
        Box::new(TreeMinimaxPlayer::new(
            *g,
            ultimate_ttt::NO_MOVE,
            h,
            MinimaxBudget::MaxEvaluations(eval_budget),
            s,
            16_384,
        ))
    };
    let f_r1 = |g: &UtttGame, s: u64| rand_uttt(g, s);
    let tm_res = run_multi_match(&root, [&f_tm, &f_r1], 4, 2, 1950);
    let tree_minimax_us_per_eval =
        (tm_res.choose_time_us[0].mean() / (eval_budget as f32)).max(0.1);

    let mcts_sims = 120usize;
    let f_mcts_1 = |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
        Box::new(MctsPlayer::new(
            *g,
            ultimate_ttt::NO_MOVE,
            Uct::new(1.414),
            PlayoutEvaluator::new(s ^ 0x55),
            MctsConfig::new(mcts_sims, s).with_arena_capacity(8192),
        ))
    };
    let f_r2 = |g: &UtttGame, s: u64| rand_uttt(g, s);
    let m1_res = run_multi_match(&root, [&f_mcts_1, &f_r2], 4, 2, 1960);
    let mcts_us_per_sim_1tree = (m1_res.choose_time_us[0].mean() / (mcts_sims as f32)).max(0.1);

    let f_mcts_4 = |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
        Box::new(RootParallelPlayer::new(
            *g,
            ultimate_ttt::NO_MOVE,
            Uct::new(1.414),
            4,
            RootParallelConfig::new(RootParallelBudget::SimulationsPerThread(30), s)
                .with_arena_capacity_per_thread(4096),
            |_t, es| PlayoutEvaluator::new(es),
        ))
    };
    let f_r3 = |g: &UtttGame, s: u64| rand_uttt(g, s);
    let m4_res = run_multi_match(&root, [&f_mcts_4, &f_r3], 4, 2, 1970);
    let mcts_us_per_sim_4tree = (m4_res.choose_time_us[0].mean() / 120.0).max(0.1);

    let (az_1, az_4) = if let Some(c) = ctx {
        let eval = c.resolve_evaluator("live")?;
        let svc = InferenceService::new(eval, 32, 1);
        let svc_ref = &svc;
        let az_sims = 32usize;
        let f_az1 = |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
            Box::new(AlphaZeroPlayer::new(
                *g,
                svc_ref,
                AlphaZeroConfig::new(1, az_sims, s),
            ))
        };
        let f_r4 = |g: &UtttGame, s: u64| rand_uttt(g, s);
        let r1 = run_multi_match(&root, [&f_az1, &f_r4], 4, 2, 1980);
        let (_, uniq1) = svc.dedup_counts();
        let moves1 = r1.choose_time_us[0].count().max(1) as f32;
        let total_us_1 = r1.choose_time_us[0].mean()
            + ((uniq1 as f32) * svc.inference_time_stats().mean()) / moves1;

        svc.reset_stats();
        let f_az4 = |g: &UtttGame, s: u64| -> Box<dyn Player<UltimateTicTacToe, 2> + '_> {
            Box::new(AlphaZeroPlayer::new(
                *g,
                svc_ref,
                AlphaZeroConfig::new(4, 12, s),
            ))
        };
        let f_r5 = |g: &UtttGame, s: u64| rand_uttt(g, s);
        let r4 = run_multi_match(&root, [&f_az4, &f_r5], 4, 2, 1990);
        let (_, uniq4) = svc.dedup_counts();
        let moves4 = r4.choose_time_us[0].count().max(1) as f32;
        let total_us_4 = r4.choose_time_us[0].mean()
            + ((uniq4 as f32) * svc.inference_time_stats().mean()) / moves4;

        (
            (total_us_1 / (az_sims as f32)).max(1.0),
            (total_us_4 / 48.0).max(1.0),
        )
    } else {
        (12.0, 9.0)
    };

    Ok(CalibrationReport {
        game: "uttt",
        alphazero_us_per_sim_1tree: az_1,
        alphazero_us_per_sim_4tree: az_4,
        mcts_us_per_sim_1tree,
        mcts_us_per_sim_4tree,
        tree_minimax_us_per_eval,
        minimax_depths,
    })
}

fn build_ttt_player<'a>(
    cfg: &'a TttEngineConfig,
    service: Option<&'a InferenceService<CandleBatchEvaluator<27, 9, 2>, 27, 9, 2>>,
    game: &TttGame,
    seed: u64,
) -> Box<dyn Player<TicTacToe, 2> + 'a> {
    match cfg.engine.as_str() {
        "minimax" => {
            if cfg.use_heuristic {
                Box::new(MinimaxPlayer::new(*game, TttHeuristic, cfg.depth, seed))
            } else {
                Box::new(MinimaxPlayer::zero(*game, cfg.depth, seed))
            }
        }
        "tree-minimax" => {
            let budget = match cfg.budget_kind.as_str() {
                "evals" => MinimaxBudget::MaxEvaluations(cfg.budget_val.max(1)),
                "ms" => {
                    MinimaxBudget::MaxDuration(Duration::from_millis(cfg.budget_val.max(1) as u64))
                }
                _ => MinimaxBudget::FixedDepth(cfg.budget_val as u32),
            };
            let cap = cfg.arena_cap.max(64);
            if cfg.use_heuristic {
                Box::new(TreeMinimaxPlayer::new(
                    *game,
                    tic_tac_toe::NO_MOVE,
                    TttHeuristic,
                    budget,
                    seed,
                    cap,
                ))
            } else {
                Box::new(TreeMinimaxPlayer::zero(
                    *game,
                    tic_tac_toe::NO_MOVE,
                    budget,
                    seed,
                    cap,
                ))
            }
        }
        "mcts" => {
            let sims = cfg.sims.max(1);
            let trees = cfg.num_trees.max(1);
            let cap = (sims * 16).max(256);
            if trees == 1 {
                let m_cfg = MctsConfig::new(sims, seed)
                    .with_opening_moves(cfg.opening_moves)
                    .with_arena_capacity(cap);
                Box::new(MctsPlayer::new(
                    *game,
                    tic_tac_toe::NO_MOVE,
                    Uct::new(cfg.c_uct),
                    PlayoutEvaluator::new(seed.wrapping_add(1)),
                    m_cfg,
                ))
            } else {
                let rp_cfg = RootParallelConfig::new(
                    RootParallelBudget::SimulationsPerThread(sims),
                    seed,
                )
                .with_opening_moves(cfg.opening_moves)
                .with_arena_capacity_per_thread(cap);
                Box::new(RootParallelPlayer::new(
                    *game,
                    tic_tac_toe::NO_MOVE,
                    Uct::new(cfg.c_uct),
                    trees,
                    rp_cfg,
                    |_t, eval_seed| PlayoutEvaluator::new(eval_seed),
                ))
            }
        }
        "alphazero" => {
            let svc = service.expect("AlphaZero engine requires an active InferenceService");
            let az_cfg = AlphaZeroConfig::new(cfg.num_trees.max(1), cfg.sims.max(1), seed)
                .with_puct(cfg.c_base, cfg.c_init)
                .with_dirichlet(cfg.dir_alpha, cfg.dir_eps)
                .with_opening_moves(cfg.opening_moves);
            Box::new(AlphaZeroPlayer::new(*game, svc, az_cfg))
        }
        _ => Box::new(RandomPlayer::new(*game, seed)),
    }
}

fn build_uttt_player<'a>(
    cfg: &'a TttEngineConfig,
    service: Option<&'a InferenceService<CandleBatchEvaluator<324, 81, 2>, 324, 81, 2>>,
    game: &UtttGame,
    seed: u64,
) -> Box<dyn Player<UltimateTicTacToe, 2> + 'a> {
    let h = UtttHeuristic::default();
    match cfg.engine.as_str() {
        "minimax" => {
            if cfg.use_heuristic {
                Box::new(MinimaxPlayer::new(*game, h, cfg.depth, seed))
            } else {
                Box::new(MinimaxPlayer::zero(*game, cfg.depth, seed))
            }
        }
        "tree-minimax" => {
            let budget = match cfg.budget_kind.as_str() {
                "evals" => MinimaxBudget::MaxEvaluations(cfg.budget_val.max(1)),
                "ms" => {
                    MinimaxBudget::MaxDuration(Duration::from_millis(cfg.budget_val.max(1) as u64))
                }
                _ => MinimaxBudget::FixedDepth(cfg.budget_val as u32),
            };
            let cap = cfg.arena_cap.max(256);
            if cfg.use_heuristic {
                Box::new(TreeMinimaxPlayer::new(
                    *game,
                    ultimate_ttt::NO_MOVE,
                    h,
                    budget,
                    seed,
                    cap,
                ))
            } else {
                Box::new(TreeMinimaxPlayer::zero(
                    *game,
                    ultimate_ttt::NO_MOVE,
                    budget,
                    seed,
                    cap,
                ))
            }
        }
        "mcts" => {
            let sims = cfg.sims.max(1);
            let trees = cfg.num_trees.max(1);
            let cap = (sims * 32).max(1024);
            if trees == 1 {
                let m_cfg = MctsConfig::new(sims, seed)
                    .with_opening_moves(cfg.opening_moves)
                    .with_arena_capacity(cap);
                Box::new(MctsPlayer::new(
                    *game,
                    ultimate_ttt::NO_MOVE,
                    Uct::new(cfg.c_uct),
                    PlayoutEvaluator::new(seed.wrapping_add(1)),
                    m_cfg,
                ))
            } else {
                let rp_cfg = RootParallelConfig::new(
                    RootParallelBudget::SimulationsPerThread(sims),
                    seed,
                )
                .with_opening_moves(cfg.opening_moves)
                .with_arena_capacity_per_thread(cap);
                Box::new(RootParallelPlayer::new(
                    *game,
                    ultimate_ttt::NO_MOVE,
                    Uct::new(cfg.c_uct),
                    trees,
                    rp_cfg,
                    |_t, eval_seed| PlayoutEvaluator::new(eval_seed),
                ))
            }
        }
        "alphazero" => {
            let svc = service.expect("AlphaZero engine requires an active InferenceService");
            let az_cfg = AlphaZeroConfig::new(cfg.num_trees.max(1), cfg.sims.max(1), seed)
                .with_puct(cfg.c_base, cfg.c_init)
                .with_dirichlet(cfg.dir_alpha, cfg.dir_eps)
                .with_opening_moves(cfg.opening_moves);
            Box::new(AlphaZeroPlayer::new(*game, svc, az_cfg))
        }
        _ => Box::new(RandomPlayer::new(*game, seed)),
    }
}

fn discover_game_models(game: LabGame, has_live: bool, current_iter: usize) -> Vec<ModelInfo> {
    let mut list = Vec::new();
    let g_str = game.as_str();
    if has_live {
        list.push(ModelInfo {
            id: "live".to_string(),
            kind: "live",
            game: g_str,
            size_bytes: 0,
            description: format!(
                "Active Live {} Model (currently at Iter #{})",
                game.title(),
                current_iter
            ),
        });
    }
    list.push(ModelInfo {
        id: "untrained".to_string(),
        kind: "builtin",
        game: g_str,
        size_bytes: 0,
        description: format!("Untrained Random {} Weights (Iter 0 Baseline)", game.title()),
    });

    if let Ok(entries) = std::fs::read_dir(".") {
        let mut disk_files: Vec<(String, u64)> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                if !name.ends_with(".safetensors") {
                    return None;
                }
                let belongs_to_uttt = name.starts_with("uttt_");
                let matches_game = match game {
                    LabGame::Uttt => belongs_to_uttt,
                    LabGame::Ttt => !belongs_to_uttt,
                };
                if matches_game {
                    let sz = e.metadata().ok().map(|m| m.len()).unwrap_or(0);
                    Some((name, sz))
                } else {
                    None
                }
            })
            .collect();
        disk_files.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, sz) in disk_files {
            list.push(ModelInfo {
                id: name,
                kind: "disk",
                game: g_str,
                size_bytes: sz,
                description: format!("Saved {} checkpoint on disk", game.title()),
            });
        }
    }
    list
}

fn sanitize_safetensors_filename(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let clean: String = trimmed
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .collect();
    if clean.is_empty() || clean.starts_with('.') || clean.contains("..") {
        return None;
    }
    if clean.ends_with(".safetensors") {
        Some(clean)
    } else {
        Some(format!("{clean}.safetensors"))
    }
}

fn ensure_game_prefixed_filename(game: LabGame, raw: &str) -> Option<String> {
    let base = sanitize_safetensors_filename(raw)?;
    let prefix = game.prefix();
    if base.starts_with("ttt_") || base.starts_with("uttt_") {
        Some(base)
    } else {
        Some(format!("{prefix}{base}"))
    }
}

fn find_header_end(buf: &[u8]) -> Option<(usize, usize)> {
    for i in 0..buf.len() {
        if i + 4 <= buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, i + 4));
        }
        if i + 2 <= buf.len() && &buf[i..i + 2] == b"\n\n" {
            return Some((i, i + 2));
        }
    }
    None
}

fn read_http_request<R: Read>(reader: &mut R) -> std::io::Result<(String, String, String)> {
    let mut data = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let mut header_end = None;
    let mut content_length = 0usize;

    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                break;
            }
            Err(e) => return Err(e),
        };
        data.extend_from_slice(&chunk[..n]);

        if header_end.is_none()
            && let Some((hdr_end_idx, body_start_idx)) = find_header_end(&data)
        {
            header_end = Some(body_start_idx);
            let header_str = String::from_utf8_lossy(&data[..hdr_end_idx]);
            for line in header_str.lines() {
                if let Some((k, v)) = line.split_once(':')
                    && k.trim().eq_ignore_ascii_case("content-length")
                    && let Ok(len) = v.trim().parse::<usize>()
                {
                    content_length = len;
                }
            }
        }

        if let Some(body_start) = header_end
            && data.len() >= body_start + content_length
        {
            break;
        }

        if data.len() >= 65_536 {
            break;
        }
    }

    if data.is_empty() {
        return Ok((String::new(), String::new(), String::new()));
    }

    let (hdr_slice, body_slice) = match header_end {
        Some(body_start) => {
            let end = (body_start + content_length).min(data.len());
            (&data[..body_start], &data[body_start..end])
        }
        None => (&data[..], &[][..]),
    };

    let header_text = String::from_utf8_lossy(hdr_slice);
    let first_line = header_text.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let raw_path = parts.next().unwrap_or("/").to_string();
    let body = String::from_utf8_lossy(body_slice).into_owned();
    Ok((method, raw_path, body))
}

fn extract_query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        let k = kv.next().unwrap_or("");
        let v = kv.next().unwrap_or("");
        if k == key && !v.is_empty() {
            return Some(v);
        }
    }
    None
}

fn handle_http_connection(mut stream: TcpStream, hub: &TelemetryHub) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let (method, raw_path, body) = read_http_request(&mut stream)?;
    if method.is_empty() {
        return Ok(());
    }
    let method = method.as_str();
    let raw_path = raw_path.as_str();
    let body = body.as_str();

    if method == "OPTIONS" {
        return write_http_response(&mut stream, "204 No Content", "text/plain", "");
    }

    if method == "GET" && (raw_path == "/" || raw_path.starts_with("/index.html")) {
        return write_http_response(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            DASHBOARD_HTML,
        );
    }

    if method == "GET" && raw_path.starts_with("/api/state") {
        let state_json = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &state_json);
    }

    if (method == "POST" || method == "GET") && raw_path.starts_with("/api/train_control") {
        let query = raw_path.split('?').nth(1).unwrap_or("");
        let action = extract_json_str(body, "action")
            .or_else(|| extract_query_param(query, "action").map(str::to_string))
            .unwrap_or_default();
        let game_opt = extract_json_str(body, "game")
            .or_else(|| extract_query_param(query, "game").map(str::to_string));
        let total_iters_opt = extract_json_f64(body, "total_iterations").or_else(|| {
            extract_query_param(query, "total_iterations").and_then(|s| s.parse::<f64>().ok())
        });
        let auto_save_opt = extract_json_f64(body, "auto_save_every").or_else(|| {
            extract_query_param(query, "auto_save_every").and_then(|s| s.parse::<f64>().ok())
        });

        {
            let mut guard = hub.inner.lock().expect("telemetry mutex poisoned");
            if let Some(g_str) = game_opt {
                let new_game = LabGame::from_str_lossy(&g_str);
                if guard.active_game != new_game {
                    let prev = guard.active_game;
                    guard.state_for_mut(prev).knobs.paused = true;
                    guard.state_for_mut(prev).step_requests = 0;
                    guard.state_for_mut(prev).pause_after_iteration = false;
                    guard.active_game = new_game;
                    println!(
                        "🎮 [Browser Lab] Switched active game: {} -> {}",
                        prev.title(),
                        new_game.title()
                    );
                }
            }

            let active_game = guard.active_game;
            let st = guard.state_for_mut(active_game);
            apply_knob_updates(&mut st.knobs, body);
            if let Some(v) = total_iters_opt {
                st.total_iterations = (v as usize).clamp(1, 10_000);
            }
            if let Some(v) = auto_save_opt {
                st.auto_save_every = (v as usize).clamp(0, 1000);
            }

            match action.as_str() {
                "start" | "resume" => {
                    if st.current_iteration >= st.total_iterations {
                        st.total_iterations = st.current_iteration + 15;
                    }
                    st.knobs.paused = false;
                    st.pause_after_iteration = false;
                    st.phase = format!(
                        "Starting {} Iter #{} / {}...",
                        active_game.title(),
                        st.current_iteration + 1,
                        st.total_iterations
                    );
                    println!(
                        "▶️  [Browser Lab] Start/Resume {} training (Iter #{}..#{}, {} games/iter, {} sims/move)",
                        active_game.title(),
                        st.current_iteration + 1,
                        st.total_iterations,
                        st.knobs.games_per_iteration,
                        st.knobs.simulations_per_move
                    );
                }
                "pause" => {
                    st.knobs.paused = true;
                    st.step_requests = 0;
                    st.pause_after_iteration = false;
                    st.phase = format!("Paused (Iter #{})", st.current_iteration);
                    println!(
                        "⏸️  [Browser Lab] Paused {} training (at Iter #{})",
                        active_game.title(),
                        st.current_iteration
                    );
                }
                "step" => {
                    st.step_requests += 1;
                    st.knobs.paused = false;
                    if st.current_iteration >= st.total_iterations {
                        st.total_iterations = st.current_iteration + 1;
                    }
                    st.phase = format!(
                        "Stepping {} Iter #{}...",
                        active_game.title(),
                        st.current_iteration + 1
                    );
                    println!(
                        "⏭️  [Browser Lab] Stepping +1 iteration for {} (Iter #{})",
                        active_game.title(),
                        st.current_iteration + 1
                    );
                }
                "reset" => {
                    st.reset_requested = true;
                    st.knobs.paused = true;
                    st.step_requests = 0;
                    st.pause_after_iteration = false;
                    st.phase = "Resetting Model & Replay Buffer...".to_string();
                    println!(
                        "🔄 [Browser Lab] Requested reset for {} model & replay buffer",
                        active_game.title()
                    );
                }
                _ => {}
            }
        }
        let updated = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &updated);
    }

    if method == "POST" && raw_path.starts_with("/api/knobs") {
        {
            let mut guard = hub.inner.lock().expect("telemetry mutex poisoned");
            let active_game = guard.active_game;
            let st = guard.state_for_mut(active_game);
            let was_paused = st.knobs.paused;
            apply_knob_updates(&mut st.knobs, body);
            if was_paused && !st.knobs.paused && st.current_iteration >= st.total_iterations {
                st.total_iterations = st.current_iteration + 15;
            }
            if !was_paused && st.knobs.paused {
                st.step_requests = 0;
                st.pause_after_iteration = false;
                st.phase = format!("Paused (Iter #{})", st.current_iteration);
            }
            println!(
                "🎛️  [Browser Lab] Updated {} knobs: games={}, sims={}, steps={}, batch={}, lr={:.5}",
                active_game.title(),
                st.knobs.games_per_iteration,
                st.knobs.simulations_per_move,
                st.knobs.train_steps,
                st.knobs.batch_size,
                st.knobs.learning_rate
            );
        }
        let updated = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &updated);
    }

    if method == "POST" && raw_path.starts_with("/api/models") {
        let query = raw_path.split('?').nth(1).unwrap_or("");
        let action = extract_json_str(body, "action")
            .or_else(|| extract_query_param(query, "action").map(str::to_string))
            .unwrap_or_default();
        let name = extract_json_str(body, "name")
            .or_else(|| extract_query_param(query, "name").map(str::to_string))
            .unwrap_or_default();
        let game = extract_json_str(body, "game")
            .or_else(|| extract_query_param(query, "game").map(str::to_string))
            .map(|s| LabGame::from_str_lossy(&s))
            .unwrap_or_else(|| hub.active_game());

        let (ttt_ctx, uttt_ctx, cur_iter) = {
            let guard = hub.inner.lock().expect("telemetry mutex poisoned");
            (
                guard.ttt_context.clone(),
                guard.uttt_context.clone(),
                guard.state_for(game).current_iteration,
            )
        };

        let raw_target_name = if action == "snapshot" && name.trim().is_empty() {
            format!("{}iter_{:03}.safetensors", game.prefix(), cur_iter)
        } else {
            name
        };

        match action.as_str() {
            "save" | "snapshot" => {
                let Some(filename) = ensure_game_prefixed_filename(game, &raw_target_name) else {
                    return write_http_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        "{\"error\":\"Invalid model name\"}",
                    );
                };
                let save_res = match game {
                    LabGame::Ttt => {
                        let Some(ctx) = ttt_ctx else {
                            return write_http_response(
                                &mut stream,
                                "400 Bad Request",
                                "application/json",
                                "{\"error\":\"No live TTT model attached\"}",
                            );
                        };
                        ctx.live_service.evaluator().save_safetensors(&filename)
                    }
                    LabGame::Uttt => {
                        let Some(ctx) = uttt_ctx else {
                            return write_http_response(
                                &mut stream,
                                "400 Bad Request",
                                "application/json",
                                "{\"error\":\"No live UTTT model attached\"}",
                            );
                        };
                        ctx.live_service.evaluator().save_safetensors(&filename)
                    }
                };
                if let Err(e) = save_res {
                    let msg = format!("{{\"error\":\"{}\"}}", escape_json(&e.to_string()));
                    return write_http_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &msg,
                    );
                }
                hub.set_phase_for(game, cur_iter, format!("Saved checkpoint {filename}"));
            }
            "load_to_live" => {
                let Some(filename) = sanitize_safetensors_filename(&raw_target_name) else {
                    return write_http_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        "{\"error\":\"Invalid model name\"}",
                    );
                };
                let load_res = match game {
                    LabGame::Ttt => {
                        let Some(ctx) = ttt_ctx else {
                            return write_http_response(
                                &mut stream,
                                "400 Bad Request",
                                "application/json",
                                "{\"error\":\"No live TTT model attached\"}",
                            );
                        };
                        ctx.live_service.pause_inference();
                        let r = ctx.live_service.evaluator().load_safetensors(&filename);
                        ctx.live_service.resume_inference();
                        r
                    }
                    LabGame::Uttt => {
                        let Some(ctx) = uttt_ctx else {
                            return write_http_response(
                                &mut stream,
                                "400 Bad Request",
                                "application/json",
                                "{\"error\":\"No live UTTT model attached\"}",
                            );
                        };
                        ctx.live_service.pause_inference();
                        let r = ctx.live_service.evaluator().load_safetensors(&filename);
                        ctx.live_service.resume_inference();
                        r
                    }
                };
                if let Err(e) = load_res {
                    let msg = format!("{{\"error\":\"{}\"}}", escape_json(&e.to_string()));
                    return write_http_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &msg,
                    );
                }
                hub.set_phase_for(game, cur_iter, format!("Loaded {filename} into Live"));
            }
            "delete" => {
                if let Some(filename) = sanitize_safetensors_filename(&raw_target_name) {
                    let _ = std::fs::remove_file(&filename);
                }
            }
            _ => {}
        }

        let updated = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &updated);
    }

    if (method == "POST" || method == "GET") && raw_path.starts_with("/api/calibrate") {
        let query = raw_path.split('?').nth(1).unwrap_or("");
        let game = extract_json_str(body, "game")
            .or_else(|| extract_query_param(query, "game").map(str::to_string))
            .map(|s| LabGame::from_str_lossy(&s))
            .unwrap_or_else(|| hub.active_game());
        println!("⚡ [Browser Lab] Running hardware timing calibration for {}...", game.title());
        match hub.calibrate_game(game) {
            Ok(report) => {
                return write_http_response(
                    &mut stream,
                    "200 OK",
                    "application/json",
                    &report.to_json(),
                );
            }
            Err(err) => {
                let msg = format!("{{\"error\":\"{}\"}}", escape_json(&err));
                return write_http_response(&mut stream, "400 Bad Request", "application/json", &msg);
            }
        }
    }

    if method == "POST" && raw_path.starts_with("/api/match") {
        let query = raw_path.split('?').nth(1).unwrap_or("");
        let game = extract_json_str(body, "game")
            .or_else(|| extract_query_param(query, "game").map(str::to_string))
            .map(|s| LabGame::from_str_lossy(&s))
            .unwrap_or_else(|| hub.active_game());
        let cfg_a = parse_engine_config(body, "a_");
        let cfg_b = parse_engine_config(body, "b_");
        let num_games = extract_json_f64(body, "num_games")
            .map(|v| v as usize)
            .unwrap_or(40);
        let num_workers = extract_json_f64(body, "num_workers")
            .map(|v| v as usize)
            .unwrap_or(4);
        let max_batch_size = extract_json_f64(body, "max_batch_size")
            .map(|v| v as usize)
            .unwrap_or(32);
        let seed = extract_json_f64(body, "seed")
            .map(|v| v as u64)
            .unwrap_or(2026);

        println!(
            "⚔️  [Browser Lab] Starting {} multi-match ({} games): {} vs {}",
            game.title(),
            num_games,
            cfg_a.summary_label(),
            cfg_b.summary_label()
        );
        match hub.run_match_for_game(
            game,
            &cfg_a,
            &cfg_b,
            num_games,
            num_workers,
            max_batch_size,
            seed,
        ) {
            Ok(rec) => {
                return write_http_response(&mut stream, "200 OK", "application/json", &rec.to_json());
            }
            Err(err) => {
                let msg = format!("{{\"error\":\"{}\"}}", escape_json(&err));
                return write_http_response(&mut stream, "400 Bad Request", "application/json", &msg);
            }
        }
    }

    if method == "GET" && raw_path.starts_with("/api/inspect_ttt") {
        let query = raw_path.split('?').nth(1).unwrap_or("");
        let (game, sims, trees, c_init, dir_eps, model) = parse_ttt_query(query);

        let (ttt_ctx, inspector) = {
            let guard = hub.inner.lock().expect("telemetry mutex poisoned");
            (guard.ttt_context.clone(), guard.inspector.clone())
        };

        if let Some(ctx) = ttt_ctx {
            if model.is_empty() || model == "live" {
                let res = inspect_ttt_position_configured(
                    &game,
                    &ctx.live_service,
                    trees,
                    sims,
                    c_init,
                    dir_eps,
                );
                return write_http_response(&mut stream, "200 OK", "application/json", &res.to_json());
            } else if let Ok(eval) = ctx.resolve_evaluator(&model) {
                let temp_svc = InferenceService::new(eval, 32, 1);
                let res =
                    inspect_ttt_position_configured(&game, &temp_svc, trees, sims, c_init, dir_eps);
                return write_http_response(&mut stream, "200 OK", "application/json", &res.to_json());
            }
        }

        if let Some(insp_fn) = inspector {
            let res = insp_fn(&game, sims);
            return write_http_response(&mut stream, "200 OK", "application/json", &res.to_json());
        } else {
            return write_http_response(
                &mut stream,
                "503 Service Unavailable",
                "application/json",
                "{\"error\":\"inspector not attached\"}",
            );
        }
    }

    write_http_response(&mut stream, "404 Not Found", "text/plain", "Not Found")
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let header = format!(
        concat!(
            "HTTP/1.1 {}\r\n",
            "Content-Type: {}\r\n",
            "Content-Length: {}\r\n",
            "Access-Control-Allow-Origin: *\r\n",
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n",
            "Access-Control-Allow-Headers: Content-Type\r\n",
            "Cache-Control: no-store\r\n",
            "Connection: close\r\n\r\n"
        ),
        status,
        content_type,
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

fn parse_ttt_query(query: &str) -> (TttGame, usize, usize, f32, f32, String) {
    let mut state = EMPTY_STATE;
    let mut player = PlayerIndex::<2>::PLAYER1;
    let mut sims = 160usize;
    let mut trees = 1usize;
    let mut c_init = 1.25f32;
    let mut dir_eps = 0.0f32;
    let mut model = "live".to_string();

    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        let k = kv.next().unwrap_or("");
        let v = kv.next().unwrap_or("");
        match k {
            "board" => {
                for (i, ch) in v.chars().take(9).enumerate() {
                    state[i] = match ch {
                        'X' | 'x' | '1' => Symbol::Player1,
                        'O' | 'o' | '2' => Symbol::Player2,
                        _ => Symbol::Empty,
                    };
                }
            }
            "player" => {
                if v == "1" || v.eq_ignore_ascii_case("o") {
                    player = PlayerIndex::<2>::PLAYER2;
                }
            }
            "sims" => {
                if let Ok(parsed) = v.parse::<usize>() {
                    sims = parsed.clamp(4, 4000);
                }
            }
            "trees" => {
                if let Ok(parsed) = v.parse::<usize>() {
                    trees = parsed.clamp(1, 64);
                }
            }
            "c_init" => {
                if let Ok(parsed) = v.parse::<f32>() {
                    c_init = parsed.clamp(0.05, 10.0);
                }
            }
            "eps" => {
                if let Ok(parsed) = v.parse::<f32>() {
                    dir_eps = parsed.clamp(0.0, 1.0);
                }
            }
            "model" if !v.is_empty() => {
                model = v.to_string();
            }
            _ => {}
        }
    }

    (
        TttGame::new(player, state),
        sims,
        trees,
        c_init,
        dir_eps,
        model,
    )
}

fn parse_engine_config(body: &str, prefix: &str) -> TttEngineConfig {
    let mut cfg = TttEngineConfig::default();
    if let Some(v) = extract_json_str(body, &format!("{prefix}engine")) {
        cfg.engine = v;
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}depth")) {
        cfg.depth = (v as u32).clamp(0, 9);
    }
    if let Some(v) = extract_json_bool(body, &format!("{prefix}use_heuristic")) {
        cfg.use_heuristic = v;
    }
    if let Some(v) = extract_json_str(body, &format!("{prefix}budget_kind")) {
        cfg.budget_kind = v;
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}budget_val")) {
        cfg.budget_val = (v as usize).clamp(1, 200_000);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}arena_cap")) {
        cfg.arena_cap = (v as usize).clamp(64, 500_000);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}sims")) {
        cfg.sims = (v as usize).clamp(1, 50_000);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}num_trees")) {
        cfg.num_trees = (v as usize).clamp(1, 64);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}c_uct")) {
        cfg.c_uct = (v as f32).clamp(0.0, 10.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}opening_moves")) {
        cfg.opening_moves = (v as usize).clamp(0, 81);
    }
    if let Some(v) = extract_json_str(body, &format!("{prefix}model")) {
        cfg.model = v;
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}c_init")) {
        cfg.c_init = (v as f32).clamp(0.01, 20.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}c_base")) {
        cfg.c_base = (v as f32).clamp(100.0, 1_000_000.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}dir_alpha")) {
        cfg.dir_alpha = (v as f32).clamp(0.01, 10.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}dir_eps")) {
        cfg.dir_eps = (v as f32).clamp(0.0, 1.0);
    }
    cfg
}

fn apply_knob_updates(knobs: &mut LiveKnobs, json: &str) {
    if let Some(v) = extract_json_bool(json, "paused") {
        knobs.paused = v;
    }
    if let Some(v) = extract_json_f64(json, "games_per_iteration") {
        knobs.games_per_iteration = (v as usize).clamp(4, 2048);
    }
    if let Some(v) = extract_json_f64(json, "simulations_per_move") {
        knobs.simulations_per_move = (v as usize).clamp(4, 2000);
    }
    if let Some(v) = extract_json_f64(json, "train_steps") {
        knobs.train_steps = (v as usize).clamp(1, 1000);
    }
    if let Some(v) = extract_json_f64(json, "batch_size") {
        knobs.batch_size = (v as usize).clamp(8, 1024);
    }
    if let Some(v) = extract_json_f64(json, "learning_rate") {
        knobs.learning_rate = v.clamp(1e-6, 1e-1);
    }
    if let Some(v) = extract_json_f64(json, "dirichlet_epsilon") {
        knobs.dirichlet_epsilon = (v as f32).clamp(0.0, 0.9);
    }
    if let Some(v) = extract_json_f64(json, "c_init") {
        knobs.c_init = (v as f32).clamp(0.1, 10.0);
    }
}

fn extract_json_after_key<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    Some(rest[colon + 1..].trim_start())
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let s = extract_json_after_key(json, key)?;
    if s.starts_with("true") {
        Some(true)
    } else if s.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let s = extract_json_after_key(json, key)?;
    let end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E')
        .unwrap_or(s.len());
    s[..end].parse::<f64>().ok()
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let s = extract_json_after_key(json, key)?;
    let s = s.strip_prefix('"')?;
    let end = s.find('"')?;
    Some(s[..end].to_string())
}

#[inline]
fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pure-Rust AlphaZero Live Lab & Multi-Engine Arena</title>
<style>
  :root {
    --bg: #0b0f19;
    --panel: #131b2e;
    --panel-alt: #18233c;
    --border: #263457;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: var(--bg);
    color: var(--text);
    padding: 16px 22px;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
  }
  .title-block h1 {
    margin: 0;
    font-size: 1.25rem;
    color: var(--accent);
  }
  .title-block p {
    margin: 4px 0 0;
    font-size: 0.82rem;
    color: var(--muted);
  }
  .game-switcher {
    display: flex;
    gap: 6px;
    background: var(--panel);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .game-btn {
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 7px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 700;
    font-size: 0.84rem;
  }
  .game-btn.active {
    background: var(--accent);
    color: #0b0f19;
  }
  .badges {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }
  .badge {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
  }
  .badge.live { border-color: var(--green); color: var(--green); }
  .badge.paused { border-color: var(--amber); color: var(--amber); }
  .tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .tab-btn {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.85rem;
  }
  .tab-btn.active {
    background: var(--accent);
    color: #0b0f19;
    border-color: var(--accent);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
    gap: 14px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .card h2 {
    margin: 0 0 6px;
    font-size: 0.95rem;
    color: var(--accent);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .card p.hint {
    margin: 0 0 10px;
    font-size: 0.76rem;
    color: var(--muted);
  }
  canvas {
    width: 100%;
    height: 205px;
    background: #0d1322;
    border-radius: 6px;
    border: 1px solid #1e293b;
  }
  .legend {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    font-size: 0.75rem;
    margin-top: 6px;
    color: var(--muted);
  }
  .legend span::before {
    content: "● ";
  }
  .train-control-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    justify-content: space-between;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 14px;
  }
  .train-btn-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }
  .knobs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 10px;
  }
  .knob-item {
    background: var(--panel-alt);
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .knob-item label {
    display: block;
    font-size: 0.73rem;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .knob-item input, .knob-item select {
    width: 100%;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 5px 7px;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.83rem;
  }
  .btn {
    background: var(--accent);
    color: #0b0f19;
    border: none;
    padding: 7px 14px;
    border-radius: 6px;
    font-weight: 700;
    cursor: pointer;
    font-size: 0.82rem;
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn.secondary {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn.green {
    background: var(--green);
    color: #0b0f19;
  }
  .btn.amber {
    background: var(--amber);
    color: #0b0f19;
  }
  .btn.danger {
    background: #3b1826;
    color: var(--rose);
    border: 1px solid var(--rose);
  }
  .play-layout {
    display: grid;
    grid-template-columns: 330px 1fr;
    gap: 18px;
  }
  @media (max-width: 820px) {
    .play-layout { grid-template-columns: 1fr; }
  }
  .board {
    display: grid;
    grid-template-columns: repeat(3, 98px);
    grid-template-rows: repeat(3, 98px);
    gap: 8px;
    margin: 12px 0;
  }
  .cell {
    background: var(--panel-alt);
    border: 2px solid var(--border);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    position: relative;
    user-select: none;
  }
  .cell:hover { border-color: var(--accent); }
  .cell .sym {
    font-size: 2.0rem;
    font-weight: 800;
  }
  .cell .sym.x { color: var(--accent); }
  .cell .sym.o { color: var(--rose); }
  .cell .overlay {
    font-size: 0.67rem;
    color: var(--muted);
    text-align: center;
    line-height: 1.2;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
  }
  th, td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  .bar-wrap {
    width: 90px;
    background: #0b0f19;
    height: 8px;
    border-radius: 4px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
    overflow: hidden;
  }
  .bar-fill { height: 100%; background: var(--accent); }
  .hidden { display: none; }
  .engine-params {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
  }
  .est-pill {
    font-size: 0.76rem;
    padding: 3px 9px;
    border-radius: 999px;
    background: #0d1322;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-weight: 600;
  }
  .fairness-banner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    background: #0d1322;
    border: 1px solid var(--border);
    margin-bottom: 14px;
    font-size: 0.82rem;
  }
</style>
</head>
<body>
<header>
  <div class="title-block">
    <h1>🦀 Pure-Rust AlphaZero Lab, Calibrated Arena & Inspector</h1>
    <p>Interactive Training Control • Tic-Tac-Toe & Ultimate Tic-Tac-Toe • Equal-Compute Match Calibration</p>
  </div>
  <div class="game-switcher">
    <button id="game-btn-ttt" class="game-btn active" onclick="switchActiveGame('ttt')">🎯 Tic-Tac-Toe (3x3)</button>
    <button id="game-btn-uttt" class="game-btn" onclick="switchActiveGame('uttt')">⚔️ Ultimate Tic-Tac-Toe (9x9)</button>
  </div>
  <div class="badges">
    <span class="badge" id="device-badge">Device: ...</span>
    <span class="badge paused" id="phase-badge">Phase: Idle</span>
    <span class="badge" id="iter-badge">Iter: 0 / 0</span>
  </div>
</header>

<div class="tabs">
  <button class="tab-btn active" id="tab-btn-lab" onclick="switchTab('lab')">📊 Live Training & System Lab</button>
  <button class="tab-btn" id="tab-btn-arena" onclick="switchTab('arena')">⚔️ Multi-Match Arena & Compute Calibrator</button>
  <button class="tab-btn" id="tab-btn-play" onclick="switchTab('play')">🎮 3x3 Play & MCTS/NN Inspector</button>
</div>

<!-- TAB 1: TRAINING & SYSTEM LAB -->
<div id="tab-lab">
  <!-- Interactive Training Lifecycle Bar -->
  <div class="train-control-bar">
    <div class="train-btn-group">
      <button class="btn green" id="btn-start-train" onclick="sendTrainControl('start')">▶ Start / Resume Training</button>
      <button class="btn amber" id="btn-pause-train" onclick="sendTrainControl('pause')">⏸ Pause</button>
      <button class="btn secondary" id="btn-step-train" onclick="sendTrainControl('step')">⏭ Step (+1 Iter)</button>
      <button class="btn secondary" onclick="snapshotLiveModel()">📸 Save Snapshot Now</button>
      <button class="btn danger" onclick="confirmResetTraining()">🔄 Reset to Untrained</button>
    </div>
    <div class="train-btn-group" style="font-size:0.8rem; color:var(--muted);">
      <label>Target Iters:
        <input type="number" id="ctrl-total-iters" value="15" min="1" max="5000" style="width:65px; background:var(--bg); color:var(--text); border:1px solid var(--border); padding:4px 6px; border-radius:4px;" onchange="updateTrainTargets()">
      </label>
      <label>Auto-Save Every:
        <input type="number" id="ctrl-autosave" value="5" min="0" max="500" style="width:55px; background:var(--bg); color:var(--text); border:1px solid var(--border); padding:4px 6px; border-radius:4px;" onchange="updateTrainTargets()"> iters (0=off)
      </label>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>
        <span>1. Arena Win + 0.5×Draw Rate (%)</span>
        <span id="latest-wdl" style="font-size:0.78rem;color:var(--green)">—</span>
      </h2>
      <p class="hint" id="arena-chart-hint">True playing strength vs Random & Minimax baselines.</p>
      <canvas id="chart-arena" width="600" height="205"></canvas>
      <div class="legend">
        <span style="color:#34d399">vs Random</span>
        <span style="color:#38bdf8" id="legend-mm-mid">vs Minimax(d=2)</span>
        <span style="color:#fbbf24" id="legend-mm-strong">vs Minimax(d=5)</span>
      </div>
    </div>

    <div class="card">
      <h2>
        <span>2. Policy KL Divergence & Root Entropy</span>
        <span id="latest-kl" style="font-size:0.78rem;color:var(--accent)">—</span>
      </h2>
      <p class="hint">KL(π_MCTS || p_NN) has a true 0.0 floor. Root Entropy H(π) tracks exploration sharpness.</p>
      <canvas id="chart-policy" width="600" height="205"></canvas>
      <div class="legend">
        <span style="color:#38bdf8">Policy KL (→ 0)</span>
        <span style="color:#a78bfa">Root MCTS Entropy H(π)</span>
        <span style="color:#fb7185">Total Loss</span>
      </div>
    </div>

    <div class="card">
      <h2>
        <span>3. Value MSE & Game Length (Moves)</span>
        <span id="latest-val" style="font-size:0.78rem;color:var(--rose)">—</span>
      </h2>
      <p class="hint">Value MSE < 0.05 prevents premature collapse; rising game length shows deeper defense.</p>
      <canvas id="chart-value" width="600" height="205"></canvas>
      <div class="legend">
        <span style="color:#fb7185">Value MSE (left)</span>
        <span style="color:#34d399">Avg Game Moves / 10 (right)</span>
      </div>
    </div>

    <div class="card">
      <h2>
        <span>4. System Queue, Batch Fill & Intra-Batch Dedup</span>
        <span id="latest-sys" style="font-size:0.78rem;color:var(--amber)">—</span>
      </h2>
      <p class="hint">Logical batch vs Unique NN evaluations after intra-batch state deduplication.</p>
      <canvas id="chart-sys" width="600" height="205"></canvas>
      <div class="legend">
        <span style="color:#fbbf24">Logical Batch</span>
        <span style="color:#a78bfa">Unique NN Batch</span>
        <span style="color:#38bdf8">Dedup Saved (%)</span>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:14px;">
    <h2>
      <span>🎛️ Live Hyperparameter Knobs (<span id="knobs-game-label">Tic-Tac-Toe</span> — Applied on Next Iteration)</span>
      <button class="btn" onclick="pushKnobs()">Apply Knobs</button>
    </h2>
    <div class="knobs-grid">
      <div class="knob-item">
        <label>Self-Play Games / Iter</label>
        <input type="number" id="k-games" min="4" max="2048" step="4" value="128">
      </div>
      <div class="knob-item">
        <label>MCTS Sims / Move</label>
        <input type="number" id="k-sims" min="4" max="1000" step="4" value="80">
      </div>
      <div class="knob-item">
        <label>SGD Steps / Iter</label>
        <input type="number" id="k-steps" min="1" max="500" step="5" value="28">
      </div>
      <div class="knob-item">
        <label>SGD Mini-Batch Size</label>
        <input type="number" id="k-batch" min="16" max="512" step="16" value="64">
      </div>
      <div class="knob-item">
        <label>Learning Rate (AdamW)</label>
        <input type="number" id="k-lr" min="0.00001" max="0.05" step="0.0005" value="0.0015">
      </div>
      <div class="knob-item">
        <label>Root Dirichlet ε</label>
        <input type="number" id="k-eps" min="0.0" max="0.75" step="0.05" value="0.25">
      </div>
      <div class="knob-item">
        <label>PUCT c_init</label>
        <input type="number" id="k-cinit" min="0.2" max="5.0" step="0.05" value="1.25">
      </div>
    </div>
  </div>
</div>

<!-- TAB 2: MULTI-MATCH ARENA, NN MODELS & COMPUTE CALIBRATOR -->
<div id="tab-arena" class="hidden">
  <!-- NN Checkpoint Manager -->
  <div class="card" style="margin-bottom:14px;">
    <h2>
      <span>🧠 Trained Neural Network Checkpoints (<span id="models-game-label">Tic-Tac-Toe</span>)</span>
      <div style="display:flex; gap:8px; align-items:center;">
        <input type="text" id="save-model-name" placeholder="e.g. iter_010" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:5px 9px;border-radius:4px;font-family:inherit;font-size:0.8rem;width:180px;">
        <button class="btn" onclick="saveLiveModel()">💾 Save Live Model</button>
        <button class="btn secondary" onclick="snapshotLiveModel()">📸 Quick Snapshot</button>
      </div>
    </h2>
    <p class="hint">Checkpoints are scoped per game (<code>ttt_*.safetensors</code> and <code>uttt_*.safetensors</code>). Select any saved checkpoint in Engine A or Engine B below to compare training stages.</p>
    <table>
      <thead>
        <tr>
          <th>Model ID</th>
          <th>Kind</th>
          <th>Size</th>
          <th>Description</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="models-tbody"></tbody>
    </table>
  </div>

  <!-- Compute & Thinking-Time Calibration Panel -->
  <div class="card" style="margin-bottom:14px;">
    <h2>
      <span>⏱️ Compute & Thinking-Time Calibrator (<span id="cal-game-label">Tic-Tac-Toe</span>)</span>
      <button class="btn green" id="btn-calibrate" onclick="runCalibration()">⚡ Measure Hardware Speed & Calibrate</button>
    </h2>
    <p class="hint">Measures actual microsecond speeds on this machine for <b>AlphaZero</b>, <b>MCTS</b>, <b>Tree-Minimax</b>, and <b>Recursive Minimax</b> so you can pit algorithms against each other at equal thinking time per move.</p>
    <div id="cal-metrics-bar" style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; font-size:0.78rem; color:var(--muted);">
      <span class="badge">Click "⚡ Measure Hardware Speed & Calibrate" to benchmark this machine</span>
    </div>
    <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding-top:8px; border-top:1px solid var(--border);">
      <label style="font-size:0.8rem; color:var(--muted);">
        Target Thinking Time / Move (ms):
        <input type="number" id="cal-target-ms" value="2.0" min="0.05" max="1000" step="0.5" style="width:80px; background:var(--bg); color:var(--text); border:1px solid var(--border); padding:5px 7px; border-radius:4px; margin-left:4px;">
      </label>
      <div style="display:flex; gap:5px; flex-wrap:wrap;">
        <button class="btn secondary" style="padding:4px 8px;font-size:0.74rem;" onclick="setTargetMs(0.5)">0.5 ms</button>
        <button class="btn secondary" style="padding:4px 8px;font-size:0.74rem;" onclick="setTargetMs(1.0)">1.0 ms</button>
        <button class="btn secondary" style="padding:4px 8px;font-size:0.74rem;" onclick="setTargetMs(2.5)">2.5 ms</button>
        <button class="btn secondary" style="padding:4px 8px;font-size:0.74rem;" onclick="setTargetMs(5.0)">5.0 ms</button>
        <button class="btn secondary" style="padding:4px 8px;font-size:0.74rem;" onclick="setTargetMs(15.0)">15 ms</button>
      </div>
      <button class="btn" onclick="calibrateBothToTarget()">⚖️ Calibrate Both Engines to Target Time</button>
      <button class="btn secondary" onclick="matchEngineTime('b_to_a')">🔗 Match B → Engine A's Time</button>
      <button class="btn secondary" onclick="matchEngineTime('a_to_b')">🔗 Match A → Engine B's Time</button>
    </div>
  </div>

  <!-- Fairness Live Banner -->
  <div class="fairness-banner" id="fairness-banner">
    <span id="fairness-summary">⚖️ Configure Engine A and Engine B below. Estimated thinking time per move updates live.</span>
    <span id="fairness-ratio-badge" class="badge">Ratio: —</span>
  </div>

  <!-- Engine A vs Engine B Configuration -->
  <div class="grid" style="margin-bottom:14px;">
    <!-- ENGINE A -->
    <div class="card">
      <h2>
        <span style="color:var(--accent)">🅰️ Engine A Configuration</span>
        <span class="est-pill" id="a-est-pill">Est: — µs/move</span>
      </h2>
      <div class="knob-item">
        <label>Engine Type</label>
        <select id="a-engine" onchange="updateEngineForm('a')">
          <option value="alphazero" selected>AlphaZero (MCTS + Neural Network)</option>
          <option value="mcts">Classical MCTS (Random Rollouts)</option>
          <option value="tree-minimax">Tree-Minimax (Arena Subtree Reuse)</option>
          <option value="minimax">Recursive Minimax (Alpha-Beta)</option>
          <option value="random">Random Player (Baseline)</option>
        </select>
      </div>

      <div id="a-params-az" class="engine-params knobs-grid">
        <div class="knob-item">
          <label>Neural Network Model</label>
          <select id="a-az-model" class="model-select" onchange="updateEstimatedTimes()"></select>
        </div>
        <div class="knob-item">
          <label>Simulations / Tree</label>
          <input type="number" id="a-az-sims" value="64" min="1" max="10000" step="8" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Root-Parallel Trees (K)</label>
          <input type="number" id="a-az-trees" value="1" min="1" max="64" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>PUCT c_init</label>
          <input type="number" id="a-az-cinit" value="1.25" min="0.1" max="10.0" step="0.05">
        </div>
        <div class="knob-item">
          <label>Root Dirichlet ε</label>
          <input type="number" id="a-az-eps" value="0.0" min="0.0" max="0.75" step="0.05">
        </div>
        <div class="knob-item">
          <label>Opening Moves (Temp)</label>
          <input type="number" id="a-az-open" value="0" min="0" max="20">
        </div>
      </div>

      <div id="a-params-mcts" class="engine-params knobs-grid hidden">
        <div class="knob-item">
          <label>Playout Sims / Tree</label>
          <input type="number" id="a-mcts-sims" value="200" min="1" max="50000" step="20" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Root-Parallel Trees (K)</label>
          <input type="number" id="a-mcts-trees" value="1" min="1" max="64" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>UCT Constant (c)</label>
          <input type="number" id="a-mcts-c" value="1.414" min="0.1" max="5.0" step="0.1">
        </div>
        <div class="knob-item">
          <label>Opening Moves</label>
          <input type="number" id="a-mcts-open" value="0" min="0" max="20">
        </div>
      </div>

      <div id="a-params-tm" class="engine-params knobs-grid hidden">
        <div class="knob-item">
          <label>Budget Mode</label>
          <select id="a-tm-kind" onchange="updateEstimatedTimes()">
            <option value="evals" selected>Max Leaf Evaluations</option>
            <option value="depth">Fixed Depth (Plies)</option>
            <option value="ms">Time Limit (ms / move)</option>
          </select>
        </div>
        <div class="knob-item">
          <label>Budget Value (Evals / Depth / ms)</label>
          <input type="number" id="a-tm-val" value="250" min="1" max="100000" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Heuristic Evaluation</label>
          <select id="a-tm-heur">
            <option value="true" selected>Enabled (Board Heuristic)</option>
            <option value="false">Disabled (Terminal Only)</option>
          </select>
        </div>
        <div class="knob-item">
          <label>Arena Capacity (Nodes)</label>
          <input type="number" id="a-tm-cap" value="16384" min="256" max="500000" step="1024">
        </div>
      </div>

      <div id="a-params-mm" class="engine-params knobs-grid hidden">
        <div class="knob-item">
          <label>Search Depth (Plies)</label>
          <input type="number" id="a-mm-depth" value="2" min="1" max="9" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Heuristic Evaluation</label>
          <select id="a-mm-heur">
            <option value="true" selected>Enabled (Board Heuristic)</option>
            <option value="false">Disabled (Terminal Only)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- ENGINE B -->
    <div class="card">
      <h2>
        <span style="color:var(--rose)">🅱️ Engine B Configuration</span>
        <span class="est-pill" id="b-est-pill">Est: — µs/move</span>
      </h2>
      <div class="knob-item">
        <label>Engine Type</label>
        <select id="b-engine" onchange="updateEngineForm('b')">
          <option value="alphazero">AlphaZero (MCTS + Neural Network)</option>
          <option value="mcts" selected>Classical MCTS (Random Rollouts)</option>
          <option value="tree-minimax">Tree-Minimax (Arena Subtree Reuse)</option>
          <option value="minimax">Recursive Minimax (Alpha-Beta)</option>
          <option value="random">Random Player (Baseline)</option>
        </select>
      </div>

      <div id="b-params-az" class="engine-params knobs-grid hidden">
        <div class="knob-item">
          <label>Neural Network Model</label>
          <select id="b-az-model" class="model-select" onchange="updateEstimatedTimes()"></select>
        </div>
        <div class="knob-item">
          <label>Simulations / Tree</label>
          <input type="number" id="b-az-sims" value="64" min="1" max="10000" step="8" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Root-Parallel Trees (K)</label>
          <input type="number" id="b-az-trees" value="1" min="1" max="64" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>PUCT c_init</label>
          <input type="number" id="b-az-cinit" value="1.25" min="0.1" max="10.0" step="0.05">
        </div>
        <div class="knob-item">
          <label>Root Dirichlet ε</label>
          <input type="number" id="b-az-eps" value="0.0" min="0.0" max="0.75" step="0.05">
        </div>
        <div class="knob-item">
          <label>Opening Moves (Temp)</label>
          <input type="number" id="b-az-open" value="0" min="0" max="20">
        </div>
      </div>

      <div id="b-params-mcts" class="engine-params knobs-grid">
        <div class="knob-item">
          <label>Playout Sims / Tree</label>
          <input type="number" id="b-mcts-sims" value="200" min="1" max="50000" step="20" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Root-Parallel Trees (K)</label>
          <input type="number" id="b-mcts-trees" value="1" min="1" max="64" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>UCT Constant (c)</label>
          <input type="number" id="b-mcts-c" value="1.414" min="0.1" max="5.0" step="0.1">
        </div>
        <div class="knob-item">
          <label>Opening Moves</label>
          <input type="number" id="b-mcts-open" value="0" min="0" max="20">
        </div>
      </div>

      <div id="b-params-tm" class="engine-params knobs-grid hidden">
        <div class="knob-item">
          <label>Budget Mode</label>
          <select id="b-tm-kind" onchange="updateEstimatedTimes()">
            <option value="evals" selected>Max Leaf Evaluations</option>
            <option value="depth">Fixed Depth (Plies)</option>
            <option value="ms">Time Limit (ms / move)</option>
          </select>
        </div>
        <div class="knob-item">
          <label>Budget Value (Evals / Depth / ms)</label>
          <input type="number" id="b-tm-val" value="250" min="1" max="100000" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Heuristic Evaluation</label>
          <select id="b-tm-heur">
            <option value="true" selected>Enabled (Board Heuristic)</option>
            <option value="false">Disabled (Terminal Only)</option>
          </select>
        </div>
        <div class="knob-item">
          <label>Arena Capacity (Nodes)</label>
          <input type="number" id="b-tm-cap" value="16384" min="256" max="500000" step="1024">
        </div>
      </div>

      <div id="b-params-mm" class="engine-params knobs-grid hidden">
        <div class="knob-item">
          <label>Search Depth (Plies)</label>
          <input type="number" id="b-mm-depth" value="2" min="1" max="9" oninput="updateEstimatedTimes()">
        </div>
        <div class="knob-item">
          <label>Heuristic Evaluation</label>
          <select id="b-mm-heur">
            <option value="true" selected>Enabled (Board Heuristic)</option>
            <option value="false">Disabled (Terminal Only)</option>
          </select>
        </div>
      </div>
    </div>
  </div>

  <!-- Match Controls & Latest Result -->
  <div class="card" style="margin-bottom:14px;">
    <h2>
      <span>🏁 Run Multi-Match Tournament (<span id="match-game-label">Tic-Tac-Toe</span> — Role-Swapped 50% X / 50% O)</span>
      <button class="btn" id="run-match-btn" onclick="runArenaMatch()">⚔️ Start Multi-Match</button>
    </h2>
    <div class="knobs-grid" style="margin-bottom:12px;">
      <div class="knob-item">
        <label>Total Games (Role-Swapped)</label>
        <input type="number" id="m-games" value="40" min="2" max="2000" step="10">
      </div>
      <div class="knob-item">
        <label>Worker Threads</label>
        <input type="number" id="m-workers" value="4" min="1" max="32">
      </div>
      <div class="knob-item">
        <label>NN Max Batch Size</label>
        <input type="number" id="m-batch" value="32" min="1" max="256">
      </div>
      <div class="knob-item">
        <label>Deterministic Seed</label>
        <input type="number" id="m-seed" value="2026" min="1" max="999999">
      </div>
    </div>
    <div id="latest-match-banner" style="padding:10px 12px; background:var(--panel-alt); border:1px solid var(--border); border-radius:6px; font-size:0.84rem; color:var(--muted);">
      Configure Engine A and Engine B above and click <b>⚔️ Start Multi-Match</b>.
    </div>
  </div>

  <!-- Match History Table -->
  <div class="card">
    <h2><span>📜 Multi-Match Tournament History (<span id="hist-game-label">Tic-Tac-Toe</span>)</span></h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Engine A vs Engine B</th>
          <th>Games</th>
          <th>A Wins / Draws / B Wins</th>
          <th>A Score %</th>
          <th>Actual Compute (A vs B µs/mv)</th>
          <th>Speed</th>
          <th>NN Batch (Unique / Dedup%)</th>
        </tr>
      </thead>
      <tbody id="matches-tbody"></tbody>
    </table>
  </div>
</div>

<!-- TAB 3: INTERACTIVE PLAY & MCTS/NN INSPECTOR -->
<div id="tab-play" class="hidden">
  <div class="play-layout">
    <div class="card">
      <h2>
        <span>Tic-Tac-Toe Board</span>
        <span id="game-status-text" style="font-size:0.8rem;color:var(--green)">Your Turn (X)</span>
      </h2>
      <p class="hint">Click a square to play or inspect. Toggle "Auto-Play AI" to let AlphaZero reply automatically.</p>
      <div class="board" id="ttt-board"></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">
        <button class="btn" onclick="resetBoard()">Reset Board</button>
        <button class="btn secondary" onclick="stepAiMove()">Let AI Play Best Move</button>
      </div>
      <div style="margin-top:12px; display:grid; grid-template-columns: 1fr 1fr; gap:8px; font-size:0.78rem; color:var(--muted);">
        <label style="grid-column: span 2;">
          Inspect TTT NN Model:
          <select id="play-model" style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px;border-radius:4px;margin-top:2px;" onchange="inspectCurrentBoard()"></select>
        </label>
        <label>
          Total MCTS Sims:
          <input type="number" id="play-sims" value="160" min="8" max="2000" step="16" style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px;border-radius:4px;" onchange="inspectCurrentBoard()">
        </label>
        <label>
          Root-Parallel Trees:
          <input type="number" id="play-trees" value="4" min="1" max="32" style="width:100%;background:var(--bg);color:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px;border-radius:4px;" onchange="inspectCurrentBoard()">
        </label>
        <label>
          PUCT c_init:
          <input type="number" id="play-cinit" value="1.25" min="0.1" max="5.0" step="0.05" style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px;border-radius:4px;" onchange="inspectCurrentBoard()">
        </label>
        <label>
          Root Dirichlet ε:
          <input type="number" id="play-eps" value="0.0" min="0.0" max="0.75" step="0.05" style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px;border-radius:4px;" onchange="inspectCurrentBoard()">
        </label>
        <label style="grid-column: span 2; margin-top:4px;">
          <input type="checkbox" id="auto-ai" checked> Auto-reply with AI move
        </label>
      </div>
    </div>

    <div class="card">
      <h2>
        <span>🔍 Live Neural Network vs. MCTS Search Breakdown</span>
        <span id="nn-value-readout" style="font-size:0.82rem;color:var(--accent)">V(s) = 0.000</span>
      </h2>
      <p class="hint">
        Compares raw Neural Network policy priors <code>P(s, a)</code> against post-search MCTS visit counts <code>N(s, a)</code> and action values <code>Q(s, a)</code>.
        <span id="inspect-dedup-readout" style="color:var(--amber);margin-left:6px;"></span>
      </p>
      <table>
        <thead>
          <tr>
            <th>Cell (Row, Col)</th>
            <th>Raw NN Prior P(s,a)</th>
            <th>MCTS Visits N(s,a)</th>
            <th>MCTS Share π(s,a)</th>
            <th>Action Value Q(s,a)</th>
          </tr>
        </thead>
        <tbody id="inspect-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
let stateData = null;
let activeGame = 'ttt';
let knobsInitializedForGame = null;
let modelsSignature = '';
let tttModelsSignature = '';
let board = ['.', '.', '.', '.', '.', '.', '.', '.', '.'];
let currentPlayer = 0;
let lastInspect = null;

function switchTab(tab) {
  document.getElementById('tab-lab').classList.toggle('hidden', tab !== 'lab');
  document.getElementById('tab-arena').classList.toggle('hidden', tab !== 'arena');
  document.getElementById('tab-play').classList.toggle('hidden', tab !== 'play');
  document.getElementById('tab-btn-lab').classList.toggle('active', tab === 'lab');
  document.getElementById('tab-btn-arena').classList.toggle('active', tab === 'arena');
  document.getElementById('tab-btn-play').classList.toggle('active', tab === 'play');
  if (tab === 'play') inspectCurrentBoard();
  if (tab === 'arena') {
    updateEstimatedTimes();
    if (!stateData || !stateData.calibration) {
      runCalibration();
    }
  }
}

async function switchActiveGame(game) {
  activeGame = game;
  knobsInitializedForGame = null;
  modelsSignature = '';
  const res = await fetch(`/api/train_control?action=set_game&game=${encodeURIComponent(game)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ action: 'set_game', game })
  });
  if (res.ok) {
    stateData = await res.json();
    renderDashboard(stateData);
  } else {
    await fetchState();
  }
}

async function sendTrainControl(action) {
  const total_iterations = parseInt(document.getElementById('ctrl-total-iters').value, 10) || 15;
  const auto_save_every = parseInt(document.getElementById('ctrl-autosave').value, 10) || 0;
  const payload = {
    action,
    game: activeGame,
    total_iterations,
    auto_save_every,
    games_per_iteration: parseInt(document.getElementById('k-games').value, 10),
    simulations_per_move: parseInt(document.getElementById('k-sims').value, 10),
    train_steps: parseInt(document.getElementById('k-steps').value, 10),
    batch_size: parseInt(document.getElementById('k-batch').value, 10),
    learning_rate: parseFloat(document.getElementById('k-lr').value),
    dirichlet_epsilon: parseFloat(document.getElementById('k-eps').value),
    c_init: parseFloat(document.getElementById('k-cinit').value),
  };
  const qs = `action=${encodeURIComponent(action)}&game=${encodeURIComponent(activeGame)}&total_iterations=${total_iterations}&auto_save_every=${auto_save_every}`;
  const res = await fetch(`/api/train_control?${qs}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  if (res.ok) {
    stateData = await res.json();
    renderDashboard(stateData);
  } else {
    await fetchState();
  }
}

async function updateTrainTargets() {
  const total_iterations = parseInt(document.getElementById('ctrl-total-iters').value, 10) || 15;
  const auto_save_every = parseInt(document.getElementById('ctrl-autosave').value, 10) || 0;
  const qs = `action=set_targets&game=${encodeURIComponent(activeGame)}&total_iterations=${total_iterations}&auto_save_every=${auto_save_every}`;
  const res = await fetch(`/api/train_control?${qs}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ action: 'set_targets', game: activeGame, total_iterations, auto_save_every })
  });
  if (res.ok) {
    stateData = await res.json();
    renderDashboard(stateData);
  } else {
    await fetchState();
  }
}

async function confirmResetTraining() {
  await sendTrainControl('reset');
}

async function snapshotLiveModel() {
  const res = await fetch(`/api/models?action=snapshot&game=${encodeURIComponent(activeGame)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ action: 'snapshot', game: activeGame })
  });
  const data = await res.json();
  if (!data.error) {
    modelsSignature = '';
    stateData = data;
    renderDashboard(stateData);
  }
}

async function fetchState() {
  try {
    const res = await fetch('/api/state');
    stateData = await res.json();
    renderDashboard(stateData);
  } catch (e) {}
}

function renderDashboard(d) {
  activeGame = d.active_game || 'ttt';
  const title = d.active_game_title || (activeGame === 'uttt' ? 'Ultimate Tic-Tac-Toe (9x9)' : 'Tic-Tac-Toe (3x3)');

  document.getElementById('game-btn-ttt').classList.toggle('active', activeGame === 'ttt');
  document.getElementById('game-btn-uttt').classList.toggle('active', activeGame === 'uttt');
  document.getElementById('knobs-game-label').textContent = title;
  document.getElementById('models-game-label').textContent = title;
  document.getElementById('cal-game-label').textContent = title;
  document.getElementById('match-game-label').textContent = title;
  document.getElementById('hist-game-label').textContent = title;

  if (activeGame === 'uttt') {
    document.getElementById('legend-mm-mid').textContent = 'vs Minimax(d=1)';
    document.getElementById('legend-mm-strong').textContent = 'vs Minimax(d=2)';
    document.getElementById('arena-chart-hint').textContent = 'Ultimate Tic-Tac-Toe W+0.5*D % vs Random, Minimax(d=1), and Minimax(d=2).';
  } else {
    document.getElementById('legend-mm-mid').textContent = 'vs Minimax(d=2)';
    document.getElementById('legend-mm-strong').textContent = 'vs Minimax(d=5 Optimal)';
    document.getElementById('arena-chart-hint').textContent = 'Tic-Tac-Toe W+0.5*D % vs Random, Minimax(d=2), and Minimax(d=5 Optimal).';
  }

  document.getElementById('device-badge').textContent = 'Device: ' + d.device_name;
  const phaseEl = document.getElementById('phase-badge');
  phaseEl.textContent = (d.knobs.paused ? '⏸ ' : '🟢 ') + d.phase;
  phaseEl.className = 'badge ' + (d.knobs.paused ? 'paused' : 'live');
  document.getElementById('iter-badge').textContent = `Iter: ${d.current_iteration} / ${d.total_iterations}`;

  if (knobsInitializedForGame !== activeGame) {
    document.getElementById('k-games').value = d.knobs.games_per_iteration;
    document.getElementById('k-sims').value = d.knobs.simulations_per_move;
    document.getElementById('k-steps').value = d.knobs.train_steps;
    document.getElementById('k-batch').value = d.knobs.batch_size;
    document.getElementById('k-lr').value = d.knobs.learning_rate;
    document.getElementById('k-eps').value = d.knobs.dirichlet_epsilon;
    document.getElementById('k-cinit').value = d.knobs.c_init;
    document.getElementById('ctrl-total-iters').value = d.total_iterations;
    document.getElementById('ctrl-autosave').value = d.auto_save_every;
    knobsInitializedForGame = activeGame;
  } else if (document.activeElement !== document.getElementById('ctrl-total-iters')) {
    document.getElementById('ctrl-total-iters').value = d.total_iterations;
  }

  renderModels(d.models || [], d.ttt_models || []);
  renderCalibration(d.calibration);
  renderMatches(d.matches || []);

  const h = d.history || [];
  if (!h.length) {
    document.getElementById('latest-wdl').textContent = '—';
    document.getElementById('latest-kl').textContent = '—';
    document.getElementById('latest-val').textContent = '—';
    document.getElementById('latest-sys').textContent = '—';
    clearCanvas('chart-arena');
    clearCanvas('chart-policy');
    clearCanvas('chart-value');
    clearCanvas('chart-sys');
    return;
  }
  const last = h[h.length - 1];
  const scorePct = (arr) => {
    const tot = arr[0] + arr[1] + arr[2];
    return tot > 0 ? ((arr[0] + 0.5 * arr[1]) / tot) * 100.0 : 0.0;
  };
  const strongLabel = activeGame === 'uttt' ? 'vs MM(d=2)' : 'vs MM(d=5)';
  document.getElementById('latest-wdl').textContent =
    `${strongLabel}: ${last.vs_mm5[0]}W-${last.vs_mm5[1]}D-${last.vs_mm5[2]}L`;
  document.getElementById('latest-kl').textContent =
    `KL=${last.policy_kl.toFixed(4)} | H=${last.root_entropy_mean.toFixed(2)}`;
  document.getElementById('latest-val').textContent =
    `MSE=${last.value_loss.toFixed(4)} | Moves=${last.game_moves_mean.toFixed(1)}`;
  document.getElementById('latest-sys').textContent =
    `Batch=${last.batch_size_mean.toFixed(1)} → Uniq=${last.unique_batch_mean.toFixed(1)} (Saved ${last.dedup_saved_pct.toFixed(1)}%) | ${Math.round(last.simulations_per_sec)} sims/s`;

  drawChart('chart-arena', h.map(x => x.iteration), [
    { data: h.map(x => scorePct(x.vs_random)), color: '#34d399' },
    { data: h.map(x => scorePct(x.vs_mm2)), color: '#38bdf8' },
    { data: h.map(x => scorePct(x.vs_mm5)), color: '#fbbf24' },
  ], 0, 100);

  drawChart('chart-policy', h.map(x => x.iteration), [
    { data: h.map(x => x.policy_kl), color: '#38bdf8' },
    { data: h.map(x => x.root_entropy_mean), color: '#a78bfa' },
    { data: h.map(x => x.total_loss), color: '#fb7185' },
  ]);

  drawChart('chart-value', h.map(x => x.iteration), [
    { data: h.map(x => x.value_loss), color: '#fb7185' },
    { data: h.map(x => x.game_moves_mean / 10.0), color: '#34d399' },
  ]);

  drawChart('chart-sys', h.map(x => x.iteration), [
    { data: h.map(x => x.batch_size_mean), color: '#fbbf24' },
    { data: h.map(x => x.unique_batch_mean), color: '#a78bfa' },
    { data: h.map(x => x.dedup_saved_pct), color: '#38bdf8' },
  ], 0, Math.max(100, last.max_batch_size));
}

function clearCanvas(id) {
  const c = document.getElementById(id);
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
}

function renderModels(models, tttModels) {
  const sig = activeGame + ':' + models.map(m => `${m.id}:${m.size_bytes}:${m.description}`).join('|');
  if (sig !== modelsSignature) {
    modelsSignature = sig;
    const tbody = document.getElementById('models-tbody');
    tbody.innerHTML = '';
    models.forEach(m => {
      const tr = document.createElement('tr');
      const szText = m.size_bytes > 0 ? `${(m.size_bytes / 1024).toFixed(1)} KB` : 'In-Memory';
      let actions = '—';
      if (m.kind === 'disk') {
        actions = `
          <button class="btn secondary" style="padding:4px 8px;font-size:0.74rem;" onclick="loadModelToLive('${m.id}')">Load → Live</button>
          <button class="btn danger" style="padding:4px 8px;font-size:0.74rem;" onclick="deleteModel('${m.id}')">Delete</button>
        `;
      }
      tr.innerHTML = `
        <td><b>${m.id}</b></td>
        <td><span class="badge">${m.kind}</span></td>
        <td>${szText}</td>
        <td style="color:var(--muted)">${m.description}</td>
        <td>${actions}</td>
      `;
      tbody.appendChild(tr);
    });

    document.querySelectorAll('.model-select').forEach(sel => {
      const prev = sel.value || 'live';
      sel.innerHTML = '';
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = `${m.id} (${m.kind})`;
        sel.appendChild(opt);
      });
      if ([...sel.options].some(o => o.value === prev)) {
        sel.value = prev;
      }
    });
  }

  const tttSig = tttModels.map(m => `${m.id}:${m.size_bytes}`).join('|');
  if (tttSig !== tttModelsSignature) {
    tttModelsSignature = tttSig;
    const playSel = document.getElementById('play-model');
    if (playSel) {
      const prev = playSel.value || 'live';
      playSel.innerHTML = '';
      tttModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = `${m.id} (${m.kind})`;
        playSel.appendChild(opt);
      });
      if ([...playSel.options].some(o => o.value === prev)) {
        playSel.value = prev;
      }
    }
  }
}

function renderCalibration(cal) {
  const bar = document.getElementById('cal-metrics-bar');
  if (!cal) {
    bar.innerHTML = `<span class="badge">Not calibrated yet for ${activeGame.toUpperCase()} — click "⚡ Measure Hardware Speed & Calibrate"</span>`;
    updateEstimatedTimes();
    return;
  }
  const depthsText = (cal.minimax_depths || [])
    .map(d => `d=${d.depth}: ${formatUs(d.us_per_move)}`)
    .join(' • ');
  bar.innerHTML = `
    <span class="badge live">AlphaZero (1T): ${cal.alphazero_us_per_sim_1tree.toFixed(1)} µs/sim</span>
    <span class="badge live">AlphaZero (4T): ${cal.alphazero_us_per_sim_4tree.toFixed(1)} µs/sim</span>
    <span class="badge">MCTS (1T): ${cal.mcts_us_per_sim_1tree.toFixed(2)} µs/sim</span>
    <span class="badge">Tree-Minimax: ${cal.tree_minimax_us_per_eval.toFixed(2)} µs/eval</span>
    <span class="badge">Minimax: ${depthsText}</span>
  `;
  updateEstimatedTimes();
}

function formatUs(us) {
  if (!us || !isFinite(us)) return '—';
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms (${Math.round(us)} µs)`;
  return `${us.toFixed(0)} µs (${(us / 1000).toFixed(2)} ms)`;
}

function estimateEngineUs(prefix) {
  const cal = stateData && stateData.calibration ? stateData.calibration : null;
  const eng = document.getElementById(`${prefix}-engine`).value;
  if (eng === 'random') return 1.0;
  if (!cal) return null;

  if (eng === 'alphazero') {
    const sims = parseInt(document.getElementById(`${prefix}-az-sims`).value, 10) || 1;
    const trees = parseInt(document.getElementById(`${prefix}-az-trees`).value, 10) || 1;
    const perSim = trees > 1 ? cal.alphazero_us_per_sim_4tree : cal.alphazero_us_per_sim_1tree;
    return sims * trees * perSim;
  }
  if (eng === 'mcts') {
    const sims = parseInt(document.getElementById(`${prefix}-mcts-sims`).value, 10) || 1;
    const trees = parseInt(document.getElementById(`${prefix}-mcts-trees`).value, 10) || 1;
    const perSim = trees > 1 ? cal.mcts_us_per_sim_4tree : cal.mcts_us_per_sim_1tree;
    return sims * trees * perSim;
  }
  if (eng === 'tree-minimax') {
    const kind = document.getElementById(`${prefix}-tm-kind`).value;
    const val = parseInt(document.getElementById(`${prefix}-tm-val`).value, 10) || 1;
    if (kind === 'ms') return val * 1000.0;
    if (kind === 'evals') return val * cal.tree_minimax_us_per_eval;
    const found = (cal.minimax_depths || []).find(d => d.depth === val);
    return found ? found.us_per_move : val * 200.0;
  }
  if (eng === 'minimax') {
    const depth = parseInt(document.getElementById(`${prefix}-mm-depth`).value, 10) || 1;
    const depths = cal.minimax_depths || [];
    const exact = depths.find(d => d.depth === depth);
    if (exact) return exact.us_per_move;
    if (depths.length > 0) {
      const last = depths[depths.length - 1];
      return last.us_per_move * Math.pow(6, depth - last.depth);
    }
  }
  return null;
}

function applyTargetUsToEngine(prefix, targetUs) {
  const cal = stateData && stateData.calibration ? stateData.calibration : null;
  if (!cal || !targetUs || targetUs <= 0) return;
  const eng = document.getElementById(`${prefix}-engine`).value;

  if (eng === 'alphazero') {
    const trees = parseInt(document.getElementById(`${prefix}-az-trees`).value, 10) || 1;
    const perSim = trees > 1 ? cal.alphazero_us_per_sim_4tree : cal.alphazero_us_per_sim_1tree;
    const sims = Math.max(2, Math.round(targetUs / (trees * Math.max(0.1, perSim))));
    document.getElementById(`${prefix}-az-sims`).value = sims;
  } else if (eng === 'mcts') {
    const trees = parseInt(document.getElementById(`${prefix}-mcts-trees`).value, 10) || 1;
    const perSim = trees > 1 ? cal.mcts_us_per_sim_4tree : cal.mcts_us_per_sim_1tree;
    const sims = Math.max(4, Math.round(targetUs / (trees * Math.max(0.05, perSim))));
    document.getElementById(`${prefix}-mcts-sims`).value = sims;
  } else if (eng === 'tree-minimax') {
    const kind = document.getElementById(`${prefix}-tm-kind`).value;
    if (kind === 'ms') {
      document.getElementById(`${prefix}-tm-val`).value = Math.max(1, Math.round(targetUs / 1000.0));
    } else if (kind === 'evals') {
      const evals = Math.max(10, Math.round(targetUs / Math.max(0.05, cal.tree_minimax_us_per_eval)));
      document.getElementById(`${prefix}-tm-val`).value = evals;
    } else {
      const best = findClosestDepth(cal.minimax_depths || [], targetUs);
      document.getElementById(`${prefix}-tm-val`).value = best;
    }
  } else if (eng === 'minimax') {
    const best = findClosestDepth(cal.minimax_depths || [], targetUs);
    document.getElementById(`${prefix}-mm-depth`).value = best;
  }
}

function findClosestDepth(depths, targetUs) {
  if (!depths.length) return 2;
  let bestDepth = depths[0].depth;
  let bestDiff = Math.abs(Math.log(Math.max(1, depths[0].us_per_move)) - Math.log(Math.max(1, targetUs)));
  depths.forEach(d => {
    const diff = Math.abs(Math.log(Math.max(1, d.us_per_move)) - Math.log(Math.max(1, targetUs)));
    if (diff < bestDiff) {
      bestDiff = diff;
      bestDepth = d.depth;
    }
  });
  return bestDepth;
}

function updateEstimatedTimes() {
  const usA = estimateEngineUs('a');
  const usB = estimateEngineUs('b');
  document.getElementById('a-est-pill').textContent = usA ? `⏱️ Est: ${formatUs(usA)} / mv` : '⏱️ Est: (Calibrate first)';
  document.getElementById('b-est-pill').textContent = usB ? `⏱️ Est: ${formatUs(usB)} / mv` : '⏱️ Est: (Calibrate first)';

  const summary = document.getElementById('fairness-summary');
  const badge = document.getElementById('fairness-ratio-badge');
  if (usA && usB) {
    const ratio = Math.max(usA, usB) / Math.max(0.1, Math.min(usA, usB));
    const fair = ratio <= 1.35;
    summary.innerHTML = `⚖️ <b>Estimated Compute / Move:</b> Engine A <code>${formatUs(usA)}</code> vs Engine B <code>${formatUs(usB)}</code>`;
    badge.textContent = fair ? `Ratio: ${ratio.toFixed(2)}x (Fair Compute ✅)` : `Ratio: ${ratio.toFixed(1)}x (Unbalanced ⚠️)`;
    badge.className = 'badge ' + (fair ? 'live' : 'paused');
  } else {
    summary.textContent = '⚖️ Click "⚡ Measure Hardware Speed & Calibrate" above to estimate per-move thinking times.';
    badge.textContent = 'Ratio: —';
    badge.className = 'badge';
  }
}

async function runCalibration() {
  const btn = document.getElementById('btn-calibrate');
  btn.disabled = true;
  btn.textContent = '⏳ Benchmarking Hardware...';
  try {
    const res = await fetch('/api/calibrate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ game: activeGame })
    });
    const cal = await res.json();
    if (!cal.error && stateData) {
      stateData.calibration = cal;
      renderCalibration(cal);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Measure Hardware Speed & Calibrate';
  }
}

function setTargetMs(ms) {
  document.getElementById('cal-target-ms').value = ms;
  calibrateBothToTarget();
}

async function calibrateBothToTarget() {
  if (!stateData || !stateData.calibration) {
    await runCalibration();
  }
  const targetMs = parseFloat(document.getElementById('cal-target-ms').value) || 2.0;
  const targetUs = targetMs * 1000.0;
  applyTargetUsToEngine('a', targetUs);
  applyTargetUsToEngine('b', targetUs);
  updateEstimatedTimes();
}

async function matchEngineTime(dir) {
  if (!stateData || !stateData.calibration) {
    await runCalibration();
  }
  if (dir === 'b_to_a') {
    const usA = estimateEngineUs('a');
    if (usA) {
      document.getElementById('cal-target-ms').value = (usA / 1000.0).toFixed(2);
      applyTargetUsToEngine('b', usA);
    }
  } else {
    const usB = estimateEngineUs('b');
    if (usB) {
      document.getElementById('cal-target-ms').value = (usB / 1000.0).toFixed(2);
      applyTargetUsToEngine('a', usB);
    }
  }
  updateEstimatedTimes();
}

function renderMatches(matches) {
  const tbody = document.getElementById('matches-tbody');
  tbody.innerHTML = '';
  [...matches].reverse().forEach(m => {
    const tr = document.createElement('tr');
    const nnText = m.nn_logical_batch > 0
      ? `${m.nn_logical_batch.toFixed(1)} → ${m.nn_unique_batch.toFixed(1)} (-${m.nn_dedup_saved_pct.toFixed(1)}%)`
      : '—';
    const ratio = Math.max(m.a_us_per_move, m.b_us_per_move) / Math.max(0.1, Math.min(m.a_us_per_move, m.b_us_per_move));
    const computeText = `${Math.round(m.a_us_per_move)} vs ${Math.round(m.b_us_per_move)} µs (${ratio.toFixed(2)}x)`;
    tr.innerHTML = `
      <td>#${m.match_id}</td>
      <td><span style="color:var(--accent)">${m.engine_a}</span> <span style="color:var(--muted)">vs</span> <span style="color:var(--rose)">${m.engine_b}</span></td>
      <td>${m.num_games}</td>
      <td><b>${m.wins_a}W</b> - ${m.draws}D - <b>${m.wins_b}L</b></td>
      <td style="color:${m.score_a_pct >= 50 ? 'var(--green)' : 'var(--rose)'}"><b>${m.score_a_pct.toFixed(1)}%</b></td>
      <td>${computeText}</td>
      <td>${m.games_per_sec.toFixed(0)} g/s (${m.duration_ms.toFixed(0)} ms)</td>
      <td>${nnText}</td>
    `;
    tbody.appendChild(tr);
  });
}

function updateEngineForm(prefix) {
  const eng = document.getElementById(`${prefix}-engine`).value;
  document.getElementById(`${prefix}-params-az`).classList.toggle('hidden', eng !== 'alphazero');
  document.getElementById(`${prefix}-params-mcts`).classList.toggle('hidden', eng !== 'mcts');
  document.getElementById(`${prefix}-params-tm`).classList.toggle('hidden', eng !== 'tree-minimax');
  document.getElementById(`${prefix}-params-mm`).classList.toggle('hidden', eng !== 'minimax');
  updateEstimatedTimes();
}

function collectEngineConfig(prefix) {
  const eng = document.getElementById(`${prefix}-engine`).value;
  const out = { [`${prefix}_engine`]: eng };
  if (eng === 'alphazero') {
    out[`${prefix}_model`] = document.getElementById(`${prefix}-az-model`).value || 'live';
    out[`${prefix}_sims`] = parseInt(document.getElementById(`${prefix}-az-sims`).value, 10);
    out[`${prefix}_num_trees`] = parseInt(document.getElementById(`${prefix}-az-trees`).value, 10);
    out[`${prefix}_c_init`] = parseFloat(document.getElementById(`${prefix}-az-cinit`).value);
    out[`${prefix}_dir_eps`] = parseFloat(document.getElementById(`${prefix}-az-eps`).value);
    out[`${prefix}_opening_moves`] = parseInt(document.getElementById(`${prefix}-az-open`).value, 10);
  } else if (eng === 'mcts') {
    out[`${prefix}_sims`] = parseInt(document.getElementById(`${prefix}-mcts-sims`).value, 10);
    out[`${prefix}_num_trees`] = parseInt(document.getElementById(`${prefix}-mcts-trees`).value, 10);
    out[`${prefix}_c_uct`] = parseFloat(document.getElementById(`${prefix}-mcts-c`).value);
    out[`${prefix}_opening_moves`] = parseInt(document.getElementById(`${prefix}-mcts-open`).value, 10);
  } else if (eng === 'tree-minimax') {
    out[`${prefix}_budget_kind`] = document.getElementById(`${prefix}-tm-kind`).value;
    out[`${prefix}_budget_val`] = parseInt(document.getElementById(`${prefix}-tm-val`).value, 10);
    out[`${prefix}_use_heuristic`] = document.getElementById(`${prefix}-tm-heur`).value === 'true';
    out[`${prefix}_arena_cap`] = parseInt(document.getElementById(`${prefix}-tm-cap`).value, 10);
  } else if (eng === 'minimax') {
    out[`${prefix}_depth`] = parseInt(document.getElementById(`${prefix}-mm-depth`).value, 10);
    out[`${prefix}_use_heuristic`] = document.getElementById(`${prefix}-mm-heur`).value === 'true';
  }
  return out;
}

async function runArenaMatch() {
  const btn = document.getElementById('run-match-btn');
  const banner = document.getElementById('latest-match-banner');
  btn.disabled = true;
  btn.textContent = '⏳ Running Match...';
  banner.textContent = 'Running parallel role-swapped tournament...';

  const payload = {
    game: activeGame,
    ...collectEngineConfig('a'),
    ...collectEngineConfig('b'),
    num_games: parseInt(document.getElementById('m-games').value, 10),
    num_workers: parseInt(document.getElementById('m-workers').value, 10),
    max_batch_size: parseInt(document.getElementById('m-batch').value, 10),
    seed: parseInt(document.getElementById('m-seed').value, 10),
  };

  try {
    const res = await fetch('/api/match', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const rec = await res.json();
    if (rec.error) {
      banner.innerHTML = `<span style="color:var(--rose)">Error: ${rec.error}</span>`;
    } else {
      const dedupInfo = rec.nn_logical_batch > 0
        ? ` | NN Batch: <b>${rec.nn_logical_batch.toFixed(1)}</b> → Unique: <b>${rec.nn_unique_batch.toFixed(1)}</b> (Saved <b>${rec.nn_dedup_saved_pct.toFixed(1)}%</b> duplicate evals)`
        : '';
      const ratio = Math.max(rec.a_us_per_move, rec.b_us_per_move) / Math.max(0.1, Math.min(rec.a_us_per_move, rec.b_us_per_move));
      banner.innerHTML = `
        <b>Match #${rec.match_id} (${rec.game.toUpperCase()}) Complete (${rec.duration_ms.toFixed(1)} ms, ${rec.games_per_sec.toFixed(0)} games/sec):</b>
        <span style="color:var(--accent)">${rec.engine_a}</span> vs <span style="color:var(--rose)">${rec.engine_b}</span> —
        <b style="color:var(--green)">${rec.wins_a}W - ${rec.draws}D - ${rec.wins_b}L</b>
        (A Score: <b>${rec.score_a_pct.toFixed(1)}%</b>) |
        Actual Compute: <b>${Math.round(rec.a_us_per_move)} µs/mv</b> vs <b>${Math.round(rec.b_us_per_move)} µs/mv</b> (<b>${ratio.toFixed(2)}x</b>)${dedupInfo}
      `;
      fetchState();
    }
  } catch (e) {
    banner.innerHTML = `<span style="color:var(--rose)">Request failed: ${e}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '⚔️ Start Multi-Match';
  }
}

async function saveLiveModel() {
  const nameInput = document.getElementById('save-model-name');
  const name = nameInput.value.trim();
  if (!name) return;
  const res = await fetch('/api/models', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ action: 'save', name, game: activeGame })
  });
  const data = await res.json();
  modelsSignature = '';
  renderDashboard(data);
}

async function loadModelToLive(name) {
  const res = await fetch('/api/models', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ action: 'load_to_live', name, game: activeGame })
  });
  const data = await res.json();
  modelsSignature = '';
  renderDashboard(data);
}

async function deleteModel(name) {
  const res = await fetch('/api/models', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ action: 'delete', name, game: activeGame })
  });
  const data = await res.json();
  modelsSignature = '';
  renderDashboard(data);
}

function drawChart(canvasId, xs, series, fixedMin, fixedMax) {
  const c = document.getElementById(canvasId);
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  if (!xs.length) return;

  let minY = fixedMin !== undefined ? fixedMin : Infinity;
  let maxY = fixedMax !== undefined ? fixedMax : -Infinity;
  if (fixedMin === undefined || fixedMax === undefined) {
    series.forEach(s => s.data.forEach(v => {
      if (v < minY) minY = v;
      if (v > maxY) maxY = v;
    }));
  }
  if (minY === maxY) { minY -= 0.5; maxY += 0.5; }
  const padL = 44, padR = 14, padT = 14, padB = 24;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  ctx.strokeStyle = '#1e293b';
  ctx.fillStyle = '#64748b';
  ctx.font = '11px monospace';
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    const val = maxY - ((maxY - minY) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), 4, y + 4);
  }

  series.forEach(s => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    s.data.forEach((v, idx) => {
      const x = padL + (xs.length === 1 ? plotW / 2 : (plotW * idx) / (xs.length - 1));
      const y = padT + plotH * (1 - (v - minY) / (maxY - minY));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

async function pushKnobs() {
  const body = {
    games_per_iteration: parseInt(document.getElementById('k-games').value, 10),
    simulations_per_move: parseInt(document.getElementById('k-sims').value, 10),
    train_steps: parseInt(document.getElementById('k-steps').value, 10),
    batch_size: parseInt(document.getElementById('k-batch').value, 10),
    learning_rate: parseFloat(document.getElementById('k-lr').value),
    dirichlet_epsilon: parseFloat(document.getElementById('k-eps').value),
    c_init: parseFloat(document.getElementById('k-cinit').value),
  };
  const res = await fetch('/api/knobs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  stateData = await res.json();
  renderDashboard(stateData);
}

// Interactive 3x3 Tic-Tac-Toe Board & Inspector
function renderBoardUI() {
  const container = document.getElementById('ttt-board');
  container.innerHTML = '';
  const totalVisits = lastInspect ? lastInspect.mcts_visits.reduce((a, b) => a + b, 0) : 0;

  for (let i = 0; i < 9; i++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    const sym = board[i];
    if (sym === 'X' || sym === 'O') {
      const sp = document.createElement('div');
      sp.className = 'sym ' + (sym === 'X' ? 'x' : 'o');
      sp.textContent = sym;
      cell.appendChild(sp);
    } else if (lastInspect && lastInspect.legal_mask[i]) {
      const prior = (lastInspect.raw_nn_priors[i] * 100).toFixed(0);
      const visits = lastInspect.mcts_visits[i];
      const share = totalVisits > 0 ? ((visits / totalVisits) * 100).toFixed(0) : 0;
      const q = lastInspect.mcts_q_values[i].toFixed(2);
      const ov = document.createElement('div');
      ov.className = 'overlay';
      ov.innerHTML = `<b style="color:#38bdf8">${share}%</b> (${visits}v)<br>NN:${prior}% Q:${q}`;
      cell.appendChild(ov);
    }
    cell.onclick = () => onCellClick(i);
    container.appendChild(cell);
  }
}

async function inspectCurrentBoard() {
  const bStr = board.join('');
  const sims = document.getElementById('play-sims').value || 160;
  const trees = document.getElementById('play-trees').value || 4;
  const cinit = document.getElementById('play-cinit').value || 1.25;
  const eps = document.getElementById('play-eps').value || 0.0;
  const model = document.getElementById('play-model').value || 'live';
  try {
    const res = await fetch(`/api/inspect_ttt?board=${bStr}&player=${currentPlayer}&sims=${sims}&trees=${trees}&c_init=${cinit}&eps=${eps}&model=${encodeURIComponent(model)}`);
    if (!res.ok) return;
    lastInspect = await res.json();
    updateInspectTable();
    renderBoardUI();
  } catch (e) {}
}

function updateInspectTable() {
  if (!lastInspect) return;
  document.getElementById('nn-value-readout').textContent =
    `Raw NN Value V(s) = ${lastInspect.raw_nn_value >= 0 ? '+' : ''}${lastInspect.raw_nn_value.toFixed(3)}`;

  const reqs = lastInspect.nn_requests || 0;
  const uniq = lastInspect.nn_unique_evals || 0;
  const savedPct = reqs > 0 && uniq < reqs ? ((1 - uniq / reqs) * 100).toFixed(1) : '0.0';
  document.getElementById('inspect-dedup-readout').textContent =
    `[NN Evals: ${uniq} unique / ${reqs} requests (${savedPct}% dedup saved)]`;

  const stEl = document.getElementById('game-status-text');
  if (lastInspect.status === 'ongoing') {
    stEl.textContent = currentPlayer === 0 ? 'Turn: Player 1 (X)' : 'Turn: Player 2 (O)';
    stEl.style.color = '#34d399';
  } else {
    stEl.textContent = `Terminal: ${lastInspect.status.toUpperCase()}`;
    stEl.style.color = '#fbbf24';
  }

  const tbody = document.getElementById('inspect-tbody');
  tbody.innerHTML = '';
  const totalVisits = lastInspect.mcts_visits.reduce((a, b) => a + b, 0);

  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3), c = i % 3;
    const legal = lastInspect.legal_mask[i];
    const priorPct = (lastInspect.raw_nn_priors[i] * 100).toFixed(1);
    const visits = lastInspect.mcts_visits[i];
    const sharePct = totalVisits > 0 ? ((visits / totalVisits) * 100).toFixed(1) : '0.0';
    const q = lastInspect.mcts_q_values[i].toFixed(3);
    const isBest = lastInspect.chosen_move === i;

    const tr = document.createElement('tr');
    if (isBest) tr.style.background = '#1e293b';
    tr.innerHTML = `
      <td>Square ${i} (${r},${c}) ${isBest ? '⭐' : ''}</td>
      <td>${legal ? `<div class="bar-wrap"><div class="bar-fill" style="width:${priorPct}%;background:#a78bfa"></div></div>${priorPct}%` : '—'}</td>
      <td>${legal ? visits : '—'}</td>
      <td>${legal ? `<div class="bar-wrap"><div class="bar-fill" style="width:${sharePct}%"></div></div>${sharePct}%` : '—'}</td>
      <td style="color:${lastInspect.mcts_q_values[i] >= 0 ? '#34d399' : '#fb7185'}">${legal ? q : '—'}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function onCellClick(idx) {
  if (!lastInspect || lastInspect.status !== 'ongoing' || !lastInspect.legal_mask[idx]) return;
  board[idx] = currentPlayer === 0 ? 'X' : 'O';
  currentPlayer = 1 - currentPlayer;
  await inspectCurrentBoard();
  if (document.getElementById('auto-ai').checked && lastInspect && lastInspect.status === 'ongoing') {
    await stepAiMove();
  }
}

async function stepAiMove() {
  await inspectCurrentBoard();
  if (lastInspect && lastInspect.status === 'ongoing' && lastInspect.chosen_move !== null) {
    board[lastInspect.chosen_move] = currentPlayer === 0 ? 'X' : 'O';
    currentPlayer = 1 - currentPlayer;
    await inspectCurrentBoard();
  }
}

function resetBoard() {
  board = ['.', '.', '.', '.', '.', '.', '.', '.', '.'];
  currentPlayer = 0;
  inspectCurrentBoard();
}

renderBoardUI();
fetchState();
setInterval(fetchState, 800);
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::AlphaZeroNetConfig;
    use crate::training::AlphaZeroTrainer;
    use rand::SeedableRng;

    #[test]
    fn telemetry_hub_json_and_ttt_position_inspector_work() {
        let dev = Device::Cpu;
        let trainer = AlphaZeroTrainer::<27, 9, 2>::new(
            AlphaZeroNetConfig::tic_tac_toe_mlp(),
            1e-3,
            1e-4,
            dev.clone(),
        )
        .expect("trainer init");
        let service = Arc::new(
            trainer
                .build_inference_service(16, &[dev])
                .expect("service init"),
        );

        let hub = TelemetryHub::new("Cpu", 10, LiveKnobs::default());
        let svc_ref = Arc::clone(&service);
        hub.set_inspector(move |game, sims| inspect_ttt_position(game, &svc_ref, sims));

        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let inspection = inspect_ttt_position(&game, &service, 32);
        assert_eq!(inspection.status, "ongoing");
        assert!(inspection.legal_mask.iter().all(|&b| b));
        assert!(inspection.chosen_move.is_some());

        let prior_sum: f32 = inspection.raw_nn_priors.iter().sum();
        assert!(
            (prior_sum - 1.0).abs() < 1e-4,
            "priors must sum to 1.0, got {prior_sum}"
        );

        let json = hub.to_json();
        assert!(json.contains("\"device_name\":\"Cpu\""));
    }

    #[test]
    fn arena_multi_match_and_calibration_work_for_ttt_and_uttt() {
        let dev = Device::Cpu;
        let ttt_cfg = AlphaZeroNetConfig::tic_tac_toe_mlp();
        let ttt_trainer =
            AlphaZeroTrainer::<27, 9, 2>::new(ttt_cfg, 1e-3, 1e-4, dev.clone()).expect("ttt init");
        let ttt_service = Arc::new(
            ttt_trainer
                .build_inference_service(16, std::slice::from_ref(&dev))
                .expect("ttt service"),
        );

        let uttt_cfg = AlphaZeroNetConfig::ultimate_ttt_mlp();
        let uttt_trainer = AlphaZeroTrainer::<324, 81, 2>::new(uttt_cfg, 1e-3, 1e-4, dev.clone())
            .expect("uttt init");
        let uttt_service = Arc::new(
            uttt_trainer
                .build_inference_service(16, std::slice::from_ref(&dev))
                .expect("uttt service"),
        );

        let hub = TelemetryHub::new("Cpu", 5, LiveKnobs::default());
        hub.attach_ttt_service(ttt_service, ttt_cfg, vec![dev.clone()]);
        hub.attach_uttt_service(uttt_service, uttt_cfg, vec![dev]);

        // 1. Calibration works for TTT
        let cal_ttt = hub.calibrate_game(LabGame::Ttt).expect("calibrate ttt");
        assert_eq!(cal_ttt.game, "ttt");
        assert!(cal_ttt.alphazero_us_per_sim_1tree > 0.0);
        assert!(!cal_ttt.minimax_depths.is_empty());

        // 2. Match works for UTTT (AlphaZero vs Tree-Minimax)
        let cfg_az = TttEngineConfig {
            engine: "alphazero".to_string(),
            model: "live".to_string(),
            sims: 8,
            num_trees: 1,
            ..Default::default()
        };
        let cfg_tm = TttEngineConfig {
            engine: "tree-minimax".to_string(),
            budget_kind: "evals".to_string(),
            budget_val: 32,
            use_heuristic: true,
            arena_cap: 2048,
            ..Default::default()
        };
        let rec = hub
            .run_match_for_game(LabGame::Uttt, &cfg_az, &cfg_tm, 4, 2, 16, 42)
            .expect("uttt match");
        assert_eq!(rec.game, "uttt");
        assert_eq!(rec.wins_a + rec.draws + rec.wins_b, 4);
        assert!(rec.a_us_per_move > 0.0);
        assert!(rec.b_us_per_move > 0.0);

        // 3. UTTT self-play + SGD train_epoch
        let mut uttt_trainer = uttt_trainer;
        let mut uttt_buffer = crate::training::ReplayBuffer::<324, 81, 2>::new(4096);
        let mut rng = rand::prelude::SmallRng::seed_from_u64(2026);
        let root_uttt = UtttGame::new(PlayerIndex::PLAYER1, UTTT_EMPTY_STATE);
        let sp_cfg = crate::alphazero::SelfPlayConfig::new(4, 8, 5131)
            .with_workers(2)
            .with_opening_moves(4)
            .with_puct(19_652.0, 1.25)
            .with_dirichlet(0.3, 0.25);
        let uttt_svc = hub
            .inner
            .lock()
            .unwrap()
            .uttt_context
            .as_ref()
            .unwrap()
            .live_service
            .clone();
        let sp_res = crate::alphazero::run_self_play(&root_uttt, &uttt_svc, &sp_cfg);
        assert_eq!(sp_res.total_games, 4);
        uttt_buffer.extend(&sp_res.positions);
        let metrics = uttt_trainer
            .train_epoch(&uttt_buffer, &uttt_svc, 16, 4, &mut rng)
            .expect("uttt train_epoch");
        assert!(metrics.total_loss.is_finite());
    }

    struct FragmentedReader {
        chunks: Vec<Vec<u8>>,
        idx: usize,
    }

    impl Read for FragmentedReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.idx >= self.chunks.len() {
                return Ok(0);
            }
            let chunk = &self.chunks[self.idx];
            let n = chunk.len().min(buf.len());
            buf[..n].copy_from_slice(&chunk[..n]);
            self.idx += 1;
            Ok(n)
        }
    }

    #[test]
    fn fragmented_http_post_body_is_reassembled_across_reads() {
        let json_body = b"{\"action\":\"start\",\"game\":\"uttt\",\"total_iterations\":30}";
        let headers = format!(
            "POST /api/train_control HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            json_body.len()
        );
        let mut reader = FragmentedReader {
            chunks: vec![headers.into_bytes(), json_body.to_vec()],
            idx: 0,
        };
        let (method, path, body) = read_http_request(&mut reader).expect("read fragmented post");
        assert_eq!(method, "POST");
        assert_eq!(path, "/api/train_control");
        assert_eq!(extract_json_str(&body, "action").as_deref(), Some("start"));
        assert_eq!(extract_json_str(&body, "game").as_deref(), Some("uttt"));
        assert_eq!(extract_json_f64(&body, "total_iterations"), Some(30.0));
    }
}
