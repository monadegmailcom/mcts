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
//! - `POST /api/models`: Saves the live NN to `checkpoints/<game>_<name>.safetensors`, takes a 1-click
//!   iteration snapshot, loads a `.safetensors` checkpoint into the live model, or deletes a checkpoint.
//! - `POST /api/calibrate`: Benchmarks actual per-move thinking times (`µs/sim`, `µs/eval`, and
//!   per-depth Minimax `µs/move`) on the current hardware for fair equal-compute matches.
//! - `POST /api/match`: Runs a parallel role-swapped multi-match (`ttt` or `uttt`) between any two
//!   configurable engines (`alphazero`, `mcts`, `tree-minimax`, `minimax`, `random`) and reports
//!   both W/D/L and actual measured thinking time (`µs/move`) per engine.
//! - `POST /api/play`: Board and engine analysis for a single game in the Play tab (see [`crate::play`]).
//! - `POST /api/elo`: Queues a calibration of the game's Elo anchors (`anchors`, `games_per_pair`);
//!   see [`crate::elo`]. Ratings of the network appear in the history and experiment points.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};


use crate::alphazero::{AlphaZeroConfig, AlphaZeroGame, AlphaZeroPlayer, ValueQSchedule};
use crate::game::PlayerIndex;
use crate::games::tic_tac_toe::{self, EMPTY_STATE, TicTacToe, TttGame, TttHeuristic};
use crate::games::ultimate_ttt::{
    self, EMPTY_STATE as UTTT_EMPTY_STATE, UltimateTicTacToe, UtttGame, UtttHeuristic,
};
use crate::games::rubiks_cube::{self, RoundStats, RubiksCube, RubiksSettings};
use crate::play_cube::{self, CubePlayRequest};
use crate::elo::EloPool;
use crate::experiments::{ExperimentPoint, ExperimentSpec, ExperimentStore, RunStatus, json_value};
use crate::hyperparams::AlphaZeroHyperparams;
use crate::checkpoints;
use crate::opponents::{self, OpponentSpec};
use crate::play::{self, PlayRequest, PlayableGame};
use crate::utilization::UtilizationMonitor;
use crate::inference::InferenceService;
use crate::matchup::{RandomPlayer, run_multi_match, run_multi_match_with_openings};
use crate::mcts::{Fpu, MctsConfig, MctsPlayer, PlayoutEvaluator, Uct};
use crate::minimax::{MinimaxBudget, MinimaxPlayer, TreeMinimaxPlayer};
use crate::backend::{Backend, NetEvaluator, NnDevice};
use crate::nn::AlphaZeroNetConfig;
use crate::player::Player;
use crate::root_parallel::{RootParallelBudget, RootParallelConfig, RootParallelPlayer};

/// Selectable game environment in the Browser Lab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabGame {
    /// Standard 3x3 Tic-Tac-Toe (`G = 27, P = 9, N = 2`).
    Ttt,
    /// 9x9 Ultimate Tic-Tac-Toe (`G = 324, P = 81, N = 2`).
    Uttt,
    /// 3x3x3 Rubik's cube, trained with the round game (`G = 324, P = 12, N = 1`; see
    /// [`rubiks_cube::RoundGame`]). A one-player puzzle: no opponents, Elo or arena.
    Rubiks,
}

impl LabGame {
    pub const ALL: [Self; 3] = [Self::Ttt, Self::Uttt, Self::Rubiks];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ttt => "ttt",
            Self::Uttt => "uttt",
            Self::Rubiks => "rubiks",
        }
    }

    #[inline]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Ttt => "Tic-Tac-Toe (3x3)",
            Self::Uttt => "Ultimate Tic-Tac-Toe (9x9)",
            Self::Rubiks => "Rubik's Cube (3x3x3)",
        }
    }

    /// Prefix of the game's checkpoint files.
    #[inline]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Ttt => "ttt_",
            Self::Uttt => "uttt_",
            Self::Rubiks => "rubiks_",
        }
    }

    /// One-player puzzles have no opponents: no arena, Elo or win/draw/loss.
    #[inline]
    pub const fn is_puzzle(self) -> bool {
        matches!(self, Self::Rubiks)
    }

    #[inline]
    pub fn from_str_lossy(raw: &str) -> Self {
        let raw = raw.trim();
        Self::ALL.into_iter().find(|g| g.as_str().eq_ignore_ascii_case(raw)).unwrap_or(Self::Ttt)
    }

    /// The game a checkpoint file belongs to, by its prefix (unprefixed files count as TTT).
    pub fn of_checkpoint(name: &str) -> Self {
        [Self::Uttt, Self::Rubiks].into_iter().find(|g| name.starts_with(g.prefix())).unwrap_or(Self::Ttt)
    }
}

/// Action returned to the background training worker loop by [`TelemetryHub::poll_training_command`].
#[derive(Clone, Debug)]
pub enum TrainingCommand {
    /// Training is currently paused; sleep briefly and poll again.
    Idle,
    /// Reset the trainer, live inference weights, replay buffer, and telemetry history for `LabGame`.
    Reset(LabGame),
    /// Execute one training iteration (`iteration` 1-based) for `LabGame`.
    RunIteration { game: LabGame, iteration: usize },
    /// Run a queued experiment to completion (already marked running). Only issued while live
    /// training is paused.
    RunExperiment { id: u64, spec: Box<ExperimentSpec> },
    /// Play a round-robin between `anchors` and fit their Elo ratings. Only issued while live
    /// training is paused.
    CalibrateElo { game: LabGame, anchors: Vec<OpponentSpec>, games_per_pair: usize },
}

/// `[wins, draws, losses]` of the trained network against one evaluation opponent.
#[derive(Clone, Debug)]
pub struct ArenaResult {
    pub opponent: OpponentSpec,
    /// The opponent's label, or why it could not play.
    pub label: String,
    pub wdl: [usize; 3],
}

/// Default Elo anchors: fixed-strength engines from very weak to strong. Saved networks can be
/// added in the Browser Lab to extend the scale upwards.
pub fn default_elo_anchors(game: LabGame) -> Vec<OpponentSpec> {
    match game {
        LabGame::Ttt => vec![OpponentSpec::Random, OpponentSpec::Minimax { depth: 1 }, OpponentSpec::Minimax { depth: 2 }],
        LabGame::Uttt => vec![
            OpponentSpec::Random,
            OpponentSpec::Minimax { depth: 1 },
            OpponentSpec::Mcts { sims: 200, exploration: 1.414 },
            OpponentSpec::Minimax { depth: 2 },
            OpponentSpec::Minimax { depth: 3 },
        ],
        LabGame::Rubiks => Vec::new(),
    }
}

/// Default evaluation opponents of the live-training arena.
pub fn default_eval_opponents(game: LabGame) -> Vec<OpponentSpec> {
    match game {
        // With the TTT heuristic, Minimax already plays perfectly at depth 2.
        LabGame::Ttt => vec![OpponentSpec::Random, OpponentSpec::Minimax { depth: 2 }],
        LabGame::Uttt => vec![
            OpponentSpec::Mcts { sims: 200, exploration: 1.414 },
            OpponentSpec::Minimax { depth: 1 },
            OpponentSpec::Minimax { depth: 2 },
        ],
        LabGame::Rubiks => Vec::new(),
    }
}

/// Round-game results of one Rubik's cube iteration.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RoundsTelemetry {
    pub self_play: RoundStats,
    pub eval: RoundStats,
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
    /// Arena results against each configured evaluation opponent (two-player games).
    pub arena: Vec<ArenaResult>,
    /// How the round games went (Rubik's cube): self-play and, if evaluated, the noise-free games.
    pub rounds: Option<RoundsTelemetry>,
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
    /// The anchored rating implied by this iteration's arena results.
    fn rating(&self, pool: Option<&EloPool>) -> Option<crate::elo::Rating> {
        pool?.estimate(self.arena.iter().map(|a| (&a.opponent, a.wdl)))
    }

    fn to_json(&self, pool: Option<&EloPool>) -> String {
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
                "\"arena\":[{}],",
                "\"elo\":{},",
                "\"rounds\":{},",
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
            self.arena
                .iter()
                .map(|r| format!(
                    "{{\"label\":\"{}\",\"wdl\":[{},{},{}]}}",
                    escape_json(&r.label),
                    r.wdl[0],
                    r.wdl[1],
                    r.wdl[2]
                ))
                .collect::<Vec<_>>()
                .join(","),
            serde_json::to_string(&self.rating(pool)).unwrap_or_else(|_| "null".into()),
            serde_json::to_string(&self.rounds).unwrap_or_else(|_| "null".into()),
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
    /// All tunable hyperparameters (read and written through `Deref`, e.g. `knobs.batch_size`).
    pub hp: AlphaZeroHyperparams,
}

impl std::ops::Deref for LiveKnobs {
    type Target = AlphaZeroHyperparams;
    fn deref(&self) -> &AlphaZeroHyperparams {
        &self.hp
    }
}

impl std::ops::DerefMut for LiveKnobs {
    fn deref_mut(&mut self) -> &mut AlphaZeroHyperparams {
        &mut self.hp
    }
}

impl Default for LiveKnobs {
    fn default() -> Self {
        Self::default_ttt()
    }
}

impl LiveKnobs {
    /// Paused knobs starting from a game's recommended hyperparameters.
    pub const fn new(hp: AlphaZeroHyperparams) -> Self {
        Self { paused: true, hp }
    }

    pub fn for_game<R: AlphaZeroGame<G, P, N>, const G: usize, const P: usize, const N: usize>() -> Self {
        Self::new(R::hyperparams())
    }

    pub fn default_ttt() -> Self {
        Self::for_game::<TicTacToe, 27, 9, 2>()
    }

    pub fn default_uttt() -> Self {
        Self::for_game::<UltimateTicTacToe, { ultimate_ttt::INPUT_SIZE }, { ultimate_ttt::POLICY_SIZE }, 2>()
    }

    pub fn default_rubiks() -> Self {
        Self::for_game::<RubiksCube, { rubiks_cube::INPUT_SIZE }, { rubiks_cube::POLICY_SIZE }, 1>()
    }

    fn to_json(self) -> String {
        format!(
            concat!(
                "{{",
                "\"paused\":{},",
                "\"games_per_iteration\":{},",
                "\"simulations_per_move\":{},",
                "\"self_play_threads\":{},",
                "\"opening_moves\":{},",
                "\"dirichlet_alpha\":{:.4},",
                "\"dirichlet_epsilon\":{:.4},",
                "\"c_init\":{:.4},",
                "\"c_base\":{:.1},",
                "\"fpu\":{},",
                "\"value_q_weight\":{:.3},",
                "\"value_q_schedule\":{},",
                "\"train_steps\":{},",
                "\"batch_size\":{},",
                "\"learning_rate\":{:.6},",
                "\"weight_decay\":{:.6},",
                "\"buffer_capacity\":{},",
                "\"eval_games\":{},",
                "\"eval_sims\":{},",
                "\"eval_threads\":{}",
                "}}"
            ),
            self.paused,
            self.games_per_iteration,
            self.simulations_per_move,
            self.self_play_threads,
            self.opening_moves,
            self.dirichlet_alpha,
            self.dirichlet_epsilon,
            self.c_init,
            self.c_base,
            serde_json::to_string(&self.fpu).unwrap_or_default(),
            self.value_q_weight,
            serde_json::to_string(&self.value_q_schedule).unwrap_or_default(),
            self.train_steps,
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            self.buffer_capacity,
            self.eval_games,
            self.eval_sims,
            self.eval_threads,
        )
    }
}

/// Which compute device the lab trains and runs inference on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceChoice {
    Cpu,
    /// The best accelerator compiled in and present (CUDA or Metal); falls back to CPU.
    Gpu,
}

impl DeviceChoice {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// Process-wide compute settings shared by both games. Changing them rebuilds the trainers and
/// inference services (weights are kept, the AdamW moment estimates are reset).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemSettings {
    /// Neural-network backend (candle, or libtorch when built with the `tch` feature).
    pub backend: Backend,
    pub device: DeviceChoice,
    /// Maximum positions per batched NN inference call.
    pub infer_batch: usize,
    /// Inference worker threads, one model replica each (CUDA: one per GPU; Metal: always 1).
    pub infer_workers: usize,
}

impl SystemSettings {
    /// The training device these settings resolve to (falls back to candle / CPU when the chosen
    /// backend or GPU is unavailable).
    pub fn train_device(&self) -> NnDevice {
        let backend = if self.backend.is_available() { self.backend } else { Backend::Candle };
        match self.device {
            DeviceChoice::Gpu => NnDevice::best(backend),
            DeviceChoice::Cpu => NnDevice::cpu(backend),
        }
    }

    /// Training device plus the inference replica devices actually used.
    pub fn devices(&self) -> (NnDevice, Vec<NnDevice>) {
        let dev = self.train_device();
        (dev, dev.replicas(self.infer_workers))
    }

    fn to_json(self, pending: bool) -> String {
        let backends = Backend::available()
            .into_iter()
            .map(|b| {
                let gpu = b.gpu_name().map_or_else(|| "null".to_string(), |g| format!("\"{}\"", escape_json(&g)));
                format!("{{\"name\":\"{}\",\"gpu\":{gpu}}}", b.as_str())
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"backend\":\"{}\",\"device\":\"{}\",\"infer_batch\":{},\"infer_workers\":{},",
                "\"backends\":[{}],\"cpu_cores\":{},\"pending\":{}}}"
            ),
            self.backend.as_str(),
            self.device.as_str(),
            self.infer_batch,
            self.infer_workers,
            backends,
            std::thread::available_parallelism().map_or(1, |n| n.get()),
            pending,
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

/// Shared inference service over a game's network.
type SharedService<const G: usize, const P: usize, const N: usize> = Arc<InferenceService<NetEvaluator<G, P, N>, G, P, N>>;

#[derive(Clone)]
struct ModelContext<const G: usize, const P: usize, const N: usize> {
    live_service: SharedService<G, P, N>,
    net_config: AlphaZeroNetConfig,
    devices: Vec<NnDevice>,
    /// Checkpoint networks loaded for the Play tab, least recently used first; shared by clones,
    /// so a game's engines load their network once instead of on every move.
    play_networks: Arc<Mutex<Vec<(PlayNetworkKey, SharedService<G, P, N>)>>>,
}

/// A checkpoint file and its modification time (a rewritten file is loaded again).
type PlayNetworkKey = (String, Option<std::time::SystemTime>);

/// Play-tab networks kept per game (enough for both sides and the analyst).
const PLAY_NETWORK_CACHE: usize = 4;

impl<const G: usize, const P: usize, const N: usize> ModelContext<G, P, N> {
    fn new(live_service: SharedService<G, P, N>, net_config: AlphaZeroNetConfig, devices: Vec<NnDevice>) -> Self {
        Self { live_service, net_config, devices, play_networks: Arc::default() }
    }
}

impl<const G: usize, const P: usize, const N: usize> ModelContext<G, P, N> {
    /// Network for a Play-tab AlphaZero engine: the live model, or a checkpoint file (loaded with
    /// the architecture recorded in it).
    fn play_network<R: AlphaZeroGame<G, P, N>>(&self, checkpoint: &str) -> Result<SharedService<G, P, N>, String> {
        if checkpoint.is_empty() || checkpoint == "live" {
            return Ok(Arc::clone(&self.live_service));
        }
        let file = checkpoints::sanitize_name(checkpoint).ok_or_else(|| format!("invalid checkpoint name '{checkpoint}'"))?;
        let path = checkpoints::path(&file);
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let key: PlayNetworkKey = (file, modified);

        let mut cache = self.play_networks.lock().expect("play network cache poisoned");
        if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(pos);
            let service = Arc::clone(&entry.1);
            cache.push(entry);
            return Ok(service);
        }
        let config = checkpoints::net_config::<R, G, P, N>(&path)?;
        let evaluator = NetEvaluator::standalone(&config, &self.devices, Some(&path))
            .map_err(|e| format!("cannot load '{}': {e}", key.0))?;
        let replicas = evaluator.num_replicas();
        let service = Arc::new(InferenceService::new(evaluator, 64, replicas));
        // Drop an older version of the same file, then the least recently used network.
        cache.retain(|(k, _)| k.0 != key.0);
        if cache.len() >= PLAY_NETWORK_CACHE {
            cache.remove(0);
        }
        cache.push((key, Arc::clone(&service)));
        Ok(service)
    }
}

impl<const G: usize, const P: usize, const N: usize> ModelContext<G, P, N> {
    /// Loads checkpoint `file` into the live network, if it has the live network's architecture.
    fn load_into_live<R: AlphaZeroGame<G, P, N>>(&self, file: &str) -> Result<(), String> {
        let path = checkpoints::path(file);
        let config = checkpoints::net_config::<R, G, P, N>(&path)?;
        if config != self.net_config {
            return Err(format!(
                "{file} holds a {} network, but the live network is a {} network",
                checkpoints::preset_name::<R, G, P, N>(&config),
                checkpoints::preset_name::<R, G, P, N>(&self.net_config)
            ));
        }
        self.live_service.pause_inference();
        let loaded = self.live_service.evaluator().load_safetensors(&path);
        self.live_service.resume_inference();
        loaded.map_err(|e| e.to_string())
    }

    /// Network for an arena engine: the live model, untrained weights of the live architecture,
    /// or a checkpoint file (any architecture; it is recorded in the file).
    fn resolve_evaluator<R: AlphaZeroGame<G, P, N>>(&self, model_id: &str) -> Result<NetEvaluator<G, P, N>, String> {
        let id = model_id.trim();
        if id.is_empty() || id == "live" {
            return Ok(self.live_service.evaluator().clone());
        }
        if id == "untrained" {
            return NetEvaluator::standalone(&self.net_config, &self.devices, None::<&str>)
                .map_err(|e| e.to_string());
        }
        let safe_name = checkpoints::sanitize_name(id)
            .ok_or_else(|| format!("Invalid checkpoint filename: {id}"))?;
        let path = checkpoints::path(&safe_name);
        let config = checkpoints::net_config::<R, G, P, N>(&path)?;
        NetEvaluator::standalone(&config, &self.devices, Some(&path))
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
    eval_opponents: Vec<OpponentSpec>,
    history: Vec<IterationTelemetry>,
    matches: Vec<TttMatchRecord>,
    calibration: Option<CalibrationReport>,
    elo: EloState,
}

/// Elo anchors of one game and the state of their calibration.
#[derive(Default)]
struct EloState {
    pool: Option<EloPool>,
    /// Requested calibration `(anchors, games per pair)`, not yet picked up by the training loop.
    request: Option<(Vec<OpponentSpec>, usize)>,
    running: bool,
    /// What the calibration is doing, or how it ended.
    status: String,
}

impl EloState {
    fn to_json(&self, game: LabGame) -> serde_json::Value {
        let anchors: Vec<serde_json::Value> = self.pool.iter().flat_map(|p| &p.anchors).map(|a| serde_json::json!({
            "label": a.opponent.label(),
            "opponent": json_value(&a.opponent),
            "elo": a.rating.elo,
            "stderr": a.rating.stderr,
        })).collect();
        serde_json::json!({
            "anchors": anchors,
            "games_per_pair": self.pool.as_ref().map(|p| p.games_per_pair),
            "calibrated_unix": self.pool.as_ref().map(|p| p.calibrated_unix),
            "default_anchors": json_value(&default_elo_anchors(game)),
            "busy": self.request.is_some() || self.running,
            "status": self.status,
        })
    }
}

fn elo_pool_path(game: LabGame) -> std::path::PathBuf {
    checkpoints::path(&format!("{}elo_pool.json", game.prefix()))
}

impl GameLabState {
    fn new(game: LabGame, total_iterations: usize, knobs: LiveKnobs, auto_save_every: usize) -> Self {
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
            eval_opponents: default_eval_opponents(game),
            history: Vec::with_capacity(total_iterations.max(64)),
            matches: Vec::new(),
            calibration: None,
            elo: EloState::default(),
        }
    }
}

struct HubInner {
    device_name: String,
    system: SystemSettings,
    /// `system` was changed from the Browser Lab and not yet applied by the training loop.
    system_pending: bool,
    active_game: LabGame,
    ttt_state: GameLabState,
    uttt_state: GameLabState,
    rubiks_state: GameLabState,
    ttt_context: Option<ModelContext<27, 9, 2>>,
    uttt_context: Option<ModelContext<{ ultimate_ttt::INPUT_SIZE }, 81, 2>>,
    rubiks_context: Option<ModelContext<{ rubiks_cube::INPUT_SIZE }, { rubiks_cube::POLICY_SIZE }, 1>>,
    /// The cube's own settings (discount, round rules), applied on the next iteration.
    rubiks_settings: RubiksSettings,
    experiments: ExperimentStore,
    /// CPU/GPU utilization, sampled while the HTTP server runs.
    utilization: Option<Arc<UtilizationMonitor>>,
}

impl HubInner {
    fn state_for(&self, game: LabGame) -> &GameLabState {
        match game {
            LabGame::Ttt => &self.ttt_state,
            LabGame::Uttt => &self.uttt_state,
            LabGame::Rubiks => &self.rubiks_state,
        }
    }

    fn state_for_mut(&mut self, game: LabGame) -> &mut GameLabState {
        match game {
            LabGame::Ttt => &mut self.ttt_state,
            LabGame::Uttt => &mut self.uttt_state,
            LabGame::Rubiks => &mut self.rubiks_state,
        }
    }

    fn active_state(&self) -> &GameLabState {
        self.state_for(self.active_game)
    }

    fn has_live_model(&self, game: LabGame) -> bool {
        match game {
            LabGame::Ttt => self.ttt_context.is_some(),
            LabGame::Uttt => self.uttt_context.is_some(),
            LabGame::Rubiks => self.rubiks_context.is_some(),
        }
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
        let mut rubiks_knobs = LiveKnobs::default_rubiks();
        rubiks_knobs.paused = knobs.paused;
        Self {
            inner: Arc::new(Mutex::new(HubInner {
                device_name: device_name.into(),
                system: SystemSettings {
                    backend: Backend::Candle,
                    device: DeviceChoice::Gpu,
                    infer_batch: 512,
                    infer_workers: 1,
                },
                system_pending: false,
                active_game: LabGame::Uttt,
                // Auto-save is off by default (0); enable it in the Browser Lab when needed.
                ttt_state: GameLabState::new(LabGame::Ttt, total_iterations, knobs, 0),
                uttt_state: GameLabState::new(LabGame::Uttt, 30, uttt_knobs, 0),
                rubiks_state: GameLabState::new(LabGame::Rubiks, 30, rubiks_knobs, 0),
                ttt_context: None,
                uttt_context: None,
                rubiks_context: None,
                rubiks_settings: RubiksSettings::default(),
                experiments: ExperimentStore::in_memory(),
                utilization: None,
            })),
        }
    }

    /// Publishes the compute settings in effect.
    pub fn set_system_info(&self, system: SystemSettings) {
        self.inner.lock().expect("telemetry mutex poisoned").system = system;
    }

    pub fn set_device_label(&self, label: impl Into<String>) {
        self.inner.lock().expect("telemetry mutex poisoned").device_name = label.into();
    }

    pub fn system_settings(&self) -> SystemSettings {
        self.inner.lock().expect("telemetry mutex poisoned").system
    }

    /// Returns the new [`SystemSettings`] once after they were changed in the Browser Lab.
    pub fn take_system_change(&self) -> Option<SystemSettings> {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        if guard.system_pending {
            guard.system_pending = false;
            Some(guard.system)
        } else {
            None
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


    /// Attaches the live Tic-Tac-Toe (`G=27, P=9, N=2`) [`InferenceService`] and [`AlphaZeroNetConfig`].
    pub fn attach_ttt_service(
        &self,
        live_service: Arc<InferenceService<NetEvaluator<27, 9, 2>, 27, 9, 2>>,
        net_config: AlphaZeroNetConfig,
        devices: Vec<NnDevice>,
    ) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.ttt_context = Some(ModelContext::new(live_service, net_config, devices));
    }

    /// Attaches the live Ultimate Tic-Tac-Toe (`G=324, P=81, N=2`) [`InferenceService`] and [`AlphaZeroNetConfig`].
    pub fn attach_uttt_service(
        &self,
        live_service: Arc<InferenceService<NetEvaluator<{ ultimate_ttt::INPUT_SIZE }, 81, 2>, { ultimate_ttt::INPUT_SIZE }, 81, 2>>,
        net_config: AlphaZeroNetConfig,
        devices: Vec<NnDevice>,
    ) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.uttt_context = Some(ModelContext::new(live_service, net_config, devices));
    }

    /// Attaches the live Rubik's cube (`G=324, P=12, N=1`) [`InferenceService`] and [`AlphaZeroNetConfig`].
    pub fn attach_rubiks_service(
        &self,
        live_service: SharedService<{ rubiks_cube::INPUT_SIZE }, { rubiks_cube::POLICY_SIZE }, 1>,
        net_config: AlphaZeroNetConfig,
        devices: Vec<NnDevice>,
    ) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.rubiks_context = Some(ModelContext::new(live_service, net_config, devices));
    }

    /// The cube's own settings in effect.
    pub fn rubiks_settings(&self) -> RubiksSettings {
        self.inner.lock().expect("telemetry mutex poisoned").rubiks_settings
    }

    pub fn set_phase(&self, iteration: usize, phase: impl Into<String>) {
        let game = self.active_game();
        self.set_phase_for(game, iteration, phase);
    }

    /// What the lab is doing right now, for the utilization samples.
    fn current_activity(&self) -> &'static str {
        let guard = self.inner.lock().expect("telemetry mutex poisoned");
        if guard.experiments.is_running() {
            return "experiment";
        }
        if LabGame::ALL.iter().any(|&g| guard.state_for(g).elo.running) {
            return "evaluation";
        }
        let phase = guard.state_for(guard.active_game).phase.to_ascii_lowercase();
        if phase.contains("self-play") {
            "self-play"
        } else if phase.contains("sgd") {
            "training"
        } else if phase.contains("evaluation") {
            "evaluation"
        } else {
            "idle"
        }
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

    /// The evaluation opponents configured for `game`'s live-training arena.
    pub fn eval_opponents_for(&self, game: LabGame) -> Vec<OpponentSpec> {
        self.inner.lock().expect("telemetry mutex poisoned").state_for(game).eval_opponents.clone()
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
        for game in LabGame::ALL {
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

        for game in LabGame::ALL {
            let elo = &mut guard.state_for_mut(game).elo;
            if let Some((anchors, games_per_pair)) = elo.request.take() {
                elo.status = "Starting calibration...".into();
                elo.running = true;
                return TrainingCommand::CalibrateElo { game, anchors, games_per_pair };
            }
        }

        if let Some((id, spec)) = guard.experiments.next_queued() {
            guard.experiments.start(id, Vec::new(), String::new());
            return TrainingCommand::RunExperiment { id, spec: Box::new(spec) };
        }
        TrainingCommand::Idle
    }

    /// Loads the calibrated Elo anchors of both games from the checkpoint folder.
    pub fn enable_elo_pools(&self) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        for game in LabGame::ALL {
            guard.state_for_mut(game).elo.pool = EloPool::load(&elo_pool_path(game));
        }
    }

    /// The calibrated Elo anchors of `game`, if any.
    pub fn elo_pool(&self, game: LabGame) -> Option<EloPool> {
        self.inner.lock().expect("telemetry mutex poisoned").state_for(game).elo.pool.clone()
    }

    pub fn set_elo_status(&self, game: LabGame, status: impl Into<String>) {
        self.inner.lock().expect("telemetry mutex poisoned").state_for_mut(game).elo.status = status.into();
    }

    /// Stores (and saves) a finished calibration, or reports why it failed.
    pub fn finish_elo_calibration(&self, game: LabGame, result: Result<EloPool, String>) {
        let status = match &result {
            Ok(pool) => match pool.save(&elo_pool_path(game)) {
                Ok(()) => format!("Calibrated {} anchors", pool.anchors.len()),
                Err(e) => format!("Calibrated, but saving failed: {e}"),
            },
            Err(e) => format!("Calibration failed: {e}"),
        };
        println!("⚖️  [Browser Lab] {} Elo: {status}", game.title());
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        let elo = &mut guard.state_for_mut(game).elo;
        elo.status = status;
        elo.running = false;
        if let Ok(pool) = result {
            elo.pool = Some(pool);
        }
    }

    /// Loads (and from now on persists) experiment results at `path`.
    pub fn enable_experiment_store(&self, path: impl Into<std::path::PathBuf>) {
        self.inner.lock().expect("telemetry mutex poisoned").experiments = ExperimentStore::load(path);
    }

    /// Records the evaluation opponents and device of a started experiment.
    pub fn experiment_started(&self, id: u64, opponents: Vec<String>, device_label: String) {
        self.inner.lock().expect("telemetry mutex poisoned").experiments.start(id, opponents, device_label);
    }

    pub fn experiment_phase(&self, id: u64, phase: impl Into<String>) {
        self.inner.lock().expect("telemetry mutex poisoned").experiments.set_phase(id, phase.into());
    }

    pub fn experiment_point(&self, id: u64, point: ExperimentPoint) {
        self.inner.lock().expect("telemetry mutex poisoned").experiments.push_point(id, point);
    }

    pub fn experiment_finished(&self, id: u64, status: RunStatus) {
        self.inner.lock().expect("telemetry mutex poisoned").experiments.finish(id, status);
    }

    /// True when the Browser Lab asked to stop experiment `id`, or live training was started.
    pub fn experiment_should_stop(&self, id: u64) -> bool {
        self.inner.lock().expect("telemetry mutex poisoned").experiments.should_stop(id)
    }

    pub fn discover_models(&self) -> Vec<ModelInfo> {
        let guard = self.inner.lock().expect("telemetry mutex poisoned");
        let game = guard.active_game;
        let iter = guard.state_for(game).current_iteration;
        discover_game_models(game, guard.has_live_model(game), iter)
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
            LabGame::Rubiks => return Err("timing calibration is for the two-player games".into()),
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
        // Each opening is played once per seat, so the game count is even.
        let games = num_games.clamp(2, 5000).div_ceil(2) * 2;
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
                        let eval = ctx.resolve_evaluator::<TicTacToe>(&cfg_a.model)?;
                        let n_rep = eval.num_replicas();
                        svc_a = Some(Arc::new(InferenceService::new(eval, batch_cap, n_rep)));
                    }
                    if b_is_az {
                        let eval = ctx.resolve_evaluator::<TicTacToe>(&cfg_b.model)?;
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

                let openings = TicTacToe::eval_openings(games / 2);
                let start = Instant::now();
                let outcome = run_multi_match_with_openings(&openings, [&factory_a, &factory_b], workers, seed);
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
                        let eval = ctx.resolve_evaluator::<UltimateTicTacToe>(&cfg_a.model)?;
                        let n_rep = eval.num_replicas();
                        svc_a = Some(Arc::new(InferenceService::new(eval, batch_cap, n_rep)));
                    }
                    if b_is_az {
                        let eval = ctx.resolve_evaluator::<UltimateTicTacToe>(&cfg_b.model)?;
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

                let openings = UltimateTicTacToe::eval_openings(games / 2);
                let start = Instant::now();
                let outcome = run_multi_match_with_openings(&openings, [&factory_a, &factory_b], workers, seed);
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
            LabGame::Rubiks => return Err("matches are for the two-player games".into()),
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
        let models = discover_game_models(game, guard.has_live_model(game), st.current_iteration);
        let models_json = models
            .iter()
            .map(ModelInfo::to_json)
            .collect::<Vec<_>>()
            .join(",");
        let history_json = st
            .history
            .iter()
            .map(|h| h.to_json(st.elo.pool.as_ref()))
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
                "\"system\":{},",
                "\"active_game\":\"{}\",",
                "\"active_game_title\":\"{}\",",
                "\"phase\":\"{}\",",
                "\"current_iteration\":{},",
                "\"total_iterations\":{},",
                "\"auto_save_every\":{},",
                "\"knobs\":{},",
                "\"eval_opponents\":{},",
                "\"models\":[{}],",
                "\"utilization\":{},",
                "\"calibration\":{},",
                "\"elo\":{},",
                "\"game_settings\":{},",
                "\"history\":[{}],",
                "\"matches\":[{}]",
                "}}"
            ),
            escape_json(&guard.device_name),
            guard.system.to_json(guard.system_pending),
            game.as_str(),
            escape_json(game.title()),
            escape_json(&st.phase),
            st.current_iteration,
            st.total_iterations,
            st.auto_save_every,
            st.knobs.to_json(),
            serde_json::to_string(&st.eval_opponents).unwrap_or_else(|_| "[]".into()),
            models_json,
            guard.utilization.as_ref().map_or_else(|| "null".to_string(), |u| u.to_json().to_string()),
            cal_json,
            st.elo.to_json(game),
            if game == LabGame::Rubiks { json_value(&guard.rubiks_settings) } else { serde_json::Value::Null },
            history_json,
            matches_json,
        )
    }

    /// Answers `POST /api/play` (see [`crate::play`]).
    fn play_json(&self, body: &str) -> Result<String, String> {
        let bad = |e: serde_json::Error| format!("bad play request: {e}");
        let game = serde_json::from_str::<serde_json::Value>(body).map_err(bad)?["game"].as_str().map(LabGame::from_str_lossy);
        let (ttt, uttt, rubiks, settings) = {
            let guard = self.inner.lock().expect("telemetry mutex poisoned");
            (guard.ttt_context.clone(), guard.uttt_context.clone(), guard.rubiks_context.clone(), guard.rubiks_settings)
        };
        let missing = || "the network for this game is not loaded yet".to_string();
        if game == Some(LabGame::Rubiks) {
            let req: CubePlayRequest = serde_json::from_str(body).map_err(bad)?;
            let ctx = rubiks.ok_or_else(missing)?;
            let response = play_cube::respond(&req, settings.discount, |ckpt| ctx.play_network::<RubiksCube>(ckpt))?;
            return serde_json::to_string(&response).map_err(|e| e.to_string());
        }
        let req: PlayRequest = serde_json::from_str(body).map_err(bad)?;
        let response = match game.unwrap_or(LabGame::Ttt) {
            LabGame::Uttt => {
                let ctx = uttt.ok_or_else(missing)?;
                play::respond::<UltimateTicTacToe, { ultimate_ttt::INPUT_SIZE }, 81>(&req, |ckpt| {
                    ctx.play_network::<UltimateTicTacToe>(ckpt)
                })?
            }
            _ => {
                let ctx = ttt.ok_or_else(missing)?;
                play::respond::<TicTacToe, 27, 9>(&req, |ckpt| ctx.play_network::<TicTacToe>(ckpt))?
            }
        };
        serde_json::to_string(&response).map_err(|e| e.to_string())
    }

    /// Experiments, plus per-game network presets and hyperparameter defaults for the run form.
    pub fn experiments_json(&self) -> String {
        let guard = self.inner.lock().expect("telemetry mutex poisoned");
        let presets = |names: Vec<(&'static str, AlphaZeroNetConfig)>| names.into_iter().map(|(n, _)| n).collect::<Vec<_>>();
        let games = serde_json::json!({
            "ttt": {
                "title": LabGame::Ttt.title(),
                "presets": presets(<TicTacToe as AlphaZeroGame<27, 9, 2>>::net_presets()),
                "defaults": json_value(&<TicTacToe as AlphaZeroGame<27, 9, 2>>::hyperparams()),
                "live": json_value(&guard.ttt_state.knobs.hp),
                "default_opponents": default_eval_opponents(LabGame::Ttt),
                "checkpoints": disk_checkpoints(LabGame::Ttt),
            },
            "uttt": {
                "title": LabGame::Uttt.title(),
                "presets": presets(<UltimateTicTacToe as AlphaZeroGame<{ ultimate_ttt::INPUT_SIZE }, { ultimate_ttt::POLICY_SIZE }, 2>>::net_presets()),
                "defaults": json_value(&<UltimateTicTacToe as AlphaZeroGame<{ ultimate_ttt::INPUT_SIZE }, { ultimate_ttt::POLICY_SIZE }, 2>>::hyperparams()),
                "live": json_value(&guard.uttt_state.knobs.hp),
                "default_opponents": default_eval_opponents(LabGame::Uttt),
                "checkpoints": disk_checkpoints(LabGame::Uttt),
            },
            "rubiks": {
                "title": LabGame::Rubiks.title(),
                "puzzle": true,
                "presets": presets(<RubiksCube as AlphaZeroGame<{ rubiks_cube::INPUT_SIZE }, { rubiks_cube::POLICY_SIZE }, 1>>::net_presets()),
                "defaults": json_value(&<RubiksCube as AlphaZeroGame<{ rubiks_cube::INPUT_SIZE }, { rubiks_cube::POLICY_SIZE }, 1>>::hyperparams()),
                "live": json_value(&guard.rubiks_state.knobs.hp),
                "settings": json_value(&guard.rubiks_settings),
                "default_settings": json_value(&RubiksSettings::default()),
                "default_opponents": [],
                "checkpoints": disk_checkpoints(LabGame::Rubiks),
            },
        });
        let backends: Vec<serde_json::Value> = Backend::available()
            .into_iter()
            .map(|b| serde_json::json!({ "name": b.as_str(), "gpu": b.gpu_name() }))
            .collect();
        let mut store: serde_json::Value = serde_json::from_str(&guard.experiments.to_json()).unwrap_or_default();
        // Ratings come from the current anchors, so a recalibration updates every run's curve.
        for run in store["runs"].as_array_mut().into_iter().flatten() {
            let Ok(spec) = serde_json::from_value::<ExperimentSpec>(run["spec"].clone()) else { continue };
            let game = LabGame::from_str_lossy(&spec.game);
            let Some(pool) = guard.state_for(game).elo.pool.as_ref() else { continue };
            let opponents = if spec.opponents.is_empty() { default_eval_opponents(game) } else { spec.opponents };
            for point in run["points"].as_array_mut().into_iter().flatten() {
                let wdl: Vec<[usize; 3]> = serde_json::from_value(point["wdl"].clone()).unwrap_or_default();
                point["elo"] = serde_json::to_value(pool.estimate(opponents.iter().zip(wdl))).unwrap_or_default();
            }
        }
        serde_json::json!({
            "store": store,
            "games": games,
            "backends": backends,
            "system": { "infer_batch": guard.system.infer_batch, "infer_workers": guard.system.infer_workers },
            "active_game": guard.active_game.as_str(),
            "live_training": !guard.active_state().knobs.paused,
        })
        .to_string()
    }

    /// Spawns a background HTTP server thread listening on `addr` (e.g. `"127.0.0.1:8080"` or
    /// `"0.0.0.0:8080"` for RunPod).
    pub fn spawn_http_server(&self, addr: &str) -> std::io::Result<String> {
        let listener = TcpListener::bind(addr)?;
        let bound_addr = listener.local_addr()?.to_string();
        let hub = self.clone();

        let activity_hub = self.clone();
        let monitor = UtilizationMonitor::start(move || activity_hub.current_activity());
        self.inner.lock().expect("telemetry mutex poisoned").utilization = Some(monitor);

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
    svc_a: Option<&InferenceService<NetEvaluator<G, P, 2>, G, P, 2>>,
    svc_b: Option<&InferenceService<NetEvaluator<G, P, 2>, G, P, 2>>,
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
        let eval = c.resolve_evaluator::<TicTacToe>("live")?;
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

fn calibrate_uttt(ctx: Option<&ModelContext<{ ultimate_ttt::INPUT_SIZE }, 81, 2>>) -> Result<CalibrationReport, String> {
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
        let eval = c.resolve_evaluator::<UltimateTicTacToe>("live")?;
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
    service: Option<&'a InferenceService<NetEvaluator<27, 9, 2>, 27, 9, 2>>,
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
    service: Option<&'a InferenceService<NetEvaluator<{ ultimate_ttt::INPUT_SIZE }, 81, 2>, { ultimate_ttt::INPUT_SIZE }, 81, 2>>,
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

/// File names of `game`'s checkpoints in the checkpoint folder.
fn disk_checkpoints(game: LabGame) -> Vec<String> {
    discover_game_models(game, false, 0).into_iter().filter(|m| m.kind == "disk").map(|m| m.id).collect()
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

    for (name, size_bytes) in checkpoints::list() {
        if LabGame::of_checkpoint(&name) == game {
            list.push(ModelInfo {
                id: name,
                kind: "disk",
                game: g_str,
                size_bytes,
                description: format!("Saved {} checkpoint on disk", game.title()),
            });
        }
    }
    list
}

fn ensure_game_prefixed_filename(game: LabGame, raw: &str) -> Option<String> {
    let base = checkpoints::sanitize_name(raw)?;
    if LabGame::ALL.iter().any(|g| base.starts_with(g.prefix())) {
        Some(base)
    } else {
        Some(format!("{}{base}", game.prefix()))
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
            &DASHBOARD_PAGE,
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
            apply_game_settings(&mut guard, body);
            let st = guard.state_for_mut(active_game);
            apply_knob_updates(&mut st.knobs, body);
            apply_opponent_updates(&mut st.eval_opponents, body);
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

    if raw_path.starts_with("/api/experiments") {
        if method == "POST" {
            let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
            let id = req["id"].as_u64().unwrap_or(0);
            let mut guard = hub.inner.lock().expect("telemetry mutex poisoned");
            let store = &mut guard.experiments;
            match req["action"].as_str().unwrap_or("") {
                "add" => {
                    if let Ok(spec) = serde_json::from_value::<ExperimentSpec>(req["spec"].clone()) {
                        let new_id = store.add(spec);
                        println!("🧪 [Browser Lab] Queued experiment #{new_id}");
                    }
                }
                "delete" => store.remove(id),
                "requeue" => store.requeue(id),
                "stop" => store.request_stop(id),
                "run_queue" => store.set_queue_running(true),
                "pause_queue" => store.set_queue_running(false),
                _ => {}
            }
        }
        let json = hub.experiments_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &json);
    }

    if method == "POST" && raw_path.starts_with("/api/elo") {
        let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
        let game = req["game"].as_str().map_or_else(|| hub.active_game(), LabGame::from_str_lossy);
        let anchors = opponents::parse_list(&req["anchors"]).unwrap_or_default();
        let games_per_pair = req["games_per_pair"].as_u64().unwrap_or(48).clamp(2, 1000) as usize;
        if anchors.len() < 2 {
            return write_http_response(&mut stream, "400 Bad Request", "application/json", r#"{"error":"need at least two anchors"}"#);
        }
        {
            let mut guard = hub.inner.lock().expect("telemetry mutex poisoned");
            let elo = &mut guard.state_for_mut(game).elo;
            elo.status = "Queued — starts while live training is paused".into();
            elo.request = Some((anchors, games_per_pair));
        }
        return write_http_response(&mut stream, "200 OK", "application/json", &hub.to_json());
    }

    if method == "POST" && raw_path.starts_with("/api/system") {
        {
            let mut guard = hub.inner.lock().expect("telemetry mutex poisoned");
            let mut sys = guard.system;
            apply_system_updates(&mut sys, body);
            if sys != guard.system {
                println!(
                    "🖥️  [Browser Lab] System settings changed: backend={}, device={}, infer_batch={}, infer_workers={} (applied before the next iteration)",
                    sys.backend.as_str(),
                    sys.device.as_str(),
                    sys.infer_batch,
                    sys.infer_workers
                );
                guard.system = sys;
                guard.system_pending = true;
            }
        }
        let updated = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &updated);
    }

    if method == "POST" && raw_path.starts_with("/api/knobs") {
        {
            let mut guard = hub.inner.lock().expect("telemetry mutex poisoned");
            let active_game = guard.active_game;
            apply_game_settings(&mut guard, body);
            let st = guard.state_for_mut(active_game);
            let was_paused = st.knobs.paused;
            apply_knob_updates(&mut st.knobs, body);
            apply_opponent_updates(&mut st.eval_opponents, body);
            if was_paused && !st.knobs.paused && st.current_iteration >= st.total_iterations {
                st.total_iterations = st.current_iteration + 15;
            }
            if !was_paused && st.knobs.paused {
                st.step_requests = 0;
                st.pause_after_iteration = false;
                st.phase = format!("Paused (Iter #{})", st.current_iteration);
            }
            println!(
                "🎛️  [Browser Lab] Updated {} knobs: {}",
                active_game.title(),
                st.knobs.to_json()
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

        let (ttt_ctx, uttt_ctx, rubiks_ctx, cur_iter) = {
            let guard = hub.inner.lock().expect("telemetry mutex poisoned");
            (
                guard.ttt_context.clone(),
                guard.uttt_context.clone(),
                guard.rubiks_context.clone(),
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
                        ctx.live_service.evaluator().save_safetensors(checkpoints::path(&filename), &ctx.net_config)
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
                        ctx.live_service.evaluator().save_safetensors(checkpoints::path(&filename), &ctx.net_config)
                    }
                    LabGame::Rubiks => {
                        let Some(ctx) = rubiks_ctx else {
                            return write_http_response(
                                &mut stream,
                                "400 Bad Request",
                                "application/json",
                                "{\"error\":\"No live cube model attached\"}",
                            );
                        };
                        ctx.live_service.evaluator().save_safetensors(checkpoints::path(&filename), &ctx.net_config)
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
                let Some(filename) = checkpoints::sanitize_name(&raw_target_name) else {
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
                        ctx.load_into_live::<TicTacToe>(&filename)
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
                        ctx.load_into_live::<UltimateTicTacToe>(&filename)
                    }
                    LabGame::Rubiks => {
                        let Some(ctx) = rubiks_ctx else {
                            return write_http_response(
                                &mut stream,
                                "400 Bad Request",
                                "application/json",
                                "{\"error\":\"No live cube model attached\"}",
                            );
                        };
                        ctx.load_into_live::<RubiksCube>(&filename)
                    }
                };
                if let Err(e) = load_res {
                    let msg = format!("{{\"error\":\"{}\"}}", escape_json(&e));
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
                if let Some(filename) = checkpoints::sanitize_name(&raw_target_name) {
                    let _ = std::fs::remove_file(checkpoints::path(&filename));
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

    if method == "POST" && raw_path.starts_with("/api/play") {
        return match hub.play_json(body) {
            Ok(json) => write_http_response(&mut stream, "200 OK", "application/json", &json),
            Err(err) => {
                let msg = serde_json::json!({ "error": err }).to_string();
                write_http_response(&mut stream, "400 Bad Request", "application/json", &msg)
            }
        };
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
    let num = |key: &str| extract_json_f64(json, key);
    if let Some(v) = extract_json_bool(json, "paused") {
        knobs.paused = v;
    }
    if let Some(v) = num("games_per_iteration") {
        knobs.games_per_iteration = (v as usize).clamp(4, 4096);
    }
    if let Some(v) = num("simulations_per_move") {
        knobs.simulations_per_move = (v as usize).clamp(4, 4000);
    }
    if let Some(v) = num("self_play_threads") {
        knobs.self_play_threads = (v as usize).min(256);
    }
    if let Some(v) = num("opening_moves") {
        knobs.opening_moves = (v as usize).min(81);
    }
    if let Some(v) = num("dirichlet_alpha") {
        knobs.dirichlet_alpha = (v as f32).clamp(0.01, 10.0);
    }
    if let Some(v) = num("dirichlet_epsilon") {
        knobs.dirichlet_epsilon = (v as f32).clamp(0.0, 0.9);
    }
    if let Some(v) = num("c_init") {
        knobs.c_init = (v as f32).clamp(0.1, 10.0);
    }
    if let Some(v) = num("c_base") {
        knobs.c_base = (v as f32).clamp(1.0, 1e7);
    }
    if let Some(fpu) = serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| serde_json::from_value::<Fpu>(v["fpu"].clone()).ok())
    {
        knobs.fpu = fpu.sanitized();
    }
    if let Some(v) = num("value_q_weight") {
        knobs.value_q_weight = (v as f32).clamp(0.0, 1.0);
    }
    if let Some(schedule) = extract_json_str(json, "value_q_schedule")
        .and_then(|v| serde_json::from_value::<ValueQSchedule>(serde_json::Value::String(v)).ok())
    {
        knobs.value_q_schedule = schedule;
    }
    if let Some(v) = num("train_steps") {
        knobs.train_steps = (v as usize).clamp(1, 10_000);
    }
    if let Some(v) = num("batch_size") {
        knobs.batch_size = (v as usize).clamp(8, 4096);
    }
    if let Some(v) = num("learning_rate") {
        knobs.learning_rate = v.clamp(1e-6, 1e-1);
    }
    if let Some(v) = num("weight_decay") {
        knobs.weight_decay = v.clamp(0.0, 0.1);
    }
    if let Some(v) = num("buffer_capacity") {
        knobs.buffer_capacity = (v as usize).clamp(1_024, 5_000_000);
    }
    if let Some(v) = num("eval_games") {
        knobs.eval_games = (v as usize).clamp(2, 2_000);
    }
    if let Some(v) = num("eval_sims") {
        knobs.eval_sims = (v as usize).min(10_000);
    }
    if let Some(v) = num("eval_threads") {
        knobs.eval_threads = (v as usize).min(256);
    }
}

/// Replaces the evaluation opponents if the request body carries a non-empty `eval_opponents` list.
/// Applies `game_settings` from a knobs request to the active game's own settings (the cube's
/// [`RubiksSettings`]; fields left out keep their value).
fn apply_game_settings(hub: &mut HubInner, body: &str) {
    if hub.active_game != LabGame::Rubiks {
        return;
    }
    let Ok(req) = serde_json::from_str::<serde_json::Value>(body) else { return };
    let Some(update) = req["game_settings"].as_object() else { return };
    let mut merged = json_value(&hub.rubiks_settings);
    for (k, v) in update {
        merged[k] = v.clone();
    }
    if let Ok(settings) = serde_json::from_value::<RubiksSettings>(merged) {
        hub.rubiks_settings = settings.sanitized();
    }
}

fn apply_opponent_updates(opponents: &mut Vec<OpponentSpec>, body: &str) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(body) else { return };
    if let Some(list) = opponents::parse_list(&v["eval_opponents"]).filter(|l| !l.is_empty()) {
        *opponents = list;
    }
}

fn apply_system_updates(sys: &mut SystemSettings, json: &str) {
    if let Some(b) = extract_json_str(json, "backend").and_then(|v| Backend::parse(&v)).filter(|b| b.is_available()) {
        sys.backend = b;
    }
    if let Some(v) = extract_json_str(json, "device") {
        sys.device = if v.eq_ignore_ascii_case("cpu") { DeviceChoice::Cpu } else { DeviceChoice::Gpu };
    }
    if let Some(v) = extract_json_f64(json, "infer_batch") {
        sys.infer_batch = (v as usize).clamp(1, 4096);
    }
    if let Some(v) = extract_json_f64(json, "infer_workers") {
        sys.infer_workers = (v as usize).clamp(1, 64);
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

/// The dashboard with the Experiments tab (`lab_experiments.html`) and the Play tab
/// (`lab_play.html`) spliced in.
static DASHBOARD_PAGE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    const EXPERIMENTS_TAB: &str = include_str!("lab_experiments.html");
    const PLAY_TAB: &str = include_str!("lab_play.html");
    const CUBE_VIEW: &str = include_str!("lab_cube.html");
    DASHBOARD_HTML
        .replacen(
            "</button>\n</div>\n\n<!-- TAB 1",
            "</button>\n  <button class=\"tab-btn\" id=\"tab-btn-experiments\" onclick=\"switchTab('experiments')\">🧪 Experiments</button>\n</div>\n\n<!-- TAB 1",
            1,
        )
        .replacen("<script>", &format!("{EXPERIMENTS_TAB}\n{}\n<script>", PLAY_TAB.replacen("<!-- CUBE VIEW -->", CUBE_VIEW, 1)), 1)
});

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
  .knob-section {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 12px 0 6px;
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
  /* The Rubik's cube is a one-player puzzle: no opponents, Elo or arena, but round-game charts. */
  body.puzzle .two-player-only, body:not(.puzzle) .puzzle-only { display: none; }
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
    <h1>🦀 Pure-Rust AlphaZero Lab, Calibrated Arena & Play</h1>
    <p>Interactive Training Control • Tic-Tac-Toe & Ultimate Tic-Tac-Toe • Equal-Compute Match Calibration</p>
  </div>
  <div class="game-switcher">
    <button id="game-btn-uttt" class="game-btn active" onclick="switchActiveGame('uttt')">⚔️ Ultimate Tic-Tac-Toe (9x9)</button>
    <button id="game-btn-ttt" class="game-btn" onclick="switchActiveGame('ttt')">🎯 Tic-Tac-Toe (3x3)</button>
    <button id="game-btn-rubiks" class="game-btn" onclick="switchActiveGame('rubiks')">🧊 Rubik's Cube</button>
  </div>
  <div class="badges">
    <span class="badge" id="device-badge">Device: ...</span>
    <span class="badge paused" id="phase-badge">Phase: Idle</span>
    <span class="badge" id="iter-badge">Iter: 0 / 0</span>
  </div>
</header>

<div class="tabs">
  <button class="tab-btn active" id="tab-btn-lab" onclick="switchTab('lab')">📊 Live Training & System Lab</button>
  <button class="tab-btn two-player-only" id="tab-btn-arena" onclick="switchTab('arena')">⚔️ Multi-Match Arena & Compute Calibrator</button>
  <button class="tab-btn" id="tab-btn-play" onclick="switchTab('play')">🎮 Play</button>
</div>

<!-- TAB 1: TRAINING & SYSTEM LAB -->
<div id="tab-lab">
  <!-- Interactive Training Lifecycle Bar -->
  <div class="train-control-bar">
    <div class="train-btn-group">
      <button class="btn green" id="btn-toggle-train" style="min-width:150px;" onclick="toggleTraining()">▶ Start Training</button>
      <button class="btn secondary" id="btn-step-train" onclick="sendTrainControl('step')">⏭ Step (+1 Iter)</button>
      <button class="btn secondary" onclick="snapshotLiveModel()">📸 Save Snapshot Now</button>
      <button class="btn danger" onclick="confirmResetTraining()">🔄 Reset to Untrained</button>
    </div>
    <div class="train-btn-group" style="font-size:0.8rem; color:var(--muted);">
      <label>Target Iters:
        <input type="number" id="ctrl-total-iters" value="15" min="1" max="5000" style="width:65px; background:var(--bg); color:var(--text); border:1px solid var(--border); padding:4px 6px; border-radius:4px;" onchange="updateTrainTargets()">
      </label>
      <label>Auto-Save Every:
        <input type="number" id="ctrl-autosave" value="0" min="0" max="500" style="width:55px; background:var(--bg); color:var(--text); border:1px solid var(--border); padding:4px 6px; border-radius:4px;" onchange="updateTrainTargets()"> iters (0=off)
      </label>
    </div>
  </div>

  <div class="grid puzzle-only">
    <div class="card">
      <h2><span>Round game score</span><span id="latest-round-score" style="font-size:0.78rem;color:var(--green)">—</span></h2>
      <p class="hint">Mean game score <i>n</i> + 1/<i>r</i> (<i>n</i>: last solved round, <i>r</i>: its moves) of the noise-free evaluation
        games (green, thin lines ± one standard error) and of the self-play games, which explore with root noise (blue). The one number
        that says whether the network gets better.</p>
      <canvas id="chart-round-score" width="600" height="205"></canvas>
      <div class="legend"><span style="color:#34d399">evaluation</span><span style="color:#38bdf8">self-play</span></div>
    </div>
    <div class="card">
      <h2><span>Solve rate per round (evaluation)</span><span id="latest-best-round" style="font-size:0.78rem;color:var(--green)">—</span></h2>
      <p class="hint">One column per iteration, one row per round (round 1 at the bottom): the share of the games reaching the round that
        solved it, from red (0 %) to green (100 %); dark: no game reached it. The table shows the latest evaluation.</p>
      <canvas id="chart-round-heat" width="600" height="205"></canvas>
      <table style="margin-top:6px"><thead><tr><th style="text-align:left">Round</th><th>reached</th><th>solved</th><th>moves (avg)</th></tr></thead>
        <tbody id="round-table"></tbody></table>
    </div>
  </div>

  <div class="grid two-player-only">
    <div class="card">
      <h2>
        <span>Strength (anchored Elo)</span>
        <span id="latest-elo" style="font-size:0.78rem;color:var(--green)">—</span>
      </h2>
      <p class="hint" id="elo-chart-hint">The rating that best explains this iteration's results against the evaluation opponents that are Elo anchors
        (see ⚖️ Elo anchors below). One scale for all opponents, iterations and experiments: +100 Elo means an expected score of 64 % against the old self.
        The thin lines mark ± one standard error.</p>
      <canvas id="chart-elo" width="600" height="205"></canvas>
    </div>
  </div>

  <!-- One win/draw/loss card per evaluation opponent (built by renderArenaCards). -->
  <div class="grid two-player-only" id="arena-cards" style="margin-top:14px;"></div>

  <!-- One card per metric (built by buildMetricCards), so every chart gets its own y-axis scale. -->
  <div class="grid" id="metric-cards" style="margin-top:14px;"></div>

  <div class="card" style="margin-top:14px;">
    <h2>
      <span>🎛️ Live Hyperparameter Knobs (<span id="knobs-game-label">Ultimate Tic-Tac-Toe</span> — Applied on Next Iteration)</span>
      <button class="btn" onclick="pushKnobs()">Apply Knobs</button>
    </h2>
    <h3 class="knob-section">Self-Play</h3>
    <div class="knobs-grid">
      <div class="knob-item" title="Games played per iteration. Value learning is limited by the number of distinct games.">
        <label>Self-Play Games / Iter</label>
        <input type="number" id="k-games" min="4" max="4096" step="4">
      </div>
      <div class="knob-item" title="PUCT simulations per self-play move.">
        <label>MCTS Sims / Move</label>
        <input type="number" id="k-sims" min="4" max="4000" step="4">
      </div>
      <div class="knob-item" title="CPU worker threads running self-play games. 0 = all cores but two.">
        <label>Self-Play Threads (0 = auto)</label>
        <input type="number" id="k-sp-threads" min="0" max="256" step="1">
      </div>
      <div class="knob-item" title="The first N moves of each game are sampled from the visit counts (exploration); later moves play the most-visited move.">
        <label>Sampled Opening Moves</label>
        <input type="number" id="k-opening" min="0" max="81" step="1">
      </div>
      <div class="knob-item" title="Concentration of the root exploration noise (smaller = spikier noise on fewer moves).">
        <label>Root Dirichlet α</label>
        <input type="number" id="k-alpha" min="0.01" max="10" step="0.05">
      </div>
      <div class="knob-item" title="Share of the root prior replaced by Dirichlet noise during self-play.">
        <label>Root Dirichlet ε</label>
        <input type="number" id="k-eps" min="0" max="0.9" step="0.05">
      </div>
      <div class="knob-item" title="Exploration constant: higher = search tries more moves.">
        <label>PUCT c_init</label>
        <input type="number" id="k-cinit" min="0.1" max="10" step="0.05">
      </div>
      <div class="knob-item" title="Exploration grows slowly with visits beyond this many (19652 in AlphaZero).">
        <label>PUCT c_base</label>
        <input type="number" id="k-cbase" min="1" max="10000000" step="100">
      </div>
      <div class="knob-item" title="First-play urgency: the value the search assumes for a move it has not tried yet. Absolute: a fixed value (0 = assume a draw, as in AlphaZero; higher = broader search). Relative: the position's value minus reduction × √(prior of the moves already tried), as in Leela Chess Zero / KataGo (typical reduction 0.2–0.4).">
        <label>FPU (untried moves)</label>
        <div style="display:flex;gap:4px">
          <select id="k-fpu-mode" style="flex:1"><option value="absolute">Absolute</option><option value="relative">Relative −</option></select>
          <input type="number" id="k-fpu" min="-1" max="2" step="0.05" style="width:70px">
        </div>
      </div>
      <div class="knob-item" title="Value target = (1-qw)·game result + qw·search value; 0 = pure AlphaZero. Constant: every position uses qw. Linear: qw on the first move, falling to 0 on the last (the average weight is then qw/2).">
        <label>Value Target qw (0–1)</label>
        <div style="display:flex;gap:4px">
          <input type="number" id="k-qw" min="0" max="1" step="0.05" style="width:60px">
          <select id="k-qw-schedule" style="flex:1"><option value="constant">Constant</option><option value="linear_to_end">Linear</option></select>
        </div>
      </div>
    </div>
    <h3 class="knob-section">Training</h3>
    <div class="knobs-grid">
      <div class="knob-item" title="Mini-batch updates per iteration.">
        <label>SGD Steps / Iter</label>
        <input type="number" id="k-steps" min="1" max="10000" step="5">
      </div>
      <div class="knob-item" title="Positions per gradient step.">
        <label>SGD Mini-Batch Size</label>
        <input type="number" id="k-batch" min="8" max="4096" step="16">
      </div>
      <div class="knob-item" title="Optimizer step size.">
        <label>Learning Rate (AdamW)</label>
        <input type="number" id="k-lr" min="0.000001" max="0.1" step="0.0005">
      </div>
      <div class="knob-item" title="L2-style regularization strength.">
        <label>Weight Decay (AdamW)</label>
        <input type="number" id="k-wd" min="0" max="0.1" step="0.0001">
      </div>
      <div class="knob-item" title="Most recent positions kept for training. UTTT: ~50 positions per game, so 300k ≈ 6k games.">
        <label>Replay Buffer (positions)</label>
        <input type="number" id="k-buffer" min="1024" max="5000000" step="1024">
      </div>
    </div>
    <h3 class="knob-section">Arena Evaluation</h3>
    <div class="knobs-grid">
      <div class="knob-item" title="Arena games vs each baseline per iteration. Openings are played twice with colors swapped.">
        <label>Eval Games / Opponent</label>
        <input type="number" id="k-eval-games" min="2" max="2000" step="2">
      </div>
      <div class="knob-item" title="AlphaZero simulations per move in the arena.">
        <label>Eval Sims / Move (0 = self-play)</label>
        <input type="number" id="k-eval-sims" min="0" max="10000" step="8">
      </div>
      <div class="knob-item" title="Arena worker threads. 0 = all cores but two.">
        <label>Eval Threads (0 = auto)</label>
        <input type="number" id="k-eval-threads" min="0" max="256" step="1">
      </div>
    </div>
  </div>

  <div class="card puzzle-only" style="margin-top:14px;">
    <h2>
      <span>🧊 Round game settings (applied on the next iteration)</span>
      <button class="btn" onclick="pushGameSettings()">Apply Settings</button>
    </h2>
    <p class="hint">Round <i>n</i> is a cube scrambled with <i>n</i> quarter turns, solvable within moves-per-turn × <i>n</i> + extra moves
      (at most the maximum); solving it leads to round <i>n</i> + 1, failing ends the game. The discount γ makes the network learn
      γ<sup>moves to solved</sup>, an estimate of the distance to the solution. Evaluation plays Eval Games round games without noise.</p>
    <div class="knobs-grid">
      <div class="knob-item" title="Per-move discount of search values and value targets"><label>Discount γ</label><input type="number" id="g-discount" step="0.01" min="0.5" max="1"></div>
      <div class="knob-item" title="Last round (26 quarter turns reach every cube)"><label>Max Rounds</label><input type="number" id="g-max_rounds" min="1" max="40"></div>
      <div class="knob-item"><label>Budget: Moves per Turn</label><input type="number" id="g-moves_per_turn" min="0" max="10"></div>
      <div class="knob-item"><label>Budget: Extra Moves</label><input type="number" id="g-extra_moves" min="0" max="255"></div>
      <div class="knob-item"><label>Budget: Max Moves</label><input type="number" id="g-max_moves" min="1" max="255"></div>
    </div>
  </div>

  <div class="card two-player-only" style="margin-top:14px;">
    <h2>
      <span>🎯 Evaluation Opponents (applied on next iteration — the trained net plays each from fixed openings, colors swapped)</span>
      <button class="btn" onclick="pushOpponents()">Apply Opponents</button>
    </h2>
    <div id="live-opps"></div>
  </div>

  <div class="card two-player-only" style="margin-top:14px;">
    <h2>
      <span>⚖️ Elo anchors (<span id="elo-game-label">Ultimate Tic-Tac-Toe</span>)</span>
      <span id="elo-status" style="font-size:0.78rem;color:var(--muted)">—</span>
    </h2>
    <p class="hint">Engines of fixed strength that define the Elo scale. Calibrating plays every pair of anchors (fixed openings, colors swapped)
      and fits their ratings, the first anchor fixed at 0. After that every evaluation against anchors yields a rating without extra games:
      add anchors near the network's strength as evaluation opponents above. Add saved networks to extend the scale upwards, and do not overwrite
      an anchor's checkpoint file afterwards. Calibration runs while live training is paused.</p>
    <table style="margin-bottom:8px"><thead><tr><th style="text-align:left">Anchor</th><th>Elo</th><th>± std. error</th></tr></thead>
      <tbody id="elo-anchor-rows"></tbody></table>
    <div id="elo-anchors"></div>
    <div style="display:flex;gap:10px;align-items:center;margin-top:8px;font-size:0.8rem;color:var(--muted)">
      <label>Games per pair <input type="number" id="elo-games" value="48" min="2" max="1000" step="2"
        style="width:60px;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:3px 6px;border-radius:4px;"></label>
      <button class="btn" id="elo-calibrate-btn" onclick="calibrateElo()">⚖️ Calibrate anchors</button>
    </div>
  </div>

  <div class="card" style="margin-top:14px;">
    <h2>
      <span>🖥️ Backend, Device &amp; Inference (both games — rebuilds the models, keeps weights, resets AdamW state)</span>
      <button class="btn" onclick="pushSystem()">Apply System</button>
    </h2>
    <div class="knobs-grid">
      <div class="knob-item" title="Neural-network library. tch = libtorch (PyTorch kernels); needs a build with --features tch (./start_lab.sh --tch). Checkpoints work with both.">
        <label>NN Backend</label>
        <select id="sys-backend" onchange="renderSystemDeviceOptions()"></select>
      </div>
      <div class="knob-item" title="Device used for training and NN inference.">
        <label>Device</label>
        <select id="sys-device">
          <option value="gpu" id="sys-device-gpu">GPU</option>
          <option value="cpu">CPU</option>
        </select>
      </div>
      <div class="knob-item" title="Maximum positions per batched NN inference call.">
        <label>Inference Batch Size</label>
        <input type="number" id="sys-infer-batch" min="1" max="4096" step="8">
      </div>
      <div class="knob-item" title="Inference worker threads, one model replica each. CUDA: up to one per GPU. Metal: always 1. CPU: any.">
        <label>Inference Workers</label>
        <input type="number" id="sys-infer-workers" min="1" max="64" step="1">
      </div>
      <div class="knob-item">
        <label>Status</label>
        <div id="sys-status" style="font-size:0.8rem; padding:5px 0;">—</div>
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
    <p class="hint">Checkpoints live in the <code>checkpoints/</code> folder and are scoped per game (<code>ttt_*.safetensors</code> and <code>uttt_*.safetensors</code>). Select any saved checkpoint in Engine A or Engine B below to compare training stages.</p>
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
    <p class="hint">Games start from the same fixed openings as the training evaluation (Ultimate TTT: random 2-move openings; Tic-Tac-Toe: the empty board), each played once per seat, so two engines that always play their best move still meet in different games.</p>
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
<script>
let stateData = null;
let activeGame = 'uttt';
let knobsInitializedForGame = null;
let systemInitialized = false;
let modelsSignature = '';

function switchTab(tab) {
  document.getElementById('tab-lab').classList.toggle('hidden', tab !== 'lab');
  document.getElementById('tab-arena').classList.toggle('hidden', tab !== 'arena');
  document.getElementById('tab-play').classList.toggle('hidden', tab !== 'play');
  document.getElementById('tab-btn-lab').classList.toggle('active', tab === 'lab');
  document.getElementById('tab-btn-arena').classList.toggle('active', tab === 'arena');
  document.getElementById('tab-btn-play').classList.toggle('active', tab === 'play');
  document.getElementById('tab-experiments').classList.toggle('hidden', tab !== 'experiments');
  document.getElementById('tab-btn-experiments').classList.toggle('active', tab === 'experiments');
  if (tab === 'experiments') expRefresh(true);
  if (tab === 'play') playActivated();
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

// [knob JSON field, input id, parser: 'int' | 'float' | 'str']
const KNOB_FIELDS = [
  ['games_per_iteration', 'k-games', 'int'],
  ['simulations_per_move', 'k-sims', 'int'],
  ['self_play_threads', 'k-sp-threads', 'int'],
  ['opening_moves', 'k-opening', 'int'],
  ['dirichlet_alpha', 'k-alpha', 'float'],
  ['dirichlet_epsilon', 'k-eps', 'float'],
  ['c_init', 'k-cinit', 'float'],
  ['c_base', 'k-cbase', 'float'],
  ['value_q_weight', 'k-qw', 'float'],
  ['value_q_schedule', 'k-qw-schedule', 'str'],
  ['train_steps', 'k-steps', 'int'],
  ['batch_size', 'k-batch', 'int'],
  ['learning_rate', 'k-lr', 'float'],
  ['weight_decay', 'k-wd', 'float'],
  ['buffer_capacity', 'k-buffer', 'int'],
  ['eval_games', 'k-eval-games', 'int'],
  ['eval_sims', 'k-eval-sims', 'int'],
  ['eval_threads', 'k-eval-threads', 'int'],
];

function collectKnobs() {
  const out = {};
  const opps = oppRead('live-opps');
  if (opps.length) out.eval_opponents = opps;
  for (const [key, id, kind] of KNOB_FIELDS) {
    const raw = document.getElementById(id).value;
    if (kind === 'str') { out[key] = raw; continue; }
    const v = kind === 'int' ? parseInt(raw, 10) : parseFloat(raw);
    if (Number.isFinite(v)) out[key] = v;
  }
  const fpu = parseFloat(document.getElementById('k-fpu').value);
  if (Number.isFinite(fpu)) out.fpu = { [document.getElementById('k-fpu-mode').value]: fpu };
  if (activeGame === 'rubiks') out.game_settings = collectGameSettings();
  return out;
}

// The cube's own settings ([JSON field, parser]); inputs are `g-<field>`.
const GAME_SETTING_FIELDS = [['discount', parseFloat], ['max_rounds', parseInt], ['moves_per_turn', parseInt], ['extra_moves', parseInt], ['max_moves', parseInt]];

function collectGameSettings() {
  const out = {};
  for (const [key, parse] of GAME_SETTING_FIELDS) {
    const v = parse(document.getElementById(`g-${key}`).value);
    if (Number.isFinite(v)) out[key] = v;
  }
  return out;
}

async function pushGameSettings() {
  const res = await fetch('/api/knobs', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ game_settings: collectGameSettings() })
  });
  stateData = await res.json();
  for (const [key] of GAME_SETTING_FIELDS) document.getElementById(`g-${key}`).value = stateData.game_settings[key];
  renderDashboard(stateData);
}

async function pushOpponents() {
  const res = await fetch('/api/knobs', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ eval_opponents: oppRead('live-opps') })
  });
  stateData = await res.json();
  oppRender('live-opps', stateData.eval_opponents, stateData.active_game);
  renderDashboard(stateData);
}

async function toggleTraining() {
  const paused = !stateData || !stateData.knobs || stateData.knobs.paused;
  await sendTrainControl(paused ? 'start' : 'pause');
}

function renderSystemDeviceOptions() {
  if (!stateData || !stateData.system) return;
  const b = stateData.system.backends.find(x => x.name === document.getElementById('sys-backend').value);
  document.getElementById('sys-device-gpu').textContent = b && b.gpu ? `GPU (${b.gpu})` : 'GPU (none available → CPU)';
}

async function pushSystem() {
  const body = {
    backend: document.getElementById('sys-backend').value,
    device: document.getElementById('sys-device').value,
    infer_batch: parseInt(document.getElementById('sys-infer-batch').value, 10),
    infer_workers: parseInt(document.getElementById('sys-infer-workers').value, 10),
  };
  const res = await fetch('/api/system', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  stateData = await res.json();
  systemInitialized = false;
  renderDashboard(stateData);
}

async function sendTrainControl(action) {
  const total_iterations = parseInt(document.getElementById('ctrl-total-iters').value, 10) || 15;
  const auto_save_every = parseInt(document.getElementById('ctrl-autosave').value, 10) || 0;
  const payload = {
    action,
    game: activeGame,
    total_iterations,
    auto_save_every,
    ...collectKnobs(),
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
  activeGame = d.active_game || 'uttt';
  const title = d.active_game_title || (activeGame === 'uttt' ? 'Ultimate Tic-Tac-Toe (9x9)' : 'Tic-Tac-Toe (3x3)');

  document.getElementById('game-btn-ttt').classList.toggle('active', activeGame === 'ttt');
  document.getElementById('game-btn-uttt').classList.toggle('active', activeGame === 'uttt');
  document.getElementById('game-btn-rubiks').classList.toggle('active', activeGame === 'rubiks');
  const puzzle = activeGame === 'rubiks';
  document.body.classList.toggle('puzzle', puzzle);
  if (puzzle && !document.getElementById('tab-arena').classList.contains('hidden')) switchTab('lab');
  document.getElementById('knobs-game-label').textContent = title;
  document.getElementById('models-game-label').textContent = title;
  document.getElementById('cal-game-label').textContent = title;
  document.getElementById('match-game-label').textContent = title;
  document.getElementById('hist-game-label').textContent = title;


  document.getElementById('device-badge').textContent = 'Device: ' + d.device_name;
  const phaseEl = document.getElementById('phase-badge');
  phaseEl.textContent = (d.knobs.paused ? '⏸ ' : '🟢 ') + d.phase;
  phaseEl.className = 'badge ' + (d.knobs.paused ? 'paused' : 'live');
  const toggleBtn = document.getElementById('btn-toggle-train');
  if (d.knobs.paused) {
    toggleBtn.textContent = d.current_iteration > 0 ? '▶ Resume Training' : '▶ Start Training';
    toggleBtn.className = 'btn green';
  } else {
    toggleBtn.textContent = '⏸ Pause Training';
    toggleBtn.className = 'btn amber';
  }

  if (d.system) {
    const sys = d.system;
    const sel = document.getElementById('sys-backend');
    const names = sys.backends.map(b => b.name).join(',');
    if (sel.dataset.names !== names) {
      sel.innerHTML = sys.backends.map(b => `<option value="${b.name}">${b.name === 'tch' ? 'tch (libtorch)' : 'candle'}</option>`).join('');
      sel.dataset.names = names;
    }
    if (!systemInitialized) {
      sel.value = sys.backend;
      renderSystemDeviceOptions();
      document.getElementById('sys-device').value = sys.device;
      document.getElementById('sys-infer-batch').value = sys.infer_batch;
      document.getElementById('sys-infer-workers').value = sys.infer_workers;
      // Keep syncing until applied: the server reports what is actually in effect
      // (e.g. Metal always runs a single inference worker).
      systemInitialized = !sys.pending;
    }
    document.getElementById('sys-status').textContent = sys.pending
      ? '⏳ Applying before next iteration...'
      : `✓ Active · ${sys.cpu_cores} CPU cores`;
  }
  document.getElementById('iter-badge').textContent = `Iter: ${d.current_iteration} / ${d.total_iterations}`;

  if (knobsInitializedForGame !== activeGame) {
    for (const [key, id] of KNOB_FIELDS) {
      if (d.knobs[key] !== undefined) document.getElementById(id).value = d.knobs[key];
    }
    const [fpuMode, fpuValue] = Object.entries(d.knobs.fpu || { absolute: 0 })[0];
    document.getElementById('k-fpu-mode').value = fpuMode;
    document.getElementById('k-fpu').value = fpuValue;
    oppRender('live-opps', d.eval_opponents || [], activeGame);
    if (d.game_settings) for (const [key] of GAME_SETTING_FIELDS) document.getElementById(`g-${key}`).value = d.game_settings[key];
    document.getElementById('ctrl-total-iters').value = d.total_iterations;
    document.getElementById('ctrl-autosave').value = d.auto_save_every;
    knobsInitializedForGame = activeGame;
  } else if (document.activeElement !== document.getElementById('ctrl-total-iters')) {
    document.getElementById('ctrl-total-iters').value = d.total_iterations;
  }

  renderModels(d.models || []);
  renderCalibration(d.calibration);
  renderMatches(d.matches || []);

  renderEloAnchors(d.elo, title);
  renderUtilization(d.utilization);
  const h = d.history || [];
  if (puzzle) {
    renderRoundCharts(h);
  } else {
    renderEloChart(h, d.elo);
    renderArenaCards(h);
  }
  renderMetricCards(h);
}

// ---- Rubik's cube: round-game score and solve rate per round ----
function renderRoundCharts(h) {
  const rows = h.filter(x => x.rounds);
  const last = rows[rows.length - 1];
  document.getElementById('latest-round-score').textContent = last
    ? `${last.rounds.eval.score.toFixed(2)} ± ${last.rounds.eval.score_stderr.toFixed(2)}` : '—';
  document.getElementById('latest-best-round').textContent = last ? `best round ${last.rounds.eval.best_round}` : '—';
  if (!rows.length) {
    clearCanvas('chart-round-score');
    clearCanvas('chart-round-heat');
    document.getElementById('round-table').innerHTML = '';
    return;
  }
  const evals = rows.map(x => x.rounds.eval);
  drawChart('chart-round-score', rows.map(x => x.iteration), [
    { data: evals.map(e => e.score + e.score_stderr), color: '#1f6f55', width: 1 },
    { data: evals.map(e => e.score - e.score_stderr), color: '#1f6f55', width: 1 },
    { data: rows.map(x => x.rounds.self_play.score), color: '#38bdf8' },
    { data: evals.map(e => e.score), color: '#34d399' },
  ], 0);
  drawRoundHeat('chart-round-heat', evals);
  const e = last.rounds.eval;
  document.getElementById('round-table').innerHTML = e.reached.map((reached, i) => {
    const solved = e.solved[i];
    const moves = solved ? (e.solve_moves[i] / solved).toFixed(1) : '—';
    return `<tr><td style="text-align:left">${i + 1}</td><td>${reached}</td><td>${(100 * solved / Math.max(1, reached)).toFixed(0)}%</td><td>${moves}</td></tr>`;
  }).join('');
}

// Heat map: one column per evaluation, one row per round, coloured by the solve rate.
function drawRoundHeat(canvasId, evals) {
  const c = document.getElementById(canvasId);
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const rounds = Math.max(1, ...evals.map(e => e.reached.length));
  const padL = 30, padB = 4, padT = 4;
  const cw = (c.width - padL) / evals.length, rh = (c.height - padT - padB) / rounds;
  ctx.font = '10px monospace';
  ctx.fillStyle = '#64748b';
  for (let r = 0; r < rounds; r += Math.ceil(rounds / 10)) {
    ctx.fillText(String(r + 1), 4, c.height - padB - r * rh - rh / 2 + 3);
  }
  evals.forEach((e, col) => {
    for (let r = 0; r < rounds; r++) {
      const reached = e.reached[r] || 0;
      const rate = reached ? e.solved[r] / reached : null;
      // Red (0) through amber to green (1).
      ctx.fillStyle = rate == null ? '#111827' : `hsl(${Math.round(120 * rate)}, 65%, ${30 + 12 * rate}%)`;
      ctx.fillRect(padL + col * cw, c.height - padB - (r + 1) * rh, Math.max(1, cw - 1), Math.max(1, rh - 1));
    }
  });
}

// ---- Strength: anchored Elo and per-opponent wins / draws / losses ----
let eloEditorGame = null;
let eloWasBusy = false;

function renderEloAnchors(elo, title) {
  document.getElementById('elo-game-label').textContent = title;
  document.getElementById('elo-status').textContent = elo.status
    || (elo.calibrated_unix ? `calibrated ${new Date(elo.calibrated_unix * 1000).toLocaleString()} · ${elo.games_per_pair} games per pair` : 'not calibrated yet');
  document.getElementById('elo-calibrate-btn').disabled = elo.busy;
  document.getElementById('elo-anchor-rows').innerHTML = elo.anchors.length
    ? elo.anchors.map(a => `<tr><td style="text-align:left">${a.label}</td><td>${a.elo.toFixed(0)}</td>`
      + `<td>${a.stderr == null ? '—' : a.stderr === 0 ? 'fixed' : a.stderr.toFixed(0)}</td></tr>`).join('')
    : '<tr><td colspan="3" class="hint" style="text-align:left">No anchors calibrated for this game yet.</td></tr>';
  // The editor shows the calibrated anchors (or the defaults) when the game changes or a
  // calibration ends, and is left alone in between.
  if (eloEditorGame !== activeGame || (eloWasBusy && !elo.busy)) {
    oppRender('elo-anchors', elo.anchors.length ? elo.anchors.map(a => a.opponent) : elo.default_anchors, activeGame);
    eloEditorGame = activeGame;
  }
  eloWasBusy = elo.busy;
}

async function calibrateElo() {
  const anchors = oppRead('elo-anchors');
  if (anchors.length < 2) { alert('Calibration needs at least two anchors.'); return; }
  const games_per_pair = parseInt(document.getElementById('elo-games').value, 10) || 48;
  const pairs = anchors.length * (anchors.length - 1) / 2;
  if (!confirm(`Play ${pairs} pairings × ${games_per_pair} games between ${anchors.length} anchors? Live training must be paused for it to start.`)) return;
  const res = await fetch('/api/elo', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ game: activeGame, anchors, games_per_pair })
  });
  if (!res.ok) { alert((await res.json()).error); return; }
  stateData = await res.json();
  renderDashboard(stateData);
}

function renderEloChart(h, elo) {
  const ratings = h.map(x => x.elo);
  const last = [...ratings].reverse().find(r => r);
  document.getElementById('latest-elo').textContent = last
    ? `${last.elo.toFixed(0)} ± ${last.stderr == null ? '?' : last.stderr.toFixed(0)}`
    : (elo.anchors.length ? 'no anchor among the evaluation opponents' : 'calibrate the Elo anchors first');
  if (!last) { clearCanvas('chart-elo'); return; }
  const band = (sign) => ratings.map(r => r && r.stderr != null ? r.elo + sign * r.stderr : NaN);
  drawChart('chart-elo', h.map(x => x.iteration), [
    { data: band(1), color: '#1f6f55', width: 1 },
    { data: band(-1), color: '#1f6f55', width: 1 },
    { data: ratings.map(r => r ? r.elo : NaN), color: '#34d399' },
  ]);
}

const WDL_SERIES = [['wins', '#34d399'], ['draws', '#94a3b8'], ['losses', '#fb7185']];

// One card per opponent seen in the history (opponents can change between iterations).
function renderArenaCards(h) {
  const grid = document.getElementById('arena-cards');
  const labels = [...new Set(h.flatMap(x => x.arena.map(a => a.label)))];
  const sig = labels.join('|');
  if (grid.dataset.labels !== sig) {
    grid.innerHTML = labels.map((l, i) => `
      <div class="card">
        <h2><span>vs ${l}</span><span id="arena-latest-${i}" style="font-size:0.78rem;color:var(--green)">—</span></h2>
        <p class="hint">Wins, draws and losses of the trained network in % of the evaluation games after every iteration.</p>
        <canvas id="arena-chart-${i}" width="600" height="170"></canvas>
        <div class="legend">${WDL_SERIES.map(([n, c]) => `<span style="color:${c}">${n}</span>`).join('')}</div>
      </div>`).join('');
    grid.dataset.labels = sig;
  }
  labels.forEach((label, i) => {
    const results = h.map(x => x.arena.find(a => a.label === label));
    const pct = (k) => results.map(r => { const n = r ? r.wdl[0] + r.wdl[1] + r.wdl[2] : 0; return n ? 100 * r.wdl[k] / n : NaN; });
    drawChart(`arena-chart-${i}`, h.map(x => x.iteration), WDL_SERIES.map(([, color], k) => ({ data: pct(k), color })), 0, 100);
    const last = [...results].reverse().find(r => r);
    document.getElementById(`arena-latest-${i}`).textContent = last ? `${last.wdl[0]}W-${last.wdl[1]}D-${last.wdl[2]}L` : '—';
  });
}

// Training-tab charts, one per metric so each has its own y-axis scale.
// series: [label, color, value of an iteration record]; latest: text for the newest record.
const METRIC_CARDS = [
  { id: 'kl', title: 'Policy KL divergence', latest: x => x.policy_kl.toFixed(3),
    hint: 'How far the network\'s move probabilities are from the search\'s visit counts, KL(π_search ‖ p_net), on the training batches. Lower = the network already predicts what search finds. It drops quickly at first and then levels off; a sudden rise means search keeps finding moves the network does not expect.',
    series: [['KL', '#38bdf8', x => x.policy_kl]] },
  { id: 'value', title: 'Value loss (MSE)', latest: x => x.value_loss.toFixed(3),
    hint: 'Squared error of the network\'s outcome prediction against the value targets (game result, blended with the search value by qw). It falls as the network learns who is winning, but never reaches 0: early positions are genuinely uncertain.',
    series: [['MSE', '#fb7185', x => x.value_loss]] },
  { id: 'entropy', title: 'Search decisiveness (root entropy)', latest: x => x.root_entropy_mean.toFixed(2),
    hint: 'Entropy of the visit distribution at the root, averaged over self-play moves. High = search spreads its visits over many moves, low = it settles on few. Falls as the network gets confident; the root noise keeps it from reaching 0.',
    series: [['H(π)', '#a78bfa', x => x.root_entropy_mean]] },
  { id: 'loss', title: 'Total training loss', latest: x => x.total_loss.toFixed(3),
    hint: 'Policy cross-entropy + value MSE, the quantity SGD minimises. The policy part includes the entropy of the search targets themselves, so it cannot fall below that; use KL and value loss to judge learning.',
    series: [['loss', '#f472b6', x => x.total_loss]] },
  { id: 'moves', title: 'Self-play game length (moves)', latest: x => x.game_moves_mean.toFixed(1),
    hint: 'Average length of the self-play games. Changes when play style changes: longer games usually mean both sides defend better.',
    series: [['moves', '#34d399', x => x.game_moves_mean]] },
  { id: 'speed', title: 'Self-play speed (simulations / s)', latest: x => Math.round(x.simulations_per_sec).toLocaleString(),
    hint: 'Search simulations per second during self-play, the throughput of CPU tree search and GPU network evaluation together.',
    series: [['sims/s', '#fbbf24', x => x.simulations_per_sec]] },
  { id: 'batch', title: 'Network batch fill', latest: x => `${x.batch_size_mean.toFixed(1)} of ${x.max_batch_size}`,
    hint: 'Average number of positions per network evaluation during self-play, out of the configured maximum batch. A full batch keeps the GPU efficient; a low fill means the CPU search threads cannot supply positions fast enough.',
    series: [['positions', '#22d3ee', x => x.batch_size_mean]], min: 0, max: h => Math.max(...h.map(x => x.max_batch_size)) },
  { id: 'time', title: 'Iteration time (s)', latest: x => `${((x.self_play_ms + x.train_ms + x.eval_ms) / 1000).toFixed(1)} s`,
    hint: 'Where each iteration spends its time: self-play (CPU search + GPU evaluation), SGD training (GPU) and the arena evaluation.',
    series: [['self-play', '#38bdf8', x => x.self_play_ms / 1000], ['training', '#fb7185', x => x.train_ms / 1000], ['evaluation', '#a78bfa', x => x.eval_ms / 1000]], min: 0 },
];

function buildMetricCards() {
  const grid = document.getElementById('metric-cards');
  if (grid.childElementCount) return;
  grid.innerHTML = `
    <div class="card">
      <h2><span>CPU &amp; GPU utilization (live, last 5 min)</span><span id="latest-util" style="font-size:0.78rem;color:var(--green)">—</span></h2>
      <p class="hint">CPU: this process's CPU time as % of all <span id="util-cores">?</span> cores. GPU: device busy % reported by the OS
        (<span id="util-gpu-src">n/a</span>; other programs count too). Sampled every second and tagged with what the lab was doing.</p>
      <canvas id="chart-util" width="600" height="170"></canvas>
      <div class="legend"><span style="color:#34d399">CPU %</span><span style="color:#fbbf24">GPU %</span></div>
      <table style="margin-top:6px"><thead><tr><th style="text-align:left">Average while</th><th>samples</th><th>CPU</th><th>GPU</th></tr></thead>
        <tbody id="util-by-activity"></tbody></table>
    </div>` + METRIC_CARDS.map(c => `
    <div class="card">
      <h2><span>${c.title}</span><span id="latest-${c.id}" style="font-size:0.78rem;color:${c.series[0][1]}">—</span></h2>
      <p class="hint">${c.hint}</p>
      <canvas id="chart-${c.id}" width="600" height="170"></canvas>
      ${c.series.length > 1 ? `<div class="legend">${c.series.map(([l, col]) => `<span style="color:${col}">${l}</span>`).join('')}</div>` : ''}
    </div>`).join('');
}

function renderUtilization(u) {
  buildMetricCards();
  const samples = (u && u.samples) || [];
  document.getElementById('util-cores').textContent = u ? u.cores : '?';
  document.getElementById('util-gpu-src').textContent = u && u.gpu_source ? u.gpu_source : 'not available';
  if (!samples.length) { clearCanvas('chart-util'); return; }
  const last = samples[samples.length - 1];
  document.getElementById('latest-util').textContent =
    `CPU ${last.cpu.toFixed(0)}%` + (last.gpu == null ? '' : ` · GPU ${last.gpu.toFixed(0)}%`) + ` · ${last.activity}`;
  drawChart('chart-util', samples.map(x => x.t), [
    { data: samples.map(x => x.cpu), color: '#34d399' },
    { data: samples.map(x => x.gpu == null ? NaN : x.gpu), color: '#fbbf24' },
  ], 0, 100);
  const groups = {};
  for (const x of samples) (groups[x.activity] = groups[x.activity] || []).push(x);
  const avg = (xs, f) => { const v = xs.map(f).filter(Number.isFinite); return v.length ? `${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(0)}%` : '—'; };
  document.getElementById('util-by-activity').innerHTML = ['self-play', 'training', 'evaluation', 'experiment', 'idle']
    .filter(a => groups[a]).map(a => `<tr><td style="text-align:left">${a}</td><td>${groups[a].length}</td>`
      + `<td>${avg(groups[a], x => x.cpu)}</td><td>${avg(groups[a], x => x.gpu == null ? NaN : x.gpu)}</td></tr>`).join('');
}

function renderMetricCards(h) {
  buildMetricCards();
  for (const c of METRIC_CARDS) {
    document.getElementById(`latest-${c.id}`).textContent = h.length ? c.latest(h[h.length - 1]) : '—';
    if (!h.length) { clearCanvas(`chart-${c.id}`); continue; }
    drawChart(`chart-${c.id}`, h.map(x => x.iteration), c.series.map(([, color, f]) => ({ data: h.map(f), color })),
      c.min, typeof c.max === 'function' ? c.max(h) : c.max);
  }
}

function clearCanvas(id) {
  const c = document.getElementById(id);
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
}

function renderModels(models) {
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
  if (data.error) {
    alert(`Could not load ${name} into the live model:\n${data.error}\n\nThe checkpoint must have the live network's architecture.`);
    await fetchState();
    return;
  }
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
      if (!Number.isFinite(v)) return;
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
    const span = maxY - minY;
    ctx.fillText(Math.abs(maxY) >= 10000 ? `${(val / 1000).toFixed(0)}k` : val.toFixed(span >= 20 ? 0 : span >= 2 ? 1 : 2), 4, y + 4);
  }

  series.forEach(s => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 2.2;
    ctx.beginPath();
    let penDown = false;
    s.data.forEach((v, idx) => {
      if (!Number.isFinite(v)) { penDown = false; return; }
      const x = padL + (xs.length === 1 ? plotW / 2 : (plotW * idx) / (xs.length - 1));
      const y = padT + plotH * (1 - (v - minY) / (maxY - minY));
      if (!penDown) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      penDown = true;
    });
    ctx.stroke();
  });
}

async function pushKnobs() {
  const body = collectKnobs();
  const res = await fetch('/api/knobs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  stateData = await res.json();
  renderDashboard(stateData);
}

fetchState();
setInterval(fetchState, 800);
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::AlphaZeroTrainer;
    use rand::SeedableRng;

    #[test]
    fn telemetry_hub_json_and_play_analysis_work() {
        let dev = NnDevice::CPU;
        let config = tic_tac_toe::net_mlp();
        let trainer = AlphaZeroTrainer::<27, 9, 2>::new(config, 1e-3, 1e-4, dev).expect("trainer init");
        let service = Arc::new(trainer.build_inference_service(16, &[dev]).expect("service init"));

        let hub = TelemetryHub::new("Cpu", 10, LiveKnobs::default());
        assert!(hub.to_json().contains("\"device_name\":\"Cpu\""));
        assert!(hub.play_json(r#"{"game":"ttt","moves":[4]}"#).is_err(), "no network attached yet");

        hub.attach_ttt_service(service, config, vec![dev]);
        let body = r#"{"game":"ttt","moves":[4],"seed":3,"engine":{"kind":"alphazero","checkpoint":"live","net":"mlp","sims":32}}"#;
        let v: serde_json::Value = serde_json::from_str(&hub.play_json(body).unwrap()).unwrap();
        assert_eq!(v["board"]["to_move"], 2);
        assert_eq!(v["board"]["legal"].as_array().unwrap().len(), 8);
        let moves = v["analysis"]["moves"].as_array().unwrap();
        let prior_sum: f64 = moves.iter().map(|m| m["prior"].as_f64().unwrap()).sum();
        assert!((prior_sum - 1.0).abs() < 1e-4, "priors must sum to 1, got {prior_sum}");
        assert_ne!(v["analysis"]["chosen"], 4, "the engine plays a free cell");
        let root_sims = |v: &serde_json::Value| v["analysis"]["simulations"].as_u64().unwrap();
        let body = body.replace(r#""seed":3,"#, r#""seed":3,"trees":4,"#);
        let four: serde_json::Value = serde_json::from_str(&hub.play_json(&body).unwrap()).unwrap();
        assert!(root_sims(&four) > 3 * root_sims(&v), "four trees search about four times as much");
        assert!(hub.play_json(r#"{"game":"ttt","moves":[4,4]}"#).is_err(), "illegal move");
    }

    #[test]
    fn arena_multi_match_and_calibration_work_for_ttt_and_uttt() {
        let dev = NnDevice::CPU;
        let ttt_cfg = tic_tac_toe::net_mlp();
        let ttt_trainer =
            AlphaZeroTrainer::<27, 9, 2>::new(ttt_cfg, 1e-3, 1e-4, dev).expect("ttt init");
        let ttt_service = Arc::new(
            ttt_trainer
                .build_inference_service(16, std::slice::from_ref(&dev))
                .expect("ttt service"),
        );

        let uttt_cfg = ultimate_ttt::net_mlp();
        let uttt_trainer = AlphaZeroTrainer::<{ ultimate_ttt::INPUT_SIZE }, 81, 2>::new(uttt_cfg, 1e-3, 1e-4, dev)
            .expect("uttt init");
        let uttt_service = Arc::new(
            uttt_trainer
                .build_inference_service(16, std::slice::from_ref(&dev))
                .expect("uttt service"),
        );

        let hub = TelemetryHub::new("Cpu", 5, LiveKnobs::default());
        hub.attach_ttt_service(ttt_service, ttt_cfg, vec![dev]);
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
        let mut uttt_buffer = crate::training::ReplayBuffer::<{ ultimate_ttt::INPUT_SIZE }, 81, 2>::new(4096);
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
