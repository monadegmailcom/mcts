//! Pure-Rust Live Training Telemetry Hub, NN Checkpoint Manager, Multi-Engine Match Arena,
//! Interactive Position Inspector, and Embedded Browser Lab.
//!
//! Replaces external Python TensorBoard dependencies with a zero-dependency pure-Rust HTTP + JSON
//! server (`std::net::TcpListener`) that serves:
//! - `GET /`: The interactive **AlphaZero Browser Lab, Multi-Engine Arena & Play Inspector**.
//! - `GET /api/state`: Live JSON snapshot of training iterations, system/queue/dedup telemetry,
//!   tunable hyperparameter knobs, registered neural network models, and multi-match history.
//! - `POST /api/knobs`: Updates live training hyperparameter knobs.
//! - `POST /api/models`: Saves the live NN to `<name>.safetensors`, loads a `.safetensors`
//!   checkpoint into the live model, or deletes a saved checkpoint.
//! - `POST /api/match`: Runs a parallel role-swapped Tic-Tac-Toe multi-match between any two
//!   configurable engines (`alphazero`, `mcts`, `tree-minimax`, `minimax`, `random`).
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
use crate::inference::{BatchEvaluator, InferenceService};
use crate::matchup::{RandomPlayer, run_multi_match};
use crate::mcts::{MctsConfig, MctsPlayer, PlayoutEvaluator, Uct};
use crate::minimax::{MinimaxBudget, MinimaxPlayer, TreeMinimaxPlayer};
use crate::nn::{AlphaZeroNetConfig, CandleBatchEvaluator};
use crate::player::Player;
use crate::root_parallel::{RootParallelBudget, RootParallelConfig, RootParallelPlayer};

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
    /// `[wins, draws, losses]` against `RandomPlayer`.
    pub vs_random: [usize; 3],
    /// `[wins, draws, losses]` against `Minimax(depth=2)`.
    pub vs_mm2: [usize; 3],
    /// `[wins, draws, losses]` against `Minimax(depth=5)` (optimal Tic-Tac-Toe oracle).
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
        Self {
            paused: false,
            games_per_iteration: 128,
            simulations_per_move: 64,
            train_steps: 60,
            batch_size: 64,
            learning_rate: 2e-3,
            dirichlet_epsilon: 0.25,
            c_init: 1.25,
        }
    }
}

impl LiveKnobs {
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
    pub size_bytes: u64,
    pub description: String,
}

impl ModelInfo {
    fn to_json(&self) -> String {
        format!(
            "{{\"id\":\"{}\",\"kind\":\"{}\",\"size_bytes\":{},\"description\":\"{}\"}}",
            escape_json(&self.id),
            self.kind,
            self.size_bytes,
            escape_json(&self.description),
        )
    }
}

/// Configuration for one side of a Tic-Tac-Toe Multi-Match in the Browser Lab.
#[derive(Clone, Debug)]
pub struct TttEngineConfig {
    /// `"alphazero"`, `"mcts"`, `"tree-minimax"`, `"minimax"`, or `"random"`.
    pub engine: String,
    /// Minimax search depth (`0..=9`).
    pub depth: u32,
    /// Whether Minimax / TreeMinimax uses `TttHeuristic` (`true`) or `ZeroHeuristic` (`false`).
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
            budget_kind: "depth".to_string(),
            budget_val: 4,
            arena_cap: 4096,
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
                "\"nn_logical_batch\":{:.2},",
                "\"nn_unique_batch\":{:.2},",
                "\"nn_dedup_saved_pct\":{:.2},",
                "\"nn_us_per_state\":{:.2}",
                "}}"
            ),
            self.match_id,
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

    // 1. Direct single-state evaluation on the service's client for raw NN priors & value.
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

    // 2. Run Root-Parallel AlphaZero search (`num_trees`) for `total_simulations` without advancing.
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
struct TttModelContext {
    live_service: Arc<InferenceService<CandleBatchEvaluator<27, 9, 2>, 27, 9, 2>>,
    net_config: AlphaZeroNetConfig,
    devices: Vec<Device>,
}

impl TttModelContext {
    fn resolve_evaluator(&self, model_id: &str) -> Result<CandleBatchEvaluator<27, 9, 2>, String> {
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
            .map_err(|e| e.to_string())
    }
}

struct HubInner {
    device_name: String,
    phase: String,
    current_iteration: usize,
    total_iterations: usize,
    knobs: LiveKnobs,
    history: Vec<IterationTelemetry>,
    matches: Vec<TttMatchRecord>,
    inspector: Option<Arc<TttInspectorFn>>,
    ttt_context: Option<TttModelContext>,
}

/// Thread-safe telemetry, model registry, multi-match arena, and live control hub shared between
/// the training loop and the embedded HTTP Browser Lab server.
#[derive(Clone)]
pub struct TelemetryHub {
    inner: Arc<Mutex<HubInner>>,
}

impl TelemetryHub {
    pub fn new(device_name: impl Into<String>, total_iterations: usize, knobs: LiveKnobs) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HubInner {
                device_name: device_name.into(),
                phase: "Initializing".to_string(),
                current_iteration: 0,
                total_iterations,
                knobs,
                history: Vec::with_capacity(total_iterations.max(64)),
                matches: Vec::new(),
                inspector: None,
                ttt_context: None,
            })),
        }
    }

    pub fn set_inspector<F>(&self, f: F)
    where
        F: Fn(&TttGame, usize) -> TttPositionInspection + Send + Sync + 'static,
    {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.inspector = Some(Arc::new(Box::new(f)));
    }

    /// Attaches the live Tic-Tac-Toe [`InferenceService`] and [`AlphaZeroNetConfig`] so the
    /// Browser Lab can manage `.safetensors` models, inspect any model, and run multi-matches
    /// with [`AlphaZeroPlayer`].
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
        guard.ttt_context = Some(TttModelContext {
            live_service,
            net_config,
            devices,
        });
    }

    pub fn set_phase(&self, iteration: usize, phase: impl Into<String>) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.current_iteration = iteration;
        guard.phase = phase.into();
    }

    pub fn record_iteration(&self, record: IterationTelemetry) {
        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        guard.current_iteration = record.iteration;
        guard.history.push(record);
    }

    pub fn knobs(&self) -> LiveKnobs {
        self.inner.lock().expect("telemetry mutex poisoned").knobs
    }

    /// Blocks while `knobs.paused` is `true`, polling every `100ms` so the user can pause/resume
    /// training from the Browser Lab.
    pub fn wait_if_paused(&self) {
        while self.knobs().paused {
            thread::sleep(Duration::from_millis(100));
        }
    }

    pub fn discover_models(&self) -> Vec<ModelInfo> {
        let has_live = self
            .inner
            .lock()
            .expect("telemetry mutex poisoned")
            .ttt_context
            .is_some();
        discover_ttt_models(has_live)
    }

    /// Executes a configured Tic-Tac-Toe Multi-Match between `cfg_a` and `cfg_b` across
    /// `num_games` role-swapped games on `num_workers` threads, recording and returning the
    /// resulting [`TttMatchRecord`].
    pub fn run_ttt_match(
        &self,
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

        let ttt_ctx = self
            .inner
            .lock()
            .expect("telemetry mutex poisoned")
            .ttt_context
            .clone();

        let a_is_az = cfg_a.engine == "alphazero";
        let b_is_az = cfg_b.engine == "alphazero";

        let mut svc_a: Option<Arc<InferenceService<CandleBatchEvaluator<27, 9, 2>, 27, 9, 2>>> =
            None;
        let mut svc_b: Option<Arc<InferenceService<CandleBatchEvaluator<27, 9, 2>, 27, 9, 2>>> =
            None;

        if a_is_az || b_is_az {
            let ctx = ttt_ctx.ok_or_else(|| {
                "AlphaZero model context is not attached to TelemetryHub".to_string()
            })?;
            if a_is_az && b_is_az && cfg_a.model == cfg_b.model {
                let eval = ctx.resolve_evaluator(&cfg_a.model)?;
                let n_rep = eval.num_replicas();
                let shared_svc = Arc::new(InferenceService::new(eval, batch_cap, n_rep));
                svc_a = Some(Arc::clone(&shared_svc));
                svc_b = Some(shared_svc);
            } else {
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
        }

        let svc_a_ref = svc_a.as_deref();
        let svc_b_ref = svc_b.as_deref();
        let factory_a =
            |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> { build_ttt_player(cfg_a, svc_a_ref, g, s) };
        let factory_b =
            |g: &TttGame, s: u64| -> Box<dyn Player<TicTacToe, 2> + '_> { build_ttt_player(cfg_b, svc_b_ref, g, s) };

        let root_game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let start = Instant::now();
        let outcome = run_multi_match(&root_game, [&factory_a, &factory_b], games, workers, seed);
        let duration_ms = (start.elapsed().as_secs_f64() * 1000.0) as f32;
        let games_per_sec = (games as f32) / (duration_ms / 1000.0).max(1e-6);

        // Aggregate NN inference statistics if AlphaZero participated.
        let mut logical_batch = 0.0f32;
        let mut unique_batch = 0.0f32;
        let mut dedup_saved_pct = 0.0f32;
        let mut us_per_state = 0.0f32;

        if let Some(ref sa) = svc_a {
            let mut b_stat = sa.batch_size_stats();
            let mut u_stat = sa.unique_batch_size_stats();
            let mut t_stat = sa.inference_time_stats();
            let (mut tot_req, mut uniq_ev) = sa.dedup_counts();

            if let Some(ref sb) = svc_b
                && !Arc::ptr_eq(sa, sb)
            {
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
        } else if let Some(ref sb) = svc_b {
            logical_batch = sb.batch_size_stats().mean();
            unique_batch = sb.unique_batch_size_stats().mean();
            us_per_state = sb.inference_time_stats().mean();
            dedup_saved_pct = sb.dedup_savings_ratio() * 100.0;
        }

        let wins_a = outcome.wins[0];
        let wins_b = outcome.wins[1];
        let draws = outcome.draws;
        let score_a_pct =
            (((wins_a as f32) + 0.5 * (draws as f32)) / (games as f32).max(1.0)) * 100.0;
        let total_moves =
            outcome.choose_time_us[0].count() + outcome.choose_time_us[1].count();
        let avg_moves = (total_moves as f32) / (games as f32).max(1.0);

        let mut guard = self.inner.lock().expect("telemetry mutex poisoned");
        let match_id = guard.matches.len() + 1;
        let record = TttMatchRecord {
            match_id,
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
            nn_logical_batch: logical_batch,
            nn_unique_batch: unique_batch,
            nn_dedup_saved_pct: dedup_saved_pct,
            nn_us_per_state: us_per_state,
        };
        guard.matches.push(record.clone());
        Ok(record)
    }

    pub fn to_json(&self) -> String {
        let guard = self.inner.lock().expect("telemetry mutex poisoned");
        let models = discover_ttt_models(guard.ttt_context.is_some());
        let models_json = models
            .iter()
            .map(ModelInfo::to_json)
            .collect::<Vec<_>>()
            .join(",");
        let history_json = guard
            .history
            .iter()
            .map(IterationTelemetry::to_json)
            .collect::<Vec<_>>()
            .join(",");
        let matches_json = guard
            .matches
            .iter()
            .map(TttMatchRecord::to_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{",
                "\"device_name\":\"{}\",",
                "\"phase\":\"{}\",",
                "\"current_iteration\":{},",
                "\"total_iterations\":{},",
                "\"knobs\":{},",
                "\"models\":[{}],",
                "\"history\":[{}],",
                "\"matches\":[{}]",
                "}}"
            ),
            escape_json(&guard.device_name),
            escape_json(&guard.phase),
            guard.current_iteration,
            guard.total_iterations,
            guard.knobs.to_json(),
            models_json,
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

fn discover_ttt_models(has_live: bool) -> Vec<ModelInfo> {
    let mut list = Vec::new();
    if has_live {
        list.push(ModelInfo {
            id: "live".to_string(),
            kind: "live",
            size_bytes: 0,
            description: "Active Live Model (shared with training loop)".to_string(),
        });
    }
    list.push(ModelInfo {
        id: "untrained".to_string(),
        kind: "builtin",
        size_bytes: 0,
        description: "Untrained Random Weights (baseline)".to_string(),
    });

    if let Ok(entries) = std::fs::read_dir(".") {
        let mut disk_files: Vec<(String, u64)> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                if name.ends_with(".safetensors") {
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
                size_bytes: sz,
                description: "Saved .safetensors checkpoint on disk".to_string(),
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

fn handle_http_connection(mut stream: TcpStream, hub: &TelemetryHub) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = [0u8; 16384];
    let n = stream.read(&mut buf)?;
    if n == 0 {
        return Ok(());
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let first_line = req.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let raw_path = parts.next().unwrap_or("/");

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
        let body = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &body);
    }

    if method == "POST" && raw_path.starts_with("/api/knobs") {
        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
        {
            let mut guard = hub.inner.lock().expect("telemetry mutex poisoned");
            apply_knob_updates(&mut guard.knobs, body);
        }
        let updated = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &updated);
    }

    if method == "POST" && raw_path.starts_with("/api/models") {
        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
        let action = extract_json_str(body, "action").unwrap_or_default();
        let name = extract_json_str(body, "name").unwrap_or_default();

        let ctx_opt = hub
            .inner
            .lock()
            .expect("telemetry mutex poisoned")
            .ttt_context
            .clone();

        match action.as_str() {
            "save" => {
                let Some(ctx) = ctx_opt else {
                    return write_http_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        "{\"error\":\"No live model attached\"}",
                    );
                };
                let Some(filename) = sanitize_safetensors_filename(&name) else {
                    return write_http_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        "{\"error\":\"Invalid model name\"}",
                    );
                };
                if let Err(e) = ctx.live_service.evaluator().save_safetensors(&filename) {
                    let msg = format!("{{\"error\":\"{}\"}}", escape_json(&e.to_string()));
                    return write_http_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &msg,
                    );
                }
                let iter = hub
                    .inner
                    .lock()
                    .expect("telemetry mutex poisoned")
                    .current_iteration;
                hub.set_phase(iter, format!("Saved checkpoint {filename}"));
            }
            "load_to_live" => {
                let Some(ctx) = ctx_opt else {
                    return write_http_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        "{\"error\":\"No live model attached\"}",
                    );
                };
                let Some(filename) = sanitize_safetensors_filename(&name) else {
                    return write_http_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        "{\"error\":\"Invalid model name\"}",
                    );
                };
                ctx.live_service.pause_inference();
                let res = ctx.live_service.evaluator().load_safetensors(&filename);
                ctx.live_service.resume_inference();
                if let Err(e) = res {
                    let msg = format!("{{\"error\":\"{}\"}}", escape_json(&e.to_string()));
                    return write_http_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &msg,
                    );
                }
                let iter = hub
                    .inner
                    .lock()
                    .expect("telemetry mutex poisoned")
                    .current_iteration;
                hub.set_phase(iter, format!("Loaded {filename} into Live"));
            }
            "delete" => {
                if let Some(filename) = sanitize_safetensors_filename(&name) {
                    let _ = std::fs::remove_file(&filename);
                }
            }
            _ => {}
        }

        let updated = hub.to_json();
        return write_http_response(&mut stream, "200 OK", "application/json", &updated);
    }

    if method == "POST" && raw_path.starts_with("/api/match") {
        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
        let cfg_a = parse_engine_config(body, "a_");
        let cfg_b = parse_engine_config(body, "b_");
        let num_games = extract_json_f64(body, "num_games")
            .map(|v| v as usize)
            .unwrap_or(100);
        let num_workers = extract_json_f64(body, "num_workers")
            .map(|v| v as usize)
            .unwrap_or(4);
        let max_batch_size = extract_json_f64(body, "max_batch_size")
            .map(|v| v as usize)
            .unwrap_or(32);
        let seed = extract_json_f64(body, "seed")
            .map(|v| v as u64)
            .unwrap_or(2026);

        match hub.run_ttt_match(&cfg_a, &cfg_b, num_games, num_workers, max_batch_size, seed) {
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

    (TttGame::new(player, state), sims, trees, c_init, dir_eps, model)
}

fn parse_engine_config(body: &str, prefix: &str) -> TttEngineConfig {
    let mut cfg = TttEngineConfig::default();
    if let Some(v) = extract_json_str(body, &format!("{prefix}engine")) {
        cfg.engine = v;
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}depth")) {
        cfg.depth = (v as u32).clamp(0, 9);
    }
    if let Some(v) = extract_json_bool(body, &format!("{prefix}heuristic")) {
        cfg.use_heuristic = v;
    }
    if let Some(v) = extract_json_str(body, &format!("{prefix}budget_kind")) {
        cfg.budget_kind = v;
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}budget_val")) {
        cfg.budget_val = (v as usize).clamp(1, 1_000_000);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}arena_cap")) {
        cfg.arena_cap = (v as usize).clamp(64, 1_000_000);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}sims")) {
        cfg.sims = (v as usize).clamp(1, 10_000);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}trees")) {
        cfg.num_trees = (v as usize).clamp(1, 64);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}c_uct")) {
        cfg.c_uct = (v as f32).clamp(0.0, 10.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}opening")) {
        cfg.opening_moves = (v as usize).clamp(0, 9);
    }
    if let Some(v) = extract_json_str(body, &format!("{prefix}model")) {
        cfg.model = v;
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}c_init")) {
        cfg.c_init = (v as f32).clamp(0.05, 10.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}c_base")) {
        cfg.c_base = (v as f32).clamp(1.0, 1_000_000.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}dir_alpha")) {
        cfg.dir_alpha = (v as f32).clamp(0.01, 10.0);
    }
    if let Some(v) = extract_json_f64(body, &format!("{prefix}dir_eps")) {
        cfg.dir_eps = (v as f32).clamp(0.0, 1.0);
    }
    cfg
}

fn apply_knob_updates(knobs: &mut LiveKnobs, body: &str) {
    if let Some(v) = extract_json_bool(body, "paused") {
        knobs.paused = v;
    }
    if let Some(v) = extract_json_f64(body, "games_per_iteration") {
        knobs.games_per_iteration = (v as usize).clamp(8, 4096);
    }
    if let Some(v) = extract_json_f64(body, "simulations_per_move") {
        knobs.simulations_per_move = (v as usize).clamp(4, 2000);
    }
    if let Some(v) = extract_json_f64(body, "train_steps") {
        knobs.train_steps = (v as usize).clamp(1, 1000);
    }
    if let Some(v) = extract_json_f64(body, "batch_size") {
        knobs.batch_size = (v as usize).clamp(8, 1024);
    }
    if let Some(v) = extract_json_f64(body, "learning_rate") {
        knobs.learning_rate = v.clamp(1e-6, 1.0);
    }
    if let Some(v) = extract_json_f64(body, "dirichlet_epsilon") {
        knobs.dirichlet_epsilon = (v as f32).clamp(0.0, 1.0);
    }
    if let Some(v) = extract_json_f64(body, "c_init") {
        knobs.c_init = (v as f32).clamp(0.1, 10.0);
    }
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let pat = format!("\"{key}\"");
    let pos = json.find(&pat)?;
    let rest = json[pos + pat.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let pos = json.find(&pat)?;
    let rest = json[pos + pat.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| {
            !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E'
        })
        .unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let pos = json.find(&pat)?;
    let rest = json[pos + pat.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
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
<title>Pure-Rust AlphaZero Lab, Multi-Engine Arena & Inspector</title>
<style>
  :root {
    --bg: #0b0f19;
    --panel: #131b2e;
    --panel-border: #23304d;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: var(--bg);
    color: var(--text);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: #0f1629;
    border-bottom: 1px solid var(--panel-border);
  }
  .brand { font-size: 18px; font-weight: 700; color: var(--accent); }
  .badges { display: flex; gap: 12px; align-items: center; font-size: 13px; }
  .badge {
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--panel);
    border: 1px solid var(--panel-border);
  }
  .tabs {
    display: flex;
    gap: 8px;
    padding: 12px 24px 0;
    border-bottom: 1px solid var(--panel-border);
  }
  .tab-btn {
    background: transparent;
    color: var(--muted);
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px 16px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
  }
  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .container { padding: 20px 24px; max-width: 1500px; margin: 0 auto; }
  .grid-4 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 18px;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 16px;
    margin-bottom: 18px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 16px;
  }
  .card h3 {
    margin: 0 0 8px 0;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }
  .metric-val { font-size: 26px; font-weight: 700; }
  .metric-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  canvas { width: 100%; height: 220px; display: block; }
  .knobs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 12px;
    align-items: end;
  }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  input[type="number"], input[type="text"], select {
    width: 100%;
    padding: 7px 10px;
    background: #0b0f19;
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    color: var(--text);
    font-family: inherit;
  }
  button.btn {
    padding: 8px 14px;
    border-radius: 6px;
    border: 1px solid var(--accent);
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent);
    font-weight: 600;
    cursor: pointer;
  }
  button.btn:hover { background: rgba(56, 189, 248, 0.28); }
  button.btn-green {
    border-color: var(--green);
    background: rgba(34, 197, 94, 0.16);
    color: var(--green);
  }
  button.btn-green:hover { background: rgba(34, 197, 94, 0.3); }
  button.btn-red {
    border-color: var(--red);
    background: rgba(239, 68, 68, 0.14);
    color: var(--red);
    padding: 5px 10px;
    font-size: 12px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td {
    padding: 8px 10px;
    text-align: right;
    border-bottom: 1px solid var(--panel-border);
  }
  th { color: var(--muted); font-weight: 600; }
  td.left, th.left { text-align: left; }
  /* Play & Inspector Tab */
  .play-layout {
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 24px;
  }
  .board {
    display: grid;
    grid-template-columns: repeat(3, 120px);
    grid-template-rows: repeat(3, 120px);
    gap: 10px;
    margin: 16px 0;
  }
  .cell {
    background: #0b0f19;
    border: 2px solid var(--panel-border);
    border-radius: 10px;
    position: relative;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 6px;
    transition: border-color 0.15s;
  }
  .cell:hover { border-color: var(--accent); }
  .cell.best-choice { border-color: var(--green); box-shadow: 0 0 12px rgba(34,197,94,0.25); }
  .sym { font-size: 42px; font-weight: 800; line-height: 1; }
  .sym.x { color: var(--accent); }
  .sym.o { color: var(--amber); }
  .overlay-stats {
    font-size: 11px;
    line-height: 1.35;
    width: 100%;
    text-align: left;
    color: var(--muted);
  }
  .bar-bg {
    width: 100%;
    height: 5px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
    margin: 3px 0;
  }
  .bar-fill { height: 100%; background: var(--green); }
  .wdl-bar {
    display: flex;
    height: 22px;
    border-radius: 6px;
    overflow: hidden;
    margin: 10px 0;
    font-size: 12px;
    font-weight: 700;
  }
  .wdl-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0b0f19;
  }
  .hidden { display: none; }
</style>
</head>
<body>
<header>
  <div class="brand">⚡ Pure-Rust AlphaZero Lab, Multi-Engine Arena & Inspector</div>
  <div class="badges">
    <span class="badge" id="device-badge">Device: ...</span>
    <span class="badge" id="phase-badge">Phase: ...</span>
    <span class="badge" id="iter-badge">Iteration: 0 / 0</span>
  </div>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('lab', this)">📊 Live Training & System Lab</button>
  <button class="tab-btn" onclick="switchTab('arena', this)">⚔️ Multi-Match Arena & NN Models</button>
  <button class="tab-btn" onclick="switchTab('play', this)">🎮 Interactive Play & MCTS/NN Inspector</button>
</div>

<!-- TAB 1: LIVE TRAINING & SYSTEM LAB -->
<div class="container" id="tab-lab">
  <div class="grid-4">
    <div class="card">
      <h3>5. System Batch Fill & Dedup</h3>
      <div class="metric-val" id="m-batch">--</div>
      <div class="metric-sub" id="m-batch-sub">Unique batch & deduplication savings</div>
    </div>
    <div class="card">
      <h3>Search & NN Throughput</h3>
      <div class="metric-val" id="m-speed">--</div>
      <div class="metric-sub" id="m-speed-sub">Latency: -- µs / state</div>
    </div>
    <div class="card">
      <h3>2 & 3. Policy KL & Root Entropy</h3>
      <div class="metric-val" id="m-kl">--</div>
      <div class="metric-sub" id="m-entropy-sub">Root Visit Entropy H(π): -- nats</div>
    </div>
    <div class="card">
      <h3>1. Arena vs Optimal Minimax(d=5)</h3>
      <div class="metric-val" id="m-arena">--</div>
      <div class="metric-sub" id="m-arena-sub">vs Random: -- | vs MM(d=2): --</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>1. Arena Win + Draw Rate (%) vs Fixed Baselines (100% = Never Loses)</h3>
      <canvas id="chart-arena" width="600" height="220"></canvas>
    </div>
    <div class="card">
      <h3>2 & 3. Policy KL-Divergence KL(π||p) [Green→0] & Root Entropy H(π) [Blue]</h3>
      <canvas id="chart-policy" width="600" height="220"></canvas>
    </div>
    <div class="card">
      <h3>4. Value MSE Loss [Amber] & Total Loss [Purple]</h3>
      <canvas id="chart-loss" width="600" height="220"></canvas>
    </div>
    <div class="card">
      <h3>5. System Pipeline: Logical Batch [Cyan] vs Unique NN Batch [Green]</h3>
      <canvas id="chart-system" width="600" height="220"></canvas>
    </div>
  </div>

  <div class="card" style="margin-bottom: 18px;">
    <h3>🎛️ Live Training Knobs (Applied on Next Iteration)</h3>
    <div class="knobs-grid">
      <div>
        <label>Games / Iteration</label>
        <input type="number" id="k-games" value="128" min="8" max="2048" step="16">
      </div>
      <div>
        <label>Simulations / Move</label>
        <input type="number" id="k-sims" value="64" min="8" max="1000" step="8">
      </div>
      <div>
        <label>SGD Steps / Iter</label>
        <input type="number" id="k-steps" value="60" min="5" max="500" step="5">
      </div>
      <div>
        <label>SGD Batch Size</label>
        <input type="number" id="k-batch" value="64" min="16" max="512" step="16">
      </div>
      <div>
        <label>Learning Rate</label>
        <input type="number" id="k-lr" value="0.002" step="0.0005">
      </div>
      <div>
        <label>Dirichlet ε</label>
        <input type="number" id="k-eps" value="0.25" min="0" max="1" step="0.05">
      </div>
      <div>
        <label>PUCT c_init</label>
        <input type="number" id="k-cinit" value="1.25" min="0.1" max="5" step="0.05">
      </div>
      <div style="display:flex; gap:8px;">
        <button class="btn" onclick="submitKnobs(null)">Apply Knobs</button>
        <button class="btn" id="pause-btn" onclick="togglePause()">Pause</button>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>📋 Iteration Telemetry Log</h3>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Iter</th>
            <th>Total Loss</th>
            <th>Policy CE</th>
            <th>Policy KL</th>
            <th>Value MSE</th>
            <th>Root H(π)</th>
            <th>Avg Moves</th>
            <th>vs Random</th>
            <th>vs MM(d=2)</th>
            <th>vs MM(d=5)</th>
            <th>Batch (Log / Uniq)</th>
            <th>Dedup %</th>
            <th>Sims/s</th>
            <th>SP / SGD / Eval (ms)</th>
          </tr>
        </thead>
        <tbody id="log-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- TAB 2: MULTI-MATCH ARENA & NN MODEL MANAGER -->
<div class="container hidden" id="tab-arena">
  <!-- NN Model Manager -->
  <div class="card" style="margin-bottom: 18px;">
    <h3>🧠 Trained Neural Network Registry & Checkpoint Manager</h3>
    <div style="display:flex; gap:12px; align-items:end; flex-wrap:wrap; margin-bottom:14px;">
      <div style="min-width:260px;">
        <label>Save Current Live NN Checkpoint As (.safetensors)</label>
        <input type="text" id="model-save-name" value="ttt_checkpoint.safetensors" placeholder="e.g. ttt_iter15.safetensors">
      </div>
      <button class="btn btn-green" onclick="saveLiveModel()">💾 Save Live Model</button>
      <button class="btn" onclick="pollState()">🔄 Refresh Models</button>
      <span id="model-msg" style="font-size:13px; color:var(--accent);"></span>
    </div>
    <table>
      <thead>
        <tr>
          <th class="left">Model ID</th>
          <th class="left">Type</th>
          <th>File Size</th>
          <th class="left">Description</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="models-tbody"></tbody>
    </table>
  </div>

  <!-- Engine A vs Engine B Configurator -->
  <div class="grid-2">
    <!-- Engine A -->
    <div class="card">
      <h3 style="color:var(--green);">🟢 Player A Engine Configuration</h3>
      <div class="knobs-grid" style="margin-bottom:12px;">
        <div style="grid-column: span 2;">
          <label>Engine Type</label>
          <select id="a-engine" onchange="updateEngineFields('a')">
            <option value="alphazero" selected>AlphaZero (Neural Network + PUCT)</option>
            <option value="mcts">MCTS (Classical UCT + Random Playouts)</option>
            <option value="tree-minimax">Tree-Minimax (Persistent Tree + Iterative Deepening)</option>
            <option value="minimax">Minimax (Stateless Alpha-Beta)</option>
            <option value="random">Random (Uniform Baseline)</option>
          </select>
        </div>
      </div>

      <div id="a-params-az" class="knobs-grid">
        <div style="grid-column: span 2;">
          <label>Neural Network Model</label>
          <select id="a-model" class="model-select"></select>
        </div>
        <div>
          <label>Sims / Tree</label>
          <input type="number" id="a-az-sims" value="80" min="1" max="5000">
        </div>
        <div>
          <label>Root Trees (K)</label>
          <input type="number" id="a-az-trees" value="1" min="1" max="64">
        </div>
        <div>
          <label>PUCT c_init</label>
          <input type="number" id="a-c-init" value="1.25" step="0.05">
        </div>
        <div>
          <label>PUCT c_base</label>
          <input type="number" id="a-c-base" value="19652" step="500">
        </div>
        <div>
          <label>Dirichlet α</label>
          <input type="number" id="a-dir-alpha" value="0.30" step="0.05">
        </div>
        <div>
          <label>Dirichlet ε (Noise)</label>
          <input type="number" id="a-dir-eps" value="0.00" step="0.05">
        </div>
        <div>
          <label>Opening Plies</label>
          <input type="number" id="a-az-open" value="0" min="0" max="9">
        </div>
      </div>

      <div id="a-params-mcts" class="knobs-grid hidden">
        <div>
          <label>Sims / Tree</label>
          <input type="number" id="a-mcts-sims" value="200" min="1" max="10000">
        </div>
        <div>
          <label>Root Trees (K)</label>
          <input type="number" id="a-mcts-trees" value="1" min="1" max="64">
        </div>
        <div>
          <label>UCT Exploration c</label>
          <input type="number" id="a-c-uct" value="1.414" step="0.1">
        </div>
        <div>
          <label>Opening Plies</label>
          <input type="number" id="a-mcts-open" value="0" min="0" max="9">
        </div>
      </div>

      <div id="a-params-tmm" class="knobs-grid hidden">
        <div>
          <label>Budget Mode</label>
          <select id="a-budget-kind">
            <option value="depth" selected>Fixed Depth (0..9)</option>
            <option value="evals">Max Node Evaluations</option>
            <option value="ms">Max Time (ms / move)</option>
          </select>
        </div>
        <div>
          <label>Budget Value</label>
          <input type="number" id="a-budget-val" value="4" min="0" max="500000">
        </div>
        <div>
          <label>Leaf Heuristic</label>
          <select id="a-tmm-heur">
            <option value="true" selected>TttHeuristic (Line threat score)</option>
            <option value="false">ZeroHeuristic (Terminal only)</option>
          </select>
        </div>
        <div>
          <label>Arena Capacity</label>
          <input type="number" id="a-arena-cap" value="4096" min="64" max="500000">
        </div>
      </div>

      <div id="a-params-mm" class="knobs-grid hidden">
        <div>
          <label>Search Depth (0=1ply, 5=optimal)</label>
          <input type="number" id="a-depth" value="2" min="0" max="9">
        </div>
        <div>
          <label>Leaf Heuristic</label>
          <select id="a-mm-heur">
            <option value="true" selected>TttHeuristic (Line threat score)</option>
            <option value="false">ZeroHeuristic (Terminal only)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Engine B -->
    <div class="card">
      <h3 style="color:var(--amber);">🟠 Player B Engine Configuration</h3>
      <div class="knobs-grid" style="margin-bottom:12px;">
        <div style="grid-column: span 2;">
          <label>Engine Type</label>
          <select id="b-engine" onchange="updateEngineFields('b')">
            <option value="alphazero">AlphaZero (Neural Network + PUCT)</option>
            <option value="mcts" selected>MCTS (Classical UCT + Random Playouts)</option>
            <option value="tree-minimax">Tree-Minimax (Persistent Tree + Iterative Deepening)</option>
            <option value="minimax">Minimax (Stateless Alpha-Beta)</option>
            <option value="random">Random (Uniform Baseline)</option>
          </select>
        </div>
      </div>

      <div id="b-params-az" class="knobs-grid hidden">
        <div style="grid-column: span 2;">
          <label>Neural Network Model</label>
          <select id="b-model" class="model-select"></select>
        </div>
        <div>
          <label>Sims / Tree</label>
          <input type="number" id="b-az-sims" value="80" min="1" max="5000">
        </div>
        <div>
          <label>Root Trees (K)</label>
          <input type="number" id="b-az-trees" value="1" min="1" max="64">
        </div>
        <div>
          <label>PUCT c_init</label>
          <input type="number" id="b-c-init" value="1.25" step="0.05">
        </div>
        <div>
          <label>PUCT c_base</label>
          <input type="number" id="b-c-base" value="19652" step="500">
        </div>
        <div>
          <label>Dirichlet α</label>
          <input type="number" id="b-dir-alpha" value="0.30" step="0.05">
        </div>
        <div>
          <label>Dirichlet ε (Noise)</label>
          <input type="number" id="b-dir-eps" value="0.00" step="0.05">
        </div>
        <div>
          <label>Opening Plies</label>
          <input type="number" id="b-az-open" value="0" min="0" max="9">
        </div>
      </div>

      <div id="b-params-mcts" class="knobs-grid">
        <div>
          <label>Sims / Tree</label>
          <input type="number" id="b-mcts-sims" value="200" min="1" max="10000">
        </div>
        <div>
          <label>Root Trees (K)</label>
          <input type="number" id="b-mcts-trees" value="1" min="1" max="64">
        </div>
        <div>
          <label>UCT Exploration c</label>
          <input type="number" id="b-c-uct" value="1.414" step="0.1">
        </div>
        <div>
          <label>Opening Plies</label>
          <input type="number" id="b-mcts-open" value="0" min="0" max="9">
        </div>
      </div>

      <div id="b-params-tmm" class="knobs-grid hidden">
        <div>
          <label>Budget Mode</label>
          <select id="b-budget-kind">
            <option value="depth" selected>Fixed Depth (0..9)</option>
            <option value="evals">Max Node Evaluations</option>
            <option value="ms">Max Time (ms / move)</option>
          </select>
        </div>
        <div>
          <label>Budget Value</label>
          <input type="number" id="b-budget-val" value="4" min="0" max="500000">
        </div>
        <div>
          <label>Leaf Heuristic</label>
          <select id="b-tmm-heur">
            <option value="true" selected>TttHeuristic (Line threat score)</option>
            <option value="false">ZeroHeuristic (Terminal only)</option>
          </select>
        </div>
        <div>
          <label>Arena Capacity</label>
          <input type="number" id="b-arena-cap" value="4096" min="64" max="500000">
        </div>
      </div>

      <div id="b-params-mm" class="knobs-grid hidden">
        <div>
          <label>Search Depth (0=1ply, 5=optimal)</label>
          <input type="number" id="b-depth" value="2" min="0" max="9">
        </div>
        <div>
          <label>Leaf Heuristic</label>
          <select id="b-mm-heur">
            <option value="true" selected>TttHeuristic (Line threat score)</option>
            <option value="false">ZeroHeuristic (Terminal only)</option>
          </select>
        </div>
      </div>
    </div>
  </div>

  <!-- Match Execution Bar -->
  <div class="card" style="margin-bottom: 18px;">
    <h3>🏁 Multi-Match Execution Settings (Role-Swapped 50% X / 50% O)</h3>
    <div class="knobs-grid">
      <div>
        <label>Total Games</label>
        <input type="number" id="m-num-games" value="100" min="2" max="2000" step="10">
      </div>
      <div>
        <label>Parallel Worker Threads</label>
        <input type="number" id="m-workers" value="4" min="1" max="16">
      </div>
      <div>
        <label>NN Max Batch Size</label>
        <input type="number" id="m-batch-cap" value="32" min="1" max="256">
      </div>
      <div>
        <label>Random Seed</label>
        <input type="number" id="m-seed" value="2026" min="1">
      </div>
      <div>
        <button class="btn btn-green" id="run-match-btn" onclick="runArenaMatch()" style="width:100%;">⚔️ Start Multi-Match</button>
      </div>
    </div>

    <div id="match-latest-box" class="hidden" style="margin-top:16px; padding-top:14px; border-top:1px solid var(--panel-border);">
      <div style="display:flex; justify-content:space-between; font-size:14px; font-weight:700;">
        <span id="match-title-a" style="color:var(--green);">Player A</span>
        <span id="match-summary-mid" style="color:var(--accent);">--</span>
        <span id="match-title-b" style="color:var(--amber);">Player B</span>
      </div>
      <div class="wdl-bar" id="match-wdl-bar"></div>
      <div style="font-size:12px; color:var(--muted);" id="match-perf-sub"></div>
    </div>
  </div>

  <!-- Match History Table -->
  <div class="card">
    <h3>🏆 Multi-Match Benchmark History</h3>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th class="left">Player A</th>
            <th class="left">Player B</th>
            <th>Games</th>
            <th>A Wins</th>
            <th>Draws</th>
            <th>B Wins</th>
            <th>A Score %</th>
            <th>Avg Moves</th>
            <th>NN Batch (Log→Uniq)</th>
            <th>Dedup Saved</th>
            <th>Games/s (ms)</th>
          </tr>
        </thead>
        <tbody id="matches-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- TAB 3: INTERACTIVE PLAY & MCTS/NN INSPECTOR -->
<div class="container hidden" id="tab-play">
  <div class="play-layout">
    <div class="card">
      <h3>🎮 Interactive Board & Neural/MCTS Overlay</h3>
      <div style="display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap;">
        <button class="btn" onclick="resetBoard()">Reset Empty</button>
        <button class="btn" onclick="stepAiMove()">🤖 Play Best AI Move</button>
        <label style="display:flex; align-items:center; gap:6px; margin:0;">
          <input type="checkbox" id="auto-ai" checked> Auto-Reply as AI
        </label>
      </div>
      <div class="knobs-grid" style="margin-bottom:10px;">
        <div style="grid-column: span 2;">
          <label>Inspect Neural Network Model</label>
          <select id="insp-model" class="model-select" onchange="inspectBoard()"></select>
        </div>
        <div>
          <label>Total Sims</label>
          <input type="number" id="insp-sims" value="160" min="8" max="2000" step="16" onchange="inspectBoard()">
        </div>
        <div>
          <label>Root Trees (K)</label>
          <input type="number" id="insp-trees" value="1" min="1" max="32" onchange="inspectBoard()">
        </div>
        <div>
          <label>PUCT c_init</label>
          <input type="number" id="insp-cinit" value="1.25" step="0.05" onchange="inspectBoard()">
        </div>
        <div>
          <label>Dirichlet ε</label>
          <input type="number" id="insp-eps" value="0.00" min="0" max="1" step="0.05" onchange="inspectBoard()">
        </div>
      </div>
      <div class="board" id="ttt-board"></div>
      <div id="play-status" style="font-size:14px; font-weight:600; color:var(--accent);"></div>
    </div>

    <div class="card">
      <h3>🔍 Live Position Diagnostics (Raw Neural Network vs. MCTS Search)</h3>
      <p style="font-size:13px; color:var(--muted); margin-top:0;">
        Click any square on the left to make a move. Compare what the selected
        <strong>Neural Network</strong> predicts in a single forward pass against how
        <strong>PUCT MCTS</strong> refines the visit distribution and Q-values, and observe
        how intra-batch deduplication collapses identical leaf evaluations!
      </p>
      <div class="grid-4" style="margin-top:12px;">
        <div class="card">
          <h3>Raw NN Value V(s)</h3>
          <div class="metric-val" id="insp-nn-val">--</div>
          <div class="metric-sub">Current player's expected return [-1, +1]</div>
        </div>
        <div class="card">
          <h3>Recommended Move</h3>
          <div class="metric-val" id="insp-best-mv">--</div>
          <div class="metric-sub" id="insp-best-sub">Most visited root child</div>
        </div>
        <div class="card">
          <h3>NN Evaluations (Dedup)</h3>
          <div class="metric-val" id="insp-dedup-val">--</div>
          <div class="metric-sub" id="insp-dedup-sub">Unique NN evals / Logical requests</div>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Cell (Row, Col)</th>
            <th>Raw NN Prior P(s,a)</th>
            <th>MCTS Visits N(s,a)</th>
            <th>MCTS Policy π(a|s)</th>
            <th>Action Value Q(s,a)</th>
          </tr>
        </thead>
        <tbody id="insp-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
let latestState = null;
let knobsInitialized = false;
let board = ['.', '.', '.', '.', '.', '.', '.', '.', '.'];
let currentPlayer = 0; // 0 = X, 1 = O
let lastInspection = null;
let knownModelIds = '';

function switchTab(tab, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-lab').classList.toggle('hidden', tab !== 'lab');
  document.getElementById('tab-arena').classList.toggle('hidden', tab !== 'arena');
  document.getElementById('tab-play').classList.toggle('hidden', tab !== 'play');
  if (tab === 'play') inspectBoard();
}

async function pollState() {
  try {
    const res = await fetch('/api/state');
    latestState = await res.json();
    renderLab(latestState);
    renderModelsAndMatches(latestState);
  } catch (e) {}
}

function renderLab(st) {
  document.getElementById('device-badge').innerText = `Device: ${st.device_name}`;
  document.getElementById('phase-badge').innerText = `Phase: ${st.phase}`;
  document.getElementById('iter-badge').innerText = `Iteration: ${st.current_iteration} / ${st.total_iterations}`;
  document.getElementById('pause-btn').innerText = st.knobs.paused ? 'Resume' : 'Pause';

  if (!knobsInitialized && st.knobs) {
    knobsInitialized = true;
    document.getElementById('k-games').value = st.knobs.games_per_iteration;
    document.getElementById('k-sims').value = st.knobs.simulations_per_move;
    document.getElementById('k-steps').value = st.knobs.train_steps;
    document.getElementById('k-batch').value = st.knobs.batch_size;
    document.getElementById('k-lr').value = st.knobs.learning_rate;
    document.getElementById('k-eps').value = st.knobs.dirichlet_epsilon;
    document.getElementById('k-cinit').value = st.knobs.c_init;
  }

  const h = st.history;
  if (!h || h.length === 0) return;
  const last = h[h.length - 1];

  document.getElementById('m-batch').innerText =
    `${last.batch_size_mean.toFixed(1)} → ${last.unique_batch_mean.toFixed(1)} uniq`;
  document.getElementById('m-batch-sub').innerText =
    `Dedup Saved: ${last.dedup_saved_pct.toFixed(1)}% | Replay Buf: ${last.buffer_len}/${last.buffer_capacity}`;

  document.getElementById('m-speed').innerText = `${last.simulations_per_sec.toFixed(0)} sims/s`;
  document.getElementById('m-speed-sub').innerText = `NN Latency: ${last.inference_us_mean.toFixed(1)} µs / state`;

  document.getElementById('m-kl').innerText = `KL: ${last.policy_kl.toFixed(4)}`;
  document.getElementById('m-entropy-sub').innerText = `Root Entropy H(π): ${last.root_entropy_mean.toFixed(3)} nats | CE: ${last.policy_loss.toFixed(3)}`;

  const mm5Total = last.vs_mm5[0] + last.vs_mm5[1] + last.vs_mm5[2];
  const mm5NoLoss = mm5Total > 0 ? (((last.vs_mm5[0] + last.vs_mm5[1]) / mm5Total) * 100).toFixed(0) : '0';
  document.getElementById('m-arena').innerText = `${mm5NoLoss}% (${last.vs_mm5[0]}W-${last.vs_mm5[1]}D-${last.vs_mm5[2]}L)`;
  document.getElementById('m-arena-sub').innerText =
    `vs Random: ${last.vs_random[0]}W-${last.vs_random[1]}D-${last.vs_random[2]}L | vs MM2: ${last.vs_mm2[0]}W-${last.vs_mm2[1]}D-${last.vs_mm2[2]}L`;

  const tbody = document.getElementById('log-tbody');
  tbody.innerHTML = h.slice().reverse().map(r => `
    <tr>
      <td><strong>#${r.iteration}</strong></td>
      <td>${r.total_loss.toFixed(4)}</td>
      <td>${r.policy_loss.toFixed(4)}</td>
      <td style="color:var(--green)">${r.policy_kl.toFixed(4)}</td>
      <td style="color:var(--amber)">${r.value_loss.toFixed(4)}</td>
      <td>${r.root_entropy_mean.toFixed(3)}</td>
      <td>${r.game_moves_mean.toFixed(1)}</td>
      <td>${r.vs_random[0]}W-${r.vs_random[1]}D-${r.vs_random[2]}L</td>
      <td>${r.vs_mm2[0]}W-${r.vs_mm2[1]}D-${r.vs_mm2[2]}L</td>
      <td><strong>${r.vs_mm5[0]}W-${r.vs_mm5[1]}D-${r.vs_mm5[2]}L</strong></td>
      <td>${r.batch_size_mean.toFixed(1)} / ${r.unique_batch_mean.toFixed(1)}</td>
      <td style="color:var(--accent)">${r.dedup_saved_pct.toFixed(1)}%</td>
      <td>${r.simulations_per_sec.toFixed(0)}</td>
      <td>${r.self_play_ms.toFixed(0)} / ${r.train_ms.toFixed(0)} / ${r.eval_ms.toFixed(0)}</td>
    </tr>
  `).join('');

  drawLineChart('chart-arena', h.map(r => r.iteration), [
    { label: 'vs Random (W+D %)', color: '#22c55e', data: h.map(r => pctNoLoss(r.vs_random)) },
    { label: 'vs MM(d=2) (W+D %)', color: '#38bdf8', data: h.map(r => pctNoLoss(r.vs_mm2)) },
    { label: 'vs MM(d=5) (W+D %)', color: '#f59e0b', data: h.map(r => pctNoLoss(r.vs_mm5)) },
  ], 0, 100);

  drawLineChart('chart-policy', h.map(r => r.iteration), [
    { label: 'Policy KL(π||p)', color: '#22c55e', data: h.map(r => r.policy_kl) },
    { label: 'Root Entropy H(π)', color: '#38bdf8', data: h.map(r => r.root_entropy_mean) },
  ]);

  drawLineChart('chart-loss', h.map(r => r.iteration), [
    { label: 'Value MSE', color: '#f59e0b', data: h.map(r => r.value_loss) },
    { label: 'Total Loss', color: '#a855f7', data: h.map(r => r.total_loss) },
  ]);

  drawLineChart('chart-system', h.map(r => r.iteration), [
    { label: 'Logical Batch', color: '#38bdf8', data: h.map(r => r.batch_size_mean) },
    { label: 'Unique NN Batch', color: '#22c55e', data: h.map(r => r.unique_batch_mean) },
  ], 0);
}

function renderModelsAndMatches(st) {
  const models = st.models || [];
  const sig = models.map(m => `${m.id}:${m.size_bytes}`).join('|');
  if (sig !== knownModelIds) {
    knownModelIds = sig;
    document.querySelectorAll('.model-select').forEach(sel => {
      const prev = sel.value;
      sel.innerHTML = models.map(m =>
        `<option value="${m.id}">${m.id} (${m.kind})</option>`
      ).join('');
      if (models.some(m => m.id === prev)) sel.value = prev;
    });

    const mbody = document.getElementById('models-tbody');
    mbody.innerHTML = models.map(m => {
      const sz = m.size_bytes > 0 ? `${(m.size_bytes / 1024).toFixed(1)} KB` : 'In-Memory';
      const actions = m.kind === 'disk'
        ? `<button class="btn" style="padding:4px 10px; font-size:12px;" onclick="loadModelToLive('${m.id}')">⬆️ Load to Live</button>
           <button class="btn btn-red" onclick="deleteModel('${m.id}')">🗑️ Delete</button>`
        : `<span style="color:var(--muted)">Built-in</span>`;
      return `
        <tr>
          <td class="left"><strong>${m.id}</strong></td>
          <td class="left">${m.kind}</td>
          <td>${sz}</td>
          <td class="left" style="color:var(--muted)">${m.description}</td>
          <td>${actions}</td>
        </tr>
      `;
    }).join('');
  }

  const matches = st.matches || [];
  const mtbody = document.getElementById('matches-tbody');
  mtbody.innerHTML = matches.slice().reverse().map(m => `
    <tr>
      <td><strong>#${m.match_id}</strong></td>
      <td class="left" style="color:var(--green)">${m.engine_a}</td>
      <td class="left" style="color:var(--amber)">${m.engine_b}</td>
      <td>${m.num_games}</td>
      <td style="color:var(--green)"><strong>${m.wins_a}</strong></td>
      <td style="color:var(--accent)">${m.draws}</td>
      <td style="color:var(--amber)"><strong>${m.wins_b}</strong></td>
      <td><strong>${m.score_a_pct.toFixed(1)}%</strong></td>
      <td>${m.avg_moves.toFixed(1)}</td>
      <td>${m.nn_logical_batch > 0 ? `${m.nn_logical_batch.toFixed(1)} → ${m.nn_unique_batch.toFixed(1)}` : '—'}</td>
      <td style="color:var(--accent)">${m.nn_logical_batch > 0 ? `${m.nn_dedup_saved_pct.toFixed(1)}%` : '—'}</td>
      <td>${m.games_per_sec.toFixed(1)}/s (${m.duration_ms.toFixed(0)} ms)</td>
    </tr>
  `).join('');
}

async function saveLiveModel() {
  const name = document.getElementById('model-save-name').value;
  const res = await fetch('/api/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'save', name })
  });
  const data = await res.json();
  if (data.error) {
    document.getElementById('model-msg').innerText = `Error: ${data.error}`;
  } else {
    document.getElementById('model-msg').innerText = `Saved ${name}!`;
    knownModelIds = '';
    renderModelsAndMatches(data);
  }
}

async function loadModelToLive(name) {
  const res = await fetch('/api/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'load_to_live', name })
  });
  const data = await res.json();
  if (!data.error) {
    document.getElementById('model-msg').innerText = `Loaded ${name} into Live model!`;
    renderLab(data);
    inspectBoard();
  }
}

async function deleteModel(name) {
  const res = await fetch('/api/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'delete', name })
  });
  const data = await res.json();
  knownModelIds = '';
  renderModelsAndMatches(data);
}

function updateEngineFields(side) {
  const eng = document.getElementById(`${side}-engine`).value;
  document.getElementById(`${side}-params-az`).classList.toggle('hidden', eng !== 'alphazero');
  document.getElementById(`${side}-params-mcts`).classList.toggle('hidden', eng !== 'mcts');
  document.getElementById(`${side}-params-tmm`).classList.toggle('hidden', eng !== 'tree-minimax');
  document.getElementById(`${side}-params-mm`).classList.toggle('hidden', eng !== 'minimax');
}

function readSidePayload(side, out) {
  const eng = document.getElementById(`${side}-engine`).value;
  out[`${side}_engine`] = eng;
  if (eng === 'alphazero') {
    out[`${side}_model`] = document.getElementById(`${side}-model`).value || 'live';
    out[`${side}_sims`] = Number(document.getElementById(`${side}-az-sims`).value);
    out[`${side}_trees`] = Number(document.getElementById(`${side}-az-trees`).value);
    out[`${side}_c_init`] = Number(document.getElementById(`${side}-c-init`).value);
    out[`${side}_c_base`] = Number(document.getElementById(`${side}-c-base`).value);
    out[`${side}_dir_alpha`] = Number(document.getElementById(`${side}-dir-alpha`).value);
    out[`${side}_dir_eps`] = Number(document.getElementById(`${side}-dir-eps`).value);
    out[`${side}_opening`] = Number(document.getElementById(`${side}-az-open`).value);
  } else if (eng === 'mcts') {
    out[`${side}_sims`] = Number(document.getElementById(`${side}-mcts-sims`).value);
    out[`${side}_trees`] = Number(document.getElementById(`${side}-mcts-trees`).value);
    out[`${side}_c_uct`] = Number(document.getElementById(`${side}-c-uct`).value);
    out[`${side}_opening`] = Number(document.getElementById(`${side}-mcts-open`).value);
  } else if (eng === 'tree-minimax') {
    out[`${side}_budget_kind`] = document.getElementById(`${side}-budget-kind`).value;
    out[`${side}_budget_val`] = Number(document.getElementById(`${side}-budget-val`).value);
    out[`${side}_heuristic`] = document.getElementById(`${side}-tmm-heur`).value === 'true';
    out[`${side}_arena_cap`] = Number(document.getElementById(`${side}-arena-cap`).value);
  } else if (eng === 'minimax') {
    out[`${side}_depth`] = Number(document.getElementById(`${side}-depth`).value);
    out[`${side}_heuristic`] = document.getElementById(`${side}-mm-heur`).value === 'true';
  }
}

async function runArenaMatch() {
  const btn = document.getElementById('run-match-btn');
  btn.disabled = true;
  btn.innerText = '⏳ Running Multi-Match...';
  const payload = {
    num_games: Number(document.getElementById('m-num-games').value),
    num_workers: Number(document.getElementById('m-workers').value),
    max_batch_size: Number(document.getElementById('m-batch-cap').value),
    seed: Number(document.getElementById('m-seed').value),
  };
  readSidePayload('a', payload);
  readSidePayload('b', payload);

  try {
    const res = await fetch('/api/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const m = await res.json();
    if (!m.error) {
      showLatestMatchBanner(m);
      await pollState();
    }
  } finally {
    btn.disabled = false;
    btn.innerText = '⚔️ Start Multi-Match';
  }
}

function showLatestMatchBanner(m) {
  document.getElementById('match-latest-box').classList.remove('hidden');
  document.getElementById('match-title-a').innerText = `${m.engine_a}: ${m.wins_a}W`;
  document.getElementById('match-summary-mid').innerText =
    `Draws: ${m.draws} | Score A: ${m.score_a_pct.toFixed(1)}%`;
  document.getElementById('match-title-b').innerText = `${m.engine_b}: ${m.wins_b}W`;

  const tot = Math.max(m.num_games, 1);
  const waPct = (m.wins_a / tot) * 100;
  const dPct = (m.draws / tot) * 100;
  const wbPct = (m.wins_b / tot) * 100;

  document.getElementById('match-wdl-bar').innerHTML = `
    <div class="wdl-seg" style="width:${waPct}%; background:var(--green);">${waPct >= 8 ? `${waPct.toFixed(0)}% W` : ''}</div>
    <div class="wdl-seg" style="width:${dPct}%; background:var(--accent);">${dPct >= 8 ? `${dPct.toFixed(0)}% D` : ''}</div>
    <div class="wdl-seg" style="width:${wbPct}%; background:var(--amber);">${wbPct >= 8 ? `${wbPct.toFixed(0)}% L` : ''}</div>
  `;

  const nnStr = m.nn_logical_batch > 0
    ? ` | NN Batch: ${m.nn_logical_batch.toFixed(1)} logical → ${m.nn_unique_batch.toFixed(1)} unique (${m.nn_dedup_saved_pct.toFixed(1)}% dedup saved, ${m.nn_us_per_state.toFixed(1)} µs/state)`
    : '';
  document.getElementById('match-perf-sub').innerText =
    `Completed ${m.num_games} games in ${m.duration_ms.toFixed(1)} ms (${m.games_per_sec.toFixed(1)} games/sec) | Avg Game Length: ${m.avg_moves.toFixed(1)} moves${nnStr}`;
}

function pctNoLoss(arr) {
  const tot = arr[0] + arr[1] + arr[2];
  return tot > 0 ? ((arr[0] + arr[1]) / tot) * 100.0 : 0;
}

function drawLineChart(canvasId, xs, series, fixedMin = null, fixedMax = null) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 44, padR = 16, padT = 24, padB = 26;
  let allVals = [];
  series.forEach(s => allVals.push(...s.data));
  let minV = fixedMin !== null ? fixedMin : Math.min(...allVals, 0);
  let maxV = fixedMax !== null ? fixedMax : Math.max(...allVals, minV + 1e-3) * 1.08;
  if (maxV <= minV) maxV = minV + 1;

  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px monospace';
  for (let i = 0; i <= 4; i++) {
    const y = padT + ((H - padT - padB) * i) / 4;
    const val = maxV - ((maxV - minV) * i) / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(val.toFixed(val >= 10 ? 0 : 2), 4, y + 4);
  }

  let lx = padL;
  series.forEach(s => {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, 6, 10, 10);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(s.label, lx + 14, 15);
    lx += ctx.measureText(s.label).width + 32;
  });

  series.forEach(s => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    s.data.forEach((v, idx) => {
      const x = xs.length <= 1
        ? padL + (W - padL - padR) / 2
        : padL + ((W - padL - padR) * idx) / (xs.length - 1);
      const y = padT + (H - padT - padB) * (1 - (v - minV) / (maxV - minV));
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    s.data.forEach((v, idx) => {
      const x = xs.length <= 1
        ? padL + (W - padL - padR) / 2
        : padL + ((W - padL - padR) * idx) / (xs.length - 1);
      const y = padT + (H - padT - padB) * (1 - (v - minV) / (maxV - minV));
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(x, y, 3.2, 0, Math.PI * 2); ctx.fill();
    });
  });
}

async function submitKnobs(pauseOverride) {
  const payload = {
    paused: pauseOverride !== null ? pauseOverride : (latestState ? latestState.knobs.paused : false),
    games_per_iteration: Number(document.getElementById('k-games').value),
    simulations_per_move: Number(document.getElementById('k-sims').value),
    train_steps: Number(document.getElementById('k-steps').value),
    batch_size: Number(document.getElementById('k-batch').value),
    learning_rate: Number(document.getElementById('k-lr').value),
    dirichlet_epsilon: Number(document.getElementById('k-eps').value),
    c_init: Number(document.getElementById('k-cinit').value),
  };
  const res = await fetch('/api/knobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  latestState = await res.json();
  renderLab(latestState);
}

function togglePause() {
  const nextPaused = latestState ? !latestState.knobs.paused : true;
  submitKnobs(nextPaused);
}

// Interactive Play & Inspector
function resetBoard() {
  board = ['.', '.', '.', '.', '.', '.', '.', '.', '.'];
  currentPlayer = 0;
  inspectBoard();
}

async function inspectBoard() {
  const bStr = board.join('');
  const sims = document.getElementById('insp-sims').value || 160;
  const trees = document.getElementById('insp-trees').value || 1;
  const cinit = document.getElementById('insp-cinit').value || 1.25;
  const eps = document.getElementById('insp-eps').value || 0.0;
  const model = document.getElementById('insp-model').value || 'live';
  try {
    const res = await fetch(`/api/inspect_ttt?board=${bStr}&player=${currentPlayer}&sims=${sims}&trees=${trees}&c_init=${cinit}&eps=${eps}&model=${encodeURIComponent(model)}`);
    lastInspection = await res.json();
    renderBoard();
  } catch (e) {}
}

function renderBoard() {
  const el = document.getElementById('ttt-board');
  const insp = lastInspection;
  const totalVisits = insp ? insp.mcts_visits.reduce((a, b) => a + b, 0) : 0;

  el.innerHTML = board.map((cell, idx) => {
    const isBest = insp && insp.chosen_move === idx;
    if (cell === 'X' || cell === 'O') {
      return `<div class="cell"><div class="sym ${cell.toLowerCase()}">${cell}</div></div>`;
    }
    if (!insp || !insp.legal_mask[idx]) {
      return `<div class="cell"></div>`;
    }
    const priorPct = (insp.raw_nn_priors[idx] * 100).toFixed(1);
    const visits = insp.mcts_visits[idx];
    const visitPct = totalVisits > 0 ? ((visits / totalVisits) * 100).toFixed(1) : '0.0';
    const q = insp.mcts_q_values[idx];
    const qColor = q >= 0.05 ? 'var(--green)' : (q <= -0.05 ? 'var(--red)' : 'var(--text)');
    return `
      <div class="cell ${isBest ? 'best-choice' : ''}" onclick="playCell(${idx})">
        <div class="overlay-stats">
          <div><strong>π: ${visitPct}%</strong> (${visits})</div>
          <div class="bar-bg"><div class="bar-fill" style="width:${visitPct}%"></div></div>
          <div>NN P: ${priorPct}%</div>
          <div style="color:${qColor}">Q: ${q >= 0 ? '+' : ''}${q.toFixed(2)}</div>
        </div>
      </div>
    `;
  }).join('');

  if (!insp) return;
  document.getElementById('insp-nn-val').innerText =
    `${insp.raw_nn_value >= 0 ? '+' : ''}${insp.raw_nn_value.toFixed(3)}`;
  document.getElementById('insp-best-mv').innerText =
    insp.chosen_move !== null ? `Cell ${insp.chosen_move} (${Math.floor(insp.chosen_move/3)}, ${insp.chosen_move%3})` : 'Terminal';

  const savedPct = insp.nn_requests > 0
    ? (((insp.nn_requests - insp.nn_unique_evals) / insp.nn_requests) * 100).toFixed(1)
    : '0.0';
  document.getElementById('insp-dedup-val').innerText =
    `${insp.nn_unique_evals} / ${insp.nn_requests}`;
  document.getElementById('insp-dedup-sub').innerText =
    `Intra-batch dedup saved ${savedPct}%`;

  const statusMap = {
    ongoing: `Turn: ${currentPlayer === 0 ? 'X (Player 1)' : 'O (Player 2)'}`,
    p1_win: '🏆 Player 1 (X) Wins!',
    p2_win: '🏆 Player 2 (O) Wins!',
    draw: '🤝 Draw Game!'
  };
  document.getElementById('play-status').innerText = statusMap[insp.status] || insp.status;

  const tbody = document.getElementById('insp-tbody');
  tbody.innerHTML = [0,1,2,3,4,5,6,7,8].map(i => {
    if (!insp.legal_mask[i]) {
      return `<tr style="opacity:0.4"><td>Cell ${i} (${Math.floor(i/3)}, ${i%3})</td><td colspan="4">Occupied (${board[i]})</td></tr>`;
    }
    const vPct = totalVisits > 0 ? ((insp.mcts_visits[i] / totalVisits) * 100).toFixed(1) : '0.0';
    return `
      <tr style="${insp.chosen_move === i ? 'color:var(--green); font-weight:700;' : ''}">
        <td>Cell ${i} (${Math.floor(i/3)}, ${i%3})</td>
        <td>${(insp.raw_nn_priors[i] * 100).toFixed(1)}%</td>
        <td>${insp.mcts_visits[i]}</td>
        <td>${vPct}%</td>
        <td>${insp.mcts_q_values[i] >= 0 ? '+' : ''}${insp.mcts_q_values[i].toFixed(3)}</td>
      </tr>
    `;
  }).join('');
}

async function playCell(idx) {
  if (!lastInspection || lastInspection.status !== 'ongoing' || board[idx] !== '.') return;
  board[idx] = currentPlayer === 0 ? 'X' : 'O';
  currentPlayer = 1 - currentPlayer;
  await inspectBoard();
  if (document.getElementById('auto-ai').checked && lastInspection && lastInspection.status === 'ongoing') {
    await stepAiMove();
  }
}

async function stepAiMove() {
  if (!lastInspection || lastInspection.status !== 'ongoing' || lastInspection.chosen_move === null) return;
  const mv = lastInspection.chosen_move;
  board[mv] = currentPlayer === 0 ? 'X' : 'O';
  currentPlayer = 1 - currentPlayer;
  await inspectBoard();
}

setInterval(pollState, 500);
pollState();
resetBoard();
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FnBatchEvaluator;
    use crate::nn::default_device;
    use crate::training::AlphaZeroTrainer;

    #[test]
    fn telemetry_hub_json_and_ttt_position_inspector_work() {
        let eval = FnBatchEvaluator(
            |_states: &[[f32; 27]], policies: &mut [[f32; 9]], values: &mut [[f32; 2]]| {
                for p in policies.iter_mut() {
                    *p = [0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.1, 0.0, 0.1];
                }
                for v in values.iter_mut() {
                    *v = [0.25, -0.25];
                }
            },
        );
        let service = InferenceService::new(eval, 16, 1);
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

        let insp = inspect_ttt_position(&game, &service, 64);
        assert_eq!(insp.status, "ongoing");
        assert!(insp.raw_nn_priors[4] > insp.raw_nn_priors[0]);
        assert!((insp.raw_nn_value - 0.25).abs() < 1e-5);
        // 8 trees x 8 simulations/tree = 64 total simulations (8 root expansions + 56 child visits).
        assert_eq!(insp.mcts_visits.iter().sum::<u64>(), 56);
        assert!(insp.chosen_move.is_some());
        assert!(insp.nn_requests >= insp.nn_unique_evals);

        let hub = TelemetryHub::new("CPU", 10, LiveKnobs::default());
        let json = hub.to_json();
        assert!(json.contains("\"device_name\":\"CPU\""));
    }

    #[test]
    fn arena_multi_match_between_all_ttt_engines_works() {
        let dev = default_device();
        let net_cfg = AlphaZeroNetConfig::tic_tac_toe_mlp();
        let trainer =
            AlphaZeroTrainer::<27, 9, 2>::new(net_cfg, 1e-3, 1e-4, dev.clone()).unwrap();
        let live_svc = Arc::new(
            trainer
                .build_inference_service(16, std::slice::from_ref(&dev))
                .unwrap(),
        );

        let hub = TelemetryHub::new("CPU", 5, LiveKnobs::default());
        hub.attach_ttt_service(live_svc, net_cfg, vec![dev]);

        // 1. AlphaZero (live) vs MCTS (classical UCT)
        let cfg_az = TttEngineConfig {
            engine: "alphazero".to_string(),
            model: "live".to_string(),
            sims: 16,
            num_trees: 2,
            ..Default::default()
        };
        let cfg_mcts = TttEngineConfig {
            engine: "mcts".to_string(),
            sims: 32,
            num_trees: 1,
            ..Default::default()
        };
        let rec1 = hub
            .run_ttt_match(&cfg_az, &cfg_mcts, 8, 2, 16, 42)
            .expect("AZ vs MCTS match must succeed");
        assert_eq!(rec1.wins_a + rec1.draws + rec1.wins_b, 8);
        assert!(rec1.nn_logical_batch >= 1.0);

        // 2. Tree-Minimax vs Minimax
        let cfg_tmm = TttEngineConfig {
            engine: "tree-minimax".to_string(),
            budget_kind: "depth".to_string(),
            budget_val: 3,
            use_heuristic: true,
            ..Default::default()
        };
        let cfg_mm = TttEngineConfig {
            engine: "minimax".to_string(),
            depth: 2,
            use_heuristic: true,
            ..Default::default()
        };
        let rec2 = hub
            .run_ttt_match(&cfg_tmm, &cfg_mm, 8, 2, 16, 99)
            .expect("TreeMinimax vs Minimax match must succeed");
        assert_eq!(rec2.wins_a + rec2.draws + rec2.wins_b, 8);
    }
}
