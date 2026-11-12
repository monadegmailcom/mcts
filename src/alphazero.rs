//! Unified `N`-player AlphaZero adapter and Game-Parallel Self-Play engine
//! (port & architectural evolution of C++ `alphazero.h`).
//!
//! ### Architectural Evolution from C++ `beat-it`
//! 1. **Game Parallelism Instead of Virtual-Loss Tree Parallelism**:
//!    In C++ `alphazero.h`, a single game spawned multiple threads that traversed the *same*
//!    MCTS tree concurrently using `std::atomic<int> visits`, `std::shared_mutex`, and a
//!    hardcoded `-1.0f` virtual loss penalty (`value_sum -= 1.0f` before evaluation, `+1.0f`
//!    after). That distorted PUCT selection away from the best path on every concurrent descent.
//!
//!    Here, each self-play worker multiplexes `M` (`concurrent_games_per_worker`) **independent,
//!    strictly single-threaded** [`MctsSearch`] state machines in waves:
//!    - Every simulation within a game sees the exact, undistorted `Q(s, a)` and `N(s, a)` of all
//!      prior simulations in that game (`0` virtual loss, `0` atomics, `20`-byte nodes).
//!    - Across the `M` concurrent games, all `NeedsEvaluation` leaves from the current simulation
//!      wave are submitted together in a single batched call to [`InferenceClient::evaluate_batch`],
//!      saturating GPU batch sizes with 100% mathematical purity.
//! 2. **Fixed Move-1 Root Dirichlet Noise**:
//!    In C++ `alphazero.h` (lines 586–593), `prepare_root()` expanded `root` while `visits == 0`
//!    and blended Dirichlet noise into `0.0` priors; then Simulation #1 saw `visits == 0`,
//!    re-expanded `root`, and overwrote the priors with un-noised softmax. Here, Dirichlet noise
//!    `(1 - eps) * P(s, a) + eps * Dir(alpha)` is always mixed *after* the root's neural network
//!    softmax priors `P(s, a)` have been evaluated.
//! 3. **Generalized to `const N: usize` Players & Curriculum Puzzles**:
//!    Supports `N = 1` single-player puzzles (e.g., 3x3x3 Rubik's Cube with curriculum scramble
//!    initializers), `N = 2` zero-sum games (Tic-Tac-Toe, Ultimate Tic-Tac-Toe), and `N >= 3`
//!    multi-player games via player-relative value vectors `[f32; N]`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::prelude::{Rng, SeedableRng, SmallRng};
use rand_distr::{Distribution, Gamma};

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::games::rubiks_cube::{self, RubiksCube};
use crate::games::tic_tac_toe::{self, Symbol as TttSymbol, TicTacToe};
use crate::games::twist_puzzle::{self, TwistPuzzle};
use crate::games::ultimate_ttt::{self, UltimateTicTacToe};
use crate::inference::{BatchEvaluator, InferenceClient, InferenceService};
use crate::mcts::{
    LeafEvaluator, LeafStep, MctsPayload, MctsSearch, Puct, assign_masked_softmax_priors,
};
use crate::statistics::Statistics;
use crate::tree::{Node, NodeId, TreeArena};

/// Trait bridging a [`GameRules<N>`] implementation to a neural network with flattened state
/// input size `G` and policy logit output size `P`.
pub trait AlphaZeroGame<const G: usize, const P: usize, const N: usize>:
    GameRules<N> + Sized
{
    /// Sentinel dummy move stored at the root node before any move has been played.
    const SENTINEL_MOVE: Self::Move;

    /// Maps a legal move to its corresponding index `0..P` in the policy logit vector.
    fn move_to_policy_index(mv: Self::Move) -> usize;

    /// Encodes `game` from `game.current_player()`'s perspective into a flattened `[f32; G]` tensor.
    fn encode_state(game: &Game<Self, N>, out: &mut [f32; G]);
}

impl AlphaZeroGame<27, 9, 2> for TicTacToe {
    const SENTINEL_MOVE: Self::Move = tic_tac_toe::NO_MOVE;

    #[inline]
    fn move_to_policy_index(mv: Self::Move) -> usize {
        mv as usize
    }

    /// Encodes 3 planes of `3x3 = 9` floats (`27` floats total; matches `games/tic_tac_toe.cpp:50-79`):
    /// - Plane 0 (`0..9`): `1.0` where `current_player` has a piece.
    /// - Plane 1 (`9..18`): `1.0` where the opponent has a piece.
    /// - Plane 2 (`18..27`): `1.0` if `current_player == PLAYER1`, else `0.0`.
    fn encode_state(game: &Game<Self, 2>, out: &mut [f32; 27]) {
        out.fill(0.0);
        let me = game.current_player();
        let my_sym = TttSymbol::from_player(me);
        let opp_sym = TttSymbol::from_player(me.next());

        for (idx, &cell) in game.state().iter().enumerate() {
            if cell == my_sym {
                out[idx] = 1.0;
            } else if cell == opp_sym {
                out[9 + idx] = 1.0;
            }
        }

        if me == PlayerIndex::<2>::PLAYER1 {
            out[18..27].fill(1.0);
        }
    }
}

impl AlphaZeroGame<{ ultimate_ttt::INPUT_SIZE }, { ultimate_ttt::POLICY_SIZE }, 2>
    for UltimateTicTacToe
{
    const SENTINEL_MOVE: Self::Move = ultimate_ttt::NO_MOVE;

    #[inline]
    fn move_to_policy_index(mv: Self::Move) -> usize {
        mv.to_policy_index()
    }

    #[inline]
    fn encode_state(game: &Game<Self, 2>, out: &mut [f32; ultimate_ttt::INPUT_SIZE]) {
        ultimate_ttt::encode_alphazero_planes(game, out);
    }
}

impl AlphaZeroGame<{ rubiks_cube::INPUT_SIZE }, { rubiks_cube::POLICY_SIZE }, 1> for RubiksCube {
    const SENTINEL_MOVE: Self::Move = rubiks_cube::NO_MOVE;

    #[inline]
    fn move_to_policy_index(mv: Self::Move) -> usize {
        mv as usize
    }

    #[inline]
    fn encode_state(game: &Game<Self, 1>, out: &mut [f32; rubiks_cube::INPUT_SIZE]) {
        rubiks_cube::encode_one_hot(game.state(), out);
    }
}

impl AlphaZeroGame<81, 8, 1> for TwistPuzzle {
    const SENTINEL_MOVE: Self::Move = twist_puzzle::NO_MOVE;

    #[inline]
    fn move_to_policy_index(mv: Self::Move) -> usize {
        mv as usize
    }

    #[inline]
    fn encode_state(game: &Game<Self, 1>, out: &mut [f32; 81]) {
        out.fill(0.0);
        for (pos, &tile) in game.state().board.iter().enumerate() {
            out[pos * 9 + (tile as usize)] = 1.0;
        }
    }
}

/// Converts a player-relative value vector `rel` (where `rel[0]` is the expected return for `actor`,
/// `rel[1]` for `(actor + 1) % N`, etc.) into absolute player indexing (`abs[p]` for `PlayerIndex(p)`).
#[inline]
pub fn relative_to_absolute_values<const N: usize>(
    actor: PlayerIndex<N>,
    rel: &[f32; N],
) -> [f32; N] {
    let mut abs = [0.0f32; N];
    let base = actor.index();
    for (offset, &val) in rel.iter().enumerate() {
        abs[(base + offset) % N] = val;
    }
    abs
}

/// Converts an absolute payoff vector `abs` (`abs[p]` for `PlayerIndex(p)`) into a player-relative
/// value vector from `actor`'s perspective (`rel[0] = abs[actor]`, `rel[1] = abs[(actor + 1) % N]`, ...).
#[inline]
pub fn absolute_to_relative_values<const N: usize>(
    actor: PlayerIndex<N>,
    abs: &[f32; N],
) -> [f32; N] {
    let mut rel = [0.0f32; N];
    let base = actor.index();
    for (offset, slot) in rel.iter_mut().enumerate() {
        *slot = abs[(base + offset) % N];
    }
    rel
}

/// Blends symmetric `Dirichlet(alpha)` exploration noise into the root node's child priors:
/// `P'(s_root, a) = (1 - epsilon) * P(s_root, a) + epsilon * noise_a`.
///
/// Must be called *after* the root's children have been expanded and assigned their neural network
/// softmax priors `P(s_root, a)`.
pub fn apply_root_dirichlet_noise<M: Copy>(
    arena: &mut TreeArena<M, MctsPayload>,
    alpha: f32,
    epsilon: f32,
    rng: &mut impl Rng,
) {
    if epsilon <= 0.0 || alpha <= 0.0 {
        return;
    }
    let children = arena.children_mut(NodeId::ROOT);
    if children.len() <= 1 {
        return;
    }

    let gamma = Gamma::new(alpha as f64, 1.0).expect("Dirichlet alpha must be > 0 and finite");
    let mut samples = Vec::with_capacity(children.len());
    let mut sum = 0.0f64;
    for _ in 0..children.len() {
        let s = gamma.sample(rng);
        samples.push(s);
        sum += s;
    }

    if sum <= f64::EPSILON {
        return;
    }

    let inv_sum = 1.0 / sum;
    for (child, &sample) in children.iter_mut().zip(samples.iter()) {
        let noise = (sample * inv_sum) as f32;
        child.payload.prior = (1.0 - epsilon) * child.payload.prior + epsilon * noise;
    }
}

/// Single-leaf [`LeafEvaluator`] adapter backed by a lock-free [`InferenceClient`].
///
/// Allows single-threaded [`crate::mcts::MctsPlayer`] and multi-threaded
/// [`crate::root_parallel::RootParallelPlayer`] to evaluate leaves through the shared batched
/// [`InferenceService`].
pub struct InferenceLeafEvaluator<'a, E, const G: usize, const P: usize, const N: usize>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    client: InferenceClient<'a, E, G, P, N>,
}

impl<'a, E, const G: usize, const P: usize, const N: usize> InferenceLeafEvaluator<'a, E, G, P, N>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    #[inline]
    pub fn new(service: &'a InferenceService<E, G, P, N>) -> Self {
        Self {
            client: service.client(16),
        }
    }
}

impl<R, E, const G: usize, const P: usize, const N: usize> LeafEvaluator<R, N>
    for InferenceLeafEvaluator<'_, E, G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    fn evaluate(&mut self, game: &Game<R, N>, children: &mut [Node<R::Move, MctsPayload>]) -> [f32; N] {
        let mut state_tensor = [0.0f32; G];
        R::encode_state(game, &mut state_tensor);
        let (policy_logits, rel_values) = self.client.evaluate_one(&state_tensor);
        assign_masked_softmax_priors(children, |mv| policy_logits[R::move_to_policy_index(mv)]);
        relative_to_absolute_values(game.current_player(), &rel_values)
    }
}

/// A single supervised training example `(state, policy_target, value_target)` produced by
/// AlphaZero self-play (generalized port of `alphazero::TrainingPosition<G, P>`).
#[derive(Clone, Debug)]
pub struct TrainingPosition<const G: usize, const P: usize, const N: usize> {
    /// Encoded game state `[f32; G]` from the acting player's perspective.
    pub state: [f32; G],
    /// Normalized MCTS root visit distribution `pi(a | s)` over all `0..P` actions.
    pub policy: [f32; P],
    /// Terminal game outcome `[f32; N]` rotated into the acting player's relative order
    /// (`value[0]` is the acting player's terminal payoff).
    pub value: [f32; N],
}

/// Hyperparameters for Game-Parallel AlphaZero Self-Play (combines `alphazero::params::Mcts`
/// and `alphazero::params::SelfPlay` from C++ `alphazero.h`).
#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    /// Total self-play games to complete across all worker threads.
    pub total_games: usize,
    /// Number of CPU self-play worker threads feeding the [`InferenceService`].
    pub num_workers: usize,
    /// Number of independent single-threaded games multiplexed concurrently per worker thread.
    /// Higher values (`16..128`) increase GPU inference batch sizes without any virtual loss!
    pub concurrent_games_per_worker: usize,
    /// Number of PUCT simulations executed per move in each game.
    pub simulations_per_move: usize,
    /// Number of initial moves per game sampled proportionally to root visit counts `N(s, a)`.
    pub opening_moves: usize,
    /// PUCT exploration constant `c_base` (default `19_652.0`).
    pub c_base: f32,
    /// PUCT exploration constant `c_init` (default `1.25`).
    pub c_init: f32,
    /// Symmetric Dirichlet parameter `alpha` for root exploration noise.
    pub dirichlet_alpha: f32,
    /// Weight `epsilon` of Dirichlet noise blended into root priors (default `0.25`; `0.0` disables).
    pub dirichlet_epsilon: f32,
    /// Node capacity per game's generational arena (`0` uses `simulations_per_move * 16`).
    pub arena_capacity: usize,
    /// Deterministic base seed.
    pub seed: u64,
}

impl SelfPlayConfig {
    pub const fn new(total_games: usize, simulations_per_move: usize, seed: u64) -> Self {
        Self {
            total_games,
            num_workers: 1,
            concurrent_games_per_worker: 32,
            simulations_per_move,
            opening_moves: 4,
            c_base: 19_652.0,
            c_init: 1.25,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            arena_capacity: 0,
            seed,
        }
    }

    pub const fn with_workers(
        mut self,
        num_workers: usize,
        concurrent_games_per_worker: usize,
    ) -> Self {
        self.num_workers = num_workers;
        self.concurrent_games_per_worker = concurrent_games_per_worker;
        self
    }

    pub const fn with_opening_moves(mut self, opening_moves: usize) -> Self {
        self.opening_moves = opening_moves;
        self
    }

    pub const fn with_dirichlet(mut self, alpha: f32, epsilon: f32) -> Self {
        self.dirichlet_alpha = alpha;
        self.dirichlet_epsilon = epsilon;
        self
    }

    pub const fn with_puct(mut self, c_base: f32, c_init: f32) -> Self {
        self.c_base = c_base;
        self.c_init = c_init;
        self
    }

    pub const fn with_arena_capacity(mut self, arena_capacity: usize) -> Self {
        self.arena_capacity = arena_capacity;
        self
    }

    #[inline]
    fn resolved_arena_capacity(&self) -> usize {
        if self.arena_capacity > 0 {
            self.arena_capacity
        } else {
            (self.simulations_per_move * 16).max(256)
        }
    }
}

/// Summary statistics and training dataset produced by a self-play run
/// (port of `alphazero::GameStats` + `SelfPlay::new_positions` in `alphazero.h`).
#[derive(Clone, Debug)]
pub struct SelfPlayResult<const G: usize, const P: usize, const N: usize> {
    pub positions: Vec<TrainingPosition<G, P, N>>,
    pub wins: [usize; N],
    pub draws: usize,
    pub total_games: usize,
    pub game_moves_stats: Statistics,
    pub root_entropy_stats: Statistics,
}

struct PendingTurn<const G: usize, const P: usize, const N: usize> {
    actor: PlayerIndex<N>,
    state: [f32; G],
    policy: [f32; P],
}

struct SelfPlaySlot<R: GameRules<N>, const G: usize, const P: usize, const N: usize> {
    search: MctsSearch<R, N, Puct>,
    sims_done: usize,
    move_count: usize,
    root_noise_applied: bool,
    active: bool,
    pending_turns: Vec<PendingTurn<G, P, N>>,
}

/// Runs Game-Parallel AlphaZero Self-Play starting from `initial_game` (rotating the starting
/// player `PlayerIndex::new((initial_game.current_player().raw() + game_idx) % N)` across games).
pub fn run_self_play<R, E, const G: usize, const P: usize, const N: usize>(
    initial_game: &Game<R, N>,
    service: &InferenceService<E, G, P, N>,
    config: &SelfPlayConfig,
) -> SelfPlayResult<G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    R::State: Sync,
    E: BatchEvaluator<G, P, N> + 'static,
{
    let base_state = initial_game.state();
    let base_player = initial_game.current_player().index();
    run_self_play_with_initializer(service, config, move |game_idx, _rng| {
        let start_player = PlayerIndex::<N>::new(((base_player + game_idx) % N) as u8);
        Game::<R, N>::new(start_player, base_state.clone())
    })
}

/// Runs Game-Parallel AlphaZero Self-Play using a custom `init_game(game_idx, &mut rng)` closure.
///
/// This enables both multi-player starting-seat rotation (`N >= 2`) and single-player (`N = 1`)
/// curriculum scramble generation (e.g., `RubiksState::scrambled(depth, budget, rng)`).
pub fn run_self_play_with_initializer<R, E, F, const G: usize, const P: usize, const N: usize>(
    service: &InferenceService<E, G, P, N>,
    config: &SelfPlayConfig,
    init_game: F,
) -> SelfPlayResult<G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
    F: Fn(usize, &mut SmallRng) -> Game<R, N> + Sync,
{
    assert!(
        config.simulations_per_move > 0,
        "simulations_per_move must be > 0"
    );
    let num_workers = config.num_workers.max(1);
    let concurrent_per_worker = config.concurrent_games_per_worker.max(1);
    let next_game_id = AtomicUsize::new(0);

    let shared_result = Mutex::new(SelfPlayResult {
        positions: Vec::new(),
        wins: [0; N],
        draws: 0,
        total_games: 0,
        game_moves_stats: Statistics::new(),
        root_entropy_stats: Statistics::new(),
    });

    std::thread::scope(|scope| {
        for worker_id in 0..num_workers {
            let svc = service;
            let cfg = *config;
            let init_ref = &init_game;
            let next_id_ref = &next_game_id;
            let result_ref = &shared_result;

            scope.spawn(move || {
                run_self_play_worker::<R, E, F, G, P, N>(
                    worker_id,
                    concurrent_per_worker,
                    svc,
                    &cfg,
                    init_ref,
                    next_id_ref,
                    result_ref,
                );
            });
        }
    });

    shared_result
        .into_inner()
        .expect("self_play result mutex poisoned")
}

#[allow(clippy::too_many_arguments)]
fn run_self_play_worker<R, E, F, const G: usize, const P: usize, const N: usize>(
    worker_id: usize,
    concurrent_games: usize,
    service: &InferenceService<E, G, P, N>,
    config: &SelfPlayConfig,
    init_game: &F,
    next_game_id: &AtomicUsize,
    shared_result: &Mutex<SelfPlayResult<G, P, N>>,
) where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
    F: Fn(usize, &mut SmallRng) -> Game<R, N> + Sync,
{
    let puct = Puct::new(config.c_base, config.c_init);
    let arena_cap = config.resolved_arena_capacity();
    let mut worker_rng = SmallRng::seed_from_u64(
        config
            .seed
            .wrapping_add((worker_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    );

    // Initialize up to `concurrent_games` active game slots for this worker.
    let mut slots: Vec<SelfPlaySlot<R, G, P, N>> = Vec::with_capacity(concurrent_games);
    for slot_idx in 0..concurrent_games {
        let game_idx = next_game_id.fetch_add(1, Ordering::Relaxed);
        if game_idx >= config.total_games {
            break;
        }
        let mut game = init_game(game_idx, &mut worker_rng);
        // Guard against a curriculum scramble that happened to start already terminal/solved.
        while !game.status().is_ongoing() {
            game = init_game(game_idx, &mut worker_rng);
        }
        let slot_seed = config
            .seed
            .wrapping_add(((worker_id * 4096 + slot_idx) as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        slots.push(SelfPlaySlot {
            search: MctsSearch::new(game, R::SENTINEL_MOVE, puct, arena_cap, slot_seed),
            sims_done: 0,
            move_count: 0,
            root_noise_applied: false,
            active: true,
            pending_turns: Vec::with_capacity(64),
        });
    }

    if slots.is_empty() {
        return;
    }

    let mut client = service.client(slots.len());
    let mut wave_meta: Vec<(usize, NodeId, PlayerIndex<N>)> = Vec::with_capacity(slots.len());
    let mut wave_states: Vec<[f32; G]> = vec![[0.0; G]; slots.len()];
    let mut wave_policies: Vec<[f32; P]> = vec![[0.0; P]; slots.len()];
    let mut wave_values: Vec<[f32; N]> = vec![[0.0; N]; slots.len()];

    let mut local_positions: Vec<TrainingPosition<G, P, N>> = Vec::new();
    let mut local_wins = [0usize; N];
    let mut local_draws = 0usize;
    let mut local_games = 0usize;
    let mut local_moves_stats = Statistics::new();
    let mut local_entropy_stats = Statistics::new();

    loop {
        wave_meta.clear();

        // Phase 1: Advance each active game slot until it either needs a neural network leaf
        // evaluation (joining the current wave) or completes all `simulations_per_move`.
        for (slot_idx, slot) in slots.iter_mut().enumerate() {
            if !slot.active {
                continue;
            }

            while slot.sims_done < config.simulations_per_move {
                // If the root already has evaluated children (e.g. from subtree reuse on move > 1),
                // mix Dirichlet noise into the root priors before selecting a child.
                if !slot.root_noise_applied
                    && config.dirichlet_epsilon > 0.0
                    && slot.search.tree().root().payload.visits > 0
                    && !slot.search.tree().active().children(NodeId::ROOT).is_empty()
                {
                    let (tree, rng) = slot.search.tree_and_rng_mut();
                    apply_root_dirichlet_noise(
                        tree.active_mut(),
                        config.dirichlet_alpha,
                        config.dirichlet_epsilon,
                        rng,
                    );
                    slot.root_noise_applied = true;
                }

                match slot.search.select_to_leaf() {
                    LeafStep::Terminal(payoffs) => {
                        slot.search.backpropagate(payoffs);
                        slot.sims_done += 1;
                    }
                    LeafStep::NeedsEvaluation { leaf_id, leaf_game } => {
                        let batch_idx = wave_meta.len();
                        R::encode_state(&leaf_game, &mut wave_states[batch_idx]);
                        wave_meta.push((slot_idx, leaf_id, leaf_game.current_player()));
                        break;
                    }
                }
            }
        }

        // Phase 2: Evaluate all collected leaf states from this wave in one batched call.
        let wave_len = wave_meta.len();
        if wave_len > 0 {
            client.evaluate_batch(
                &wave_states[..wave_len],
                &mut wave_policies[..wave_len],
                &mut wave_values[..wave_len],
            );

            for (i, &(slot_idx, leaf_id, actor)) in wave_meta.iter().enumerate() {
                let slot = &mut slots[slot_idx];
                let logits = &wave_policies[i];
                let rel_val = &wave_values[i];
                let abs_payoffs = relative_to_absolute_values(actor, rel_val);

                {
                    let children = slot.search.tree_mut().active_mut().children_mut(leaf_id);
                    assign_masked_softmax_priors(children, |mv| {
                        logits[R::move_to_policy_index(mv)]
                    });
                }

                // If `leaf_id == NodeId::ROOT`, this was the root's initial evaluation (Move 1 or
                // unexpanded subtree). Apply Dirichlet noise immediately *after* setting softmax
                // priors so subsequent simulations explore with perturbed root priors!
                if leaf_id == NodeId::ROOT
                    && !slot.root_noise_applied
                    && config.dirichlet_epsilon > 0.0
                {
                    let (tree, rng) = slot.search.tree_and_rng_mut();
                    apply_root_dirichlet_noise(
                        tree.active_mut(),
                        config.dirichlet_alpha,
                        config.dirichlet_epsilon,
                        rng,
                    );
                    slot.root_noise_applied = true;
                }

                slot.search.backpropagate(abs_payoffs);
                slot.sims_done += 1;
            }
        }

        // Phase 3: For any slot that has completed `simulations_per_move`, record its training
        // target, play the chosen move, and recycle the slot if the game reached a terminal state.
        let mut any_active = false;
        for slot in slots.iter_mut() {
            if !slot.active {
                continue;
            }
            if slot.sims_done < config.simulations_per_move {
                any_active = true;
                continue;
            }

            // Record search entropy and extract normalized visit-count policy target `pi(a | s)`.
            local_entropy_stats.update(slot.search.root_visit_entropy());
            let mut policy_target = [0.0f32; P];
            {
                let children = slot.search.tree().active().children(NodeId::ROOT);
                let total_visits: u64 = children.iter().map(|c| c.payload.visits as u64).sum();
                if total_visits > 0 {
                    let inv = 1.0 / (total_visits as f32);
                    for child in children {
                        let idx = R::move_to_policy_index(child.mv);
                        policy_target[idx] = (child.payload.visits as f32) * inv;
                    }
                }
            }

            let mut encoded_root = [0.0f32; G];
            R::encode_state(slot.search.root_game(), &mut encoded_root);
            slot.pending_turns.push(PendingTurn {
                actor: slot.search.root_game().current_player(),
                state: encoded_root,
                policy: policy_target,
            });

            // Select or sample move and advance the single-threaded search tree.
            slot.move_count += 1;
            let (chosen_id, chosen_mv) = if slot.move_count <= config.opening_moves {
                slot.search.sample_root_child_by_visits()
            } else {
                slot.search.best_root_child()
            };
            slot.search.advance_to_child(chosen_id, chosen_mv);
            slot.sims_done = 0;
            slot.root_noise_applied = false;

            // Check if the game has terminated.
            if let GameStatus::Terminal(final_abs_payoffs) = slot.search.root_game().status() {
                for turn in slot.pending_turns.drain(..) {
                    local_positions.push(TrainingPosition {
                        state: turn.state,
                        policy: turn.policy,
                        value: absolute_to_relative_values(turn.actor, &final_abs_payoffs),
                    });
                }
                local_moves_stats.update(slot.move_count as f32);
                local_games += 1;

                if let Some(winner_idx) = unique_winner_index(&final_abs_payoffs) {
                    local_wins[winner_idx] += 1;
                } else {
                    local_draws += 1;
                }

                // Recycle this slot for the next game if any remain.
                let next_idx = next_game_id.fetch_add(1, Ordering::Relaxed);
                if next_idx < config.total_games {
                    let mut new_game = init_game(next_idx, &mut worker_rng);
                    while !new_game.status().is_ongoing() {
                        new_game = init_game(next_idx, &mut worker_rng);
                    }
                    slot.search.reset(new_game, R::SENTINEL_MOVE);
                    slot.move_count = 0;
                    any_active = true;
                } else {
                    slot.active = false;
                }
            } else {
                any_active = true;
            }
        }

        if !any_active {
            break;
        }
    }

    let mut guard = shared_result
        .lock()
        .expect("self_play result mutex poisoned");
    guard.positions.extend(local_positions);
    for (dst, &src) in guard.wins.iter_mut().zip(local_wins.iter()) {
        *dst += src;
    }
    guard.draws += local_draws;
    guard.total_games += local_games;
    guard.game_moves_stats.join(&local_moves_stats);
    guard.root_entropy_stats.join(&local_entropy_stats);
}

#[inline]
fn unique_winner_index<const N: usize>(payoffs: &[f32; N]) -> Option<usize> {
    if N == 1 {
        return (payoffs[0] > 0.0).then_some(0);
    }
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0usize;
    let mut count = 0usize;
    for (idx, &v) in payoffs.iter().enumerate() {
        if v > best_val + 1e-6 {
            best_val = v;
            best_idx = idx;
            count = 1;
        } else if (v - best_val).abs() <= 1e-6 {
            count += 1;
        }
    }
    (count == 1).then_some(best_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::rubiks_cube::{RubiksGame, RubiksState};
    use crate::games::tic_tac_toe::{EMPTY_STATE, TttGame, TttHeuristic};
    use crate::inference::FnBatchEvaluator;

    #[test]
    fn move_1_dirichlet_noise_perturbs_root_priors_without_being_overwritten() {
        // Uniform model returning identical 0.0 logits for all 9 Tic-Tac-Toe moves.
        let uniform_eval = FnBatchEvaluator(
            |_states: &[[f32; 27]], policies: &mut [[f32; 9]], values: &mut [[f32; 2]]| {
                for p in policies.iter_mut() {
                    *p = [0.0; 9];
                }
                for v in values.iter_mut() {
                    *v = [0.0, 0.0];
                }
            },
        );
        let service = InferenceService::new(uniform_eval, 16, 1);
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

        // Run a single game with 32 simulations per move and epsilon = 0.25.
        let cfg = SelfPlayConfig::new(1, 32, 42)
            .with_workers(1, 1)
            .with_dirichlet(0.3, 0.25);
        let res = run_self_play(&game, &service, &cfg);
        assert_eq!(res.total_games, 1);
        assert!(!res.positions.is_empty());

        // Also directly test that after 1 simulation + Dirichlet noise, priors are non-uniform
        // and sum to 1.0 (catching the C++ Move-1 overwrite bug).
        let mut search = MctsSearch::new(game, tic_tac_toe::NO_MOVE, Puct::default(), 512, 99);
        if let LeafStep::NeedsEvaluation { leaf_id, .. } = search.select_to_leaf() {
            assert_eq!(leaf_id, NodeId::ROOT);
            let children = search.tree_mut().active_mut().children_mut(leaf_id);
            assign_masked_softmax_priors(children, |_mv| 0.0);
            let (tree, rng) = search.tree_and_rng_mut();
            apply_root_dirichlet_noise(tree.active_mut(), 0.3, 0.25, rng);
            search.backpropagate([0.0, 0.0]);
        } else {
            panic!("expected root to need evaluation on simulation 1");
        }

        // Run simulation 2: must NOT overwrite the root's Dirichlet-perturbed priors!
        let priors_after_sim1: Vec<f32> = search
            .tree()
            .active()
            .children(NodeId::ROOT)
            .iter()
            .map(|c| c.payload.prior)
            .collect();
        let sum: f32 = priors_after_sim1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Priors must not all be equal to 1/9 anymore:
        assert!(
            priors_after_sim1
                .iter()
                .any(|&p| (p - 1.0 / 9.0).abs() > 1e-3),
            "expected Dirichlet noise to perturb uniform priors, got {priors_after_sim1:?}"
        );

        let mut leaf_eval = InferenceLeafEvaluator::new(&service);
        search.simulate_once(&mut leaf_eval);
        let priors_after_sim2: Vec<f32> = search
            .tree()
            .active()
            .children(NodeId::ROOT)
            .iter()
            .map(|c| c.payload.prior)
            .collect();
        assert_eq!(priors_after_sim1, priors_after_sim2);
    }

    #[test]
    fn game_parallel_self_play_multiplexes_64_games_with_high_batch_utilization() {
        // Heuristic-backed mock NN for Tic-Tac-Toe.
        let evaluator = FnBatchEvaluator(
            |states: &[[f32; 27]], policies: &mut [[f32; 9]], values: &mut [[f32; 2]]| {
                for ((s, p), v) in states.iter().zip(policies.iter_mut()).zip(values.iter_mut()) {
                    // Slight center & corner logit preference.
                    *p = [0.2, 0.0, 0.2, 0.0, 0.5, 0.0, 0.2, 0.0, 0.2];
                    let my_pieces: f32 = s[0..9].iter().sum();
                    let opp_pieces: f32 = s[9..18].iter().sum();
                    let adv = ((my_pieces - opp_pieces) * 0.1).clamp(-0.9, 0.9);
                    *v = [adv, -adv];
                }
            },
        );

        let service = InferenceService::new(evaluator, 64, 1);
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let cfg = SelfPlayConfig::new(64, 24, 2026)
            .with_workers(2, 32)
            .with_opening_moves(2);

        let result = run_self_play(&game, &service, &cfg);
        assert_eq!(result.total_games, 64);
        assert_eq!(result.wins[0] + result.wins[1] + result.draws, 64);
        assert!(result.positions.len() >= 64 * 5);

        // Verify every generated training position has a normalized policy distribution and valid value.
        for pos in &result.positions {
            let pol_sum: f32 = pos.policy.iter().sum();
            assert!(
                (pol_sum - 1.0).abs() < 1e-4,
                "policy target must sum to 1.0, got {pol_sum}"
            );
            assert!(
                (pos.value[0] + pos.value[1]).abs() < 1e-5,
                "2-player zero-sum terminal values must sum to 0, got {:?}",
                pos.value
            );
        }

        // Verify Game Parallelism achieved large batch sizes without virtual loss!
        let batch_stats = service.batch_size_stats();
        assert!(
            batch_stats.mean() >= 16.0,
            "expected wave batching across 64 concurrent games to produce large batches, got {batch_stats}"
        );
        let _ = TttHeuristic;
    }

    #[test]
    fn single_player_rubiks_cube_curriculum_self_play_solves_short_scrambles() {
        // Untrained uniform-prior + sticker-count value estimator for 3x3x3 Rubik's Cube (N = 1).
        let cube_eval = FnBatchEvaluator(
            |states: &[[f32; 324]], policies: &mut [[f32; 12]], values: &mut [[f32; 1]]| {
                for ((s, p), v) in states.iter().zip(policies.iter_mut()).zip(values.iter_mut()) {
                    *p = [0.0; 12];
                    // Count how many stickers match their face color in the 324 one-hot encoding:
                    // For face `f in 0..6`, stickers `f*9 .. (f+1)*9` are solved when channel `f == 1.0`.
                    let mut solved_stickers = 0usize;
                    for face in 0..6 {
                        for idx in (face * 9)..((face + 1) * 9) {
                            if s[idx * 6 + face] > 0.5 {
                                solved_stickers += 1;
                            }
                        }
                    }
                    let score = ((solved_stickers as f32) / 54.0) * 2.0 - 1.0;
                    *v = [score];
                }
            },
        );

        let service = InferenceService::new(cube_eval, 32, 1);
        // 16 curriculum games at scramble depth 2 with step budget 2 and 200 simulations/move.
        let cfg = SelfPlayConfig::new(16, 200, 777)
            .with_workers(2, 8)
            .with_opening_moves(0)
            .with_dirichlet(0.3, 0.0);

        let result = run_self_play_with_initializer(&service, &cfg, |_idx, rng| {
            RubiksGame::new(PlayerIndex::P0, RubiksState::scrambled(2, 2, rng))
        });

        assert_eq!(result.total_games, 16);
        // With 48 sequential PUCT simulations + sticker heuristic, all depth-2 scrambles are solved!
        assert_eq!(
            result.wins[0], 16,
            "expected all 16 depth-2 Rubik's Cube scrambles to be solved, wins={:?}",
            result.wins
        );
        for pos in &result.positions {
            assert_eq!(pos.value, [1.0]);
        }
    }
}
