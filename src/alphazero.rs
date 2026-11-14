//! Unified `N`-player AlphaZero adapter and Game-Parallel Self-Play engine
//! (port & architectural evolution of C++ `alphazero.h`).
//!
//! ### Architectural Evolution from C++ `beat-it`
//! 1. **AlphaZero = MCTS + `Puct` + `NnEvaluator`**:
//!    Instead of duplicating MCTS tree traversal, backpropagation, and root Dirichlet noise inside
//!    a separate AlphaZero engine, [`NnEvaluator`] implements [`crate::mcts::LeafEvaluator`]
//!    (via [`LeafEvaluator::poll_evaluate`]). As a result:
//!    - Single-tree AlphaZero, Root-Parallel AlphaZero ([`AlphaZeroPlayer`]), and Self-Play
//!      ([`SelfPlaySlot`]) all run the exact same [`MctsSearch::poll_simulations`] core from
//!      [`crate::mcts`].
//!    - [`AlphaZeroPlayer`] is literally [`RootParallelPlayer<R, N, Puct, NnEvaluator>`]
//!      (`num_trees = K > 1` for single-game root parallelism; `num_trees = 1` for multi-match).
//! 2. **Unified Dynamic Game Multiplexing (`run_parallel_games` + `PollableSlot`)**:
//!    Both [`crate::matchup::run_multi_match`] and [`run_self_play_with_initializer`] share the
//!    same on-demand slot-scaling engine ([`crate::matchup::run_parallel_games`]) and implement
//!    [`crate::matchup::PollableSlot`] via named struct methods instead of large stateful closures.
//! 3. **Fixed Move-1 Root Dirichlet Noise**:
//!    In C++ `alphazero.h` (lines 586–593), `prepare_root()` expanded `root` while `visits == 0`
//!    and blended Dirichlet noise into `0.0` priors; then Simulation #1 saw `visits == 0`,
//!    re-expanded `root`, and overwrote the priors with un-noised softmax. Here, Dirichlet noise
//!    `(1 - eps) * P(s, a) + eps * Dir(alpha)` is always mixed inside [`MctsSearch::poll_simulations`]
//!    *after* the root's neural network softmax priors `P(s, a)` have been evaluated.
//! 4. **Generalized to `const N: usize` Players & Curriculum Puzzles**:
//!    Supports `N = 1` single-player puzzles (e.g., 3x3x3 Rubik's Cube with curriculum scramble
//!    initializers), `N = 2` zero-sum games (Tic-Tac-Toe, Ultimate Tic-Tac-Toe), and `N >= 3`
//!    multi-player games via player-relative value vectors `[f32; N]`.

use std::task::Poll;

use rand::prelude::{Rng, SeedableRng, SmallRng};
use rand_distr::{Distribution, Gamma};

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::games::rubiks_cube::{self, RubiksCube};
use crate::games::tic_tac_toe::{self, Symbol as TttSymbol, TicTacToe};
use crate::games::twist_puzzle::{self, TwistPuzzle};
use crate::games::ultimate_ttt::{self, UltimateTicTacToe};
use crate::inference::{BatchEvaluator, InferenceClient, InferenceService};
use crate::matchup::{PollableSlot, run_parallel_games};
use crate::mcts::{
    LeafEvaluator, MctsPayload, MctsSearch, Puct, assign_masked_softmax_priors,
};
use crate::player::Player;
use crate::root_parallel::{
    AggregatedRootChild, RootParallelBudget, RootParallelConfig, RootParallelPlayer,
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

/// Blends symmetric `Dirichlet(alpha)` exploration noise into a slice of root child nodes:
/// `P'(s_root, a) = (1 - epsilon) * P(s_root, a) + epsilon * noise_a`.
///
/// Must be called *after* the root's children have been expanded and assigned their neural network
/// softmax priors `P(s_root, a)`.
pub fn apply_dirichlet_noise_to_children<M: Copy>(
    children: &mut [Node<M, MctsPayload>],
    alpha: f32,
    epsilon: f32,
    rng: &mut impl Rng,
) {
    if epsilon <= 0.0 || alpha <= 0.0 || children.len() <= 1 {
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

/// Blends symmetric `Dirichlet(alpha)` exploration noise into the root node's child priors in `arena`.
#[inline]
pub fn apply_root_dirichlet_noise<M: Copy>(
    arena: &mut TreeArena<M, MctsPayload>,
    alpha: f32,
    epsilon: f32,
    rng: &mut impl Rng,
) {
    apply_dirichlet_noise_to_children(arena.children_mut(NodeId::ROOT), alpha, epsilon, rng);
}

/// Asynchronous neural-network [`LeafEvaluator`] that submits encoded leaf positions to a
/// batched [`InferenceService`], assigns masked softmax policy priors when the batch response
/// arrives, and blends root Dirichlet exploration noise via [`LeafEvaluator::prepare_root`].
///
/// Pairing [`MctsSearch`] with [`Puct`] and [`NnEvaluator`] turns standard MCTS into AlphaZero
/// without polluting classical MCTS configs with neural network or Dirichlet noise parameters.
pub struct NnEvaluator<'a, E, const G: usize, const P: usize, const N: usize>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    client: InferenceClient<'a, E, G, P, N>,
    rng: SmallRng,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    waiting: bool,
    unnotified: bool,
}

impl<'a, E, const G: usize, const P: usize, const N: usize> NnEvaluator<'a, E, G, P, N>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    #[inline]
    pub fn new(service: &'a InferenceService<E, G, P, N>, seed: u64) -> Self {
        Self {
            client: service.client(1),
            rng: SmallRng::seed_from_u64(seed),
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.0,
            waiting: false,
            unnotified: false,
        }
    }

    /// Configures root Dirichlet exploration noise `(alpha, epsilon)` blended into root child
    /// priors inside [`LeafEvaluator::prepare_root`].
    #[inline]
    pub const fn with_dirichlet(mut self, alpha: f32, epsilon: f32) -> Self {
        self.dirichlet_alpha = alpha;
        self.dirichlet_epsilon = epsilon;
        self
    }

    /// Wakes an idle background inference worker if this evaluator has enqueued an unnotified request.
    #[inline]
    pub fn notify(&mut self) {
        if self.unnotified {
            self.client.notify_service();
            self.unnotified = false;
        }
    }
}

impl<R, E, const G: usize, const P: usize, const N: usize> LeafEvaluator<R, N>
    for NnEvaluator<'_, E, G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    const IS_ASYNC: bool = true;

    fn poll_evaluate(
        &mut self,
        game: &Game<R, N>,
        children: &mut [Node<R::Move, MctsPayload>],
    ) -> Poll<[f32; N]> {
        if !self.waiting {
            let mut state = [0.0f32; G];
            R::encode_state(game, &mut state);
            self.client.submit(0, state);
            self.waiting = true;
            self.unnotified = true;
        }

        if let Some(resp) = self.client.try_pop() {
            self.waiting = false;
            assign_masked_softmax_priors(children, |mv| {
                resp.policy_logits[R::move_to_policy_index(mv)]
            });
            Poll::Ready(relative_to_absolute_values(
                game.current_player(),
                &resp.value,
            ))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn notify_pending(&mut self) {
        self.notify();
    }

    #[inline]
    fn prepare_root(&mut self, root_children: &mut [Node<R::Move, MctsPayload>]) {
        apply_dirichlet_noise_to_children(
            root_children,
            self.dirichlet_alpha,
            self.dirichlet_epsilon,
            &mut self.rng,
        );
    }

    fn evaluate(
        &mut self,
        game: &Game<R, N>,
        children: &mut [Node<R::Move, MctsPayload>],
    ) -> [f32; N] {
        loop {
            if let Poll::Ready(payoffs) = self.poll_evaluate(game, children) {
                return payoffs;
            }
            self.notify();
            self.client.wait_for_response();
        }
    }
}

/// Configuration for [`AlphaZeroPlayer`].
#[derive(Clone, Copy, Debug)]
pub struct AlphaZeroConfig {
    /// Number of independent root-parallel trees `K` multiplexed concurrently per move
    /// (`0` defaults to `service.max_batch_size()`, saturating a full GPU batch in a single game;
    /// `1` uses a single tree per game, ideal when `MultiMatch` multiplexes many games in parallel).
    pub num_trees: usize,
    /// Number of PUCT simulations executed per tree on each move
    /// (total simulations per move = `num_trees * simulations_per_tree`).
    pub simulations_per_tree: usize,
    /// Number of initial moves sampled proportionally to aggregated root visits (`0` = always argmax).
    pub opening_moves: usize,
    /// PUCT exploration constant `c_base` (default `19_652.0`).
    pub c_base: f32,
    /// PUCT exploration constant `c_init` (default `1.25`).
    pub c_init: f32,
    /// Symmetric Dirichlet parameter `alpha` for per-tree root exploration noise.
    pub dirichlet_alpha: f32,
    /// Weight `epsilon` of Dirichlet noise blended into each tree's root priors (`0.0` disables).
    pub dirichlet_epsilon: f32,
    /// Node capacity per tree's generational arena (`0` uses `(simulations_per_tree * 16).max(256)`).
    pub arena_capacity_per_tree: usize,
    /// Deterministic base seed.
    pub seed: u64,
}

impl AlphaZeroConfig {
    pub const fn new(num_trees: usize, simulations_per_tree: usize, seed: u64) -> Self {
        Self {
            num_trees,
            simulations_per_tree,
            opening_moves: 0,
            c_base: 19_652.0,
            c_init: 1.25,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.15,
            arena_capacity_per_tree: 0,
            seed,
        }
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

    pub const fn with_arena_capacity_per_tree(mut self, capacity: usize) -> Self {
        self.arena_capacity_per_tree = capacity;
        self
    }

    #[inline]
    fn resolved_arena_capacity(&self) -> usize {
        if self.arena_capacity_per_tree > 0 {
            self.arena_capacity_per_tree
        } else {
            (self.simulations_per_tree * 16).max(256)
        }
    }
}

/// Unified Single-Game and Multi-Match AlphaZero [`Player`] built by composing
/// [`RootParallelPlayer`] with [`Puct`] and [`NnEvaluator`].
///
/// - **In single-game play (`num_trees = K > 1`, Root Parallelism)**: Maintains `K` independent
///   [`MctsSearch`] trees rooted at the current position, steps all `K` trees cooperatively via
///   [`MctsSearch::poll_simulations`] so the GPU evaluates batches of up to `K` states at once,
///   sums the root visits across all `K` trees, and chooses the most visited move.
/// - **In multi-match evaluation ([`crate::matchup::run_multi_match`], `num_trees = 1` or `K`)**:
///   Implements [`Player::poll_choose_move`] so each worker thread can dynamically scale up to
///   `64` concurrent games and batch leaf evaluations across all active games automatically.
pub struct AlphaZeroPlayer<'a, R, E, const G: usize, const P: usize, const N: usize>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    inner: RootParallelPlayer<R, N, Puct, NnEvaluator<'a, E, G, P, N>>,
}

impl<'a, R, E, const G: usize, const P: usize, const N: usize> AlphaZeroPlayer<'a, R, E, G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    pub fn new(
        game: Game<R, N>,
        service: &'a InferenceService<E, G, P, N>,
        config: AlphaZeroConfig,
    ) -> Self {
        assert!(
            config.simulations_per_tree > 0,
            "simulations_per_tree must be > 0"
        );
        let num_trees = if config.num_trees > 0 {
            config.num_trees
        } else {
            service.max_batch_size()
        }
        .max(1);

        let rp_config = RootParallelConfig::new(
            RootParallelBudget::SimulationsPerThread(config.simulations_per_tree),
            config.seed,
        )
        .with_opening_moves(config.opening_moves)
        .with_arena_capacity_per_thread(config.resolved_arena_capacity());

        Self {
            inner: RootParallelPlayer::new(
                game,
                R::SENTINEL_MOVE,
                Puct::new(config.c_base, config.c_init),
                num_trees,
                rp_config,
                |_t, eval_seed| {
                    NnEvaluator::new(service, eval_seed)
                        .with_dirichlet(config.dirichlet_alpha, config.dirichlet_epsilon)
                },
            ),
        }
    }

    #[inline]
    pub fn num_trees(&self) -> usize {
        self.inner.num_threads()
    }

    /// Aggregates `(mv, visits, value_sum)` across all `K` root-parallel trees.
    #[inline]
    pub fn aggregated_root_children(&self) -> Vec<AggregatedRootChild<R::Move>> {
        self.inner.aggregated_root_children()
    }
}

impl<R, E, const G: usize, const P: usize, const N: usize> Player<R, N>
    for AlphaZeroPlayer<'_, R, E, G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    #[inline]
    fn poll_choose_move(&mut self) -> Poll<R::Move> {
        self.inner.poll_choose_move()
    }

    #[inline]
    fn notify_pending(&mut self) {
        self.inner.notify_pending();
    }

    #[inline]
    fn choose_move(&mut self) -> R::Move {
        self.inner.choose_move()
    }

    #[inline]
    fn apply_external_move(&mut self, mv: R::Move) {
        self.inner.apply_external_move(mv);
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
    /// Number of CPU self-play worker threads feeding the [`InferenceService`]
    /// (`0` automatically uses `std::thread::available_parallelism()`).
    pub num_workers: usize,
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
            num_workers: 0,
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

    pub const fn with_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
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

impl<const G: usize, const P: usize, const N: usize> SelfPlayResult<G, P, N> {
    fn empty() -> Self {
        Self {
            positions: Vec::new(),
            wins: [0; N],
            draws: 0,
            total_games: 0,
            game_moves_stats: Statistics::new(),
            root_entropy_stats: Statistics::new(),
        }
    }

    fn record_game(&mut self, completed: CompletedSelfPlayGame<G, P, N>) {
        self.positions.extend(completed.positions);
        if let Some(winner_idx) = unique_winner_index(&completed.payoffs) {
            self.wins[winner_idx] += 1;
        } else {
            self.draws += 1;
        }
        self.total_games += 1;
        self.game_moves_stats.update(completed.moves_played as f32);
        self.root_entropy_stats.join(&completed.root_entropy_stats);
    }

    fn merge(&mut self, other: Self) {
        self.positions.extend(other.positions);
        for p in 0..N {
            self.wins[p] += other.wins[p];
        }
        self.draws += other.draws;
        self.total_games += other.total_games;
        self.game_moves_stats.join(&other.game_moves_stats);
        self.root_entropy_stats.join(&other.root_entropy_stats);
    }
}

struct PendingTurn<const G: usize, const P: usize, const N: usize> {
    actor: PlayerIndex<N>,
    state: [f32; G],
    policy: [f32; P],
}

struct CompletedSelfPlayGame<const G: usize, const P: usize, const N: usize> {
    positions: Vec<TrainingPosition<G, P, N>>,
    payoffs: [f32; N],
    moves_played: usize,
    root_entropy_stats: Statistics,
}

/// A single self-play game slot pairing [`MctsSearch`] with [`NnEvaluator`] and recording
/// `(state, policy_target)` training examples at every turn.
struct SelfPlaySlot<'a, R, E, const G: usize, const P: usize, const N: usize>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    search: MctsSearch<R, N, Puct>,
    evaluator: NnEvaluator<'a, E, G, P, N>,
    simulations_per_move: usize,
    opening_moves: usize,
    move_count: usize,
    pending_turns: Vec<PendingTurn<G, P, N>>,
    root_entropy_stats: Statistics,
}

impl<'a, R, E, const G: usize, const P: usize, const N: usize> SelfPlaySlot<'a, R, E, G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    fn new(
        game: Game<R, N>,
        service: &'a InferenceService<E, G, P, N>,
        config: &SelfPlayConfig,
        slot_seed: u64,
    ) -> Self {
        let puct = Puct::new(config.c_base, config.c_init);
        let search = MctsSearch::new(
            game,
            R::SENTINEL_MOVE,
            puct,
            config.resolved_arena_capacity(),
            slot_seed,
        );
        let evaluator = NnEvaluator::new(service, slot_seed ^ 0xBF58_476D_1CE4_E5B9)
            .with_dirichlet(config.dirichlet_alpha, config.dirichlet_epsilon);

        Self {
            search,
            evaluator,
            simulations_per_move: config.simulations_per_move,
            opening_moves: config.opening_moves,
            move_count: 0,
            pending_turns: Vec::with_capacity(64),
            root_entropy_stats: Statistics::new(),
        }
    }

    /// Resets this slot for a new game, reusing its generational tree arena allocation in-place.
    fn reset(&mut self, new_game: Game<R, N>) {
        self.search.reset(new_game, R::SENTINEL_MOVE);
        self.move_count = 0;
        self.pending_turns.clear();
        self.root_entropy_stats.reset();
    }

    /// Records `(encoded_root_state, visit_distribution_policy)` for the current turn, chooses a
    /// move (proportional sampling during `opening_moves`, argmax thereafter), and advances the
    /// tree to the chosen child.
    fn record_turn_and_advance(&mut self) {
        self.root_entropy_stats
            .update(self.search.root_visit_entropy());

        let mut policy_target = [0.0f32; P];
        {
            let children = self.search.tree().active().children(NodeId::ROOT);
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
        R::encode_state(self.search.root_game(), &mut encoded_root);
        self.pending_turns.push(PendingTurn {
            actor: self.search.root_game().current_player(),
            state: encoded_root,
            policy: policy_target,
        });

        self.move_count += 1;
        let (chosen_id, chosen_mv) = if self.move_count <= self.opening_moves {
            self.search.sample_root_child_by_visits()
        } else {
            self.search.best_root_child()
        };
        self.search.advance_to_child(chosen_id, chosen_mv);
    }
}

impl<R, E, const G: usize, const P: usize, const N: usize> PollableSlot
    for SelfPlaySlot<'_, R, E, G, P, N>
where
    R: AlphaZeroGame<G, P, N>,
    E: BatchEvaluator<G, P, N> + 'static,
{
    type Output = CompletedSelfPlayGame<G, P, N>;

    fn poll_step(&mut self) -> Poll<Self::Output> {
        loop {
            if let GameStatus::Terminal(final_abs_payoffs) = self.search.root_game().status() {
                let positions = self
                    .pending_turns
                    .drain(..)
                    .map(|turn| TrainingPosition {
                        state: turn.state,
                        policy: turn.policy,
                        value: absolute_to_relative_values(turn.actor, &final_abs_payoffs),
                    })
                    .collect();
                return Poll::Ready(CompletedSelfPlayGame {
                    positions,
                    payoffs: final_abs_payoffs,
                    moves_played: self.move_count,
                    root_entropy_stats: self.root_entropy_stats,
                });
            }

            if self
                .search
                .poll_simulations(self.simulations_per_move, &mut self.evaluator)
                .is_pending()
            {
                return Poll::Pending;
            }

            self.record_turn_and_advance();
        }
    }

    #[inline]
    fn notify_pending(&mut self) {
        self.evaluator.notify();
    }
}

struct SelfPlayWorkerState<const G: usize, const P: usize, const N: usize> {
    worker_id: usize,
    rng: SmallRng,
    result: SelfPlayResult<G, P, N>,
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
/// Powered by [`run_parallel_games`] and [`SelfPlaySlot`]: dynamically scales active game slots
/// up to `max_slots_per_worker` on demand and reuses each slot's tree arena across games.
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
    if config.total_games == 0 {
        return SelfPlayResult::empty();
    }

    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let num_workers = if config.num_workers > 0 {
        config.num_workers
    } else {
        default_threads
    }
    .min(config.total_games);

    // Derive the per-worker concurrent game cap directly from the InferenceService's pipeline
    // capacity (`2 * max_batch_size * num_inference_workers`), bounding memory in the opening phase.
    let total_pipeline_cap = (service.max_batch_size() * service.num_workers() * 2)
        .min(config.total_games)
        .max(num_workers);
    let max_slots_per_worker = total_pipeline_cap.div_ceil(num_workers);

    let sample_ongoing_game = |game_idx: usize, rng: &mut SmallRng| -> Game<R, N> {
        let mut game = init_game(game_idx, rng);
        while !game.status().is_ongoing() {
            game = init_game(game_idx, rng);
        }
        game
    };

    let worker_states = run_parallel_games(
        config.total_games,
        num_workers,
        max_slots_per_worker,
        |worker_id| SelfPlayWorkerState {
            worker_id,
            rng: SmallRng::seed_from_u64(
                config
                    .seed
                    .wrapping_add((worker_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            ),
            result: SelfPlayResult::empty(),
        },
        |w, slot_idx, game_idx| {
            let game = sample_ongoing_game(game_idx, &mut w.rng);
            let slot_seed = config.seed.wrapping_add(
                ((w.worker_id * 4096 + slot_idx) as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9),
            );
            SelfPlaySlot::new(game, service, config, slot_seed)
        },
        |w, slot, game_idx| {
            let game = sample_ongoing_game(game_idx, &mut w.rng);
            slot.reset(game);
        },
        |w, _game_idx, completed| w.result.record_game(completed),
    );

    let mut total = SelfPlayResult::empty();
    for w in worker_states {
        total.merge(w.result);
    }
    total
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
    use crate::mcts::LeafStep;

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
            .with_workers(1)
            .with_dirichlet(0.3, 0.25);
        let res = run_self_play(&game, &service, &cfg);
        assert_eq!(res.total_games, 1);
        assert!(!res.positions.is_empty());

        // Also directly test that after 1 simulation + Dirichlet noise via `poll_simulations`,
        // priors are non-uniform and sum to 1.0 (catching the C++ Move-1 overwrite bug).
        let mut search = MctsSearch::new(game, tic_tac_toe::NO_MOVE, Puct::default(), 512, 99);
        let mut nn_eval = NnEvaluator::new(&service, 99).with_dirichlet(0.3, 0.25);
        search.run_simulations(1, &mut nn_eval);

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
        assert!(
            priors_after_sim1
                .iter()
                .any(|&p| (p - 1.0 / 9.0).abs() > 1e-3),
            "expected Dirichlet noise to perturb uniform priors, got {priors_after_sim1:?}"
        );

        if let LeafStep::NeedsEvaluation { leaf_id, .. } = search.select_to_leaf() {
            assert_ne!(leaf_id, NodeId::ROOT);
            let children = search.tree_mut().active_mut().children_mut(leaf_id);
            assign_masked_softmax_priors(children, |_mv| 0.0);
            search.backpropagate([0.0, 0.0]);
        }
        let priors_after_sim2: Vec<f32> = search
            .tree()
            .active()
            .children(NodeId::ROOT)
            .iter()
            .map(|c| c.payload.prior)
            .collect();
        assert_eq!(priors_after_sim1, priors_after_sim2);

        // Also verify single-game Root-Parallel `AlphaZeroPlayer` batches across its `K` trees
        // without needing single-state evaluation.
        service.reset_stats();
        let mut az_player = AlphaZeroPlayer::new(
            game,
            &service,
            AlphaZeroConfig::new(16, 10, 123),
        );
        let mv = az_player.choose_move();
        assert!((0..9).contains(&mv));
        assert!(
            service.batch_size_stats().mean() >= 8.0,
            "expected single-game AlphaZeroPlayer to batch across 16 root trees, got {}",
            service.batch_size_stats()
        );
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
            .with_workers(2)
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
            .with_workers(2)
            .with_opening_moves(0)
            .with_dirichlet(0.3, 0.0);

        let result = run_self_play_with_initializer(&service, &cfg, |_idx, rng| {
            RubiksGame::new(PlayerIndex::P0, RubiksState::scrambled(2, 2, rng))
        });

        assert_eq!(result.total_games, 16);
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
