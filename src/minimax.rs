//! Generic `N`-player Minimax (`Max^n` & Negamax Alpha-Beta) and iterative-deepening
//! `TreeMinimax` (port & generalization of C++ `minimax.h`, `minimax.cpp`, and `minimax-tree.h`).
//!
//! ### Architectural Evolution from C++ `beat-it`
//! - **Unified Sign Convention**: Every evaluator returns `[f32; N]` where higher is **always**
//!   better for player `p` (eliminating C++ `minimax.cpp`'s asymmetry where `Player1` minimized
//!   negative scores while `Player2` maximized positive scores).
//! - **Zero Indirect Calls in Inner Loop**: Replaces per-node `std::function<bool(double, double)>`
//!   and copied `std::function<double(Game const&)>` closures with monomorphized [`HeuristicEvaluator`]
//!   generics and branchless Negamax sign-flipping (`-negamax_ab(..., -beta, -alpha)`).
//! - **Generalized to `N` Players (`Max^n`)**: Automatically uses fast Negamax Alpha-Beta for
//!   2-player zero-sum games (`N == 2`) and exact `Max^n` search for `N = 1` single-player puzzles
//!   and `N >= 3` multi-player games.
//! - **12-Byte `TreeMinimax` Nodes & Slice Move Ordering**: Because [`Node`](crate::tree::Node) has
//!   no `parent` back-pointer, evaluated children can be stably sorted **in-place** via
//!   `children_mut(id)[..evaluated].sort_by(...)` without invalidating grandchild `NodeId` ranges—
//!   replacing C++ `boost::intrusive::list::splice` + linked-list merge-sort on ~280-byte nodes.
//!
//! ### Depth Convention (Matches C++ `beat-it` 1-to-1)
//! Following `beat-it`'s `minimax.h` / `minimax-tree.h`:
//! - `depth = 0` expands all legal root moves and evaluates the resulting child positions (1 ply).
//! - `depth = d` searches `d + 1` plies from the root (`depth = 2` is 3-ply lookahead).

use std::time::{Duration, Instant};

use rand::prelude::{SeedableRng, SliceRandom, SmallRng};

use crate::game::{Game, GameRules, GameStatus};
use crate::player::Player;
use crate::tree::{GenerationalTree, NodeId};

/// Base magnitude used to convert terminal payoffs (typically `[-1.0, +1.0]`) into search scores
/// that strictly dominate heuristic estimates while preferring faster wins and slower losses.
pub const TERMINAL_SCORE_BASE: f32 = 100_000.0;

#[inline]
fn terminal_score_for(payoff: f32, remaining_depth: u32) -> f32 {
    if payoff.abs() <= 1e-6 {
        0.0
    } else {
        payoff * (TERMINAL_SCORE_BASE + (remaining_depth as f32))
    }
}

#[inline]
fn terminal_vector_for<const N: usize>(payoffs: &[f32; N], remaining_depth: u32) -> [f32; N] {
    std::array::from_fn(|p| terminal_score_for(payoffs[p], remaining_depth))
}

/// Static position evaluator for non-terminal leaf states at the Minimax horizon.
pub trait HeuristicEvaluator<R: GameRules<N>, const N: usize> {
    /// Estimates a non-terminal game state as `[f32; N]` (`result[p]` is better when larger).
    fn evaluate_heuristic(&self, game: &Game<R, N>) -> [f32; N];
}

/// Default neutral heuristic returning `[0.0; N]` (matches C++ `minimax::Player::score` default,
/// used for games like Nim or pure terminal lookahead).
#[derive(Clone, Copy, Debug, Default)]
pub struct ZeroHeuristic;

impl<R: GameRules<N>, const N: usize> HeuristicEvaluator<R, N> for ZeroHeuristic {
    #[inline]
    fn evaluate_heuristic(&self, _game: &Game<R, N>) -> [f32; N] {
        [0.0; N]
    }
}

/// Closure adapter implementing [`HeuristicEvaluator`].
pub struct FnHeuristic<F>(pub F);

impl<R, const N: usize, F> HeuristicEvaluator<R, N> for FnHeuristic<F>
where
    R: GameRules<N>,
    F: Fn(&Game<R, N>) -> [f32; N],
{
    #[inline]
    fn evaluate_heuristic(&self, game: &Game<R, N>) -> [f32; N] {
        (self.0)(game)
    }
}

/// Stateless depth-limited Minimax agent (port & `N`-player generalization of C++ `minimax::Player`).
///
/// - When `N == 2` (and `zero_sum_pruning` is enabled, the default), runs Negamax with Alpha-Beta
///   pruning (propagating `alpha` right from the root move loop).
/// - For `N == 1` (single-player puzzles) or `N >= 3` (multi-player games), runs `Max^n` lookahead.
pub struct MinimaxPlayer<R: GameRules<N>, const N: usize, H = ZeroHeuristic> {
    game: Game<R, N>,
    heuristic: H,
    depth: u32,
    rng: SmallRng,
    move_buf: Vec<R::Move>,
    eval_calls: usize,
    last_score: f32,
    zero_sum_pruning: bool,
}

impl<R: GameRules<N>, const N: usize> MinimaxPlayer<R, N, ZeroHeuristic> {
    /// Creates a Minimax player with [`ZeroHeuristic`] (`[0.0; N]` at non-terminal leaves).
    pub fn zero(game: Game<R, N>, depth: u32, seed: u64) -> Self {
        Self::new(game, ZeroHeuristic, depth, seed)
    }
}

impl<R: GameRules<N>, const N: usize, H: HeuristicEvaluator<R, N>> MinimaxPlayer<R, N, H> {
    pub fn new(game: Game<R, N>, heuristic: H, depth: u32, seed: u64) -> Self {
        Self {
            game,
            heuristic,
            depth,
            rng: SmallRng::seed_from_u64(seed),
            move_buf: Vec::new(),
            eval_calls: 0,
            last_score: 0.0,
            zero_sum_pruning: N == 2,
        }
    }

    /// Enables or disables 2-player zero-sum Alpha-Beta pruning (enabled by default when `N == 2`;
    /// disable if evaluating a general-sum 2-player cooperative/asymmetric game via `Max^n`).
    pub fn with_zero_sum_pruning(mut self, enabled: bool) -> Self {
        assert!(!enabled || N == 2, "zero-sum Alpha-Beta pruning requires N == 2");
        self.zero_sum_pruning = enabled;
        self
    }

    #[inline]
    pub const fn game(&self) -> &Game<R, N> {
        &self.game
    }

    #[inline]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    #[inline]
    pub const fn eval_calls(&self) -> usize {
        self.eval_calls
    }

    #[inline]
    pub const fn last_score(&self) -> f32 {
        self.last_score
    }

    #[inline]
    pub const fn heuristic(&self) -> &H {
        &self.heuristic
    }
}

impl<R, const N: usize, H> Player<R, N> for MinimaxPlayer<R, N, H>
where
    R: GameRules<N>,
    H: HeuristicEvaluator<R, N>,
{
    fn choose_move(&mut self) -> R::Move {
        assert!(
            matches!(self.game.status(), GameStatus::Ongoing),
            "MinimaxPlayer::choose_move called on an already terminal game"
        );

        self.game.write_valid_moves(&mut self.move_buf);
        assert!(!self.move_buf.is_empty(), "MinimaxPlayer::choose_move found no valid moves");

        // Randomize tie-breaking among equally optimal moves (matches `std::ranges::shuffle` in C++).
        self.move_buf.shuffle(&mut self.rng);

        let actor = self.game.current_player();
        let actor_idx = actor.index();
        let mut best_move = self.move_buf[0];
        let mut best_score = f32::NEG_INFINITY;

        if N == 2 && self.zero_sum_pruning {
            for i in 0..self.move_buf.len() {
                let mv = self.move_buf[i];
                let next_game = self.game.apply(mv);
                let score = if next_game.current_player() == actor {
                    negamax_ab(
                        &next_game,
                        &self.heuristic,
                        self.depth,
                        best_score,
                        f32::INFINITY,
                        &mut self.eval_calls,
                    )
                } else {
                    -negamax_ab(
                        &next_game,
                        &self.heuristic,
                        self.depth,
                        f32::NEG_INFINITY,
                        -best_score,
                        &mut self.eval_calls,
                    )
                };

                if score > best_score {
                    best_score = score;
                    best_move = mv;
                }
            }
        } else {
            for i in 0..self.move_buf.len() {
                let mv = self.move_buf[i];
                let next_game = self.game.apply(mv);
                let vec_score = maxn_eval(
                    &next_game,
                    &self.heuristic,
                    self.depth,
                    &mut self.eval_calls,
                );
                let score = vec_score[actor_idx];
                if score > best_score {
                    best_score = score;
                    best_move = mv;
                }
            }
        }

        self.last_score = best_score;
        self.game = self.game.apply(best_move);
        best_move
    }

    fn apply_external_move(&mut self, mv: R::Move) {
        self.game = self.game.apply(mv);
    }
}

/// Recursive 2-player zero-sum Negamax with Alpha-Beta pruning.
///
/// Returns the minimax value from `game.current_player()`'s perspective.
pub fn negamax_ab<R, const N: usize, H>(
    game: &Game<R, N>,
    heuristic: &H,
    depth: u32,
    mut alpha: f32,
    beta: f32,
    calls: &mut usize,
) -> f32
where
    R: GameRules<N>,
    H: HeuristicEvaluator<R, N>,
{
    *calls += 1;
    let actor = game.current_player();
    let actor_idx = actor.index();

    if let GameStatus::Terminal(payoffs) = game.status() {
        return terminal_score_for(payoffs[actor_idx], depth);
    }
    if depth == 0 {
        return heuristic.evaluate_heuristic(game)[actor_idx];
    }

    let mut best = f32::NEG_INFINITY;
    for mv in game.valid_moves() {
        let next_game = game.apply(mv);
        let score = if next_game.current_player() == actor {
            negamax_ab(&next_game, heuristic, depth - 1, alpha, beta, calls)
        } else {
            -negamax_ab(&next_game, heuristic, depth - 1, -beta, -alpha, calls)
        };

        if score > best {
            best = score;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }

    best
}

/// Recursive `N`-player `Max^n` evaluation (`N = 1` single-player lookahead, `N >= 3` multi-player).
pub fn maxn_eval<R, const N: usize, H>(
    game: &Game<R, N>,
    heuristic: &H,
    depth: u32,
    calls: &mut usize,
) -> [f32; N]
where
    R: GameRules<N>,
    H: HeuristicEvaluator<R, N>,
{
    *calls += 1;

    if let GameStatus::Terminal(payoffs) = game.status() {
        return terminal_vector_for(&payoffs, depth);
    }
    if depth == 0 {
        return heuristic.evaluate_heuristic(game);
    }

    let actor_idx = game.current_player().index();
    let mut best_vec = [f32::NEG_INFINITY; N];

    for mv in game.valid_moves() {
        let next_game = game.apply(mv);
        let candidate = maxn_eval(&next_game, heuristic, depth - 1, calls);
        if candidate[actor_idx] > best_vec[actor_idx] {
            best_vec = candidate;
        }
    }

    best_vec
}

/// Stopping budget for iterative-deepening [`TreeMinimaxPlayer`].
///
/// Replaces C++ `minimax-tree.h`'s fixed `max_depth` loop with an algebraic sum type supporting
/// both fixed-horizon search and anytime dynamic-depth search capped by evaluation count or clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimaxBudget {
    /// Searches up to `max_depth + 1` plies from the root (matches [`MinimaxPlayer`] at
    /// `max_depth` 1-to-1, stopping early if the entire remaining game tree reaches terminal states).
    FixedDepth(u32),
    /// Dynamically deepens (`1, 2, 3, ...` plies) up to at most `max_evals` node evaluations per
    /// turn. Always completes the 1-ply root scan (`d = 1`) so every legal move is scored once.
    MaxEvaluations(usize),
    /// Dynamically deepens (`1, 2, 3, ...` plies) until `duration` elapses on this turn.
    /// Always completes the 1-ply root scan (`d = 1`) so a valid legal move is guaranteed.
    MaxDuration(Duration),
}

/// Zero-sized error sentinel used to unwind [`TreeMinimaxPlayer::eval_node`] cleanly via Rust's
/// `?` operator when [`MinimaxBudget::MaxEvaluations`] or [`MinimaxBudget::MaxDuration`] expires
/// mid-iteration—without C++ exception stack-unwinding or corrupting partial node orderings.
#[derive(Clone, Copy, Debug)]
struct SearchAborted;

struct TurnTracker {
    budget: MinimaxBudget,
    turn_evals: usize,
    deadline: Option<Instant>,
    allow_abort: bool,
    hit_horizon: bool,
}

impl TurnTracker {
    #[inline]
    fn new(budget: MinimaxBudget) -> Self {
        let deadline = match budget {
            MinimaxBudget::MaxDuration(dur) => Some(Instant::now() + dur),
            _ => None,
        };
        Self {
            budget,
            turn_evals: 0,
            deadline,
            allow_abort: false,
            hit_horizon: false,
        }
    }

    /// Checks whether the budget is already spent before starting a new depth iteration `d >= 2`.
    #[inline]
    fn is_exhausted(&self) -> bool {
        match self.budget {
            MinimaxBudget::FixedDepth(_) => false,
            MinimaxBudget::MaxEvaluations(limit) => self.turn_evals >= limit,
            MinimaxBudget::MaxDuration(_) => {
                self.deadline.is_some_and(|dl| Instant::now() >= dl)
            }
        }
    }

    /// Accounts for one node evaluation, returning `Err(SearchAborted)` if an active dynamic
    /// budget is exceeded.
    #[inline]
    fn step(&mut self) -> Result<(), SearchAborted> {
        if self.allow_abort {
            match self.budget {
                MinimaxBudget::FixedDepth(_) => {}
                MinimaxBudget::MaxEvaluations(limit) => {
                    if self.turn_evals >= limit {
                        return Err(SearchAborted);
                    }
                }
                MinimaxBudget::MaxDuration(_) => {
                    // Poll OS clock every 64 evaluations (~0.05 ns/node amortized overhead).
                    if (self.turn_evals & 63) == 0
                        && self.deadline.is_some_and(|dl| Instant::now() >= dl)
                    {
                        return Err(SearchAborted);
                    }
                }
            }
        }
        self.turn_evals += 1;
        Ok(())
    }
}

/// Per-node evaluation payload for [`TreeMinimaxPlayer`] (`4` bytes; total `Node<u8, _>` is **12 bytes**!).
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeMinimaxPayload {
    /// Cached evaluation from the **parent** node's acting-player perspective, enabling direct
    /// descending move ordering (`b.evaluation.total_cmp(&a.evaluation)`).
    pub evaluation: f32,
}

/// Iterative-deepening Alpha-Beta player with persistent move-ordered search tree
/// (port & anytime generalization of C++ `minimax::tree::Player` in `minimax-tree.h`).
///
/// Stably sorts evaluated children after each completed node visit so Principal Variation moves
/// are searched first on deeper iterations and subsequent turns (via [`GenerationalTree::advance_to`]).
///
/// ### Dynamic-Depth Safeguards (`MaxEvaluations` / `MaxDuration`)
/// 1. **Abort-Safe Unwinding (`?`)**: If the budget expires mid-iteration (`d >= 2`), `eval_node`
///    returns `Err(SearchAborted)` immediately before overwriting unfinished parent orderings.
/// 2. **Root PV Salvaging**: At the root, if `M0` (from `d - 1`) and a better move `M_k` both
///    complete cleanly at depth `d` before a later sibling aborts, `M_k` (searched with open
///    `beta = +INF`, hence exact) is promoted to index `0` via `rotate_right(1)`.
/// 3. **Exact Endgame Stop (`!hit_horizon`)**: If an iteration finishes without hitting `depth == 0`
///    anywhere in the unpruned tree, every leaf was a terminal outcome—deepening stops immediately
///    instead of re-searching a solved endgame until the budget runs out.
pub struct TreeMinimaxPlayer<R: GameRules<2>, H = ZeroHeuristic> {
    game: Game<R, 2>,
    tree: GenerationalTree<R::Move, TreeMinimaxPayload>,
    heuristic: H,
    budget: MinimaxBudget,
    rng: SmallRng,
    move_buf: Vec<R::Move>,
    eval_calls: usize,
    last_turn_evals: usize,
    last_completed_depth: u32,
}

impl<R: GameRules<2>> TreeMinimaxPlayer<R, ZeroHeuristic> {
    pub fn zero(
        game: Game<R, 2>,
        sentinel_move: R::Move,
        budget: MinimaxBudget,
        seed: u64,
        arena_capacity: usize,
    ) -> Self {
        Self::new(game, sentinel_move, ZeroHeuristic, budget, seed, arena_capacity)
    }
}

impl<R: GameRules<2>, H: HeuristicEvaluator<R, 2>> TreeMinimaxPlayer<R, H> {
    pub fn new(
        game: Game<R, 2>,
        sentinel_move: R::Move,
        heuristic: H,
        budget: MinimaxBudget,
        seed: u64,
        arena_capacity: usize,
    ) -> Self {
        Self {
            game,
            tree: GenerationalTree::new(
                arena_capacity.max(64),
                sentinel_move,
                TreeMinimaxPayload::default(),
            ),
            heuristic,
            budget,
            rng: SmallRng::seed_from_u64(seed),
            move_buf: Vec::new(),
            eval_calls: 0,
            last_turn_evals: 0,
            last_completed_depth: 0,
        }
    }

    #[inline]
    pub const fn game(&self) -> &Game<R, 2> {
        &self.game
    }

    #[inline]
    pub const fn tree(&self) -> &GenerationalTree<R::Move, TreeMinimaxPayload> {
        &self.tree
    }

    #[inline]
    pub const fn budget(&self) -> MinimaxBudget {
        self.budget
    }

    /// Cumulative node evaluations across all turns played so far.
    #[inline]
    pub const fn eval_calls(&self) -> usize {
        self.eval_calls
    }

    /// Node evaluations performed during the most recent `choose_move()` call.
    #[inline]
    pub const fn last_turn_evals(&self) -> usize {
        self.last_turn_evals
    }

    /// Deepest full ply horizon `d` from the root (`1` = 1-ply `depth 0`, `6` = 6-ply `depth 5`)
    /// completed during the most recent `choose_move()` call.
    #[inline]
    pub const fn last_completed_depth(&self) -> u32 {
        self.last_completed_depth
    }

    fn ensure_children(&mut self, node_id: NodeId, game: &Game<R, 2>) {
        if !self.tree.active().get(node_id).has_children() {
            game.write_valid_moves(&mut self.move_buf);
            self.move_buf.shuffle(&mut self.rng);
            let iter = self
                .move_buf
                .iter()
                .copied()
                .map(|mv| (mv, TreeMinimaxPayload::default()));
            self.tree.active_mut().expand_from_iter(node_id, iter);
        }
    }

    /// Searches one iterative-deepening ply `plies >= 1` at `root_id`, returning `true` if the
    /// full root iteration completed or `false` if aborted mid-iteration by `tracker`.
    fn search_root_iteration(
        &mut self,
        root_id: NodeId,
        root_game: &Game<R, 2>,
        plies: u32,
        tracker: &mut TurnTracker,
    ) -> bool {
        let actor = root_game.current_player();
        let root_snapshot = *self.tree.active().get(root_id);
        let child_count = root_snapshot.child_count();
        let first_idx = root_snapshot.first_child().index();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_offset = 0usize;
        let mut completed_children = 0usize;
        let mut aborted = false;

        for offset in 0..child_count {
            let child_id = NodeId::from_index(first_idx + offset);
            let mv = self.tree.active().get(child_id).mv;
            let child_game = root_game.apply(mv);

            let res = if child_game.current_player() == actor {
                self.eval_node(
                    child_id,
                    &child_game,
                    plies - 1,
                    best_score,
                    f32::INFINITY,
                    tracker,
                )
            } else {
                self.eval_node(
                    child_id,
                    &child_game,
                    plies - 1,
                    f32::NEG_INFINITY,
                    -best_score,
                    tracker,
                )
                .map(|v| -v)
            };

            match res {
                Ok(score) => {
                    self.tree.active_mut().get_mut(child_id).payload.evaluation = score;
                    if score > best_score {
                        best_score = score;
                        best_offset = offset;
                    }
                    completed_children = offset + 1;
                }
                Err(SearchAborted) => {
                    aborted = true;
                    break;
                }
            }
        }

        if !aborted {
            self.tree
                .active_mut()
                .children_mut(root_id)
                .sort_by(|a, b| b.payload.evaluation.total_cmp(&a.payload.evaluation));
            self.tree.root_mut().payload.evaluation = best_score;
            true
        } else {
            // Root PV salvaging: if child 0 (`d - 1`'s best move) AND a subsequent move `best_offset`
            // both completed at depth `plies` with `score(best_offset) > score(0)` (exact because
            // `beta = +INF`), promote `best_offset` to index 0 while preserving remaining order!
            if completed_children >= 1 && best_offset > 0 {
                self.tree.active_mut().children_mut(root_id)[0..=best_offset].rotate_right(1);
                self.tree.root_mut().payload.evaluation = best_score;
            }
            false
        }
    }

    fn eval_node(
        &mut self,
        node_id: NodeId,
        game: &Game<R, 2>,
        depth: u32,
        mut alpha: f32,
        beta: f32,
        tracker: &mut TurnTracker,
    ) -> Result<f32, SearchAborted> {
        tracker.step()?;

        let actor = game.current_player();
        let actor_idx = actor.index();

        if let GameStatus::Terminal(payoffs) = game.status() {
            return Ok(terminal_score_for(payoffs[actor_idx], depth));
        }
        if depth == 0 {
            tracker.hit_horizon = true;
            return Ok(self.heuristic.evaluate_heuristic(game)[actor_idx]);
        }

        self.ensure_children(node_id, game);

        let parent_snapshot = *self.tree.active().get(node_id);
        let child_count = parent_snapshot.child_count();
        let first_idx = parent_snapshot.first_child().index();

        let mut best = f32::NEG_INFINITY;
        let mut evaluated_count = 0usize;

        for offset in 0..child_count {
            let child_id = NodeId::from_index(first_idx + offset);
            let mv = self.tree.active().get(child_id).mv;
            let child_game = game.apply(mv);

            let score = if child_game.current_player() == actor {
                self.eval_node(child_id, &child_game, depth - 1, alpha, beta, tracker)?
            } else {
                -self.eval_node(child_id, &child_game, depth - 1, -beta, -alpha, tracker)?
            };

            self.tree.active_mut().get_mut(child_id).payload.evaluation = score;

            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            evaluated_count = offset + 1;
            if alpha >= beta {
                break;
            }
        }

        // Stable-sort the evaluated prefix in-place so the best moves come first on deeper
        // iterations and future turns. Because `Node` stores `(first_child, child_count)` with
        // no parent back-pointers, swapping children leaves all grandchild slices intact!
        self.tree.active_mut().children_mut(node_id)[..evaluated_count]
            .sort_by(|a, b| b.payload.evaluation.total_cmp(&a.payload.evaluation));

        Ok(best)
    }
}

impl<R, H> Player<R, 2> for TreeMinimaxPlayer<R, H>
where
    R: GameRules<2>,
    H: HeuristicEvaluator<R, 2>,
{
    fn choose_move(&mut self) -> R::Move {
        assert!(
            matches!(self.game.status(), GameStatus::Ongoing),
            "TreeMinimaxPlayer::choose_move called on an already terminal game"
        );

        let root_id = self.tree.root_id();
        let root_game = self.game.clone();
        self.ensure_children(root_id, &root_game);
        assert!(
            self.tree.root().has_children(),
            "TreeMinimaxPlayer::choose_move found no valid moves"
        );

        let max_plies = match self.budget {
            MinimaxBudget::FixedDepth(max_depth) => max_depth.saturating_add(1),
            MinimaxBudget::MaxEvaluations(_) | MinimaxBudget::MaxDuration(_) => u32::MAX,
        };

        let mut tracker = TurnTracker::new(self.budget);
        self.last_completed_depth = 0;

        for plies in 1..=max_plies {
            if plies >= 2 && tracker.is_exhausted() {
                break;
            }
            tracker.allow_abort =
                !matches!(self.budget, MinimaxBudget::FixedDepth(_)) && plies >= 2;
            tracker.hit_horizon = false;

            let completed = self.search_root_iteration(root_id, &root_game, plies, &mut tracker);
            if !completed {
                break;
            }
            self.last_completed_depth = plies;

            // If no branch in the entire unpruned search stopped at `depth == 0`, every leaf was
            // an exact `GameStatus::Terminal` outcome—further deepening cannot change the move!
            if !tracker.hit_horizon {
                break;
            }
        }

        self.last_turn_evals = tracker.turn_evals;
        self.eval_calls += tracker.turn_evals;

        let best_child_id = self.tree.root().first_child();
        let chosen_mv = self.tree.active().get(best_child_id).mv;

        self.game = self.game.apply(chosen_mv);
        self.tree.advance_to(best_child_id);
        chosen_mv
    }

    fn apply_external_move(&mut self, mv: R::Move) {
        self.game = self.game.apply(mv);
        self.tree.advance_or_reset(mv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerIndex;
    use crate::games::twist_puzzle::{TwistGame, TwistState};
    use crate::matchup::{play_match, run_multi_match};
    use crate::tree::Node;

    #[test]
    fn tree_minimax_node_size_is_12_bytes() {
        assert_eq!(std::mem::size_of::<Node<u8, TreeMinimaxPayload>>(), 12);
        assert_eq!(std::mem::size_of::<Node<(u8, u8), TreeMinimaxPayload>>(), 12);
    }

    #[test]
    fn maxn_single_player_minimax_solves_twist_puzzle_by_shortest_path() {
        // Scramble 3 moves deep; depth-2 Minimax searches `depth + 1 = 3` plies and prefers
        // larger `remaining_depth`, solving in <= 3 moves!
        let mut rng = SmallRng::seed_from_u64(99);
        let start = TwistState::scrambled(3, 6, &mut rng);
        let game = TwistGame::new(PlayerIndex::P0, start);
        let solver = MinimaxPlayer::zero(game, 2, 42);
        let outcome = play_match(game, &mut [solver]);

        assert_eq!(outcome.payoffs, [1.0]);
        assert!(
            outcome.moves_played <= 3,
            "expected shortest solution <= 3 moves, took {}",
            outcome.moves_played
        );
    }

    /// Minimal 3-player take-away game to verify `Max^n` multi-player lookahead + `MultiMatch`
    /// cyclic 3-seat rotation:
    /// Players `0, 1, 2` take `1` or `2` tokens from a pile; whoever takes the **last** token wins!
    #[derive(Clone, Copy, Debug)]
    struct ThreePlayerTakeAway;

    impl GameRules<3> for ThreePlayerTakeAway {
        type Move = u8;
        type State = u8; // remaining tokens
        type MoveIter<'a> = std::vec::IntoIter<u8>;

        fn valid_moves<'a>(state: &'a Self::State, _player: PlayerIndex<3>) -> Self::MoveIter<'a> {
            let max_take = (*state).min(2);
            (1..=max_take).collect::<Vec<_>>().into_iter()
        }

        fn apply(state: &Self::State, _player: PlayerIndex<3>, mv: Self::Move) -> Self::State {
            assert!((1..=2).contains(&mv) && mv <= *state);
            *state - mv
        }

        fn status(state: &Self::State, current_player: PlayerIndex<3>) -> GameStatus<3> {
            if *state == 0 {
                // Previous player took the last token and won (`+1.0` vs `-0.5` for the other two).
                let prev_raw = ((current_player.index() + 2) % 3) as u8;
                let winner = PlayerIndex::<3>::new(prev_raw);
                GameStatus::zero_sum_winner(winner)
            } else {
                GameStatus::Ongoing
            }
        }
    }

    #[test]
    fn three_player_maxn_outplays_shallow_players_with_3_seat_rotation() {
        let initial = Game::<ThreePlayerTakeAway, 3>::new(PlayerIndex::new(0), 10);

        let shallow_0 = |g: &Game<ThreePlayerTakeAway, 3>, seed: u64| -> Box<dyn Player<ThreePlayerTakeAway, 3>> {
            Box::new(MinimaxPlayer::zero(*g, 0, seed))
        };
        let shallow_1 = |g: &Game<ThreePlayerTakeAway, 3>, seed: u64| -> Box<dyn Player<ThreePlayerTakeAway, 3>> {
            Box::new(MinimaxPlayer::zero(*g, 0, seed))
        };
        let deep_2 = |g: &Game<ThreePlayerTakeAway, 3>, seed: u64| -> Box<dyn Player<ThreePlayerTakeAway, 3>> {
            Box::new(MinimaxPlayer::zero(*g, 9, seed))
        };

        // 30 rounds -> each of the 3 participants plays each of the 3 seats 10 times!
        let summary = run_multi_match(&initial, [&shallow_0, &shallow_1, &deep_2], 30, 3, 2026);
        assert_eq!(summary.rounds, 30);
        assert!(
            summary.wins[2] > summary.wins[0] && summary.wins[2] > summary.wins[1],
            "expected deep Max^n agent to dominate shallow agents, got wins={:?}",
            summary.wins
        );
    }

    #[test]
    fn max_evaluations_budget_strictly_caps_evals_and_dynamically_deepens() {
        use crate::games::tic_tac_toe::{EMPTY_STATE, NO_MOVE, Symbol, TttGame, TttHeuristic};

        let budget_limit = 180usize;
        let mut game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let mut p1 = TreeMinimaxPlayer::new(
            game,
            NO_MOVE,
            TttHeuristic,
            MinimaxBudget::MaxEvaluations(budget_limit),
            123,
            4096,
        );
        let mut p2 = TreeMinimaxPlayer::new(
            game,
            NO_MOVE,
            TttHeuristic,
            MinimaxBudget::MaxEvaluations(budget_limit),
            456,
            4096,
        );

        let mut p1_depths = Vec::new();
        while game.status().is_ongoing() {
            if game.current_player() == PlayerIndex::PLAYER1 {
                let mv = p1.choose_move();
                assert!(
                    p1.last_turn_evals() <= budget_limit,
                    "exceeded MaxEvaluations({budget_limit}): {}",
                    p1.last_turn_evals()
                );
                p1_depths.push(p1.last_completed_depth());
                p2.apply_external_move(mv);
                game = game.apply(mv);
            } else {
                let mv = p2.choose_move();
                assert!(p2.last_turn_evals() <= budget_limit);
                p1.apply_external_move(mv);
                game = game.apply(mv);
            }
        }

        // In the 9-move opening, `180` evaluations reaches ~2-3 plies; by the endgame with fewer
        // legal moves + Alpha-Beta pruning, the exact same budget solves all remaining plies!
        assert!(
            p1_depths.first().copied().unwrap_or(0) <= 3,
            "unexpectedly deep opening depth on 180-eval budget: {p1_depths:?}"
        );

        // Verify `!hit_horizon` early-stop: with 2 empty squares left, a 10_000-eval budget stops
        // after solving those 2 plies (< 20 evaluations) instead of spinning forever!
        let mut near_end = [
            Symbol::Player1,
            Symbol::Player2,
            Symbol::Player1,
            Symbol::Player2,
            Symbol::Player1,
            Symbol::Player2,
            Symbol::Player2,
            Symbol::Empty,
            Symbol::Empty,
        ];
        near_end[7] = Symbol::Empty;
        let endgame = TttGame::new(PlayerIndex::PLAYER1, near_end);
        let mut solver = TreeMinimaxPlayer::new(
            endgame,
            NO_MOVE,
            TttHeuristic,
            MinimaxBudget::MaxEvaluations(10_000),
            99,
            1024,
        );
        let _mv = solver.choose_move();
        assert!(
            solver.last_turn_evals() < 30,
            "expected `!hit_horizon` early-stop in 2-ply endgame, used {} evals",
            solver.last_turn_evals()
        );
    }

    #[test]
    fn higher_evaluation_budget_dominates_lower_budget_and_duration_budget_works() {
        use crate::games::nim::{NO_MOVE, Nim, NimGame};

        // 5-heap Misère Nim: 400 evaluations/turn reaches significantly deeper than 30 evaluations/turn!
        let game = NimGame::<5>::new(PlayerIndex::PLAYER1, [1, 2, 3, 4, 5]);
        let f_small = |g: &NimGame<5>, seed: u64| -> Box<dyn Player<Nim<5>, 2>> {
            Box::new(TreeMinimaxPlayer::zero(
                *g,
                NO_MOVE,
                MinimaxBudget::MaxEvaluations(30),
                seed,
                2048,
            ))
        };
        let f_large = |g: &NimGame<5>, seed: u64| -> Box<dyn Player<Nim<5>, 2>> {
            Box::new(TreeMinimaxPlayer::zero(
                *g,
                NO_MOVE,
                MinimaxBudget::MaxEvaluations(500),
                seed,
                8192,
            ))
        };

        let summary = run_multi_match(&game, [&f_small, &f_large], 80, 4, 2026);
        assert_eq!(summary.draws, 0);
        assert!(
            summary.wins[1] > summary.wins[0],
            "expected MaxEvaluations(500) to beat MaxEvaluations(30), got wins={:?}",
            summary.wins
        );

        // Verify `MaxDuration` completes cleanly and plays a full game under strict clock control.
        let mut p_clock_1 = TreeMinimaxPlayer::zero(
            game,
            NO_MOVE,
            MinimaxBudget::MaxDuration(Duration::from_micros(300)),
            1,
            4096,
        );
        let mut p_clock_2 = TreeMinimaxPlayer::zero(
            game,
            NO_MOVE,
            MinimaxBudget::MaxDuration(Duration::from_micros(300)),
            2,
            4096,
        );
        let clock_outcome = play_match(game, &mut [&mut p_clock_1, &mut p_clock_2]);
        assert!(clock_outcome.moves_played >= 5);
        assert!(p_clock_1.last_completed_depth() >= 1);
    }
}
