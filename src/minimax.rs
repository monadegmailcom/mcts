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

/// Per-node evaluation payload for [`TreeMinimaxPlayer`] (`4` bytes; total `Node<u8, _>` is **12 bytes**!).
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeMinimaxPayload {
    /// Cached evaluation from the **parent** node's acting-player perspective, enabling direct
    /// descending move ordering (`b.evaluation.total_cmp(&a.evaluation)`).
    pub evaluation: f32,
}

/// Iterative-deepening Alpha-Beta player with persistent move-ordered search tree
/// (port of C++ `minimax::tree::Player` in `minimax-tree.h`).
///
/// Searches depths `0 ..= max_depth + 1` from the root, stably sorting evaluated children
/// after each node visit so Principal Variation moves are searched first on deeper iterations
/// and subsequent turns (via [`GenerationalTree::advance_to`]).
pub struct TreeMinimaxPlayer<R: GameRules<2>, H = ZeroHeuristic> {
    game: Game<R, 2>,
    tree: GenerationalTree<R::Move, TreeMinimaxPayload>,
    heuristic: H,
    max_depth: u32,
    rng: SmallRng,
    move_buf: Vec<R::Move>,
    eval_calls: usize,
}

impl<R: GameRules<2>> TreeMinimaxPlayer<R, ZeroHeuristic> {
    pub fn zero(
        game: Game<R, 2>,
        sentinel_move: R::Move,
        max_depth: u32,
        seed: u64,
        arena_capacity: usize,
    ) -> Self {
        Self::new(game, sentinel_move, ZeroHeuristic, max_depth, seed, arena_capacity)
    }
}

impl<R: GameRules<2>, H: HeuristicEvaluator<R, 2>> TreeMinimaxPlayer<R, H> {
    pub fn new(
        game: Game<R, 2>,
        sentinel_move: R::Move,
        heuristic: H,
        max_depth: u32,
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
            max_depth,
            rng: SmallRng::seed_from_u64(seed),
            move_buf: Vec::new(),
            eval_calls: 0,
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
    pub const fn eval_calls(&self) -> usize {
        self.eval_calls
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

    fn eval_node(
        &mut self,
        node_id: NodeId,
        game: &Game<R, 2>,
        depth: u32,
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        self.eval_calls += 1;
        let actor = game.current_player();
        let actor_idx = actor.index();

        if let GameStatus::Terminal(payoffs) = game.status() {
            return terminal_score_for(payoffs[actor_idx], depth);
        }
        if depth == 0 {
            return self.heuristic.evaluate_heuristic(game)[actor_idx];
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
                self.eval_node(child_id, &child_game, depth - 1, alpha, beta)
            } else {
                -self.eval_node(child_id, &child_game, depth - 1, -beta, -alpha)
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

        best
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
        for d in 0..=(self.max_depth + 1) {
            let val = self.eval_node(root_id, &root_game, d, f32::NEG_INFINITY, f32::INFINITY);
            self.tree.root_mut().payload.evaluation = val;
        }

        let best_child_id = self.tree.root().first_child();
        assert!(
            best_child_id.is_some(),
            "TreeMinimaxPlayer::choose_move found no valid moves"
        );
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
}
