//! Unified single-threaded Root-to-Leaf Monte Carlo Tree Search engine.
//!
//! Unifies classical playout-based MCTS (`montecarlo.h`) and AlphaZero (`alphazero.h`) into a
//! single generic engine parameterized by:
//! 1. [`SelectionPolicy`]: e.g. [`Uct`] (classical MCTS) or [`Puct`] (AlphaZero).
//! 2. [`LeafEvaluator`]: e.g. [`PlayoutEvaluator`] (random rollouts) or a neural-network
//!    prior + value evaluator.
//!
//! ### Why Root-to-Leaf State Reconstruction?
//! Instead of storing `Game<R, N>` inside unexpanded `PreNode`s and dispatching via virtual
//! `NodeVisitor`s (`PreNode` vs `FixNode`), each simulation starts from `root_game` and applies
//! `curr_game = curr_game.apply(child.mv)` along the selected path (typically 5–40 plies).
//! This removes `State`, `vptr`, `parent` pointers, and `std::shared_mutex` from tree nodes,
//! shrinking every node to **20 bytes** (for 1–2 byte moves).
//!
//! ### Ready for Lock-Free Game Parallelism & Root Parallelism
//! Search is decomposed into two state-machine primitives—[`MctsSearch::select_to_leaf`] and
//! [`MctsSearch::backpropagate`]—so that:
//! - Synchronous callers simply invoke [`MctsSearch::simulate_once`] / [`MctsSearch::run_simulations`].
//! - Batched GPU Self-Play workers can step hundreds of independent single-threaded `MctsSearch`
//!   instances to their leaves, batch all leaf states into a single GPU forward pass, write priors,
//!   and call `backpropagate`—achieving full GPU saturation with **zero virtual loss** and 100%
//!   informed PUCT selections.

use std::task::Poll;

use rand::prelude::{IndexedRandom, RngExt, SeedableRng, SliceRandom, SmallRng};

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::player::Player;
use crate::tree::{GenerationalTree, Node, NodeId};

/// Per-node statistics stored inside [`Node<M, MctsPayload>`].
///
/// Memory layout: `4 + 4 + 4 = 12` bytes. Combined with `NodeId` (4B), `child_count: u16` (2B),
/// and a 1–2 byte move `M`, the entire `Node<M, MctsPayload>` is **20 bytes**.
#[derive(Clone, Copy, Debug, Default)]
pub struct MctsPayload {
    /// Number of simulations that have passed through this node.
    pub visits: u32,
    /// Accumulated payoff sum from the perspective of the player who **chose** the move
    /// leading into this node (eliminating sign-flipping assumptions so `N = 1`, `N = 2`,
    /// and `N > 2` work identically).
    pub value_sum: f32,
    /// Prior probability `P(s, a)` assigned when the parent node was evaluated.
    pub prior: f32,
}

impl MctsPayload {
    /// Mean action value `Q(s, a)` for the player who chose the action leading to this node.
    #[inline]
    pub fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / (self.visits as f32)
        }
    }
}

/// Strategy trait for scoring child nodes during tree descent (compile-time monomorphized).
///
/// Selection at a parent first computes [`Self::Parent`] once (quantities shared by all
/// siblings, e.g. the exploration factor), then scores each child against it.
pub trait SelectionPolicy: Copy {
    /// Per-parent quantities computed once before the parent's children are scored.
    type Parent: Copy;

    fn parent<M>(&self, parent_visits: u32, children: &[Node<M, MctsPayload>]) -> Self::Parent;

    fn score(&self, child: &MctsPayload, parent: Self::Parent) -> f32;
}

/// Classical Upper Confidence Bound for Trees (UCT).
///
/// Unvisited children receive `f32::INFINITY` so every legal move is tried once before
/// revisiting any sibling (matching `montecarlo.h`'s progressive move iterator).
#[derive(Clone, Copy, Debug)]
pub struct Uct {
    /// Exploration constant `c` (note: payoffs are in `[-1.0, 1.0]`, so `c ≈ 1.4` corresponds
    /// to `0.7` in `[0.0, 1.0]` scale).
    pub exploration: f32,
}

impl Uct {
    pub const fn new(exploration: f32) -> Self {
        Self { exploration }
    }
}

impl Default for Uct {
    fn default() -> Self {
        Self {
            exploration: std::f32::consts::SQRT_2,
        }
    }
}

impl SelectionPolicy for Uct {
    /// `ln(N_parent)`.
    type Parent = f32;

    #[inline]
    fn parent<M>(&self, parent_visits: u32, _children: &[Node<M, MctsPayload>]) -> f32 {
        (parent_visits as f32).ln()
    }

    #[inline]
    fn score(&self, child: &MctsPayload, ln_parent_visits: f32) -> f32 {
        if child.visits == 0 {
            return f32::INFINITY;
        }
        let exploit = child.q_value();
        let explore = self.exploration * (ln_parent_visits / (child.visits as f32)).sqrt();
        exploit + explore
    }
}

/// First-play urgency: the `Q` that [`Puct`] assumes for a child that has not been visited yet.
///
/// It decides how wide the search is: an optimistic value makes it try every move before going
/// deeper, a pessimistic one makes it follow the policy prior narrowly. Matters most with few
/// simulations and many legal moves. Values are payoffs in `[-1, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fpu {
    /// A fixed value. `Absolute(0.0)` ("assume a draw") is the AlphaGo Zero / AlphaZero choice;
    /// it widens the search in losing positions and narrows it in winning ones.
    Absolute(f32),
    /// The parent's value minus `reduction * sqrt(sum of the priors of the visited children)`
    /// (Leela Chess Zero, KataGo): an untried move is assumed a bit worse than the position, and
    /// more so once the moves the network likes have been explored. The parent's value is the
    /// visit-weighted mean `Q` of its visited children, from the choosing player's view.
    Relative(f32),
}

impl Default for Fpu {
    fn default() -> Self {
        Self::Absolute(0.0)
    }
}

impl Fpu {
    /// Clamps the value to a sane range (`[-1, 1]` absolute, `[0, 2]` relative reduction).
    pub fn sanitized(self) -> Self {
        match self {
            Self::Absolute(v) => Self::Absolute(v.clamp(-1.0, 1.0)),
            Self::Relative(r) => Self::Relative(r.clamp(0.0, 2.0)),
        }
    }

    /// The `Q` for unvisited children of a parent with these `children`.
    fn unvisited_q<M>(self, children: &[Node<M, MctsPayload>]) -> f32 {
        match self {
            Self::Absolute(v) => v,
            Self::Relative(reduction) => {
                let (mut visits, mut value_sum, mut visited_prior) = (0u32, 0.0f32, 0.0f32);
                for c in children.iter().filter(|c| c.payload.visits > 0) {
                    visits += c.payload.visits;
                    value_sum += c.payload.value_sum;
                    visited_prior += c.payload.prior;
                }
                if visits == 0 {
                    // Nothing visited yet: every child gets the same `Q`, so its value is irrelevant.
                    return 0.0;
                }
                value_sum / visits as f32 - reduction * visited_prior.sqrt()
            }
        }
    }
}

/// Predictor + Upper Confidence Bound for Trees (AlphaZero PUCT).
///
/// Formula (matching `alphazero.h`):
/// `Q(s, a) + P(s, a) * (ln((N_p + c_base + 1) / c_base) + c_init) * sqrt(N_p) / (1 + N_c)`,
/// with `Q` of unvisited children given by [`Fpu`].
#[derive(Clone, Copy, Debug)]
pub struct Puct {
    pub c_base: f32,
    pub c_init: f32,
    pub fpu: Fpu,
}

impl Puct {
    pub const fn new(c_base: f32, c_init: f32) -> Self {
        Self {
            c_base,
            c_init,
            fpu: Fpu::Absolute(0.0),
        }
    }

    pub const fn with_fpu(mut self, fpu: Fpu) -> Self {
        self.fpu = fpu;
        self
    }
}

/// Quantities [`Puct`] shares across the children of one parent.
#[derive(Clone, Copy, Debug)]
pub struct PuctParent {
    /// `(ln((N_p + c_base + 1) / c_base) + c_init) * sqrt(N_p)`.
    explore: f32,
    /// `Q` assumed for unvisited children.
    unvisited_q: f32,
}

impl Default for Puct {
    fn default() -> Self {
        Self::new(19_652.0, 1.25)
    }
}

impl SelectionPolicy for Puct {
    type Parent = PuctParent;

    #[inline]
    fn parent<M>(&self, parent_visits: u32, children: &[Node<M, MctsPayload>]) -> PuctParent {
        let n = parent_visits as f32;
        PuctParent {
            explore: (((n + self.c_base + 1.0) / self.c_base).ln() + self.c_init) * n.sqrt(),
            unvisited_q: self.fpu.unvisited_q(children),
        }
    }

    #[inline]
    fn score(&self, child: &MctsPayload, parent: PuctParent) -> f32 {
        let q = if child.visits == 0 { parent.unvisited_q } else { child.q_value() };
        q + parent.explore * child.prior / ((child.visits + 1) as f32)
    }
}

/// Pluggable leaf evaluation strategy unifying playout MCTS and AlphaZero neural evaluation.
pub trait LeafEvaluator<R: GameRules<N>, const N: usize> {
    /// Whether `poll_evaluate` submits work to an external asynchronous batch service (`true` for
    /// [`crate::alphazero::NnEvaluator`]) rather than computing synchronously on the calling CPU
    /// thread (`false` for [`PlayoutEvaluator`]).
    const IS_ASYNC: bool = false;

    /// Synchronously evaluates a newly expanded non-terminal `game` position.
    ///
    /// `children` holds the newly allocated child nodes for all valid moves from `game`,
    /// pre-initialized with uniform priors `1.0 / children.len()`.
    /// - A rollout evaluator leaves `child.payload.prior` uniform and returns terminal playout payoffs.
    /// - An AlphaZero evaluator writes masked softmax NN priors into `child.payload.prior` and
    ///   returns the NN value vector `[f32; N]`.
    fn evaluate(
        &mut self,
        game: &Game<R, N>,
        children: &mut [Node<R::Move, MctsPayload>],
    ) -> [f32; N];

    /// Non-blocking evaluation step enabling [`MctsSearch::poll_simulations`] to multiplex
    /// asynchronous neural network requests across parallel trees or games.
    #[inline]
    fn poll_evaluate(
        &mut self,
        game: &Game<R, N>,
        children: &mut [Node<R::Move, MctsPayload>],
    ) -> Poll<[f32; N]> {
        Poll::Ready(self.evaluate(game, children))
    }

    /// Wakes any background evaluation service before the caller parks waiting for responses.
    #[inline]
    fn notify_pending(&mut self) {}

    /// Hook invoked once per turn on the root node's children after their initial priors have
    /// been assigned (either immediately after the root is evaluated on Move 1, or at the start
    /// of the turn's search when the root was reused from a previous turn's subtree).
    ///
    /// Classical evaluators ([`PlayoutEvaluator`]) leave this as a no-op, whereas
    /// [`crate::alphazero::NnEvaluator`] uses it to blend root Dirichlet exploration noise.
    #[inline]
    fn prepare_root(&mut self, _root_children: &mut [Node<R::Move, MctsPayload>]) {}
}

/// Classical uniform random rollout evaluator (port of `montecarlo::Player::playout`).
#[derive(Clone, Debug)]
pub struct PlayoutEvaluator<M> {
    rng: SmallRng,
    move_buf: Vec<M>,
    playout_count: u64,
}

impl<M> PlayoutEvaluator<M> {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            move_buf: Vec::with_capacity(32),
            playout_count: 0,
        }
    }

    #[inline]
    pub const fn playout_count(&self) -> u64 {
        self.playout_count
    }
}

impl<R: GameRules<N>, const N: usize> LeafEvaluator<R, N> for PlayoutEvaluator<R::Move> {
    fn evaluate(
        &mut self,
        game: &Game<R, N>,
        _children: &mut [Node<R::Move, MctsPayload>],
    ) -> [f32; N] {
        self.playout_count += 1;
        let mut sim_game = game.clone();
        loop {
            if let GameStatus::Terminal(payoffs) = sim_game.status() {
                return payoffs;
            }
            sim_game.write_valid_moves(&mut self.move_buf);
            let &mv = self
                .move_buf
                .choose(&mut self.rng)
                .expect("Ongoing game state produced no valid moves during playout");
            sim_game = sim_game.apply(mv);
        }
    }
}

/// Adapter allowing any closure `FnMut(&Game<R, N>, &mut [Node<R::Move, MctsPayload>]) -> [f32; N]`
/// to serve as a [`LeafEvaluator`] (convenient for custom heuristics and AlphaZero policy+value tests).
pub struct FnEvaluator<F>(pub F);

impl<R, const N: usize, F> LeafEvaluator<R, N> for FnEvaluator<F>
where
    R: GameRules<N>,
    F: FnMut(&Game<R, N>, &mut [Node<R::Move, MctsPayload>]) -> [f32; N],
{
    #[inline]
    fn evaluate(
        &mut self,
        game: &Game<R, N>,
        children: &mut [Node<R::Move, MctsPayload>],
    ) -> [f32; N] {
        (self.0)(game, children)
    }
}

/// Result of stepping the MCTS tree from [`NodeId::ROOT`] down to a leaf.
#[derive(Debug)]
pub enum LeafStep<R: GameRules<N>, const N: usize> {
    /// Traversal reached a terminal state with known exact payoffs `[f32; N]`.
    Terminal([f32; N]),
    /// Traversal reached and expanded a fresh non-terminal leaf that now needs evaluation.
    NeedsEvaluation {
        leaf_id: NodeId,
        leaf_game: Game<R, N>,
    },
}

/// Single-threaded MCTS tree and root-to-leaf traversal state machine.
pub struct MctsSearch<R: GameRules<N>, const N: usize, S: SelectionPolicy> {
    root_game: Game<R, N>,
    tree: GenerationalTree<R::Move, MctsPayload>,
    selector: S,
    rng: SmallRng,
    root_prepared: bool,
    sims_done: usize,
    pending_leaf: Option<(NodeId, Game<R, N>)>,
    /// Reusable root-to-leaf `(child_id, acting_player_who_chose_child)` stack.
    path_buf: Vec<(NodeId, PlayerIndex<N>)>,
    /// Reusable buffer for bulk move generation during leaf expansion.
    move_buf: Vec<R::Move>,
    /// Per-move discount `γ` of backed-up values (see [`Self::with_discount`]).
    discount: f32,
}

impl<R: GameRules<N>, const N: usize, S: SelectionPolicy> MctsSearch<R, N, S> {
    /// Creates a new MCTS search tree rooted at `root_game`.
    ///
    /// `dummy_move` is stored on the initial root node (before any move has been played).
    pub fn new(
        root_game: Game<R, N>,
        dummy_move: R::Move,
        selector: S,
        arena_capacity: usize,
        seed: u64,
    ) -> Self {
        Self {
            root_game,
            tree: GenerationalTree::new(arena_capacity, dummy_move, MctsPayload::default()),
            selector,
            rng: SmallRng::seed_from_u64(seed),
            root_prepared: false,
            sims_done: 0,
            pending_leaf: None,
            path_buf: Vec::with_capacity(64),
            move_buf: Vec::with_capacity(32),
            discount: 1.0,
        }
    }

    /// Discounts backed-up values by `γ` per move: a payoff reached `k` moves below a node counts
    /// `γ^k` there, so the search prefers short wins (default `1.0`: no discount). For puzzles,
    /// where solved is worth `1` and failure `0`, the value of a position then becomes
    /// `γ^(moves to solved)`.
    pub const fn with_discount(mut self, discount: f32) -> Self {
        self.discount = discount;
        self
    }

    #[inline]
    pub const fn root_game(&self) -> &Game<R, N> {
        &self.root_game
    }

    #[inline]
    pub const fn tree(&self) -> &GenerationalTree<R::Move, MctsPayload> {
        &self.tree
    }

    #[inline]
    pub fn tree_mut(&mut self) -> &mut GenerationalTree<R::Move, MctsPayload> {
        &mut self.tree
    }

    #[inline]
    pub const fn selector(&self) -> &S {
        &self.selector
    }

    /// Walks from `ROOT` down to a leaf by applying moves on the fly (`curr_game.apply(mv)`),
    /// expanding the leaf if it is non-terminal and unvisited.
    pub fn select_to_leaf(&mut self) -> LeafStep<R, N> {
        self.path_buf.clear();
        let mut curr_id = NodeId::ROOT;
        let mut curr_game = self.root_game.clone();

        loop {
            let node = *self.tree.active().get(curr_id);

            if node.payload.visits == 0 {
                // First visit to `curr_id`: check terminal status before expanding.
                if let GameStatus::Terminal(payoffs) = curr_game.status() {
                    return LeafStep::Terminal(payoffs);
                }

                curr_game.write_valid_moves(&mut self.move_buf);
                assert!(
                    !self.move_buf.is_empty(),
                    "Ongoing game state must yield at least one valid move"
                );
                // Shuffle move order once at expansion to eliminate board-index tie bias.
                self.move_buf.shuffle(&mut self.rng);

                let uniform_prior = 1.0 / (self.move_buf.len() as f32);
                self.tree.active_mut().expand_from_iter(
                    curr_id,
                    self.move_buf.iter().copied().map(|mv| {
                        (
                            mv,
                            MctsPayload {
                                visits: 0,
                                value_sum: 0.0,
                                prior: uniform_prior,
                            },
                        )
                    }),
                );

                return LeafStep::NeedsEvaluation {
                    leaf_id: curr_id,
                    leaf_game: curr_game,
                };
            }

            if !node.has_children() {
                // Already-visited node with 0 children is guaranteed to be terminal.
                let payoffs = curr_game
                    .status()
                    .payoffs()
                    .expect("visited childless node must be terminal");
                return LeafStep::Terminal(payoffs);
            }

            // Expanded interior node: pick child maximizing `SelectionPolicy::score`.
            let parent_visits = node.payload.visits;
            let (best_child_id, best_mv) = self.select_best_child(curr_id, parent_visits);
            let acting_player = curr_game.current_player();

            self.path_buf.push((best_child_id, acting_player));
            curr_game = curr_game.apply(best_mv);
            curr_id = best_child_id;
        }
    }

    /// Backpropagates `payoffs` (`[f32; N]`, the value of the leaf) along the active root-to-leaf
    /// path recorded by [`Self::select_to_leaf`], discounted by `γ` per move above the leaf.
    #[inline]
    pub fn backpropagate(&mut self, payoffs: [f32; N]) {
        let arena = self.tree.active_mut();
        arena.get_mut(NodeId::ROOT).payload.visits += 1;
        let mut factor = 1.0;
        for &(node_id, acting_player) in self.path_buf.iter().rev() {
            let payload = &mut arena.get_mut(node_id).payload;
            payload.visits += 1;
            payload.value_sum += factor * payoffs[acting_player.index()];
            factor *= self.discount;
        }
    }

    /// Non-blocking simulation driver that steps this tree toward `target_simulations` using
    /// `evaluator.poll_evaluate(...)`.
    ///
    /// - For synchronous evaluators ([`PlayoutEvaluator`]), completes all `target_simulations`
    ///   in a single call and returns `Poll::Ready(())`.
    /// - For asynchronous evaluators ([`crate::alphazero::NnEvaluator`]), pauses and returns
    ///   `Poll::Pending` whenever a leaf is waiting on a neural network batch response, resuming
    ///   from that exact leaf on the next poll!
    pub fn poll_simulations<E: LeafEvaluator<R, N>>(
        &mut self,
        target_simulations: usize,
        evaluator: &mut E,
    ) -> Poll<()> {
        while self.sims_done < target_simulations {
            if self.pending_leaf.is_none() {
                if !self.root_prepared
                    && self.tree.root().payload.visits > 0
                    && !self.tree.active().children(NodeId::ROOT).is_empty()
                {
                    evaluator.prepare_root(self.tree.active_mut().children_mut(NodeId::ROOT));
                    self.root_prepared = true;
                }

                match self.select_to_leaf() {
                    LeafStep::Terminal(payoffs) => {
                        self.backpropagate(payoffs);
                        self.sims_done += 1;
                        continue;
                    }
                    LeafStep::NeedsEvaluation { leaf_id, leaf_game } => {
                        self.pending_leaf = Some((leaf_id, leaf_game));
                    }
                }
            }

            let (leaf_id, leaf_game) = self
                .pending_leaf
                .as_ref()
                .expect("pending_leaf must be present");
            let leaf_id = *leaf_id;
            let children = self.tree.active_mut().children_mut(leaf_id);
            match evaluator.poll_evaluate(leaf_game, children) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(payoffs) => {
                    if leaf_id == NodeId::ROOT && !self.root_prepared {
                        evaluator.prepare_root(children);
                        self.root_prepared = true;
                    }
                    self.pending_leaf = None;
                    self.backpropagate(payoffs);
                    self.sims_done += 1;
                }
            }
        }
        self.sims_done = 0;
        Poll::Ready(())
    }

    /// Executes a single complete MCTS simulation (Select -> Expand -> Evaluate -> Backpropagate).
    #[inline]
    pub fn simulate_once<E: LeafEvaluator<R, N>>(&mut self, evaluator: &mut E) {
        let payoffs = match self.select_to_leaf() {
            LeafStep::Terminal(payoffs) => payoffs,
            LeafStep::NeedsEvaluation { leaf_id, leaf_game } => {
                let children = self.tree.active_mut().children_mut(leaf_id);
                evaluator.evaluate(&leaf_game, children)
            }
        };
        self.backpropagate(payoffs);
    }

    /// Runs `simulations` sequential MCTS iterations.
    pub fn run_simulations<E: LeafEvaluator<R, N>>(
        &mut self,
        simulations: usize,
        evaluator: &mut E,
    ) {
        loop {
            if self.poll_simulations(simulations, evaluator).is_ready() {
                return;
            }
            evaluator.notify_pending();
            std::thread::park();
        }
    }

    /// Returns the `(NodeId, R::Move)` of the root child with the highest visit count
    /// (breaking ties by higher `q_value`).
    pub fn best_root_child(&self) -> (NodeId, R::Move) {
        self.tree
            .active()
            .children_with_ids(NodeId::ROOT)
            .max_by(|(_, a), (_, b)| {
                a.payload
                    .visits
                    .cmp(&b.payload.visits)
                    .then_with(|| a.payload.q_value().total_cmp(&b.payload.q_value()))
            })
            .map(|(id, node)| (id, node.mv))
            .expect("root has no expanded children to choose from")
    }

    /// Samples a root child proportionally to its visit count `N(s_root, a)`
    /// (used for AlphaZero opening move diversity; port of `choose_opening_move`).
    pub fn sample_root_child_by_visits(&mut self) -> (NodeId, R::Move) {
        let children = self.tree.active().children(NodeId::ROOT);
        assert!(
            !children.is_empty(),
            "root has no expanded children to sample"
        );

        let total_visits: u64 = children.iter().map(|c| c.payload.visits as u64).sum();
        if total_visits == 0 {
            let idx = self.rng.random_range(0..children.len());
            let first = self.tree.root().first_child().index();
            return (NodeId::from_index(first + idx), children[idx].mv);
        }

        let mut target = self.rng.random_range(0..total_visits);
        let first = self.tree.root().first_child().index();
        for (offset, child) in children.iter().enumerate() {
            let v = child.payload.visits as u64;
            if target < v {
                return (NodeId::from_index(first + offset), child.mv);
            }
            target -= v;
        }
        unreachable!("visit sum sampling must select a child")
    }

    /// Advances both `root_game` and the generational arena tree by `mv`,
    /// reusing the surviving subtree if `mv` was already expanded.
    pub fn advance(&mut self, mv: R::Move) -> bool {
        self.root_game = self.root_game.apply(mv);
        self.sims_done = 0;
        self.root_prepared = false;
        self.pending_leaf = None;
        self.tree.advance_or_reset(mv)
    }

    /// Advances `root_game` by `mv` and promotes `child_id` directly to root without scanning.
    #[inline]
    pub fn advance_to_child(&mut self, child_id: NodeId, mv: R::Move) {
        self.root_game = self.root_game.apply(mv);
        self.sims_done = 0;
        self.root_prepared = false;
        self.pending_leaf = None;
        self.tree.advance_to(child_id);
    }

    /// Resets the search state machine to a fresh `root_game` in `O(1)` while retaining all
    /// allocated arena and scratch-buffer capacities (used to recycle slots across self-play games).
    #[inline]
    pub fn reset(&mut self, root_game: Game<R, N>, dummy_move: R::Move) {
        self.root_game = root_game;
        self.sims_done = 0;
        self.root_prepared = false;
        self.pending_leaf = None;
        self.tree.reset_root(dummy_move, MctsPayload::default());
        self.path_buf.clear();
        self.move_buf.clear();
    }

    /// Shannon entropy `-sum(p_i * ln(p_i))` of the root's normalized child visit distribution
    /// (port of `alphazero::shannon_entropy` for search sharpness diagnostics).
    pub fn root_visit_entropy(&self) -> f32 {
        let children = self.tree.active().children(NodeId::ROOT);
        let visit_sum: u64 = children.iter().map(|c| c.payload.visits as u64).sum();
        if visit_sum == 0 {
            return 0.0;
        }
        let total = visit_sum as f32;
        let mut entropy = 0.0f32;
        for child in children {
            if child.payload.visits > 0 {
                let p = (child.payload.visits as f32) / total;
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Visit-weighted expected value at the root for `root_game.current_player()`.
    pub fn root_value_estimate(&self) -> f32 {
        let children = self.tree.active().children(NodeId::ROOT);
        let mut total_visits = 0u64;
        let mut total_value = 0.0f32;
        for child in children {
            total_visits += child.payload.visits as u64;
            total_value += child.payload.value_sum;
        }
        if total_visits == 0 {
            0.0
        } else {
            total_value / (total_visits as f32)
        }
    }

    #[inline]
    fn select_best_child(&self, parent_id: NodeId, parent_visits: u32) -> (NodeId, R::Move) {
        let arena = self.tree.active();
        let children = arena.children(parent_id);
        let base = arena.get(parent_id).first_child().index();

        let parent = self.selector.parent(parent_visits, children);
        let mut best_offset = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (offset, child) in children.iter().enumerate() {
            let score = self.selector.score(&child.payload, parent);
            if score > best_score {
                best_score = score;
                best_offset = offset;
            }
        }
        (
            NodeId::from_index(base + best_offset),
            children[best_offset].mv,
        )
    }
}

/// Search budget and arena sizing parameters for [`MctsPlayer`]
/// (analogous to `alphazero::params::GamePlay` in C++ `alphazero.h`).
#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub simulations: usize,
    pub opening_moves: usize,
    pub arena_capacity: usize,
    pub seed: u64,
}

impl MctsConfig {
    pub const fn new(simulations: usize, seed: u64) -> Self {
        Self {
            simulations,
            opening_moves: 0,
            arena_capacity: simulations * 16,
            seed,
        }
    }

    pub const fn with_opening_moves(mut self, opening_moves: usize) -> Self {
        self.opening_moves = opening_moves;
        self
    }

    pub const fn with_arena_capacity(mut self, arena_capacity: usize) -> Self {
        self.arena_capacity = arena_capacity;
        self
    }
}

/// Ready-to-play MCTS [`Player`] pairing [`MctsSearch`] with any [`LeafEvaluator`].
///
/// - With `S = Uct` and `E = PlayoutEvaluator<R::Move>`, this is classical playout MCTS.
/// - With `S = Puct` and `E = NnEvaluator`, this is single-tree AlphaZero.
pub struct MctsPlayer<R: GameRules<N>, const N: usize, S: SelectionPolicy, E: LeafEvaluator<R, N>> {
    search: MctsSearch<R, N, S>,
    evaluator: E,
    simulations: usize,
    opening_moves: usize,
    move_count: usize,
}

impl<R, const N: usize, S, E> MctsPlayer<R, N, S, E>
where
    R: GameRules<N>,
    S: SelectionPolicy,
    E: LeafEvaluator<R, N>,
{
    pub fn new(
        game: Game<R, N>,
        dummy_move: R::Move,
        selector: S,
        evaluator: E,
        config: MctsConfig,
    ) -> Self {
        assert!(
            config.simulations > 0,
            "simulations must be greater than zero"
        );
        Self {
            search: MctsSearch::new(
                game,
                dummy_move,
                selector,
                config.arena_capacity,
                config.seed,
            ),
            evaluator,
            simulations: config.simulations,
            opening_moves: config.opening_moves,
            move_count: 0,
        }
    }

    #[inline]
    pub const fn search(&self) -> &MctsSearch<R, N, S> {
        &self.search
    }

    #[inline]
    pub const fn evaluator(&self) -> &E {
        &self.evaluator
    }
}

impl<R, const N: usize, S, E> Player<R, N> for MctsPlayer<R, N, S, E>
where
    R: GameRules<N>,
    S: SelectionPolicy,
    E: LeafEvaluator<R, N>,
{
    fn poll_choose_move(&mut self) -> Poll<R::Move> {
        if self
            .search
            .poll_simulations(self.simulations, &mut self.evaluator)
            .is_pending()
        {
            return Poll::Pending;
        }

        self.move_count += 1;
        let (chosen_id, chosen_mv) = if self.move_count <= self.opening_moves {
            self.search.sample_root_child_by_visits()
        } else {
            self.search.best_root_child()
        };

        self.search.advance_to_child(chosen_id, chosen_mv);
        Poll::Ready(chosen_mv)
    }

    #[inline]
    fn notify_pending(&mut self) {
        self.evaluator.notify_pending();
    }

    fn choose_move(&mut self) -> R::Move {
        loop {
            if let Poll::Ready(mv) = self.poll_choose_move() {
                return mv;
            }
            self.notify_pending();
            std::thread::park();
        }
    }

    fn apply_external_move(&mut self, mv: R::Move) {
        self.move_count += 1;
        self.search.advance(mv);
    }
}

/// Applies masked softmax over raw policy logits for a slice of legal child nodes,
/// writing normalized prior probabilities `P(s, a)` directly into `child.payload.prior`
/// (reusable helper for AlphaZero NN evaluators; port of `alphazero::Player::backpropagation`'s
/// softmax normalization).
pub fn assign_masked_softmax_priors<M: Copy>(
    children: &mut [Node<M, MctsPayload>],
    mut move_to_logit: impl FnMut(M) -> f32,
) {
    if children.is_empty() {
        return;
    }
    // Subtract max logit for numerical stability (`exp(x - max_x)` never overflows to `inf`).
    let mut max_logit = f32::NEG_INFINITY;
    for child in children.iter_mut() {
        let logit = move_to_logit(child.mv);
        child.payload.prior = logit;
        if logit > max_logit {
            max_logit = logit;
        }
    }

    let mut sum = 0.0f32;
    for child in children.iter_mut() {
        let p = (child.payload.prior - max_logit).exp();
        child.payload.prior = p;
        sum += p;
    }

    let inv_sum = 1.0 / sum;
    for child in children.iter_mut() {
        child.payload.prior *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discounted_backup_values_a_solve_one_move_later_at_gamma() {
        use crate::games::rubiks_cube::{NO_MOVE, RubiksGame, RubiksState, apply_move_stickers, inverse_move};
        // Scramble `U R`: the only 2-move solution is `R' U'`; the budget ends the search at depth 2.
        let stickers = apply_move_stickers(&apply_move_stickers(&RubiksState::solved(0).stickers, 0), 6);
        let game = RubiksGame::new(PlayerIndex::P0, RubiksState { stickers, remaining_steps: 2, last_move: NO_MOVE });
        let mut search = MctsSearch::new(game, NO_MOVE, Uct::new(1.4), 4096, 3).with_discount(0.5);
        let mut zero = FnEvaluator(|_: &RubiksGame, _: &mut [Node<u8, MctsPayload>]| [0.0]);
        search.run_simulations(400, &mut zero);

        let tree = search.tree().active();
        let (first_id, _) = tree.children_with_ids(NodeId::ROOT).find(|(_, c)| c.mv == inverse_move(6)).unwrap();
        let (_, solve) = tree.children_with_ids(first_id).find(|(_, c)| c.mv == inverse_move(0)).unwrap();
        assert!(solve.payload.visits > 10, "the search finds the solution");
        // The solved cube is worth 1 where it is reached and 0.5 one move earlier; the other
        // grandchildren end unsolved (0).
        assert_eq!(solve.payload.value_sum, solve.payload.visits as f32);
        assert_eq!(tree.get(first_id).payload.value_sum, 0.5 * solve.payload.visits as f32);
    }

    #[test]
    fn mcts_node_size_is_20_bytes_for_1_and_2_byte_moves() {
        assert_eq!(std::mem::size_of::<MctsPayload>(), 12);
        assert_eq!(std::mem::size_of::<Node<u8, MctsPayload>>(), 20);
        assert_eq!(std::mem::size_of::<Node<u16, MctsPayload>>(), 20);
    }

    #[test]
    fn masked_softmax_is_numerically_stable_for_large_logits() {
        let mut arena = crate::tree::TreeArena::<u8, MctsPayload>::with_capacity(4);
        let root = arena.init_root(0, MctsPayload::default());
        let children = arena.expand_from_iter(
            root,
            [
                (0u8, MctsPayload::default()),
                (1u8, MctsPayload::default()),
            ],
        );
        // Raw `exp(1001.0)` would overflow `f32::MAX` without max-logit subtraction!
        assign_masked_softmax_priors(children, |mv| if mv == 0 { 1001.0 } else { 1000.0 });
        let expected_p0 = std::f32::consts::E / (std::f32::consts::E + 1.0);
        assert!((children[0].payload.prior - expected_p0).abs() < 1e-6);
        assert!((children[1].payload.prior - (1.0 - expected_p0)).abs() < 1e-6);
    }

    fn payload(visits: u32, value_sum: f32, prior: f32) -> MctsPayload {
        MctsPayload { visits, value_sum, prior }
    }

    #[test]
    fn fpu_values_for_unvisited_children() {
        let mut arena = crate::tree::TreeArena::<u8, MctsPayload>::with_capacity(4);
        let root = arena.init_root(0, MctsPayload::default());
        // Two visited children (priors 0.5 + 0.25, mean Q = (2.0 + 1.0) / 6 = 0.5), one unvisited.
        let children = arena.expand_from_iter(
            root,
            [(0u8, payload(4, 2.0, 0.5)), (1u8, payload(2, 1.0, 0.25)), (2u8, payload(0, 0.0, 0.25))],
        );
        assert_eq!(Fpu::Absolute(-0.3).unvisited_q(children), -0.3);
        let relative = Fpu::Relative(0.2).unvisited_q(children);
        assert!((relative - (0.5 - 0.2 * 0.75f32.sqrt())).abs() < 1e-6);
        // Nothing visited yet: all siblings share the same Q, so selection goes by prior alone.
        let fresh = [Node::new(0u8, payload(0, 0.0, 0.7)), Node::new(1u8, payload(0, 0.0, 0.3))];
        assert_eq!(Fpu::Relative(0.2).unvisited_q(&fresh), 0.0);
    }

    #[test]
    fn fpu_controls_search_width() {
        // One visited child with mean value `q` (prior 0.9) and an unvisited sibling (prior 0.1)
        // whose exploration bonus alone is too small to be picked. Returns whether the search
        // would try the unvisited move next.
        let tries_new_move = |fpu: Fpu, q: f32| {
            let mut arena = crate::tree::TreeArena::<u8, MctsPayload>::with_capacity(4);
            let root = arena.init_root(0, MctsPayload::default());
            let children = arena.expand_from_iter(root, [(0u8, payload(5, 5.0 * q, 0.9)), (1u8, payload(0, 0.0, 0.1))]);
            let puct = Puct::default().with_fpu(fpu);
            let parent = puct.parent(5, children);
            puct.score(&children[1].payload, parent) > puct.score(&children[0].payload, parent)
        };
        // Optimistic absolute FPU always widens.
        assert!(tries_new_move(Fpu::Absolute(1.0), 0.6));
        // Absolute "draw" FPU deepens in a winning position but widens in a losing one ...
        assert!(!tries_new_move(Fpu::Absolute(0.0), 0.6));
        assert!(tries_new_move(Fpu::Absolute(0.0), -0.6));
        // ... while relative FPU behaves the same in both.
        assert!(!tries_new_move(Fpu::Relative(0.25), 0.6));
        assert!(!tries_new_move(Fpu::Relative(0.25), -0.6));
    }

    #[test]
    fn fpu_json_and_sanitizing() {
        assert_eq!(serde_json::to_string(&Fpu::Relative(0.25)).unwrap(), r#"{"relative":0.25}"#);
        assert_eq!(serde_json::from_str::<Fpu>(r#"{"absolute":-0.5}"#).unwrap(), Fpu::Absolute(-0.5));
        assert_eq!(Fpu::Absolute(3.0).sanitized(), Fpu::Absolute(1.0));
        assert_eq!(Fpu::Relative(-1.0).sanitized(), Fpu::Relative(0.0));
    }
}
