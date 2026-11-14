//! Root-Parallel MCTS / AlphaZero Tournament Player (`RootParallelPlayer`).
//!
//! ### Why Root Parallelism for Single-Game / Tournament Play?
//! During self-play training ([`crate::alphazero::run_self_play`]), **Game Parallelism**
//! multiplexes `M` independent games to saturate GPU batches and CPU cores without virtual loss.
//!
//! In a tournament match or interactive human vs. AI game, however, only **one game** is being
//! played at a time. **Root Parallelism** (Chaslot et al., *Parallel Monte-Carlo Tree Search*)
//! parallelizes a single game across `T` threads without locks, atomics, or virtual loss:
//! - Each of the `T` workers owns a completely independent, single-threaded [`MctsSearch`] arena
//!   and [`LeafEvaluator`] seeded with a distinct RNG stream.
//! - On [`Player::choose_move`], all `T` workers search from the root in parallel via
//!   [`std::thread::scope`] under a simulation or wall-clock time budget ([`RootParallelBudget`]).
//! - Because [`GameRules::write_valid_moves`] emits legal moves in a deterministic order, root
//!   child `i` corresponds to the same move across all `T` trees. The main thread sums `visits`
//!   and `value_sum` across all `T` roots in `O(T * B)` time, chooses the move with highest
//!   combined visits (or samples proportionally in the opening), and calls `advance(chosen_mv)`
//!   on all `T` workers so **every thread retains its surviving subtree across turns**.
//! - For neural-network-backed single-game play, [`crate::alphazero::AlphaZeroPlayer`] applies
//!   this same Root Parallelism over the non-blocking [`crate::inference::InferenceService`] event
//!   loop, multiplexing `K` root trees in batches without blocking on single-state evaluations.

use std::task::Poll;
use std::time::{Duration, Instant};

use rand::prelude::{RngExt, SeedableRng, SmallRng};

use crate::game::{Game, GameRules};
use crate::mcts::{LeafEvaluator, MctsSearch, SelectionPolicy};
use crate::player::Player;
use crate::tree::NodeId;

/// Search budget for [`RootParallelPlayer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootParallelBudget {
    /// Splits `total` simulations across the `T` worker threads (`total / T` per thread).
    TotalSimulations(usize),
    /// Runs `sims_per_thread` simulations on each of the `T` worker threads (`T * sims` total).
    SimulationsPerThread(usize),
    /// Runs simulations on all `T` threads until the wall-clock `Duration` elapses.
    MaxDuration(Duration),
}

/// Configuration for [`RootParallelPlayer`].
#[derive(Clone, Copy, Debug)]
pub struct RootParallelConfig {
    pub budget: RootParallelBudget,
    pub opening_moves: usize,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub arena_capacity_per_thread: usize,
    pub seed: u64,
}

impl RootParallelConfig {
    pub const fn new(budget: RootParallelBudget, seed: u64) -> Self {
        let default_cap = match budget {
            RootParallelBudget::TotalSimulations(total) => {
                let c = total * 8;
                if c > 2048 { c } else { 2048 }
            }
            RootParallelBudget::SimulationsPerThread(sims) => {
                let c = sims * 16;
                if c > 2048 { c } else { 2048 }
            }
            RootParallelBudget::MaxDuration(_) => 65_536,
        };
        Self {
            budget,
            opening_moves: 0,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.0,
            arena_capacity_per_thread: default_cap,
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

    pub const fn with_arena_capacity_per_thread(mut self, capacity: usize) -> Self {
        self.arena_capacity_per_thread = capacity;
        self
    }
}

struct RootWorker<R: GameRules<N>, const N: usize, S: SelectionPolicy, E: LeafEvaluator<R, N>> {
    search: MctsSearch<R, N, S>,
    evaluator: E,
    done: bool,
}

/// Aggregated statistics for a single legal root move across all `T` root-parallel threads.
#[derive(Clone, Copy, Debug)]
pub struct AggregatedRootChild<M> {
    pub mv: M,
    pub visits: u64,
    pub value_sum: f64,
}

impl<M: Copy> AggregatedRootChild<M> {
    #[inline]
    pub fn q_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / (self.visits as f64)
        }
    }
}

/// Multi-threaded Root-Parallel MCTS / AlphaZero [`Player`].
pub struct RootParallelPlayer<R, const N: usize, S, E>
where
    R: GameRules<N>,
    S: SelectionPolicy,
    E: LeafEvaluator<R, N>,
{
    workers: Box<[RootWorker<R, N, S, E>]>,
    budget: RootParallelBudget,
    opening_moves: usize,
    move_count: usize,
    rng: SmallRng,
    last_total_simulations: usize,
}

impl<R, const N: usize, S, E> RootParallelPlayer<R, N, S, E>
where
    R: GameRules<N>,
    R::Move: Send + Sync,
    R::State: Send + Sync,
    S: SelectionPolicy + Copy + Send + Sync,
    E: LeafEvaluator<R, N> + Send,
{
    /// Creates a new [`RootParallelPlayer`] with `num_threads` independent worker trees,
    /// constructing each thread's [`LeafEvaluator`] via `make_evaluator(thread_id, thread_seed)`.
    pub fn new(
        game: Game<R, N>,
        dummy_move: R::Move,
        selector: S,
        num_threads: usize,
        config: RootParallelConfig,
        mut make_evaluator: impl FnMut(usize, u64) -> E,
    ) -> Self {
        let threads = num_threads.max(1);
        let workers = (0..threads)
            .map(|t| {
                let thread_seed = config
                    .seed
                    .wrapping_add(((t as u64) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let eval_seed = thread_seed ^ 0xBF58_476D_1CE4_E5B9;
                RootWorker {
                    search: MctsSearch::new(
                        game.clone(),
                        dummy_move,
                        selector,
                        config.arena_capacity_per_thread,
                        thread_seed,
                    )
                    .with_dirichlet(config.dirichlet_alpha, config.dirichlet_epsilon),
                    evaluator: make_evaluator(t, eval_seed),
                    done: false,
                }
            })
            .collect::<Box<[_]>>();

        Self {
            workers,
            budget: config.budget,
            opening_moves: config.opening_moves,
            move_count: 0,
            rng: SmallRng::seed_from_u64(config.seed ^ 0x94D0_49BB_1331_11EB),
            last_total_simulations: 0,
        }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }

    /// Total simulations executed across all threads during the most recent `choose_move()` call.
    #[inline]
    pub const fn last_total_simulations(&self) -> usize {
        self.last_total_simulations
    }

    /// Aggregates `(mv, visits, value_sum)` across all `T` worker root nodes.
    pub fn aggregated_root_children(&self) -> Vec<AggregatedRootChild<R::Move>> {
        let first_children = self.workers[0].search.tree().active().children(NodeId::ROOT);
        let mut agg: Vec<AggregatedRootChild<R::Move>> = first_children
            .iter()
            .map(|c| AggregatedRootChild {
                mv: c.mv,
                visits: 0,
                value_sum: 0.0,
            })
            .collect();

        for worker in &self.workers {
            let children = worker.search.tree().active().children(NodeId::ROOT);
            debug_assert_eq!(
                children.len(),
                agg.len(),
                "all root-parallel workers must expand identical legal root moves"
            );
            for child in children {
                let slot = agg
                    .iter_mut()
                    .find(|s| s.mv == child.mv)
                    .expect("worker root child move must exist in aggregated move list");
                slot.visits += child.payload.visits as u64;
                slot.value_sum += child.payload.value_sum as f64;
            }
        }
        agg
    }

    #[inline]
    fn worker_target_simulations(budget: RootParallelBudget, t: usize, num_threads: usize) -> usize {
        match budget {
            RootParallelBudget::TotalSimulations(total) => {
                let base = total / num_threads;
                let rem = usize::from(t < (total % num_threads));
                (base + rem).max(1)
            }
            RootParallelBudget::SimulationsPerThread(sims) => sims.max(1),
            RootParallelBudget::MaxDuration(_) => 1,
        }
    }

    fn run_parallel_search(&mut self) -> usize {
        let num_threads = self.workers.len();
        let budget = self.budget;
        let mut sims_per_worker = vec![0usize; num_threads];

        std::thread::scope(|scope| {
            for (t, (worker, out_sims)) in self
                .workers
                .iter_mut()
                .zip(sims_per_worker.iter_mut())
                .enumerate()
            {
                scope.spawn(move || {
                    match budget {
                        RootParallelBudget::TotalSimulations(_)
                        | RootParallelBudget::SimulationsPerThread(_) => {
                            let my_sims = Self::worker_target_simulations(budget, t, num_threads);
                            worker.search.run_simulations(my_sims, &mut worker.evaluator);
                            *out_sims = my_sims;
                        }
                        RootParallelBudget::MaxDuration(duration) => {
                            let deadline = Instant::now() + duration;
                            let mut count = 0usize;
                            // Always run at least 1 simulation so the root is guaranteed to expand.
                            loop {
                                worker.search.simulate_once(&mut worker.evaluator);
                                count += 1;
                                if (count & 15) == 0 && Instant::now() >= deadline {
                                    break;
                                }
                            }
                            *out_sims = count;
                        }
                    }
                });
            }
        });

        sims_per_worker.into_iter().sum()
    }

    fn finalize_move_choice(&mut self) -> R::Move {
        let agg = self.aggregated_root_children();
        assert!(
            !agg.is_empty(),
            "RootParallelPlayer root has no legal moves to choose from"
        );

        self.move_count += 1;
        let chosen_idx = if self.move_count <= self.opening_moves {
            let total_visits: u64 = agg.iter().map(|c| c.visits).sum();
            if total_visits == 0 {
                self.rng.random_range(0..agg.len())
            } else {
                let mut target = self.rng.random_range(0..total_visits);
                let mut picked = agg.len() - 1;
                for (idx, child) in agg.iter().enumerate() {
                    if target < child.visits {
                        picked = idx;
                        break;
                    }
                    target -= child.visits;
                }
                picked
            }
        } else {
            agg.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.visits
                        .cmp(&b.visits)
                        .then_with(|| a.q_value().total_cmp(&b.q_value()))
                })
                .map(|(idx, _)| idx)
                .unwrap()
        };

        let chosen_mv = agg[chosen_idx].mv;
        for worker in &mut self.workers {
            worker.search.advance(chosen_mv);
            worker.done = false;
        }
        chosen_mv
    }
}

impl<R, const N: usize, S, E> Player<R, N> for RootParallelPlayer<R, N, S, E>
where
    R: GameRules<N>,
    R::Move: Send + Sync,
    R::State: Send + Sync,
    S: SelectionPolicy + Copy + Send + Sync,
    E: LeafEvaluator<R, N> + Send,
{
    fn poll_choose_move(&mut self) -> Poll<R::Move> {
        if matches!(self.budget, RootParallelBudget::MaxDuration(_)) {
            self.last_total_simulations = self.run_parallel_search();
            return Poll::Ready(self.finalize_move_choice());
        }

        let num_threads = self.workers.len();
        let budget = self.budget;
        let mut all_done = true;

        for (t, worker) in self.workers.iter_mut().enumerate() {
            if !worker.done {
                let my_sims = Self::worker_target_simulations(budget, t, num_threads);
                if worker
                    .search
                    .poll_simulations(my_sims, &mut worker.evaluator)
                    .is_ready()
                {
                    worker.done = true;
                } else {
                    all_done = false;
                }
            }
        }

        if !all_done {
            return Poll::Pending;
        }

        self.last_total_simulations = (0..num_threads)
            .map(|t| Self::worker_target_simulations(budget, t, num_threads))
            .sum();
        Poll::Ready(self.finalize_move_choice())
    }

    #[inline]
    fn notify_pending(&mut self) {
        for worker in &mut self.workers {
            if !worker.done {
                worker.evaluator.notify_pending();
            }
        }
    }

    fn choose_move(&mut self) -> R::Move {
        if E::IS_ASYNC || self.workers.len() == 1 {
            loop {
                if let Poll::Ready(mv) = self.poll_choose_move() {
                    return mv;
                }
                self.notify_pending();
                std::thread::park();
            }
        } else {
            self.last_total_simulations = self.run_parallel_search();
            self.finalize_move_choice()
        }
    }

    fn apply_external_move(&mut self, mv: R::Move) {
        self.move_count += 1;
        for worker in &mut self.workers {
            worker.search.advance(mv);
            worker.done = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphazero::{AlphaZeroConfig, AlphaZeroPlayer};
    use crate::game::PlayerIndex;
    use crate::games::tic_tac_toe::{EMPTY_STATE, NO_MOVE, TicTacToe, TttGame, TttHeuristic};
    use crate::inference::{FnBatchEvaluator, InferenceService};
    use crate::matchup::run_multi_match;
    use crate::mcts::{PlayoutEvaluator, Uct};
    use crate::minimax::MinimaxPlayer;

    #[test]
    fn root_parallel_mcts_aggregates_visits_and_holds_minimax() {
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

        // 4 threads x 200 simulations = 800 total simulations per move.
        let f_root_par = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(RootParallelPlayer::new(
                *g,
                NO_MOVE,
                Uct::new(1.0),
                4,
                RootParallelConfig::new(RootParallelBudget::TotalSimulations(800), seed),
                |_t, eval_seed| PlayoutEvaluator::new(eval_seed),
            ))
        };
        let f_mm2 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 2, seed))
        };

        let summary = run_multi_match(&game, [&f_root_par, &f_mm2], 20, 2, 2026);
        assert!(
            summary.wins[0] + summary.draws >= 19,
            "expected 4-thread RootParallelPlayer to draw/beat Minimax(depth=2), got {summary:?}"
        );
    }

    #[test]
    fn root_parallel_time_budget_and_alphazero_inference_batching_work() {
        // Test 1: Time-bounded RootParallelBudget::MaxDuration.
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let mut timed_player = RootParallelPlayer::new(
            game,
            NO_MOVE,
            Uct::default(),
            4,
            RootParallelConfig::new(
                RootParallelBudget::MaxDuration(Duration::from_millis(5)),
                123,
            ),
            |_t, eval_seed| PlayoutEvaluator::new(eval_seed),
        );
        let mv = timed_player.choose_move();
        assert!((0..9).contains(&mv));
        assert!(timed_player.last_total_simulations() >= 4);

        // Test 2: Root-Parallel `AlphaZeroPlayer` with 16 parallel root trees sharing an InferenceService.
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
        let mut az_root_par = AlphaZeroPlayer::new(
            game,
            &service,
            AlphaZeroConfig::new(16, 20, 999),
        );
        let az_mv = az_root_par.choose_move();
        assert!((0..9).contains(&az_mv));
        assert!(service.batch_size_stats().mean() >= 8.0);
    }
}
