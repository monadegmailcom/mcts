//! Single-game and multithreaded `N`-player tournament runners (port & generalization of C++
//! `match.h` and `evaluation::RandomPlayer` / `TimingPlayerWrapper` in `evaluation.h`).
//!
//! Note on naming: `match` is a reserved control-flow keyword in Rust, so this module is named
//! `matchup` (echoing `evaluate_matchup` in `evaluation.h`).
//!
//! ### Architectural Evolution from C++ `beat-it`
//! - **Arbitrary `N` Players & Non-Alternating Turns**: [`play_match`] queries
//!   `game.current_player()` at every ply instead of swapping 2 player pointers, supporting
//!   `N = 1` puzzles, `N = 2` duels, `N >= 3` multi-player games, and extra-turn rules.
//! - **Cyclic Seat Rotation**: [`run_multi_match`] rotates `N` participants across all `N` seats
//!   (`seat = (participant + round) % N`) so every agent experiences every seat equally.
//! - **Data-Race-Free Deterministic Seeding**: Replaces C++ `MultiMatch`'s unsynchronized shared
//!   `std::mt19937 g` across worker threads with a pure SplitMix64 seed mixer over
//!   `(base_seed, round_index, participant_index)`.
//! - **Fearless Concurrency (`std::thread::scope`)**: Workers borrow `&initial_game` and player
//!   factories directly from the caller's stack without `Arc` or `'static` bounds, returning
//!   thread-local summaries by value on join—zero `Mutex` needed!

use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

use rand::prelude::{IndexedRandom, SeedableRng, SmallRng};

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::player::Player;
use crate::statistics::Statistics;

/// Uniformly random legal-move agent for any `N`-player game (port of `evaluation::RandomPlayer`).
///
/// Reuses an internal `move_buf` so `choose_move()` performs zero heap allocations after warmup.
pub struct RandomPlayer<R: GameRules<N>, const N: usize> {
    game: Game<R, N>,
    rng: SmallRng,
    move_buf: Vec<R::Move>,
}

impl<R: GameRules<N>, const N: usize> RandomPlayer<R, N> {
    pub fn new(game: Game<R, N>, seed: u64) -> Self {
        Self {
            game,
            rng: SmallRng::seed_from_u64(seed),
            move_buf: Vec::new(),
        }
    }

    #[inline]
    pub const fn game(&self) -> &Game<R, N> {
        &self.game
    }
}

impl<R: GameRules<N>, const N: usize> Player<R, N> for RandomPlayer<R, N> {
    fn choose_move(&mut self) -> R::Move {
        self.game.write_valid_moves(&mut self.move_buf);
        let &mv = self
            .move_buf
            .choose(&mut self.rng)
            .expect("RandomPlayer::choose_move called on position with no legal moves");
        self.game = self.game.apply(mv);
        mv
    }

    fn apply_external_move(&mut self, mv: R::Move) {
        self.game = self.game.apply(mv);
    }
}

/// Result and per-seat timing breakdown of a single completed game.
#[derive(Clone, Debug)]
pub struct MatchOutcome<const N: usize> {
    /// Terminal payoffs indexed by game seat (`0..N`).
    pub payoffs: [f32; N],
    /// Total number of plies played from the starting position to termination.
    pub moves_played: usize,
    /// Cumulative wall-clock time spent in `choose_move` + `apply_external_move` per seat.
    pub seat_durations: [Duration; N],
    /// Per-move `choose_move()` latency distribution in microseconds (`us`) per seat.
    pub choose_time_us: [Statistics; N],
}

impl<const N: usize> MatchOutcome<N> {
    /// If a single seat strictly achieved the highest terminal payoff, returns that winner.
    pub fn unique_winner(&self) -> Option<PlayerIndex<N>> {
        unique_winner_index(&self.payoffs).map(|idx| PlayerIndex::new(idx as u8))
    }

    /// Returns `true` when all `N` seats tied with identical terminal payoffs (e.g. `[0.0; N]`).
    pub fn is_draw(&self) -> bool {
        all_payoffs_equal(&self.payoffs)
    }
}

/// Plays a single game from `game` until termination.
///
/// Works seamlessly with homogeneous arrays (`&mut [p1, p2]`), stack trait objects
/// (`&mut [&mut p1 as &mut dyn Player<R, N>, &mut p2]`), or boxed players (`&mut [b1, b2]`).
pub fn play_match<R, const N: usize, P>(game: Game<R, N>, players: &mut [P; N]) -> MatchOutcome<N>
where
    R: GameRules<N>,
    P: Player<R, N>,
{
    play_match_observed(game, players, |_, _, _| {})
}

/// Plays a single game from `game` until termination, invoking `on_move(&game_after, actor, mv)`
/// after every ply (zero-cost closure replacement for `Match::report` in C++ `match.h`).
pub fn play_match_observed<R, const N: usize, P, F>(
    mut game: Game<R, N>,
    players: &mut [P; N],
    mut on_move: F,
) -> MatchOutcome<N>
where
    R: GameRules<N>,
    P: Player<R, N>,
    F: FnMut(&Game<R, N>, PlayerIndex<N>, R::Move),
{
    const { assert!(N >= 1, "play_match requires at least 1 player (N >= 1)") };

    let mut seat_durations = [Duration::ZERO; N];
    let mut choose_time_us = [Statistics::new(); N];
    let mut moves_played = 0usize;

    loop {
        if let GameStatus::Terminal(payoffs) = game.status() {
            return MatchOutcome {
                payoffs,
                moves_played,
                seat_durations,
                choose_time_us,
            };
        }

        let actor = game.current_player();
        let actor_idx = actor.index();

        let start = Instant::now();
        let mv = players[actor_idx].choose_move();
        let choose_elapsed = start.elapsed();

        seat_durations[actor_idx] += choose_elapsed;
        choose_time_us[actor_idx].update((choose_elapsed.as_secs_f64() * 1_000_000.0) as f32);

        game = game.apply(mv);

        for (seat, player) in players.iter_mut().enumerate() {
            if seat != actor_idx {
                let notify_start = Instant::now();
                player.apply_external_move(mv);
                seat_durations[seat] += notify_start.elapsed();
            }
        }

        moves_played += 1;
        on_move(&game, actor, mv);
    }
}

/// Thread-safe factory signature constructing a fresh boxed [`Player`] for a given starting
/// `Game` and deterministic per-round-per-participant 64-bit seed.
pub type BoxedPlayerFactory<'a, R, const N: usize> =
    &'a (dyn Fn(&Game<R, N>, u64) -> Box<dyn Player<R, N> + 'a> + Sync + 'a);

/// Aggregated tournament statistics across `rounds` games for `N` participants
/// (generalized port of C++ `MultiMatch` getters + `TimingPlayerWrapper` stats).
#[derive(Clone, Debug)]
pub struct MultiMatchSummary<const N: usize> {
    pub rounds: usize,
    /// Sole first-place finishes per participant (`0..N`).
    /// For `N == 1`, counts rounds with strictly positive terminal payoff (`payoffs[0] > 0.0`).
    pub wins: [usize; N],
    /// Rounds where all `N` participants tied (for `N == 1`, rounds with `payoffs[0] == 0.0`).
    pub draws: usize,
    /// Rounds in `N >= 3` games where `2..N` participants tied for highest payoff.
    pub partial_ties: usize,
    /// Sum of terminal payoffs earned by each participant across all rounds.
    pub payoff_sums: [f64; N],
    /// Total thinking + notification wall-clock duration per participant.
    pub durations: [Duration; N],
    /// Per-move `choose_move()` latency statistics (in microseconds) per participant.
    pub choose_time_us: [Statistics; N],
}

impl<const N: usize> MultiMatchSummary<N> {
    fn empty() -> Self {
        Self {
            rounds: 0,
            wins: [0; N],
            draws: 0,
            partial_ties: 0,
            payoff_sums: [0.0; N],
            durations: [Duration::ZERO; N],
            choose_time_us: [Statistics::new(); N],
        }
    }

    /// Average payoff per game for `participant`.
    #[inline]
    pub fn mean_payoff(&self, participant: usize) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.payoff_sums[participant] / (self.rounds as f64)
        }
    }

    fn record_round(&mut self, round_idx: usize, outcome: &MatchOutcome<N>) {
        self.rounds += 1;

        // Participant `p` sat at seat `(p + round_idx) % N`.
        for p in 0..N {
            let seat = (p + round_idx) % N;
            self.payoff_sums[p] += outcome.payoffs[seat] as f64;
            self.durations[p] += outcome.seat_durations[seat];
            self.choose_time_us[p].join(&outcome.choose_time_us[seat]);
        }

        if N == 1 {
            if outcome.payoffs[0] > 0.0 {
                self.wins[0] += 1;
            } else if outcome.payoffs[0] == 0.0 {
                self.draws += 1;
            }
            return;
        }

        let mut max_payoff = f32::NEG_INFINITY;
        for &v in &outcome.payoffs {
            if v > max_payoff {
                max_payoff = v;
            }
        }

        let mut top_count = 0usize;
        let mut last_top_seat = 0usize;
        for (seat, &v) in outcome.payoffs.iter().enumerate() {
            if (v - max_payoff).abs() <= 1e-6 {
                top_count += 1;
                last_top_seat = seat;
            }
        }

        if top_count == 1 {
            // Invert `seat = (p + round_idx) % N` -> `p = (seat + N - (round_idx % N)) % N`.
            let winner_participant = (last_top_seat + N - (round_idx % N)) % N;
            self.wins[winner_participant] += 1;
        } else if top_count == N {
            self.draws += 1;
        } else {
            self.partial_ties += 1;
        }
    }

    fn merge(&mut self, other: &Self) {
        self.rounds += other.rounds;
        self.draws += other.draws;
        self.partial_ties += other.partial_ties;
        for p in 0..N {
            self.wins[p] += other.wins[p];
            self.payoff_sums[p] += other.payoff_sums[p];
            self.durations[p] += other.durations[p];
            self.choose_time_us[p].join(&other.choose_time_us[p]);
        }
    }
}

struct MatchSlot<'a, R: GameRules<N>, const N: usize> {
    round_idx: usize,
    game: Game<R, N>,
    seats: [Box<dyn Player<R, N> + 'a>; N],
    moves_played: usize,
    seat_durations: [Duration; N],
    choose_time_us: [Statistics; N],
    turn_accum_duration: Duration,
}

/// Runs `rounds` games across `num_threads` OS threads with cyclic seat rotation
/// (port & `N`-player generalization of C++ `MultiMatch`).
///
/// - On round `r`, participant `p` (`0..N`) is assigned to seat `(p + r) % N`.
/// - When one or more participants return [`Poll::Pending`] from [`Player::poll_choose_move`]
///   (such as [`crate::alphazero::AlphaZeroPlayer`] waiting on a neural network evaluation),
///   each worker thread automatically multiplexes up to `64` concurrent games on demand so
///   their neural network evaluations are batched together!
/// - When all participants are synchronous CPU players, `poll_choose_move` always returns
///   [`Poll::Ready`], so each worker thread allocates only `1` slot and runs games sequentially.
/// - Because seeding is derived deterministically from `(base_seed, r, p)`, total win/draw/payoff
///   counts are **100% deterministic and identical** regardless of `num_threads` (`1` vs `8`)!
pub fn run_multi_match<'a, R, const N: usize>(
    initial_game: &Game<R, N>,
    factories: [BoxedPlayerFactory<'a, R, N>; N],
    rounds: usize,
    num_threads: usize,
    base_seed: u64,
) -> MultiMatchSummary<N>
where
    R: GameRules<N>,
    R::State: Sync,
{
    const { assert!(N >= 1, "run_multi_match requires at least 1 player (N >= 1)") };

    if rounds == 0 {
        return MultiMatchSummary::empty();
    }

    let worker_count = num_threads.max(1).min(rounds);
    let max_slots_per_worker = rounds.div_ceil(worker_count).clamp(1, 64);
    let next_round = AtomicUsize::new(0);

    let make_slot = |r: usize| -> MatchSlot<'a, R, N> {
        let seats: [Box<dyn Player<R, N> + 'a>; N] = std::array::from_fn(|seat| {
            let participant = (seat + N - (r % N)) % N;
            let seed = mix_round_seed(base_seed, r as u64, participant as u64);
            factories[participant](initial_game, seed)
        });
        MatchSlot {
            round_idx: r,
            game: initial_game.clone(),
            seats,
            moves_played: 0,
            seat_durations: [Duration::ZERO; N],
            choose_time_us: [Statistics::new(); N],
            turn_accum_duration: Duration::ZERO,
        }
    };

    // Advances `slot` until an asynchronous player returns `Poll::Pending` (`true`) or the round
    // pool is exhausted (`false`).
    let step_slot = |slot: &mut MatchSlot<'a, R, N>,
                     local_summary: &mut MultiMatchSummary<N>|
     -> bool {
        loop {
            if let GameStatus::Terminal(payoffs) = slot.game.status() {
                let outcome = MatchOutcome {
                    payoffs,
                    moves_played: slot.moves_played,
                    seat_durations: slot.seat_durations,
                    choose_time_us: slot.choose_time_us,
                };
                local_summary.record_round(slot.round_idx, &outcome);

                let r = next_round.fetch_add(1, Ordering::Relaxed);
                if r < rounds {
                    *slot = make_slot(r);
                    continue;
                } else {
                    return false;
                }
            }

            let actor_idx = slot.game.current_player().index();
            let start = Instant::now();
            match slot.seats[actor_idx].poll_choose_move() {
                Poll::Pending => {
                    slot.turn_accum_duration += start.elapsed();
                    return true;
                }
                Poll::Ready(mv) => {
                    let choose_elapsed = slot.turn_accum_duration + start.elapsed();
                    slot.turn_accum_duration = Duration::ZERO;
                    slot.seat_durations[actor_idx] += choose_elapsed;
                    slot.choose_time_us[actor_idx]
                        .update((choose_elapsed.as_secs_f64() * 1_000_000.0) as f32);

                    slot.game = slot.game.apply(mv);
                    for (seat, player) in slot.seats.iter_mut().enumerate() {
                        if seat != actor_idx {
                            let notify_start = Instant::now();
                            player.apply_external_move(mv);
                            slot.seat_durations[seat] += notify_start.elapsed();
                        }
                    }
                    slot.moves_played += 1;
                }
            }
        }
    };

    let run_worker = || {
        let mut local_summary = MultiMatchSummary::<N>::empty();
        let mut slots: Vec<MatchSlot<'a, R, N>> = Vec::new();
        let mut pool_exhausted = false;

        loop {
            // 1. Step all currently active slots that have received responses.
            let mut i = 0;
            while i < slots.len() {
                if step_slot(&mut slots[i], &mut local_summary) {
                    i += 1;
                } else {
                    slots.swap_remove(i);
                }
            }

            // 2. If all existing slots are waiting on an async evaluation (or none exist yet),
            //    dynamically start a new round on demand up to `max_slots_per_worker`.
            if slots.len() < max_slots_per_worker && !pool_exhausted {
                let r = next_round.fetch_add(1, Ordering::Relaxed);
                if r < rounds {
                    let mut slot = make_slot(r);
                    if step_slot(&mut slot, &mut local_summary) {
                        slots.push(slot);
                    }
                    continue;
                } else {
                    pool_exhausted = true;
                }
            }

            if slots.is_empty() {
                break;
            }

            // 3. All active slots are waiting on async evaluations: flush notifications and park.
            for slot in &mut slots {
                let actor_idx = slot.game.current_player().index();
                slot.seats[actor_idx].notify_pending();
            }
            std::thread::park();
        }
        local_summary
    };

    // Fast single-threaded path avoids OS thread spawn overhead altogether.
    if worker_count == 1 {
        return run_worker();
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(run_worker));
        }

        let mut total = MultiMatchSummary::<N>::empty();
        for handle in handles {
            let worker_summary = handle.join().expect("MultiMatch worker thread panicked");
            total.merge(&worker_summary);
        }
        total
    })
}

/// SplitMix64-based bijection mapping `(base_seed, round, participant)` to an independent `u64` seed.
#[inline]
fn mix_round_seed(base_seed: u64, round: u64, participant: u64) -> u64 {
    let mut z = base_seed
        .wrapping_add(round.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(participant.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn unique_winner_index<const N: usize>(payoffs: &[f32; N]) -> Option<usize> {
    if N == 1 {
        return (payoffs[0] > 0.0).then_some(0);
    }
    let mut max_val = f32::NEG_INFINITY;
    for &v in payoffs {
        if v > max_val {
            max_val = v;
        }
    }
    let mut count = 0usize;
    let mut idx = 0usize;
    for (i, &v) in payoffs.iter().enumerate() {
        if (v - max_val).abs() <= 1e-6 {
            count += 1;
            idx = i;
        }
    }
    (count == 1).then_some(idx)
}

#[inline]
fn all_payoffs_equal<const N: usize>(payoffs: &[f32; N]) -> bool {
    let first = payoffs[0];
    payoffs.iter().all(|&v| (v - first).abs() <= 1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphazero::{AlphaZeroConfig, AlphaZeroPlayer};
    use crate::games::tic_tac_toe::{EMPTY_STATE, TicTacToe, TttGame, TttHeuristic};
    use crate::inference::{FnBatchEvaluator, InferenceService};
    use crate::minimax::MinimaxPlayer;

    #[test]
    fn multithreaded_multi_match_is_bit_exact_deterministic_across_thread_counts() {
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let f1 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(RandomPlayer::new(*g, seed))
        };
        let f2 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(RandomPlayer::new(*g, seed))
        };

        let single = run_multi_match(&game, [&f1, &f2], 120, 1, 42);
        let multi = run_multi_match(&game, [&f1, &f2], 120, 4, 42);

        assert_eq!(single.rounds, 120);
        assert_eq!(single.wins, multi.wins);
        assert_eq!(single.draws, multi.draws);
        assert!((single.payoff_sums[0] - multi.payoff_sums[0]).abs() < 1e-9);
        assert!((single.payoff_sums[1] - multi.payoff_sums[1]).abs() < 1e-9);
    }

    #[test]
    fn multi_match_multiplexes_single_tree_alphazero_games_into_nn_batches() {
        let evaluator = FnBatchEvaluator(
            |states: &[[f32; 27]], policies: &mut [[f32; 9]], values: &mut [[f32; 2]]| {
                for ((s, p), v) in states.iter().zip(policies.iter_mut()).zip(values.iter_mut()) {
                    *p = [0.2, 0.0, 0.2, 0.0, 0.5, 0.0, 0.2, 0.0, 0.2];
                    let my_pieces: f32 = s[0..9].iter().sum();
                    let opp_pieces: f32 = s[9..18].iter().sum();
                    let adv = ((my_pieces - opp_pieces) * 0.1).clamp(-0.9, 0.9);
                    *v = [adv, -adv];
                }
            },
        );
        let service = InferenceService::new(evaluator, 32, 1);
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

        // `num_trees = 1` (single MCTS tree per game, zero root parallelism!):
        // `run_multi_match` automatically interleaves the 32 games via `poll_choose_move`.
        let f_az = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(AlphaZeroPlayer::new(
                *g,
                &service,
                AlphaZeroConfig::new(1, 24, seed).with_dirichlet(0.3, 0.0),
            ))
        };
        let f_mm = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2> + '_> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 2, seed))
        };

        let summary_1t = run_multi_match(&game, [&f_az, &f_mm], 32, 1, 2026);
        let batch_stats = service.batch_size_stats();
        assert_eq!(summary_1t.rounds, 32);
        assert!(
            batch_stats.mean() >= 8.0,
            "expected MultiMatch to batch single-tree AlphaZero evaluations across concurrent games, got {batch_stats}"
        );

        let summary_2t = run_multi_match(&game, [&f_az, &f_mm], 32, 2, 2026);
        assert_eq!(summary_1t.wins, summary_2t.wins);
        assert_eq!(summary_1t.draws, summary_2t.draws);
    }
}
