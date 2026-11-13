//! Generic N-player agent interface (port & generalization of C++ `player.h`).

use std::task::Poll;

use crate::game::GameRules;

/// Stateful game-playing agent for an `N`-player game `R`.
///
/// Unlike C++ `Player<MoveT>` where `choose_move()` and `apply_opponent_move(move)` assumed
/// strictly alternating 2-player turns, in an `N`-player game there may be `N - 1` opponent
/// moves between two calls to `choose_move()` (or `0` opponent moves when `N = 1`!).
///
/// Contract:
/// - `choose_move(&mut self)` chooses a legal move for the current turn **and** advances the
///   player's internal state/tree by that move (matching C++ `beat-it`'s `choose_move`).
/// - `apply_external_move(&mut self, mv)` notifies this player of a move played by another player.
/// - `poll_choose_move(&mut self)` enables a worker thread (such as [`crate::matchup::run_multi_match`])
///   to multiplex multiple games concurrently when an agent (like [`crate::alphazero::AlphaZeroPlayer`])
///   is waiting on an asynchronous neural network batch evaluation.
pub trait Player<R: GameRules<N>, const N: usize> {
    /// Selects a valid move for the current game position and advances internal state by it.
    fn choose_move(&mut self) -> R::Move;

    /// Advances internal state by a valid move played by another participant.
    fn apply_external_move(&mut self, mv: R::Move);

    /// Non-blocking step for multiplexing multiple games on a single worker thread.
    ///
    /// - Returns `Poll::Ready(mv)` when the player has selected a move and advanced its state.
    /// - Returns `Poll::Pending` when the player has enqueued an asynchronous evaluation request
    ///   (e.g. to an [`crate::inference::InferenceService`]) and is waiting for the batch response.
    #[inline]
    fn poll_choose_move(&mut self) -> Poll<R::Move> {
        Poll::Ready(self.choose_move())
    }

    /// Flushes/wakes any background evaluation service before the caller parks waiting for responses.
    #[inline]
    fn notify_pending(&mut self) {}
}

impl<R: GameRules<N>, const N: usize, P: Player<R, N> + ?Sized> Player<R, N> for Box<P> {
    #[inline]
    fn choose_move(&mut self) -> R::Move {
        (**self).choose_move()
    }

    #[inline]
    fn apply_external_move(&mut self, mv: R::Move) {
        (**self).apply_external_move(mv);
    }

    #[inline]
    fn poll_choose_move(&mut self) -> Poll<R::Move> {
        (**self).poll_choose_move()
    }

    #[inline]
    fn notify_pending(&mut self) {
        (**self).notify_pending();
    }
}

impl<R: GameRules<N>, const N: usize, P: Player<R, N> + ?Sized> Player<R, N> for &mut P {
    #[inline]
    fn choose_move(&mut self) -> R::Move {
        (**self).choose_move()
    }

    #[inline]
    fn apply_external_move(&mut self, mv: R::Move) {
        (**self).apply_external_move(mv);
    }

    #[inline]
    fn poll_choose_move(&mut self) -> Poll<R::Move> {
        (**self).poll_choose_move()
    }

    #[inline]
    fn notify_pending(&mut self) {
        (**self).notify_pending();
    }
}
