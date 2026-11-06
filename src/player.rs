//! Generic N-player agent interface (port & generalization of C++ `player.h`).

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
pub trait Player<R: GameRules<N>, const N: usize> {
    /// Selects a valid move for the current game position and advances internal state by it.
    fn choose_move(&mut self) -> R::Move;

    /// Advances internal state by a valid move played by another participant.
    fn apply_external_move(&mut self, mv: R::Move);
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
}
