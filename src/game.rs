//! Core N-player game abstraction (port & generalization of C++ `game.h`).
//!
//! Generalizes two-player zero-sum games to `const N: usize` players:
//! - `N = 1`: Single-player puzzles and optimization (e.g., Rubik's Cube, Twist Puzzle).
//! - `N = 2`: Two-player zero-sum or general-sum games (e.g., Tic-Tac-Toe, Nim, Chess).
//! - `N > 2`: Multi-player games with arbitrary `[f32; N]` payoff vectors.

use std::fmt::{self, Debug};

/// Strongly-typed player index guaranteed to satisfy `0 <= index < N`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlayerIndex<const N: usize>(u8);

impl<const N: usize> PlayerIndex<N> {
    /// Player 0 (always valid whenever `N >= 1`).
    pub const P0: Self = Self::new(0);

    /// Constructs a checked `PlayerIndex<N>`.
    ///
    /// Usable in `const` contexts (`PlayerIndex::<2>::new(1)` is validated at compile time).
    #[inline]
    pub const fn new(index: u8) -> Self {
        assert!(N >= 1 && N <= (u8::MAX as usize), "N must be in 1..=255");
        assert!((index as usize) < N, "player index out of bounds for N");
        Self(index)
    }

    /// Returns the player index as `usize` for array indexing (`payoffs[player.index()]`).
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the underlying `u8` representation.
    #[inline]
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// Advances to the next player in round-robin order `(index + 1) % N`.
    ///
    /// - For `N = 1`: `0 -> 0` (no-op).
    /// - For `N = 2`: `0 -> 1`, `1 -> 0` (replaces C++ `toggle(PlayerIndex)`).
    /// - For `N > 2`: `0 -> 1 -> ... -> N - 1 -> 0`.
    #[inline]
    pub const fn next(self) -> Self {
        Self(((self.0 as usize + 1) % N) as u8)
    }
}

impl PlayerIndex<2> {
    pub const PLAYER1: Self = Self::new(0);
    pub const PLAYER2: Self = Self::new(1);

    #[inline]
    pub const fn toggle(self) -> Self {
        self.next()
    }
}

impl<const N: usize> fmt::Display for PlayerIndex<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.0 + 1)
    }
}

/// Status of a game position for `N` players.
///
/// Unifies terminal game outcomes across single-player solvers (`N = 1`),
/// two-player games (`N = 2`), and multi-player games (`N > 2`) using a
/// stack-allocated payoff vector `[f32; N]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GameStatus<const N: usize> {
    /// The game has not finished; at least one valid move must exist.
    Ongoing,
    /// The game has terminated with explicit payoffs for each player `0..N`.
    ///
    /// Standard convention: `+1.0` for win/solved, `0.0` for draw, `-1.0` for loss.
    Terminal([f32; N]),
}

impl<const N: usize> GameStatus<N> {
    /// Terminal draw where all `N` players receive `0.0`.
    #[inline]
    pub const fn draw() -> Self {
        Self::Terminal([0.0; N])
    }

    /// Terminal outcome where `winner` receives `+1.0` and all other players receive `-1.0`.
    ///
    /// For `N = 1`, returns `Terminal([1.0])` ("puzzle solved").
    /// For `N = 2`, returns `Terminal([1.0, -1.0])` or `Terminal([-1.0, 1.0])`.
    #[inline]
    pub fn single_winner(winner: PlayerIndex<N>) -> Self {
        let mut payoffs = [-1.0; N];
        payoffs[winner.index()] = 1.0;
        Self::Terminal(payoffs)
    }

    /// Strict zero-sum terminal outcome where `winner` receives `+1.0` and the
    /// remaining `N - 1` players split `-1.0` equally (`-1.0 / (N - 1)`), summing to `0.0`.
    ///
    /// For `N = 2`, identical to [`Self::single_winner`].
    #[inline]
    pub fn zero_sum_winner(winner: PlayerIndex<N>) -> Self {
        if N == 1 {
            return Self::Terminal([1.0; N]);
        }
        let loss = -1.0 / ((N - 1) as f32);
        let mut payoffs = [loss; N];
        payoffs[winner.index()] = 1.0;
        Self::Terminal(payoffs)
    }

    #[inline]
    pub const fn is_ongoing(&self) -> bool {
        matches!(self, Self::Ongoing)
    }

    #[inline]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }

    #[inline]
    pub const fn payoffs(&self) -> Option<[f32; N]> {
        match *self {
            Self::Ongoing => None,
            Self::Terminal(p) => Some(p),
        }
    }
}

/// Compile-time game specification (replaces C++ `GameState<MoveT, StateT>` specialization
/// and `boost::iterator_facade` without CRTP boilerplate).
///
/// Implementors are typically zero-sized marker types (e.g., `pub struct TicTacToe;`).
pub trait GameRules<const N: usize>: Sized {
    /// Action/move representation (small `Copy` value, e.g. `u8` or compact struct).
    type Move: Copy + Eq + Debug + Send + Sync;

    /// Board/environment state representation.
    type State: Clone + Debug + Send + Sync;

    /// Zero-allocation lazy iterator over valid moves (Rust GAT: Generic Associated Type).
    type MoveIter<'a>: Iterator<Item = Self::Move>
    where
        Self: 'a;

    /// Returns a lazy iterator over all legal moves for `player` in `state`.
    fn valid_moves<'a>(state: &'a Self::State, player: PlayerIndex<N>) -> Self::MoveIter<'a>;

    /// Appends all legal moves into a reusable buffer `out`, avoiding heap allocations.
    ///
    /// Games can override this when a tight bulk scan is faster than stepping `MoveIter`
    /// (preserving the optimization in C++ `GameStateBase::get_valid_moves`).
    #[inline]
    fn write_valid_moves(state: &Self::State, player: PlayerIndex<N>, out: &mut Vec<Self::Move>) {
        out.clear();
        out.extend(Self::valid_moves(state, player));
    }

    /// Pure state transition: returns the new state after `player` plays a valid `mv`.
    fn apply(state: &Self::State, player: PlayerIndex<N>, mv: Self::Move) -> Self::State;

    /// Determines whose turn is next after `current_player` plays `mv` leading to `next_state`.
    ///
    /// Defaults to round-robin `current_player.next()`, but can be overridden for games
    /// with extra turns (e.g., Mancala, Dots-and-Boxes) or player elimination.
    #[inline]
    fn next_player(
        _next_state: &Self::State,
        current_player: PlayerIndex<N>,
        _mv: Self::Move,
    ) -> PlayerIndex<N> {
        current_player.next()
    }

    /// Evaluates whether `(state, current_player)` is `Ongoing` or `Terminal([f32; N])`.
    fn status(state: &Self::State, current_player: PlayerIndex<N>) -> GameStatus<N>;
}

/// Pairs the active `PlayerIndex<N>` with `R::State` (port of C++ `Game<MoveT, StateT>`).
pub struct Game<R: GameRules<N>, const N: usize> {
    player: PlayerIndex<N>,
    state: R::State,
}

// Manual `Clone`, `Copy`, `Debug`, `PartialEq`, `Eq` implementations so `R` itself
// does not need `#[derive(Clone, ...)]`—only `R::State` does.
impl<R: GameRules<N>, const N: usize> Clone for Game<R, N> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            player: self.player,
            state: self.state.clone(),
        }
    }
}

impl<R: GameRules<N>, const N: usize> Copy for Game<R, N> where R::State: Copy {}

impl<R: GameRules<N>, const N: usize> Debug for Game<R, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Game")
            .field("player", &self.player)
            .field("state", &self.state)
            .finish()
    }
}

impl<R: GameRules<N>, const N: usize> PartialEq for Game<R, N>
where
    R::State: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.player == other.player && self.state == other.state
    }
}

impl<R: GameRules<N>, const N: usize> Eq for Game<R, N> where R::State: Eq {}

impl<R: GameRules<N>, const N: usize> Game<R, N> {
    #[inline]
    pub const fn new(player: PlayerIndex<N>, state: R::State) -> Self {
        Self { player, state }
    }

    #[inline]
    pub const fn current_player(&self) -> PlayerIndex<N> {
        self.player
    }

    #[inline]
    pub const fn state(&self) -> &R::State {
        &self.state
    }

    #[inline]
    pub fn status(&self) -> GameStatus<N> {
        R::status(&self.state, self.player)
    }

    #[inline]
    pub fn valid_moves(&self) -> R::MoveIter<'_> {
        R::valid_moves(&self.state, self.player)
    }

    #[inline]
    pub fn write_valid_moves(&self, out: &mut Vec<R::Move>) {
        R::write_valid_moves(&self.state, self.player, out);
    }

    #[inline]
    pub fn apply(&self, mv: R::Move) -> Self {
        let next_state = R::apply(&self.state, self.player, mv);
        let next_player = R::next_player(&next_state, self.player, mv);
        Self {
            player: next_player,
            state: next_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_index_cycles_for_1_2_and_3_players() {
        // N = 1 (single-player puzzle)
        let p1 = PlayerIndex::<1>::P0;
        assert_eq!(p1.next(), PlayerIndex::<1>::P0);

        // N = 2 (two-player toggle)
        let a = PlayerIndex::<2>::PLAYER1;
        let b = a.toggle();
        assert_eq!(b, PlayerIndex::<2>::PLAYER2);
        assert_eq!(b.toggle(), a);

        // N = 3 (three-player round-robin)
        let q0 = PlayerIndex::<3>::new(0);
        let q1 = q0.next();
        let q2 = q1.next();
        assert_eq!(q1.index(), 1);
        assert_eq!(q2.index(), 2);
        assert_eq!(q2.next(), q0);
    }

    #[test]
    fn zero_sum_payoffs_sum_to_zero_for_multiplayer() {
        let status = GameStatus::<4>::zero_sum_winner(PlayerIndex::<4>::new(2));
        let payoffs = status.payoffs().unwrap();
        assert_eq!(payoffs[2], 1.0);
        let sum: f32 = payoffs.iter().sum();
        assert!(sum.abs() < 1e-6);
    }
}
