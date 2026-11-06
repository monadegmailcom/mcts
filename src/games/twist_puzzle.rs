//! Single-player (`N = 1`) 3x3 Quadrant Rotation Puzzle ("2D Mini-Rubik's Cube").
//!
//! Demonstrates how [`GameRules<1>`] and [`crate::mcts::MctsPlayer`] handle single-player
//! combinatorial puzzles with zero special-casing or sign-flipping hacks—serving as a fast,
//! lightweight curriculum stepping stone toward a full 3x3x3 Rubik's Cube solver.
//!
//! ### Rules
//! - Board is a 3x3 grid of tiles `0..9` (`SOLVED_BOARD = [0, 1, 2, 3, 4, 5, 6, 7, 8]`).
//! - There are `8` rotation moves (`0..8`):
//!   - `0..4`: Clockwise quarter-turn of one of the four 2x2 quadrants (TL, TR, BL, BR).
//!   - `4..8`: Counter-clockwise quarter-turn of one of the four 2x2 quadrants.
//! - Reaching `SOLVED_BOARD` yields `GameStatus::Terminal([1.0])`.
//! - Exhausting `remaining_steps == 0` without solving yields `GameStatus::Terminal([-1.0])`.

use rand::prelude::{Rng, RngExt};

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};

pub type TwistMove = u8;
pub const NO_MOVE: TwistMove = u8::MAX;
pub const NUM_MOVES: u8 = 8;
pub const SOLVED_BOARD: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

/// The four 2x2 quadrants in clockwise index order `[top_left, top_right, bottom_right, bottom_left]`.
const QUADRANTS: [[usize; 4]; 4] = [
    [0, 1, 4, 3], // Top-Left 2x2
    [1, 2, 5, 4], // Top-Right 2x2
    [3, 4, 7, 6], // Bottom-Left 2x2
    [4, 5, 8, 7], // Bottom-Right 2x2
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TwistState {
    pub board: [u8; 9],
    pub remaining_steps: u8,
}

impl TwistState {
    pub const fn solved(remaining_steps: u8) -> Self {
        Self {
            board: SOLVED_BOARD,
            remaining_steps,
        }
    }

    #[inline]
    pub fn is_solved(&self) -> bool {
        self.board == SOLVED_BOARD
    }

    /// Generates a scrambled puzzle state `scramble_depth` random rotations away from solved,
    /// avoiding immediate inverse move cancellations (`mv ^ 4`).
    pub fn scrambled(scramble_depth: u8, step_budget: u8, rng: &mut impl Rng) -> Self {
        let mut board = SOLVED_BOARD;
        let mut prev_move: Option<u8> = None;
        for _ in 0..scramble_depth {
            let mv = loop {
                let candidate = rng.random_range(0..NUM_MOVES);
                if prev_move != Some(candidate ^ 4) {
                    break candidate;
                }
            };
            board = rotate_board(&board, mv);
            prev_move = Some(mv);
        }
        Self {
            board,
            remaining_steps: step_budget,
        }
    }
}

#[inline]
pub fn rotate_board(board: &[u8; 9], mv: TwistMove) -> [u8; 9] {
    assert!(mv < NUM_MOVES, "invalid TwistPuzzle move {mv}");
    let mut next = *board;
    let q = QUADRANTS[(mv & 3) as usize];
    if mv < 4 {
        // Clockwise: q[0] -> q[1] -> q[2] -> q[3] -> q[0]
        next[q[1]] = board[q[0]];
        next[q[2]] = board[q[1]];
        next[q[3]] = board[q[2]];
        next[q[0]] = board[q[3]];
    } else {
        // Counter-clockwise (exact inverse of `mv - 4`)
        next[q[0]] = board[q[1]];
        next[q[1]] = board[q[2]];
        next[q[2]] = board[q[3]];
        next[q[3]] = board[q[0]];
    }
    next
}

/// Zero-sized game rules marker for the single-player (`N = 1`) Twist Puzzle.
#[derive(Clone, Copy, Debug)]
pub struct TwistPuzzle;

pub type TwistGame = Game<TwistPuzzle, 1>;

impl GameRules<1> for TwistPuzzle {
    type Move = TwistMove;
    type State = TwistState;
    type MoveIter<'a> = std::ops::Range<u8>;

    #[inline]
    fn valid_moves<'a>(_state: &'a Self::State, _player: PlayerIndex<1>) -> Self::MoveIter<'a> {
        0..NUM_MOVES
    }

    #[inline]
    fn apply(state: &Self::State, _player: PlayerIndex<1>, mv: Self::Move) -> Self::State {
        assert!(
            state.remaining_steps > 0,
            "cannot apply move when remaining_steps == 0"
        );
        Self::State {
            board: rotate_board(&state.board, mv),
            remaining_steps: state.remaining_steps - 1,
        }
    }

    #[inline]
    fn status(state: &Self::State, _current_player: PlayerIndex<1>) -> GameStatus<1> {
        if state.is_solved() {
            GameStatus::Terminal([1.0])
        } else if state.remaining_steps == 0 {
            GameStatus::Terminal([-1.0])
        } else {
            GameStatus::Ongoing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::{MctsConfig, MctsPlayer, PlayoutEvaluator, Uct};
    use crate::player::Player;

    #[test]
    fn inverse_move_restores_solved_board() {
        for mv in 0..4u8 {
            let rotated = rotate_board(&SOLVED_BOARD, mv);
            assert_ne!(rotated, SOLVED_BOARD);
            let restored = rotate_board(&rotated, mv + 4);
            assert_eq!(restored, SOLVED_BOARD);
        }
    }

    #[test]
    fn single_player_mcts_solves_scrambled_rotation_puzzle() {
        use rand::prelude::{SeedableRng, SmallRng};
        let mut rng = SmallRng::seed_from_u64(2026);
        // Scramble 3 moves from solved with a 5-step budget (`N = 1` single-player search!).
        let state = TwistState::scrambled(3, 5, &mut rng);
        assert!(!state.is_solved());

        let mut game = TwistGame::new(PlayerIndex::P0, state);
        let mut solver = MctsPlayer::new(
            game,
            NO_MOVE,
            Uct::new(1.4),
            PlayoutEvaluator::new(77),
            MctsConfig::new(1_200, 99).with_arena_capacity(16_384),
        );

        while game.status().is_ongoing() {
            let mv = solver.choose_move();
            game = game.apply(mv);
            // Notice: for `N = 1`, there are 0 opponent moves between solver moves!
        }

        assert_eq!(game.status(), GameStatus::Terminal([1.0]));
        assert!(game.state().is_solved());
    }
}
