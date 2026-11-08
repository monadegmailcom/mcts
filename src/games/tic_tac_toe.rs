//! Two-player Tic-Tac-Toe (`N = 2`; port of C++ `games/tic_tac_toe.h` & `games/tic_tac_toe.cpp`).

use std::fmt;

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::minimax::HeuristicEvaluator;

/// Marker occupancy of a single Tic-Tac-Toe square.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Symbol {
    Empty = 0,
    Player1 = 1,
    Player2 = 2,
}

impl Symbol {
    #[inline]
    pub const fn from_player(player: PlayerIndex<2>) -> Self {
        match player.raw() {
            0 => Self::Player1,
            _ => Self::Player2,
        }
    }

    #[inline]
    pub const fn as_char(self) -> char {
        match self {
            Self::Empty => ' ',
            Self::Player1 => 'X',
            Self::Player2 => 'O',
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

/// Square index `0..9` (row-major).
pub type TttMove = u8;

/// 3x3 board state (`9` bytes, `Copy` value—allocated entirely inline on the stack).
pub type TttState = [Symbol; 9];

pub const NO_MOVE: TttMove = 9;
pub const EMPTY_STATE: TttState = [Symbol::Empty; 9];

/// All 8 winning 3-in-a-row line indices.
pub const WIN_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8], // rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8], // columns
    [0, 4, 8],
    [2, 4, 6], // diagonals
];

/// Zero-sized game rules marker for 2-player Tic-Tac-Toe.
#[derive(Clone, Copy, Debug)]
pub struct TicTacToe;

pub type TttGame = Game<TicTacToe, 2>;

/// Lazy, zero-allocation iterator over empty squares `0..9`
/// (replaces `boost::iterator_facade` + `next_valid_move` in `game.h` / `tic_tac_toe.cpp`).
pub struct TttMoveIter<'a> {
    state: &'a TttState,
    cursor: u8,
}

impl Iterator for TttMoveIter<'_> {
    type Item = TttMove;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while (self.cursor as usize) < 9 {
            let idx = self.cursor;
            self.cursor += 1;
            if self.state[idx as usize] == Symbol::Empty {
                return Some(idx);
            }
        }
        None
    }
}

impl GameRules<2> for TicTacToe {
    type Move = TttMove;
    type State = TttState;
    type MoveIter<'a> = TttMoveIter<'a>;

    #[inline]
    fn valid_moves<'a>(state: &'a Self::State, _player: PlayerIndex<2>) -> Self::MoveIter<'a> {
        TttMoveIter { state, cursor: 0 }
    }

    #[inline]
    fn write_valid_moves(
        state: &Self::State,
        _player: PlayerIndex<2>,
        out: &mut Vec<Self::Move>,
    ) {
        out.clear();
        for (idx, &cell) in state.iter().enumerate() {
            if cell == Symbol::Empty {
                out.push(idx as u8);
            }
        }
    }

    #[inline]
    fn apply(state: &Self::State, player: PlayerIndex<2>, mv: Self::Move) -> Self::State {
        let idx = mv as usize;
        assert!(idx < 9, "invalid Tic-Tac-Toe move index: {mv}");
        assert!(
            state[idx] == Symbol::Empty,
            "invalid Tic-Tac-Toe move {mv}: square already occupied"
        );
        let mut next = *state;
        next[idx] = Symbol::from_player(player);
        next
    }

    fn status(state: &Self::State, _current_player: PlayerIndex<2>) -> GameStatus<2> {
        for &[a, b, c] in &WIN_LINES {
            let s = state[a];
            if s != Symbol::Empty && s == state[b] && s == state[c] {
                let winner = match s {
                    Symbol::Player1 => PlayerIndex::<2>::PLAYER1,
                    Symbol::Player2 => PlayerIndex::<2>::PLAYER2,
                    Symbol::Empty => unreachable!(),
                };
                return GameStatus::single_winner(winner);
            }
        }

        if state.contains(&Symbol::Empty) {
            GameStatus::Ongoing
        } else {
            GameStatus::draw()
        }
    }
}

/// Classic unblocked-line heuristic for Tic-Tac-Toe (port of `ttt::minimax::score` in
/// `games/tic_tac_toe.cpp`, normalized to `[p1_advantage, p2_advantage]`).
///
/// For each of the 8 winning lines:
/// - If Player 2 has 0 marks on the line, Player 1 gains `+p1_marks`.
/// - If Player 1 has 0 marks on the line, Player 2 gains `+p2_marks`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TttHeuristic;

impl TttHeuristic {
    #[inline]
    pub fn score_state(state: &TttState) -> [f32; 2] {
        let mut p1_adv = 0i32;
        for &[a, b, c] in &WIN_LINES {
            let mut p1 = 0i32;
            let mut p2 = 0i32;
            for &cell_idx in &[a, b, c] {
                match state[cell_idx] {
                    Symbol::Player1 => p1 += 1,
                    Symbol::Player2 => p2 += 1,
                    Symbol::Empty => {}
                }
            }
            if p2 == 0 {
                p1_adv += p1;
            } else if p1 == 0 {
                p1_adv -= p2;
            }
        }
        let s = p1_adv as f32;
        [s, -s]
    }
}

impl HeuristicEvaluator<TicTacToe, 2> for TttHeuristic {
    #[inline]
    fn evaluate_heuristic(&self, game: &TttGame) -> [f32; 2] {
        Self::score_state(game.state())
    }
}

/// Formats a Tic-Tac-Toe game board for terminal display.
pub fn format_board(game: &TttGame) -> String {
    let s = game.state();
    format!(
        "player {}\n-----\n|{}{}{}|\n|{}{}{}|\n|{}{}{}|\n-----",
        game.current_player().raw() + 1,
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        s[8],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::{
        FnEvaluator, MctsConfig, MctsPlayer, PlayoutEvaluator, Puct, Uct,
        assign_masked_softmax_priors,
    };
    use crate::player::Player;

    #[test]
    fn game_progression_matches_cpp_test() {
        let mut game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        assert_eq!(game.status(), GameStatus::Ongoing);
        assert_eq!(game.valid_moves().collect::<Vec<_>>(), (0..9).collect::<Vec<_>>());

        // Replays exact move sequence from `test.cpp::ttt_game`
        for &mv in &[4, 0, 7, 2, 8] {
            game = game.apply(mv);
        }
        assert_eq!(game.current_player(), PlayerIndex::PLAYER2);
        assert_eq!(game.valid_moves().collect::<Vec<_>>(), vec![1, 3, 5, 6]);
        assert_eq!(game.status(), GameStatus::Ongoing);

        // Player 2 completes top row [0, 1, 2]
        game = game.apply(1);
        assert_eq!(
            game.status(),
            GameStatus::single_winner(PlayerIndex::PLAYER2)
        );
    }

    #[test]
    fn uct_playout_mcts_finds_immediate_win_and_blocks_threat() {
        // Player 1 (`X`) has 0 and 1; square 2 wins immediately!
        // X | X | .
        // O | O | .
        // . | . | .
        let mut board = EMPTY_STATE;
        board[0] = Symbol::Player1;
        board[1] = Symbol::Player1;
        board[3] = Symbol::Player2;
        board[4] = Symbol::Player2;

        let game = TttGame::new(PlayerIndex::PLAYER1, board);
        let mut player = MctsPlayer::new(
            game,
            NO_MOVE,
            Uct::default(),
            PlayoutEvaluator::new(123),
            MctsConfig::new(400, 456).with_arena_capacity(2048),
        );
        assert_eq!(player.choose_move(), 2);

        // If it is Player 1's turn when Player 2 (`O`) threatens square 5 ([3, 4, _])
        // and Player 1 has only [0, 8], Player 1 MUST block square 5!
        let mut block_board = EMPTY_STATE;
        block_board[0] = Symbol::Player1;
        block_board[8] = Symbol::Player1;
        block_board[3] = Symbol::Player2;
        block_board[4] = Symbol::Player2;

        let block_game = TttGame::new(PlayerIndex::PLAYER1, block_board);
        let mut blocker = MctsPlayer::new(
            block_game,
            NO_MOVE,
            Uct::default(),
            PlayoutEvaluator::new(789),
            MctsConfig::new(600, 101).with_arena_capacity(2048),
        );
        assert_eq!(blocker.choose_move(), 5);
    }

    #[test]
    fn puct_alphazero_style_search_draws_self_play_and_blocks_tactics() {
        // Demonstrates AlphaZero-style PUCT search using a prior + value evaluator
        // (same MctsPlayer engine, zero virtual loss, 100% sequential informed PUCT).
        let mut block_board = EMPTY_STATE;
        block_board[0] = Symbol::Player1;
        block_board[8] = Symbol::Player1;
        block_board[3] = Symbol::Player2;
        block_board[4] = Symbol::Player2;

        let game = TttGame::new(PlayerIndex::PLAYER1, block_board);

        // Mock untrained NN returning uniform logits (0.0) and neutral value [0.0, 0.0]:
        // Pure sequential PUCT search alone discovers Player 2's 1-move win if square 5 is not
        // blocked and converges to move 5!
        let uniform_nn = FnEvaluator(|_g: &TttGame, children: &mut [crate::tree::Node<u8, _>]| {
            assign_masked_softmax_priors(children, |_mv| 0.0);
            [0.0, 0.0]
        });

        let mut az_player = MctsPlayer::new(
            game,
            NO_MOVE,
            Puct::new(19_652.0, 1.25),
            uniform_nn,
            MctsConfig::new(400, 42).with_arena_capacity(4096),
        );
        assert_eq!(az_player.choose_move(), 5);
    }

    #[test]
    fn minimax_depth_0_vs_depth_5_matches_cpp_test() {
        use crate::matchup::run_multi_match;
        use crate::minimax::MinimaxPlayer;

        // Port of `test.cpp::tic_tac_toe_match`: depth 0 never beats depth 5 over 100 games.
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let f_depth0 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 0, seed))
        };
        let f_depth5 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 5, seed))
        };

        let summary = run_multi_match(&game, [&f_depth0, &f_depth5], 100, 4, 2026);
        assert_eq!(
            summary.wins[0], 0,
            "depth 0 should never beat depth 5, got summary={summary:?}"
        );
        assert!(summary.wins[1] + summary.draws == 100);
    }

    #[test]
    fn tree_minimax_draws_perfect_minimax_and_reuses_ordered_subtrees() {
        use crate::matchup::{play_match, run_multi_match};
        use crate::minimax::{MinimaxBudget, MinimaxPlayer, TreeMinimaxPlayer};

        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);

        // Single-game sanity check: verify generational tree retains surviving nodes across turns.
        let mut mm = MinimaxPlayer::new(game, TttHeuristic, 4, 11);
        let mut tree_mm = TreeMinimaxPlayer::new(
            game,
            NO_MOVE,
            TttHeuristic,
            MinimaxBudget::FixedDepth(4),
            22,
            4096,
        );
        let outcome = play_match(
            game,
            &mut [&mut mm as &mut dyn Player<TicTacToe, 2>, &mut tree_mm],
        );
        assert!(outcome.is_draw(), "expected perfect draw, got {outcome:?}");
        assert!(tree_mm.eval_calls() > 0);

        // Multi-game tournament at depth 5 vs dynamic MaxEvaluations(600): every game draws!
        let f_mm = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 5, seed))
        };
        let f_tree = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(TreeMinimaxPlayer::new(
                *g,
                NO_MOVE,
                TttHeuristic,
                MinimaxBudget::MaxEvaluations(600),
                seed,
                8192,
            ))
        };
        let summary = run_multi_match(&game, [&f_mm, &f_tree], 40, 4, 99);
        assert_eq!(summary.draws, 40, "expected 40/40 draws, got {summary:?}");
    }

    #[test]
    fn sequential_mcts_without_virtual_loss_holds_minimax_depth_2() {
        use crate::matchup::run_multi_match;
        use crate::minimax::MinimaxPlayer;

        // Directly validates the architectural premise: pure single-threaded MCTS (zero virtual
        // loss distortion) with 800 rollouts per move draws or beats Minimax(depth=2) consistently!
        let game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let f_mcts = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(MctsPlayer::new(
                *g,
                NO_MOVE,
                Uct::new(1.0),
                PlayoutEvaluator::new(seed ^ 0xA5A5_5A5A),
                MctsConfig::new(800, seed).with_arena_capacity(2048),
            ))
        };
        let f_mm2 = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
            Box::new(MinimaxPlayer::new(*g, TttHeuristic, 2, seed))
        };

        let summary = run_multi_match(&game, [&f_mcts, &f_mm2], 40, 4, 777);
        assert!(
            summary.wins[0] + summary.draws >= 38,
            "expected pure single-threaded MCTS to draw/beat Minimax depth 2 almost every game, got {summary:?}"
        );
    }
}
