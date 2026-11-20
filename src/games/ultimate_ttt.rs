//! Two-player Ultimate Tic-Tac-Toe (`N = 2`; port & optimization of C++
//! `games/ultimate_ttt.h` & `games/ultimate_ttt.cpp`).
//!
//! ### Architectural Improvements over C++ `beat-it`
//! - **No `mutable std::optional<GameResult> game_result_cache`**: Instead of mutating a cached
//!   field behind `State const&` inside `result()` (a thread-safety hazard when sharing states
//!   across threads), [`UtttState`] updates `outcome: SubBoardStatus` eagerly inside `apply()`
//!   only on turns where a sub-board actually finishes. [`UltimateTicTacToe::status`] becomes a
//!   1-byte branchless read (`92` bytes total `Copy` state, `Sync` by construction).
//! - **20-Byte MCTS Nodes & 12-Byte `TreeMinimax` Nodes**: Because [`UtttMove`] is `2` bytes and
//!   states are reconstructed along the root-to-leaf path, MCTS nodes shrink from ~280 bytes in
//!   C++ down to **20 bytes** (14x smaller), and `TreeMinimax` nodes down to **12 bytes** (23x smaller).

use std::fmt;

use crate::alphazero::AlphaZeroGame;
use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::hyperparams::AlphaZeroHyperparams;
use crate::nn::AlphaZeroNetConfig;
use crate::play::PlayableGame;
use crate::training::Symmetry;
use crate::games::tic_tac_toe::{
    EMPTY_STATE as EMPTY_SMALL_STATE, Symbol, TicTacToe, TttHeuristic, TttState, WIN_LINES,
};
use crate::minimax::HeuristicEvaluator;

/// Sentinel sub-board index indicating the acting player is free to choose any undecided sub-board.
pub const NO_BIG_MOVE: u8 = 9;

/// Sentinel move used to initialize the root of a search tree before any move has been played.
pub const NO_MOVE: UtttMove = UtttMove {
    big_move: NO_BIG_MOVE,
    small_move: NO_BIG_MOVE,
};

/// Number of spatial cells / policy logits (`9 x 9 = 81`).
pub const POLICY_SIZE: usize = 81;
/// Number of 9x9 feature planes for AlphaZero (see [`encode_alphazero_planes`]).
pub const INPUT_PLANES: usize = 7;
/// Flattened neural network input tensor length (`7 * 81 = 567`).
pub const INPUT_SIZE: usize = INPUT_PLANES * POLICY_SIZE;

/// A move in Ultimate Tic-Tac-Toe (`2` bytes): sub-board `big_move` (`0..9`) and cell `small_move` (`0..9`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UtttMove {
    pub big_move: u8,
    pub small_move: u8,
}

impl UtttMove {
    #[inline]
    pub const fn new(big_move: u8, small_move: u8) -> Self {
        Self {
            big_move,
            small_move,
        }
    }

    /// Maps `(big_move, small_move)` to global 9x9 row-major policy index `0..81`
    /// (matches `uttt::alphazero::Player::move_to_policy_index` in `ultimate_ttt.cpp`).
    #[inline]
    pub const fn to_policy_index(self) -> usize {
        let row = ((self.big_move as usize) / 3) * 3 + ((self.small_move as usize) / 3);
        let col = ((self.big_move as usize) % 3) * 3 + ((self.small_move as usize) % 3);
        row * 9 + col
    }

    /// Inverse of [`Self::to_policy_index`]: maps global 9x9 index `0..81` back to `UtttMove`.
    #[inline]
    pub const fn from_policy_index(idx: usize) -> Self {
        assert!(idx < 81, "UTTT policy index out of range 0..81");
        let row = idx / 9;
        let col = idx % 9;
        let big_move = ((row / 3) * 3 + (col / 3)) as u8;
        let small_move = ((row % 3) * 3 + (col % 3)) as u8;
        Self {
            big_move,
            small_move,
        }
    }
}

impl fmt::Display for UtttMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.big_move, self.small_move)
    }
}

/// Compact 1-byte status of a 3x3 sub-board (or the overall 3x3 macro-board).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SubBoardStatus {
    Ongoing = 0,
    Player1Win = 1,
    Player2Win = 2,
    Draw = 3,
}

impl SubBoardStatus {
    #[inline]
    pub const fn as_char(self) -> char {
        match self {
            Self::Ongoing => ' ',
            Self::Player1Win => 'X',
            Self::Player2Win => 'O',
            Self::Draw => 'D',
        }
    }

    #[inline]
    fn from_ttt_status(status: GameStatus<2>) -> Self {
        match status {
            GameStatus::Ongoing => Self::Ongoing,
            GameStatus::Terminal(payoffs) => {
                if payoffs[0] > payoffs[1] {
                    Self::Player1Win
                } else if payoffs[1] > payoffs[0] {
                    Self::Player2Win
                } else {
                    Self::Draw
                }
            }
        }
    }
}

/// Complete Ultimate Tic-Tac-Toe board state (`92` bytes, `Copy`, zero heap allocation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UtttState {
    pub small_states: [TttState; 9],
    pub big_state: [SubBoardStatus; 9],
    /// Sub-board index `0..9` forced for the next player, or [`NO_BIG_MOVE`] (`9`) if free choice.
    pub next_big_move: u8,
    /// Eagerly maintained macro-board outcome.
    pub outcome: SubBoardStatus,
}

pub const EMPTY_STATE: UtttState = UtttState {
    small_states: [EMPTY_SMALL_STATE; 9],
    big_state: [SubBoardStatus::Ongoing; 9],
    next_big_move: NO_BIG_MOVE,
    outcome: SubBoardStatus::Ongoing,
};

#[inline]
fn compute_macro_outcome(big_state: &[SubBoardStatus; 9]) -> SubBoardStatus {
    for &[a, b, c] in &WIN_LINES {
        let s = big_state[a];
        if (s == SubBoardStatus::Player1Win || s == SubBoardStatus::Player2Win)
            && s == big_state[b]
            && s == big_state[c]
        {
            return s;
        }
    }
    if big_state.contains(&SubBoardStatus::Ongoing) {
        SubBoardStatus::Ongoing
    } else {
        SubBoardStatus::Draw
    }
}

/// Zero-sized game rules marker for 2-player Ultimate Tic-Tac-Toe.
#[derive(Clone, Copy, Debug)]
pub struct UltimateTicTacToe;

pub type UtttGame = Game<UltimateTicTacToe, 2>;

/// Lazy, zero-allocation move iterator over legal `(big_move, small_move)` pairs.
pub struct UtttMoveIter<'a> {
    state: &'a UtttState,
    big: u8,
    big_end: u8,
    small: u8,
}

impl Iterator for UtttMoveIter<'_> {
    type Item = UtttMove;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.big < self.big_end {
            let b = self.big as usize;
            if self.state.big_state[b] == SubBoardStatus::Ongoing {
                while self.small < 9 {
                    let s = self.small;
                    self.small += 1;
                    if self.state.small_states[b][s as usize] == Symbol::Empty {
                        return Some(UtttMove {
                            big_move: self.big,
                            small_move: s,
                        });
                    }
                }
            }
            self.big += 1;
            self.small = 0;
        }
        None
    }
}

impl GameRules<2> for UltimateTicTacToe {
    type Move = UtttMove;
    type State = UtttState;
    type MoveIter<'a> = UtttMoveIter<'a>;

    #[inline]
    fn valid_moves<'a>(state: &'a Self::State, _player: PlayerIndex<2>) -> Self::MoveIter<'a> {
        if state.outcome != SubBoardStatus::Ongoing {
            return UtttMoveIter {
                state,
                big: 9,
                big_end: 9,
                small: 0,
            };
        }
        if state.next_big_move != NO_BIG_MOVE {
            UtttMoveIter {
                state,
                big: state.next_big_move,
                big_end: state.next_big_move + 1,
                small: 0,
            }
        } else {
            UtttMoveIter {
                state,
                big: 0,
                big_end: 9,
                small: 0,
            }
        }
    }

    #[inline]
    fn write_valid_moves(
        state: &Self::State,
        _player: PlayerIndex<2>,
        out: &mut Vec<Self::Move>,
    ) {
        out.clear();
        if state.outcome != SubBoardStatus::Ongoing {
            return;
        }
        if state.next_big_move != NO_BIG_MOVE {
            let b = state.next_big_move;
            for (s, &cell) in state.small_states[b as usize].iter().enumerate() {
                if cell == Symbol::Empty {
                    out.push(UtttMove {
                        big_move: b,
                        small_move: s as u8,
                    });
                }
            }
        } else {
            for b in 0..9u8 {
                if state.big_state[b as usize] == SubBoardStatus::Ongoing {
                    for (s, &cell) in state.small_states[b as usize].iter().enumerate() {
                        if cell == Symbol::Empty {
                            out.push(UtttMove {
                                big_move: b,
                                small_move: s as u8,
                            });
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn apply(state: &Self::State, player: PlayerIndex<2>, mv: Self::Move) -> Self::State {
        let big = mv.big_move as usize;
        assert!(big < 9, "invalid UTTT big_move: {}", mv.big_move);
        assert!(
            state.next_big_move == NO_BIG_MOVE || mv.big_move == state.next_big_move,
            "invalid UTTT big_move {}: forced to play in sub-board {}",
            mv.big_move,
            state.next_big_move
        );
        assert!(
            state.big_state[big] == SubBoardStatus::Ongoing,
            "cannot play in already-finished sub-board {big}"
        );

        let mut next = *state;
        next.small_states[big] = TicTacToe::apply(&state.small_states[big], player, mv.small_move);

        let sub_status =
            SubBoardStatus::from_ttt_status(TicTacToe::status(&next.small_states[big], player));
        next.big_state[big] = sub_status;

        if sub_status != SubBoardStatus::Ongoing {
            next.outcome = compute_macro_outcome(&next.big_state);
        }

        next.next_big_move = if next.big_state[mv.small_move as usize] == SubBoardStatus::Ongoing {
            mv.small_move
        } else {
            NO_BIG_MOVE
        };

        next
    }

    #[inline]
    fn status(state: &Self::State, _current_player: PlayerIndex<2>) -> GameStatus<2> {
        match state.outcome {
            SubBoardStatus::Ongoing => GameStatus::Ongoing,
            SubBoardStatus::Player1Win => GameStatus::single_winner(PlayerIndex::PLAYER1),
            SubBoardStatus::Player2Win => GameStatus::single_winner(PlayerIndex::PLAYER2),
            SubBoardStatus::Draw => GameStatus::draw(),
        }
    }
}

/// Two-tier Ultimate Tic-Tac-Toe position heuristic (port of `uttt::score_function` in
/// `games/ultimate_ttt.cpp`, normalized to `[p1_advantage, p2_advantage]`).
///
/// Combines macro-board unblocked line counts weighted by `macro_weight` (default `9.0` in
/// `beat-it`) with the sum of [`TttHeuristic`] scores across all undecided sub-boards.
#[derive(Clone, Copy, Debug)]
pub struct UtttHeuristic {
    pub macro_weight: f32,
}

impl Default for UtttHeuristic {
    #[inline]
    fn default() -> Self {
        Self { macro_weight: 9.0 }
    }
}

impl UtttHeuristic {
    #[inline]
    pub const fn new(macro_weight: f32) -> Self {
        Self { macro_weight }
    }

    pub fn score_state(&self, state: &UtttState) -> [f32; 2] {
        let mut macro_p1_adv = 0i32;

        for &[a, b, c] in &WIN_LINES {
            let mut skip_line = false;
            let mut p1 = 0i32;
            let mut p2 = 0i32;

            for &idx in &[a, b, c] {
                match state.big_state[idx] {
                    SubBoardStatus::Draw => {
                        skip_line = true;
                        break;
                    }
                    SubBoardStatus::Player1Win => p1 += 1,
                    SubBoardStatus::Player2Win => p2 += 1,
                    SubBoardStatus::Ongoing => {}
                }
            }

            if skip_line {
                continue;
            }
            if p2 == 0 {
                macro_p1_adv += p1;
            } else if p1 == 0 {
                macro_p1_adv -= p2;
            }
        }

        let mut p1_score = (macro_p1_adv as f32) * self.macro_weight;
        for i in 0..9 {
            if state.big_state[i] == SubBoardStatus::Ongoing {
                p1_score += TttHeuristic::score_state(&state.small_states[i])[0];
            }
        }

        [p1_score, -p1_score]
    }
}

impl HeuristicEvaluator<UltimateTicTacToe, 2> for UtttHeuristic {
    #[inline]
    fn evaluate_heuristic(&self, game: &UtttGame) -> [f32; 2] {
        self.score_state(game.state())
    }
}

/// Encodes `game` into the 7x9x9 (`567` floats) relative feature representation for AlphaZero:
/// - Plane 0: `1.0` where `game.current_player()` holds a cell.
/// - Plane 1: `1.0` where the opponent holds a cell.
/// - Plane 2: `1.0` on every currently legal move (empty cell of a playable sub-board).
/// - Plane 3: `1.0` on all 9 cells of each sub-board won by the current player.
/// - Plane 4: `1.0` on all 9 cells of each sub-board won by the opponent.
/// - Plane 5: `1.0` on all 9 cells of each drawn (full, undecided) sub-board.
/// - Plane 6: `1.0` everywhere if `current_player == PLAYER1`, else `0.0`.
///
/// Planes 3–5 expose the macro board directly: without them the network has to rediscover
/// "three in a row inside a 3x3 block" from raw pieces before it can reason about the macro game,
/// which makes value learning far slower.
pub fn encode_alphazero_planes(game: &UtttGame, out: &mut [f32; INPUT_SIZE]) {
    out.fill(0.0);
    let state = game.state();
    let current_player = game.current_player();
    let my_sym = Symbol::from_player(current_player);
    let opp_sym = Symbol::from_player(current_player.toggle());
    let (my_win, opp_win) = if current_player == PlayerIndex::PLAYER1 {
        (SubBoardStatus::Player1Win, SubBoardStatus::Player2Win)
    } else {
        (SubBoardStatus::Player2Win, SubBoardStatus::Player1Win)
    };

    for big in 0..9u8 {
        let small_board = &state.small_states[big as usize];
        let big_status = state.big_state[big as usize];
        let sub_playable = state.outcome == SubBoardStatus::Ongoing
            && big_status == SubBoardStatus::Ongoing
            && (state.next_big_move == NO_BIG_MOVE || state.next_big_move == big);
        let status_plane = if big_status == my_win {
            Some(3)
        } else if big_status == opp_win {
            Some(4)
        } else if big_status == SubBoardStatus::Draw {
            Some(5)
        } else {
            None
        };

        for small in 0..9u8 {
            let spatial = UtttMove::new(big, small).to_policy_index();
            let cell = small_board[small as usize];
            if cell == my_sym {
                out[spatial] = 1.0;
            } else if cell == opp_sym {
                out[81 + spatial] = 1.0;
            } else if sub_playable {
                out[2 * 81 + spatial] = 1.0;
            }
            if let Some(plane) = status_plane {
                out[plane * 81 + spatial] = 1.0;
            }
        }
    }

    if current_player == PlayerIndex::PLAYER1 {
        out[6 * 81..7 * 81].fill(1.0);
    }
}


/// Formats an Ultimate Tic-Tac-Toe position for terminal display.
pub fn format_board(game: &UtttGame) -> String {
    let state = game.state();
    let mut out = String::with_capacity(512);

    for big_row in 0..3 {
        for small_row in 0..3 {
            for big_col in 0..3 {
                let big_idx = big_row * 3 + big_col;
                if state.big_state[big_idx] != SubBoardStatus::Ongoing {
                    for small_col in 0..3 {
                        if small_row == 1 && small_col == 1 {
                            out.push(state.big_state[big_idx].as_char());
                            out.push(' ');
                        } else {
                            out.push_str("  ");
                        }
                    }
                } else {
                    for small_col in 0..3 {
                        let sym = state.small_states[big_idx][small_row * 3 + small_col];
                        out.push(sym.as_char());
                        out.push(' ');
                    }
                }
                if big_col != 2 {
                    out.push(' ');
                }
            }
            out.push('\n');
        }
        if big_row != 2 {
            out.push('\n');
        }
    }

    out.push_str("\nbig state:\n-----\n");
    for idx in 0..9 {
        if idx % 3 == 0 {
            out.push('|');
        }
        out.push(state.big_state[idx].as_char());
        if (idx + 1) % 3 == 0 {
            out.push_str("|\n");
        }
    }
    out.push_str("-----\nnext big move: (");
    if state.next_big_move != NO_BIG_MOVE {
        out.push(char::from(b'0' + state.next_big_move));
    } else {
        out.push_str("choose free");
    }
    out.push_str(")\n");
    out
}

// ---------------------------------------------------------------------------------------------
// AlphaZero integration
// ---------------------------------------------------------------------------------------------

/// Block network: one shared encoder per `3x3` sub-board plus a global MLP (the default; learns
/// as much per game as [`net_resnet`] at a fraction of its compute).
pub const fn net_block() -> AlphaZeroNetConfig {
    AlphaZeroNetConfig::BlockNet {
        in_planes: INPUT_PLANES,
        board_height: 9,
        board_width: 9,
        block_height: 3,
        block_width: 3,
        local_dim: 64,
        hidden_dim: 256,
        num_res_blocks: 3,
        head_hidden: 64,
    }
}

/// Convolutional ResNet over the `9x9` grid.
pub const fn net_resnet() -> AlphaZeroNetConfig {
    AlphaZeroNetConfig::ResNet2d {
        in_channels: INPUT_PLANES,
        board_height: 9,
        board_width: 9,
        channels: 64,
        num_res_blocks: 4,
        policy_channels: 8,
        value_channels: 4,
        fc_hidden: 128,
    }
}

/// Flat residual MLP over the whole encoding (no spatial structure; a baseline).
pub const fn net_mlp() -> AlphaZeroNetConfig {
    AlphaZeroNetConfig::ResMlp {
        hidden_dim: 128,
        num_res_blocks: 2,
        head_hidden: 96,
    }
}

impl AlphaZeroGame<INPUT_SIZE, POLICY_SIZE, 2> for UltimateTicTacToe {
    const SENTINEL_MOVE: Self::Move = NO_MOVE;

    #[inline]
    fn move_to_policy_index(mv: Self::Move) -> usize {
        mv.to_policy_index()
    }

    #[inline]
    fn encode_state(game: &Game<Self, 2>, out: &mut [f32; INPUT_SIZE]) {
        encode_alphazero_planes(game, out);
    }

    fn net_config() -> AlphaZeroNetConfig {
        net_block()
    }

    fn net_presets() -> Vec<(&'static str, AlphaZeroNetConfig)> {
        vec![("block", net_block()), ("resnet", net_resnet()), ("mlp", net_mlp())]
    }

    /// The 8 dihedral symmetries of Ultimate Tic-Tac-Toe for replay-buffer augmentation.
    ///
    /// Rotating/reflecting the whole 9x9 grid applies the same transform to the macro board and to
    /// every sub-board, which preserves the "cell `k` sends the opponent to sub-board `k`" rule, so all
    /// 8 transforms are exact symmetries of the game (and of [`encode_alphazero_planes`]).
    fn symmetries() -> Vec<Symmetry> {
        Symmetry::dihedral_grid(INPUT_PLANES, 9)
    }

    /// Many cheap games per iteration: 1024 games keep the network batches full (self-play ~1.7x
    /// faster than with 256 on the dev Mac) and in a 15-minute comparison learned faster per
    /// minute and at least as well per game. Enough SGD steps to see each new position ~4 times
    /// (`1024 games * ~50 positions * 4 / 256`), a replay buffer holding ~6k games, and a value
    /// target blended with the search value (`qw = 0.5`, meant to damp the noise of early,
    /// near-random game outcomes; compare against `qw = 0` in the Experiments tab before relying on it).
    fn hyperparams() -> AlphaZeroHyperparams {
        AlphaZeroHyperparams {
            games_per_iteration: 1024,
            simulations_per_move: 64,
            opening_moves: 8,
            value_q_weight: 0.5,
            train_steps: 800,
            batch_size: 256,
            learning_rate: 2.0e-3,
            buffer_capacity: 300_000,
            eval_games: 48,
            eval_sims: 0,
            ..AlphaZeroHyperparams::GENERIC
        }
    }
}

impl PlayableGame<INPUT_SIZE, POLICY_SIZE> for UltimateTicTacToe {
    type Heuristic = UtttHeuristic;

    fn start() -> UtttGame {
        UtttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE)
    }

    fn heuristic() -> UtttHeuristic {
        UtttHeuristic::default()
    }

    /// Seeded random 2-ply openings: rarely decisive, and the seat swap cancels what they favour.
    fn eval_openings(count: usize) -> Vec<UtttGame> {
        crate::matchup::random_openings(&Self::start(), count.max(1), 2, 4242)
    }

    fn cells(game: &UtttGame) -> Vec<u8> {
        let state = game.state();
        (0..81)
            .map(|idx| {
                let mv = UtttMove::from_policy_index(idx);
                state.small_states[mv.big_move as usize][mv.small_move as usize] as u8
            })
            .collect()
    }

    fn blocks(game: &UtttGame) -> Vec<u8> {
        game.state().big_state.iter().map(|&s| s as u8).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matchup::run_multi_match;
    use crate::mcts::{MctsConfig, MctsPayload, MctsPlayer, PlayoutEvaluator, Uct};
    use crate::minimax::{MinimaxBudget, MinimaxPlayer, TreeMinimaxPayload, TreeMinimaxPlayer};
    use crate::player::Player;
    use crate::tree::Node;

    #[test]
    fn uttt_node_sizes_are_20_and_12_bytes() {
        assert_eq!(std::mem::size_of::<UtttMove>(), 2);
        assert_eq!(std::mem::size_of::<UtttState>(), 92);
        assert_eq!(std::mem::size_of::<Node<UtttMove, MctsPayload>>(), 20);
        assert_eq!(std::mem::size_of::<Node<UtttMove, TreeMinimaxPayload>>(), 12);
    }

    #[test]
    fn policy_index_bijection_covers_all_81_cells() {
        let mut seen = [false; 81];
        for big in 0..9u8 {
            for small in 0..9u8 {
                let mv = UtttMove::new(big, small);
                let idx = mv.to_policy_index();
                assert!(!seen[idx]);
                seen[idx] = true;
                assert_eq!(UtttMove::from_policy_index(idx), mv);
            }
        }
    }

    #[test]
    fn uttt_progression_matches_cpp_test() {
        // Exact port of `test.cpp::uttt_game`
        let mut game = UtttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        assert_eq!(game.status(), GameStatus::Ongoing);
        assert_eq!(game.valid_moves().count(), 81);

        game = game.apply(UtttMove::new(4, 4));
        assert_eq!(game.current_player(), PlayerIndex::PLAYER2);
        let expected_after_4_4: Vec<UtttMove> = [0, 1, 2, 3, 5, 6, 7, 8]
            .into_iter()
            .map(|s| UtttMove::new(4, s))
            .collect();
        assert_eq!(game.valid_moves().collect::<Vec<_>>(), expected_after_4_4);

        game = game.apply(UtttMove::new(4, 1));
        assert_eq!(game.current_player(), PlayerIndex::PLAYER1);
        let expected_after_4_1: Vec<UtttMove> =
            (0..9).map(|s| UtttMove::new(1, s)).collect();
        assert_eq!(game.valid_moves().collect::<Vec<_>>(), expected_after_4_1);
    }

    #[test]
    fn deeper_minimax_and_dynamic_tree_minimax_outplay_shallow_agents() {
        let game = UtttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let h = UtttHeuristic::default();

        // 1. Deeper Minimax (depth 3) dominates shallow Minimax (depth 1) across 20 games.
        let f_d1 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
            Box::new(MinimaxPlayer::new(*g, h, 1, seed))
        };
        let f_d3 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
            Box::new(MinimaxPlayer::new(*g, h, 3, seed))
        };
        let mm_summary = run_multi_match(&game, [&f_d1, &f_d3], 20, 4, 2026);
        assert!(
            mm_summary.wins[1] > mm_summary.wins[0],
            "expected UTTT Minimax(d=3) to beat Minimax(d=1), got {mm_summary:?}"
        );

        // 2. Fair compute-capped duel: MctsPlayer(400 sims) vs TreeMinimax(MaxEvaluations(400))
        // and verify TreeMinimax(MaxEvaluations(600)) easily beats Minimax(d=1).
        let f_tree_600 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
            Box::new(TreeMinimaxPlayer::new(
                *g,
                NO_MOVE,
                h,
                MinimaxBudget::MaxEvaluations(600),
                seed,
                16_384,
            ))
        };
        let tree_vs_d1 = run_multi_match(&game, [&f_d1, &f_tree_600], 20, 4, 77);
        assert!(
            tree_vs_d1.wins[1] > tree_vs_d1.wins[0],
            "expected TreeMinimax(MaxEvaluations(600)) to beat Minimax(d=1), got {tree_vs_d1:?}"
        );

        // 3. Single-threaded UCT playout MCTS works cleanly on Ultimate Tic-Tac-Toe too!
        let f_mcts = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
            Box::new(MctsPlayer::new(
                *g,
                NO_MOVE,
                Uct::new(0.7),
                PlayoutEvaluator::new(seed ^ 0x1234_5678),
                MctsConfig::new(300, seed).with_arena_capacity(8192),
            ))
        };
        let mcts_vs_rand = run_multi_match(
            &game,
            [
                &f_mcts,
                &|g: &UtttGame, seed: u64| Box::new(crate::matchup::RandomPlayer::new(*g, seed)),
            ],
            12,
            4,
            999,
        );
        assert_eq!(
            mcts_vs_rand.wins[0], 12,
            "expected 300-sim UCT MCTS to win 12/12 vs RandomPlayer in UTTT, got {mcts_vs_rand:?}"
        );
    }

    #[test]
    fn dihedral_symmetries_commute_with_rules_and_encoding() {
        use rand::prelude::{IndexedRandom, SeedableRng, SmallRng};
        let syms = <UltimateTicTacToe as AlphaZeroGame<INPUT_SIZE, POLICY_SIZE, 2>>::symmetries();
        assert_eq!(syms.len(), 8);
        let mut rng = SmallRng::seed_from_u64(7);
        let mut moves = Vec::new();
        let mut moves_t = Vec::new();
        for sym in &syms {
            // policy_src maps destination cell -> source cell; invert to map a move forward.
            let mut fwd = [0usize; 81];
            for (dst, &src) in sym.policy_src.iter().enumerate() {
                fwd[src as usize] = dst;
            }
            let map = |mv: UtttMove| UtttMove::from_policy_index(fwd[mv.to_policy_index()]);
            for _ in 0..20 {
                let mut g = UtttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
                let mut gt = g;
                while g.status().is_ongoing() {
                    g.write_valid_moves(&mut moves);
                    gt.write_valid_moves(&mut moves_t);
                    let mut mapped: Vec<usize> =
                        moves.iter().map(|&m| map(m).to_policy_index()).collect();
                    let mut actual: Vec<usize> =
                        moves_t.iter().map(|m| m.to_policy_index()).collect();
                    mapped.sort_unstable();
                    actual.sort_unstable();
                    assert_eq!(mapped, actual, "legal moves must map under the symmetry");

                    let mut enc = [0.0f32; INPUT_SIZE];
                    let mut enc_t = [0.0f32; INPUT_SIZE];
                    encode_alphazero_planes(&g, &mut enc);
                    encode_alphazero_planes(&gt, &mut enc_t);
                    for (i, &src) in sym.state_src.iter().enumerate() {
                        assert_eq!(enc_t[i], enc[src as usize], "encoding must commute");
                    }

                    let &mv = moves.choose(&mut rng).unwrap();
                    g = g.apply(mv);
                    gt = gt.apply(map(mv));
                }
                assert_eq!(g.status(), gt.status());
            }
        }
    }

    #[test]
    fn block_net_layout_matches_uttt_move_indexing() {
        use crate::nn::BlockLayout;
        use candle_core::{Device, Tensor};
        let planes = 2;
        let layout = BlockLayout { planes, rows: 3, cols: 3, bh: 3, bw: 3 };
        let data: Vec<f32> = (0..planes * 81).map(|i| i as f32).collect();
        let xs = Tensor::from_slice(&data, (1, planes * 81), &Device::Cpu).unwrap();
        let tokens = layout.tokens(&xs).unwrap().to_vec3::<f32>().unwrap();
        for big in 0..9u8 {
            for plane in 0..planes {
                for small in 0..9u8 {
                    let expected = plane * 81 + UtttMove::new(big, small).to_policy_index();
                    assert_eq!(tokens[0][big as usize][plane * 9 + small as usize], expected as f32);
                }
            }
        }
        // Round trip: logits laid out per (big, small) must land on `to_policy_index`.
        let per_sub: Vec<f32> = (0..81).map(|i| i as f32).collect();
        let p = Tensor::from_slice(&per_sub, (1, 9, 9), &Device::Cpu).unwrap();
        let grid = layout.logits_to_grid(&p).unwrap().to_vec2::<f32>().unwrap();
        for big in 0..9u8 {
            for small in 0..9u8 {
                let idx = UtttMove::new(big, small).to_policy_index();
                assert_eq!(grid[0][idx], (big as usize * 9 + small as usize) as f32);
            }
        }
    }
}
