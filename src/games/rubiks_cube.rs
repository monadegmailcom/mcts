//! Single-player (`N = 1`) 3x3x3 Rubik's Cube in the Quarter-Turn Metric (`12` moves).
//!
//! Designed for curriculum-trained AlphaZero / *DeepCubeA*-style neural solvers:
//! - **56-Byte Single-Cache-Line State**: [`RubiksState`] stores all `54` facelets (`[Color; 54]`),
//!   `remaining_steps: u8`, and `last_move: u8` in **56 bytes** (`Copy`, zero heap allocation).
//! - **Compile-Time 3D Geometry (`const fn`)**: All 12 face-turn permutations (`MOVE_PERMS`) are
//!   synthesized at compile time from 3D right-hand-rule cross-product rotations on `(pos, normal)`
//!   coordinates—making manual sticker-index transcription bugs impossible and allowing LLVM to
//!   vectorize each cube turn into a handful of SIMD byte-shuffle instructions.
//! - **Curriculum Scrambler**: [`RubiksState::scrambled`] generates scrambles at controlled depths
//!   `k = 1, 2, ...` from the solved cube while avoiding redundant same-axis cancellations.
//! - **324-Float One-Hot Neural Encoding**: [`encode_one_hot`] serializes the 54 stickers x 6
//!   colors into `[f32; 324]` (matching *DeepCubeA* and sharing the exact `324` input width of
//!   [`crate::games::ultimate_ttt::INPUT_SIZE`]).

use std::fmt;

use rand::prelude::{Rng, RngExt};

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::minimax::HeuristicEvaluator;

/// Quarter-turn move index `0..12`:
/// `0: U, 1: U', 2: D, 3: D', 4: L, 5: L', 6: R, 7: R', 8: F, 9: F', 10: B, 11: B'`.
pub type RubiksMove = u8;

/// Number of legal quarter-turn actions (`6` faces x `{CW, CCW}` = `12`).
pub const NUM_MOVES: u8 = 12;
pub const POLICY_SIZE: usize = 12;
pub const NUM_STICKERS: usize = 54;
pub const NUM_FACES: usize = 6;
/// One-hot neural network input size (`54` stickers x `6` colors = `324`).
pub const INPUT_SIZE: usize = NUM_STICKERS * NUM_FACES;

/// Sentinel indicating no prior move (all 12 quarter-turns are allowed at the root of a scramble).
pub const NO_MOVE: RubiksMove = 12;

/// Standard Singmaster notation names for moves `0..12`.
pub const MOVE_NAMES: [&str; 12] = [
    "U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'",
];

/// Returns the exact inverse quarter-turn (`U <-> U'`, `R <-> R'`, etc.).
#[inline]
pub const fn inverse_move(mv: RubiksMove) -> RubiksMove {
    debug_assert!(mv < NUM_MOVES);
    mv ^ 1
}

/// Face / sticker color (`0..6`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    Up = 0,    // White
    Down = 1,  // Yellow
    Left = 2,  // Orange
    Right = 3, // Red
    Front = 4, // Green
    Back = 5,  // Blue
}

impl Color {
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Up,
            1 => Self::Down,
            2 => Self::Left,
            3 => Self::Right,
            4 => Self::Front,
            5 => Self::Back,
            _ => panic!("invalid Rubik's Color index"),
        }
    }

    #[inline]
    pub const fn as_char(self) -> char {
        match self {
            Self::Up => 'W',
            Self::Down => 'Y',
            Self::Left => 'O',
            Self::Right => 'R',
            Self::Front => 'G',
            Self::Back => 'B',
        }
    }
}

/// Solved 54-sticker array (`stickers[face * 9 .. (face + 1) * 9] == face`).
pub const SOLVED_STICKERS: [u8; NUM_STICKERS] = {
    let mut arr = [0u8; NUM_STICKERS];
    let mut i = 0;
    while i < NUM_STICKERS {
        arr[i] = (i / 9) as u8;
        i += 1;
    }
    arr
};

// ---------------------------------------------------------------------------
// Compile-time 3D geometry synthesis of all 12 quarter-turn permutations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Sticker3D {
    pos: [i8; 3],
    normal: [i8; 3],
}

const FACE_NORMALS: [[i8; 3]; 6] = [
    [0, 1, 0],  // 0: U (+y)
    [0, -1, 0], // 1: D (-y)
    [-1, 0, 0], // 2: L (-x)
    [1, 0, 0],  // 3: R (+x)
    [0, 0, 1],  // 4: F (+z)
    [0, 0, -1], // 5: B (-z)
];

const fn sticker_3d(idx: usize) -> Sticker3D {
    let face = idx / 9;
    let cell = idx % 9;
    let row = (cell / 3) as i8;
    let col = (cell % 3) as i8;

    let pos = match face {
        // U (+y): viewed from above, Back (-z) at top, Front (+z) at bottom, Left (-x) to Right (+x)
        0 => [-1 + col, 1, -1 + row],
        // D (-y): viewed from below, Front (+z) at top, Back (-z) at bottom, Left (-x) to Right (+x)
        1 => [-1 + col, -1, 1 - row],
        // L (-x): viewed from left, Up (+y) at top, Back (-z) at left, Front (+z) at right
        2 => [-1, 1 - row, -1 + col],
        // R (+x): viewed from right, Up (+y) at top, Front (+z) at left, Back (-z) at right
        3 => [1, 1 - row, 1 - col],
        // F (+z): viewed from front, Up (+y) at top, Left (-x) at left, Right (+x) at right
        4 => [-1 + col, 1 - row, 1],
        // B (-z): viewed from behind, Up (+y) at top, Right (+x) at left, Left (-x) at right
        5 => [1 - col, 1 - row, -1],
        _ => panic!("invalid face"),
    };

    Sticker3D {
        pos,
        normal: FACE_NORMALS[face],
    }
}

#[inline]
const fn dot3(a: [i8; 3], b: [i8; 3]) -> i8 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rodrigues' 90-degree clockwise rotation `(v . n) * n + (v x n)` when looking against outward
/// unit normal `n` in our right-handed `(+x = Right, +y = Up, +z = Front)` basis.
#[inline]
const fn rotate_cw_90(v: [i8; 3], n: [i8; 3]) -> [i8; 3] {
    let parallel = dot3(v, n);
    [
        parallel * n[0] + (v[1] * n[2] - v[2] * n[1]),
        parallel * n[1] + (v[2] * n[0] - v[0] * n[2]),
        parallel * n[2] + (v[0] * n[1] - v[1] * n[0]),
    ]
}

#[inline]
const fn eq3(a: [i8; 3], b: [i8; 3]) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
}

/// Pull-gather permutation table: `next[i] = current[MOVE_PERMS[mv][i] as usize]`.
const MOVE_PERMS: [[u8; NUM_STICKERS]; 12] = {
    let mut coords = [Sticker3D {
        pos: [0, 0, 0],
        normal: [0, 0, 0],
    }; NUM_STICKERS];
    let mut idx = 0;
    while idx < NUM_STICKERS {
        coords[idx] = sticker_3d(idx);
        idx += 1;
    }

    let mut perms = [[0u8; NUM_STICKERS]; 12];
    let mut face = 0;
    while face < 6 {
        let axis = FACE_NORMALS[face];
        let mut cw = [0u8; NUM_STICKERS];
        let mut ccw = [0u8; NUM_STICKERS];

        let mut src = 0;
        while src < NUM_STICKERS {
            let s = coords[src];
            let dst = if dot3(s.pos, axis) == 1 {
                let new_pos = rotate_cw_90(s.pos, axis);
                let new_norm = rotate_cw_90(s.normal, axis);
                let mut found = NUM_STICKERS;
                let mut k = 0;
                while k < NUM_STICKERS {
                    if eq3(coords[k].pos, new_pos) && eq3(coords[k].normal, new_norm) {
                        found = k;
                        break;
                    }
                    k += 1;
                }
                assert!(found < NUM_STICKERS, "3D rotation did not map to a valid sticker");
                found
            } else {
                src
            };

            // Sticker at `src` moves to `dst` under CW, and `dst` moves to `src` under CCW.
            cw[dst] = src as u8;
            ccw[src] = dst as u8;
            src += 1;
        }

        perms[face * 2] = cw;
        perms[face * 2 + 1] = ccw;
        face += 1;
    }
    perms
};

/// Applies a single quarter-turn `mv` (`0..12`) to a 54-sticker array.
#[inline]
pub fn apply_move_stickers(stickers: &[u8; NUM_STICKERS], mv: RubiksMove) -> [u8; NUM_STICKERS] {
    assert!(mv < NUM_MOVES, "invalid Rubik's Cube move {mv}");
    let perm = &MOVE_PERMS[mv as usize];
    let mut next = [0u8; NUM_STICKERS];
    for i in 0..NUM_STICKERS {
        next[i] = stickers[perm[i] as usize];
    }
    next
}

/// Parses a space-separated Singmaster move sequence (including `2` half-turns such as
/// `"R U R' U'"` or `"U R2 F B'"`) into quarter-turn [`RubiksMove`] steps.
pub fn parse_singmaster(seq: &str) -> Result<Vec<RubiksMove>, String> {
    let mut out = Vec::new();
    for token in seq.split_whitespace() {
        let mut chars = token.chars();
        let base_face = match chars.next() {
            Some('U') => 0u8,
            Some('D') => 2u8,
            Some('L') => 4u8,
            Some('R') => 6u8,
            Some('F') => 8u8,
            Some('B') => 10u8,
            _ => return Err(format!("unknown Rubik's face in token '{token}'")),
        };
        match chars.as_str() {
            "" => out.push(base_face),
            "'" => out.push(base_face + 1),
            "2" => {
                out.push(base_face);
                out.push(base_face);
            }
            _ => return Err(format!("invalid Rubik's modifier in token '{token}'")),
        }
    }
    Ok(out)
}

/// Complete single-player (`N = 1`) 3x3x3 Rubik's Cube game state (`56` bytes, `Copy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RubiksState {
    pub stickers: [u8; NUM_STICKERS],
    pub remaining_steps: u8,
    pub last_move: RubiksMove,
}

impl RubiksState {
    #[inline]
    pub const fn solved(remaining_steps: u8) -> Self {
        Self {
            stickers: SOLVED_STICKERS,
            remaining_steps,
            last_move: NO_MOVE,
        }
    }

    #[inline]
    pub fn is_solved(&self) -> bool {
        self.stickers == SOLVED_STICKERS
    }

    /// Generates a curriculum scramble `scramble_depth` quarter-turns away from solved,
    /// avoiding immediate inverse cancellations (`X X'`) and 3-turn same-face loops (`X X X`),
    /// then resets `last_move = NO_MOVE` so the solver sees only the scrambled cube and budget.
    pub fn scrambled(scramble_depth: u8, step_budget: u8, rng: &mut impl Rng) -> Self {
        let mut stickers = SOLVED_STICKERS;
        let mut prev_mv: Option<u8> = None;
        let mut prev_prev_mv: Option<u8> = None;

        for _ in 0..scramble_depth {
            let mv = loop {
                let candidate = rng.random_range(0..NUM_MOVES);
                // Never immediately undo the previous quarter-turn (`X` followed by `X'`).
                if prev_mv == Some(candidate ^ 1) {
                    continue;
                }
                // Never apply three identical quarter-turns (`X X X` == `X'`, shorter as 1 turn).
                if prev_mv == Some(candidate) && prev_prev_mv == Some(candidate) {
                    continue;
                }
                break candidate;
            };
            stickers = apply_move_stickers(&stickers, mv);
            prev_prev_mv = prev_mv;
            prev_mv = Some(mv);
        }

        Self {
            stickers,
            remaining_steps: step_budget,
            last_move: NO_MOVE,
        }
    }
}

impl fmt::Display for RubiksState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = |face: usize, row: usize, col: usize| -> char {
            Color::from_u8(self.stickers[face * 9 + row * 3 + col]).as_char()
        };
        // Unfolded net:
        //       [U]
        // [L] [F] [R] [B]
        //       [D]
        for r in 0..3 {
            writeln!(f, "      {} {} {}", c(0, r, 0), c(0, r, 1), c(0, r, 2))?;
        }
        for r in 0..3 {
            writeln!(
                f,
                "{} {} {}  {} {} {}  {} {} {}  {} {} {}",
                c(2, r, 0),
                c(2, r, 1),
                c(2, r, 2),
                c(4, r, 0),
                c(4, r, 1),
                c(4, r, 2),
                c(3, r, 0),
                c(3, r, 1),
                c(3, r, 2),
                c(5, r, 0),
                c(5, r, 1),
                c(5, r, 2),
            )?;
        }
        for r in 0..3 {
            writeln!(f, "      {} {} {}", c(1, r, 0), c(1, r, 1), c(1, r, 2))?;
        }
        Ok(())
    }
}

/// Zero-sized game rules marker for the single-player (`N = 1`) 3x3x3 Rubik's Cube.
#[derive(Clone, Copy, Debug)]
pub struct RubiksCube;

pub type RubiksGame = Game<RubiksCube, 1>;

/// Zero-allocation iterator over legal quarter-turns (`0..12`, skipping `last_move ^ 1` when set).
pub struct RubiksMoveIter {
    cursor: u8,
    end: u8,
    forbidden: u8,
}

impl Iterator for RubiksMoveIter {
    type Item = RubiksMove;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.cursor < self.end {
            let mv = self.cursor;
            self.cursor += 1;
            if mv != self.forbidden {
                return Some(mv);
            }
        }
        None
    }
}

impl GameRules<1> for RubiksCube {
    type Move = RubiksMove;
    type State = RubiksState;
    type MoveIter<'a> = RubiksMoveIter;

    #[inline]
    fn valid_moves<'a>(state: &'a Self::State, _player: PlayerIndex<1>) -> Self::MoveIter<'a> {
        if state.is_solved() || state.remaining_steps == 0 {
            return RubiksMoveIter {
                cursor: 0,
                end: 0,
                forbidden: NO_MOVE,
            };
        }
        let forbidden = if state.last_move < NUM_MOVES {
            state.last_move ^ 1
        } else {
            NO_MOVE
        };
        RubiksMoveIter {
            cursor: 0,
            end: NUM_MOVES,
            forbidden,
        }
    }

    #[inline]
    fn write_valid_moves(
        state: &Self::State,
        _player: PlayerIndex<1>,
        out: &mut Vec<Self::Move>,
    ) {
        out.clear();
        if state.is_solved() || state.remaining_steps == 0 {
            return;
        }
        let forbidden = if state.last_move < NUM_MOVES {
            state.last_move ^ 1
        } else {
            NO_MOVE
        };
        for mv in 0..NUM_MOVES {
            if mv != forbidden {
                out.push(mv);
            }
        }
    }

    #[inline]
    fn apply(state: &Self::State, _player: PlayerIndex<1>, mv: Self::Move) -> Self::State {
        assert!(
            state.remaining_steps > 0,
            "cannot apply Rubik's move when remaining_steps == 0"
        );
        RubiksState {
            stickers: apply_move_stickers(&state.stickers, mv),
            remaining_steps: state.remaining_steps - 1,
            last_move: mv,
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

/// Lightweight sticker-alignment heuristic in `[-1.0, +1.0]` for baseline `Max^1` / MCTS tests
/// prior to neural network training (counts the fraction of the 48 non-center stickers matching
/// their solved face color, mapped to `[-1.0, +1.0]`).
#[derive(Clone, Copy, Debug, Default)]
pub struct StickerMatchHeuristic;

impl HeuristicEvaluator<RubiksCube, 1> for StickerMatchHeuristic {
    #[inline]
    fn evaluate_heuristic(&self, game: &RubiksGame) -> [f32; 1] {
        let s = &game.state().stickers;
        let mut matched = 0u32;
        for (i, &color) in s.iter().enumerate() {
            // Skip the 6 fixed face centers (`i % 9 == 4`).
            if i % 9 != 4 && color == SOLVED_STICKERS[i] {
                matched += 1;
            }
        }
        // Map `0..=48` into `[-1.0, +1.0]` (`< 1.0` for all unsolved states so `TERMINAL_SCORE_BASE` dominates).
        [(matched as f32) / 24.0 - 1.0]
    }
}

/// Encodes the 54 stickers into a `[f32; 324]` one-hot vector (`54 x 6`) for AlphaZero / DeepCubeA.
#[inline]
pub fn encode_one_hot(state: &RubiksState, out: &mut [f32; INPUT_SIZE]) {
    out.fill(0.0);
    for (idx, &color) in state.stickers.iter().enumerate() {
        out[idx * NUM_FACES + (color as usize)] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::{SeedableRng, SmallRng};

    use crate::matchup::play_match;
    use crate::mcts::{MctsConfig, MctsPlayer, PlayoutEvaluator, Uct};
    use crate::minimax::MinimaxPlayer;

    #[test]
    fn rubiks_state_fits_in_single_64_byte_cache_line() {
        assert_eq!(std::mem::size_of::<RubiksState>(), 56);
    }

    #[test]
    fn every_quarter_turn_has_order_4_and_exact_inverse() {
        for mv in 0..NUM_MOVES {
            let once = apply_move_stickers(&SOLVED_STICKERS, mv);
            assert_ne!(once, SOLVED_STICKERS, "move {} left cube unchanged", MOVE_NAMES[mv as usize]);

            let back = apply_move_stickers(&once, inverse_move(mv));
            assert_eq!(back, SOLVED_STICKERS);

            let mut cur = SOLVED_STICKERS;
            for step in 1..=4 {
                cur = apply_move_stickers(&cur, mv);
                if step < 4 {
                    assert_ne!(cur, SOLVED_STICKERS);
                } else {
                    assert_eq!(cur, SOLVED_STICKERS);
                }
            }
        }
    }

    #[test]
    fn group_theory_identities_sexy_move_order_6_ru_order_105_and_superflip_involution() {
        // 1. `[R, U, R', U']` ("Sexy Move") has exact order 6 in the 3x3x3 Rubik's group.
        let sexy = parse_singmaster("R U R' U'").unwrap();
        let mut s = SOLVED_STICKERS;
        for rep in 1..=6 {
            for &mv in &sexy {
                s = apply_move_stickers(&s, mv);
            }
            if rep < 6 {
                assert_ne!(s, SOLVED_STICKERS, "R U R' U' returned to solved early at rep {rep}");
            } else {
                assert_eq!(s, SOLVED_STICKERS, "R U R' U' failed order-6 identity");
            }
        }

        // 2. `R U` has exact order 105 (LCM(3, 5, 7) from corner/edge permutation + orientation!).
        let ru = parse_singmaster("R U").unwrap();
        let mut s_ru = SOLVED_STICKERS;
        for rep in 1..=105 {
            for &mv in &ru {
                s_ru = apply_move_stickers(&s_ru, mv);
            }
            if rep < 105 {
                assert_ne!(s_ru, SOLVED_STICKERS, "R U returned to solved early at rep {rep}");
            } else {
                assert_eq!(s_ru, SOLVED_STICKERS, "R U failed order-105 identity");
            }
        }

        // 3. Michael Reid's 20-HTM Superflip (`U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2`)
        // is an involution (order 2): applying it once flips all 12 edges (leaving centers + corners
        // in place), and applying it a second time restores `SOLVED_STICKERS`!
        let superflip =
            parse_singmaster("U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2").unwrap();
        let mut flipped = SOLVED_STICKERS;
        for &mv in &superflip {
            flipped = apply_move_stickers(&flipped, mv);
        }
        assert_ne!(flipped, SOLVED_STICKERS);
        // Verify all 8 corners (the 4 even indices `0, 2, 6, 8` on every face!) match solved colors!
        for face in 0..6 {
            for &corner_cell in &[0usize, 2, 6, 8] {
                assert_eq!(
                    flipped[face * 9 + corner_cell],
                    face as u8,
                    "superflip disturbed corner ({face}, {corner_cell})"
                );
            }
        }
        // Apply Superflip a second time -> exact solved cube!
        for &mv in &superflip {
            flipped = apply_move_stickers(&flipped, mv);
        }
        assert_eq!(flipped, SOLVED_STICKERS);
    }

    #[test]
    fn single_player_maxn_and_mcts_solve_3_move_rubiks_cube_scrambles() {
        let mut rng = SmallRng::seed_from_u64(2026);

        // 1. `Max^1` (`MinimaxPlayer` with `N = 1`) finds exact shortest <= 3-move solutions!
        for trial in 0..10 {
            let state = RubiksState::scrambled(3, 5, &mut rng);
            if state.is_solved() {
                continue;
            }
            let game = RubiksGame::new(PlayerIndex::P0, state);
            let solver = MinimaxPlayer::new(game, StickerMatchHeuristic, 2, trial);
            let outcome = play_match(game, &mut [solver]);
            assert_eq!(outcome.payoffs, [1.0]);
            assert!(outcome.moves_played <= 3);
        }

        // 2. Pure single-threaded UCT MCTS (`N = 1`, zero neural network yet!) solves 3-move scrambles!
        let scramble_3 = RubiksState::scrambled(3, 4, &mut rng);
        let game = RubiksGame::new(PlayerIndex::P0, scramble_3);
        let mcts_solver = MctsPlayer::new(
            game,
            NO_MOVE,
            Uct::new(1.4),
            PlayoutEvaluator::new(101),
            MctsConfig::new(2_500, 202).with_arena_capacity(32_768),
        );
        let outcome = play_match(game, &mut [mcts_solver]);
        assert_eq!(
            outcome.payoffs,
            [1.0],
            "expected N=1 MCTS to solve 3-move 3x3x3 Rubik's scramble"
        );
    }
}
