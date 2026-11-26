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
//! - **Round Game**: [`RoundGame`] chains scrambles of rising depth into one game, a curriculum
//!   that follows the network's skill (trained by `examples/train_rubiks.rs`).
//!
//! Solving the cube pays `1`; running out of moves pays `0` (not `-1`), so that with a discounted
//! search ([`RubiksSettings::discount`]) the value of a cube is `γ^(moves to solved)` and an
//! unsolved cube is simply worth nothing.

use std::fmt;

use rand::prelude::{Rng, RngExt};
use serde::{Deserialize, Serialize};

use crate::alphazero::{AlphaZeroGame, Episode, SelfPlayConfig, SelfPlayResult, run_chained_self_play};
use crate::hyperparams::AlphaZeroHyperparams;
use crate::inference::{BatchEvaluator, InferenceService};
use crate::mcts::Fpu;
use crate::game::{Game, GameRules, GameStatus, PlayerIndex};
use crate::minimax::HeuristicEvaluator;
use crate::nn::AlphaZeroNetConfig;

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

/// Where sticker `idx` sits: the position of its cubie (`x` right, `y` up, `z` front; each in
/// `-1..=1`) and the outward normal of its face. For drawing the cube.
pub const fn sticker_geometry(idx: usize) -> ([i8; 3], [i8; 3]) {
    let s = sticker_3d(idx);
    (s.pos, s.normal)
}

/// The face turned by `mv`, as its outward normal: the move turns the layer of cubies on that
/// side by 90°, clockwise as seen from outside for even moves (`U`, `D`, ...) and
/// counter-clockwise for the primed ones.
pub const fn move_axis(mv: RubiksMove) -> [i8; 3] {
    FACE_NORMALS[(mv / 2) as usize]
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

    /// The cube after `moves` from solved, with `remaining_steps` moves left and every move
    /// allowed next (like a fresh scramble).
    pub fn after_moves(moves: &[RubiksMove], remaining_steps: u8) -> Self {
        let stickers = moves.iter().fold(SOLVED_STICKERS, |s, &mv| apply_move_stickers(&s, mv));
        Self { stickers, remaining_steps, last_move: NO_MOVE }
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
            GameStatus::Terminal([0.0])
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

// ---------------------------------------------------------------------------------------------
// AlphaZero integration
// ---------------------------------------------------------------------------------------------

impl AlphaZeroGame<INPUT_SIZE, POLICY_SIZE, 1> for RubiksCube {
    const SENTINEL_MOVE: Self::Move = NO_MOVE;

    #[inline]
    fn move_to_policy_index(mv: Self::Move) -> usize {
        mv as usize
    }

    #[inline]
    fn encode_state(game: &Game<Self, 1>, out: &mut [f32; INPUT_SIZE]) {
        encode_one_hot(game.state(), out);
    }

    /// Residual MLP over the one-hot stickers, sized like DeepCubeA's network (~15M parameters)
    /// so that it can learn to solve any cube: 1024 wide, 6 residual blocks (~13M parameters).
    /// Convolutions do not fit: neighbouring stickers lie on different faces.
    fn net_config() -> AlphaZeroNetConfig {
        AlphaZeroNetConfig::ResMlp {
            hidden_dim: 1024,
            num_res_blocks: 6,
            head_hidden: 256,
        }
    }

    /// Settings for the [`RoundGame`] (the cube-specific ones are in [`RubiksSettings`]):
    /// - search: 200 simulations per move, no temperature moves (a wrong move only costs moves)
    ///   and root noise for exploration;
    /// - `fpu` 0: an untried move is assumed to lead nowhere (worth what an unsolved cube is worth);
    /// - training: 500 steps of 512 positions; a game yields about 20 positions at first and about
    ///   90 once the network reaches round 11, so SGD sees each new position about 25 times in the
    ///   first iteration and about 6 times later; the buffer holds about 11 iterations of 44k
    ///   positions (see `docs/playbooks/rubiks-cube.md`).
    fn hyperparams() -> AlphaZeroHyperparams {
        AlphaZeroHyperparams {
            games_per_iteration: 512,
            simulations_per_move: 200,
            opening_moves: 0,
            dirichlet_alpha: 0.5,
            dirichlet_epsilon: 0.25,
            fpu: Fpu::Absolute(0.0),
            value_q_weight: 0.0,
            train_steps: 500,
            batch_size: 512,
            learning_rate: 1e-3,
            buffer_capacity: 500_000,
            eval_games: 128,
            eval_sims: 400,
            ..AlphaZeroHyperparams::GENERIC
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Round game
// ---------------------------------------------------------------------------------------------

/// Cube-specific training settings, on top of the shared [`AlphaZeroHyperparams`]. In JSON the
/// round rules sit next to `discount` (`{"discount": 0.9, "max_rounds": 26, ...}`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RubiksSettings {
    /// Per-move discount `γ` of search values and value targets
    /// ([`crate::alphazero::SelfPlayConfig::discount`]): the network learns `γ^(moves to solved)`,
    /// so `0.9` values a cube solved in 5 moves at `0.59` and one solved in 20 moves at `0.12`.
    pub discount: f32,
    /// Rules of the game played in self-play and evaluation.
    #[serde(flatten)]
    pub rounds: RoundGame,
}

impl Default for RubiksSettings {
    fn default() -> Self {
        Self { discount: 0.9, rounds: RoundGame::default() }
    }
}

impl RubiksSettings {
    /// Clamps the settings to sane ranges.
    pub fn sanitized(self) -> Self {
        let r = self.rounds;
        let max_rounds = r.max_rounds.clamp(1, 40);
        Self {
            discount: self.discount.clamp(0.5, 1.0),
            rounds: RoundGame {
                first_round: r.first_round.clamp(1, max_rounds),
                max_rounds,
                moves_per_turn: r.moves_per_turn.min(10),
                extra_moves: r.extra_moves,
                max_moves: r.max_moves.max(1),
            },
        }
    }

    /// The self-play games of one iteration: [`Self::rounds`] with `hp`'s search settings, root
    /// noise and this discount.
    pub fn self_play<E>(&self, service: &InferenceService<E, INPUT_SIZE, POLICY_SIZE, 1>, hp: &AlphaZeroHyperparams, seed: u64) -> SelfPlayResult<INPUT_SIZE, POLICY_SIZE, 1>
    where
        E: BatchEvaluator<INPUT_SIZE, POLICY_SIZE, 1> + 'static,
    {
        Self::play(self.rounds, service, &hp.self_play_config(seed).with_discount(self.discount))
    }

    /// The evaluation games: [`Self::eval_rounds`], `hp.eval_games` games at
    /// `hp.eval_simulations()` per move, without noise.
    pub fn evaluate<E>(&self, service: &InferenceService<E, INPUT_SIZE, POLICY_SIZE, 1>, hp: &AlphaZeroHyperparams, seed: u64) -> SelfPlayResult<INPUT_SIZE, POLICY_SIZE, 1>
    where
        E: BatchEvaluator<INPUT_SIZE, POLICY_SIZE, 1> + 'static,
    {
        let config = SelfPlayConfig::new(hp.eval_games, hp.eval_simulations(), seed)
            .with_workers(hp.eval_thread_count())
            .with_opening_moves(0)
            .with_puct(hp.c_base, hp.c_init)
            .with_fpu(hp.fpu)
            .with_dirichlet(0.3, 0.0)
            .with_discount(self.discount);
        Self::play(self.eval_rounds(), service, &config)
    }

    /// The rules of evaluation games: those of self-play, but always from round 1, so that
    /// evaluation scores stay comparable when [`RoundGame::first_round`] changes.
    pub fn eval_rounds(&self) -> RoundGame {
        RoundGame { first_round: 1, ..self.rounds }
    }

    fn play<E>(rounds: RoundGame, service: &InferenceService<E, INPUT_SIZE, POLICY_SIZE, 1>, config: &SelfPlayConfig) -> SelfPlayResult<INPUT_SIZE, POLICY_SIZE, 1>
    where
        E: BatchEvaluator<INPUT_SIZE, POLICY_SIZE, 1> + 'static,
    {
        run_chained_self_play(service, config, &|played, rng| rounds.next_round(played, rng))
    }
}

/// How a set of round games went.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RoundStats {
    pub games: usize,
    /// Mean game score and its standard error.
    pub score: f64,
    pub score_stderr: f64,
    /// Highest round solved in any game.
    pub best_round: usize,
    /// Per round, from round 1 to the highest one played: games that played it, solved it, and
    /// the total moves of those solves. Rounds below [`RoundGame::first_round`] count as not
    /// reached.
    pub reached: Vec<usize>,
    pub solved: Vec<usize>,
    pub solve_moves: Vec<usize>,
}

impl RoundStats {
    /// Statistics of round games played by `rules`, whose rounds ended as `games`
    /// ([`SelfPlayResult::episodes`]).
    pub fn new(rules: &RoundGame, games: &[Vec<Episode<1>>]) -> Self {
        let skipped = rules.first_round as usize - 1;
        let rounds = skipped + games.iter().map(Vec::len).max().unwrap_or(0);
        let mut stats = Self {
            games: games.len(),
            reached: vec![0; rounds],
            solved: vec![0; rounds],
            solve_moves: vec![0; rounds],
            ..Self::default()
        };
        let scores: Vec<f64> = games.iter().map(|g| rules.score(g)).collect();
        let n = scores.len() as f64;
        stats.score = scores.iter().sum::<f64>() / n.max(1.0);
        stats.score_stderr =
            (scores.iter().map(|s| (s - stats.score).powi(2)).sum::<f64>() / (n - 1.0).max(1.0) / n.max(1.0)).sqrt();
        for game in games {
            for (i, round) in game.iter().enumerate().map(|(i, round)| (skipped + i, round)) {
                stats.reached[i] += 1;
                if solved(round) {
                    stats.solved[i] += 1;
                    stats.solve_moves[i] += round.moves;
                }
            }
        }
        stats.best_round = stats.solved.iter().rposition(|&s| s > 0).map_or(0, |i| i + 1);
        stats
    }

    /// One line per round played: games that reached it, solve rate and average moves of the solves.
    pub fn table(&self) -> String {
        let mut out = String::from("      round | reached | solved | moves (avg)\n");
        for (i, (&reached, &solved)) in self.reached.iter().zip(&self.solved).enumerate() {
            if reached == 0 {
                continue;
            }
            let moves = if solved > 0 { format!("{:.1}", self.solve_moves[i] as f64 / solved as f64) } else { "-".into() };
            let rate = 100.0 * solved as f64 / reached.max(1) as f64;
            out += &format!("      {:>5} | {reached:>7} | {rate:>5.0}% | {moves:>6}\n", i + 1);
        }
        out
    }
}

/// A game of rounds of rising difficulty: round `n` is a cube scrambled with `n` quarter turns
/// ([`RubiksState::scrambled`]) to be solved within [`Self::move_budget`] moves. The game starts
/// at `first_round`; solving round `n` leads to round `n + 1`, up to `max_rounds`; failing ends
/// the game.
///
/// Solving round `n` in `r` moves scores `n + 1/r`, and a game scores what its last solved round
/// scored: a higher round always scores more, and fewer moves earn a bonus. Rounds below
/// `first_round` count as solved, so a game that fails its first round scores `first_round - 1`.
///
/// Each round is one episode of [`crate::alphazero::run_chained_self_play`]. The next scramble is
/// random, so rounds are searched and learned on their own; the game decides *which* cubes are
/// played, moving on to deeper scrambles as the network improves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoundGame {
    /// First round played. Self-play can skip the rounds the network has mastered, since their
    /// cubes still come up on the way to solving deeper scrambles (a cube `k` moves from solved is
    /// a round-`k` cube); evaluation always starts at round 1 ([`RubiksSettings::eval_rounds`]).
    pub first_round: u8,
    /// Last round (26 quarter turns reach every cube).
    pub max_rounds: u8,
    /// Round `n` allows `moves_per_turn × n + extra_moves` moves, at most `max_moves`: room for
    /// detours that grows with the scramble, while a failed round (which ends every game) stays
    /// cheap and does not swamp the training data with unsolved positions.
    pub moves_per_turn: u8,
    pub extra_moves: u8,
    pub max_moves: u8,
}

impl Default for RoundGame {
    fn default() -> Self {
        Self { first_round: 1, max_rounds: 26, moves_per_turn: 2, extra_moves: 10, max_moves: 100 }
    }
}

impl RoundGame {
    /// Moves allowed in round `round` (1-based).
    pub fn move_budget(&self, round: usize) -> u8 {
        (self.moves_per_turn as usize * round + self.extra_moves as usize).min(self.max_moves as usize) as u8
    }

    /// The cube of the round after `played` (`first_round` if `played` is empty), or `None` when
    /// the game is over: the last round failed or `max_rounds` were solved.
    pub fn next_round(&self, played: &[Episode<1>], rng: &mut impl Rng) -> Option<RubiksGame> {
        let round = self.first_round as usize + played.len();
        if played.last().is_some_and(|r| !solved(r)) || round > self.max_rounds as usize {
            return None;
        }
        // Scrambles that cancel out (e.g. `U D U' D'`) are drawn again.
        loop {
            let state = RubiksState::scrambled(round as u8, self.move_budget(round), rng);
            if !state.is_solved() {
                return Some(RubiksGame::new(PlayerIndex::P0, state));
            }
        }
    }

    /// Score of a game whose rounds ended as `played`: `n + 1/r` of its last solved round `n`,
    /// or `first_round - 1` if it solved none.
    pub fn score(&self, played: &[Episode<1>]) -> f64 {
        let skipped = self.first_round as usize - 1;
        played
            .iter()
            .rposition(solved)
            .map_or(skipped as f64, |i| (skipped + i + 1) as f64 + 1.0 / played[i].moves as f64)
    }
}

fn solved(round: &Episode<1>) -> bool {
    round.payoffs[0] > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::{SeedableRng, SmallRng};

    use crate::matchup::play_match;
    use crate::mcts::{MctsConfig, MctsPlayer, PlayoutEvaluator, Uct};
    use crate::minimax::MinimaxPlayer;

    #[test]
    fn round_game_continues_after_solves_and_scores_the_last_solved_round() {
        let game = RoundGame { max_rounds: 3, ..RoundGame::default() };
        let mut rng = SmallRng::seed_from_u64(1);
        let solved = |moves| Episode { payoffs: [1.0], moves };
        let failed = Episode { payoffs: [0.0], moves: 10 };

        let first = game.next_round(&[], &mut rng).unwrap();
        assert!(!first.state().is_solved());
        assert_eq!(first.state().remaining_steps, 12);
        assert_eq!([2, 20, 45, 50].map(|n| game.move_budget(n)), [14, 50, 100, 100]);
        assert!(game.next_round(&[solved(1), solved(4)], &mut rng).is_some(), "round 3 follows");
        assert!(game.next_round(&[solved(1), failed], &mut rng).is_none(), "a failed round ends the game");
        assert!(game.next_round(&[solved(1), solved(2), solved(3)], &mut rng).is_none(), "max_rounds reached");

        assert_eq!(game.score(&[]), 0.0);
        assert_eq!(game.score(&[failed]), 0.0);
        assert_eq!(game.score(&[solved(1), solved(4), failed]), 2.25);
        assert_eq!(game.score(&[solved(1), solved(2), solved(3)]), 3.0 + 1.0 / 3.0);
    }

    #[test]
    fn round_game_can_start_at_a_later_round_and_counts_the_skipped_rounds() {
        let game = RoundGame { first_round: 5, max_rounds: 7, ..RoundGame::default() };
        let mut rng = SmallRng::seed_from_u64(2);
        let solved = |moves| Episode { payoffs: [1.0], moves };
        let failed = Episode { payoffs: [0.0], moves: 20 };

        let first = game.next_round(&[], &mut rng).unwrap();
        assert_eq!(first.state().remaining_steps, game.move_budget(5), "the game starts with round 5");
        assert!(game.next_round(&[solved(5), solved(6), solved(7)], &mut rng).is_none(), "max_rounds reached");
        assert_eq!(game.score(&[failed]), 4.0, "rounds 1-4 count as solved");
        assert!((game.score(&[solved(5), failed]) - 5.2).abs() < 1e-12);

        let stats = RoundStats::new(&game, &[vec![solved(5), failed], vec![failed]]);
        assert_eq!(stats.reached, [0, 0, 0, 0, 2, 1]);
        assert_eq!(stats.solved, [0, 0, 0, 0, 1, 0]);
        assert_eq!(stats.best_round, 5);
        assert!((stats.score - 4.6).abs() < 1e-12);
        assert_eq!(stats.table().lines().count(), 3, "a header and rounds 5 and 6:\n{}", stats.table());

        let settings = RubiksSettings { rounds: game, ..RubiksSettings::default() };
        assert_eq!(settings.eval_rounds().first_round, 1, "evaluation always starts at round 1");
    }

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
