//! The Rubik's cube in the Browser Lab's Play tab.
//!
//! Stateless like [`crate::play`]: the browser keeps the moves made from the solved cube (scramble
//! and turns) and sends them with every request; the server returns the stickers to draw, the
//! cube's geometry, and optionally an AlphaZero analysis of the position: per move the network's
//! prior, the search's visits and value, and the move it plays.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::alphazero::{AlphaZeroGame, NnEvaluator};
use crate::backend::NetEvaluator;
use crate::game::PlayerIndex;
use crate::games::rubiks_cube::{
    INPUT_SIZE, MOVE_NAMES, NO_MOVE, NUM_MOVES, NUM_STICKERS, POLICY_SIZE, RubiksCube, RubiksGame, RubiksMove, RubiksState,
    move_axis, sticker_geometry,
};
use crate::inference::{BatchEvaluator, InferenceService};
use crate::mcts::{MctsSearch, Puct};
use crate::tree::NodeId;
use crate::opponents::OpponentSpec;

/// Inference service over the cube's network.
pub type CubeService = InferenceService<NetEvaluator<INPUT_SIZE, POLICY_SIZE, 1>, INPUT_SIZE, POLICY_SIZE, 1>;

/// `POST /api/play` body for `"game": "rubiks"`.
#[derive(Debug, Deserialize)]
pub struct CubePlayRequest {
    /// Moves made from the solved cube, scramble included (`0..12`, see [`MOVE_NAMES`]).
    #[serde(default)]
    pub moves: Vec<RubiksMove>,
    /// Moves the engine may still use to solve the cube.
    #[serde(default = "default_moves_left")]
    pub moves_left: u8,
    /// Engine to analyse the cube with (AlphaZero only; `None`: cube only).
    #[serde(default)]
    pub engine: Option<OpponentSpec>,
    #[serde(default)]
    pub seed: u64,
}

const fn default_moves_left() -> u8 {
    100
}

#[derive(Debug, Serialize)]
pub struct CubeResponse {
    /// Colour of every sticker (`0..6`: U white, D yellow, L orange, R red, F green, B blue).
    pub stickers: Vec<u8>,
    pub solved: bool,
    pub geometry: Geometry,
    pub analysis: Option<CubeAnalysis>,
}

/// Everything needed to draw the cube and animate its moves.
#[derive(Debug, Serialize)]
pub struct Geometry {
    /// Per sticker: the position of its cubie (`x` right, `y` up, `z` front, each `-1..=1`) and
    /// the outward normal of its face.
    pub stickers: Vec<([i8; 3], [i8; 3])>,
    /// Per move: its name and the face it turns (outward normal); even moves turn clockwise as
    /// seen from outside, odd (primed) moves counter-clockwise.
    pub moves: Vec<(&'static str, [i8; 3])>,
}

impl Geometry {
    fn new() -> Self {
        Self {
            stickers: (0..NUM_STICKERS).map(sticker_geometry).collect(),
            moves: (0..NUM_MOVES).map(|mv| (MOVE_NAMES[mv as usize], move_axis(mv))).collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CubeAnalysis {
    pub engine: String,
    /// The move the engine plays.
    pub chosen: RubiksMove,
    /// Per move, most visited first.
    pub moves: Vec<CubeMoveStat>,
    /// Search value of the cube along the chosen move: `γ^(moves to solved)` as the search
    /// estimates it (`0`: no solve in sight). Averaging over the search's detours makes it a
    /// cautious estimate.
    pub value: f32,
    /// Moves to solved along the chosen move (`None`: no solve in sight).
    pub distance: Option<f32>,
    pub simulations: u64,
    pub millis: f64,
}

#[derive(Debug, Serialize)]
pub struct CubeMoveStat {
    #[serde(rename = "move")]
    pub mv: RubiksMove,
    pub prior: f32,
    pub visits: u64,
    /// Value of the cube after the move (`γ^(moves to solved)`; `0` if unvisited).
    pub q: f32,
    /// Moves to solved through this move, as implied by `q` (`None` when it is `0`).
    pub distance: Option<f32>,
}

/// Answers a Play-tab request. `discount` is the cube's training discount (the network's values
/// are `γ^(moves to solved)`); `network` provides the network of a checkpoint (`"live"`: the live one).
pub fn respond(
    req: &CubePlayRequest,
    discount: f32,
    network: impl FnOnce(&str) -> Result<Arc<CubeService>, String>,
) -> Result<CubeResponse, String> {
    if let Some(&bad) = req.moves.iter().find(|&&mv| mv >= NUM_MOVES) {
        return Err(format!("invalid move {bad}"));
    }
    let state = RubiksState::after_moves(&req.moves, req.moves_left);
    let analysis = match &req.engine {
        Some(OpponentSpec::AlphaZero { checkpoint, sims }) if !state.is_solved() && state.remaining_steps > 0 => {
            Some(analyse(state, network(checkpoint)?.as_ref(), *sims, discount, req.seed, checkpoint))
        }
        Some(OpponentSpec::AlphaZero { .. }) | None => None,
        Some(other) => return Err(format!("{} cannot solve the cube; use an AlphaZero network", other.label())),
    };
    Ok(CubeResponse { stickers: state.stickers.to_vec(), solved: state.is_solved(), geometry: Geometry::new(), analysis })
}

fn analyse<E>(
    state: RubiksState,
    service: &InferenceService<E, INPUT_SIZE, POLICY_SIZE, 1>,
    sims: usize,
    discount: f32,
    seed: u64,
    checkpoint: &str,
) -> CubeAnalysis
where
    E: BatchEvaluator<INPUT_SIZE, POLICY_SIZE, 1> + 'static,
{
    let start = Instant::now();
    let hp = RubiksCube::hyperparams();
    let puct = Puct::new(hp.c_base, hp.c_init).with_fpu(hp.fpu);
    let sims = sims.clamp(1, 100_000);
    let mut search = MctsSearch::new(RubiksGame::new(PlayerIndex::P0, state), NO_MOVE, puct, (sims * 16).max(256), seed)
        .with_discount(discount);
    let mut evaluator = NnEvaluator::new(service, seed).with_dirichlet(0.3, 0.0);
    search.run_simulations(sims, &mut evaluator);

    let mut moves: Vec<CubeMoveStat> = search
        .tree()
        .active()
        .children(NodeId::ROOT)
        .iter()
        .map(|c| {
            let q = c.payload.q_value();
            // `q = γ^d` after the move, so the move itself makes it `d + 1`.
            let distance = (q > 0.0 && discount < 1.0).then(|| q.ln() / discount.ln() + 1.0);
            CubeMoveStat { mv: c.mv, prior: c.payload.prior, visits: c.payload.visits as u64, q, distance }
        })
        .collect();
    moves.sort_by(|a, b| b.visits.cmp(&a.visits).then(b.q.total_cmp(&a.q)));
    let chosen = search.best_root_child().1;
    let best = moves.iter().find(|m| m.mv == chosen);
    // The cube is worth `γ ×` what it is worth after its best move.
    let value = best.map_or(0.0, |m| (discount * m.q).max(0.0));
    let distance = best.and_then(|m| m.distance);
    CubeAnalysis {
        engine: format!("AlphaZero {} @{sims}", if checkpoint.is_empty() { "live" } else { checkpoint }),
        chosen,
        moves,
        value,
        distance,
        simulations: sims as u64,
        millis: start.elapsed().as_secs_f64() * 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FnBatchEvaluator;

    #[test]
    fn cube_play_returns_stickers_geometry_and_a_solving_move() {
        // `R U`: solved by `U' R'`. A blind network (uniform priors, value 0): search alone finds it.
        let blind = FnBatchEvaluator(|_: &[[f32; 324]], policies: &mut [[f32; 12]], values: &mut [[f32; 1]]| {
            policies.fill([0.0; 12]);
            values.fill([0.0]);
        });
        let service = InferenceService::new(blind, 16, 1);
        let req = CubePlayRequest {
            moves: vec![6, 0],
            moves_left: 2,
            engine: Some(OpponentSpec::AlphaZero { checkpoint: "live".into(), sims: 400 }),
            seed: 1,
        };
        let state = RubiksState::after_moves(&req.moves, req.moves_left);
        assert!(!state.is_solved());
        let analysis = analyse(state, &service, 400, 0.9, req.seed, "live");
        assert_eq!(analysis.chosen, 1, "U' undoes the last turn");
        // Solved two moves later: `γ² = 0.81`, a little less after averaging over the detours.
        assert!(analysis.value > 0.6 && analysis.value <= 0.81, "value {}", analysis.value);
        assert!((2.0..5.0).contains(&analysis.distance.unwrap()));
        assert_eq!(analysis.moves[0].mv, 1);

        let geometry = Geometry::new();
        assert_eq!(geometry.stickers.len(), 54);
        assert_eq!(geometry.moves[0], ("U", [0, 1, 0]));
        let solved = respond(&CubePlayRequest { moves: vec![6, 7], moves_left: 5, engine: None, seed: 0 }, 0.9, |_| unreachable!()).unwrap();
        assert!(solved.solved && solved.analysis.is_none());
        assert!(respond(&CubePlayRequest { moves: vec![12], moves_left: 5, engine: None, seed: 0 }, 0.9, |_| unreachable!()).is_err());
    }
}
