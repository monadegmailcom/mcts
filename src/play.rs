//! Single games for the Browser Lab's Play tab.
//!
//! The browser keeps the move list and sends it with every request, so the server holds no
//! per-game state: it replays the moves from the start position and returns the board to draw
//! and, optionally, one engine's analysis of that position — the move the engine plays and, for
//! search engines, per-move statistics for the heat map (search share, network prior, win rate).

use std::sync::Arc;
use std::time::Instant;

use rand::prelude::{IndexedRandom, SeedableRng, SmallRng};
use serde::{Deserialize, Serialize};

use crate::alphazero::{AlphaZeroConfig, AlphaZeroGame, AlphaZeroPlayer};
use crate::backend::NetEvaluator;
use crate::game::{Game, GameStatus};
use crate::inference::InferenceService;
use crate::mcts::{PlayoutEvaluator, Uct};
use crate::minimax::{HeuristicEvaluator, TERMINAL_SCORE_BASE, root_move_scores};
use crate::opponents::OpponentSpec;
use crate::root_parallel::{AggregatedRootChild, RootParallelBudget, RootParallelConfig, RootParallelPlayer};

/// Inference service over a two-player game's network.
pub type NetService<const G: usize, const P: usize> = InferenceService<NetEvaluator<G, P, 2>, G, P, 2>;

/// What the Play tab needs from a two-player board game beyond [`AlphaZeroGame`]. Cells are
/// addressed by policy index, so the network's outputs map directly onto the board.
pub trait PlayableGame<const G: usize, const P: usize>: AlphaZeroGame<G, P, 2> {
    type Heuristic: HeuristicEvaluator<Self, 2> + Copy + Send + Sync;

    fn start() -> Game<Self, 2>;

    /// Leaf evaluation for Minimax.
    fn heuristic() -> Self::Heuristic;

    /// Starting positions for evaluation matches, each played once per seat. Deterministic engines
    /// (AlphaZero playing its best move, Minimax) would otherwise replay the same few games from
    /// the start position. Must be deterministic so results are comparable over time.
    fn eval_openings(count: usize) -> Vec<Game<Self, 2>>;

    /// Owner of every cell in policy-index order: `0` empty, `1` first player, `2` second player.
    fn cells(game: &Game<Self, 2>) -> Vec<u8>;

    /// Status of each sub-board for boards made of sub-boards (`0` open, `1`/`2` won by that
    /// player, `3` drawn); empty for plain boards.
    fn blocks(_game: &Game<Self, 2>) -> Vec<u8> {
        Vec::new()
    }
}

/// `POST /api/play` body.
#[derive(Debug, Deserialize)]
pub struct PlayRequest {
    pub game: String,
    /// Moves played so far, as cells (policy indices).
    #[serde(default)]
    pub moves: Vec<usize>,
    /// Engine to analyse the position with (`None`: board only).
    #[serde(default)]
    pub engine: Option<OpponentSpec>,
    /// Root-parallel trees for an AlphaZero engine (`sims` each, visit counts summed). With one game
    /// on the board, the trees' positions share the network batches, so more trees search more in
    /// about the same time.
    #[serde(default = "one_tree")]
    pub trees: usize,
    #[serde(default)]
    pub seed: u64,
}

const fn one_tree() -> usize {
    1
}

/// The position to draw.
#[derive(Debug, Serialize)]
pub struct Board {
    pub cells: Vec<u8>,
    pub blocks: Vec<u8>,
    /// Legal cells for the side to move (empty when the game is over).
    pub legal: Vec<usize>,
    /// Side to move: `1` (X) or `2` (O).
    pub to_move: u8,
    /// `"x"`, `"o"` or `"draw"` once the game is over.
    pub result: Option<&'static str>,
}

/// One engine's view of a position.
#[derive(Debug, Serialize)]
pub struct Analysis {
    pub engine: String,
    /// The cell the engine plays.
    pub chosen: usize,
    /// Per-move statistics, best first (empty for Random).
    pub moves: Vec<MoveStat>,
    /// Search value for the side to move in `[-1, 1]` (search engines only).
    pub value: Option<f32>,
    pub simulations: u64,
    pub millis: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct MoveStat {
    pub cell: usize,
    /// Network prior (AlphaZero only).
    pub prior: Option<f32>,
    /// Search visits (MCTS, AlphaZero).
    pub visits: u64,
    /// Mean search value of the move for the side to move in `[-1, 1]` (MCTS, AlphaZero; `0` if unvisited).
    pub q: f32,
    /// Minimax score for the side to move, in the game heuristic's units (Minimax only).
    pub score: Option<f32>,
    /// `"win"` or `"loss"` when Minimax proved the result of the move.
    pub proven: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct PlayResponse {
    pub board: Board,
    pub analysis: Option<Analysis>,
}

/// Replays `moves` from the start position.
pub fn replay<R: PlayableGame<G, P>, const G: usize, const P: usize>(moves: &[usize]) -> Result<Game<R, 2>, String> {
    moves.iter().try_fold(R::start(), |game, &cell| {
        if game.status().is_terminal() {
            return Err("the game is already over".to_string());
        }
        let mv = game
            .valid_moves()
            .find(|&mv| R::move_to_policy_index(mv) == cell)
            .ok_or_else(|| format!("cell {cell} is not a legal move"))?;
        Ok(game.apply(mv))
    })
}

pub fn board<R: PlayableGame<G, P>, const G: usize, const P: usize>(game: &Game<R, 2>) -> Board {
    let result = match game.status() {
        GameStatus::Ongoing => None,
        GameStatus::Terminal([a, b]) if a > b => Some("x"),
        GameStatus::Terminal([a, b]) if b > a => Some("o"),
        GameStatus::Terminal(_) => Some("draw"),
    };
    Board {
        cells: R::cells(game),
        blocks: R::blocks(game),
        legal: if result.is_none() { game.valid_moves().map(R::move_to_policy_index).collect() } else { Vec::new() },
        to_move: game.current_player().raw() + 1,
        result,
    }
}

/// Lets `engine` search the (ongoing) `game`. `network` must be given for an AlphaZero engine,
/// which searches with `trees` root-parallel trees.
pub fn analyse<R: PlayableGame<G, P>, const G: usize, const P: usize>(
    game: &Game<R, 2>,
    engine: &OpponentSpec,
    network: Option<&NetService<G, P>>,
    trees: usize,
    seed: u64,
) -> Result<Analysis, String> {
    let start = Instant::now();
    let mut rng = SmallRng::seed_from_u64(seed);
    let (chosen, moves, value) = match engine {
        OpponentSpec::Random => {
            let moves: Vec<_> = game.valid_moves().collect();
            (*moves.choose(&mut rng).ok_or("no legal move")?, Vec::new(), None)
        }
        OpponentSpec::Minimax { depth } => {
            let scored = root_move_scores(game, &R::heuristic(), *depth);
            let best = scored.iter().map(|&(_, s)| s).fold(f32::NEG_INFINITY, f32::max);
            // Random tie-break among the best moves, as `MinimaxPlayer` does.
            let ties: Vec<_> = scored.iter().filter(|&&(_, s)| s == best).map(|&(mv, _)| mv).collect();
            let chosen = *ties.choose(&mut rng).ok_or("no legal move")?;
            let mut moves: Vec<MoveStat> = scored
                .iter()
                .map(|&(mv, score)| MoveStat {
                    cell: R::move_to_policy_index(mv),
                    score: Some(score),
                    proven: (score.abs() >= TERMINAL_SCORE_BASE).then_some(if score > 0.0 { "win" } else { "loss" }),
                    ..MoveStat::default()
                })
                .collect();
            moves.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            (chosen, moves, None)
        }
        OpponentSpec::Mcts { sims, exploration } => {
            let config = RootParallelConfig::new(RootParallelBudget::SimulationsPerThread(*sims), seed);
            let root = RootParallelPlayer::new(game.clone(), R::SENTINEL_MOVE, Uct::new(*exploration), 1, config, |_, s| {
                PlayoutEvaluator::new(s)
            })
            .evaluate_root();
            search_stats::<R, G, P>(&root, false)
        }
        OpponentSpec::AlphaZero { sims, .. } => {
            let network = network.ok_or("no network for the AlphaZero engine")?;
            let config = AlphaZeroConfig::new(trees, *sims, seed).with_dirichlet(0.3, 0.0);
            let root = AlphaZeroPlayer::new(game.clone(), network, config).evaluate_root();
            search_stats::<R, G, P>(&root, true)
        }
    };
    Ok(Analysis {
        engine: match engine {
            OpponentSpec::AlphaZero { .. } if trees > 1 => format!("{} ×{trees} trees", engine.label()),
            _ => engine.label(),
        },
        chosen: R::move_to_policy_index(chosen),
        simulations: moves.iter().map(|m| m.visits).sum(),
        moves,
        value,
        millis: start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Move played, per-move statistics (most visited first) and root value of a finished search.
fn search_stats<R: PlayableGame<G, P>, const G: usize, const P: usize>(
    root: &[AggregatedRootChild<R::Move>],
    with_priors: bool,
) -> (R::Move, Vec<MoveStat>, Option<f32>) {
    let visits: u64 = root.iter().map(|c| c.visits).sum();
    let value = (visits > 0).then(|| (root.iter().map(|c| c.value_sum).sum::<f64>() / visits as f64) as f32);
    let mut moves: Vec<MoveStat> = root
        .iter()
        .map(|c| MoveStat {
            cell: R::move_to_policy_index(c.mv),
            prior: with_priors.then_some(c.prior),
            visits: c.visits,
            q: c.q_value() as f32,
            ..MoveStat::default()
        })
        .collect();
    moves.sort_by_key(|m| std::cmp::Reverse(m.visits));
    (most_visited(root), moves, value)
}

fn most_visited<M: Copy>(root: &[AggregatedRootChild<M>]) -> M {
    root.iter().max_by_key(|c| c.visits).expect("an ongoing game has a legal move").mv
}

/// Answers a Play request for game `R`. `network` resolves an AlphaZero engine's
/// `(checkpoint, net preset)` to an inference service.
pub fn respond<R: PlayableGame<G, P>, const G: usize, const P: usize>(
    req: &PlayRequest,
    network: impl FnOnce(&str, &str) -> Result<Arc<NetService<G, P>>, String>,
) -> Result<PlayResponse, String> {
    let game = replay::<R, G, P>(&req.moves)?;
    let analysis = match &req.engine {
        Some(engine) if game.status().is_ongoing() => {
            let engine = engine.clone().sanitized();
            let net = match &engine {
                OpponentSpec::AlphaZero { checkpoint, net, .. } => Some(network(checkpoint, net)?),
                _ => None,
            };
            Some(analyse(&game, &engine, net.as_deref(), req.trees.clamp(1, 64), req.seed)?)
        }
        _ => None,
    };
    Ok(PlayResponse { board: board(&game), analysis })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::tic_tac_toe::TicTacToe;
    use crate::games::ultimate_ttt::UltimateTicTacToe;

    #[test]
    fn replay_and_board() {
        // X takes the top row while O plays the middle row: X wins on its third move.
        let game = replay::<TicTacToe, 27, 9>(&[0, 3, 1, 4, 2]).unwrap();
        let b = board(&game);
        assert_eq!(b.cells, [1, 1, 1, 2, 2, 0, 0, 0, 0]);
        assert_eq!((b.result, b.legal.len()), (Some("x"), 0));
        assert!(replay::<TicTacToe, 27, 9>(&[0, 0]).is_err(), "occupied cell");
        assert!(replay::<TicTacToe, 27, 9>(&[0, 3, 1, 4, 2, 5]).is_err(), "game over");

        // UTTT: the first move (centre cell of the top-left sub-board) sends O to the centre sub-board.
        let game = replay::<UltimateTicTacToe, { crate::games::ultimate_ttt::INPUT_SIZE }, 81>(&[10]).unwrap();
        let b = board(&game);
        assert_eq!((b.cells[10], b.to_move, b.blocks.len()), (1, 2, 9));
        assert!(b.legal.iter().all(|&c| (3..6).contains(&(c / 9)) && (3..6).contains(&(c % 9))));
    }

    #[test]
    fn search_engines_report_heat_map_statistics() {
        let game = replay::<TicTacToe, 27, 9>(&[0, 4]).unwrap();
        let mcts = OpponentSpec::Mcts { sims: 400, exploration: 1.414 };
        let a = analyse(&game, &mcts, None, 1, 7).unwrap();
        assert_eq!(a.moves.len(), 7);
        assert_eq!(a.simulations, a.moves.iter().map(|m| m.visits).sum::<u64>());
        assert_eq!(a.chosen, a.moves[0].cell, "plays the most visited move");
        assert!(a.moves.iter().all(|m| m.prior.is_none()));
        // Minimax scores every move; O must block X's diagonal 0-4-8 at cell 8 or lose.
        let game = replay::<TicTacToe, 27, 9>(&[0, 1, 4]).unwrap();
        let a = analyse(&game, &OpponentSpec::Minimax { depth: 2 }, None, 1, 7).unwrap();
        assert_eq!((a.chosen, a.moves.len(), a.value), (8, 6, None));
        assert_eq!(a.moves[0].cell, 8);
        assert!(a.moves[1..].iter().all(|m| m.proven == Some("loss")), "every other move loses");
    }
}
