//! Microbenchmark comparing Root-to-Leaf MCTS memory footprint and throughput.
//!
//! Run with: `cargo run --release --example bench_mcts`

use std::time::Instant;

use rand::prelude::{SeedableRng, SmallRng};

use mcts::game::PlayerIndex;
use mcts::games::tic_tac_toe::{EMPTY_STATE, NO_MOVE, TicTacToe, TttGame, TttHeuristic, TttMove};
use mcts::games::twist_puzzle::{
    NO_MOVE as TWIST_NO_MOVE, TwistGame, TwistMove, TwistState,
};
use mcts::matchup::run_multi_match;
use mcts::mcts::{
    FnEvaluator, MctsConfig, MctsPayload, MctsPlayer, PlayoutEvaluator, Puct, Uct,
    assign_masked_softmax_priors,
};
use mcts::minimax::{MinimaxPlayer, TreeMinimaxPayload, TreeMinimaxPlayer};
use mcts::player::Player;
use mcts::tree::Node;

fn main() {
    println!("=== Rust Root-to-Leaf MCTS Memory Footprint ===");
    println!(
        "sizeof(Node<TttMove, MctsPayload>):   {} bytes",
        std::mem::size_of::<Node<TttMove, MctsPayload>>()
    );
    println!(
        "sizeof(Node<u16, MctsPayload>):       {} bytes",
        std::mem::size_of::<Node<u16, MctsPayload>>()
    );
    println!(
        "sizeof(Node<TttMove, TreeMinimax>):   {} bytes",
        std::mem::size_of::<Node<TttMove, TreeMinimaxPayload>>()
    );
    println!(
        "sizeof(MctsPayload):                  {} bytes",
        std::mem::size_of::<MctsPayload>()
    );
    println!(
        "sizeof(SmallRng [rand crate]):        {} bytes",
        std::mem::size_of::<SmallRng>()
    );
    println!();

    // 1. Playout MCTS vs Playout MCTS on Tic-Tac-Toe (500 full games, 400 sims/move)
    let games = 500usize;
    let sims_per_move = 400usize;
    let mut total_moves = 0usize;
    let mut draws = 0usize;
    let start = Instant::now();

    for round in 0..games {
        let starter = if round % 2 == 0 {
            PlayerIndex::PLAYER1
        } else {
            PlayerIndex::PLAYER2
        };
        let mut game = TttGame::new(starter, EMPTY_STATE);
        let seed_base = (round as u64) * 17 + 1;

        let mut p1 = MctsPlayer::new(
            game,
            NO_MOVE,
            Uct::new(0.8),
            PlayoutEvaluator::new(seed_base),
            MctsConfig::new(sims_per_move, seed_base + 1).with_arena_capacity(4096),
        );
        let mut p2 = MctsPlayer::new(
            game,
            NO_MOVE,
            Uct::new(0.8),
            PlayoutEvaluator::new(seed_base + 2),
            MctsConfig::new(sims_per_move, seed_base + 3).with_arena_capacity(4096),
        );

        let mut p1_turn = true;
        while game.status().is_ongoing() {
            let mv = if p1_turn {
                let m = p1.choose_move();
                p2.apply_external_move(m);
                m
            } else {
                let m = p2.choose_move();
                p1.apply_external_move(m);
                m
            };
            game = game.apply(mv);
            total_moves += 1;
            p1_turn = !p1_turn;
        }
        if game.status().payoffs() == Some([0.0, 0.0]) {
            draws += 1;
        }
    }
    let elapsed_uct = start.elapsed();
    let total_sims_uct = total_moves * sims_per_move;
    println!("=== Benchmark 1: Tic-Tac-Toe UCT Playout MCTS ({games} games, {sims_per_move} sims/move) ===");
    println!(
        "Total moves: {total_moves}, Total simulations: {total_sims_uct}, Draws: {draws}/{games}"
    );
    println!(
        "Elapsed: {:.3} ms ({:.2}M simulations/sec)",
        elapsed_uct.as_secs_f64() * 1000.0,
        (total_sims_uct as f64) / elapsed_uct.as_secs_f64() / 1e6
    );
    println!();

    // 2. Pure AlphaZero PUCT tree-overhead benchmark (zero-cost mock NN evaluator + softmax)
    let mut total_moves_puct = 0usize;
    let start_puct = Instant::now();
    for round in 0..games {
        let mut game = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
        let seed_base = (round as u64) * 31 + 7;

        let make_eval = || {
            FnEvaluator(|_g: &TttGame, children: &mut [Node<TttMove, MctsPayload>]| {
                // Realistic center > corner > edge prior logits + masked softmax
                assign_masked_softmax_priors(children, |mv| match mv {
                    4 => 1.2,
                    0 | 2 | 6 | 8 => 0.6,
                    _ => 0.1,
                });
                [0.0, 0.0]
            })
        };

        let mut p1 = MctsPlayer::new(
            game,
            NO_MOVE,
            Puct::default(),
            make_eval(),
            MctsConfig::new(sims_per_move, seed_base).with_arena_capacity(4096),
        );
        let mut p2 = MctsPlayer::new(
            game,
            NO_MOVE,
            Puct::default(),
            make_eval(),
            MctsConfig::new(sims_per_move, seed_base + 1).with_arena_capacity(4096),
        );

        let mut p1_turn = true;
        while game.status().is_ongoing() {
            let mv = if p1_turn {
                let m = p1.choose_move();
                p2.apply_external_move(m);
                m
            } else {
                let m = p2.choose_move();
                p1.apply_external_move(m);
                m
            };
            game = game.apply(mv);
            total_moves_puct += 1;
            p1_turn = !p1_turn;
        }
    }
    let elapsed_puct = start_puct.elapsed();
    let total_sims_puct = total_moves_puct * sims_per_move;
    println!("=== Benchmark 2: Pure AlphaZero PUCT Tree Overhead ({games} games, {sims_per_move} sims/move) ===");
    println!(
        "Total moves: {total_moves_puct}, Total simulations: {total_sims_puct}"
    );
    println!(
        "Elapsed: {:.3} ms ({:.2}M PUCT tree traversals + softmax/sec)",
        elapsed_puct.as_secs_f64() * 1000.0,
        (total_sims_puct as f64) / elapsed_puct.as_secs_f64() / 1e6
    );
    println!();

    // 3. Single-player (`N = 1`) TwistPuzzle solving benchmark
    let puzzles = 100usize;
    let mut solved = 0usize;
    let mut rng = SmallRng::seed_from_u64(12345);
    let start_puzzle = Instant::now();
    for i in 0..puzzles {
        let state = TwistState::scrambled(3, 5, &mut rng);
        let mut game = TwistGame::new(PlayerIndex::P0, state);
        let mut solver = MctsPlayer::new(
            game,
            TWIST_NO_MOVE,
            Uct::new(1.4),
            PlayoutEvaluator::<TwistMove>::new(i as u64 + 100),
            MctsConfig::new(1_000, i as u64 + 200).with_arena_capacity(16_384),
        );
        while game.status().is_ongoing() {
            let mv = solver.choose_move();
            game = game.apply(mv);
        }
        if game.state().is_solved() {
            solved += 1;
        }
    }
    let elapsed_puzzle = start_puzzle.elapsed();
    println!("=== Benchmark 3: N=1 TwistPuzzle (3-move scrambles, 1000 sims/step) ===");
    println!(
        "Solved: {solved}/{puzzles} in {:.3} ms",
        elapsed_puzzle.as_secs_f64() * 1000.0
    );
    println!();

    // 4. Multithreaded `MultiMatch`: Minimax(depth=5) vs TreeMinimax(depth=5) over 500 games
    let ttt_root = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
    let f_mm = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
        Box::new(MinimaxPlayer::new(*g, TttHeuristic, 5, seed))
    };
    let f_tree_mm = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
        Box::new(TreeMinimaxPlayer::new(*g, NO_MOVE, TttHeuristic, 5, seed, 8192))
    };

    let start_mm = Instant::now();
    let summary = run_multi_match(&ttt_root, [&f_mm, &f_tree_mm], 500, 8, 2026);
    let elapsed_mm = start_mm.elapsed();
    println!("=== Benchmark 4: MultiMatch (8 threads, 500 games, Minimax d=5 vs TreeMinimax d=5) ===");
    println!(
        "Wins: {:?}, Draws: {}, Wall time: {:.3} ms",
        summary.wins,
        summary.draws,
        elapsed_mm.as_secs_f64() * 1000.0
    );
    println!(
        "Mean choose_move() time — Minimax: {:.2} us, TreeMinimax: {:.2} us",
        summary.choose_time_us[0].mean(),
        summary.choose_time_us[1].mean()
    );
}
