//! Microbenchmark comparing Root-to-Leaf MCTS memory footprint and throughput.
//!
//! Run with: `cargo run --release --example bench_mcts`

use std::time::Instant;

use rand::prelude::{SeedableRng, SmallRng};

use mcts::game::PlayerIndex;
use mcts::games::rubiks_cube::{
    NO_MOVE as RUBIKS_NO_MOVE, RubiksGame, RubiksMove, RubiksState,
};
use mcts::games::tic_tac_toe::{EMPTY_STATE, NO_MOVE, TicTacToe, TttGame, TttHeuristic, TttMove};
use mcts::games::twist_puzzle::{
    NO_MOVE as TWIST_NO_MOVE, TwistGame, TwistMove, TwistState,
};
use mcts::games::ultimate_ttt::{
    EMPTY_STATE as UTTT_EMPTY, NO_MOVE as UTTT_NO_MOVE, UltimateTicTacToe, UtttGame, UtttHeuristic,
};
use mcts::matchup::run_multi_match;
use mcts::mcts::{
    FnEvaluator, MctsConfig, MctsPayload, MctsPlayer, PlayoutEvaluator, Puct, Uct,
    assign_masked_softmax_priors,
};
use mcts::minimax::{MinimaxBudget, MinimaxPlayer, TreeMinimaxPayload, TreeMinimaxPlayer};
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

    // 4. Multithreaded `MultiMatch`: Minimax(depth=5) vs TreeMinimax(MaxEvaluations(600)) over 500 games
    let ttt_root = TttGame::new(PlayerIndex::PLAYER1, EMPTY_STATE);
    let f_mm = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
        Box::new(MinimaxPlayer::new(*g, TttHeuristic, 5, seed))
    };
    let f_tree_mm = |g: &TttGame, seed: u64| -> Box<dyn Player<TicTacToe, 2>> {
        Box::new(TreeMinimaxPlayer::new(
            *g,
            NO_MOVE,
            TttHeuristic,
            MinimaxBudget::MaxEvaluations(600),
            seed,
            8192,
        ))
    };

    let start_mm = Instant::now();
    let summary = run_multi_match(&ttt_root, [&f_mm, &f_tree_mm], 500, 8, 2026);
    let elapsed_mm = start_mm.elapsed();
    println!("=== Benchmark 4: MultiMatch (8 threads, 500 games, Minimax d=5 vs TreeMinimax MaxEvaluations(600)) ===");
    println!(
        "Wins: {:?}, Draws: {}, Wall time: {:.3} ms",
        summary.wins,
        summary.draws,
        elapsed_mm.as_secs_f64() * 1000.0
    );
    println!(
        "Mean choose_move() time — Minimax(d=5): {:.2} us, TreeMinimax(600 evals): {:.2} us",
        summary.choose_time_us[0].mean(),
        summary.choose_time_us[1].mean()
    );
    println!();

    // 5. Ultimate Tic-Tac-Toe (`N = 2`, 81 branching):
    //    (a) Minimax(d=4) vs TreeMinimax(FixedDepth(4)) — measuring move-ordering pruning speedup
    //    (b) MctsPlayer(1600 sims) vs TreeMinimax(MaxEvaluations(1600)) — equal budget head-to-head
    let uttt_root = UtttGame::new(PlayerIndex::PLAYER1, UTTT_EMPTY);
    let uttt_h = UtttHeuristic::default();

    let f_uttt_mm4 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
        Box::new(MinimaxPlayer::new(*g, uttt_h, 4, seed))
    };
    let f_uttt_tree4 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
        Box::new(TreeMinimaxPlayer::new(
            *g,
            UTTT_NO_MOVE,
            uttt_h,
            MinimaxBudget::FixedDepth(4),
            seed,
            65_536,
        ))
    };
    let uttt_d4 = run_multi_match(&uttt_root, [&f_uttt_mm4, &f_uttt_tree4], 64, 8, 2026);
    println!("=== Benchmark 5a: Ultimate TTT FixedDepth=4 (64 games, 8 threads) ===");
    println!(
        "Wins [Minimax, TreeMinimax]: {:?}, Draws: {}",
        uttt_d4.wins, uttt_d4.draws
    );
    println!(
        "Mean choose_move() time — Minimax(d=4): {:.2} us vs TreeMinimax(d=4): {:.2} us ({:.2}x speedup!)",
        uttt_d4.choose_time_us[0].mean(),
        uttt_d4.choose_time_us[1].mean(),
        uttt_d4.choose_time_us[0].mean() / uttt_d4.choose_time_us[1].mean()
    );
    println!();

    let f_uttt_mcts = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
        Box::new(MctsPlayer::new(
            *g,
            UTTT_NO_MOVE,
            Uct::new(0.5),
            PlayoutEvaluator::new(seed ^ 0xDEAD_BEEF),
            MctsConfig::new(1_600, seed).with_arena_capacity(32_768),
        ))
    };
    let f_uttt_tree_1600 = |g: &UtttGame, seed: u64| -> Box<dyn Player<UltimateTicTacToe, 2>> {
        Box::new(TreeMinimaxPlayer::new(
            *g,
            UTTT_NO_MOVE,
            uttt_h,
            MinimaxBudget::MaxEvaluations(1_600),
            seed,
            32_768,
        ))
    };
    let uttt_budget = run_multi_match(&uttt_root, [&f_uttt_mcts, &f_uttt_tree_1600], 64, 8, 909);
    println!("=== Benchmark 5b: Ultimate TTT Equal-Budget Duel (1600 sims MCTS vs 1600 evals TreeMinimax, 64 games) ===");
    println!(
        "Wins [UCT-MCTS(1600), TreeMinimax(1600)]: {:?}, Draws: {}",
        uttt_budget.wins, uttt_budget.draws
    );
    println!(
        "Mean choose_move() time — UCT-MCTS(1600): {:.2} us vs TreeMinimax(1600): {:.2} us",
        uttt_budget.choose_time_us[0].mean(),
        uttt_budget.choose_time_us[1].mean()
    );
    println!();

    // 6. Single-Player (`N = 1`) 3x3x3 Rubik's Cube (4-move scrambles, pure UCT MCTS before NN)
    let cube_trials = 50usize;
    let mut cube_solved = 0usize;
    let mut cube_rng = SmallRng::seed_from_u64(333);
    let start_cube = Instant::now();
    for i in 0..cube_trials {
        let state = RubiksState::scrambled(4, 5, &mut cube_rng);
        let mut game = RubiksGame::new(PlayerIndex::P0, state);
        let mut solver = MctsPlayer::new(
            game,
            RUBIKS_NO_MOVE,
            Uct::new(1.4),
            PlayoutEvaluator::<RubiksMove>::new(i as u64 + 500),
            MctsConfig::new(4_000, i as u64 + 900).with_arena_capacity(65_536),
        );
        while game.status().is_ongoing() {
            let mv = solver.choose_move();
            game = game.apply(mv);
        }
        if game.state().is_solved() {
            cube_solved += 1;
        }
    }
    let elapsed_cube = start_cube.elapsed();
    println!("=== Benchmark 6: N=1 3x3x3 Rubik's Cube (4-move scrambles, 4000 UCT sims/step, no NN yet) ===");
    println!(
        "Solved: {cube_solved}/{cube_trials} in {:.3} ms",
        elapsed_cube.as_secs_f64() * 1000.0
    );
}
