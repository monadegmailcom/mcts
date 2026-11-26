# Playbook: Rubik's cube

This playbook collects experience and recommendations for the 3×3×3 cube, from Nov 21 to Nov 26,
2026.

- How to supervise a run in general: [training-supervision.md](training-supervision.md). The cube
  has no opponents and no Elo; §5 says what replaces them.
- Engine tuning and hardware numbers: [engine-tuning.md](engine-tuning.md).

## 1. The game as a training problem

- **Moves:** 12 quarter turns. Half turns count as two moves.
- **One player:** a cube is worth `1` when solved and `0` when the move budget runs out (not `-1`).
  - With the per-move discount γ, the network learns `γ^(moves to solved)`, i.e. how far a cube is
    from solved.
  - The search backs values up with the same discount, so a solve one move later is worth γ times
    as much.
- **Input:** 54 stickers × 6 colors, one-hot (324 values). Neither the last move nor the moves
  left are part of the input.
- **Symmetries:** none are used. The cube's 48 symmetries would need a color relabelling and are
  not implemented.
- **Scrambles:** round `n` is `n` random quarter turns. `X X'` and `X X X` are never drawn, and
  scrambles that cancel out are redrawn. Up to about round 10 the shortest solution is usually the
  scramble reversed, so it is `n` moves long.
- **Every solve passes the easy cubes:** solving a 12-move scramble visits cubes 11, 10, …, 1 moves
  from solved. That is why self-play can skip the rounds it has mastered without forgetting them.

### The round game

The cube is trained as a game of rounds of rising difficulty (`RoundGame` in
`src/games/rubiks_cube.rs`):

- Round `n` is a cube scrambled with `n` quarter turns, to be solved within
  `moves_per_turn × n + extra_moves` moves, at most `max_moves` (default `2n + 10`, at most 100).
- Solving it leads to round `n + 1`, up to `max_rounds` (26: every cube is at most 26 quarter turns
  from solved). Failing ends the game.
- A game scores `n + 1/r` for its last solved round `n`, solved in `r` moves.
- Self-play games start at `first_round`. The skipped rounds count as solved, so a game that fails
  its first round scores `first_round − 1`. **Evaluation always starts at round 1**, so its scores
  stay comparable when `first_round` changes.
- Each round is one episode: searched, labelled and learned on its own. The game only decides
  *which* cubes are played, moving on to deeper scrambles as the network improves.

**Why a budget that grows with the round:** with a fixed 100-move budget, failed rounds made up
about 95% of the positions: every failure adds 100 positions labelled 0. With `2n + 10` a failure
costs 12–60 moves.

## 2. Network

| Preset | Architecture | Verdict |
|---|---|---|
| ResMlp (default) | 1024 wide, 6 residual blocks, heads 256; about 13M parameters | Chosen to be able to learn a full solver when fully trained (DeepCubeA used a similar MLP). Solved rounds 1–9 every time after about 25 minutes on the M4 |

- Convolutions don't fit: neighbouring stickers lie on different faces.
- Smaller or larger nets have not been compared.

## 3. Settings and hyperparameters

The cube's own settings (`RubiksSettings`, the lab's *Rubik's Cube* card, `train_rubiks` options)
sit next to the shared knobs.

| Setting | Default | Used Nov 25 | Note |
|---|---|---|---|
| `discount` γ | 0.9 | 0.9 | 20 moves from solved is worth 0.12, 25 moves 0.07. Raise to 0.95 once rounds ≥ 15 matter (§6) |
| `first_round` | 1 | 1 | Self-play only. See §6 for when to raise it |
| `max_rounds` | 26 | 26 | |
| `moves_per_turn` / `extra_moves` / `max_moves` | 2 / 10 / 100 | the same | Never the limit so far: solved rounds take about `n` moves |
| `games_per_iteration` | 512 | 512 | |
| `simulations_per_move` | 200 | 200 | Never varied |
| `opening_moves` | 0 | 0 | No temperature moves: a wrong move only costs moves |
| `dirichlet_alpha` / `_epsilon` | 0.5 / 0.25 | the same | Never varied |
| FPU | absolute 0 | the same | An untried move is assumed to lead nowhere |
| `value_q_weight` | 0 | 0 | |
| `train_steps` × `batch_size` | 500 × 512 | the same | Reuse ≈ 6 at 87 positions per game; 25 in the first iteration (§5) |
| `learning_rate` | 1e-3 | 1e-3 | Never varied |
| `buffer_capacity` | 500k | 500k | About 11 iterations at 44k positions each |
| `eval_games` / `eval_sims` / `eval_every` | 128 / 400 / 1 | the same | Evaluation took 45–49% of each iteration; see §6 |

## 4. Networks and their scores

| Checkpoint | How it was trained | Evaluation score (128 games, 400 sims) |
|---|---|---|
| `rubiks_alphazero` in `mcts-rubiks/checkpoints/` | Live run Nov 25, 30 iterations from scratch, defaults (§3), 77 min on the M4 | **10.82 ± 0.09**: rounds 1–9 solved every time; from the start, round 10 is solved in 88%, round 11 in 56%, round 12 in 23%, round 13 in 5% of the games |

Save it under a run name (`rubiks_run1_iter0030`) before continuing: the live checkpoint is
rewritten after every iteration.

## 5. Run history and what it taught

1. **Nov 21 – design.**
   - Failure pays 0 instead of −1, so that with the discount an unsolved cube is simply worth
     nothing and the value is `γ^(moves to solved)`.
   - The fixed 100-move budget made 95% of the positions failures; the budget now grows with the
     round.
   - The discount is a cube setting, not a shared hyperparameter: two-player games don't have one.
2. **Nov 21 – first checks** (scratch lab, small settings): a one-iteration network solved a
   5-turn scramble in 9 moves; a short experiment went from score 2.5 to 5.1 in 80 s.
3. **Nov 25 – first live run**, 30 iterations from scratch with the defaults, M4 (tch MPS):

   | Iteration | Self-play score | Evaluation score | Rounds always solved | Policy loss |
   |---|---|---|---|---|
   | 1 | 2.62 | 5.62 ± 0.08 | 1–4 | 2.21 |
   | 10 | 9.03 | 9.91 ± 0.11 | 1–9 | 1.44 |
   | 20 | 9.92 | 10.43 ± 0.12 | 1–9 | 1.30 |
   | 30 | 10.12 | 10.82 ± 0.09 | 1–9 | 1.24 |

   - **Fast start, slow tail:** about +1 per iteration at first, only +0.4 over iterations 20–30.
     Not a plateau yet: the policy loss still falls and the policy KL per iteration is steady at
     0.15.
   - **Solutions are short:** rounds up to 13 are solved in about `n` moves. The network reverses
     the scramble; when it fails, it has lost the way, not run out of moves.
   - **Where self-play's positions went** (iteration 30, 44k positions, 87 per game):
     - rounds 1–8, solved almost always: 40%;
     - the failed last round, at its full budget of about 32 moves: 37%;
     - rounds 9 and up, where the learning happens: 23%.

     This led to `first_round` (Nov 26).
   - **Time per iteration:** about 155 s: self-play 65 s (110–140k sims/s), training 15 s,
     evaluation 60–95 s. Iteration 10 took twice as long; the cause is unknown (another process?).
   - **Sample reuse** falls as the games get longer: 25 in iteration 1 (20 positions per game),
     about 6 from iteration 10 on.

## 6. Where to go next

1. **Continue the run** from `rubiks_alphazero` (iteration 30), saved as `rubiks_run1_iter0030`:
   - `first_round` 7. Rule of thumb: three rounds below the first round the evaluation solves in
     fewer than 95% of the games that reach it (iteration 30: round 10, 89%). Raise it as that
     round moves up.
   - `eval_every` 5, or `eval_games` 64 (± about 0.15 instead of 0.11). Evaluation now costs as
     much as self-play.
   - Expect the self-play score to jump when `first_round` changes (skipped rounds count as
     solved). Judge by the evaluation score.
2. **Learning rate:** once 10 iterations add less than about 0.2 to the evaluation score, drop it
   from 1e-3 to 3e-4. As with UTTT, expect a one-time gain, not a new slope.
3. **Discount:** at round 15 and beyond, raise γ from 0.9 to 0.95, so that cubes 20 and 25 moves
   from solved are worth 0.36 and 0.28 instead of 0.12 and 0.07. Changing γ changes every value
   target; expect a dip of a few iterations until the buffer has turned over.
4. **Sims per move:** 200 was never varied. With search depth mattering more for deep scrambles,
   compare 200 vs 400 once `first_round` has removed the easy rounds.
5. **Production:** a long run on the RunPod CUDA pod. Run the engine-tuning playbook for the cube
   first: its throughput on CUDA has not been measured.

**Ideas, not tried:**

- An adaptive `first_round` that follows the evaluation's solve rates on its own, for unattended
  runs.
- A smaller `extra_moves`: failed rounds are 37% of the positions, and solved rounds never needed
  the extra room so far.
- The cube's 48 symmetries (with color relabelling) as data augmentation.
