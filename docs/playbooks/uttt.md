# Playbook: Ultimate Tic-Tac-Toe

This playbook collects experience and recommendations for UTTT, from Nov 14 to Nov 25, 2026.

- How to supervise a run in general: [training-supervision.md](training-supervision.md).
- Engine tuning and hardware numbers: [engine-tuning.md](engine-tuning.md).

## 1. The game as a training problem

- **Moves:** 81, with about 53 positions per self-play game, so 1024 games give about 54k positions.
- **Symmetries:** the 8 dihedral symmetries of the 9 × 9 grid are exact (unit-tested). The replay
  buffer uses them for augmentation.
- **Draws:** many. Between nets of similar strength 50–62% of the games are drawn; against deep27h
  it was 1138 of 1824 games. Draws narrow the score range, so a few points of score are a real
  difference.
- **Input:** 7 planes: own and opponent stones, legal moves, own and opponent sub-boards won,
  drawn sub-boards, side to move.
- **Network initialization:** the last policy and value layers start at zero. A value head
  saturated at init (predicting about +1 everywhere) never learned.
- **Evaluation:** 24 fixed random 2-ply openings, each played with both colors (seed 4242), argmax,
  no noise.
- **Classic opponents are saturated** after about 5 minutes of training: MCTS-200 with random
  rollouts and Minimax depths 1–3. After that, evaluate only against AlphaZero checkpoints.

## 2. Network presets

| Preset | Architecture | Verdict |
|---|---|---|
| `block` | Shared encoder per 3 × 3 sub-board (local 64, hidden 256), 3 residual blocks, about 590k parameters | Fast; plateaus at about 1370–1390 Elo |
| `block_deep` | The same with 6 residual blocks, about 1M parameters | Learns as fast per minute as `block` early on; reached about 1500 and more after hours; costs only 14% sims/s on CUDA. Plateaued at deep27h's strength through runs 3–5 |
| `block_deeper` | The same with 10 residual blocks | **The default since Nov 27**, for run6. Not measured yet |
| `resnet` | 4 × 64 convolutions | 6.6× slower than `block`, not better per game |
| `mlp` | Flat MLP | Learned poorly |

- There is no growth from `block` to `block_deep`; a deeper net trains from scratch.
- **Any size:** the knobs `net_blocks` and `net_width` resize the default (`block_deeper`, 10 × 256)
  for the next Reset or experiment, e.g. 8 × 384. See
  [training-supervision.md](training-supervision.md) §4, "Capacity".

## 3. Hyperparameters

| Knob | Code default | Proven on the A4000 pod | Note |
|---|---|---|---|
| `games_per_iteration` | 1024 | 1024 | 256 leaves the batches half empty; 2048 didn't help |
| `simulations_per_move` | 64 | **200** | 64 is best from scratch for short budgets; 200 helped late in training; 400 didn't |
| `train_steps` × `batch_size` | 800 × 256 | 200 × 1024 | Same samples, 3.7× faster SGD; reuse ≈ 3.8. Run5: reuse 1.9–7.6 all equal at the plateau |
| `learning_rate` | 2e-3 | 2e-3 from scratch, 6e-4 to continue, 2e-4 for the final hours | See §5 |
| `weight_decay` | 1e-4 | 1e-4 | Never varied |
| `net_blocks` / `net_width` | 0 / 0 (= 10 × 256) | 6 × 256 (`block_deep`) through run5 | Act on Reset only |
| `buffer_capacity` | 300k | 300k | About 5.5 iterations; never varied |
| `opening_moves` | 8 | 8 | Moves sampled by visit count, for variety |
| `dirichlet_alpha` / `_epsilon` | 0.3 / 0.25 | the same | Never varied |
| `c_init` / `c_base` | 1.25 / 19652 | the same | Never varied |
| `value_q_weight` | 0.5 | 0.5 | The 0 vs 0.5 comparison ran on a buggy build; no clean evidence |
| FPU | absolute 0 | the same | A relative variant exists; never compared |
| `eval_games` / `eval_sims` / `eval_every` | 48 / 0 / 1 | 48 / **100** / 10 | Always set `eval_sims` |
| System: `infer_batch` / `infer_workers` | 512 / 2 per GPU | the same | See engine-tuning |

## 4. Networks and their strength

Elo pool `checkpoints/uttt_elo_pool.json`, 48 games per pair (SE 26–29). Calibrated Nov 21;
deep27h and run5_iter0670 added on Nov 27 (the old pairings were kept, only the new ones played).
The refit moved every anchor down by 30–65 Elo, so compare Elo values only within one pool
version. The Nov 21 pool is kept as `checkpoints/uttt_elo_pool_2026-11-21.json`.

| Anchor | Elo Nov 27 | Elo Nov 21 |
|---|---|---|
| Random | 0 | 0 |
| Minimax d1 | 527 | 556 |
| MCTS-200, random rollouts | 571 | 603 |
| Minimax d2 | 629 | 663 |
| Minimax d3 | 780 | 819 |
| `uttt_exp1_block` @100 | 1150 | 1197 |
| `uttt_exp4_block` @100 | 1176 | 1221 |
| `uttt_live200` @100 | 1307 | 1371 |
| `uttt_deep27h_lr2e-4_block_deep` @100 | 1444 ± 28 | – |
| `uttt_run5_iter0670` @100 | 1443 ± 28 | – |

- deep27h vs run5_iter0670 head-to-head: 6-31-11 (45%), within the noise of the run5 evaluations
  (49–50% over thousands of games).
- Elo above about 1450 is no longer extrapolated: the pool has two anchors there.

| Checkpoint | Preset | How it was trained | Strength |
|---|---|---|---|
| `uttt_exp1_block` | block | 15 min, 256 games × 200 steps, lr 2e-3, 64 sims | 1197 |
| `uttt_exp4_block` | block | 15 min, 1024 games × 800 steps, 64 sims. The file was overwritten once: the current one is the 64-sim run from Nov 21 | 1221 |
| `uttt_live200` | block | Live run, 200 iterations, lr 2e-3, 64 sims; the 3-block plateau | 1371 |
| `uttt_deep90_lr2e-4_block_deep` | block_deep | 60 min at lr 2e-3, 100 sims, then 30 min at lr 2e-4 | about 1335; 43% vs live200 |
| `uttt_deep24h_block_deep` | block_deep | 24 h on the pod, lr 2e-3, 64 sims, 1024 × 800 | 67% vs live200, about 1485; plateau after about 15 h |
| `uttt_deep27h_lr2e-4_block_deep` | block_deep | deep24h + 168 min at lr 2e-4 | 57.3% vs deep24h, about 72% vs live200, about 1530?; **the reference since run3** |
| `uttt_run3_iterNNNN` | block_deep | Run3, 64 sims, lr 2e-3 → 6e-4 → 2e-4; snapshots every 2 h | 45–50% vs deep27h; best about iter 2000 (49.7%) |
| `uttt_run4_iter4703` | block_deep | Run3 final + 300 iterations at 200 sims, lr 6e-4 | **about 50.6% vs deep27h (252-714-234), the strongest net so far** |
| `uttt_run4_iter4794` | block_deep | iter 4703 + 90 iterations at 400 sims | about 47–48% vs deep27h |
| `uttt_run5_iter0670` | block_deep | iter 4703 + 670 iterations (11 h) at 200 sims, 2048 games, lr 6e-4, reuse 1.9–7.6 | about 49–50% vs deep27h, as strong as iter 4703; **the second reference for run6** |

- The pod's live network and log from the end of run4 are in `checkpoints/pod_backup_2026-11-25/`.
- The run4 change log is `checkpoints/uttt_run4_changes.log`.

## 5. Run history and what it taught

1. **Nov 14 – "nothing learns".** There were five causes, none of them a hyperparameter:
   - too few distinct games (about 3k);
   - a saturated value head at init;
   - a flat MLP;
   - a misleading evaluation: deterministic games from the empty board, 12 games, sims silently
     capped;
   - a candle Metal crash.
2. **Nov 15 – the candle LayerNorm had no backward pass.** Everything in front of it never
   learned, so all earlier architecture comparisons were invalid. Since then the default is
   libtorch with the `block` net.
3. **Nov 20 – 15-minute experiments from scratch:**
   - **Games per iteration:** 1024 games × 800 steps learned 1.7× faster per minute than 256 × 200,
     with equal final strength (52.6% head-to-head).
   - **Sims per move:** 32 ≈ 64 > 128 (head-to-head: 64 vs 32 49%, 128 vs 64 46%). At short budgets
     the number of games counts more than the quality of the targets.
4. **Nov 21 – the 3-block plateau.** Neither the levers nor capacity lifted it much:
   - live200 rose to about 65% vs exp4 by iteration 140, then stayed flat;
   - lr 2e-4: +15–40 Elo;
   - 128 sims: +15–25 Elo;
   - `block_deep` for 90 min stayed 50 Elo below live200.

   Verdict: capacity, with hours of training needed.
5. **Nov 22–23 – deep24h on the pod.** It climbed against live200 from 43% to 67% over about
   15 h, then plateaued. 168 min at lr 2e-4 added about +50 Elo, and that became deep27h.
6. **Nov 23–24 – run3**, continuing from deep27h: 64 sims, lr 2e-3 → 6e-4 → 2e-4, 24 h.
   - The losses kept falling and the EWA gain went to about 0.
   - The score vs deep27h stayed at 45–50%.

   Verdict: **the lr is not the bottleneck.** Raising it to 2e-3 first cost strength; lowering it
   only recovered that.
7. **Nov 24–25 – run4**, from the run3 final net, lr 6e-4:

   | Sims/move | Score vs deep27h | Notes |
   |---|---|---|
   | 200 | 46.4% → about 50.6% within the first hour | Then flat for 3.6 h |
   | 400 | 47–48% | 80 s per iteration instead of 43 s |

   - **Sims help up to about 200.**
   - A run3 net at 64 sims is not a fair start for 200 sims: part of the first hour's gain is the
     adaptation to the new targets.
   - Run4 ended in a RunPod hardware fault at iteration 4796. Its last setting, 200 sims with 100
     train steps from 4703, never ran.
8. **Nov 26–27 – run5**, from `uttt_run4_iter4703` on one A40: 200 sims, 2048 games, lr 6e-4.

   | Setting | Hours | Score vs deep27h |
   |---|---|---|
   | 400 × 1024 SGD (reuse 3.8) | 4.5 | 49.7% ± 0.9 |
   | 800 × 1024 (reuse 7.6) | 3.8 | 47.6% ± 1.0 |
   | 200 × 1024 (reuse 1.9) | 1.9 | 49.7% ± 1.4 |
   | 200 × 1024, `value_q_weight` 0 | 0.8 | 50.8% ± 2.2, too short |

   - vs live200: flat at 65–68%; the Elo estimate flat at about 1480–1500.
   - Held-out and train loss flat, no overfitting.

   Verdict: **`block_deep` is at capacity.** Three runs (about 60 h) ended at deep27h's strength
   with lr, sims and reuse varied. Stopped at iteration 670 for run6 with a bigger network.

## 6. Where to go next

The open question: how do we get clearly above deep27h, to 55% or more?

1. **Continue run4** from `uttt_run4_iter4703`.
   - Settings: 200 sims, lr 6e-4, 1024 games, 200 × 1024 SGD.
   - Opponents: deep27h @100 and live200 @100, `eval_sims` 100, `eval_every` 10.
   - First confirm that the 50.6% holds over about 2 h.
2. **Sample reuse.** The reuse is about 3.8, not "20×" as estimated during run4.
   - Halving `train_steps` to 100 tests the *lower* end, about 1.9.
   - The opposite direction (400 × 1024, about 7.6) is just as open.
   - Try one of them, 2 h each.
3. **Exploration and value target**, one at a time:
   - `value_q_weight` 0.5 vs 0 vs the `linear_to_end` schedule;
   - Dirichlet α 0.3 vs 1.0. The AlphaZero rule of thumb is α ≈ 10 / legal moves. Most UTTT moves
     are confined to one sub-board, so at most 9 legal moves, which points to a larger α. Not
     measured.
   - `opening_moves` 8 vs 16.
4. **Capacity.** If the levers stay flat, build a bigger preset (8–10 blocks or wider) and train it
   from scratch on the pod.
   - deep24h needed about 15 h to plateau.
   - Plan at least 24 h, and use the 15-minute experiments only for the speed check.
5. **Measurement.**
   - Done Nov 27: deep27h and run5_iter0670 are anchors in the Elo pool (§4).
   - Once a net scores ≥ 65% vs deep27h, make it the new reference.

**Ideas, not tried:**

- Playout-cap randomization (KataGo): many fast moves with few sims, a few moves with many sims
  for the policy targets. It would combine "more games" with "better targets". Proposed on Nov 20
  and declined.
- More sims during evaluation, to see whether 400-sim training pays off with deeper search.
