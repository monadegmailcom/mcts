# Playbook: supervising a training run

This playbook covers watching a long live-training run and stepping in. It applies to any game.

- Tune the engine first, with [engine-tuning.md](engine-tuning.md).
- Game-specific experience is in the game's playbook: [uttt.md](uttt.md),
  [rubiks-cube.md](rubiks-cube.md).
- Tool: [`experiments/lab_checkin.py`](../../experiments/lab_checkin.py), which saves a snapshot
  and prints the window table.

## 1. Before the run

1. **Name the question.** Write one sentence, e.g. "Do more sims per move make the network stronger
   than X?" A run without a question turns into fiddling with knobs.
2. **Pick the reference opponents** (`eval_opponents`):
   - **Main metric:** a fixed AlphaZero checkpoint close to the current strength, scoring 40–60%.
   - **Anchor:** a weaker checkpoint or classic engine from the Elo pool, for the Elo estimate and
     as a regression check. Only anchors count toward the Elo.
   - Opponents above about 90% are saturated and measure nothing.
3. **Fix the evaluation:**
   - Both sides at the same sims, e.g. `eval_sims: 100` and opponents `@100`.
   - Never leave `eval_sims` at `0`. It couples the evaluation to `simulations_per_move`, so every
     sims change also changes the yardstick.
   - An earlier 64-vs-100 sims mismatch understated the network by about 150 Elo.
4. **Cost of evaluation:**
   - `eval_games` 48 per opponent.
   - `eval_every` such that evaluation takes ≤ 10% of the wall time.
5. **Snapshots:**
   - Names: `<game>_<run>_iter<NNNN>`. Never reuse a name; an overwritten `exp4` file cost the
     evidence once.
   - On a pod without a volume, everything not downloaded is lost on terminate. Download a
     snapshot at every check-in.
   - `auto_save_every` in `train_control` saves `<game>_iter_NNN` snapshots on the pod itself.
6. **Change log:** `checkpoints/<game>_<run>_changes.log`, one line per change:
   `<iteration>  <change and reason>`. Log the start configuration as the first line.
7. **Infrastructure:**
   - Keep the Mac awake for scheduled check-ins: `caffeinate -i -t <s>` (a closed lid still sleeps).
   - SSH tunnel to the pod's lab: `ssh -f -N -L 18080:localhost:8080 ...`.
   - The pod's SSH port changes when it restarts.
8. **Run length:** `total_iterations` is clamped to 1..10000.

## 2. The metrics and how to read them

| Metric | What it is | How to read it |
|---|---|---|
| **Score vs reference** | (W + D/2) / N against the fixed checkpoint | The main metric. 48 games ±7 points; 1 h of evaluations (1000–1800 games) ±1.5–3 points. Compare 2 h spans; differences below 2 SE are noise |
| **Anchored Elo** | Bradley–Terry fit against the anchors in the opponents | ±34 per evaluation of 3 × 48 games. Only useful averaged over many evaluations |
| **Train loss** (`policy_kl`, `value_loss`) | KL from the search policy, and value MSE, on the SGD batches | KL has a floor (about 0.2 for UTTT): with Dirichlet ε 0.25 part of the target is unpredictable. A flat KL is not a stall |
| **Held-out loss** | KL + MSE on 2% of the games, held out whole, the last 8192 positions | Overfitting: held-out loss rises while the train loss falls. Only comparable while the self-play settings are unchanged (see below) |
| **EWA gain** | held-out(current weights) − held-out(EWA weights, window `ewa_window` steps) | What a lower learning rate would gain right away. > 0.003: step noise, lower lr. About 0: lr is not the lever |
| Self-play stats | Game length, draws, root entropy | Watch for collapse: game length jumps, draws go to 100%, entropy goes to 0 |
| Speed | sims/s, phase times | A sudden drop points to the hardware or another process, not to the learning |

**Losses are only comparable within one self-play setting.**

- The held-out positions come from the current self-play.
- When `simulations_per_move`, the noise or `value_q_weight` changes, the targets change too:
  - more sims gives sharper policies, so a higher KL;
  - more sims gives better values, so a lower MSE.
- After such a change, judge only by the score.

**Sample reuse:** how often SGD sees each position, on average.

```
reuse = train_steps × batch_size / new positions per iteration
new positions per iteration = games_per_iteration × positions per game
```

- `buffer_capacity` sets how many iterations a position stays in the buffer. It does not change
  the reuse, only how old the samples are.
- Example: UTTT with 1024 games × 53 positions = 54k positions per iteration.
  - 200 × 1024 or 800 × 256 gives reuse ≈ 3.8.
  - A 300k buffer holds about 5.5 iterations.

## 3. Check-in routine (every ~2 h)

1. **Reachable?** If the lab or the pod is down:
   - read `lab.log`, `nvidia-smi` and the provider's notices;
   - report, and restart only with the owner's OK.
2. **Snapshot and table:**

   ```bash
   python3 experiments/lab_checkin.py run5 5000 --api http://localhost:18080 --ssh root@<ip>:<port>
   ```

3. **Judge** with the decision rules (§4): compare the last 2 h with the 2 h before.
4. **At most one change.** Before it, note the last good snapshot. Then log the change.
5. **Report:** the table, the conclusion, the change with its reason, and the plan for the next
   check-in.

## 4. Decision rules

**Principles:**

- One lever at a time.
- Hold each setting for at least 2 h, or about 1000 evaluation games against the reference.
- Judge only by the score against the reference over 2 h spans.
- Use every lever, not just the learning rate. One run spent 24 h only lowering the lr and learned
  nothing about the other levers.

| Symptom | Action |
|---|---|
| Score still rising | Change nothing |
| Score flat for 3–4 h, EWA gain > 0.003 | Lower the lr by about 3× |
| Score flat, EWA gain ≈ 0 | The lr is not the lever. Next: sims/move, then sample reuse |
| Held-out rises while the train loss falls (same self-play settings) | Lower the reuse: fewer `train_steps`, or more `games_per_iteration` |
| Score drops by > 2 SE over 2 h after a change | Revert the change; if the drop persists, load the last good snapshot |
| Losses spike, NaN, game length or draws collapse | Load the last good snapshot, revert the last change, check the lr |
| Score vs reference > 65–70% | Saturated: take a stronger reference, e.g. a snapshot of this run; calibrate it as an anchor (`POST /api/elo`) |
| All levers flat over several hours | The network is at capacity: train a bigger preset from scratch |
| Last ~3 h of a run | lr down to about 1/10 for the one-time gain |

**What each lever did so far** (mostly UTTT; see [uttt.md](uttt.md)):

- **Learning rate:**
  - Lowering it gives a *one-time* gain of +15 to +40 Elo within minutes to an hour, then flat. It
    doesn't lift a plateau.
  - Raising it again for a network trained at a low lr first costs strength: 47% against its own
    starting net.
- **Sims/move:**
  - At short budgets, more games beat better targets: 32 ≈ 64 > 128 sims in 15 minutes from
    scratch.
  - Late in training more sims help, up to a point: 64 → 200 gave about +30 Elo within an hour.
    200 → 400 gave nothing at a 100-sim evaluation, and each iteration took twice as long.
- **Sample reuse and buffer:** not cleanly tested yet. Reuse was about 4× in all runs.
- **Exploration** (`dirichlet_alpha`, `dirichlet_epsilon`, `c_init`, `opening_moves`),
  **`value_q_weight`** and **FPU:** never compared cleanly. Try them last, one at a time.
- **Capacity:** the 3-block net plateaued at about 1370–1390 Elo. The 6-block net reached about
  1500 and more.

## 5. Operating rules and pitfalls

- **Load → Live and system changes:**
  - Both rebuild the trainer. They keep the weights and the replay buffer, but the AdamW moments
    and the EWA weights restart.
  - Expect a small bump, and ignore the EWA gain for the next ~`ewa_window` steps.
- **Lab restart:** the replay buffer and the chart history are lost; they live in memory only.
  - The first iterations train on a nearly empty buffer, so the reuse is very high.
  - Lower `train_steps` until the buffer holds a few iterations, or accept a short dip.
- **Same games after a start:** the evaluation right after a start replays the same seeded games.
  - Two starts from the same network give the identical result (1489 Elo twice).
  - A "jump" there is not evidence.
- **Evaluation openings:**
  - Use fixed random openings with colors swapped.
  - Without them, deterministic players replay the same few games.
  - With openings sampled from the policy, the test set changes as the policy changes.
- **Final comparison:** for the final verdict, play a head-to-head match of ≥ 200 games (±3.5
  points). Don't rely on opponents that are already saturated.
- **Strength vs games:** Elo grows roughly with the log of training time. Doubling the run time
  gives a constant gain, not a doubled one.
- **Bugs first:** when nothing learns from the start, check for bugs before tuning.
  - Every parameter must change after a step.
  - The value head must not start saturated: zero-init the last layers.
  - There must be enough distinct games: the value target is per game, so thousands of games are
    needed, not thousands of positions.
- **Crash on a healthy-looking GPU:** with `CUDA error: unknown error`, look for a hardware notice
  from the provider first. On Nov 25 it was a host fault.
- **Measure before you claim** a bottleneck, a jump or a trend. State the uncertainty (SE) with
  every number.

## 6. End of the run

1. Find the best snapshot and confirm it with a head-to-head of ≥ 200 games against the reference.
2. **Report:**
   - the question and the answer;
   - the best snapshot and its score;
   - what each lever showed;
   - the change log.
3. Add the findings to the game's playbook. If a new network becomes the reference, calibrate it
   into the Elo pool.
4. Stop the scheduled check-ins, the tunnels and `caffeinate`.
