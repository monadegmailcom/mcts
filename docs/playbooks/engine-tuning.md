# Playbook: tuning the engine on new hardware

Run this playbook on every new machine before training: a new pod type, GPU or Mac. Also run it
after changes to the backend, the network preset, or the self-play and inference pipeline.

The goal is the most training per wall-clock minute. Do it before any training-parameter
experiment. Otherwise those experiments compare the engine instead of the hyperparameters.

Tools:

- [`experiments/engine_sweep.py`](../../experiments/engine_sweep.py): sweeps the settings against a
  running lab.
- `examples/bench_nn.rs`: measures network latency per batch size.

The measured results are in [Reference results](#5-reference-results).

## 1. Speed settings and learning settings

| Setting | Where | Changes what the network learns? |
|---|---|---|
| `backend` (`tch` / `candle`), `device` | System panel, `--backend`, `MCTS_BACKEND` | No. Only after a backend bug, see [§6](#6-pitfalls) |
| `infer_batch` (positions per network call) | System panel | No |
| `infer_workers` (model replicas, spread over the GPUs) | System panel | No |
| `self_play_threads`, `eval_threads` (`0` = all cores but two) | Knobs | No |
| `OMP_NUM_THREADS` | Environment (`mcts-lab` sets `1`) | No |
| `eval_every`, `eval_games` | Knobs | No, but they cost time |
| `games_per_iteration` | Knobs | Yes: games in flight, and how fresh the data is |
| `train_steps` × `batch_size` | Knobs | Yes: samples per iteration, and the number of updates |
| Network preset | Experiments tab, or the checkpoint | Yes |

**Rules:**

- Tune the speed-only settings freely, by throughput.
- For the coupled settings, keep the learning-relevant quantities fixed while you change how they
  are split:
  - samples per iteration (`train_steps × batch_size`);
  - games per iteration.
- `batch_size` × `train_steps` at the same product is *not* the same learning: fewer, larger steps
  are fewer updates. 200 × 1024 trained UTTT well at lr 6e-4 to 2e-3. Confirm with a short strength
  comparison when you change it a lot.

## 2. How the pipeline works

You need this to read the numbers.

- **Closed loop.**
  - Each self-play game has exactly one network request in flight; there is no virtual loss.
  - Games in flight = `min(games_per_iteration, infer_batch × infer_workers × 2)`.
  - Throughput = games in flight / round-trip time of one request.
  - The queues never "fill up". A game's next request exists only after its last answer has come
    back.
- **Inference worker.** Each worker is a serial loop, with its own model replica:
  1. collect a batch from the queue;
  2. copy it to the GPU;
  3. run the forward pass (many small kernels);
  4. read the results back;
  5. wake the games.

  The GPU idles during the host-side steps. A second worker per GPU overlaps them.
- **Fixed cost per network call.** One call costs about the same for 16 and for 512 positions
  (MPS: 1.2–1.6 ms). So batches should be as large as the games in flight allow.
- **Batch tail.** At the end of each self-play phase, games finish and the batches shrink.
- **SGD** runs on GPU 0 in one thread and is bound by kernel launches. On the A4000 a step of the
  `block_deep` net costs about 20 ms for batch sizes 256 to 1024. So few large steps are faster
  than many small ones.
- **Evaluation** has few games in flight (2 × `eval_games` per opponent), so GPU use is low.
  Amortize it with `eval_every`.

**Consequence:** 30–40% GPU and about 25% CPU utilization are normal for this design.

- `nvidia-smi` "utilization" is the share of time a kernel runs, not how much of the GPU's compute
  it uses.
- Don't chase 100%. Chase simulations/s and seconds per iteration.
- Run4 on the A4000, by phase:

  | Phase | Share of wall time | CPU | GPU |
  |---|---|---|---|
  | Self-play | 80% | 27% | 39% |
  | SGD | 9% | 7% | 23% |
  | Evaluation | 8% | 18% | 30% |

## 3. What to measure

`/api/state` → `history`, per iteration:

| Field | Meaning |
|---|---|
| `simulations_per_sec` | Self-play throughput |
| `self_play_ms`, `train_ms`, `eval_ms` | Phase times |
| `batch_size_mean` / `_std` / `_max` vs `max_batch_size` | How full the network calls are |
| `inference_us_mean` | Time per network call |

`/api/state` → `utilization` gives CPU and GPU per phase.

- **Target:** seconds per iteration at fixed learning settings, `self_play + train + eval / eval_every`.
- **Secondary:** simulations/s.
- `sims/s` depends only weakly on `simulations_per_move`. Still, sweep at the sims you will train
  with.

## 4. Procedure

### Step 0: check the machine (5 min)

- **GPU:** `nvidia-smi` shows the GPU, and the driver supports CUDA ≥ 12.8.
- **CPU count:**
  - On pods, `nproc` shows the *host's* cores (128 on the A4000 pod). The usable count is the
    cgroup quota: `cat /sys/fs/cgroup/cpu.max`, quota / period. On the A4000 pod that was 15, while
    RunPod advertised 18.
  - The lab's `system.cpu_cores` shows what the lab uses.
- **Lab log:** the first lines of `lab.log` name the device. libtorch must report CUDA.
  - GPU at 2–4% while the process uses about 1600% CPU means libtorch runs on the CPU.
    `libtorch_cuda` was not linked; `build.rs` forces it.
  - All cores busy and self-play 6–7× slow means libtorch's OpenMP pool spins one thread per host
    core. `OMP_NUM_THREADS=1` fixes it; `mcts-lab` sets it.
- **Quiet machine:** nothing else runs during measurements (no `cargo build`, no second lab).

### Step 1: network micro-benchmark (5 min, optional)

```bash
cargo run --release --features tch,cuda --example bench_nn nets=block,block_deep backends=tch,candle
```

- On the Mac, use `--features tch,metal`.
- It shows the latency per batch size (the fixed overhead) and the time per training step, per
  backend.
- Expect libtorch to beat candle: 2–4× for the block nets, about 17× for the ResNet on MPS.

### Step 2: sweep in the lab (30–60 min)

- Start a scratch lab: port 8090 on the Mac, or the pod's lab before you load the network to
  train. Then run:

  ```bash
  python3 experiments/engine_sweep.py --api http://localhost:8090 --gpus 1
  ```

- Through an SSH tunnel to a pod: `--api http://localhost:18080`. Python is not installed on the
  pod.
- For each configuration, the script:
  - trains 3 iterations against a random opponent with 2 games and no evaluation;
  - drops the first iteration as warm-up;
  - prints self-play and SGD time, sims/s and the mean batch.
- **It trains the live network and leaves the last configuration set.**
- Change one variable at a time; keep the winner before you move on. The standard sweep runs these
  steps in order. Pass a JSON file for other configurations.
  1. Backend: `tch` vs `candle`.
  2. Inference workers per GPU: 1, 2, 3.
  3. Inference batch: 256, 512, 1024. Keep games in flight ≥ 2 × batch × workers.
  4. Games per iteration: 1024 vs 2048. More games also make an iteration longer and the data
     older.
  5. SGD split at a constant product: 200 × 1024, 400 × 512, 800 × 256.
  6. `eval_every` and `eval_games`: evaluation should take ≤ 10% of the wall time.
- Self-play threads: leave them on auto. More threads helped nowhere so far.

### Step 3: apply and record

- Set the winners in the System panel and the knobs. Add the hardware's row to
  [Reference results](#5-reference-results).
- **When to change system settings:** a change rebuilds the trainer and the inference service.
  - It keeps the weights but restarts the AdamW moments and the EWA weights.
  - Make the change before a run, not in the middle of an experiment.

## 5. Reference results

All numbers are for UTTT at 64 sims/move with 1024 games per iteration unless stated.

| Hardware | Setup | sims/s |
|---|---|---|
| Mac M4 (10 cores), tch on MPS | `block`, batch 512, 1 worker | 245–265k |
| Mac M4, candle on Metal | `block`, after the LayerNorm fix | 62k |
| RunPod RTX A4000, 15 vCPU (cgroup), EPYC 7702 | tch `block`, 1 worker | 259k |
| same | tch `block_deep`, 1 worker | 223k |
| same | candle `block` / `block_deep` | 194k / 144k |
| same | tch `block_deep`, 2 workers, SGD 200 × 1024 | 274–291k (300k at 200 sims) |

**A4000 sweep** (Nov 23, `block_deep`, SGD 800 × 256 unless stated):

| Setup | sims/s | Mean batch |
|---|---|---|
| Batch 256 | 121k | 232 |
| Batch 512 | 197k | 459 |
| Batch 1024 | 208k | 669 |
| Batch 512, 2048 games | 190k | |
| Batch 1024, 2048 games | 169k | |
| 1 worker | 152k | 433 |
| **2 workers** | **217k** | 359 |
| 4 workers | 162k | 181 |
| 8 workers (old default) | 82k | 94 |

**SGD on the A4000.** About the same 205k samples per iteration, split differently:

| Split | Time per iteration |
|---|---|
| 800 × 256 | 15.6 s |
| 400 × 512 | 7.9 s |
| 200 × 1024 | 4.2 s |

**Iteration time on the A4000**, before and after tuning (Nov 23):

| Phase | Before | After |
|---|---|---|
| Self-play | 17–27 s | 12 s |
| SGD | 15.6 s | 4.3 s |
| Evaluation | 25–32 s every iteration | 31 s every 10th iteration |
| **Iteration** | **57 s** | **19 s** |

**Mac M4, network call cost** (block net, tch on MPS):

| Positions per call | 16 | 256 | 512 | 1024 |
|---|---|---|---|---|
| ms per call | 1.16 | 1.19 | 1.56 | 2.39 |
| positions/s | 14k | 215k | 328k | 429k |

**Lessons:**

- **Batch size:**
  - Inference batch 64 gave 22–28k sims/s on the Mac; 256 gave 97k. The default is now 512.
  - At 256 games per iteration the batches averaged about 180, because of the tail. That is why
    the UTTT default is 1024 games.
- **Pod vs Mac:** the A4000 pod was only as fast as the M4 with one worker.
  - The pod is not CPU-limited (`top` showed about 240%).
  - The bottleneck is the serial inference loop, so more vCPUs alone won't help.
- **Network preset:** `block_deep` costs only 14% sims/s on CUDA, against 1.7× the inference
  latency on MPS.

## 6. Pitfalls

**What did not help:**

- 2048 games per iteration.
- Inference batch 1024 (a tie on CUDA, slower on MPS).
- 3 or more workers per GPU.
- Two batches in flight per worker on MPS: libtorch serializes MPS calls behind a global lock.
- More self-play threads.
- Updating the EWA weights only every 8 SGD steps.

**Bugs that looked like slow hardware or a network that doesn't learn:**

- **Linker:** it dropped `libtorch_cuda`, so libtorch ran on the CPU.
- **OpenMP threads:** see Step 0.
- **candle on Metal:**
  - It segfaulted with two networks at once; fixed by a global lock.
  - It is slower than candle on the CPU for convolutions.
- **candle LayerNorm:** it had no backward pass, so every layer in front of it never learned.
  - Since then, a test checks that every parameter changes after a step.
  - Another test checks that both backends follow the same loss curve from identical weights.
  - After a backend change, check that both still hold.

**Measurement mistakes:**

- **Warm-up:** the first iteration after a start or a system change is slower. Discard it.
- **Utilization is not a bottleneck diagnosis.** "CPU-bound" was claimed once without a
  measurement, and it was wrong. Measure phase times and batch sizes.
- **Hardware faults:** a `CUDA error: unknown error` on a GPU that looks healthy can be a host
  fault. Check the provider's notices before debugging code.

## 7. Ideas not tried yet for more throughput

| Idea | Expected effect |
|---|---|
| Virtual loss, several requests per game | More games in flight from the same games |
| CUDA graphs for the forward pass | Fewer kernel launches per call |
| Pinned host memory | Faster copies to and from the GPU |
| SGD overlapped with the next self-play phase | Hides the SGD time |
| Timing instrumentation of the inference workers | Shows where a batch's 2.3 ms go (about 0.45 ms is GPU compute) |
