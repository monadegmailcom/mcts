# Running the Browser Lab on RunPod

The [Dockerfile](Dockerfile) builds an image for NVIDIA GPUs:

- `train_alphazero` built with `--features cuda,tch`: both backends run on CUDA, with libtorch
  2.9.1 (CUDA 12.8 build) as the default.
- An SSH server that accepts your RunPod SSH key.
- The Browser Lab, started in the background on port 8080.
- The build environment (Rust, the CUDA compiler, libtorch with headers) and the build in `/src`,
  so code changes are built on the pod ([Developing on the pod](#developing-on-the-pod)).

Lab state lives in `/workspace/mcts`: `checkpoints/`, `lab_experiments.json` and the Elo pools.
On RunPod that is the pod's volume, so it survives restarts and new image versions.

## Building the image

GitHub Actions builds the image ([workflow](.github/workflows/docker.yml)) on an x86_64 runner
and pushes it to `ghcr.io/monadegmailcom/mcts-runpod`. It is tagged `latest` and with the short
commit hash. Builds run only on demand, from `main` or another branch:

```bash
gh workflow run docker.yml
```

```bash
gh workflow run docker.yml --ref <branch>
```

```bash
gh run watch
```

A build takes about 20 minutes.

**One-time step:** after the first build, make the package public, so RunPod can pull it without
registry credentials. On GitHub, open your profile → **Packages** → `mcts-runpod` → **Package
settings** → **Change visibility** → **Public**.

## RunPod template (one-time)

Under **Templates → New Template**:

| Field | Value |
|---|---|
| Container image | `ghcr.io/monadegmailcom/mcts-runpod:latest` |
| Container disk | 30 GB (the unpacked image is about 12 GB, mostly CUDA libraries) |
| Volume disk / mount path | 20 GB at `/workspace`, or a Network Volume for Spot pods |
| Expose HTTP ports | `8080` |
| Expose TCP ports | `22` |

Put your public SSH key under **Settings → SSH Public Keys**. RunPod passes it to the container
as `PUBLIC_KEY`.

When deploying, pick a GPU whose host supports **CUDA 12.8** or newer (the CUDA version filter
in the deploy dialog). Older drivers cannot run the CUDA 12.8 libraries.

Optional environment variables:

| Variable | Effect |
|---|---|
| `MCTS_AUTOSTART=0` | Don't start the lab; start it yourself over SSH with `mcts-lab` |
| `MCTS_BACKEND=candle` | Default backend (`tch` otherwise); also switchable in the lab's System panel |
| `MCTS_DATA_DIR` | Where the lab state lives (`/workspace/mcts`) |

## Using the pod

- **Lab:** open `https://<pod-id>-8080.proxy.runpod.net` (**Connect → HTTP Service** in RunPod).
  - **This URL is public and the lab has no login:** anyone who has the URL can start and stop
    training and overwrite checkpoints.
  - If that is not acceptable, leave port 8080 out of the template and open the lab through an
    SSH tunnel at <http://localhost:8080> instead:

    ```bash
    ssh -L 8080:localhost:8080 root@<ip> -p <port>
    ```

- **SSH:** `ssh root@<ip> -p <port>`, with the IP and port from **Connect → SSH over exposed TCP**.
  On the pod:

  ```bash
  mcts-lab                          # restart the lab (e.g. after a crash), extra args go to train_alphazero
  ```

  ```bash
  tail -f /workspace/mcts/lab.log   # lab output
  ```

  ```bash
  nvidia-smi                        # GPU load
  ```

- **Checkpoints to and from the Mac:**

  ```bash
  scp -P <port> checkpoints/uttt_deep90_lr2e-4_block_deep.safetensors root@<ip>:/workspace/mcts/checkpoints/
  ```

  ```bash
  scp -P <port> 'root@<ip>:/workspace/mcts/checkpoints/*.safetensors' checkpoints/
  ```

  The Elo pool is `checkpoints/uttt_elo_pool.json`; copy it along so ratings stay on the same scale.
  Its network anchors (`uttt_exp4_block`, `uttt_live200`) must be in `checkpoints/` too.

## Workflow: tune with experiments, then train

1. Start the pod and open the lab, through the proxy URL or the SSH tunnel.
2. **Tune with experiments.** In the Experiments tab, queue short runs that differ in one
   setting at a time: inference batch, inference workers, games per iteration, replay buffer
   size, train steps. Each run reports self-play speed (simulations/s) and strength per training
   minute.
3. **Train.** Copy the winning values into the live knobs and the System panel, then start live
   training on the Training tab.

Experiments can also be queued without the browser, with the lab's HTTP API (here through the SSH
tunnel). Start with the spec of an earlier run (`GET /api/experiments` lists them) and change
fields:

```bash
curl -s localhost:8080/api/experiments -d '{"action":"add","spec":{...}}'
```

```bash
curl -s localhost:8080/api/experiments -d '{"action":"run_queue"}'
```

## Developing on the pod

A fix doesn't need a new image: sync the source from your checkout into `/src` on the pod and
build it there. The build reuses the image's compiled dependencies and CUDA kernels, so only the
`mcts` crate recompiles.

```bash
rsync -az --delete -e 'ssh -p <port>' Cargo.toml Cargo.lock build.rs src examples root@<ip>:/src/
```

On the pod:

```bash
mcts-build    # builds and installs train_alphazero; a running lab keeps the old binary
```

```bash
mcts-lab      # restarts the lab with the new binary (stops running training and experiments)
```

Commit and push from your checkout as usual. The next image build picks the changes up, and
`/src` on a running pod is replaced only when the pod restarts with a new image.

To work with git on the pod instead, clone the repository and forward your SSH agent, so the pod
can push with your key without storing it (`ssh -A root@<ip> -p <port>`, then
`git clone git@github.com:monadegmailcom/mcts.git /workspace/mcts-src`), then build it with
`mcts-build /workspace/mcts-src`.

## Notes

- **CPU, not GPU, limits self-play:** each self-play game searches on its own CPU thread, and the
  network calls of all games are batched on the GPU. A pod with 15 vCPUs and an RTX A4000 kept the
  GPU at 20–40% load and ran as fast as an Apple M4 (about 260k simulations/s for UTTT with the
  `block` network). For more speed, pick a pod by its vCPU count first.

- **Multiple GPUs:** self-play and evaluation games spread their network calls over the GPUs, one
  model replica per *inference worker*, round-robin. Several workers per GPU overlap the host-side
  work of their batches. Live training takes the worker count from the System
  panel. Each experiment has its own *Inference Workers* setting, which defaults to the System
  panel, so experiments run on the same setup as training. Training steps (SGD) run on GPU 0.
- **GPU architecture:** the candle backend's CUDA kernels are compiled for compute capability 8.0
  (Ampere) and run on every newer GPU. Older GPUs (V100, T4) need a rebuild with
  `CUDA_COMPUTE_CAP` set in the Dockerfile. libtorch supports them as is.
