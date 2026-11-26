"""Engine throughput sweep against a running Browser Lab (see docs/playbooks/engine-tuning.md).

Each configuration sets the System panel (backend, inference batch and workers) and knobs, trains
the live network for a few iterations with a minimal evaluation, and prints the phase times and
the self-play throughput, averaged over the iterations after the first (warm-up).

It trains the lab's live network and leaves the last configuration in place: run it on a scratch
lab or before loading the network you want to train.

Usage:
    python3 experiments/engine_sweep.py [--api URL] [--game uttt] [--iters 3] [--gpus 1] [CONFIGS.json]

CONFIGS.json is a list of {"name": ..., "system": {...}, "knobs": {...}}; without it the standard
sweep below runs. Values not set by a configuration keep the base values.
"""
import argparse
import json
import time
import urllib.request


def standard_sweep(gpus):
    base = {"games_per_iteration": 1024, "train_steps": 200, "batch_size": 1024}
    system = {"infer_batch": 512, "infer_workers": 2 * gpus}
    sweep = [("baseline", {}, {})]
    sweep += [(f"{w} workers per GPU", {"infer_workers": w * gpus}, {}) for w in (1, 3)]
    sweep += [(f"infer batch {b}", {"infer_batch": b}, {}) for b in (256, 1024)]
    sweep += [("2048 games", {}, {"games_per_iteration": 2048})]
    sweep += [(f"SGD {s} x {b}", {}, {"train_steps": s, "batch_size": b}) for s, b in ((800, 256), (400, 512))]
    return [{"name": n, "system": {**system, **s}, "knobs": {**base, **k}} for n, s, k in sweep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="?")
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--game", default="uttt")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        return json.load(urllib.request.urlopen(urllib.request.Request(args.api + path, data), timeout=60))

    def wait(done):
        while True:
            st = call("/api/state")
            if done(st):
                return st
            time.sleep(5)

    configs = json.load(open(args.configs)) if args.configs else standard_sweep(args.gpus)
    print(f"{'config':28s} {'self-play':>9s} {'SGD':>7s} {'sims/s':>8s} {'batch':>11s}  system")
    for cfg in configs:
        call("/api/system", cfg.get("system", {}))
        wait(lambda st: not st["system"]["pending"])
        start = call("/api/state")["current_iteration"]
        body = {"action": "start", "game": args.game, "total_iterations": start + args.iters,
                "eval_every": 1000, "eval_games": 2, "eval_opponents": [{"kind": "random"}]}
        call("/api/train_control", {**body, **cfg.get("knobs", {})})
        st = wait(lambda st: st["current_iteration"] >= start + args.iters and st["knobs"]["paused"])
        hs = [h for h in st["history"] if h["iteration"] > start + 1]

        def avg(key):
            return sum(h[key] for h in hs) / len(hs)

        sys_ = st["system"]
        print(f"{cfg['name']:28s} {avg('self_play_ms') / 1000:8.1f}s {avg('train_ms') / 1000:6.1f}s "
              f"{avg('simulations_per_sec') / 1000:7.0f}k {avg('batch_size_mean'):5.0f}/{hs[-1]['max_batch_size']:<5d}"
              f"  {sys_['backend']} batch {sys_['infer_batch']} x {sys_['infer_workers']} workers", flush=True)


if __name__ == "__main__":
    main()
