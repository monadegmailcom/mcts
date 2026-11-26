"""Engine throughput sweep as Browser Lab experiments (see docs/playbooks/engine-tuning.md).

Queues one short experiment per configuration. All start from the same checkpoint and differ only
in the engine settings: backend, inference batch and workers, games per iteration, self-play
threads, SGD split. Experiments have their own network, replay buffer and inference service, so
the live network and its training charts stay untouched. They run only while live training is
paused.

For each configuration it prints:
- self-play and SGD time per iteration;
- games per training minute (the target);
- self-play simulations/s and the experiment's final losses, a NaN check.

The first iteration is warm-up and is left out of the averages. The experiment records stay in the
Experiments tab; their weight files are deleted.

Usage:
    python3 experiments/engine_sweep.py --init uttt_run4_iter4703.safetensors --net block_deep \
        [--api URL] [--game uttt] [--iters 4] [--gpus 1] [--set simulations_per_move=200 --set ...] [CONFIGS.json]

The base hyperparameters are the lab's current knobs of the active game, overridden by --set.
CONFIGS.json is a list of {"name": ..., "system": {"infer_batch", "infer_workers", "backend"},
"knobs": {...}}; without it the standard sweep runs.
"""
import argparse
import json
import math
import time
import urllib.request


def standard_sweep(hp, system, gpus, cores):
    games, steps, batch = hp["games_per_iteration"], hp["train_steps"], hp["batch_size"]
    sweep = [("baseline", {}, {})]
    sweep += [(f"{w} workers per GPU", {"infer_workers": w * gpus}, {}) for w in (1, 3)]
    sweep += [(f"infer batch {b}", {"infer_batch": b}, {}) for b in (256, 1024)]
    # Twice the games with twice the SGD steps keeps the sample reuse.
    sweep += [(f"{2 * games} games, {2 * steps} steps", {}, {"games_per_iteration": 2 * games, "train_steps": 2 * steps})]
    sweep += [(f"SGD {2 * steps} x {batch // 2}", {}, {"train_steps": 2 * steps, "batch_size": batch // 2})]
    sweep += [(f"{cores} self-play threads", {}, {"self_play_threads": cores})]
    return [{"name": n, "system": {**system, **s}, "knobs": k} for n, s, k in sweep]


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def summarize(run, sims):
    """Per-iteration means over the points after the warm-up iteration."""
    pts = sorted(run["points"], key=lambda p: p["iteration"])
    pairs = [(a, b) for a, b in zip(pts, pts[1:]) if a["iteration"] >= 1]
    if not pairs:
        return None
    n = sum(b["iteration"] - a["iteration"] for a, b in pairs)
    train = sum(b["train_secs"] - a["train_secs"] for a, b in pairs)
    games = sum(b["games"] - a["games"] for a, b in pairs)
    self_play = sum((b["positions"] - a["positions"]) * sims / b["sims_per_sec"] for a, b in pairs if b["sims_per_sec"])
    last = pts[-1]
    losses = (last["value_loss"], last["policy_kl"])
    return {
        "self_play": self_play / n,
        "sgd": (train - self_play) / n,
        "games_per_min": 60 * games / train,
        "sims_per_sec": sum(b["sims_per_sec"] for _, b in pairs) / len(pairs),
        "nan": any(x is None or (isinstance(x, float) and math.isnan(x)) for x in losses),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="?")
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--init", required=True, help="checkpoint to start every configuration from")
    ap.add_argument("--net", required=True, help="network preset of that checkpoint, e.g. block_deep")
    ap.add_argument("--game", help="switch the lab to this game first (default: the active game)")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE", help="override a knob (repeatable)")
    args = ap.parse_args()

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        return json.load(urllib.request.urlopen(urllib.request.Request(args.api + path, data), timeout=60))

    if args.game:
        call("/api/train_control", {"action": "set_game", "game": args.game})
    st = call("/api/state")
    if not st["knobs"]["paused"]:
        raise SystemExit("Live training is running; experiments would wait until it is paused.")
    game = st["active_game"]
    hp = {k: v for k, v in st["knobs"].items() if k not in ("paused", "eval_every")}
    hp.update(dict((k, parse_value(v)) for k, v in (s.split("=", 1) for s in args.set)))
    system = {"backend": st["system"]["backend"], "infer_batch": st["system"]["infer_batch"],
              "infer_workers": st["system"]["infer_workers"]}
    configs = json.load(open(args.configs)) if args.configs else standard_sweep(hp, system, args.gpus, st["system"]["cpu_cores"])

    ids = []
    for cfg in configs:
        sys_ = {**system, **cfg.get("system", {})}
        spec = {"name": f"engine: {cfg['name']}", "game": game, "net": args.net, "backend": sys_["backend"],
                "device": "gpu", "hp": {**hp, **cfg.get("knobs", {})}, "budget_minutes": 1e6,
                "max_iterations": args.iters, "eval_every": 1, "eval_sims": 1, "eval_games": 2,
                "infer_batch": sys_["infer_batch"], "infer_workers": sys_["infer_workers"],
                "opponents": [{"kind": "random"}], "init_checkpoint": args.init}
        run = max(call("/api/experiments", {"action": "add", "spec": spec})["store"]["runs"], key=lambda r: r["id"])
        if run["spec"]["name"] != spec["name"] or run["status"]["state"] != "Queued":
            raise SystemExit(f"The lab did not queue '{spec['name']}' (spec rejected).")
        ids.append((run["id"], cfg, sys_))
    call("/api/experiments", {"action": "run_queue"})

    print(f"{'config':32s} {'self-play':>9s} {'SGD':>7s} {'games/min':>9s} {'sims/s':>7s}  system")
    for run_id, cfg, sys_ in ids:
        while True:
            run = next(r for r in call("/api/experiments")["store"]["runs"] if r["id"] == run_id)
            if run["status"]["state"] != "Queued" and run["status"]["state"] != "Running":
                break
            time.sleep(10)
        call("/api/models", {"action": "delete", "name": f"{game}_exp{run_id}_{args.net}.safetensors", "game": game})
        s = summarize(run, run["spec"]["hp"]["simulations_per_move"])
        if run["status"]["state"] != "Done" or s is None:
            print(f"{cfg['name']:32s} {run['status']['state']}: {run['status'].get('message', '')}", flush=True)
            continue
        print(f"{cfg['name']:32s} {s['self_play']:8.1f}s {s['sgd']:6.1f}s {s['games_per_min']:9.0f} "
              f"{s['sims_per_sec'] / 1000:6.0f}k  {sys_['backend']} batch {sys_['infer_batch']} x "
              f"{sys_['infer_workers']} workers" + ("  NaN LOSSES" if s["nan"] else ""), flush=True)


if __name__ == "__main__":
    main()
