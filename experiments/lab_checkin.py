"""Check-in on a live training run in the Browser Lab (see docs/playbooks/training-supervision.md).

Saves a snapshot of the live network (and copies it to the local checkpoints folder when the lab
runs on a remote host), then summarizes the training since a start iteration in windows of wall
time: iterations, self-play speed, anchored Elo, score and W-D-L against each evaluation
opponent, train losses, held-out loss and EWA gain. Ends with the run's change log.

Usage:
    python3 experiments/lab_checkin.py RUN START_ITER [--api URL] [--game uttt] [--window-hours 1]
                                       [--ssh root@HOST:PORT] [--no-snapshot]

The change log is checkpoints/<game>_<RUN>_changes.log, one line per change:
"<iteration>  <change and reason>".
"""
import argparse
import json
import os
import subprocess
import urllib.request

CHECKPOINTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("start", type=int)
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--game", default="uttt")
    ap.add_argument("--window-hours", type=float, default=1.0)
    ap.add_argument("--ssh", help="root@HOST:PORT of a remote lab; its snapshot is copied here")
    ap.add_argument("--no-snapshot", action="store_true")
    args = ap.parse_args()

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        return json.load(urllib.request.urlopen(urllib.request.Request(args.api + path, data), timeout=30))

    st = call("/api/state")
    k, it = st["knobs"], st["current_iteration"]
    print(f"phase: {st['phase']}")
    print(f"iteration {it} / {st['total_iterations']} | net {st['live_net']} | {st['device_name']}")
    print("knobs: " + ", ".join(f"{x}={k[x]}" for x in (
        "learning_rate", "train_steps", "batch_size", "games_per_iteration", "simulations_per_move",
        "buffer_capacity", "ewa_window", "eval_sims", "eval_every")))

    if not args.no_snapshot:
        name = f"{args.game}_{args.run}_iter{it:04d}.safetensors"
        call("/api/models", {"action": "save", "name": name, "game": args.game})
        if args.ssh:
            host, port = args.ssh.rsplit(":", 1)
            subprocess.run(["scp", "-q", "-o", "IdentitiesOnly=yes", "-P", port,
                            f"{host}:/workspace/mcts/checkpoints/{name}", os.path.join(CHECKPOINTS, name)], check=True)
        print(f"snapshot: {name}")

    h = [x for x in st["history"] if x["iteration"] > args.start]
    if h:
        secs = [(x["self_play_ms"] + x["train_ms"] + x["eval_ms"]) / 1000 for x in h]
        print(f"\n{len(h)} iterations in {sum(secs) / 3600:.1f} h ({mean(secs):.1f} s each); "
              f"remaining {(st['total_iterations'] - it) * mean(secs[-50:]) / 3600:.1f} h")

        windows, t, cur = [], 0.0, []
        for x, s in zip(h, secs):
            cur.append(x)
            t += s
            if t >= args.window_hours * 3600:
                windows.append(cur)
                cur, t = [], 0.0
        if cur:
            windows.append(cur)

        labels = [a["label"] for a in next((x["arena"] for x in reversed(h) if x["arena"]), [])]
        print("\niterations    sims/s  Elo (evals)  " + "".join(f"vs {a[:27]:27s}  " for a in labels)
              + "train KL/MSE  held-out  EWA gain")
        for w in windows:
            ev = [x for x in w if x["arena"]]
            elo = [x["elo"]["elo"] for x in ev if x.get("elo")]
            elo_txt = f"{mean(elo):6.0f} ({len(elo):3d})" if elo else f"{'-':>6s}      "
            cols = []
            for label in labels:
                wdl = [sum(r["wdl"][i] for x in ev for r in x["arena"] if r["label"] == label) for i in range(3)]
                n = sum(wdl)
                cols.append(f"{100 * (wdl[0] + wdl[1] / 2) / n:5.1f}% {wdl[0]:4d}-{wdl[1]:4d}-{wdl[2]:4d}" if n else "-")
            ho = [x["held_out"] for x in w if x["held_out"]]
            held = mean([o["kl"] + o["value"] for o in ho])
            ewa = mean([o["ewa_kl"] + o["ewa_value"] for o in ho])
            print(f"{w[0]['iteration']:5d}-{w[-1]['iteration']:<5d} "
                  f"{mean([x['simulations_per_sec'] for x in w]) / 1000:6.0f}k  {elo_txt}  "
                  + "".join(f"{c:30s}  " for c in cols)
                  + f"{mean([x['policy_kl'] for x in w]):.3f}/{mean([x['value_loss'] for x in w]):.3f}   "
                  f"{held:.4f}   {held - ewa:+.4f}")

    log = os.path.join(CHECKPOINTS, f"{args.game}_{args.run}_changes.log")
    if os.path.exists(log):
        print("\nchanges:\n" + open(log).read().rstrip())


if __name__ == "__main__":
    main()
