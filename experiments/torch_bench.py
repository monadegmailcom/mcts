"""PyTorch (= libtorch kernels) counterpart of `examples/bench_nn.rs`.

Same networks, batch sizes and measurements as the candle benchmark:
- inference: numpy batch -> device -> forward (no_grad) -> numpy (what self-play pays per call)
- training: one forward + backward + AdamW step

Run: python experiments/torch_bench.py [devices=cpu,mps,cuda] [nets=resnet6,resnet4,sub,mlp] [secs=1.5]
     add `bn=1` to use BatchNorm (as in beat-it) instead of GroupNorm(1) (as in the candle net).
     add `compile=1` to also try torch.compile on CUDA.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PLANES, BOARD, P, N = 7, 9, 81, 2
G = PLANES * BOARD * BOARD

args = dict(a.split("=", 1) for a in sys.argv[1:])
USE_BN = args.get("bn", "0") == "1"


def norm2d(c):
    return nn.BatchNorm2d(c) if USE_BN else nn.GroupNorm(1, c)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(c, c, 3, padding=1, bias=False), norm2d(c)
        self.c2, self.n2 = nn.Conv2d(c, c, 3, padding=1, bias=False), norm2d(c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        return F.relu(x + self.n2(self.c2(h)))


class ResNet(nn.Module):
    def __init__(self, ch, blocks, pch, vch, fc):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(PLANES, ch, 3, padding=1, bias=False), norm2d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.pconv, self.pnorm = nn.Conv2d(ch, pch, 1), norm2d(pch)
        self.pfc1, self.pfc2 = nn.Linear(pch * 81, fc), nn.Linear(fc, P)
        self.vconv, self.vnorm = nn.Conv2d(ch, vch, 1), norm2d(vch)
        self.vfc1, self.vfc2 = nn.Linear(vch * 81, fc), nn.Linear(fc, N)

    def forward(self, x):
        x = self.blocks(self.inp(x.view(-1, PLANES, BOARD, BOARD)))
        p = F.relu(self.pnorm(self.pconv(x))).flatten(1)
        v = F.relu(self.vnorm(self.vconv(x))).flatten(1)
        return self.pfc2(F.relu(self.pfc1(p))), torch.tanh(self.vfc2(F.relu(self.vfc1(v))))


class MlpBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f1, self.n1, self.f2, self.n2 = nn.Linear(d, d), nn.LayerNorm(d), nn.Linear(d, d), nn.LayerNorm(d)

    def forward(self, x):
        return F.relu(x + self.n2(self.f2(F.relu(self.n1(self.f1(x))))))


class ResMlp(nn.Module):
    def __init__(self, hidden=128, blocks=2, head=96):
        super().__init__()
        self.inp, self.n = nn.Linear(G, hidden), nn.LayerNorm(hidden)
        self.blocks = nn.Sequential(*[MlpBlock(hidden) for _ in range(blocks)])
        self.p = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, P))
        self.v = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, N), nn.Tanh())

    def forward(self, x):
        h = self.blocks(F.relu(self.n(self.inp(x))))
        return self.p(h), self.v(h)


class SubBoard(nn.Module):
    def __init__(self, local=64, hidden=256, blocks=3, head=64):
        super().__init__()
        self.l1, self.ln1 = nn.Linear(PLANES * 9, local), nn.LayerNorm(local)
        self.l2, self.ln2 = nn.Linear(local, local), nn.LayerNorm(local)
        self.g, self.gn = nn.Linear(9 * local, hidden), nn.LayerNorm(hidden)
        self.blocks = nn.Sequential(*[MlpBlock(hidden) for _ in range(blocks)])
        self.ctx = nn.Linear(hidden, local)
        self.p1, self.p2 = nn.Linear(local, head), nn.Linear(head, 9)
        self.v = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, N), nn.Tanh())

    def forward(self, x):
        b = x.shape[0]
        t = x.view(b, PLANES, 3, 3, 3, 3).permute(0, 2, 4, 1, 3, 5).reshape(b, 9, PLANES * 9)
        h = F.relu(self.ln1(self.l1(t)))
        h = F.relu(h + self.ln2(self.l2(h)))
        g = self.blocks(F.relu(self.gn(self.g(h.reshape(b, -1)))))
        pt = F.relu(h + self.ctx(g).unsqueeze(1))
        logits = self.p2(F.relu(self.p1(pt))).view(b, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(b, 81)
        return logits, self.v(g)


NETS = {
    "resnet6": lambda: ResNet(64, 6, 32, 4, 256),  # beat-it uttt_config.json
    "resnet4": lambda: ResNet(64, 4, 8, 4, 128),   # AlphaZeroNetConfig::ultimate_ttt_resnet
    "sub": SubBoard,
    "mlp": ResMlp,
}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def time_ms(secs, f, dev):
    for _ in range(3):
        f()
    sync(dev)
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < secs or n < 3:
        f()
        n += 1
    sync(dev)
    return (time.perf_counter() - t0) * 1000 / n


def bench(name, dev, secs, compile_=False):
    torch.manual_seed(0)
    net = NETS[name]().to(dev)
    fwd = torch.compile(net) if compile_ else net
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(7)
    line = f"torch  {name + ('+c' if compile_ else ''):<8} {dev.type:<4}"
    for b in (64, 256, 1024):
        states = (rng.random((b, G)) < 0.2).astype(np.float32)
        net.eval()

        def infer():
            with torch.no_grad():
                p, v = fwd(torch.from_numpy(states).to(dev))
                torch.softmax(p, 1).cpu().numpy(), v.cpu().numpy()

        ms = time_ms(secs, infer, dev)
        line += f" | infer b{b:<4} {ms:7.2f} ms ({b / ms * 1000:7.0f} pos/s)"
    net.train()
    for b in (256, 1024):
        s = torch.from_numpy((rng.random((b, G)) < 0.2).astype(np.float32)).to(dev)
        pi = torch.full((b, P), 1.0 / P, device=dev)
        z = torch.tensor([[0.5, -0.5]] * b, device=dev)

        def step():
            logits, v = fwd(s)
            loss = -(pi * F.log_softmax(logits, 1)).sum(1).mean() + F.mse_loss(v, z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        ms = time_ms(secs, step, dev)
        line += f" | train b{b:<4} {ms:7.2f} ms"
    print(line, flush=True)


if __name__ == "__main__":
    secs = float(args.get("secs", "1.5"))
    nets = args.get("nets", "resnet6,resnet4,sub,mlp").split(",")
    print(f"# torch {torch.__version__}, threads={torch.get_num_threads()}, norm={'BatchNorm' if USE_BN else 'GroupNorm(1)'}")
    for d in args.get("devices", "cpu,mps,cuda").split(","):
        if d == "mps" and not torch.backends.mps.is_available():
            continue
        if d == "cuda" and not torch.cuda.is_available():
            continue
        for n in nets:
            bench(n, torch.device(d), secs)
            if d == "cuda" and args.get("compile", "0") == "1":
                bench(n, torch.device(d), secs, compile_=True)
