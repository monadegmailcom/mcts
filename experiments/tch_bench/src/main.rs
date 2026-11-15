//! libtorch-from-Rust (tch-rs) counterpart of `examples/bench_nn.rs` and
//! `experiments/torch_bench.py`: same networks, batch sizes and measurements.
//!
//! Build against a PyTorch install (macOS dev box):
//!   LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 PATH=<venv>/bin:$PATH \
//!   DYLD_LIBRARY_PATH=<venv>/lib/python3.x/site-packages/torch/lib cargo run --release
//! or against a standalone libtorch (Linux/CUDA, e.g. a RunPod Docker image):
//!   LIBTORCH=/opt/libtorch LD_LIBRARY_PATH=/opt/libtorch/lib cargo run --release
//! Options: devices=cpu,mps,cuda nets=resnet6,resnet6bn,sub,mlp secs=1.5

use std::time::Instant;

use tch::nn::{self, Module, ModuleT, OptimizerConfig};
use tch::{Device, Kind, Tensor};

const PLANES: i64 = 7;
const G: i64 = PLANES * 81;
const P: i64 = 81;
const N: i64 = 2;

#[derive(Debug)]
enum Norm2d {
    Bn(nn::BatchNorm),
    Gn(nn::GroupNorm),
}

impl Norm2d {
    fn new(p: nn::Path, c: i64, bn: bool) -> Self {
        if bn { Self::Bn(nn::batch_norm2d(p, c, Default::default())) } else { Self::Gn(nn::group_norm(p, 1, c, Default::default())) }
    }
    fn fwd(&self, x: &Tensor, train: bool) -> Tensor {
        match self {
            Self::Bn(b) => b.forward_t(x, train),
            Self::Gn(g) => g.forward(x),
        }
    }
}

fn conv(p: nn::Path, i: i64, o: i64, k: i64, bias: bool) -> nn::Conv2D {
    nn::conv2d(p, i, o, k, nn::ConvConfig { padding: k / 2, bias, ..Default::default() })
}

#[derive(Debug)]
struct ResNet {
    inp: nn::Conv2D,
    inp_n: Norm2d,
    blocks: Vec<(nn::Conv2D, Norm2d, nn::Conv2D, Norm2d)>,
    pconv: nn::Conv2D,
    pn: Norm2d,
    pfc1: nn::Linear,
    pfc2: nn::Linear,
    vconv: nn::Conv2D,
    vn: Norm2d,
    vfc1: nn::Linear,
    vfc2: nn::Linear,
}

impl ResNet {
    fn new(p: &nn::Path, ch: i64, blocks: i64, pch: i64, vch: i64, fc: i64, bn: bool) -> Self {
        Self {
            inp: conv(p / "inp", PLANES, ch, 3, false),
            inp_n: Norm2d::new(p / "inp_n", ch, bn),
            blocks: (0..blocks)
                .map(|i| {
                    let b = p / format!("b{i}");
                    (conv(&b / "c1", ch, ch, 3, false), Norm2d::new(&b / "n1", ch, bn), conv(&b / "c2", ch, ch, 3, false), Norm2d::new(&b / "n2", ch, bn))
                })
                .collect(),
            pconv: conv(p / "pconv", ch, pch, 1, true),
            pn: Norm2d::new(p / "pn", pch, bn),
            pfc1: nn::linear(p / "pfc1", pch * 81, fc, Default::default()),
            pfc2: nn::linear(p / "pfc2", fc, P, Default::default()),
            vconv: conv(p / "vconv", ch, vch, 1, true),
            vn: Norm2d::new(p / "vn", vch, bn),
            vfc1: nn::linear(p / "vfc1", vch * 81, fc, Default::default()),
            vfc2: nn::linear(p / "vfc2", fc, N, Default::default()),
        }
    }
    fn fwd(&self, x: &Tensor, t: bool) -> (Tensor, Tensor) {
        let mut x = self.inp_n.fwd(&x.view([-1, PLANES, 9, 9]).apply(&self.inp), t).relu();
        for (c1, n1, c2, n2) in &self.blocks {
            let h = n1.fwd(&x.apply(c1), t).relu();
            x = (&x + n2.fwd(&h.apply(c2), t)).relu();
        }
        let p = self.pn.fwd(&x.apply(&self.pconv), t).relu().flatten(1, -1);
        let v = self.vn.fwd(&x.apply(&self.vconv), t).relu().flatten(1, -1);
        (p.apply(&self.pfc1).relu().apply(&self.pfc2), v.apply(&self.vfc1).relu().apply(&self.vfc2).tanh())
    }
}

#[derive(Debug)]
struct MlpBlock(nn::Linear, nn::LayerNorm, nn::Linear, nn::LayerNorm);
impl MlpBlock {
    fn new(p: nn::Path, d: i64) -> Self {
        Self(nn::linear(&p / "f1", d, d, Default::default()), nn::layer_norm(&p / "n1", vec![d], Default::default()), nn::linear(&p / "f2", d, d, Default::default()), nn::layer_norm(&p / "n2", vec![d], Default::default()))
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        (x + x.apply(&self.0).apply(&self.1).relu().apply(&self.2).apply(&self.3)).relu()
    }
}

#[derive(Debug)]
struct SubBoard {
    l1: nn::Linear,
    ln1: nn::LayerNorm,
    l2: nn::Linear,
    ln2: nn::LayerNorm,
    g: nn::Linear,
    gn: nn::LayerNorm,
    blocks: Vec<MlpBlock>,
    ctx: nn::Linear,
    p1: nn::Linear,
    p2: nn::Linear,
    v1: nn::Linear,
    v2: nn::Linear,
}

impl SubBoard {
    fn new(p: &nn::Path) -> Self {
        let (local, hidden, head) = (64, 256, 64);
        let lin = |n: &str, i, o| nn::linear(p / n, i, o, Default::default());
        let ln = |n: &str, d| nn::layer_norm(p / n, vec![d], Default::default());
        Self {
            l1: lin("l1", PLANES * 9, local),
            ln1: ln("ln1", local),
            l2: lin("l2", local, local),
            ln2: ln("ln2", local),
            g: lin("g", 9 * local, hidden),
            gn: ln("gn", hidden),
            blocks: (0..3).map(|i| MlpBlock::new(p / format!("b{i}"), hidden)).collect(),
            ctx: lin("ctx", hidden, local),
            p1: lin("p1", local, head),
            p2: lin("p2", head, 9),
            v1: lin("v1", hidden, head),
            v2: lin("v2", head, N),
        }
    }
    fn fwd(&self, x: &Tensor) -> (Tensor, Tensor) {
        let b = x.size()[0];
        let t = x.view([b, PLANES, 3, 3, 3, 3]).permute([0, 2, 4, 1, 3, 5]).reshape([b, 9, PLANES * 9]);
        let h = t.apply(&self.l1).apply(&self.ln1).relu();
        let h = (&h + h.apply(&self.l2).apply(&self.ln2)).relu();
        let mut g = h.reshape([b, -1]).apply(&self.g).apply(&self.gn).relu();
        for blk in &self.blocks {
            g = blk.fwd(&g);
        }
        let pt = (&h + g.apply(&self.ctx).unsqueeze(1)).relu();
        let logits = pt.apply(&self.p1).relu().apply(&self.p2).view([b, 3, 3, 3, 3]).permute([0, 1, 3, 2, 4]).reshape([b, 81]);
        (logits, g.apply(&self.v1).relu().apply(&self.v2).tanh())
    }
}

#[derive(Debug)]
struct ResMlp {
    inp: nn::Linear,
    n: nn::LayerNorm,
    blocks: Vec<MlpBlock>,
    p1: nn::Linear,
    p2: nn::Linear,
    v1: nn::Linear,
    v2: nn::Linear,
}

impl ResMlp {
    fn new(p: &nn::Path) -> Self {
        let lin = |n: &str, i, o| nn::linear(p / n, i, o, Default::default());
        Self {
            inp: lin("inp", G, 128),
            n: nn::layer_norm(p / "n", vec![128], Default::default()),
            blocks: (0..2).map(|i| MlpBlock::new(p / format!("b{i}"), 128)).collect(),
            p1: lin("p1", 128, 96),
            p2: lin("p2", 96, P),
            v1: lin("v1", 128, 96),
            v2: lin("v2", 96, N),
        }
    }
    fn fwd(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = x.apply(&self.inp).apply(&self.n).relu();
        for blk in &self.blocks {
            h = blk.fwd(&h);
        }
        (h.apply(&self.p1).relu().apply(&self.p2), h.apply(&self.v1).relu().apply(&self.v2).tanh())
    }
}

enum Net {
    Res(ResNet),
    Sub(SubBoard),
    Mlp(ResMlp),
}

impl Net {
    fn fwd(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        match self {
            Self::Res(n) => n.fwd(x, train),
            Self::Sub(n) => n.fwd(x),
            Self::Mlp(n) => n.fwd(x),
        }
    }
}

fn sync(dev: Device) {
    match dev {
        Device::Cuda(i) => tch::Cuda::synchronize(i as i64),
        // A tiny readback forces the MPS command queue to drain.
        Device::Mps => {
            let _ = Tensor::zeros([1], (Kind::Float, dev)).double_value(&[0]);
        }
        _ => {}
    }
}

fn time_ms(secs: f64, dev: Device, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    sync(dev);
    let t0 = Instant::now();
    let mut n = 0;
    while t0.elapsed().as_secs_f64() < secs || n < 3 {
        f();
        n += 1;
    }
    sync(dev);
    t0.elapsed().as_secs_f64() * 1000.0 / n as f64
}

fn bench(name: &str, dev: Device, secs: f64) {
    tch::manual_seed(0);
    let vs = nn::VarStore::new(dev);
    let root = vs.root();
    let net = match name {
        "resnet6" => Net::Res(ResNet::new(&root, 64, 6, 32, 4, 256, false)),
        "resnet6bn" => Net::Res(ResNet::new(&root, 64, 6, 32, 4, 256, true)),
        "resnet4" => Net::Res(ResNet::new(&root, 64, 4, 8, 4, 128, false)),
        "sub" => Net::Sub(SubBoard::new(&root)),
        "mlp" => Net::Mlp(ResMlp::new(&root)),
        o => panic!("unknown net {o}"),
    };
    let mut opt = nn::AdamW { wd: 1e-4, ..Default::default() }.build(&vs, 1e-3).expect("optimizer");
    let mut line = format!("tch    {name:<9} {:<4}", format!("{dev:?}").to_lowercase().chars().take(4).collect::<String>());
    let mut seed = 7u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        if seed % 5 == 0 { 1.0f32 } else { 0.0 }
    };
    for b in [64i64, 256, 1024] {
        let host: Vec<f32> = (0..b * G).map(|_| rnd()).collect();
        let mut pol = vec![0f32; (b * P) as usize];
        let mut val = vec![0f32; (b * N) as usize];
        let ms = time_ms(secs, dev, || {
            let _g = tch::no_grad_guard();
            let x = Tensor::from_slice(&host).view([b, G]).to_device(dev);
            let (p, v) = net.fwd(&x, false);
            p.softmax(1, Kind::Float).to_device(Device::Cpu).copy_data(&mut pol, (b * P) as usize);
            v.to_device(Device::Cpu).copy_data(&mut val, (b * N) as usize);
        });
        line += &format!(" | infer b{b:<4} {ms:>7.2} ms ({:>7.0} pos/s)", b as f64 / ms * 1000.0);
    }
    for b in [256i64, 1024] {
        let x = Tensor::from_slice(&(0..b * G).map(|_| rnd()).collect::<Vec<_>>()).view([b, G]).to_device(dev);
        let pi = Tensor::full([b, P], 1.0 / P as f64, (Kind::Float, dev));
        let z = Tensor::from_slice(&[0.5f32, -0.5]).view([1, N]).repeat([b, 1]).to_device(dev);
        let ms = time_ms(secs, dev, || {
            let (logits, v) = net.fwd(&x, true);
            let loss = -(&pi * logits.log_softmax(1, Kind::Float)).sum_dim_intlist(1, false, Kind::Float).mean(Kind::Float) + v.mse_loss(&z, tch::Reduction::Mean);
            opt.backward_step(&loss);
        });
        line += &format!(" | train b{b:<4} {ms:>7.2} ms");
    }
    println!("{line}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |k: &str, d: &str| args.iter().find_map(|a| a.strip_prefix(&format!("{k}=")).map(str::to_string)).unwrap_or_else(|| d.to_string());
    let secs: f64 = arg("secs", "1.5").parse().unwrap();
    println!("# tch {} threads={}", env!("CARGO_PKG_VERSION"), tch::get_num_threads());
    for d in arg("devices", "cpu,mps,cuda").split(',') {
        let dev = match d {
            "cpu" => Device::Cpu,
            "mps" if tch::utils::has_mps() => Device::Mps,
            "cuda" if tch::Cuda::is_available() => Device::Cuda(0),
            _ => continue,
        };
        for net in arg("nets", "resnet6,resnet6bn,resnet4,sub,mlp").split(',') {
            bench(net, dev, secs);
        }
    }
}
