//! libtorch backend (via `tch-rs`, cargo feature `tch`) for [`AlphaZeroNetConfig`] networks.
//!
//! Mirrors the candle implementation in [`crate::nn`] layer by layer **with identical parameter
//! names and shapes**, so `.safetensors` checkpoints are interchangeable between the backends.
//!
//! Thread model: one [`TchModel`] (VarStore + network) per device replica behind a `Mutex`. The
//! trainer and the inference worker share replica 0; training pauses the inference service, so
//! the lock is uncontended in practice. Apple MPS work is additionally serialized process-wide.

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use tch::nn::{self, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use crate::backend::NnDevice;
use crate::nn::{AlphaZeroNetConfig, BlockLayout};

type Result<T> = candle_core::Result<T>;

fn err(e: tch::TchError) -> candle_core::Error {
    candle_core::Error::Msg(format!("libtorch: {e}"))
}

static MPS_LOCK: Mutex<()> = Mutex::new(());

fn mps_guard(dev: Device) -> Option<MutexGuard<'static, ()>> {
    (dev == Device::Mps).then(|| MPS_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Maps a backend-neutral device to a libtorch device (`gpu` = CUDA device `i`, else Apple MPS).
pub fn tch_device(dev: NnDevice) -> Device {
    match dev.gpu {
        None => Device::Cpu,
        Some(i) if tch::Cuda::is_available() => Device::Cuda(i),
        Some(_) => Device::Mps,
    }
}

/// Number of GPUs libtorch can use (CUDA devices, or 1 for Apple MPS).
pub fn gpu_count() -> usize {
    if tch::Cuda::is_available() {
        tch::Cuda::device_count() as usize
    } else if tch::utils::has_mps() {
        1
    } else {
        0
    }
}

pub fn gpu_name() -> Option<String> {
    if tch::Cuda::is_available() {
        Some(format!("{} × CUDA", tch::Cuda::device_count()))
    } else if tch::utils::has_mps() {
        Some("MPS".to_string())
    } else {
        None
    }
}

// ------------------------------------------------------------------------------------------------
// Layers (names match `crate::nn`)
// ------------------------------------------------------------------------------------------------

fn linear(p: nn::Path, i: usize, o: usize) -> nn::Linear {
    nn::linear(p, i as i64, o as i64, Default::default())
}

/// Final policy/value layer, zero-initialized like `crate::nn::linear_zero`.
fn linear_zero(p: nn::Path, i: usize, o: usize) -> nn::Linear {
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.0),
        bs_init: Some(nn::Init::Const(0.0)),
        bias: true,
    };
    nn::linear(p, i as i64, o as i64, cfg)
}

fn layer_norm(p: nn::Path, d: usize) -> nn::LayerNorm {
    nn::layer_norm(p, vec![d as i64], nn::LayerNormConfig { eps: 1e-5, ..Default::default() })
}

fn conv(p: nn::Path, i: usize, o: usize, k: usize, bias: bool) -> nn::Conv2D {
    let cfg = nn::ConvConfig { padding: (k / 2) as i64, bias, ..Default::default() };
    nn::conv2d(p, i as i64, o as i64, k as i64, cfg)
}

/// `GroupNorm(1, C)` built from `layer_norm` over `(C, H, W)` plus a per-channel affine: same math
/// as candle's `group_norm(1, ..)`, and (unlike libtorch's fused group norm) it has a backward
/// pass on Apple MPS.
#[derive(Debug)]
struct GroupNorm1 {
    weight: Tensor,
    bias: Tensor,
}

impl GroupNorm1 {
    fn new(p: nn::Path, c: usize) -> Self {
        Self {
            weight: p.var("weight", &[c as i64], nn::Init::Const(1.0)),
            bias: p.var("bias", &[c as i64], nn::Init::Const(0.0)),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let s = x.size();
        let y = x.layer_norm(&s[1..], None::<Tensor>, None::<Tensor>, 1e-5, false);
        y * self.weight.view([1, s[1], 1, 1]) + self.bias.view([1, s[1], 1, 1])
    }
}

#[derive(Debug)]
struct MlpBlock {
    fc1: nn::Linear,
    norm1: nn::LayerNorm,
    fc2: nn::Linear,
    norm2: nn::LayerNorm,
}

impl MlpBlock {
    fn new(p: nn::Path, d: usize) -> Self {
        Self {
            fc1: linear(&p / "fc1", d, d),
            norm1: layer_norm(&p / "norm1", d),
            fc2: linear(&p / "fc2", d, d),
            norm2: layer_norm(&p / "norm2", d),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = x.apply(&self.fc1).apply(&self.norm1).relu();
        (x + h.apply(&self.fc2).apply(&self.norm2)).relu()
    }
}

#[derive(Debug)]
struct ConvBlock {
    conv1: nn::Conv2D,
    norm1: GroupNorm1,
    conv2: nn::Conv2D,
    norm2: GroupNorm1,
}

#[derive(Debug)]
enum Trunk {
    ResNet {
        shape: [i64; 3],
        init_conv: nn::Conv2D,
        init_norm: GroupNorm1,
        blocks: Vec<ConvBlock>,
        policy_conv: nn::Conv2D,
        policy_norm: GroupNorm1,
        policy_fc1: nn::Linear,
        policy_fc2: nn::Linear,
        value_conv: nn::Conv2D,
        value_norm: GroupNorm1,
        value_fc1: nn::Linear,
        value_fc2: nn::Linear,
    },
    ResMlp {
        input_fc: nn::Linear,
        input_norm: nn::LayerNorm,
        blocks: Vec<MlpBlock>,
        policy_fc1: nn::Linear,
        policy_fc2: nn::Linear,
        value_fc1: nn::Linear,
        value_fc2: nn::Linear,
    },
    Block {
        layout: BlockLayout,
        local_dim: usize,
        local_fc1: nn::Linear,
        local_norm1: nn::LayerNorm,
        local_fc2: nn::Linear,
        local_norm2: nn::LayerNorm,
        global_fc: nn::Linear,
        global_norm: nn::LayerNorm,
        blocks: Vec<MlpBlock>,
        policy_ctx: nn::Linear,
        policy_fc1: nn::Linear,
        policy_fc2: nn::Linear,
        value_fc1: nn::Linear,
        value_fc2: nn::Linear,
    },
}

/// libtorch policy-value network (`[B, G]` -> `([B, P] logits, [B, N] tanh values)`).
#[derive(Debug)]
pub struct TchNet {
    trunk: Trunk,
}

impl TchNet {
    pub fn new(p: &nn::Path, cfg: &AlphaZeroNetConfig, g: usize, pol: usize, n: usize) -> Self {
        let trunk = match *cfg {
            AlphaZeroNetConfig::ResNet2d {
                in_channels,
                board_height,
                board_width,
                channels,
                num_res_blocks,
                policy_channels,
                value_channels,
                fc_hidden,
            } => {
                assert_eq!(in_channels * board_height * board_width, g, "ResNet2d dimensions C*H*W must equal G");
                let area = board_height * board_width;
                Trunk::ResNet {
                    shape: [in_channels as i64, board_height as i64, board_width as i64],
                    init_conv: conv(p / "init_conv", in_channels, channels, 3, false),
                    init_norm: GroupNorm1::new(p / "init_norm", channels),
                    blocks: (0..num_res_blocks)
                        .map(|i| {
                            let b = p / format!("res_{i}");
                            ConvBlock {
                                conv1: conv(&b / "conv1", channels, channels, 3, false),
                                norm1: GroupNorm1::new(&b / "norm1", channels),
                                conv2: conv(&b / "conv2", channels, channels, 3, false),
                                norm2: GroupNorm1::new(&b / "norm2", channels),
                            }
                        })
                        .collect(),
                    policy_conv: conv(p / "policy_conv", channels, policy_channels, 1, true),
                    policy_norm: GroupNorm1::new(p / "policy_norm", policy_channels),
                    policy_fc1: linear(p / "policy_fc1", policy_channels * area, fc_hidden),
                    policy_fc2: linear_zero(p / "policy_fc2", fc_hidden, pol),
                    value_conv: conv(p / "value_conv", channels, value_channels, 1, true),
                    value_norm: GroupNorm1::new(p / "value_norm", value_channels),
                    value_fc1: linear(p / "value_fc1", value_channels * area, fc_hidden),
                    value_fc2: linear_zero(p / "value_fc2", fc_hidden, n),
                }
            }
            AlphaZeroNetConfig::ResMlp { hidden_dim, num_res_blocks, head_hidden } => Trunk::ResMlp {
                input_fc: linear(p / "input_fc", g, hidden_dim),
                input_norm: layer_norm(p / "input_norm", hidden_dim),
                blocks: (0..num_res_blocks).map(|i| MlpBlock::new(p / format!("res_{i}"), hidden_dim)).collect(),
                policy_fc1: linear(p / "policy_fc1", hidden_dim, head_hidden),
                policy_fc2: linear_zero(p / "policy_fc2", head_hidden, pol),
                value_fc1: linear(p / "value_fc1", hidden_dim, head_hidden),
                value_fc2: linear_zero(p / "value_fc2", head_hidden, n),
            },
            AlphaZeroNetConfig::BlockNet {
                in_planes,
                board_height,
                board_width,
                block_height,
                block_width,
                local_dim,
                hidden_dim,
                num_res_blocks,
                head_hidden,
            } => {
                let layout = BlockLayout {
                    planes: in_planes,
                    rows: board_height / block_height,
                    cols: board_width / block_width,
                    bh: block_height,
                    bw: block_width,
                };
                assert_eq!(in_planes * board_height * board_width, g, "BlockNet requires G == in_planes * H * W");
                assert_eq!(board_height * board_width, pol, "BlockNet requires P == H * W");
                let (tokens, cells) = (layout.num_blocks(), layout.cells());
                Trunk::Block {
                    layout,
                    local_dim,
                    local_fc1: linear(p / "local_fc1", in_planes * cells, local_dim),
                    local_norm1: layer_norm(p / "local_norm1", local_dim),
                    local_fc2: linear(p / "local_fc2", local_dim, local_dim),
                    local_norm2: layer_norm(p / "local_norm2", local_dim),
                    global_fc: linear(p / "global_fc", tokens * local_dim, hidden_dim),
                    global_norm: layer_norm(p / "global_norm", hidden_dim),
                    blocks: (0..num_res_blocks).map(|i| MlpBlock::new(p / format!("res_{i}"), hidden_dim)).collect(),
                    policy_ctx: linear(p / "policy_ctx", hidden_dim, local_dim),
                    policy_fc1: linear(p / "policy_fc1", local_dim, head_hidden),
                    policy_fc2: linear_zero(p / "policy_fc2", head_hidden, cells),
                    value_fc1: linear(p / "value_fc1", hidden_dim, head_hidden),
                    value_fc2: linear_zero(p / "value_fc2", head_hidden, n),
                }
            }
        };
        Self { trunk }
    }

    pub fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let b = xs.size()[0];
        match &self.trunk {
            Trunk::ResNet {
                shape,
                init_conv,
                init_norm,
                blocks,
                policy_conv,
                policy_norm,
                policy_fc1,
                policy_fc2,
                value_conv,
                value_norm,
                value_fc1,
                value_fc2,
            } => {
                let x = xs.view([b, shape[0], shape[1], shape[2]]);
                let mut h = init_norm.forward(&x.apply(init_conv)).relu();
                for blk in blocks {
                    let t = blk.norm1.forward(&h.apply(&blk.conv1)).relu();
                    h = (&h + blk.norm2.forward(&t.apply(&blk.conv2))).relu();
                }
                let p = policy_norm.forward(&h.apply(policy_conv)).relu().flatten(1, -1);
                let v = value_norm.forward(&h.apply(value_conv)).relu().flatten(1, -1);
                (p.apply(policy_fc1).relu().apply(policy_fc2), v.apply(value_fc1).relu().apply(value_fc2).tanh())
            }
            Trunk::ResMlp { input_fc, input_norm, blocks, policy_fc1, policy_fc2, value_fc1, value_fc2 } => {
                let mut h = xs.apply(input_fc).apply(input_norm).relu();
                for blk in blocks {
                    h = blk.forward(&h);
                }
                (h.apply(policy_fc1).relu().apply(policy_fc2), h.apply(value_fc1).relu().apply(value_fc2).tanh())
            }
            Trunk::Block {
                layout: l,
                local_dim,
                local_fc1,
                local_norm1,
                local_fc2,
                local_norm2,
                global_fc,
                global_norm,
                blocks,
                policy_ctx,
                policy_fc1,
                policy_fc2,
                value_fc1,
                value_fc2,
            } => {
                let (planes, rows, cols, bh, bw) = (l.planes as i64, l.rows as i64, l.cols as i64, l.bh as i64, l.bw as i64);
                let tokens = xs
                    .view([b, planes, rows, bh, cols, bw])
                    .permute([0, 2, 4, 1, 3, 5])
                    .reshape([b, rows * cols, planes * bh * bw]);
                let t = tokens.apply(local_fc1).apply(local_norm1).relu();
                let t = (&t + t.apply(local_fc2).apply(local_norm2)).relu();
                let mut g = t.reshape([b, rows * cols * *local_dim as i64]).apply(global_fc).apply(global_norm).relu();
                for blk in blocks {
                    g = blk.forward(&g);
                }
                let pt = (&t + g.apply(policy_ctx).unsqueeze(1)).relu();
                let logits = pt
                    .apply(policy_fc1)
                    .relu()
                    .apply(policy_fc2)
                    .view([b, rows, cols, bh, bw])
                    .permute([0, 1, 3, 2, 4])
                    .reshape([b, rows * bh * cols * bw]);
                (logits, g.apply(value_fc1).relu().apply(value_fc2).tanh())
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Model replicas, evaluator and trainer
// ------------------------------------------------------------------------------------------------

/// One network replica on one device.
pub struct TchModel {
    pub vs: nn::VarStore,
    pub net: TchNet,
    pub device: Device,
}

impl TchModel {
    pub fn new(cfg: &AlphaZeroNetConfig, device: Device, g: usize, p: usize, n: usize) -> Self {
        let _mps = mps_guard(device);
        let vs = nn::VarStore::new(device);
        let net = TchNet::new(&vs.root(), cfg, g, p, n);
        Self { vs, net, device }
    }
}

pub type SharedModel = Arc<Mutex<TchModel>>;

fn lock(m: &SharedModel) -> MutexGuard<'_, TchModel> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Multi-replica libtorch evaluator: replica 0 is shared with the trainer, further replicas
/// (one per extra GPU) are refreshed by [`TchEvaluator::sync_workers`]. Clones share weights.
#[derive(Clone)]
pub struct TchEvaluator {
    replicas: Vec<SharedModel>,
    cfg: AlphaZeroNetConfig,
    dims: (usize, usize, usize),
}

impl TchEvaluator {
    pub fn new(master: SharedModel, cfg: AlphaZeroNetConfig, devices: &[NnDevice], dims: (usize, usize, usize)) -> Result<Self> {
        let mut replicas = vec![master];
        for d in devices.iter().skip(1) {
            replicas.push(Arc::new(Mutex::new(TchModel::new(&cfg, tch_device(*d), dims.0, dims.1, dims.2))));
        }
        let eval = Self { replicas, cfg, dims };
        eval.sync_workers()?;
        Ok(eval)
    }

    pub fn standalone(cfg: &AlphaZeroNetConfig, devices: &[NnDevice], dims: (usize, usize, usize), checkpoint: Option<&Path>) -> Result<Self> {
        let dev = tch_device(devices.first().copied().unwrap_or(NnDevice::CPU_TCH));
        let mut model = TchModel::new(cfg, dev, dims.0, dims.1, dims.2);
        if let Some(path) = checkpoint {
            let _mps = mps_guard(dev);
            model.vs.load(path).map_err(err)?;
        }
        Self::new(Arc::new(Mutex::new(model)), *cfg, devices, dims)
    }

    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }

    pub fn sync_workers(&self) -> Result<()> {
        if self.replicas.len() <= 1 {
            return Ok(());
        }
        let master = lock(&self.replicas[0]);
        for r in &self.replicas[1..] {
            lock(r).vs.copy(&master.vs).map_err(err)?;
        }
        Ok(())
    }

    pub fn save_safetensors(&self, path: &Path) -> Result<()> {
        let m = lock(&self.replicas[0]);
        let _mps = mps_guard(m.device);
        m.vs.save(path).map_err(err)
    }

    pub fn load_safetensors(&self, path: &Path) -> Result<()> {
        {
            let mut m = lock(&self.replicas[0]);
            let _mps = mps_guard(m.device);
            m.vs.load(path).map_err(err)?;
        }
        self.sync_workers()
    }

    pub fn reset_random_weights(&self) -> Result<()> {
        {
            let mut m = lock(&self.replicas[0]);
            let fresh = TchModel::new(&self.cfg, m.device, self.dims.0, self.dims.1, self.dims.2);
            let _mps = mps_guard(m.device);
            m.vs.copy(&fresh.vs).map_err(err)?;
        }
        self.sync_workers()
    }

    /// Writes raw policy logits and values for `b` flat states (`states.len() == b * G`).
    pub fn evaluate_flat(&self, worker_id: usize, states: &[f32], b: usize, out_p: &mut [f32], out_v: &mut [f32]) {
        let (g, p, n) = self.dims;
        // Host-side copies happen outside the locks, so other users of this replica (or of the
        // GPU) are only blocked while the device works.
        let x = Tensor::from_slice(states).view([b as i64, g as i64]);
        let (logits, v) = {
            let m = lock(&self.replicas[worker_id % self.replicas.len()]);
            let _mps = mps_guard(m.device);
            let _ng = tch::no_grad_guard();
            let (logits, v) = m.net.forward(&x.to_device(m.device));
            // Device tensors are dropped before the guards (reverse declaration order).
            (logits.to_device(Device::Cpu), v.to_device(Device::Cpu))
        };
        logits.copy_data(out_p, b * p);
        v.copy_data(out_v, b * n);
    }
}

/// AdamW trainer over the shared replica-0 model.
pub struct TchTrainer {
    model: SharedModel,
    opt: nn::Optimizer,
    cfg: AlphaZeroNetConfig,
    dims: (usize, usize, usize),
}

impl TchTrainer {
    pub fn new(cfg: AlphaZeroNetConfig, lr: f64, wd: f64, device: NnDevice, dims: (usize, usize, usize)) -> Result<Self> {
        let model = TchModel::new(&cfg, tch_device(device), dims.0, dims.1, dims.2);
        let opt = Self::optimizer(&model.vs, lr, wd)?;
        Ok(Self { model: Arc::new(Mutex::new(model)), opt, cfg, dims })
    }

    fn optimizer(vs: &nn::VarStore, lr: f64, wd: f64) -> Result<nn::Optimizer> {
        nn::AdamW { beta1: 0.9, beta2: 0.999, wd, eps: 1e-8, amsgrad: false }.build(vs, lr).map_err(err)
    }

    pub fn shared_model(&self) -> SharedModel {
        Arc::clone(&self.model)
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.opt.set_lr(lr);
    }

    pub fn set_weight_decay(&mut self, wd: f64) {
        self.opt.set_weight_decay(wd);
    }

    /// Re-creates the optimizer (fresh moment estimates), e.g. after the weights were reset.
    pub fn reset_optimizer(&mut self, lr: f64, wd: f64) -> Result<()> {
        let m = lock(&self.model);
        self.opt = Self::optimizer(&m.vs, lr, wd)?;
        Ok(())
    }

    pub fn train_step(&mut self, states: &[f32], policies: &[f32], values: &[f32], b: usize) -> Result<(f32, f32, f32)> {
        let (g, p, n) = self.dims;
        let m = lock(&self.model);
        let _mps = mps_guard(m.device);
        let dev = m.device;
        let x = Tensor::from_slice(states).view([b as i64, g as i64]).to_device(dev);
        let pi = Tensor::from_slice(policies).view([b as i64, p as i64]).to_device(dev);
        let z = Tensor::from_slice(values).view([b as i64, n as i64]).to_device(dev);
        let (logits, v) = m.net.forward(&x);
        let policy_loss = -(&pi * logits.log_softmax(1, Kind::Float)).sum(Kind::Float) / b as f64;
        let value_loss = (&v - &z).square().mean(Kind::Float);
        let total = &policy_loss + &value_loss;
        self.opt.backward_step(&total);
        let mut out = [0f32; 3];
        Tensor::stack(&[total, policy_loss, value_loss], 0).detach().to_device(Device::Cpu).copy_data(&mut out, 3);
        Ok((out[0], out[1], out[2]))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let m = lock(&self.model);
        let _mps = mps_guard(m.device);
        m.vs.save(path).map_err(err)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let mut m = lock(&self.model);
        let _mps = mps_guard(m.device);
        m.vs.load(path).map_err(err)
    }

    pub const fn net_config(&self) -> &AlphaZeroNetConfig {
        &self.cfg
    }
}
