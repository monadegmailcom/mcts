//! Pure-Rust AlphaZero Policy-Value Neural Networks and Multi-Device `BatchEvaluator`
//! powered by HuggingFace [`candle_core`] and [`candle_nn`].
//!
//! ### Architectural Highlights
//! 1. **100% Pure Rust (CPU, Apple Metal, NVIDIA Multi-GPU CUDA, & WebAssembly Ready)**:
//!    Replaces C++ `libtorch_util.h` (`libtorch` C++ dynamic library) with pure Rust.
//!    Compiles out-of-the-box on macOS, Linux, iOS, Android, and `wasm32` without C++ toolchains,
//!    while supporting GPU acceleration via Cargo features (`--features metal` or `--features cuda`).
//! 2. **Two Configurable Trunk Architectures**:
//!    - [`AlphaZeroNetConfig::ResNet2d`]: 2D Convolutional Residual Tower with `1x1` policy/value
//!      bottleneck heads (port of `AlphaZeroCNN` in `train/alphazero_cnn.py` for spatial board
//!      games like Tic-Tac-Toe and Ultimate Tic-Tac-Toe).
//!    - [`AlphaZeroNetConfig::ResMlp`]: Pre-activated / LayerNorm Residual MLP Tower for flat
//!      or non-spatial state encodings (such as 3x3x3 Rubik's Cube's `324`-dim one-hot state).
//! 3. **Zero Train/Eval Drift via `GroupNorm` & `LayerNorm`**:
//!    Uses per-sample normalization (`GroupNorm(1, C)` on 2D feature maps and `LayerNorm` on MLP
//!    layers) rather than `BatchNorm`, ensuring that evaluating a state in a batch of size `1`,
//!    `16`, or `64` produces identical logits and values and every `Var` in [`VarMap`] is a pure
//!    learnable parameter saved/loaded by `.safetensors`.

use std::path::Path;
use std::sync::OnceLock;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, GroupNorm, Linear, VarBuilder, VarMap, conv2d, conv2d_no_bias, group_norm,
    linear,
};

/// LayerNorm over the last dimension that is **differentiable**.
///
/// `candle_nn::LayerNorm` dispatches contiguous inputs to a fused kernel registered with
/// `apply_op3_no_bwd`: it has no backward pass, so gradients silently stop at every LayerNorm
/// and no layer before it (nor the norm's own affine parameters) ever learns. This version is
/// built from primitive ops (`candle_nn::ops::layer_norm_slow`), with the same parameter names
/// (`weight`, `bias`) so checkpoints are unchanged.
#[derive(Clone, Debug)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm_slow(xs, &self.weight, &self.bias, self.eps)
    }
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?,
        bias: vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.0))?,
        eps: eps as f32,
    })
}

use crate::inference::BatchEvaluator;

/// Process-wide cache of discovered hardware accelerator devices so that [`default_device`] and
/// [`discover_devices`] always return clones of the **exact same** `Device` instance (`DeviceId`).
///
/// In `candle_core`, each call to `Device::new_metal(0)` or `Device::new_cuda(idx)` allocates a
/// distinct internal `DeviceId`, which would cause `Tensor::matmul` to fail with a `device mismatch`
/// error if a model is built on one `Device::new_metal(0)` call and evaluated on another!
static CACHED_DEVICES: OnceLock<Vec<Device>> = OnceLock::new();

/// Process-wide lock serializing all candle work on Metal devices.
///
/// candle's Metal backend is not safe for concurrent use of one device from several threads:
/// allocating a buffer (`MetalDevice::new_buffer` -> `IOGPUMetalResidencySet commit`) while another
/// thread runs kernels or releases buffers crashes with `SIGSEGV` inside `objc_release`. Every
/// [`crate::inference::InferenceService`] owns its own worker thread, so this happens as soon as two
/// networks share the GPU (e.g. a live model plus a baseline/snapshot model in an arena match).
static METAL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

thread_local! {
    static METAL_LOCK_HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Guard returned by [`metal_guard`]; releases the process-wide Metal lock on drop.
pub struct MetalGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl Drop for MetalGuard {
    fn drop(&mut self) {
        METAL_LOCK_HELD.with(|h| h.set(false));
    }
}

/// Acquires the process-wide Metal lock if `device` is a Metal device (re-entrant per thread:
/// returns `None` when this thread already holds it, or for CPU/CUDA devices).
///
/// Hold the guard across *all* tensor work including the drop of temporaries: bind it first
/// (`let _metal = metal_guard(&device);`) so it is released last.
pub fn metal_guard(device: &Device) -> Option<MetalGuard> {
    if !device.is_metal() || METAL_LOCK_HELD.with(|h| h.get()) {
        return None;
    }
    let guard = METAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    METAL_LOCK_HELD.with(|h| h.set(true));
    Some(MetalGuard { _guard: guard })
}

fn init_hardware_devices() -> Vec<Device> {
    let mut cuda_devices = Vec::new();
    for idx in 0..16 {
        if let Ok(dev) = Device::new_cuda(idx) {
            cuda_devices.push(dev);
        } else {
            break;
        }
    }
    if !cuda_devices.is_empty() {
        return cuda_devices;
    }
    if let Ok(dev) = Device::new_metal(0) {
        return vec![dev];
    }
    vec![Device::Cpu]
}

/// Selects the best single accelerator device available on the current machine
/// (`CUDA(0)` -> `Metal(0)` -> `Cpu`), sharing the same underlying `DeviceId` as [`discover_devices`].
pub fn default_device() -> Device {
    CACHED_DEVICES.get_or_init(init_hardware_devices)[0].clone()
}

/// The accelerator devices found at startup (CUDA GPUs or the Metal GPU); empty on CPU-only builds.
pub fn accelerator_devices() -> Vec<Device> {
    CACHED_DEVICES
        .get_or_init(init_hardware_devices)
        .iter()
        .filter(|d| !d.is_cpu())
        .cloned()
        .collect()
}

/// Discovers up to `max_workers` inference devices for [`crate::inference::InferenceService`].
///
/// - On a multi-GPU RunPod node (`--features cuda`), binds `worker_id` `0..K` to `CUDA(0)..CUDA(K-1)`.
/// - On Apple Silicon (`--features metal`), binds to `Metal(0)`.
/// - Otherwise falls back to `Device::Cpu`.
pub fn discover_devices(max_workers: usize) -> Vec<Device> {
    let limit = max_workers.max(1);
    let all = CACHED_DEVICES.get_or_init(init_hardware_devices);
    all.iter().take(limit).cloned().collect()
}

/// Architecture configuration for [`AlphaZeroNet`]. Checkpoints store it in their metadata
/// (see [`crate::checkpoints`]).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AlphaZeroNetConfig {
    /// 2D Convolutional Residual Network (for spatial board games where
    /// `in_channels * board_height * board_width == G`).
    ResNet2d {
        in_channels: usize,
        board_height: usize,
        board_width: usize,
        channels: usize,
        num_res_blocks: usize,
        policy_channels: usize,
        value_channels: usize,
        fc_hidden: usize,
    },
    /// Residual Multi-Layer Perceptron with LayerNorm (for any flattened state vector `G`,
    /// including 3x3x3 Rubik's Cube and 2D games).
    ResMlp {
        hidden_dim: usize,
        num_res_blocks: usize,
        head_hidden: usize,
    },
    /// Block network for boards made of identical rectangular blocks (e.g. the nine `3x3`
    /// sub-boards of Ultimate Tic-Tac-Toe; `1x1` blocks give one token per cell).
    ///
    /// Input: `in_planes` row-major `board_height x board_width` planes (`G == in_planes * H * W`);
    /// policy: one logit per cell, row-major (`P == H * W`).
    ///
    /// One small encoder with **shared weights** embeds each block (blocks that follow the same
    /// rules are the inductive bias that makes convolutions work on boards), a global residual
    /// MLP mixes the block embeddings, and a shared per-block head (local embedding + projected
    /// global context) emits the policy logits of each block's cells. Built from matmuls only, so
    /// it runs at MLP speed on every backend.
    BlockNet {
        in_planes: usize,
        board_height: usize,
        board_width: usize,
        block_height: usize,
        block_width: usize,
        local_dim: usize,
        hidden_dim: usize,
        num_res_blocks: usize,
        head_hidden: usize,
        /// Project the global state into one policy context per block instead of one shared by
        /// all blocks. With a shared context the policy head sees only a block's contents, not
        /// where the block sits, so two blocks with equal contents get equal logits even when one
        /// of them decides the board (e.g. a free move in Ultimate Tic-Tac-Toe). Per block, the
        /// projection's bias acts as a learned position embedding. Costs
        /// `(blocks - 1) * local_dim * (hidden_dim + 1)` parameters (+132k for UTTT's 10 x 256 net)
        /// and negligible compute. Files written before the field existed read as `false`.
        #[serde(default)]
        per_block_context: bool,
    },
}

impl AlphaZeroNetConfig {
    /// Generic default for games that do not choose an architecture: a residual MLP over the
    /// flat state encoding (works for any `G`, `P`, `N`). Game-specific architectures live in the
    /// game modules (`AlphaZeroGame::net_config`).
    pub const fn generic_mlp() -> Self {
        Self::ResMlp {
            hidden_dim: 256,
            num_res_blocks: 3,
            head_hidden: 128,
        }
    }

    /// Depth (residual blocks) and width (hidden size of the residual tower: `hidden_dim`, or
    /// `channels` for [`Self::ResNet2d`]).
    pub const fn size(&self) -> (usize, usize) {
        match *self {
            Self::ResNet2d { channels, num_res_blocks, .. } => (num_res_blocks, channels),
            Self::ResMlp { hidden_dim, num_res_blocks, .. } | Self::BlockNet { hidden_dim, num_res_blocks, .. } => {
                (num_res_blocks, hidden_dim)
            }
        }
    }

    /// This architecture with `blocks` residual blocks and a residual tower `width` wide (see
    /// [`Self::size`]); `0` keeps the current value. Heads and encoders keep their sizes.
    #[must_use]
    pub fn with_size(mut self, blocks: usize, width: usize) -> Self {
        match &mut self {
            Self::ResNet2d { channels, num_res_blocks, .. } => {
                if blocks > 0 {
                    *num_res_blocks = blocks;
                }
                if width > 0 {
                    *channels = width;
                }
            }
            Self::ResMlp { hidden_dim, num_res_blocks, .. } | Self::BlockNet { hidden_dim, num_res_blocks, .. } => {
                if blocks > 0 {
                    *num_res_blocks = blocks;
                }
                if width > 0 {
                    *hidden_dim = width;
                }
            }
        }
        self
    }
}

#[derive(Clone, Debug)]
struct ConvResidualBlock {
    conv1: Conv2d,
    norm1: GroupNorm,
    conv2: Conv2d,
    norm2: GroupNorm,
}

impl ConvResidualBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d_no_bias(channels, channels, 3, cfg, vb.pp("conv1"))?;
        let norm1 = group_norm(1, channels, 1e-5, vb.pp("norm1"))?;
        let conv2 = conv2d_no_bias(channels, channels, 3, cfg, vb.pp("conv2"))?;
        let norm2 = group_norm(1, channels, 1e-5, vb.pp("norm2"))?;
        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(xs)?;
        let h = self.norm1.forward(&h)?.relu()?;
        let h = self.conv2.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        (xs + h)?.relu()
    }
}

#[derive(Clone, Debug)]
struct ResNet2dTrunk {
    in_channels: usize,
    board_height: usize,
    board_width: usize,
    init_conv: Conv2d,
    init_norm: GroupNorm,
    blocks: Vec<ConvResidualBlock>,
    policy_conv: Conv2d,
    policy_norm: GroupNorm,
    policy_fc1: Linear,
    policy_fc2: Linear,
    value_conv: Conv2d,
    value_norm: GroupNorm,
    value_fc1: Linear,
    value_fc2: Linear,
}

#[derive(Clone, Debug)]
struct MlpResidualBlock {
    fc1: Linear,
    norm1: LayerNorm,
    fc2: Linear,
    norm2: LayerNorm,
}

impl MlpResidualBlock {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_dim, hidden_dim, vb.pp("fc1"))?;
        let norm1 = layer_norm(hidden_dim, 1e-5, vb.pp("norm1"))?;
        let fc2 = linear(hidden_dim, hidden_dim, vb.pp("fc2"))?;
        let norm2 = layer_norm(hidden_dim, 1e-5, vb.pp("norm2"))?;
        Ok(Self {
            fc1,
            norm1,
            fc2,
            norm2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(xs)?;
        let h = self.norm1.forward(&h)?.relu()?;
        let h = self.fc2.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        (xs + h)?.relu()
    }
}

#[derive(Clone, Debug)]
struct ResMlpTrunk {
    input_fc: Linear,
    input_norm: LayerNorm,
    blocks: Vec<MlpResidualBlock>,
    policy_fc1: Linear,
    policy_fc2: Linear,
    value_fc1: Linear,
    value_fc2: Linear,
}

#[derive(Clone, Debug)]
struct BlockTrunk {
    layout: BlockLayout,
    local_dim: usize,
    /// Policy contexts: 1 (shared by all blocks) or one per block.
    ctx_blocks: usize,
    local_fc1: Linear,
    local_norm1: LayerNorm,
    local_fc2: Linear,
    local_norm2: LayerNorm,
    global_fc: Linear,
    global_norm: LayerNorm,
    blocks: Vec<MlpResidualBlock>,
    policy_ctx: Linear,
    policy_fc1: Linear,
    policy_fc2: Linear,
    value_fc1: Linear,
    value_fc2: Linear,
}

#[derive(Clone, Debug)]
enum Trunk {
    ResNet2d(Box<ResNet2dTrunk>),
    ResMlp(ResMlpTrunk),
    Block(Box<BlockTrunk>),
}

/// Generic `N`-player AlphaZero Policy-Value Network with flattened state input size `G`,
/// policy logit output dimension `P`, and relative value vector output dimension `N`.
#[derive(Clone, Debug)]
pub struct AlphaZeroNet<const G: usize, const P: usize, const N: usize> {
    trunk: Trunk,
}

impl<const G: usize, const P: usize, const N: usize> AlphaZeroNet<G, P, N> {
    /// Constructs a new [`AlphaZeroNet`] initialized into `vb`.
    pub fn new(vb: VarBuilder, config: &AlphaZeroNetConfig) -> Result<Self> {
        match *config {
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
                assert_eq!(
                    in_channels * board_height * board_width,
                    G,
                    "ResNet2d dimensions C*H*W must equal G"
                );
                let pad1 = Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                };
                let pad0 = Conv2dConfig::default();

                let init_conv = conv2d_no_bias(in_channels, channels, 3, pad1, vb.pp("init_conv"))?;
                let init_norm = group_norm(1, channels, 1e-5, vb.pp("init_norm"))?;

                let mut blocks = Vec::with_capacity(num_res_blocks);
                for idx in 0..num_res_blocks {
                    blocks.push(ConvResidualBlock::new(
                        channels,
                        vb.pp(format!("res_{idx}")),
                    )?);
                }

                let spatial_area = board_height * board_width;
                let policy_conv = conv2d(
                    channels,
                    policy_channels,
                    1,
                    pad0,
                    vb.pp("policy_conv"),
                )?;
                let policy_norm = group_norm(1, policy_channels, 1e-5, vb.pp("policy_norm"))?;
                let policy_fc1 = linear(
                    policy_channels * spatial_area,
                    fc_hidden,
                    vb.pp("policy_fc1"),
                )?;
                let policy_fc2 = linear_zero(fc_hidden, P, vb.pp("policy_fc2"))?;

                let value_conv = conv2d(
                    channels,
                    value_channels,
                    1,
                    pad0,
                    vb.pp("value_conv"),
                )?;
                let value_norm = group_norm(1, value_channels, 1e-5, vb.pp("value_norm"))?;
                let value_fc1 = linear(
                    value_channels * spatial_area,
                    fc_hidden,
                    vb.pp("value_fc1"),
                )?;
                let value_fc2 = linear_zero(fc_hidden, N, vb.pp("value_fc2"))?;

                Ok(Self {
                    trunk: Trunk::ResNet2d(Box::new(ResNet2dTrunk {
                        in_channels,
                        board_height,
                        board_width,
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
                    })),
                })
            }
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
                per_block_context,
            } => {
                let layout = BlockLayout {
                    planes: in_planes,
                    rows: board_height / block_height,
                    cols: board_width / block_width,
                    bh: block_height,
                    bw: block_width,
                };
                assert!(
                    board_height % block_height == 0 && board_width % block_width == 0,
                    "BlockNet blocks must tile the board"
                );
                assert_eq!(in_planes * board_height * board_width, G, "BlockNet requires G == in_planes * H * W");
                assert_eq!(board_height * board_width, P, "BlockNet requires P == H * W");
                let (num_tokens, cells) = (layout.num_blocks(), layout.cells());
                let ctx_blocks = if per_block_context { num_tokens } else { 1 };
                let mut blocks = Vec::with_capacity(num_res_blocks);
                for idx in 0..num_res_blocks {
                    blocks.push(MlpResidualBlock::new(hidden_dim, vb.pp(format!("res_{idx}")))?);
                }
                Ok(Self {
                    trunk: Trunk::Block(Box::new(BlockTrunk {
                        layout,
                        local_dim,
                        ctx_blocks,
                        local_fc1: linear(in_planes * cells, local_dim, vb.pp("local_fc1"))?,
                        local_norm1: layer_norm(local_dim, 1e-5, vb.pp("local_norm1"))?,
                        local_fc2: linear(local_dim, local_dim, vb.pp("local_fc2"))?,
                        local_norm2: layer_norm(local_dim, 1e-5, vb.pp("local_norm2"))?,
                        global_fc: linear(num_tokens * local_dim, hidden_dim, vb.pp("global_fc"))?,
                        global_norm: layer_norm(hidden_dim, 1e-5, vb.pp("global_norm"))?,
                        blocks,
                        policy_ctx: linear(hidden_dim, ctx_blocks * local_dim, vb.pp("policy_ctx"))?,
                        policy_fc1: linear(local_dim, head_hidden, vb.pp("policy_fc1"))?,
                        policy_fc2: linear_zero(head_hidden, cells, vb.pp("policy_fc2"))?,
                        value_fc1: linear(hidden_dim, head_hidden, vb.pp("value_fc1"))?,
                        value_fc2: linear_zero(head_hidden, N, vb.pp("value_fc2"))?,
                    })),
                })
            }
            AlphaZeroNetConfig::ResMlp {
                hidden_dim,
                num_res_blocks,
                head_hidden,
            } => {
                let input_fc = linear(G, hidden_dim, vb.pp("input_fc"))?;
                let input_norm = layer_norm(hidden_dim, 1e-5, vb.pp("input_norm"))?;

                let mut blocks = Vec::with_capacity(num_res_blocks);
                for idx in 0..num_res_blocks {
                    blocks.push(MlpResidualBlock::new(
                        hidden_dim,
                        vb.pp(format!("res_{idx}")),
                    )?);
                }

                let policy_fc1 = linear(hidden_dim, head_hidden, vb.pp("policy_fc1"))?;
                let policy_fc2 = linear_zero(head_hidden, P, vb.pp("policy_fc2"))?;

                let value_fc1 = linear(hidden_dim, head_hidden, vb.pp("value_fc1"))?;
                let value_fc2 = linear_zero(head_hidden, N, vb.pp("value_fc2"))?;

                Ok(Self {
                    trunk: Trunk::ResMlp(ResMlpTrunk {
                        input_fc,
                        input_norm,
                        blocks,
                        policy_fc1,
                        policy_fc2,
                        value_fc1,
                        value_fc2,
                    }),
                })
            }
        }
    }

    /// Evaluates a batch of encoded states `xs` of shape `[B, G]`, returning
    /// `(policy_logits, values)` of shapes `([B, P], [B, N])` with `values` in `[-1.0, 1.0]`.
    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let b_sz = xs.dim(0)?;
        match &self.trunk {
            Trunk::ResNet2d(net) => {
                let x = xs.reshape((
                    b_sz,
                    net.in_channels,
                    net.board_height,
                    net.board_width,
                ))?;
                let mut h = net.init_conv.forward(&x)?;
                h = net.init_norm.forward(&h)?.relu()?;
                for block in &net.blocks {
                    h = block.forward(&h)?;
                }

                let p = net.policy_conv.forward(&h)?;
                let p = net.policy_norm.forward(&p)?.relu()?.flatten_from(1)?;
                let p = net.policy_fc1.forward(&p)?.relu()?;
                let policy_logits = net.policy_fc2.forward(&p)?;

                let v = net.value_conv.forward(&h)?;
                let v = net.value_norm.forward(&v)?.relu()?.flatten_from(1)?;
                let v = net.value_fc1.forward(&v)?.relu()?;
                let values = net.value_fc2.forward(&v)?.tanh()?;

                Ok((policy_logits, values))
            }
            Trunk::Block(net) => {
                let x = net.layout.tokens(xs)?;
                let t = net.local_fc1.forward(&x)?;
                let t = net.local_norm1.forward(&t)?.relu()?;
                let t2 = net.local_fc2.forward(&t)?;
                let t = (&t + net.local_norm2.forward(&t2)?)?.relu()?;

                let mut g = net
                    .global_fc
                    .forward(&t.reshape((b_sz, net.layout.num_blocks() * net.local_dim))?)?;
                g = net.global_norm.forward(&g)?.relu()?;
                for block in &net.blocks {
                    g = block.forward(&g)?;
                }

                let ctx = net.policy_ctx.forward(&g)?.reshape((b_sz, net.ctx_blocks, net.local_dim))?;
                let p = t.broadcast_add(&ctx)?.relu()?;
                let p = net.policy_fc1.forward(&p)?.relu()?;
                let policy_logits = net.layout.logits_to_grid(&net.policy_fc2.forward(&p)?)?;

                let v = net.value_fc1.forward(&g)?.relu()?;
                let values = net.value_fc2.forward(&v)?.tanh()?;
                Ok((policy_logits, values))
            }
            Trunk::ResMlp(net) => {
                let mut h = net.input_fc.forward(xs)?;
                h = net.input_norm.forward(&h)?.relu()?;
                for block in &net.blocks {
                    h = block.forward(&h)?;
                }

                let p = net.policy_fc1.forward(&h)?.relu()?;
                let policy_logits = net.policy_fc2.forward(&p)?;

                let v = net.value_fc1.forward(&h)?.relu()?;
                let values = net.value_fc2.forward(&v)?.tanh()?;

                Ok((policy_logits, values))
            }
        }
    }
}

/// Linear layer with all-zero initial weights and bias, used for the final policy and value layers:
/// an untrained network then outputs uniform priors and value `0` (a clean pure-search baseline)
/// instead of arbitrary, often tanh-saturated values whose vanishing gradients stall value learning.
fn linear_zero(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get_with_hints((out_dim, in_dim), "weight", candle_nn::Init::Const(0.0))?;
    let b = vb.get_with_hints(out_dim, "bias", candle_nn::Init::Const(0.0))?;
    Ok(Linear::new(w, Some(b)))
}

/// How a [`AlphaZeroNetConfig::BlockNet`] cuts row-major board planes into block tokens.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockLayout {
    pub planes: usize,
    /// Blocks per board column / row.
    pub rows: usize,
    pub cols: usize,
    /// Cells per block row / column.
    pub bh: usize,
    pub bw: usize,
}

impl BlockLayout {
    pub const fn num_blocks(&self) -> usize {
        self.rows * self.cols
    }

    pub const fn cells(&self) -> usize {
        self.bh * self.bw
    }

    /// `[B, planes * H * W]` -> `[B, blocks, planes * bh * bw]`: token `k = block_row * cols +
    /// block_col` holds that block's cells (`planes x bh x bw`).
    pub fn tokens(&self, xs: &Tensor) -> Result<Tensor> {
        let b = xs.dim(0)?;
        xs.reshape((b, self.planes, self.rows, self.bh, self.cols, self.bw))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, self.num_blocks(), self.planes * self.cells()))
    }

    /// `[B, blocks, bh * bw]` per-block logits -> `[B, H * W]` row-major board logits (inverse
    /// layout of [`Self::tokens`] for a single plane).
    pub fn logits_to_grid(&self, p: &Tensor) -> Result<Tensor> {
        let b = p.dim(0)?;
        p.reshape((b, self.rows, self.cols, self.bh, self.bw))?
            .permute((0, 1, 3, 2, 4))?
            .contiguous()?
            .reshape((b, self.num_blocks() * self.cells()))
    }
}

#[derive(Clone)]
struct WorkerReplica<const G: usize, const P: usize, const N: usize> {
    varmap: VarMap,
    net: AlphaZeroNet<G, P, N>,
    device: Device,
}

/// Multi-device [`BatchEvaluator`] wrapping one [`AlphaZeroNet`] replica per inference worker.
///
/// - Worker `0` shares `master_varmap` directly with [`crate::training::AlphaZeroTrainer`].
/// - Additional workers (`worker_id >= 1`, e.g., secondary CUDA GPUs on a multi-GPU RunPod machine)
///   hold device-local [`VarMap`] replicas that are synchronized via [`Self::sync_workers`].
#[derive(Clone)]
pub struct CandleEvaluator<const G: usize, const P: usize, const N: usize> {
    replicas: Box<[WorkerReplica<G, P, N>]>,
}

impl<const G: usize, const P: usize, const N: usize> CandleEvaluator<G, P, N> {
    /// Creates a new [`CandleEvaluator`] across `devices`, sharing `master_varmap` and
    /// `master_net` on `devices[0]` and cloning weights onto `devices[1..]` if multiple GPUs
    /// are provided.
    pub fn new(
        master_varmap: &VarMap,
        master_net: &AlphaZeroNet<G, P, N>,
        config: &AlphaZeroNetConfig,
        devices: &[Device],
    ) -> Result<Self> {
        assert!(!devices.is_empty(), "devices slice must not be empty");
        let _metal = metal_guard(&devices[0]);
        let mut replicas = Vec::with_capacity(devices.len());

        // Always bind Replica 0 to the exact `Device` (`DeviceId`) that `master_varmap`'s tensors
        // were allocated on so `evaluate_batch` can never hit a `DeviceId` mismatch against `master_net`.
        let master_device = master_varmap
            .all_vars()
            .first()
            .map(|v| v.device().clone())
            .unwrap_or_else(|| devices[0].clone());

        replicas.push(WorkerReplica {
            varmap: master_varmap.clone(),
            net: master_net.clone(),
            device: master_device,
        });

        for dev in &devices[1..] {
            let replica_varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&replica_varmap, DType::F32, dev);
            let replica_net = AlphaZeroNet::new(vb, config)?;
            replicas.push(WorkerReplica {
                varmap: replica_varmap,
                net: replica_net,
                device: dev.clone(),
            });
        }

        let eval = Self {
            replicas: replicas.into_boxed_slice(),
        };
        eval.sync_workers()?;
        Ok(eval)
    }

    /// Creates a standalone [`CandleEvaluator`] initialized on `devices`, optionally loading
    /// weights from a `.safetensors` file at `checkpoint_path`.
    pub fn standalone<Pth: AsRef<Path>>(
        config: &AlphaZeroNetConfig,
        devices: &[Device],
        checkpoint_path: Option<Pth>,
    ) -> Result<Self> {
        assert!(!devices.is_empty(), "devices slice must not be empty");
        let _metal = metal_guard(&devices[0]);
        let mut master_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&master_varmap, DType::F32, &devices[0]);
        let master_net = AlphaZeroNet::new(vb, config)?;
        if let Some(path) = checkpoint_path {
            master_varmap.load(path)?;
        }
        Self::new(&master_varmap, &master_net, config, devices)
    }

    #[inline]
    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }

    /// Synchronizes weights from Worker `0` (`master_varmap`) to all secondary GPU worker replicas
    /// in parallel.
    ///
    /// When `num_replicas() == 1` (single GPU or CPU), this is a zero-cost no-op because Worker `0`
    /// already shares `master_varmap`'s underlying `Var` storage in-place!
    pub fn sync_workers(&self) -> Result<()> {
        if self.replicas.len() <= 1 {
            return Ok(());
        }
        let _metal = metal_guard(&self.replicas[0].device);
        let master_guard = self.replicas[0]
            .varmap
            .data()
            .lock()
            .expect("master varmap mutex poisoned");

        std::thread::scope(|scope| {
            let master_ref = &*master_guard;
            let handles: Vec<_> = self.replicas[1..]
                .iter()
                .map(|replica| {
                    scope.spawn(move || -> Result<()> {
                        let replica_guard = replica
                            .varmap
                            .data()
                            .lock()
                            .expect("replica varmap mutex poisoned");
                        for (name, master_var) in master_ref.iter() {
                            if let Some(dst_var) = replica_guard.get(name) {
                                let transferred =
                                    master_var.as_tensor().to_device(&replica.device)?;
                                dst_var.set(&transferred)?;
                            }
                        }
                        Ok(())
                    })
                })
                .collect();

            for handle in handles {
                handle.join().expect("sync_workers thread panicked")?;
            }
            Ok(())
        })
    }

    /// Saves the model weights to `path` in `.safetensors` format.
    pub fn save_safetensors<Pth: AsRef<Path>>(&self, path: Pth) -> Result<()> {
        let _metal = metal_guard(&self.replicas[0].device);
        self.replicas[0].varmap.save(path)
    }

    /// Loads model weights from a `.safetensors` file at `path` and synchronizes all replicas.
    pub fn load_safetensors<Pth: AsRef<Path>>(&self, path: Pth) -> Result<()> {
        let _metal = metal_guard(&self.replicas[0].device);
        let mut master_varmap = self.replicas[0].varmap.clone();
        master_varmap.load(path)?;
        self.sync_workers()
    }

    /// Re-initializes the live model weights in-place to fresh random values and synchronizes all replicas.
    pub fn reset_random_weights(&self, net_config: &AlphaZeroNetConfig) -> Result<()> {
        let _metal = metal_guard(&self.replicas[0].device);
        let fresh_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&fresh_varmap, DType::F32, &self.replicas[0].device);
        let _fresh_net = AlphaZeroNet::<G, P, N>::new(vb, net_config)?;
        {
            let src_guard = fresh_varmap
                .data()
                .lock()
                .expect("fresh varmap mutex poisoned");
            let dst_guard = self.replicas[0]
                .varmap
                .data()
                .lock()
                .expect("master varmap mutex poisoned");
            for (name, src_var) in src_guard.iter() {
                if let Some(dst_var) = dst_guard.get(name) {
                    dst_var.set(src_var.as_tensor())?;
                }
            }
        }
        self.sync_workers()
    }
}

impl<const G: usize, const P: usize, const N: usize> BatchEvaluator<G, P, N>
    for CandleEvaluator<G, P, N>
{
    fn evaluate_batch(
        &self,
        worker_id: usize,
        states: &[[f32; G]],
        out_policies: &mut [[f32; P]],
        out_values: &mut [[f32; N]],
    ) {
        let b = states.len();
        if b == 0 {
            return;
        }
        let replica = &self.replicas[worker_id % self.replicas.len()];
        let _metal = metal_guard(&replica.device);

        let input = Tensor::from_slice(states.as_flattened(), (b, G), &replica.device)
            .expect("failed to construct input batch tensor");
        let (policy_tensor, value_tensor) = replica
            .net
            .forward(&input)
            .expect("AlphaZeroNet forward pass failed");

        let policy_vec: Vec<f32> = policy_tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("failed to extract policy logits from tensor");
        let value_vec: Vec<f32> = value_tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("failed to extract value predictions from tensor");

        out_policies[..b]
            .as_flattened_mut()
            .copy_from_slice(&policy_vec[..(b * P)]);
        out_values[..b]
            .as_flattened_mut()
            .copy_from_slice(&value_vec[..(b * N)]);
    }
}
