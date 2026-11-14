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

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, GroupNorm, LayerNorm, Linear, VarBuilder, VarMap, conv2d, conv2d_no_bias,
    group_norm, layer_norm, linear,
};

use crate::inference::BatchEvaluator;

/// Selects the best single accelerator device available on the current machine
/// (`CUDA(0)` -> `Metal(0)` -> `Cpu`).
pub fn default_device() -> Device {
    if let Ok(dev) = Device::new_cuda(0) {
        return dev;
    }
    if let Ok(dev) = Device::new_metal(0) {
        return dev;
    }
    Device::Cpu
}

/// Discovers up to `max_workers` inference devices for [`crate::inference::InferenceService`].
///
/// - On a multi-GPU RunPod node (`--features cuda`), binds `worker_id` `0..K` to `CUDA(0)..CUDA(K-1)`.
/// - On Apple Silicon (`--features metal`), binds to `Metal(0)`.
/// - Otherwise falls back to `Device::Cpu`.
pub fn discover_devices(max_workers: usize) -> Vec<Device> {
    let limit = max_workers.max(1);
    let mut cuda_devices = Vec::new();
    for idx in 0..limit {
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

/// Architecture configuration for [`AlphaZeroNet`].
#[derive(Clone, Copy, Debug)]
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
}

impl AlphaZeroNetConfig {
    /// Compact 2D ResNet configuration for `3x3` Tic-Tac-Toe (`G = 27, P = 9, N = 2`).
    pub const fn tic_tac_toe_resnet() -> Self {
        Self::ResNet2d {
            in_channels: 3,
            board_height: 3,
            board_width: 3,
            channels: 32,
            num_res_blocks: 2,
            policy_channels: 8,
            value_channels: 4,
            fc_hidden: 64,
        }
    }

    /// Fast Residual MLP configuration for `3x3` Tic-Tac-Toe (`G = 27, P = 9, N = 2`).
    pub const fn tic_tac_toe_mlp() -> Self {
        Self::ResMlp {
            hidden_dim: 64,
            num_res_blocks: 2,
            head_hidden: 64,
        }
    }

    /// 2D ResNet configuration for `9x9` Ultimate Tic-Tac-Toe (`G = 324, P = 81, N = 2`).
    pub const fn ultimate_ttt_resnet() -> Self {
        Self::ResNet2d {
            in_channels: 4,
            board_height: 9,
            board_width: 9,
            channels: 64,
            num_res_blocks: 4,
            policy_channels: 8,
            value_channels: 4,
            fc_hidden: 128,
        }
    }

    /// Residual MLP configuration for `N = 1` 3x3x3 Rubik's Cube (`G = 324, P = 12, N = 1`).
    pub const fn rubiks_cube_mlp() -> Self {
        Self::ResMlp {
            hidden_dim: 256,
            num_res_blocks: 4,
            head_hidden: 128,
        }
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
enum Trunk {
    ResNet2d(Box<ResNet2dTrunk>),
    ResMlp(ResMlpTrunk),
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
                let policy_fc2 = linear(fc_hidden, P, vb.pp("policy_fc2"))?;

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
                let value_fc2 = linear(fc_hidden, N, vb.pp("value_fc2"))?;

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
                let policy_fc2 = linear(head_hidden, P, vb.pp("policy_fc2"))?;

                let value_fc1 = linear(hidden_dim, head_hidden, vb.pp("value_fc1"))?;
                let value_fc2 = linear(head_hidden, N, vb.pp("value_fc2"))?;

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
pub struct CandleBatchEvaluator<const G: usize, const P: usize, const N: usize> {
    replicas: Box<[WorkerReplica<G, P, N>]>,
}

impl<const G: usize, const P: usize, const N: usize> CandleBatchEvaluator<G, P, N> {
    /// Creates a new [`CandleBatchEvaluator`] across `devices`, sharing `master_varmap` and
    /// `master_net` on `devices[0]` and cloning weights onto `devices[1..]` if multiple GPUs
    /// are provided.
    pub fn new(
        master_varmap: &VarMap,
        master_net: &AlphaZeroNet<G, P, N>,
        config: &AlphaZeroNetConfig,
        devices: &[Device],
    ) -> Result<Self> {
        assert!(!devices.is_empty(), "devices slice must not be empty");
        let mut replicas = Vec::with_capacity(devices.len());

        replicas.push(WorkerReplica {
            varmap: master_varmap.clone(),
            net: master_net.clone(),
            device: devices[0].clone(),
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
        self.replicas[0].varmap.save(path)
    }

    /// Loads model weights from a `.safetensors` file at `path` and synchronizes all replicas.
    pub fn load_safetensors<Pth: AsRef<Path>>(&self, path: Pth) -> Result<()> {
        let mut master_varmap = self.replicas[0].varmap.clone();
        master_varmap.load(path)?;
        self.sync_workers()
    }
}

impl<const G: usize, const P: usize, const N: usize> BatchEvaluator<G, P, N>
    for CandleBatchEvaluator<G, P, N>
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
