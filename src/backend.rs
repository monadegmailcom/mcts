//! Neural-network backend selection: **candle** (pure Rust, always available) or **libtorch**
//! via `tch-rs` (cargo feature `tch`). Both run the same [`AlphaZeroNetConfig`] architectures
//! with identical parameter names, so `.safetensors` checkpoints work with either backend.
//!
//! The backend is a *runtime* choice carried by [`NnDevice`]: [`crate::training::AlphaZeroTrainer`]
//! and [`NetEvaluator`] dispatch to the selected implementation, so the Browser Lab can switch
//! backends (and compare them) without recompiling.

use std::fmt;
use std::path::Path;

use candle_core::{Device, Result};

use crate::inference::BatchEvaluator;
use crate::nn::{AlphaZeroNetConfig, CandleEvaluator, accelerator_devices};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// candle (pure Rust; CPU, Apple Metal, CUDA).
    Candle,
    /// libtorch via tch-rs (CPU, Apple MPS, CUDA). Requires the `tch` cargo feature.
    Tch,
}

impl Backend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Tch => "tch",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "candle" => Some(Self::Candle),
            "tch" | "libtorch" | "torch" => Some(Self::Tch),
            _ => None,
        }
    }

    /// Whether this backend was compiled in.
    pub const fn is_available(self) -> bool {
        match self {
            Self::Candle => true,
            Self::Tch => cfg!(feature = "tch"),
        }
    }

    /// Backends compiled into this binary.
    pub fn available() -> Vec<Self> {
        [Self::Candle, Self::Tch].into_iter().filter(|b| b.is_available()).collect()
    }

    /// Number of GPUs this backend can use (0 = CPU only).
    pub fn gpu_count(self) -> usize {
        match self {
            Self::Candle => accelerator_devices().len(),
            #[cfg(feature = "tch")]
            Self::Tch => crate::nn_tch::gpu_count(),
            #[cfg(not(feature = "tch"))]
            Self::Tch => 0,
        }
    }

    /// Human-readable accelerator name (`"Metal"`, `"MPS"`, `"2 × CUDA"`), if any.
    pub fn gpu_name(self) -> Option<String> {
        match self {
            Self::Candle => {
                let gpus = accelerator_devices();
                match gpus.first() {
                    None => None,
                    Some(d) if d.is_metal() => Some("Metal".to_string()),
                    Some(_) => Some(format!("{} × CUDA", gpus.len())),
                }
            }
            #[cfg(feature = "tch")]
            Self::Tch => crate::nn_tch::gpu_name(),
            #[cfg(not(feature = "tch"))]
            Self::Tch => None,
        }
    }

    /// Whether several model replicas on this backend's GPUs actually run in parallel
    /// (false for Apple GPUs: one device, serialized command submission).
    pub fn parallel_gpus(self) -> bool {
        self.gpu_name().is_some_and(|n| n.contains("CUDA"))
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A backend plus a device on it (`gpu: None` = CPU, `Some(i)` = the backend's GPU `i`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NnDevice {
    pub backend: Backend,
    pub gpu: Option<usize>,
}

impl NnDevice {
    pub const CPU: Self = Self { backend: Backend::Candle, gpu: None };
    pub const CPU_TCH: Self = Self { backend: Backend::Tch, gpu: None };

    pub const fn cpu(backend: Backend) -> Self {
        Self { backend, gpu: None }
    }

    pub const fn gpu(backend: Backend, index: usize) -> Self {
        Self { backend, gpu: Some(index) }
    }

    /// GPU 0 of `backend` if it has one, else its CPU.
    pub fn best(backend: Backend) -> Self {
        if backend.gpu_count() > 0 { Self::gpu(backend, 0) } else { Self::cpu(backend) }
    }

    /// Training device plus `workers` inference replica devices: GPUs round-robin for CUDA,
    /// a single replica on Apple GPUs, `workers` CPU replicas on CPU.
    pub fn replicas(self, workers: usize) -> Vec<Self> {
        let workers = workers.max(1);
        match self.gpu {
            None => vec![self; workers],
            Some(_) if !self.backend.parallel_gpus() => vec![self],
            Some(_) => {
                let n = self.backend.gpu_count().max(1);
                (0..workers.min(n)).map(|i| Self::gpu(self.backend, i)).collect()
            }
        }
    }

    /// The matching candle device (cached, so all users share one `DeviceId`).
    pub fn candle_device(self) -> Device {
        match self.gpu {
            Some(i) => accelerator_devices().get(i).cloned().unwrap_or(Device::Cpu),
            None => Device::Cpu,
        }
    }

    pub fn is_gpu(self) -> bool {
        self.gpu.is_some() && self.backend.gpu_count() > 0
    }
}

impl fmt::Display for NnDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.gpu, self.backend.gpu_name()) {
            (Some(i), Some(name)) if name.contains("CUDA") => write!(f, "{} CUDA:{i}", self.backend),
            (Some(_), Some(name)) => write!(f, "{} {name}", self.backend),
            _ => write!(f, "{} CPU", self.backend),
        }
    }
}

/// Batched NN evaluator over any backend (one model replica per inference worker). Clones share
/// the same weights.
#[derive(Clone)]
pub enum NetEvaluator<const G: usize, const P: usize, const N: usize> {
    Candle(CandleEvaluator<G, P, N>),
    #[cfg(feature = "tch")]
    Tch(crate::nn_tch::TchEvaluator),
}

impl<const G: usize, const P: usize, const N: usize> NetEvaluator<G, P, N> {
    /// A fresh evaluator on `devices` (backend of `devices[0]`), optionally loading weights.
    pub fn standalone<Pth: AsRef<Path>>(
        config: &AlphaZeroNetConfig,
        devices: &[NnDevice],
        checkpoint_path: Option<Pth>,
    ) -> Result<Self> {
        assert!(!devices.is_empty(), "devices slice must not be empty");
        match devices[0].backend {
            Backend::Candle => {
                let devs: Vec<Device> = devices.iter().map(|d| d.candle_device()).collect();
                Ok(Self::Candle(CandleEvaluator::standalone(config, &devs, checkpoint_path)?))
            }
            #[cfg(feature = "tch")]
            Backend::Tch => Ok(Self::Tch(crate::nn_tch::TchEvaluator::standalone(
                config,
                devices,
                (G, P, N),
                checkpoint_path.as_ref().map(|p| p.as_ref()),
            )?)),
            #[cfg(not(feature = "tch"))]
            Backend::Tch => Err(candle_core::Error::Msg("built without the `tch` feature".into())),
        }
    }

    pub fn num_replicas(&self) -> usize {
        match self {
            Self::Candle(e) => e.num_replicas(),
            #[cfg(feature = "tch")]
            Self::Tch(e) => e.num_replicas(),
        }
    }

    /// Copies replica 0's weights to all other replicas (no-op with a single replica).
    pub fn sync_workers(&self) -> Result<()> {
        match self {
            Self::Candle(e) => e.sync_workers(),
            #[cfg(feature = "tch")]
            Self::Tch(e) => e.sync_workers(),
        }
    }

    pub fn save_safetensors<Pth: AsRef<Path>>(&self, path: Pth) -> Result<()> {
        match self {
            Self::Candle(e) => e.save_safetensors(path),
            #[cfg(feature = "tch")]
            Self::Tch(e) => e.save_safetensors(path.as_ref()),
        }
    }

    pub fn load_safetensors<Pth: AsRef<Path>>(&self, path: Pth) -> Result<()> {
        match self {
            Self::Candle(e) => e.load_safetensors(path),
            #[cfg(feature = "tch")]
            Self::Tch(e) => e.load_safetensors(path.as_ref()),
        }
    }

    pub fn reset_random_weights(&self, net_config: &AlphaZeroNetConfig) -> Result<()> {
        match self {
            Self::Candle(e) => e.reset_random_weights(net_config),
            #[cfg(feature = "tch")]
            Self::Tch(e) => e.reset_random_weights(),
        }
    }
}

impl<const G: usize, const P: usize, const N: usize> BatchEvaluator<G, P, N> for NetEvaluator<G, P, N> {
    fn evaluate_batch(
        &self,
        worker_id: usize,
        states: &[[f32; G]],
        out_policies: &mut [[f32; P]],
        out_values: &mut [[f32; N]],
    ) {
        match self {
            Self::Candle(e) => e.evaluate_batch(worker_id, states, out_policies, out_values),
            #[cfg(feature = "tch")]
            Self::Tch(e) => {
                let b = states.len();
                if b > 0 {
                    e.evaluate_flat(
                        worker_id,
                        states.as_flattened(),
                        b,
                        out_policies[..b].as_flattened_mut(),
                        out_values[..b].as_flattened_mut(),
                    );
                }
            }
        }
    }
}
