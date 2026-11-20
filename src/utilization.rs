//! CPU and GPU utilization of the lab, sampled once a second for the Browser Lab.
//!
//! - CPU: CPU time of this process (all threads) per wall-clock second, as % of all cores.
//! - GPU: the busy % the operating system reports for the whole device (macOS: `ioreg`'s
//!   IOAccelerator "Device Utilization %"; elsewhere: `nvidia-smi`, averaged over all GPUs), so
//!   other programs using the GPU count too. `None` where neither is available.
//!
//! Each sample is tagged with what the lab was doing (self-play, training, evaluation), so the
//! averages per activity show which part of an iteration leaves CPU or GPU idle.

use std::collections::VecDeque;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

/// Samples kept (one per second).
const HISTORY: usize = 300;

#[derive(Clone, Debug, Serialize)]
pub struct Sample {
    /// Seconds since the monitor started.
    pub t: f64,
    pub cpu: f32,
    pub gpu: Option<f32>,
    pub activity: &'static str,
}

pub struct UtilizationMonitor {
    cores: usize,
    gpu_source: Option<&'static str>,
    samples: Mutex<VecDeque<Sample>>,
}

impl UtilizationMonitor {
    /// Starts sampling once a second; `activity` names what the lab is doing at that moment.
    pub fn start(activity: impl Fn() -> &'static str + Send + 'static) -> Arc<Self> {
        let gpu_source = if gpu_percent(GPU_SOURCES[0]).is_some() {
            Some(GPU_SOURCES[0])
        } else {
            GPU_SOURCES[1..].iter().copied().find(|&s| gpu_percent(s).is_some())
        };
        let monitor = Arc::new(Self {
            cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
            gpu_source,
            samples: Mutex::new(VecDeque::with_capacity(HISTORY)),
        });
        let m = Arc::clone(&monitor);
        std::thread::spawn(move || {
            let start = Instant::now();
            let (mut last_wall, mut last_cpu) = (Instant::now(), process_cpu_seconds());
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let (wall, cpu) = (Instant::now(), process_cpu_seconds());
                let busy = (cpu - last_cpu) / wall.duration_since(last_wall).as_secs_f64() / m.cores as f64;
                (last_wall, last_cpu) = (wall, cpu);
                let sample = Sample {
                    t: start.elapsed().as_secs_f64(),
                    cpu: (100.0 * busy).clamp(0.0, 100.0) as f32,
                    gpu: m.gpu_source.and_then(gpu_percent),
                    activity: activity(),
                };
                let mut samples = m.samples.lock().expect("utilization samples poisoned");
                if samples.len() == HISTORY {
                    samples.pop_front();
                }
                samples.push_back(sample);
            }
        });
        monitor
    }

    pub fn to_json(&self) -> serde_json::Value {
        let samples = self.samples.lock().expect("utilization samples poisoned");
        serde_json::json!({
            "cores": self.cores,
            "gpu_source": self.gpu_source,
            "samples": *samples,
        })
    }
}

/// CPU time (user + system) of this process so far, in seconds.
fn process_cpu_seconds() -> f64 {
    // SAFETY: `getrusage` only writes into the zeroed struct we pass.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    };
    let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    secs(usage.ru_utime) + secs(usage.ru_stime)
}

const GPU_SOURCES: [&str; 2] = ["ioreg", "nvidia-smi"];

/// Device busy % from `source`, or `None` if it is unavailable.
fn gpu_percent(source: &str) -> Option<f32> {
    match source {
        "ioreg" => {
            let out = Command::new("ioreg").args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]).output().ok()?;
            let text = String::from_utf8_lossy(&out.stdout);
            let key = "\"Device Utilization %\"=";
            let rest = &text[text.find(key)? + key.len()..];
            rest[..rest.find(|c: char| !c.is_ascii_digit())?].parse().ok()
        }
        "nvidia-smi" => {
            let out = Command::new("nvidia-smi")
                .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
                .output()
                .ok()?;
            let values: Vec<f32> =
                String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse().ok()).collect();
            (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_cpu_time_grows_with_work() {
        let before = process_cpu_seconds();
        let mut x = 0u64;
        for i in 0..50_000_000u64 {
            x = x.wrapping_mul(31).wrapping_add(i);
        }
        std::hint::black_box(x);
        assert!(process_cpu_seconds() > before);
    }
}
