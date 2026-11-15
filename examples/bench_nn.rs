//! Neural-network throughput benchmark for the candle backend (counterpart of
//! `experiments/torch_bench.py`, which runs the identical networks on PyTorch/libtorch).
//!
//! Measures, per network and device:
//! - inference: `evaluate_batch` latency incl. host->device upload and readback (what self-play
//!   pays per batched NN call), for several batch sizes;
//! - training: one forward + backward + AdamW step (`train_step_flat`).
//!
//! Run:
//! - CPU only:           `cargo run --release --example bench_nn`
//! - CPU + Apple GPU:    `cargo run --release --features metal --example bench_nn`
//! - CPU + NVIDIA GPU:   `cargo run --release --features cuda --example bench_nn`
//!
//! Options (`key=value`): `nets=resnet6,resnet4,block,mlp` `devices=cpu,gpu` `backends=candle,tch`
//! `secs=1.5`. The tch backend needs `--features tch` (see `start_lab.sh` for the libtorch setup).

use std::time::Instant;

use mcts::backend::{Backend, NnDevice};
use mcts::games::ultimate_ttt::{self, INPUT_PLANES, INPUT_SIZE};
use mcts::inference::BatchEvaluator;
use mcts::nn::AlphaZeroNetConfig;
use mcts::training::AlphaZeroTrainer;
use rand::prelude::{RngExt, SeedableRng, SmallRng};

const P: usize = 81;
const N: usize = 2;

fn net_by_name(name: &str) -> AlphaZeroNetConfig {
    match name {
        // The network of the C++/libtorch predecessor (beat-it `uttt_config.json`).
        "resnet6" => AlphaZeroNetConfig::ResNet2d {
            in_channels: INPUT_PLANES,
            board_height: 9,
            board_width: 9,
            channels: 64,
            num_res_blocks: 6,
            policy_channels: 32,
            value_channels: 4,
            fc_hidden: 256,
        },
        "resnet4" => ultimate_ttt::net_resnet(),
        "block" => ultimate_ttt::net_block(),
        "mlp" => ultimate_ttt::net_mlp(),
        other => panic!("unknown net {other}"),
    }
}

/// Runs `f` repeatedly for about `secs` seconds after warm-up and returns the mean time in ms.
fn time_ms(secs: f64, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    let mut n = 0usize;
    while start.elapsed().as_secs_f64() < secs || n < 3 {
        f();
        n += 1;
    }
    start.elapsed().as_secs_f64() * 1000.0 / n as f64
}

fn bench(net_name: &str, device: NnDevice, secs: f64) -> candle_core::Result<()> {
    let cfg = net_by_name(net_name);
    let mut trainer = AlphaZeroTrainer::<INPUT_SIZE, P, N>::new(cfg, 1e-3, 1e-4, device)?;
    let service = trainer.build_inference_service(1024, &[device])?;
    let eval = service.evaluator();
    let mut rng = SmallRng::seed_from_u64(7);
    let mut line = format!("{:<16} {net_name:<8}", device.to_string());

    for b in [64usize, 256, 1024] {
        let states: Vec<[f32; INPUT_SIZE]> = (0..b)
            .map(|_| std::array::from_fn(|_| if rng.random_bool(0.2) { 1.0 } else { 0.0 }))
            .collect();
        let mut pol = vec![[0.0f32; P]; b];
        let mut val = vec![[0.0f32; N]; b];
        let ms = time_ms(secs, || eval.evaluate_batch(0, &states, &mut pol, &mut val));
        line += &format!(" | infer b{b:<4} {ms:>7.2} ms ({:>7.0} pos/s)", b as f64 / ms * 1000.0);
    }
    for b in [256usize, 1024] {
        let states: Vec<f32> =
            (0..b * INPUT_SIZE).map(|_| if rng.random_bool(0.2) { 1.0 } else { 0.0 }).collect();
        let policies: Vec<f32> = (0..b * P).map(|_| 1.0 / P as f32).collect();
        let values: Vec<f32> = (0..b * N).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let ms = time_ms(secs, || {
            trainer.train_step_flat(&states, &policies, &values, b).expect("train step");
        });
        line += &format!(" | train b{b:<4} {ms:>7.2} ms");
    }
    println!("{line}");
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |key: &str, default: &str| -> String {
        args.iter()
            .find_map(|a| a.strip_prefix(&format!("{key}=")).map(str::to_string))
            .unwrap_or_else(|| default.to_string())
    };
    let nets = arg("nets", "resnet6,resnet4,block,mlp");
    let devices = arg("devices", "cpu,gpu");
    let secs: f64 = arg("secs", "1.5").parse().expect("secs");

    let backends = arg("backends", &Backend::available().iter().map(|b| b.as_str()).collect::<Vec<_>>().join(","));
    for backend in backends.split(',').map(|b| Backend::parse(b).expect("backend")) {
        if !backend.is_available() {
            println!("({backend} not compiled in — build with `--features tch`)");
            continue;
        }
        for dev_name in devices.split(',') {
            let device = match dev_name {
                "cpu" => NnDevice::cpu(backend),
                "gpu" if backend.gpu_count() > 0 => NnDevice::gpu(backend, 0),
                "gpu" => {
                    println!("({backend}: no GPU available — skipping gpu)");
                    continue;
                }
                other => panic!("unknown device {other}"),
            };
            for net in nets.split(',') {
                bench(net, device, secs)?;
            }
        }
    }
    Ok(())
}
