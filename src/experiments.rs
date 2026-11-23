//! Browser-Lab **experiments**: queued, comparable training runs.
//!
//! Each run trains a fresh network from scratch (own trainer, replay buffer and inference
//! service) for a training-time budget and evaluates it periodically with a *fixed* protocol
//! (same AlphaZero simulations, openings and opponents for every run), so the strength curves of
//! different architectures, backends or hyperparameters are directly comparable — per minute of
//! training and per self-play game. Runs execute one at a time on the training thread; results
//! are persisted to a JSON file so they survive lab restarts.
//!
//! This module holds the data model, queue and persistence; the training itself is driven by the
//! lab binary (`examples/train_alphazero.rs`) via [`crate::telemetry::TrainingCommand::RunExperiment`].

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::hyperparams::AlphaZeroHyperparams;
use crate::opponents::OpponentSpec;

/// What to train and how to evaluate it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentSpec {
    pub name: String,
    /// `"ttt"` or `"uttt"`.
    pub game: String,
    /// Network preset name (see `AlphaZeroGame::net_presets`).
    pub net: String,
    /// `"candle"` or `"tch"`.
    pub backend: String,
    /// `"gpu"` or `"cpu"`.
    pub device: String,
    pub hp: AlphaZeroHyperparams,
    /// Training-time budget in minutes (self-play + SGD; evaluation time is not counted).
    pub budget_minutes: f64,
    /// Hard cap on iterations (`0` = no cap).
    pub max_iterations: usize,
    /// Evaluate every this many iterations (plus at iteration 0 and at the end).
    pub eval_every: usize,
    /// AlphaZero simulations per move in evaluation games (identical for all runs).
    pub eval_sims: usize,
    /// Games per opponent and evaluation.
    pub eval_games: usize,
    /// Maximum positions per batched NN inference call (larger batches use GPUs much better).
    pub infer_batch: usize,
    /// Inference workers, one model replica each, as for live training (CUDA: spread over the
    /// GPUs, several per GPU allowed; Apple GPUs: always 1).
    pub infer_workers: usize,
    /// Evaluation opponents (empty = the game's defaults).
    pub opponents: Vec<OpponentSpec>,
    /// Checkpoint to start from (empty = fresh random weights). Must match the `net` architecture.
    pub init_checkpoint: String,
}

impl Default for ExperimentSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            game: "uttt".into(),
            net: "default".into(),
            backend: "candle".into(),
            device: "gpu".into(),
            hp: AlphaZeroHyperparams::GENERIC,
            budget_minutes: 20.0,
            max_iterations: 0,
            eval_every: 5,
            eval_sims: 100,
            eval_games: 48,
            infer_batch: 256,
            infer_workers: 1,
            opponents: Vec::new(),
            init_checkpoint: String::new(),
        }
    }
}

/// One evaluation of a run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExperimentPoint {
    pub iteration: usize,
    pub games: usize,
    pub positions: usize,
    /// Cumulative training time (self-play + SGD) in seconds.
    pub train_secs: f64,
    /// Cumulative wall-clock time including evaluations, in seconds.
    pub wall_secs: f64,
    /// Score `(wins + draws / 2) / games` in percent vs each of the run's `opponents`.
    pub scores: Vec<f32>,
    /// `[wins, draws, losses]` vs each of the run's `opponents` (empty in runs recorded before
    /// these were kept).
    #[serde(default)]
    pub wdl: Vec<[usize; 3]>,
    pub value_loss: f32,
    pub policy_kl: f32,
    pub sims_per_sec: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "message")]
pub enum RunStatus {
    Queued,
    Running,
    Done,
    Stopped,
    Failed(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentRun {
    pub id: u64,
    pub spec: ExperimentSpec,
    pub status: RunStatus,
    /// Labels of the evaluation opponents (order of `ExperimentPoint::scores`).
    #[serde(default)]
    pub opponents: Vec<String>,
    #[serde(default)]
    pub device_label: String,
    #[serde(default)]
    pub points: Vec<ExperimentPoint>,
    #[serde(default)]
    pub created_unix: u64,
    /// Current activity of a running run (not persisted).
    #[serde(skip)]
    pub phase: String,
}

#[derive(Serialize, Deserialize, Default)]
struct StoreFile {
    next_id: u64,
    runs: Vec<ExperimentRun>,
}

/// Queue + results of all experiments, persisted to `path`.
pub struct ExperimentStore {
    path: Option<PathBuf>,
    next_id: u64,
    runs: Vec<ExperimentRun>,
    queue_running: bool,
    stop_requested: Option<u64>,
}

impl ExperimentStore {
    /// An empty store that is not persisted.
    pub fn in_memory() -> Self {
        Self { path: None, next_id: 1, runs: Vec::new(), queue_running: false, stop_requested: None }
    }

    /// Loads `path` if it exists; a run that was running when the lab stopped is marked stopped.
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let file: StoreFile = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let mut runs = file.runs;
        for r in &mut runs {
            if r.status == RunStatus::Running {
                r.status = RunStatus::Stopped;
            }
        }
        let next_id = file.next_id.max(runs.iter().map(|r| r.id + 1).max().unwrap_or(1));
        Self { path: Some(path), next_id, runs, queue_running: false, stop_requested: None }
    }

    fn save(&self) {
        let Some(path) = &self.path else { return };
        let file = StoreFile { next_id: self.next_id, runs: self.runs.clone() };
        if let Ok(json) = serde_json::to_string_pretty(&file) {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, json).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
    }

    fn run_mut(&mut self, id: u64) -> Option<&mut ExperimentRun> {
        self.runs.iter_mut().find(|r| r.id == id)
    }

    pub fn add(&mut self, mut spec: ExperimentSpec) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        if spec.name.trim().is_empty() {
            spec.name = format!("{}-{}-{}-{} #{id}", spec.game, spec.net, spec.backend, spec.device);
        }
        spec.eval_every = spec.eval_every.max(1);
        spec.eval_games = spec.eval_games.max(2);
        spec.eval_sims = spec.eval_sims.max(1);
        spec.infer_batch = spec.infer_batch.clamp(1, 4096);
        spec.hp.fpu = spec.hp.fpu.sanitized();
        let created_unix = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        self.runs.push(ExperimentRun {
            id,
            spec,
            status: RunStatus::Queued,
            opponents: Vec::new(),
            device_label: String::new(),
            points: Vec::new(),
            created_unix,
            phase: String::new(),
        });
        self.save();
        id
    }

    /// Deletes a run (a running run is stopped first by the caller via [`Self::request_stop`]).
    pub fn remove(&mut self, id: u64) {
        self.runs.retain(|r| r.id != id || r.status == RunStatus::Running);
        self.save();
    }

    /// Puts a finished run back into the queue with its results cleared.
    pub fn requeue(&mut self, id: u64) {
        if let Some(r) = self.run_mut(id).filter(|r| r.status != RunStatus::Running) {
            r.status = RunStatus::Queued;
            r.points.clear();
            r.phase.clear();
        }
        self.save();
    }

    pub fn set_queue_running(&mut self, running: bool) {
        self.queue_running = running;
    }

    pub const fn queue_running(&self) -> bool {
        self.queue_running
    }

    pub fn is_running(&self) -> bool {
        self.runs.iter().any(|r| r.status == RunStatus::Running)
    }

    /// The next queued run to start, if the queue is running and nothing else runs.
    pub fn next_queued(&self) -> Option<(u64, ExperimentSpec)> {
        if !self.queue_running || self.is_running() {
            return None;
        }
        self.runs.iter().find(|r| r.status == RunStatus::Queued).map(|r| (r.id, r.spec.clone()))
    }

    pub fn start(&mut self, id: u64, opponents: Vec<String>, device_label: String) {
        if let Some(r) = self.run_mut(id) {
            r.status = RunStatus::Running;
            r.opponents = opponents;
            r.device_label = device_label;
            r.points.clear();
        }
        self.stop_requested = None;
        self.save();
    }

    pub fn set_phase(&mut self, id: u64, phase: String) {
        if let Some(r) = self.run_mut(id) {
            r.phase = phase;
        }
    }

    pub fn push_point(&mut self, id: u64, point: ExperimentPoint) {
        if let Some(r) = self.run_mut(id) {
            r.points.push(point);
        }
        self.save();
    }

    pub fn finish(&mut self, id: u64, status: RunStatus) {
        if let Some(r) = self.run_mut(id) {
            r.status = status;
            r.phase.clear();
        }
        if self.stop_requested == Some(id) {
            self.stop_requested = None;
        }
        if !self.runs.iter().any(|r| r.status == RunStatus::Queued) {
            self.queue_running = false;
        }
        self.save();
    }

    pub fn request_stop(&mut self, id: u64) {
        self.stop_requested = Some(id);
    }

    pub fn should_stop(&self, id: u64) -> bool {
        self.stop_requested == Some(id)
    }

    /// `{"queue_running":..,"runs":[..]}` (runs include the transient `phase`).
    pub fn to_json(&self) -> String {
        let runs: Vec<serde_json::Value> = self
            .runs
            .iter()
            .map(|r| {
                let mut v = json_value(r);
                v["phase"] = serde_json::Value::String(r.phase.clone());
                v
            })
            .collect();
        serde_json::json!({ "queue_running": self.queue_running, "runs": runs }).to_string()
    }
}

/// Score in percent: `(wins + draws / 2) / games` of participant 0.
pub fn score_pct(wins: usize, draws: usize, games: usize) -> f32 {
    if games == 0 { 0.0 } else { (wins as f32 + 0.5 * draws as f32) / games as f32 * 100.0 }
}

/// Like `serde_json::to_value`, but `f32` fields keep their short decimal form (`0.3`, not the
/// widened `0.30000001192092896`), so hyperparameters read cleanly in the browser.
pub(crate) fn json_value<T: Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_string(value).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_roundtrip_queue_and_recovery() {
        let path = std::env::temp_dir().join(format!("mcts_exp_test_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut store = ExperimentStore::load(&path);
        let a = store.add(ExperimentSpec { name: "a".into(), ..Default::default() });
        let b = store.add(ExperimentSpec::default());
        assert!(store.next_queued().is_none(), "queue not running yet");
        store.set_queue_running(true);
        assert_eq!(store.next_queued().unwrap().0, a);
        store.start(a, vec!["MCTS".into()], "dev".into());
        assert!(store.next_queued().is_none(), "one run at a time");
        store.push_point(a, ExperimentPoint { iteration: 1, scores: vec![50.0], ..Default::default() });

        // A crash while running: reloading marks the run stopped and keeps its points.
        let reloaded = ExperimentStore::load(&path);
        let ra = reloaded.runs.iter().find(|r| r.id == a).unwrap();
        assert_eq!(ra.status, RunStatus::Stopped);
        assert_eq!(ra.points.len(), 1);
        assert!(reloaded.runs.iter().any(|r| r.id == b && r.spec.name.contains(&format!("#{b}"))));
        assert_eq!(reloaded.next_id, b + 1);

        store.finish(a, RunStatus::Done);
        assert_eq!(store.next_queued().unwrap().0, b);
        let _ = std::fs::remove_file(&path);
    }
}
