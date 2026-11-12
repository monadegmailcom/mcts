//! Lock-free batched neural network inference pipeline (port & generalization of C++
//! `inference.h` and `libtorch::InferenceService` in `libtorch_util.h`).
//!
//! ### Architectural Evolution from C++ `beat-it`
//! - **Rescued Lock-Free Queue (`crossbeam_queue::ArrayQueue`)**: Replaces `boost::lockfree::queue`
//!   and raw `void* node` pointers with safe, bounded, cache-padded MPMC ring buffers.
//! - **Zero Lost-Wakeup Deadlocks (`std::thread::park` / `Thread::unpark`)**: In C++ `alphazero.h`
//!   (lines 356–365), combining a lock-free queue with `std::condition_variable::notify_one()`
//!   suffered from lost wakeups (requiring a 1000ms fallback timeout). Rust's `Thread::unpark()`
//!   uses an atomic permit token: if `unpark()` fires before the client calls `thread::park()`,
//!   the permit is retained and `park()` returns immediately in `O(1)` without blocking.
//! - **Generalized to `const N: usize` Players**: Value outputs are `[f32; N]` (relative to the
//!   acting player at the evaluated state) rather than a 2-player scalar.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::Instant;

use crossbeam_queue::ArrayQueue;
use crossbeam_utils::Backoff;

use crate::statistics::Statistics;

/// Batched evaluator interface for neural network (or heuristic/mock) inference backends.
///
/// - `G`: Flattened input state tensor dimension (e.g. `27` for TTT, `324` for UTTT and Rubik's Cube).
/// - `P`: Policy logit output dimension (e.g. `9` for TTT, `81` for UTTT, `12` for Rubik's Cube).
/// - `N`: Number of players (`[f32; N]` value vector in player-relative order: `out_values[i][0]`
///   is the expected return for the acting player at `states[i]`, `out_values[i][1]` for the next
///   player `(actor + 1) % N`, etc.).
pub trait BatchEvaluator<const G: usize, const P: usize, const N: usize>: Send + Sync {
    fn evaluate_batch(
        &self,
        worker_id: usize,
        states: &[[f32; G]],
        out_policies: &mut [[f32; P]],
        out_values: &mut [[f32; N]],
    );
}

/// Closure adapter implementing [`BatchEvaluator`].
pub struct FnBatchEvaluator<F>(pub F);

impl<const G: usize, const P: usize, const N: usize, F> BatchEvaluator<G, P, N>
    for FnBatchEvaluator<F>
where
    F: Fn(&[[f32; G]], &mut [[f32; P]], &mut [[f32; N]]) + Send + Sync,
{
    #[inline]
    fn evaluate_batch(
        &self,
        _worker_id: usize,
        states: &[[f32; G]],
        out_policies: &mut [[f32; P]],
        out_values: &mut [[f32; N]],
    ) {
        (self.0)(states, out_policies, out_values);
    }
}

/// Completed neural network evaluation for a single submitted state.
pub struct InferenceResponse<const P: usize, const N: usize> {
    pub tag: usize,
    pub policy_logits: [f32; P],
    pub value: [f32; N],
}

struct InferenceRequest<const G: usize, const P: usize, const N: usize> {
    tag: usize,
    state: [f32; G],
    reply_queue: Arc<ArrayQueue<InferenceResponse<P, N>>>,
    caller: Thread,
}

#[derive(Clone, Copy, Default)]
struct WorkerStats {
    batch_size: Statistics,
    inference_time_us: Statistics,
}

struct ServiceShared<E, const G: usize, const P: usize, const N: usize> {
    evaluator: E,
    max_batch_size: usize,
    request_queue: ArrayQueue<InferenceRequest<G, P, N>>,
    stop: AtomicBool,
    paused: AtomicBool,
    worker_parked: Box<[AtomicBool]>,
    worker_stats: Box<[Mutex<WorkerStats>]>,
}

/// Background batched inference service (port of `inference::Service<G, P>` +
/// `libtorch::InferenceService<G, P>`).
///
/// Collects individual or wave-batched state evaluation requests from multiple client threads
/// via a shared lock-free [`ArrayQueue`], groups them into contiguous batches up to
/// `max_batch_size`, invokes `E::evaluate_batch`, and routes responses back to each client's
/// lock-free reply queue.
pub struct InferenceService<E, const G: usize, const P: usize, const N: usize>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    shared: Arc<ServiceShared<E, G, P, N>>,
    workers: Box<[JoinHandle<()>]>,
}

impl<E, const G: usize, const P: usize, const N: usize> InferenceService<E, G, P, N>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    /// Starts a new batched inference service with `num_workers` background threads and
    /// maximum batch size `max_batch_size`.
    pub fn new(evaluator: E, max_batch_size: usize, num_workers: usize) -> Self {
        assert!(max_batch_size > 0, "max_batch_size must be >= 1");
        assert!(num_workers > 0, "num_workers must be >= 1");

        let queue_capacity = (max_batch_size * num_workers * 4).max(64);
        let worker_parked = (0..num_workers)
            .map(|_| AtomicBool::new(false))
            .collect::<Box<[_]>>();
        let worker_stats = (0..num_workers)
            .map(|_| Mutex::new(WorkerStats::default()))
            .collect::<Box<[_]>>();

        let shared = Arc::new(ServiceShared {
            evaluator,
            max_batch_size,
            request_queue: ArrayQueue::new(queue_capacity),
            stop: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            worker_parked,
            worker_stats,
        });

        let workers = (0..num_workers)
            .map(|worker_id| {
                let shared_ref = Arc::clone(&shared);
                thread::spawn(move || {
                    Self::worker_loop(worker_id, shared_ref);
                })
            })
            .collect::<Box<[_]>>();

        Self { shared, workers }
    }

    #[inline]
    pub fn max_batch_size(&self) -> usize {
        self.shared.max_batch_size
    }

    #[inline]
    pub fn evaluator(&self) -> &E {
        &self.shared.evaluator
    }

    /// Creates a reusable client handle with its own pre-allocated lock-free reply queue.
    pub fn client(&self, max_in_flight: usize) -> InferenceClient<'_, E, G, P, N> {
        let cap = max_in_flight.max(self.shared.max_batch_size).max(16);
        InferenceClient {
            service: self,
            reply_queue: Arc::new(ArrayQueue::new(cap)),
        }
    }

    /// Pauses background inference workers (used when yielding GPU to a training step).
    pub fn pause_inference(&self) {
        self.shared.paused.store(true, Ordering::Release);
    }

    /// Resumes background inference workers after a training step.
    pub fn resume_inference(&self) {
        self.shared.paused.store(false, Ordering::Release);
        self.wake_workers();
    }

    /// Aggregated batch-size distribution across all inference worker threads.
    pub fn batch_size_stats(&self) -> Statistics {
        let mut total = Statistics::new();
        for w in &self.shared.worker_stats {
            let guard = w.lock().expect("worker_stats mutex poisoned");
            total.join(&guard.batch_size);
        }
        total
    }

    /// Aggregated per-sample inference latency distribution (in microseconds) across workers.
    pub fn inference_time_stats(&self) -> Statistics {
        let mut total = Statistics::new();
        for w in &self.shared.worker_stats {
            let guard = w.lock().expect("worker_stats mutex poisoned");
            total.join(&guard.inference_time_us);
        }
        total
    }

    /// Resets accumulated batch-size and inference-time statistics on all workers.
    pub fn reset_stats(&self) {
        for w in &self.shared.worker_stats {
            let mut guard = w.lock().expect("worker_stats mutex poisoned");
            guard.batch_size.reset();
            guard.inference_time_us.reset();
        }
    }

    /// Walks the worker list, tests each worker's atomic idle flag, wakes the first parked
    /// worker found, and returns immediately (zero mutex locks, zero syscalls if all are busy).
    #[inline]
    fn wake_one_worker(&self) {
        for (parked, handle) in self.shared.worker_parked.iter().zip(self.workers.iter()) {
            if parked.swap(false, Ordering::SeqCst) {
                handle.thread().unpark();
                return;
            }
        }
    }

    #[inline]
    fn wake_workers(&self) {
        for (parked, handle) in self.shared.worker_parked.iter().zip(self.workers.iter()) {
            parked.store(false, Ordering::SeqCst);
            handle.thread().unpark();
        }
    }

    fn worker_loop(worker_id: usize, shared: Arc<ServiceShared<E, G, P, N>>) {
        let max_batch_size = shared.max_batch_size;
        let mut req_batch: Vec<InferenceRequest<G, P, N>> = Vec::with_capacity(max_batch_size);
        let mut state_batch: Vec<[f32; G]> = Vec::with_capacity(max_batch_size);
        let mut policy_batch: Vec<[f32; P]> = vec![[0.0; P]; max_batch_size];
        let mut value_batch: Vec<[f32; N]> = vec![[0.0; N]; max_batch_size];

        while !shared.stop.load(Ordering::Acquire) {
            while shared.paused.load(Ordering::Acquire) && !shared.stop.load(Ordering::Acquire) {
                thread::park();
            }
            if shared.stop.load(Ordering::Acquire) {
                break;
            }

            // Drain up to `max_batch_size` pending requests from the lock-free queue.
            while req_batch.len() < max_batch_size {
                if let Some(req) = shared.request_queue.pop() {
                    state_batch.push(req.state);
                    req_batch.push(req);
                } else {
                    break;
                }
            }

            let b = req_batch.len();
            if b == 0 {
                // Pure event-driven sleep: announce we are parked (`SeqCst`), re-check that no
                // request arrived just before the store, and sleep until a producer unparks us.
                shared.worker_parked[worker_id].store(true, Ordering::SeqCst);
                if shared.request_queue.is_empty()
                    && !shared.stop.load(Ordering::Acquire)
                    && !shared.paused.load(Ordering::Acquire)
                {
                    thread::park();
                }
                shared.worker_parked[worker_id].store(false, Ordering::Relaxed);
                continue;
            }

            let start = Instant::now();
            shared.evaluator.evaluate_batch(
                worker_id,
                &state_batch[..b],
                &mut policy_batch[..b],
                &mut value_batch[..b],
            );
            let elapsed_us_per_item = (start.elapsed().as_secs_f64() * 1_000_000.0) / (b as f64);

            {
                let mut st = shared.worker_stats[worker_id]
                    .lock()
                    .expect("worker_stats mutex poisoned");
                st.batch_size.update(b as f32);
                st.inference_time_us.update(elapsed_us_per_item as f32);
            }

            // First enqueue all responses into their respective client reply queues, then unpark
            // each waiting client once its entire batch slice is ready to be drained.
            for ((req, &policy_logits), &value) in req_batch
                .iter()
                .zip(policy_batch[..b].iter())
                .zip(value_batch[..b].iter())
            {
                let mut resp = InferenceResponse {
                    tag: req.tag,
                    policy_logits,
                    value,
                };
                let push_backoff = Backoff::new();
                while let Err(returned) = req.reply_queue.push(resp) {
                    resp = returned;
                    req.caller.unpark();
                    push_backoff.snooze();
                }
            }
            let mut prev_queue: Option<*const ArrayQueue<InferenceResponse<P, N>>> = None;
            for req in req_batch.drain(..) {
                let q_ptr = Arc::as_ptr(&req.reply_queue);
                if prev_queue != Some(q_ptr) {
                    req.caller.unpark();
                    prev_queue = Some(q_ptr);
                }
            }
            state_batch.clear();
        }
    }
}

impl<E, const G: usize, const P: usize, const N: usize> Drop for InferenceService<E, G, P, N>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.wake_workers();
        for handle in std::mem::take(&mut self.workers) {
            let _ = handle.join();
        }
    }
}

/// Per-worker client handle for submitting single or wave-batched states to an [`InferenceService`].
pub struct InferenceClient<'a, E, const G: usize, const P: usize, const N: usize>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    service: &'a InferenceService<E, G, P, N>,
    reply_queue: Arc<ArrayQueue<InferenceResponse<P, N>>>,
}

impl<E, const G: usize, const P: usize, const N: usize> InferenceClient<'_, E, G, P, N>
where
    E: BatchEvaluator<G, P, N> + 'static,
{
    /// Enqueues a single state evaluation request tagged with `tag` directly into the shared
    /// lock-free [`InferenceService`] request queue without blocking for the response.
    ///
    /// Call [`Self::notify_service`] after enqueuing one or more requests to wake an idle
    /// inference worker.
    #[inline]
    pub fn submit(&self, tag: usize, state: [f32; G]) {
        let mut req = InferenceRequest {
            tag,
            state,
            reply_queue: Arc::clone(&self.reply_queue),
            caller: thread::current(),
        };
        let backoff = Backoff::new();
        while let Err(returned) = self.service.shared.request_queue.push(req) {
            req = returned;
            self.service.wake_one_worker();
            backoff.snooze();
        }
    }

    /// Wakes one idle background inference worker (if any are currently parked).
    #[inline]
    pub fn notify_service(&self) {
        self.service.wake_one_worker();
    }

    /// Pops the next completed [`InferenceResponse`] from this client's lock-free reply queue,
    /// returning `None` if no responses have arrived yet.
    #[inline]
    pub fn try_pop(&self) -> Option<InferenceResponse<P, N>> {
        self.reply_queue.pop()
    }

    /// Parks the calling thread until at least one response is available in `reply_queue`.
    #[inline]
    pub fn wait_for_response(&self) {
        while self.reply_queue.is_empty() {
            thread::park();
        }
    }

    /// Convenience helper that submits a slice of `states` and blocks until all responses return.
    pub fn evaluate_batch(
        &mut self,
        states: &[[f32; G]],
        out_policies: &mut [[f32; P]],
        out_values: &mut [[f32; N]],
    ) {
        let count = states.len();
        if count == 0 {
            return;
        }
        assert!(out_policies.len() >= count && out_values.len() >= count);

        let mut submitted = 0usize;
        let mut received = 0usize;

        while received < count {
            let mut pushed_any = false;
            while submitted < count && (submitted - received) < self.reply_queue.capacity() {
                self.submit(submitted, states[submitted]);
                submitted += 1;
                pushed_any = true;
            }
            if pushed_any {
                self.notify_service();
            }

            while let Some(resp) = self.try_pop() {
                out_policies[resp.tag] = resp.policy_logits;
                out_values[resp.tag] = resp.value;
                received += 1;
            }

            if received < count && submitted == count {
                self.wait_for_response();
            }
        }
    }

    /// Convenience helper evaluating a single state `[f32; G]` through the batched service.
    #[inline]
    pub fn evaluate_one(&mut self, state: &[f32; G]) -> ([f32; P], [f32; N]) {
        self.submit(0, *state);
        self.notify_service();
        loop {
            if let Some(resp) = self.try_pop() {
                return (resp.policy_logits, resp.value);
            }
            thread::park();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_free_inference_service_batches_multi_thread_requests_without_deadlock() {
        // Mock model: policy[i] = sum(state) + i, value[0] = state[0], value[1] = -state[0].
        let evaluator = FnBatchEvaluator(
            |states: &[[f32; 4]], policies: &mut [[f32; 3]], values: &mut [[f32; 2]]| {
                for ((s, p), v) in states.iter().zip(policies.iter_mut()).zip(values.iter_mut()) {
                    let sum: f32 = s.iter().sum();
                    *p = [sum, sum + 1.0, sum + 2.0];
                    *v = [s[0], -s[0]];
                }
            },
        );

        let service = InferenceService::new(evaluator, 32, 1);

        // Spawn 4 client threads, each submitting 25 batches of 8 states = 800 total evaluations.
        std::thread::scope(|s| {
            for t in 0..4 {
                let svc = &service;
                s.spawn(move || {
                    let mut client = svc.client(16);
                    let mut states = [[0.0f32; 4]; 8];
                    let mut policies = [[0.0f32; 3]; 8];
                    let mut values = [[0.0f32; 2]; 8];

                    for step in 0..25 {
                        for (i, st) in states.iter_mut().enumerate() {
                            let v = ((t * 100 + step) * 8 + i) as f32;
                            *st = [v, 1.0, 2.0, 3.0];
                        }
                        client.evaluate_batch(&states, &mut policies, &mut values);
                        for i in 0..8 {
                            let v = ((t * 100 + step) * 8 + i) as f32;
                            let expected_sum = v + 6.0;
                            assert!((policies[i][0] - expected_sum).abs() < 1e-5);
                            assert!((values[i][0] - v).abs() < 1e-5);
                            assert!((values[i][1] + v).abs() < 1e-5);
                        }
                    }
                });
            }
        });

        let batch_stats = service.batch_size_stats();
        assert!(batch_stats.mean() >= 8.0, "expected batched execution, got {batch_stats}");
    }
}
