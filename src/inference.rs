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
#[derive(Clone, Copy, Debug)]
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
    unique_batch_size: Statistics,
    inference_time_us: Statistics,
    total_requests: u64,
    unique_evaluations: u64,
}

#[inline]
fn canonical_bits(x: f32) -> u32 {
    if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

#[inline]
fn hash_state<const G: usize>(state: &[f32; G]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in state {
        h ^= canonical_bits(x) as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[inline]
fn states_equal<const G: usize>(a: &[f32; G], b: &[f32; G]) -> bool {
    for (&x, &y) in a.iter().zip(b.iter()) {
        if canonical_bits(x) != canonical_bits(y) {
            return false;
        }
    }
    true
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
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Current number of pending state evaluation requests waiting in the shared lock-free queue.
    #[inline]
    pub fn queue_len(&self) -> usize {
        self.shared.request_queue.len()
    }

    /// Maximum capacity of the shared lock-free request queue.
    #[inline]
    pub fn queue_capacity(&self) -> usize {
        self.shared.request_queue.capacity()
    }

    #[inline]
    pub fn evaluator(&self) -> &E {
        &self.shared.evaluator
    }

    /// Creates a reusable client handle with its own pre-allocated lock-free reply queue.
    pub fn client(&self, max_in_flight: usize) -> InferenceClient<'_, E, G, P, N> {
        let cap = max_in_flight.max(1);
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

    /// Aggregated logical batch-size distribution (total requests served per batch) across workers.
    pub fn batch_size_stats(&self) -> Statistics {
        let mut total = Statistics::new();
        for w in &self.shared.worker_stats {
            let guard = w.lock().expect("worker_stats mutex poisoned");
            total.join(&guard.batch_size);
        }
        total
    }

    /// Aggregated unique state batch-size distribution (after intra-batch deduplication) across workers.
    pub fn unique_batch_size_stats(&self) -> Statistics {
        let mut total = Statistics::new();
        for w in &self.shared.worker_stats {
            let guard = w.lock().expect("worker_stats mutex poisoned");
            total.join(&guard.unique_batch_size);
        }
        total
    }

    /// Returns `(total_requests, unique_evaluations)` accumulated across all workers.
    pub fn dedup_counts(&self) -> (u64, u64) {
        let mut total_reqs = 0u64;
        let mut unique_evals = 0u64;
        for w in &self.shared.worker_stats {
            let guard = w.lock().expect("worker_stats mutex poisoned");
            total_reqs += guard.total_requests;
            unique_evals += guard.unique_evaluations;
        }
        (total_reqs, unique_evals)
    }

    /// Fraction `[0.0, 1.0]` of logical evaluation requests eliminated by intra-batch deduplication.
    pub fn dedup_savings_ratio(&self) -> f32 {
        let (total, unique) = self.dedup_counts();
        if total == 0 || unique >= total {
            0.0
        } else {
            1.0 - (unique as f32) / (total as f32)
        }
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

    /// Resets accumulated batch-size, deduplication, and inference-time statistics on all workers.
    pub fn reset_stats(&self) {
        for w in &self.shared.worker_stats {
            let mut guard = w.lock().expect("worker_stats mutex poisoned");
            guard.batch_size.reset();
            guard.unique_batch_size.reset();
            guard.inference_time_us.reset();
            guard.total_requests = 0;
            guard.unique_evaluations = 0;
        }
    }

    /// Walks the worker list, tests each worker's atomic idle flag, and wakes enough parked
    /// workers to cover the currently queued batches (zero mutex locks, zero syscalls if busy).
    #[inline]
    fn wake_one_worker(&self) {
        let q_len = self.shared.request_queue.len();
        let mut needed = q_len.div_ceil(self.shared.max_batch_size).max(1);
        for (parked, handle) in self.shared.worker_parked.iter().zip(self.workers.iter()) {
            if parked.swap(false, Ordering::SeqCst) {
                handle.thread().unpark();
                needed -= 1;
                if needed == 0 {
                    return;
                }
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
        let max_logical_requests = max_batch_size * 4;
        let mut req_batch: Vec<InferenceRequest<G, P, N>> =
            Vec::with_capacity(max_logical_requests);
        let mut req_to_unique: Vec<usize> = Vec::with_capacity(max_logical_requests);
        let mut state_batch: Vec<[f32; G]> = Vec::with_capacity(max_batch_size);
        let mut state_hashes: Vec<u64> = Vec::with_capacity(max_batch_size);
        let mut policy_batch: Vec<[f32; P]> = vec![[0.0; P]; max_batch_size];
        let mut value_batch: Vec<[f32; N]> = vec![[0.0; N]; max_batch_size];
        let mut postponed: Vec<(
            Arc<ArrayQueue<InferenceResponse<P, N>>>,
            Thread,
            InferenceResponse<P, N>,
        )> = Vec::with_capacity(max_logical_requests);
        let mut callers_to_wake: Vec<Thread> = Vec::with_capacity(max_logical_requests);

        while !shared.stop.load(Ordering::Acquire) {
            while shared.paused.load(Ordering::Acquire) && !shared.stop.load(Ordering::Acquire) {
                thread::park();
            }
            if shared.stop.load(Ordering::Acquire) {
                break;
            }

            // Drain pending requests from the lock-free queue, deduplicating identical encoded
            // board states `[f32; G]` within the batch so the neural network evaluates each
            // unique state at most once per batch!
            while state_batch.len() < max_batch_size && req_batch.len() < max_logical_requests {
                if let Some(req) = shared.request_queue.pop() {
                    let h = hash_state(&req.state);
                    let mut existing = None;
                    for (idx, &cand_h) in state_hashes.iter().enumerate() {
                        if cand_h == h && states_equal(&state_batch[idx], &req.state) {
                            existing = Some(idx);
                            break;
                        }
                    }
                    let unique_idx = match existing {
                        Some(idx) => idx,
                        None => {
                            let idx = state_batch.len();
                            state_batch.push(req.state);
                            state_hashes.push(h);
                            idx
                        }
                    };
                    req_to_unique.push(unique_idx);
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

            let u = state_batch.len();
            let start = Instant::now();
            shared.evaluator.evaluate_batch(
                worker_id,
                &state_batch[..u],
                &mut policy_batch[..u],
                &mut value_batch[..u],
            );
            let elapsed_us_per_item = (start.elapsed().as_secs_f64() * 1_000_000.0) / (b as f64);

            {
                let mut st = shared.worker_stats[worker_id]
                    .lock()
                    .expect("worker_stats mutex poisoned");
                st.batch_size.update(b as f32);
                st.unique_batch_size.update(u as f32);
                st.inference_time_us.update(elapsed_us_per_item as f32);
                st.total_requests += b as u64;
                st.unique_evaluations += u as u64;
            }

            // Deliver all `b` responses into their lock-free reply queues FIRST, fanning out
            // each unique state's `(policy_logits, value)` to every matching request, and only
            // unpark caller threads once the entire batch of responses is visible in their queues!
            for (req, unique_idx) in req_batch.drain(..).zip(req_to_unique.drain(..)) {
                let resp = InferenceResponse {
                    tag: req.tag,
                    policy_logits: policy_batch[unique_idx],
                    value: value_batch[unique_idx],
                };
                match req.reply_queue.push(resp) {
                    Ok(()) => {
                        if !callers_to_wake.iter().any(|t| t.id() == req.caller.id()) {
                            callers_to_wake.push(req.caller);
                        }
                    }
                    Err(returned) => {
                        req.caller.unpark();
                        postponed.push((req.reply_queue, req.caller, returned));
                    }
                }
            }

            for caller in callers_to_wake.drain(..) {
                caller.unpark();
            }

            let push_backoff = Backoff::new();
            while !postponed.is_empty() && !shared.stop.load(Ordering::Acquire) {
                postponed.retain(|(queue, caller, resp)| match queue.push(*resp) {
                    Ok(()) => {
                        caller.unpark();
                        false
                    }
                    Err(_) => {
                        caller.unpark();
                        true
                    }
                });
                if !postponed.is_empty() {
                    push_backoff.snooze();
                }
            }
            postponed.clear();
            state_batch.clear();
            state_hashes.clear();
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

        // Spawn 4 client threads, each submitting 25 waves of 8 states = 800 total evaluations.
        std::thread::scope(|s| {
            for t in 0..4 {
                let svc = &service;
                s.spawn(move || {
                    let client = svc.client(16);

                    for step in 0..25 {
                        for i in 0..8 {
                            let v = ((t * 100 + step) * 8 + i) as f32;
                            client.submit(i, [v, 1.0, 2.0, 3.0]);
                        }
                        client.notify_service();

                        let mut received = 0usize;
                        while received < 8 {
                            while let Some(resp) = client.try_pop() {
                                let v = ((t * 100 + step) * 8 + resp.tag) as f32;
                                let expected_sum = v + 6.0;
                                assert!((resp.policy_logits[0] - expected_sum).abs() < 1e-5);
                                assert!((resp.value[0] - v).abs() < 1e-5);
                                assert!((resp.value[1] + v).abs() < 1e-5);
                                received += 1;
                            }
                            if received < 8 {
                                client.wait_for_response();
                            }
                        }
                    }
                });
            }
        });

        let batch_stats = service.batch_size_stats();
        assert!(batch_stats.mean() >= 8.0, "expected batched execution, got {batch_stats}");
    }

    #[test]
    fn intra_batch_deduplication_evaluates_unique_states_once_and_broadcasts_results() {
        use std::sync::atomic::AtomicUsize;

        let evaluated_states_count = Arc::new(AtomicUsize::new(0));
        let count_ref = Arc::clone(&evaluated_states_count);

        let evaluator = FnBatchEvaluator(
            move |states: &[[f32; 4]], policies: &mut [[f32; 3]], values: &mut [[f32; 2]]| {
                count_ref.fetch_add(states.len(), Ordering::Relaxed);
                for ((s, p), v) in states.iter().zip(policies.iter_mut()).zip(values.iter_mut()) {
                    let sum: f32 = s.iter().sum();
                    *p = [sum, sum + 1.0, sum + 2.0];
                    *v = [s[0], -s[0]];
                }
            },
        );

        let service = InferenceService::new(evaluator, 32, 1);
        service.pause_inference();

        // Submit 16 requests covering only 2 distinct states before resuming the worker.
        let client = service.client(16);
        for i in 0..16 {
            let first = if i % 2 == 0 { 10.0 } else { 20.0 };
            client.submit(i, [first, 1.0, 2.0, 3.0]);
        }

        service.resume_inference();

        let mut received = 0usize;
        while received < 16 {
            while let Some(resp) = client.try_pop() {
                let expected_first = if resp.tag % 2 == 0 { 10.0 } else { 20.0 };
                assert!((resp.value[0] - expected_first).abs() < 1e-5);
                assert!((resp.policy_logits[0] - (expected_first + 6.0)).abs() < 1e-5);
                received += 1;
            }
            if received < 16 {
                client.wait_for_response();
            }
        }

        // Only 2 unique states should have been evaluated by the NN backend!
        assert_eq!(evaluated_states_count.load(Ordering::Relaxed), 2);
        assert_eq!(service.dedup_counts(), (16, 2));
        assert!((service.dedup_savings_ratio() - 0.875).abs() < 1e-5);
    }
}
