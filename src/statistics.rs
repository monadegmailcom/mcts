//! Online summary statistics accumulator (port of C++ `statistics.h` / `statistics.cpp`).

use std::fmt;

/// Accumulates running `mean`, `stddev`, `min`, `max`, and `count` in `O(1)` memory.
///
/// Uses `f64` accumulators internally to avoid catastrophic cancellation in variance
/// computation on constant or large sample streams.
#[derive(Clone, Copy, Debug)]
pub struct Statistics {
    sum: f64,
    sum_square: f64,
    min: f32,
    max: f32,
    count: usize,
}

impl Default for Statistics {
    fn default() -> Self {
        Self::new()
    }
}

impl Statistics {
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
            sum_square: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            count: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        let v = value as f64;
        self.sum += v;
        self.sum_square += v * v;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    #[inline]
    pub fn update_usize(&mut self, value: usize) {
        self.update(value as f32);
    }

    #[inline]
    pub const fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub const fn min(&self) -> f32 {
        self.min
    }

    #[inline]
    pub const fn max(&self) -> f32 {
        self.max
    }

    #[inline]
    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        (self.sum / (self.count as f64)) as f32
    }

    #[inline]
    pub fn stddev(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.sum / n;
        let variance = (self.sum_square / n - mean * mean).max(0.0);
        variance.sqrt() as f32
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn join(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        self.sum += other.sum;
        self.sum_square += other.sum_square;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.count += other.count;
    }
}

impl fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mean = {:.4}, stddev = {:.4}, min = {:.4}, max = {:.4}, count = {}",
            self.mean(),
            self.stddev(),
            self.min(),
            self.max(),
            self.count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_stream_never_produces_nan_stddev() {
        let mut s = Statistics::new();
        for _ in 0..1_000 {
            s.update(64.0);
        }
        assert_eq!(s.count(), 1_000);
        assert_eq!(s.mean(), 64.0);
        assert_eq!(s.stddev(), 0.0);
    }

    #[test]
    fn join_combines_partitions() {
        let mut a = Statistics::new();
        a.update(2.0);
        a.update(4.0);

        let mut b = Statistics::new();
        b.update(6.0);
        b.update(8.0);

        a.join(&b);
        assert_eq!(a.count(), 4);
        assert_eq!(a.min(), 2.0);
        assert_eq!(a.max(), 8.0);
        assert!((a.mean() - 5.0).abs() < 1e-6);
    }
}
