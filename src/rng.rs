//! Fast, compact (32-byte), deterministic `Xoshiro256++` pseudo-random number generator.
//!
//! Replaces C++ `std::mt19937` (which occupies ~5,000 bytes of state per player instance)
//! with a 32-byte state machine suitable for spawning hundreds of Game-Parallel self-play
//! workers without L2/L3 cache pollution.

/// Deterministic `Xoshiro256++` PRNG seeded via `SplitMix64`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Creates a new deterministic PRNG from any 64-bit seed (including `0`).
    pub const fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let s0 = splitmix64(&mut sm);
        let s1 = splitmix64(&mut sm);
        let s2 = splitmix64(&mut sm);
        let s3 = splitmix64(&mut sm);
        Self {
            state: [s0, s1, s2, s3],
        }
    }

    /// Generates the next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[0].wrapping_add(self.state[3]))
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Generates a uniform `f32` in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let upper_24 = (self.next_u64() >> 40) as u32;
        (upper_24 as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Generates an unbiased uniform index in `0..upper`.
    ///
    /// # Panics
    /// Panics if `upper == 0`.
    #[inline]
    pub fn random_range(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be positive");
        let bound = upper as u64;
        // Rejection threshold to eliminate modulo bias: (2^64 % bound)
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            let product = (r as u128).wrapping_mul(bound as u128);
            if (product as u64) >= threshold {
                return (product >> 64) as usize;
            }
        }
    }

    /// Performs an in-place Fisher-Yates shuffle on `slice`.
    #[inline]
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.random_range(i + 1);
            slice.swap(i, j);
        }
    }
}

#[inline]
const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_is_compact() {
        assert_eq!(std::mem::size_of::<Rng>(), 32);
    }

    #[test]
    fn deterministic_and_unbiased_bounds() {
        let mut a = Rng::from_seed(42);
        let mut b = Rng::from_seed(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        for _ in 0..100 {
            let idx = a.random_range(7);
            assert!(idx < 7);
            let f = a.next_f32();
            assert!((0.0..1.0).contains(&f));
        }
    }
}
