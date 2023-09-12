use mersenne_twister::MT19937;
use rand::{Rng, SeedableRng};
use std::simd::u32x8;

const N: usize = 624;
const M: usize = 397;
const ONE: u32x8 = u32x8::from_array([1, 1, 1, 1, 1, 1, 1, 1]);
const MATRIX_A: u32x8 = u32x8::from_array([
    0x9908b0df, 0x9908b0df, 0x9908b0df, 0x9908b0df, 0x9908b0df, 0x9908b0df, 0x9908b0df, 0x9908b0df,
]);
const UPPER_MASK: u32x8 = u32x8::from_array([
    0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
]);
const LOWER_MASK: u32x8 = u32x8::from_array([
    0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
]);

pub struct MultiMT19937 {
    idx: usize,
    state: [u32x8; N],
}

const UNINITIALIZED: MultiMT19937 = MultiMT19937 {
    idx: 0,
    state: [u32x8::from_array([0, 0, 0, 0, 0, 0, 0, 0]); N],
};

impl MultiMT19937 {
    #[inline]
    pub fn from_seed(seed: u32x8) -> MultiMT19937 {
        let mut mt = UNINITIALIZED;
        mt.reseed(seed);
        mt
    }

    pub fn reseed(&mut self, seed: u32x8) {
        self.idx = N;
        self.state[0] = seed;
        for i in 1..N {
            self.state[i] = u32x8::splat(1812433253)
                * (self.state[i - 1] ^ (self.state[i - 1] >> u32x8::splat(30)))
                + u32x8::splat(i as u32);
        }
    }

    #[inline]
    pub fn next_u32x8(&mut self) -> u32x8 {
        debug_assert!(self.idx != 0);
        if self.idx >= N {
            self.fill_next_state();
        }
        let x = self.state[self.idx];
        self.idx += 1;
        temper(x)
    }

    #[inline]
    pub fn next_u5x8(&mut self) -> u32x8 {
        self.next_u32x8() >> u32x8::splat(27)
    }

    #[inline]
    pub fn discard(&mut self, n: usize) {
        for _ in 0..n {
            self.next_u32x8();
        }
    }

    fn fill_next_state(&mut self) {
        for i in 0..N - M {
            let x = (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
            self.state[i] = self.state[i + M] ^ (x >> u32x8::splat(1)) ^ ((x & ONE) * MATRIX_A);
        }
        for i in N - M..N - 1 {
            let x = (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
            self.state[i] = self.state[i + M - N] ^ (x >> u32x8::splat(1)) ^ ((x & ONE) * MATRIX_A);
        }
        let x = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
        self.state[N - 1] = self.state[M - 1] ^ (x >> u32x8::splat(1)) ^ ((x & ONE) * MATRIX_A);
        self.idx = 0;
    }
}

#[inline]
fn temper(mut x: u32x8) -> u32x8 {
    x ^= x >> u32x8::splat(11);
    x ^= (x << u32x8::splat(7)) & u32x8::splat(0x9d2c5680);
    x ^= (x << u32x8::splat(15)) & u32x8::splat(0xefc60000);
    x ^= x >> u32x8::splat(18);
    x
}

#[test]
fn test_compare_with_mt() {
    let mut mts: Vec<MT19937> = Vec::new();
    let mut vec: Vec<u32> = Vec::new();
    for i in 0..8 {
        mts.push(MT19937::from_seed(i as u32));
        for _ in 0..1000 {
            mts[i].next_u32();
        }
        vec.push(mts[i].next_u32());
    }

    let mut multi_mt: MultiMT19937 =
        MultiMT19937::from_seed(u32x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]));
    for _ in 0..1000 {
        multi_mt.next_u32x8();
    }

    assert_eq!(
        multi_mt.next_u32x8().to_array(),
        [vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]]
    );
}
