// This implementation is largely based on rust-mersenne-twister licensed under MIT / Apache 2.0 dual license
// GitHub: https://github.com/dcrewi/rust-mersenne-twister
//
// Copyright (c) 2015,2017 rust-mersenne-twister developers
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

use std::simd::{u32x8, Simd};

const N: usize = 624;
const M: usize = 397;
const ONE: u32x8 = Simd::from_array([1; 8]);
const MATRIX_A: u32x8 = Simd::from_array([0x9908b0df; 8]);
const UPPER_MASK: u32x8 = Simd::from_array([0x80000000; 8]);
const LOWER_MASK: u32x8 = Simd::from_array([0x7fffffff; 8]);

pub struct MultiMT19937 {
    idx: usize,
    state: [u32x8; N],
}

const UNINITIALIZED: MultiMT19937 = MultiMT19937 {
    idx: 0,
    state: [Simd::from_array([0; 8]); N],
};

impl Default for MultiMT19937 {
    #[inline]
    fn default() -> MultiMT19937 {
        UNINITIALIZED
    }
}

impl MultiMT19937 {
    #[inline]
    pub fn reseed(&mut self, seed: u32x8) {
        self.idx = N;
        self.state[0] = seed;
        for i in 1..N {
            self.state[i] = Simd::splat(1812433253)
                * (self.state[i - 1] ^ (self.state[i - 1] >> Simd::splat(30)))
                + Simd::splat(i as u32);
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
    pub fn next_iv(&mut self) -> u32x8 {
        self.next_u32x8() >> Simd::splat(27)
    }

    #[inline]
    pub fn discard(&mut self, n: u32) {
        for _ in 0..n {
            self.next_u32x8();
        }
    }

    #[inline]
    fn fill_next_state(&mut self) {
        for i in 0..N - M {
            let x = (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
            self.state[i] = self.state[i + M] ^ (x >> Simd::splat(1)) ^ ((x & ONE) * MATRIX_A);
        }
        for i in N - M..N - 1 {
            let x = (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
            self.state[i] = self.state[i + M - N] ^ (x >> Simd::splat(1)) ^ ((x & ONE) * MATRIX_A);
        }
        let x = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
        self.state[N - 1] = self.state[M - 1] ^ (x >> Simd::splat(1)) ^ ((x & ONE) * MATRIX_A);
        self.idx = 0;
    }
}

#[inline]
fn temper(mut x: u32x8) -> u32x8 {
    x ^= x >> Simd::splat(11);
    x ^= (x << Simd::splat(7)) & Simd::splat(0x9d2c5680);
    x ^= (x << Simd::splat(15)) & Simd::splat(0xefc60000);
    x ^= x >> Simd::splat(18);
    x
}

#[test]
fn test_compare_with_mt() {
    let mut mt = MultiMT19937::default();
    mt.reseed(Simd::from_array([0, 1, 2, 3, 4, 5, 6, 7]));
    mt.discard(1000);
    assert_eq!(
        mt.next_u32x8().to_array(),
        [
            1333075495, 375733240, 1144994631, 454162887, 3777932409, 3818223146, 3836374258,
            4142999817
        ]
    );
}
