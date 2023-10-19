#![feature(portable_simd)]
mod multi_mt;

use crate::multi_mt::MultiMT19937;
use multiversion::multiversion;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use smallvec::SmallVec;
use std::simd::{u32x8, Simd, SimdPartialEq};

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: u32 = 63;

pub type IVs = (u32, u32, u32, u32, u32, u32);
pub type Frame = u32;
pub type Seed = u32;

pub struct Hit {
    pub seed: Seed,
    pub frame1: Frame,
    pub frame2: Frame,
}

#[inline]
fn encode_ivs(iv: IVs) -> u32 {
    (iv.0 << 25) | (iv.1 << 20) | (iv.2 << 15) | (iv.3 << 10) | (iv.4 << 5) | iv.5
}

// Advances f_max + 6 - f_min frames
#[inline]
fn find_frame(
    mt: &mut MultiMT19937,
    ivs: u32,
    (f_min, f_max): (Frame, Frame), // right-closed
) -> u32x8 {
    let mut curr = (mt.next_iv() << Simd::splat(20))
        | (mt.next_iv() << Simd::splat(15))
        | (mt.next_iv() << Simd::splat(10))
        | (mt.next_iv() << Simd::splat(5))
        | mt.next_iv();
    let mut frame: u32x8 = Simd::splat(0);
    for i in f_min..=f_max {
        curr = (curr & Simd::splat(0x1FFFFFF)) << Simd::splat(5) | mt.next_iv();
        frame = curr.simd_eq(Simd::splat(ivs)).select(Simd::splat(i), frame);
    }
    frame
}

#[multiversion(targets = "simd")]
fn find_seed_simd(
    seed_hi: u32,
    ivs1: u32,
    ivs2: u32,
    frame_range1: (Frame, Frame), // right-closed
    frame_range2: (Frame, Frame), // right-closed
) -> SmallVec<[Hit; 1]> {
    let mut results = SmallVec::new();
    let mut mt = MultiMT19937::default();

    let (f1_min, f1_max) = frame_range1;
    let (f2_min, f2_max) = frame_range2;

    let seed_min = seed_hi << 16;
    let seed_max = (seed_hi << 16) | 0xFFFF;
    for s in (seed_min..=seed_max).step_by(8) {
        let seed = Simd::from_array([s, s | 1, s | 2, s | 3, s | 4, s | 5, s | 6, s | 7]);
        mt.init(seed);
        mt.reserve((PRE_ADVANCE_FRAME + f1_max + 6) as usize);

        mt.discard((PRE_ADVANCE_FRAME + f1_min) as usize);

        // Advances f1_max + 6 - f1_min frames
        let f1 = find_frame(&mut mt, ivs1, frame_range1);
        if f1.simd_eq(Simd::splat(0)).all() {
            continue;
        }

        mt.reserve((PRE_ADVANCE_FRAME + f2_max + 6) as usize);
        mt.discard((f2_min - f1_max - 6) as usize);

        // Advances f2_max + 6 - f2_min frames
        let f2 = find_frame(&mut mt, ivs2, frame_range2);
        if f2.simd_eq(Simd::splat(0)).all() {
            continue;
        }

        for i in 0..8 {
            if f1[i] != 0 && f2[i] != 0 {
                results.push(Hit {
                    seed: s | (i as u32),
                    frame1: f1[i],
                    frame2: f2[i],
                });
            }
        }
    }
    results
}

pub fn find_seed(
    seed_hi_range: (Seed, Seed), // right-closed
    ivs1: IVs,
    ivs2: IVs,
    frame_range1: (Frame, Frame), // right-closed
    frame_range2: (Frame, Frame), // right-closed
    notify_progress: impl Fn(&[Hit], u32) -> () + Sync,
) -> Vec<Hit> {
    let (seed_hi_min, seed_hi_max) = seed_hi_range;
    assert!(seed_hi_min <= seed_hi_max && seed_hi_max <= 0xFFFF);
    assert!(
        ivs1.0 <= 31
            && ivs1.1 <= 31
            && ivs1.2 <= 31
            && ivs1.3 <= 31
            && ivs1.4 <= 31
            && ivs1.5 <= 31
    );
    assert!(
        ivs2.0 <= 31
            && ivs2.1 <= 31
            && ivs2.2 <= 31
            && ivs2.3 <= 31
            && ivs2.4 <= 31
            && ivs2.5 <= 31
    );
    assert!(
        frame_range1.0 <= frame_range1.1
            && frame_range1.1 + 6 <= frame_range2.0
            && frame_range2.0 <= frame_range2.1
            && frame_range2.1 <= 3000
    );

    let ivs1 = encode_ivs(ivs1);
    let ivs2 = encode_ivs(ivs2);

    let len = seed_hi_max - seed_hi_min + 1;

    (seed_hi_min..=seed_hi_max)
        .into_par_iter()
        .flat_map_iter(|seed_hi| {
            let hits = find_seed_simd(seed_hi, ivs1, ivs2, frame_range1, frame_range2);
            notify_progress(&hits, len);
            hits
        })
        .collect()
}
