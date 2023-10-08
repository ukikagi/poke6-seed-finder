use crate::multi_mt::MultiMT19937;
use multiversion::multiversion;
use rayon::prelude::*;
use std::simd::{u32x8, Simd, SimdPartialEq};

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: u32 = 63;
const W: u32 = 16;

pub type IVs = (u32, u32, u32, u32, u32, u32);
pub type Frame = u32;
pub type Seed = u32;

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
) -> Vec<(Seed, Frame, Frame)> {
    let mut results = Vec::new();
    let mut mt = MultiMT19937::default();

    let (f1_min, f1_max) = frame_range1;
    let (f2_min, f2_max) = frame_range2;

    let seed_min = seed_hi << W;
    let seed_max = seed_min | ((1 << W) - 1);
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
            if f1[i] != 0 || f2[i] != 0 {
                results.push((s | (i as u32), f1[i], f2[i]));
            }
        }
    }
    results
}

pub fn find_seed<F>(
    (seed_min, seed_max): (u32, u32), // right-closed
    ivs1: IVs,
    ivs2: IVs,
    frame_range1: (Frame, Frame), // right-closed
    frame_range2: (Frame, Frame), // right-closed
    update: F,
) -> Vec<(Seed, Frame, Frame)>
where
    F: Fn(&[(Seed, Frame, Frame)], u32) -> () + Sync,
{
    let ivs1 = encode_ivs(ivs1);
    let ivs2 = encode_ivs(ivs2);

    let seed_hi_l = seed_min >> W;
    let seed_hi_r = (seed_max >> W) + 1;
    let len = seed_hi_r - seed_hi_l;

    (seed_hi_l..seed_hi_r)
        .into_par_iter()
        .flat_map(|seed_hi| {
            let results = find_seed_simd(seed_hi, ivs1, ivs2, frame_range1, frame_range2);
            update(&results, len);
            results
        })
        .filter(|&(s, _, _)| seed_min <= s && s <= seed_max)
        .collect()
}
