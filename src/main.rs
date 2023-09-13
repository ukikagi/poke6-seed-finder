#![feature(portable_simd)]
mod multi_mt;

use indicatif::ParallelProgressIterator;
use multi_mt::MultiMT19937;
use rayon::prelude::*;
use std::simd::{u32x8, Simd, SimdPartialEq};

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: u32 = 63;

type IVs = (u32, u32, u32, u32, u32, u32);
type Frame = u32;
type Seed = u32;

#[inline]
fn ivs_to_u32(iv: IVs) -> u32 {
    (iv.0 << 25) | (iv.1 << 20) | (iv.2 << 15) | (iv.3 << 10) | (iv.4 << 5) | iv.5
}

// Advances f_max + 6 - f_min frames
#[inline]
fn find_frame(
    mt: &mut MultiMT19937,
    ivs: u32,
    (f_min, f_max): (Frame, Frame), // right-exclusive
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

fn find_frame_pair(
    seed_hi16: u32,
    ivs1: u32,
    ivs2: u32,
    (f1_min, f1_max): (Frame, Frame), // right-closed
    (f2_min, f2_max): (Frame, Frame), // right-closed
) -> Vec<(Seed, Frame, Frame)> {
    let mut results = Vec::new();
    let mut mt = MultiMT19937::default();

    for s in (seed_hi16 << 16..(seed_hi16 + 1) << 16).step_by(8) {
        let seed = Simd::from_array([s, s | 1, s | 2, s | 3, s | 4, s | 5, s | 6, s | 7]);
        mt.reseed(seed);

        mt.discard(PRE_ADVANCE_FRAME + f1_min);

        // Advances f1_max + 6 - f1_min frames
        let f1 = find_frame(&mut mt, ivs1, (f1_min, f1_max));
        if f1.simd_eq(Simd::splat(0)).all() {
            continue;
        }

        mt.discard(f2_min - f1_max - 6);

        // Advances f2_max + 6 - f2_min frames
        let f2 = find_frame(&mut mt, ivs2, (f2_min, f2_max));

        for i in 0..8 {
            if f1[i] != 0 && f2[i] != 0 {
                results.push((s | (i as u32), f1[i], f2[i]));
            }
        }
    }
    results
}

fn find_seeds(
    (seed_min, seed_max): (u32, u32), // right-closed
    ivs1: IVs,
    ivs2: IVs,
    frame_range1: (Frame, Frame), // right-closed
    frame_range2: (Frame, Frame), // right-closed
) -> Vec<(Seed, Frame, Frame)> {
    let iv1 = ivs_to_u32(ivs1);
    let iv2 = ivs_to_u32(ivs2);

    let seed_hi16_l = seed_min >> 16;
    let seed_hi16_r = (seed_max >> 16) + 1;

    (seed_hi16_l..seed_hi16_r)
        .into_par_iter()
        .progress()
        .flat_map(|seed_hi16| find_frame_pair(seed_hi16, iv1, iv2, frame_range1, frame_range2))
        .filter(|(s, _, _)| seed_min <= *s && *s <= seed_max)
        .collect()
}

fn main() {
    let now = std::time::Instant::now();

    let result = find_seeds(
        (0x00000000, 0xffffffff),
        (3, 5, 26, 31, 6, 19),
        (22, 27, 22, 1, 7, 27),
        (600, 800),
        (1500, 1700),
    );
    println!("Completed!");
    println!("Elapsed: {:?}", now.elapsed());

    for (seed, frame1, frame2) in result {
        println!("Seed: {:08X}, Frame1: {}, Frame2: {}", seed, frame1, frame2);
    }
}
