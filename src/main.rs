#![feature(portable_simd)]
mod multi_mt;

use std::simd::{u32x8, SimdPartialEq};

use indicatif::ParallelProgressIterator;
use multi_mt::MultiMT19937;
use rayon::prelude::*;

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: usize = 63;

type IV = (u32, u32, u32, u32, u32, u32);
type Frame = usize;
type Seed = u32;

#[inline]
fn iv_to_u32(iv: IV) -> u32 {
    (iv.0 << 25) | (iv.1 << 20) | (iv.2 << 15) | (iv.3 << 10) | (iv.4 << 5) | iv.5
}

// Advances f_r + 5 - f_l frames
#[inline]
fn find_frame(
    mt: &mut MultiMT19937,
    iv: u32,
    (frame_l, frame_r): (Frame, Frame), // right-exclusive
) -> u32x8 {
    let mut curr = (mt.next_u5x8() << u32x8::splat(20))
        | (mt.next_u5x8() << u32x8::splat(15))
        | (mt.next_u5x8() << u32x8::splat(10))
        | (mt.next_u5x8() << u32x8::splat(5))
        | mt.next_u5x8();
    let mut frame: u32x8 = u32x8::splat(0);
    for i in frame_l..frame_r {
        curr = (curr & u32x8::splat(0x1FFFFFF)) << u32x8::splat(5) | mt.next_u5x8();
        frame = curr
            .simd_eq(u32x8::splat(iv))
            .select(u32x8::splat(i as u32), frame);
    }
    frame
}

fn find_frame_pair(
    seed_hi29: u32,
    iv1: u32,
    iv2: u32,
    (f1_l, f1_r): (Frame, Frame), // right-exclusive
    (f2_l, f2_r): (Frame, Frame), // right-exclusive
) -> Vec<(Seed, Frame, Frame)> {
    let seed_begin = seed_hi29 << 3;
    let mut mt = MultiMT19937::from_seed(u32x8::from_array([
        seed_begin,
        seed_begin | 1,
        seed_begin | 2,
        seed_begin | 3,
        seed_begin | 4,
        seed_begin | 5,
        seed_begin | 6,
        seed_begin | 7,
    ]));
    mt.discard(PRE_ADVANCE_FRAME + f1_l);

    // Advances f1_r + 5 - f1_l frames
    let f1 = find_frame(&mut mt, iv1, (f1_l, f1_r));
    if f1.simd_eq(u32x8::splat(0)).all() {
        return vec![];
    }

    mt.discard(f2_l - f1_r - 5);

    // Advances f2_r + 5 - f2_l frames
    let f2 = find_frame(&mut mt, iv2, (f2_l, f2_r));

    let mut results = Vec::new();
    for i in 0..8 {
        if f1[i] != 0 && f2[i] != 0 {
            results.push((seed_begin | (i as u32), f1[i] as usize, f2[i] as usize));
        }
    }
    results
}

fn find_seeds(
    iv1: IV,
    iv2: IV,
    frame_range1: (Frame, Frame),     // right-exclusive
    frame_range2: (Frame, Frame),     // right-exclusive
    (seed_min, seed_max): (u32, u32), // right-inclusive
) -> Vec<(Seed, Frame, Frame)> {
    let iv1 = iv_to_u32(iv1);
    let iv2 = iv_to_u32(iv2);

    let seed_hi29_l = seed_min >> 3;
    let seed_hi29_r = (seed_max >> 3) + 1;

    (seed_hi29_l..seed_hi29_r)
        .into_par_iter()
        .progress()
        .flat_map(|seed_hi29| find_frame_pair(seed_hi29, iv1, iv2, frame_range1, frame_range2))
        .filter(|(s, _, _)| seed_min <= *s && *s <= seed_max)
        .collect()
}

fn main() {
    let now = std::time::Instant::now();

    let result = find_seeds(
        (3, 5, 26, 31, 6, 19),
        (22, 27, 22, 1, 7, 27),
        (600, 800),
        (1500, 1700),
        (0x30000000, 0x40000000),
    );
    println!("Completed!");
    println!("Elapsed: {:?}", now.elapsed());

    for (seed, frame1, frame2) in result {
        println!("Seed: {:x}, Frame1: {}, Frame2: {}", seed, frame1, frame2);
    }
}
