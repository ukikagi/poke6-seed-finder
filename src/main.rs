#![feature(portable_simd)]
mod multi_mt;

use dialoguer::Input;
use indicatif::{ParallelProgressIterator, ProgressBar, WeakProgressBar};
use multi_mt::MultiMT19937;
use multiversion::multiversion;
use rayon::prelude::*;
use std::{
    io,
    simd::{u32x8, Simd, SimdPartialEq},
};

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: u32 = 63;
const W: u32 = 16;

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
fn find_frame_pair(
    seed_hi: u32,
    ivs1: u32,
    ivs2: u32,
    (f1_min, f1_max): (Frame, Frame), // right-closed
    (f2_min, f2_max): (Frame, Frame), // right-closed
    wpb: WeakProgressBar,
) -> Vec<(Seed, Frame, Frame)> {
    let mut results = Vec::new();
    let mut mt = MultiMT19937::default();

    for s in (seed_hi << W..(seed_hi + 1) << W).step_by(8) {
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
                let seed = s | (i as u32);
                results.push((seed, f1[i], f2[i]));

                if let Some(pb) = wpb.upgrade() {
                    pb.println(format!(
                        "Hit! => Seed: {:08X}, Frame1: {}, Frame2: {}",
                        seed, f1[i], f2[i]
                    ));
                }
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

    let seed_hi_l = seed_min >> W;
    let seed_hi_r = (seed_max >> W) + 1;

    let progress_bar = ProgressBar::new((seed_hi_r - seed_hi_l) as u64);
    let wpb = progress_bar.downgrade();

    (seed_hi_l..seed_hi_r)
        .into_par_iter()
        .progress_with(progress_bar)
        .flat_map(|seed_hi| {
            find_frame_pair(seed_hi, iv1, iv2, frame_range1, frame_range2, wpb.clone())
        })
        .filter(|&(s, _, _)| seed_min <= s && s <= seed_max)
        .collect()
}

fn input_vec(prompt: &str, init: &str, radix: u32) -> Vec<u32> {
    Input::<String>::new()
        .with_prompt(prompt)
        .with_initial_text(init)
        .interact_text()
        .unwrap()
        .trim()
        .split(" ")
        .map(|x| u32::from_str_radix(x, radix).unwrap())
        .collect()
}

fn main() -> io::Result<()> {
    let ivs1 = input_vec("IVs of Wild1", "3 5 26 31 6 19", 10);
    assert!(ivs1.len() == 6 && ivs1.iter().all(|&iv| iv <= 31));

    let frame1 = input_vec("Frames of Wild1", "600 800", 10);
    assert!(frame1.len() == 2);

    let ivs2 = input_vec("IVs of Wild2", "22 27 22 1 7 27", 10);
    assert!(ivs2.len() == 6 && ivs2.iter().all(|&iv| iv <= 31));

    let frame2 = input_vec("Frames of Wild1", "1500 1700", 10);
    assert!(frame2.len() == 2);

    assert!(
        frame1[0] < frame1[1]
            && frame1[1] + 6 < frame2[0]
            && frame2[0] < frame2[1]
            && frame2[1] < 10000
    );

    let seed_range = input_vec("Seed range", "00000000 FFFFFFFF", 16);
    assert!(seed_range.len() == 2);

    println!();
    let now = std::time::Instant::now();

    let result = find_seeds(
        (seed_range[0], seed_range[1]),
        (ivs1[0], ivs1[1], ivs1[2], ivs1[3], ivs1[4], ivs1[5]),
        (ivs2[0], ivs2[1], ivs2[2], ivs2[3], ivs2[4], ivs2[5]),
        (frame1[0], frame1[1]),
        (frame2[0], frame2[1]),
    );

    println!("Done!");
    println!("Elapsed: {:?}", now.elapsed());

    println!();
    println!("Results:");
    for (seed, frame1, frame2) in result {
        println!(
            "- Seed: {:08X}, Frame1: {}, Frame2: {}",
            seed, frame1, frame2
        );
    }
    println!();

    let _ = Input::<String>::new()
        .with_prompt("Press Enter to quit")
        .allow_empty(true)
        .interact();
    Ok(())
}
