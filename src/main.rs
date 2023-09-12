use indicatif::ParallelProgressIterator;
use mersenne_twister::MT19937;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: usize = 63;

type IV = (u32, u32, u32, u32, u32, u32);
type Frame = usize;
type Seed = u32;

#[inline]
fn discard(mt: &mut MT19937, n: usize) {
    for _ in 0..n {
        mt.next_u32();
    }
}

#[inline]
fn iv_to_u32(iv: IV) -> u32 {
    (iv.0 << 25) | (iv.1 << 20) | (iv.2 << 15) | (iv.3 << 10) | (iv.4 << 5) | iv.5
}

#[inline]
fn next(mt: &mut MT19937) -> u32 {
    mt.next_u32() >> 27
}

fn find_frame_pair(
    seed: Seed,
    iv1: u32,
    iv2: u32,
    (f1_l, f1_r): (Frame, Frame), // right-exclusive
    (f2_l, f2_r): (Frame, Frame), // right-exclusive
) -> Option<(Seed, Frame, Frame)> {
    let mut mt = MT19937::from_seed(seed);
    discard(&mut mt, PRE_ADVANCE_FRAME + f1_l);

    let mut curr = iv_to_u32((
        0,
        next(&mut mt),
        next(&mut mt),
        next(&mut mt),
        next(&mut mt),
        next(&mut mt),
    ));
    let mut f1: usize = 0;
    for i in f1_l..f1_r {
        curr = (curr & 0b11111_11111_11111_11111_11111) << 5 | next(&mut mt);
        if curr == iv1 {
            f1 = i;
        }
    }
    if f1 == 0 {
        return None;
    }

    discard(&mut mt, f2_l - f1_r - 5);

    curr = iv_to_u32((
        0,
        next(&mut mt),
        next(&mut mt),
        next(&mut mt),
        next(&mut mt),
        next(&mut mt),
    ));
    let mut f2: usize = 0;
    for i in f2_l..f2_r {
        curr = (curr & 0b11111_11111_11111_11111_11111) << 5 | next(&mut mt);
        if curr == iv2 {
            f2 = i;
        }
    }
    if f2 == 0 {
        return None;
    }

    Some((seed, f1, f2))
}

fn find_seeds(
    iv1: IV,
    iv2: IV,
    frame_range1: (Frame, Frame),       // right-exclusive
    frame_range2: (Frame, Frame),       // right-exclusive
    (seed_min, seed_max): (Seed, Seed), // right-inclusive
) -> Vec<(Seed, Frame, Frame)> {
    let iv1 = iv_to_u32(iv1);
    let iv2 = iv_to_u32(iv2);

    (seed_min..=seed_max)
        .into_par_iter()
        .progress_count((seed_max as u64) - (seed_min as u64) + 1)
        .flat_map(|seed| find_frame_pair(seed, iv1, iv2, frame_range1, frame_range2))
        .collect()
}

fn main() {
    let now = std::time::Instant::now();

    let result = find_seeds(
        (3, 5, 26, 31, 6, 19),
        (22, 27, 22, 1, 7, 27),
        (600, 800),
        (1500, 1700),
        (0x30000000, 0x3fffffff),
    );
    println!("Completed!");
    println!("Elapsed: {:?}", now.elapsed());

    for (seed, frame1, frame2) in result {
        println!("Seed: {:x}, Frame1: {}, Frame2: {}", seed, frame1, frame2);
    }
}
