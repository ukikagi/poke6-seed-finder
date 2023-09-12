use boyer_moore_magiclen::byte::BMByte;
use indicatif::ParallelProgressIterator;
use mersenne_twister::MT19937;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: usize = 63;

type IV = (u8, u8, u8, u8, u8, u8);
type Frame = usize;
type Seed = u32;

fn discard(rng: &mut MT19937, n: usize) {
    for _ in 0..n {
        rng.next_u32();
    }
}

fn find_frame_pair(
    seed: Seed,
    bmb1: &BMByte,
    bmb2: &BMByte,
    (f1_min, f1_max): (Frame, Frame), // right-exclusive
    (f2_min, f2_max): (Frame, Frame), // right-exclusive
) -> Option<(Seed, Frame, Frame)> {
    let mut mt = MT19937::from_seed(seed);
    discard(&mut mt, PRE_ADVANCE_FRAME + f1_min);

    let mut rand_pool: Vec<u8> = Vec::with_capacity(f1_max - f1_min);
    for _ in f1_min..f1_max {
        rand_pool.push((mt.next_u32() >> 27) as u8)
    }
    let f1 = f1_min + bmb1.find_first_in(&rand_pool)?;

    discard(&mut mt, f2_min - f1_max);

    rand_pool.clear();
    rand_pool.reserve(f2_max - f2_min);
    for _ in f2_min..f2_max {
        rand_pool.push((mt.next_u32() >> 27) as u8)
    }
    let f2 = f2_min + bmb2.find_first_in(&rand_pool)?;

    Some((seed, f1, f2))
}

fn find_seeds(
    iv1: IV,
    iv2: IV,
    frame_range1: (Frame, Frame),       // right-exclusive
    frame_range2: (Frame, Frame),       // right-exclusive
    (seed_min, seed_max): (Seed, Seed), // right-inclusive
) -> Vec<(Seed, Frame, Frame)> {
    let bmb1 = BMByte::from(vec![iv1.0, iv1.1, iv1.2, iv1.3, iv1.4, iv1.5]).unwrap();
    let bmb2 = BMByte::from(vec![iv2.0, iv2.1, iv2.2, iv2.3, iv2.4, iv2.5]).unwrap();

    (seed_min..=seed_max)
        .into_par_iter()
        .progress_count((seed_max as u64) - (seed_min as u64) + 1)
        .flat_map(|seed| find_frame_pair(seed, &bmb1, &bmb2, frame_range1, frame_range2))
        .collect()
}

fn main() {
    let now = std::time::Instant::now();

    let result = find_seeds(
        (3, 5, 26, 31, 6, 19),
        (22, 27, 22, 1, 7, 27),
        (600, 800),
        (1500, 1700),
        (0x00000000, 0xffffffff),
    );
    println!("Completed!");
    println!("Elapsed: {:?}", now.elapsed());

    for (seed, frame1, frame2) in result {
        println!("Seed: {:x}, Frame1: {}, Frame2: {}", seed, frame1, frame2);
    }
}
