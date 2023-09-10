use mersenne_twister::MT19937;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

//  Misc consumption 60 + 1 + EC + PID
const PRE_ADVANCE_FRAME: usize = 63;

type IV = (u8, u8, u8, u8, u8, u8);
type Frame = usize;
type Seed = u32;

fn find_frame(rand_pool: &[u8], iv: IV, (frame_min, frame_max): (Frame, Frame)) -> Option<Frame> {
    for i in frame_min..=frame_max {
        if rand_pool[i + 0] == iv.0
            && rand_pool[i + 1] == iv.1
            && rand_pool[i + 2] == iv.2
            && rand_pool[i + 3] == iv.3
            && rand_pool[i + 4] == iv.4
            && rand_pool[i + 5] == iv.5
        {
            return Some(i);
        }
    }
    None
}

fn find_frame_pair(
    seed: Seed,
    iv1: IV,
    iv2: IV,
    frame_range1: (Frame, Frame), // right-inclusive
    frame_range2: (Frame, Frame), // right-inclusive
) -> Option<(Seed, Frame, Frame)> {
    let mut mt = MT19937::from_seed(seed);
    for _ in 0..PRE_ADVANCE_FRAME {
        mt.next_u32();
    }

    let pool_size = frame_range2.1 + 6;
    let mut rand_pool: Vec<u8> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        rand_pool.push((mt.next_u32() >> 27) as u8)
    }

    Some((
        seed,
        find_frame(&rand_pool, iv1, frame_range1)?,
        find_frame(&rand_pool, iv2, frame_range2)?,
    ))
}

fn find_seeds(
    iv1: IV,
    iv2: IV,
    frame_range1: (Frame, Frame),       // right-inclusive
    frame_range2: (Frame, Frame),       // right-inclusive
    (seed_min, seed_max): (Seed, Seed), // right-inclusive
) -> Vec<(Seed, Frame, Frame)> {
    (seed_min..=seed_max)
        .into_par_iter()
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
        (0x33000000, 0x34000000),
    );
    for (seed, frame1, frame2) in result {
        println!("{:x} {} {}", seed, frame1, frame2);
    }

    println!("{:?}", now.elapsed().as_secs())
}
