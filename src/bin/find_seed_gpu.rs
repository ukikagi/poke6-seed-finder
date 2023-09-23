type IVs = (u32, u32, u32, u32, u32, u32);
type Frame = u32;

const W: u32 = 20;

#[inline]
fn encode_ivs(iv: IVs) -> u32 {
    (iv.0 << 25) | (iv.1 << 20) | (iv.2 << 15) | (iv.3 << 10) | (iv.4 << 5) | iv.5
}

extern "C" {
    fn find_seed_gpu(
        seed_min: u32,
        seed_max: u32,
        ivs1: u32,
        ivs2: u32,
        f1_min: u32,
        f1_max: u32,
        f2_min: u32,
        f2_max: u32,
    );
}

fn find_frame_pair(
    seed_hi: u32,
    ivs1: u32,
    ivs2: u32,
    (f1_min, f1_max): (Frame, Frame), // right-closed
    (f2_min, f2_max): (Frame, Frame), // right-closed
) {
    let seed_min = seed_hi << W;
    let seed_max = seed_min | ((1 << W) - 1);
    println!("seed_range: {:08X} - {:08X}", seed_min, seed_max);
    unsafe {
        find_seed_gpu(
            seed_min, seed_max, ivs1, ivs2, f1_min, f1_max, f2_min, f2_max,
        );
    }
}

fn find_seeds(
    (seed_min, seed_max): (u32, u32), // right-closed
    ivs1: IVs,
    ivs2: IVs,
    frame_range1: (Frame, Frame), // right-closed
    frame_range2: (Frame, Frame), // right-closed
) {
    let iv1 = encode_ivs(ivs1);
    let iv2 = encode_ivs(ivs2);

    let seed_hi_l = seed_min >> W;
    let seed_hi_r = (seed_max >> W) + 1;

    (seed_hi_l..seed_hi_r)
        .for_each(|seed_hi| find_frame_pair(seed_hi, iv1, iv2, frame_range1, frame_range2));
}

fn main() {
    let now = std::time::Instant::now();
    find_seeds(
        (0xDE000000, 0xDEFFFFFF),
        (11, 7, 6, 7, 6, 7),
        (5, 8, 1, 2, 14, 12),
        (600, 800),
        (1500, 1700),
    );
    println!("Done!");
    println!("Elapsed: {:?}", now.elapsed());
}
