#![feature(portable_simd)]
mod find_seed;
mod multi_mt;

use dialoguer::Input;
use indicatif::ProgressBar;
use std::io;

use crate::find_seed::{Frame, Seed};

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
    // TODO: Clean up these validation logic

    let ivs1 = input_vec("IVs of Wild1", "11 7 6 7 6 7", 10);
    assert!(
        ivs1.len() == 6 && ivs1.iter().all(|&iv| iv <= 31),
        "IVs must be space-delimited 6 integers between 0 and 31."
    );

    let frame1 = input_vec("Frames of Wild1", "600 800", 10);
    assert!(
        frame1.len() == 2,
        "Frames must be space-delimited 2 integers."
    );
    assert!(
        frame1[0] <= frame1[1],
        "Min frame must be smaller than max frame."
    );
    assert!(frame1[1] <= 3000, "Frames must be <= 3000");

    let ivs2 = input_vec("IVs of Wild2", "5 8 1 2 14 12", 10);
    assert!(
        ivs2.len() == 6 && ivs2.iter().all(|&iv| iv <= 31),
        "IVs must be space-delimited 6 integers between 0 and 31."
    );

    let frame2 = input_vec("Frames of Wild2", "1500 1700", 10);
    assert!(
        frame2.len() == 2,
        "Frames must be space-delimited 2 integers."
    );
    assert!(
        frame2[0] <= frame2[1],
        "Min frame must be smaller than max frame."
    );
    assert!(frame2[1] <= 3000, "Frames must be <= 3000");

    assert!(
        frame1[1] + 6 <= frame2[0],
        "Min frame of Wild2 must be >= Max frame of Wild1 + 6"
    );

    let seed_range = input_vec("Seed range", "00000000 FFFFFFFF", 16);
    assert!(
        seed_range.len() == 2,
        "Seed range must be space-delimited 2 integers in hex."
    );

    println!();
    let now = std::time::Instant::now();

    let pb = ProgressBar::new(0);
    let notify = move |res: &[(Seed, Frame, Frame)], len: u32| {
        pb.set_length(len as u64);
        for (s, f1, f2) in res {
            let msg = format!("Hit! => Seed: {:08X}, Frame1: {}, Frame2: {}", s, f1, f2);
            pb.println(msg);
        }
        pb.inc(1);
    };

    let result = find_seed::find_seed(
        (seed_range[0], seed_range[1]),
        (ivs1[0], ivs1[1], ivs1[2], ivs1[3], ivs1[4], ivs1[5]),
        (ivs2[0], ivs2[1], ivs2[2], ivs2[3], ivs2[4], ivs2[5]),
        (frame1[0], frame1[1]),
        (frame2[0], frame2[1]),
        notify,
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
        .with_prompt("Press Ctrl+C to quit")
        .interact();
    Ok(())
}
