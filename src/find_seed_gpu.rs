extern "C" {
    fn find_seed_gpu();
}

#[test]
fn test() {
    unsafe {
        find_seed_gpu();
    }
}
