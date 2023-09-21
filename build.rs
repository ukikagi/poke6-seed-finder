extern crate cc;

fn main() {
    println!("cargo:rerun-if-changed=src/kernel.cu");

    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .file("src/kernel.cu")
        .compile("kernel.a");
}
