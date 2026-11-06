use std::env;
use std::path::PathBuf;

/// Compiles and links tau.rs's native kernels for every build.
fn main() -> cudaforge::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    // Watching the directory makes Cargo rerun this script when a new kernel
    // source is added, before cudaforge can emit per-file dependency tracking.
    println!("cargo:rerun-if-changed=src/kernels");
    println!("cargo:rerun-if-env-changed=TAU_CUDA_ARCH");

    let output_dir = PathBuf::from(
        env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR to build scripts"),
    );
    let toolkit = cudaforge::CudaToolkit::detect()?;
    let mut builder = cudaforge::KernelBuilder::new()
        .source_dir("src/kernels")
        .arg("-std=c++17")
        .arg("-O3");
    if let Ok(architecture) = env::var("TAU_CUDA_ARCH") {
        builder = builder.compute_cap_arch(&architecture);
    }
    builder.build_lib(output_dir.join("libtau_cuda_kernels.a"))?;

    println!("cargo:rustc-link-search=native={}", output_dir.display());
    // The standalone crate test harness does not depend on Candle/cudarc, so it
    // needs its own search path for the CUDA runtime and cuBLAS shared libraries.
    println!(
        "cargo:rustc-link-search=native={}",
        toolkit.lib_dir.display()
    );
    println!("cargo:rustc-link-lib=static=tau_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !env::var("TARGET").is_ok_and(|target| target.contains("msvc")) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}
