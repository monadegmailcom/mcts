//! With the `tch` feature, embed libtorch's library directory as an rpath so binaries, tests and
//! examples find `libtorch*.so/.dylib` without `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`.
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_TCH_LIBTORCH_LIB");
    if std::env::var_os("CARGO_FEATURE_TCH").is_none() {
        return;
    }
    // Exported by torch-sys (`links = "tch"`).
    if let Ok(lib) = std::env::var("DEP_TCH_LIBTORCH_LIB") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
        // A CUDA libtorch registers CUDA in libtorch_cuda's static initializers. No symbol of it
        // is referenced, so the linker's --as-needed (default on Linux) drops it and libtorch sees
        // no GPU (`tch::Cuda::is_available()` is false).
        if std::path::Path::new(&lib).join("libtorch_cuda.so").exists() {
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda,--as-needed");
        }
    }
}
