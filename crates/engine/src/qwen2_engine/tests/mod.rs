use std::path::PathBuf;

const QWEN_TEST_MODEL: &str = "/mnt/shared/models/Qwen2.5-0.5B-Instruct";

mod generation;
mod kv_cache;
mod tokenization;

/// Resolves the shared Qwen2.5 validation model used by ignored tests.
///
/// `TAU_QWEN_TEST_MODEL` may override the repository-local default mount. The
/// function returns the selected path without checking whether it exists so each
/// test can choose whether absence is a skip or a failure.
fn qwen_test_model_path() -> PathBuf {
    std::env::var("TAU_QWEN_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(QWEN_TEST_MODEL))
}
