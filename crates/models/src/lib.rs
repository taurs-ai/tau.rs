//! Architecture façades, reusable model components, and loading utilities.

pub(crate) mod components;
pub mod dtype;
mod error;
mod facade;
mod generation_config;
mod interface;
mod loader;
mod metadata;
mod observations;
mod options;
pub(crate) mod qwen2;
pub(crate) mod qwen3;
pub(crate) mod qwen3_5;
pub(crate) mod qwen3_5_common;
pub(crate) mod qwen3_5_moe;
pub mod safetensors;
pub(crate) mod tensor_contract;
mod tokenizer;

#[cfg(test)]
mod test_support;

pub use dtype::ModelDType;
pub use error::ModelLoadError;
pub use options::CudaGraphStats;
pub(crate) use options::GraphPhase;
pub use options::{AttentionBlockSize, CudaGraphMode, ModelLoadOptions};
pub use safetensors::{ModelShard, TensorInfo};
pub use tau_cuda_kernels::{
    MoeDecodeConfig, MoeDecodeOptions, MoeDecodeProfile, MoeDecodeResolution,
    MoeDecodeSelection, MoePrefillConfig, MoePrefillTile,
};

pub use components::new_capture_device;
pub use facade::{Qwen3_5Model, Qwen3_5MoeModel};
pub use generation_config::{DEFAULT_MAX_NEW_TOKENS, GenerationConfig};
pub use interface::CausalLanguageModel;
pub use loader::{LoadedCausalModel, LoadedModel, load_model, load_model_with_options};
pub use metadata::{ModelConfig, load_model_config};
pub use qwen2::Qwen2ForCausalLM;
pub use qwen3::Qwen3ForCausalLM;

pub use tokenizer::{load_tokenizer, validate_tokenizer};
