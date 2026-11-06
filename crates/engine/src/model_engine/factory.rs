//! Model-family engine dispatch, device setup and post-load KV configuration.
//! Checkpoint parsing/allocation remain in the existing model loader. Graph
//! eligibility is checked before loading tensors onto a private capture device.

use super::kv_budget::KvCacheBudget;
use super::{EngineOptions, model_id_from_path};
use crate::chat::{ChatInferenceEngine, ChatTemplateSelection};
use crate::model::{ModelDType, ModelLoadError, ModelLoadOptions};
use crate::qwen2_engine::Qwen2SafetensorsEngine;
use crate::qwen3_5_engine::Qwen3_5SafetensorsEngine;
use crate::qwen3_5_moe_engine::Qwen3_5MoeSafetensorsEngine;
use crate::qwen3_engine::Qwen3SafetensorsEngine;
use candle::Device;
use std::path::{Path, PathBuf};
use tau_models::CausalLanguageModel;
use tau_models::{LoadedCausalModel, LoadedModel, load_model_with_options};

/// One model instance still serves requests serially. This is a budget divisor,
/// not a concurrency implementation; a future scheduler must own separate state.
const CONCURRENT_REQUESTS: usize = 1;

/// Loads a supported model directory into its matching chat inference engine.
///
/// The model directory's `config.json` determines the model family. The shared
/// loader initializes the corresponding Candle model, and this factory wraps it in
/// the family-specific engine responsible for generation and chat formatting.
pub fn load_model_engine(
    path: impl AsRef<Path>,
    device: Device,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    load_model_engine_with_dtype(path, device, ModelDType::F32)
}

/// Loads a supported model using an explicitly selected runtime dtype.
///
/// The model directory declares which reduced-precision dtypes it supports. FP32
/// remains available through `load_model_engine`, while callers that select BF16
/// use this function and receive a startup error when the artifact does not declare
/// BF16 support.
pub fn load_model_engine_with_dtype(
    path: impl AsRef<Path>,
    device: Device,
    dtype: ModelDType,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    load_model_engine_with_runtime_options(
        path,
        device,
        dtype,
        EngineOptions::default(),
    )
}

/// Loads a model with explicit construction options and default serving options.
pub fn load_model_engine_with_options(
    path: impl AsRef<Path>,
    device: Device,
    dtype: ModelDType,
    options: ModelLoadOptions,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    load_model_engine_with_runtime_options(
        path,
        device,
        dtype,
        EngineOptions::from(options),
    )
}

/// Constructs a model and engine from one authoritative configuration.
/// Only derived construction settings enter the loader; runtime options stay in
/// the engine.
/// The same kernel-policy value drives model construction and later trace labels.
pub fn load_model_engine_with_runtime_options(
    path: impl AsRef<Path>,
    device: Device,
    dtype: ModelDType,
    engine_options: EngineOptions,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    load_model_engine_with_chat_template(path, device, dtype, engine_options, None)
}

/// Loads an engine with construction-only chat-template selection. Existing load
/// entry points retain automatic checkpoint selection; runtime options stay Copy.
/// Overrides are supported for the shared Qwen3.5 dense/MoE chat paths.
pub fn load_model_engine_with_chat_template(
    path: impl AsRef<Path>,
    device: Device,
    dtype: ModelDType,
    engine_options: EngineOptions,
    chat_template: Option<&Path>,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    let selection = chat_template
        .map(|path| ChatTemplateSelection::new().with_file(path))
        .unwrap_or_default();
    load_model_engine_with_chat_template_selection(
        path,
        device,
        dtype,
        engine_options,
        &selection,
    )
}

/// Loads an engine with startup template selection independent of runtime options.
/// Existing loading APIs delegate here with automatic or file-only selection.
pub fn load_model_engine_with_chat_template_selection(
    path: impl AsRef<Path>,
    device: Device,
    dtype: ModelDType,
    engine_options: EngineOptions,
    selection: &ChatTemplateSelection,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    if engine_options.prefill_chunk_size() == 0 {
        return Err(ModelLoadError::InvalidConfig(
            "prefill chunk size must be positive".to_owned(),
        ));
    }
    if engine_options.attention_block_size() != crate::AttentionBlockSize::default()
        && (engine_options.cuda_graph() != crate::CudaGraphMode::Full
            || !device.is_cuda()
            || dtype != ModelDType::BF16
            || !engine_options.optimize_kernel())
    {
        return Err(ModelLoadError::InvalidConfig(
            "non-default attention block size requires CUDA/BF16, optimized \
             kernels and --cuda-graph full"
                .to_owned(),
        ));
    }
    if engine_options.moe_prefill_config().is_some() {
        if !device.is_cuda()
            || dtype != ModelDType::BF16
            || !engine_options.optimize_kernel()
        {
            return Err(ModelLoadError::InvalidConfig(
                "explicit MoE prefill configuration requires CUDA/BF16 and optimized kernels".to_owned()));
        }
        let metadata = tau_models::load_model_config(path.as_ref())?;
        if !matches!(
            metadata.model_type.as_str(),
            "qwen3_5_moe" | "qwen3_5_moe_text"
        ) {
            return Err(ModelLoadError::InvalidConfig(
                "explicit MoE prefill configuration requires a Qwen3.5/3.6 MoE model"
                    .to_owned(),
            ));
        }
    }
    let root = path.as_ref().to_path_buf();
    if engine_options.cuda_graph().is_enabled() {
        load_cuda_graph_model_engine(root, device, dtype, engine_options, selection)
    } else {
        load_eager_model_engine(root, device, dtype, engine_options, selection)
    }
}

/// Loads every supported family on the caller's ordinary execution device.
/// Graph fallbacks retain the requested options so traces still report the reason.
fn load_eager_model_engine(
    root: PathBuf,
    device: Device,
    dtype: ModelDType,
    engine_options: EngineOptions,
    selection: &ChatTemplateSelection,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    let loaded = load_model_with_options(
        &root,
        &device,
        dtype,
        ModelLoadOptions::from(engine_options),
    )?;

    if selection.is_explicit()
        && !matches!(
            &loaded,
            LoadedModel::Qwen3_5(_) | LoadedModel::Qwen3_5Moe(_)
        )
    {
        return Err(ModelLoadError::InvalidConfig(
            "chat template selection is supported for Qwen3.5 dense/MoE engines"
                .to_owned(),
        ));
    }
    match loaded {
        LoadedModel::Qwen2(loaded) => Ok(Box::new(
            Qwen2SafetensorsEngine::from_loaded(root, device, loaded, engine_options),
        )),
        LoadedModel::Qwen3(loaded) => Ok(Box::new(
            Qwen3SafetensorsEngine::from_loaded(root, device, loaded, engine_options),
        )),
        LoadedModel::Qwen3_5(mut loaded) => {
            let engine_options =
                configure_kv_cache_budget(&mut loaded, &device, engine_options)?;
            Ok(Box::new(Qwen3_5SafetensorsEngine::from_loaded(
                root,
                device,
                loaded,
                engine_options,
                selection,
            )?))
        }
        LoadedModel::Qwen3_5Moe(mut loaded) => {
            let engine_options =
                configure_kv_cache_budget(&mut loaded, &device, engine_options)?;
            Ok(Box::new(Qwen3_5MoeSafetensorsEngine::from_loaded(
                root,
                device,
                loaded,
                engine_options,
                selection,
            )?))
        }
    }
}

/// Resolves graph support before allocating a model or creating a capture device.
fn load_cuda_graph_model_engine(
    root: PathBuf,
    device: Device,
    dtype: ModelDType,
    engine_options: EngineOptions,
    selection: &ChatTemplateSelection,
) -> Result<Box<dyn ChatInferenceEngine>, ModelLoadError> {
    let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
        eprintln!(
            "tau-engine CUDA graph policy {}: eager (unsupported_device)",
            engine_options.cuda_graph()
        );
        return load_eager_model_engine(root, device, dtype, engine_options, selection);
    };
    let metadata = tau_models::load_model_config(&root)?;
    let model_name = model_id_from_path(&root);
    let dense = metadata.architecture == "Qwen3_5ForConditionalGeneration"
        && metadata.model_type == "qwen3_5";
    let moe_decode = metadata.architecture == "Qwen3_5MoeForConditionalGeneration"
        && metadata.model_type == "qwen3_5_moe"
        && engine_options.cuda_graph() == crate::CudaGraphMode::Full;
    if !dense && !moe_decode {
        if engine_options.attention_block_size() != crate::AttentionBlockSize::default()
        {
            return Err(ModelLoadError::InvalidConfig(
                "non-default attention block size requires the Qwen3.5 \
                 prepared decode path"
                    .to_owned(),
            ));
        }
        eprintln!(
            "tau-engine CUDA graph policy {}: eager (unsupported_model_family)",
            engine_options.cuda_graph()
        );
        return load_eager_model_engine(root, device, dtype, engine_options, selection);
    }
    let reason = if !engine_options.optimize_kernel() {
        Some("optimized_kernels_disabled")
    } else if dtype != ModelDType::BF16 {
        Some("unsupported_dtype")
    } else {
        None
    };
    if let Some(reason) = reason {
        eprintln!(
            "tau-engine CUDA graph policy {}: eager ({reason})",
            engine_options.cuda_graph()
        );
        return load_eager_model_engine(root, device, dtype, engine_options, selection);
    }
    if moe_decode {
        eprintln!(
            "tau-engine CUDA graph policy full: MoE prefill stays eager; \
             decode selects a complete prepared forward when supported"
        );
    } else {
        eprintln!(
            "tau-engine CUDA graph policy {}: prefill uses eligible GDN-layer and \
         attention-MLP graphs; short chunks stay eager{}",
            engine_options.cuda_graph(),
            if engine_options.cuda_graph() == crate::CudaGraphMode::Full {
                "; decode selects a complete prepared forward when supported, \
             otherwise narrower coverage"
            } else {
                "; decode graphs stay within each layer and attention remains eager"
            }
        );
    }
    eprintln!(
        "tau-engine prepared attention block size: {} context tokens; \
         CUDA threads per block: 256",
        engine_options.attention_block_size()
    );

    // The caller's device may already have tensors. Never disable its tracking.
    // Create a separate device handle before allocating any model tensors instead.
    // SAFETY: This fresh device and all its tensors stay private to the engine.
    // Graph ownership makes these engines !Send/!Sync; &mut self serializes
    // prefill, input writes, reset, replay and readback on its original thread.
    let device = unsafe { tau_models::new_capture_device(gpu_id) }
        .map_err(|error| ModelLoadError::Candle(error.to_string()))?;
    let loaded = load_model_with_options(
        &root,
        &device,
        dtype,
        ModelLoadOptions::from(engine_options),
    )?;
    match loaded {
        LoadedModel::Qwen3_5(mut loaded) => {
            loaded.model.set_cuda_graph(engine_options.cuda_graph());
            let engine_options =
                configure_kv_cache_budget(&mut loaded, &device, engine_options)?;
            Ok(Box::new(Qwen3_5SafetensorsEngine::from_loaded(
                root,
                device,
                loaded,
                engine_options,
                selection,
            )?))
        }
        LoadedModel::Qwen3_5Moe(mut loaded) => {
            loaded.model.set_cuda_graph(engine_options.cuda_graph());
            let engine_options =
                configure_kv_cache_budget(&mut loaded, &device, engine_options)?;
            Ok(Box::new(Qwen3_5MoeSafetensorsEngine::from_loaded(
                root,
                device,
                loaded,
                engine_options,
                selection,
            )?))
        }
        _ => Err(ModelLoadError::InvalidConfig(format!(
            "CUDA graph loading is not supported for {model_name}"
        ))),
    }
}

/// Capacity is counted in token slots across all attention layers, not bytes in
/// one layer. Always use the loaded model's real head counts and runtime dtype.
fn configure_kv_cache_budget<M: tau_models::CausalLanguageModel>(
    loaded: &mut LoadedCausalModel<M>,
    device: &Device,
    mut options: EngineOptions,
) -> Result<EngineOptions, ModelLoadError> {
    if !options.auto_kv_cache_budget() || !device.is_cuda() {
        return Ok(options);
    }
    let (kv_bytes, workspace_bytes) = loaded
        .model
        .kv_memory_per_token(options.prefill_chunk_size().min(loaded.context_window))
        .map_err(|e| ModelLoadError::Candle(e.to_string()))?;
    if kv_bytes == 0 {
        return Ok(options);
    }
    let cuda = device
        .as_cuda_device()
        .map_err(|e| ModelLoadError::Candle(e.to_string()))?;
    let stream = cuda.cuda_stream();
    stream
        .synchronize()
        .map_err(|e| ModelLoadError::Candle(e.to_string()))?;
    let (free, total) = stream
        .context()
        .mem_get_info()
        .map_err(|e| ModelLoadError::Candle(e.to_string()))?;
    let budget = KvCacheBudget::from_memory(
        free,
        total,
        kv_bytes,
        workspace_bytes,
        CONCURRENT_REQUESTS,
    )?;
    let capacity = budget.per_request_capacity(loaded.context_window);
    loaded
        .model
        .set_kv_cache_capacity(capacity)
        .map_err(|e| ModelLoadError::Candle(e.to_string()))?;
    eprintln!(
        "tau-engine KV cache budget: max_kv_cache_capacity={} tokens, \
         concurrent_requests={}, per_request_capacity={} tokens, \
         kv_bytes_per_token={}, workspace_bytes_per_token={}, headroom_bytes={}, \
         free_vram_bytes={}, total_vram_bytes={}",
        budget.max_kv_cache_capacity,
        budget.concurrent_requests,
        capacity,
        kv_bytes,
        workspace_bytes,
        budget.headroom_bytes,
        free,
        total,
    );
    options.kv_cache_budget = Some(budget);
    Ok(options)
}
