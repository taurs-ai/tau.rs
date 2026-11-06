//! Options contracts for the shared causal engine.

use super::*;

/// Loader conversion preserves kernel policy and defaults engine-only options.
#[test]
fn engine_options_preserve_kernel_policy_at_the_loading_boundary() {
    let defaults = EngineOptions::default();
    assert_eq!(
        defaults.optimize_kernel(),
        ModelLoadOptions::default().optimize_kernel()
    );
    assert_eq!(defaults.cuda_graph(), crate::CudaGraphMode::None);
    assert_eq!(
        EngineOptions::new().cuda_graph(),
        crate::CudaGraphMode::None
    );
    assert!(!defaults.tracing());
    assert!(!defaults.metrics());
    assert!(!defaults.vram_metrics());
    for optimize_kernel in [false, true] {
        for cuda_graph in [
            crate::CudaGraphMode::None,
            crate::CudaGraphMode::Layer,
            crate::CudaGraphMode::Full,
        ] {
            for tracing in [false, true] {
                let engine = EngineOptions::new()
                    .with_optimize_kernel(optimize_kernel)
                    .with_cuda_graph(cuda_graph)
                    .with_tracing(tracing);
                let load = ModelLoadOptions::from(engine);
                assert_eq!(load.optimize_kernel(), optimize_kernel);
                assert_eq!(engine.cuda_graph(), cuda_graph);
                assert_eq!(engine.tracing(), tracing);
                let legacy = EngineOptions::from(load);
                assert_eq!(legacy.optimize_kernel(), optimize_kernel);
                assert_eq!(legacy.cuda_graph(), crate::CudaGraphMode::None);
                assert!(!legacy.tracing());
                assert!(!legacy.metrics());
                assert!(!legacy.vram_metrics());
            }
        }
    }
}

#[test]
fn vram_collection_is_independent_and_not_a_model_loading_option() {
    let enabled = EngineOptions::new().with_vram_metrics(true);
    assert!(enabled.vram_metrics());
    assert!(!enabled.metrics());
    assert!(!enabled.tracing());
    assert!(!enabled.with_vram_metrics(false).vram_metrics());
    let legacy = EngineOptions::from(ModelLoadOptions::from(enabled));
    assert!(!legacy.vram_metrics());
}

#[test]
fn attention_geometry_survives_both_loader_conversions() {
    use crate::AttentionBlockSize;
    for size in [
        AttentionBlockSize::Tokens32,
        AttentionBlockSize::Tokens64,
        AttentionBlockSize::Tokens128,
        AttentionBlockSize::Tokens256,
    ] {
        let engine = EngineOptions::new().with_attention_block_size(size);
        let load = ModelLoadOptions::from(engine);
        assert_eq!(load.attention_block_size(), size);
        assert_eq!(EngineOptions::from(load).attention_block_size(), size);
    }
}

/// Callers can explicitly clear a previously enabled CUDA graph preference.
#[test]
fn cuda_graph_preference_can_be_disabled() {
    let options = EngineOptions::new()
        .with_cuda_graph(crate::CudaGraphMode::Layer)
        .with_cuda_graph(crate::CudaGraphMode::None);
    assert_eq!(options.cuda_graph(), crate::CudaGraphMode::None);
}

/// Engine construction retains the graph preference alongside runtime options.
#[test]
fn engine_retains_cuda_graph_preference() {
    for cuda_graph in [
        crate::CudaGraphMode::None,
        crate::CudaGraphMode::Layer,
        crate::CudaGraphMode::Full,
    ] {
        let options = EngineOptions::new().with_cuda_graph(cuda_graph);
        let engine = sequence_engine(options);
        assert_eq!(engine.options, options);
        assert_eq!(engine.options.cuda_graph(), cuda_graph);
    }
}
