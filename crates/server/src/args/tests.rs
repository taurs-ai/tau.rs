//! CLI defaults, accepted spellings, validation and engine-option propagation.

use super::*;

/// Both public spellings carry one construction-only name; file preference is
/// retained independently so source selection can apply file precedence.
#[test]
fn parses_named_templates_and_compatibility_alias() {
    let defaults = parse_args(std::iter::empty(), "tau-server").unwrap();
    for flag in ["--chat-template-name", "--hf-chat-template-name"] {
        for raw in [
            vec![flag.to_owned(), "alternate".to_owned()],
            vec![format!("{flag}=alternate")],
        ] {
            let parsed = parse_args(raw.into_iter(), "tau-server").unwrap();
            assert_eq!(parsed.chat_template_selection().name(), Some("alternate"));
            assert_eq!(parsed.engine_options(), defaults.engine_options());
        }
        for raw in [
            vec![flag.to_owned()],
            vec![format!("{flag}=")],
            vec![flag.to_owned(), String::new()],
        ] {
            assert!(parse_args(raw.into_iter(), "tau-server").is_err());
        }
    }
    let parsed = parse_args(
        [
            "--chat-template-name",
            "missing",
            "--chat-template",
            "/tmp/file.jinja",
        ]
        .into_iter()
        .map(str::to_owned),
        "tau-server",
    )
    .unwrap();
    assert_eq!(parsed.chat_template_selection().name(), Some("missing"));
    assert_eq!(
        parsed.chat_template_selection().file(),
        Some(std::path::Path::new("/tmp/file.jinja"))
    );
}

/// Both aliases fail before device initialization when no model was selected.
#[test]
fn named_template_selection_requires_a_model() {
    for flag in ["--chat-template-name", "--hf-chat-template-name"] {
        let parsed = parse_args(
            [flag, "default"].into_iter().map(str::to_owned),
            "tau-server",
        )
        .unwrap();
        let error = crate::load_engine(&parsed).err().unwrap();
        assert!(error.contains("requires --model"));
    }
}

/// A file override is construction-only and accepts both existing CLI forms.
#[test]
fn parses_chat_template_override_without_changing_runtime_options() {
    let default = parse_args(std::iter::empty(), "tau-server").unwrap();
    assert!(default.chat_template.is_none());
    for raw in [
        vec!["--chat-template", "/tmp/chat.jinja"],
        vec!["--chat-template=/tmp/chat.jinja"],
    ] {
        let parsed =
            parse_args(raw.into_iter().map(str::to_owned), "tau-server").unwrap();
        assert_eq!(
            parsed.chat_template.as_deref(),
            Some(std::path::Path::new("/tmp/chat.jinja"))
        );
        assert_eq!(parsed.engine_options(), default.engine_options());
    }
    for raw in [
        vec!["--chat-template"],
        vec!["--chat-template="],
        vec!["--chat-template", ""],
    ] {
        assert!(parse_args(raw.into_iter().map(str::to_owned), "tau-server").is_err());
    }
}

/// Omitting the flag leaves CUDA graphs disabled through the engine boundary.
#[test]
fn cuda_graph_is_disabled_by_default() {
    let args = parse_args(std::iter::empty(), "tau-server").unwrap();
    assert_eq!(args.cuda_graph, CudaGraphMode::None);
    assert_eq!(args.engine_options().cuda_graph(), CudaGraphMode::None);
}

/// All enum values reach the engine through both CLI argument forms.
#[test]
fn parses_cuda_graph_modes_with_both_cli_forms() {
    for (value, expected) in [
        ("none", CudaGraphMode::None),
        ("layer", CudaGraphMode::Layer),
        ("full", CudaGraphMode::Full),
    ] {
        for raw in [
            vec!["--cuda-graph".to_owned(), value.to_owned()],
            vec![format!("--cuda-graph={value}")],
        ] {
            let args = parse_args(raw.into_iter(), "tau-server").unwrap();
            assert_eq!(args.cuda_graph, expected);
            assert_eq!(args.engine_options().cuda_graph(), expected);
        }
    }
}

/// Each runtime option reaches the engine independently of the other options.
#[test]
fn propagates_independent_engine_options() {
    for cuda_graph in [
        CudaGraphMode::None,
        CudaGraphMode::Layer,
        CudaGraphMode::Full,
    ] {
        for optimize_kernel in [false, true] {
            for tracing in [false, true] {
                let mut raw = vec![
                    format!("--cuda-graph={cuda_graph}"),
                    format!("--optimize-kernel={optimize_kernel}"),
                ];
                if tracing {
                    raw.push("--tracing".to_owned());
                }
                let args = parse_args(raw.into_iter(), "tau-server").unwrap();
                let options = args.engine_options();
                assert!(options.auto_kv_cache_budget());
                assert_eq!(options.cuda_graph(), cuda_graph);
                assert_eq!(options.optimize_kernel(), optimize_kernel);
                assert_eq!(options.tracing(), tracing);
            }
        }
    }
}

/// Graph policy requires a lowercase enum value; legacy booleans are rejected.
#[test]
fn rejects_invalid_cuda_graph_values() {
    for value in ["", "sometimes", "1", "TRUE", "true", "false", "--tracing"] {
        for raw in [
            vec!["--cuda-graph".to_owned(), value.to_owned()],
            vec![format!("--cuda-graph={value}")],
        ] {
            let error = parse_args(raw.into_iter(), "tau-server").unwrap_err();
            assert!(error.contains("invalid --cuda-graph value"));
            assert!(error.contains("expected none, layer, or full"));
        }
    }
}

/// A bare flag cannot silently enable CUDA graphs.
#[test]
fn rejects_missing_cuda_graph_value() {
    let error =
        parse_args(["--cuda-graph".to_owned()].into_iter(), "tau-server").unwrap_err();
    assert!(error.contains("--cuda-graph none|layer|full"));
}

/// Keeps graph coverage help and legacy-value migration guidance intact.
#[test]
fn cuda_graph_help_explains_policy() {
    let help = usage("tau-server");
    assert!(help.contains("any layer may remain eager"));
    assert!(help.contains("Does not"));
    assert!(help.contains("require capturing the whole forward pass"));
    for value in ["true", "false"] {
        let error =
            parse_args([format!("--cuda-graph={value}")].into_iter(), "tau-server")
                .unwrap_err();
        assert!(error.contains("replace false with none and true with layer"));
    }
}

#[test]
fn parses_positive_prefill_chunk_sizes() {
    let defaults = parse_args(std::iter::empty(), "tau-server").unwrap();
    assert_eq!(defaults.engine_options().prefill_chunk_size(), 256);
    for size in [1, 4, 64, 256, 512] {
        for raw in [
            vec!["--prefill-chunk-size".to_owned(), size.to_string()],
            vec![format!("--prefill-chunk-size={size}")],
        ] {
            assert_eq!(
                parse_args(raw.into_iter(), "tau-server")
                    .unwrap()
                    .engine_options()
                    .prefill_chunk_size(),
                size
            );
        }
    }
    for value in ["0", "-1", "abc", "", "184467440737095516160"] {
        let error = parse_args(
            [format!("--prefill-chunk-size={value}")].into_iter(),
            "tau-server",
        )
        .unwrap_err();
        assert!(error.contains("positive integer"));
    }
    assert!(
        parse_args(
            ["--prefill-chunk-size".to_owned()].into_iter(),
            "tau-server"
        )
        .is_err()
    );
}

#[test]
fn attention_block_size_is_independent_of_prefill_and_rejects_invalid_values() {
    let defaults = parse_args(std::iter::empty(), "tau-server").unwrap();
    assert_eq!(
        defaults.engine_options().attention_block_size().tokens(),
        64
    );
    for size in [32, 64, 128, 256] {
        for raw in [
            vec!["--attention-block-size".to_owned(), size.to_string()],
            vec![format!("--attention-block-size={size}")],
        ] {
            let mut raw = raw;
            raw.push("--prefill-chunk-size=512".to_owned());
            let options = parse_args(raw.into_iter(), "tau-server")
                .unwrap()
                .engine_options();
            assert_eq!(options.attention_block_size().tokens(), size);
            assert_eq!(options.prefill_chunk_size(), 512);
        }
    }
    for value in ["", "0", "16", "65", "512", "-1", "abc"] {
        let error = parse_args(
            [format!("--attention-block-size={value}")].into_iter(),
            "tau-server",
        )
        .unwrap_err();
        assert!(error.contains("expected 32, 64, 128, or 256"));
    }
    assert!(
        parse_args(
            ["--attention-block-size".to_owned()].into_iter(),
            "tau-server"
        )
        .is_err()
    );
}

#[test]
fn rejects_removed_component_graph_flags() {
    for flag in [
        "--decoder-gdn-graph",
        "--decoder-gdn-layer-graph",
        "--decoder-mlp-graph",
        "--check-decoder-mlp-graph",
    ] {
        for raw in [
            vec![flag.to_owned(), "true".to_owned()],
            vec![format!("{flag}=false")],
        ] {
            assert!(parse_args(raw.into_iter(), "tau-server").is_err());
        }
    }
}

/// Request instrumentation is explicit and independent of kernel selection.
#[test]
fn tracing_is_opt_in_and_independent_of_kernel_selection() {
    let defaults = parse_args(std::iter::empty(), "tau-server").unwrap();
    assert!(!defaults.tracing);
    let args = parse_args(
        ["--tracing", "--optimize-kernel", "false"]
            .map(str::to_owned)
            .into_iter(),
        "tau-server",
    )
    .unwrap();
    assert!(args.tracing);
    assert!(!args.optimize_kernel);
}

#[test]
fn metrics_are_opt_in_and_independent_of_tracing_and_kernel_policy() {
    let defaults = parse_args(std::iter::empty(), "tau-server").unwrap();
    assert!(!defaults.engine_options().metrics());
    assert!(!defaults.engine_options().vram_metrics());
    for tracing in [false, true] {
        for optimize_kernel in [false, true] {
            let mut raw = vec![
                "--metrics".to_owned(),
                format!("--optimize-kernel={optimize_kernel}"),
            ];
            if tracing {
                raw.push("--tracing".to_owned());
            }
            let options = parse_args(raw.into_iter(), "tau-server")
                .unwrap()
                .engine_options();
            assert!(options.metrics());
            assert!(!options.vram_metrics());
            assert_eq!(options.tracing(), tracing);
            assert_eq!(options.optimize_kernel(), optimize_kernel);
        }
    }
}

#[test]
fn vram_metrics_are_independent_of_other_collection_modes() {
    for metrics in [false, true] {
        for tracing in [false, true] {
            let mut raw = vec!["--vram-metrics".to_owned()];
            if metrics {
                raw.push("--metrics".to_owned());
            }
            if tracing {
                raw.push("--tracing".to_owned());
            }
            let options = parse_args(raw.into_iter(), "tau-server")
                .unwrap()
                .engine_options();
            assert!(options.vram_metrics());
            assert_eq!(options.metrics(), metrics);
            assert_eq!(options.tracing(), tracing);
        }
    }
    // Sampling frequency is internal; no user-facing interval option yet.
    for raw in ["--vram-sample-interval-ms=10", "--vram-metrics=false"] {
        assert!(parse_args([raw.to_owned()].into_iter(), "tau-server").is_err());
    }
    assert!(usage("tau-server").contains("--vram-metrics"));
}

/// Confirms tau.rs optimized kernels are preferred when the flag is omitted.
#[test]
fn optimized_kernels_are_enabled_by_default() {
    let args = parse_args(std::iter::empty(), "tau-server").unwrap();
    assert!(args.optimize_kernel);
}

/// Accepts both boolean modes in separated and equals-style forms.
#[test]
fn parses_optimize_kernel_modes_with_both_cli_forms() {
    for (value, expected) in [("true", true), ("false", false)] {
        let separated = parse_args(
            ["--optimize-kernel".to_owned(), value.to_owned()].into_iter(),
            "tau-server",
        )
        .unwrap();
        assert_eq!(separated.optimize_kernel, expected);

        let equals = parse_args(
            [format!("--optimize-kernel={value}")].into_iter(),
            "tau-server",
        )
        .unwrap();
        assert_eq!(equals.optimize_kernel, expected);
    }
}

/// Rejects the obsolete underscore spelling rather than keeping a hidden alias.
#[test]
fn rejects_underscore_optimize_kernel_alias() {
    let error = parse_args(
        ["--optimize_kernel".to_owned(), "true".to_owned()].into_iter(),
        "tau-server",
    )
    .unwrap_err();
    assert!(error.contains("--optimize-kernel"));
}

/// Rejects optimize-kernel values other than explicit booleans.
#[test]
fn rejects_unknown_optimize_kernel_value() {
    let error = parse_args(
        ["--optimize-kernel".to_owned(), "sometimes".to_owned()].into_iter(),
        "tau-server",
    )
    .unwrap_err();
    assert!(error.contains("invalid --optimize-kernel value"));
}

/// Both spellings carry every tile setting and independent down geometry through
/// the engine boundary; incompatible padding and unsupported variants fail early.
#[test]
fn parses_moe_prefill_profiles_and_rejects_inconsistent_startup_choices() {
    for raw in [
        vec![
            "--moe-prefill-config",
            "16,64,64,4,2",
            "--moe-prefill-down-config",
            "16,32,32,2,1",
        ],
        vec![
            "--moe-prefill-config=16,64,64,4,2",
            "--moe-prefill-down-config=16,32,32,2,1",
        ],
    ] {
        let args =
            parse_args(raw.into_iter().map(str::to_owned), "tau-server").unwrap();
        let config = args.engine_options().moe_prefill_config().unwrap();
        assert_eq!(config.gate_up().values(), [16, 64, 64, 4, 2]);
        assert_eq!(config.down().values(), [16, 32, 32, 2, 1]);
        let load = tau_engine::ModelLoadOptions::from(args.engine_options());
        assert_eq!(load.moe_prefill_config(), Some(config));
        assert_eq!(EngineOptions::from(load).moe_prefill_config(), Some(config));
    }
    for raw in [
        vec!["--moe-prefill-config"],
        vec!["--moe-prefill-config=16,16,64,4,2"],
        vec!["--moe-prefill-down-config=16,64,64,4,2"],
        vec![
            "--moe-prefill-config=16,64,64,4,2",
            "--moe-prefill-down-config=32,64,64,4,2",
        ],
    ] {
        assert!(parse_args(raw.into_iter().map(str::to_owned), "tau-server").is_err());
    }
}

/// Decode settings survive both construction conversions. Either projection can
/// retain baseline independently, and malformed profiles fail during parsing.
#[test]
fn parses_independent_decode_profiles_through_model_construction() {
    for raw in [
        vec![
            "--moe-decode-config",
            "tiled:64,128,4,2",
            "--moe-decode-down-config",
            "baseline",
        ],
        vec![
            "--moe-decode-config=tiled:64,128,4,2",
            "--moe-decode-down-config=baseline",
        ],
    ] {
        let args =
            parse_args(raw.into_iter().map(str::to_owned), "tau-server").unwrap();
        let config = args.engine_options().moe_decode_config().unwrap();
        assert_eq!(config.gate_up.words(), [1, 64, 128, 4, 2]);
        assert_eq!(config.down.words(), [0, 16, 16, 1, 1]);
        let load = tau_engine::ModelLoadOptions::from(args.engine_options());
        assert_eq!(load.moe_decode_config(), Some(config));
        assert_eq!(EngineOptions::from(load).moe_decode_config(), Some(config));
    }
    let args = parse_args(
        ["--moe-decode-down-config=tiled:32,64,2,1"]
            .into_iter()
            .map(str::to_owned),
        "tau-server",
    )
    .unwrap();
    assert_eq!(
        args.moe_decode_options
            .unwrap()
            .exact_config()
            .unwrap()
            .gate_up,
        tau_engine::MoeDecodeProfile::default()
    );
    assert!(
        parse_args(
            ["--moe-decode-config=tiled:32,64,8,1"]
                .into_iter()
                .map(str::to_owned),
            "tau-server"
        )
        .is_err()
    );
}

/// Budget intent survives every construction boundary; exact controls retain
/// their legacy semantics and cannot accidentally turn auto into no override.
#[test]
fn preserves_auto_decode_options_through_engine_and_loader_options() {
    let args = parse_args(
        [
            "--moe-decode-config=auto:32,64,8,2",
            "--moe-decode-down-config=baseline",
        ]
        .into_iter()
        .map(str::to_owned),
        "tau-server",
    )
    .unwrap();
    let engine = args.engine_options();
    let options = engine.moe_decode_options().unwrap();
    assert_eq!(options.gate_up.to_string(), "auto:32,64,8,2");
    assert!(engine.moe_decode_config().is_none());
    let load = tau_engine::ModelLoadOptions::from(engine);
    assert_eq!(load.moe_decode_options(), Some(options));
    assert_eq!(
        EngineOptions::from(load).moe_decode_options(),
        Some(options)
    );
    for value in ["auto:32,64,0,2", "auto:32,64,8,3"] {
        assert!(
            parse_args(
                [format!("--moe-decode-config={value}")].into_iter(),
                "tau-server"
            )
            .is_err()
        );
    }
}
