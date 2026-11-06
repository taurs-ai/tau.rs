//! Component parity and execution coverage before changing ordinary prefill.
use super::*;

/// Compare the candidate with the retained expert loop using non-tile-aligned
/// projections and changing routes. Native calls must occur, including tails;
/// reusing larger scratch must not change a later short result.
fn check_grouped(
    tokens_to_check: &[usize],
    clone_each: bool,
    tile: crate::MoePrefillTile,
) {
    check_grouped_config(
        tokens_to_check,
        clone_each,
        crate::MoePrefillConfig::new(tile, tile).unwrap(),
        33,
        17,
    );
}

/// Share exact reference, lifetime and receipt assertions across odd dimensions
/// and aligned multi-stage operands, including independent projection profiles.
fn check_grouped_config(
    tokens_to_check: &[usize],
    clone_each: bool,
    tiles: crate::MoePrefillConfig,
    h: usize,
    i: usize,
) {
    let e = 256;
    let device = Device::new_cuda(0).unwrap();
    let tensor = |data: Vec<f32>, shape: &[usize]| {
        Tensor::from_vec(data, shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    };
    let values = |n, seed| {
        (0..n)
            .map(|x| (((x * 17 + seed) % 31) as f32 - 15.) / 128.)
            .collect::<Vec<_>>()
    };
    let mut router = vec![0f32; e * h];
    for expert in 0..e {
        router[expert * h] = expert as f32 / 32.;
    }
    let tensors = HashMap::from([
        ("gate.weight".to_owned(), tensor(router, &[e, h])),
        (
            "experts.gate_up_proj".to_owned(),
            tensor(values(e * 2 * i * h, 1), &[e, 2 * i, h]),
        ),
        (
            "experts.down_proj".to_owned(),
            tensor(values(e * h * i, 3), &[e, h, i]),
        ),
    ]);
    let config = SparseMoeConfig {
        hidden_size: h,
        expert_intermediate_size: i,
        expert_activation: Activation::Silu,
        num_experts: e,
        experts_per_token: 8,
        router_bias: false,
        router_score_function: RouterScoreFunction::Softmax,
        normalize_selected_weights: true,
        shared_expert: None,
        weight_layout: SparseMoeWeightLayout::Packed,
    };
    let block = SparseMoeBlock::new(
        &config,
        true,
        VarBuilder::from_tensors(tensors, DType::BF16, &device),
    )
    .unwrap()
    .with_prefill_config(Some(tiles))
    .unwrap();
    let bits = |x: Tensor| {
        x.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    let mut retained = None;
    for &tokens in tokens_to_check {
        let mut data = values(tokens * h, 7);
        for token in 0..tokens {
            data[token * h] = [4., -4., 0.][token % 3];
        }
        let input = tensor(data, &[1, tokens, h]);
        let expected = bits(
            block
                .forward_phase(&input, Execution::ReferencePrefill)
                .unwrap(),
        );
        let runner = block
            .prefill_runner
            .as_ref()
            .expect("native prefill eligibility");
        let before = runner.calls();
        let prior = runner.workspace_info();
        let actual = bits(
            block
                .forward_phase(&input, Execution::GroupedPrefill)
                .unwrap(),
        );
        assert_eq!(runner.calls(), before + 1, "native path silently fell back");
        assert_eq!(actual, expected, "token count {tokens}");
        assert_eq!(
            runner.executed_tiles(),
            [tiles.gate_up().values(), tiles.down().values()]
        );
        let info = runner.workspace_info().unwrap();
        if let Some(old) = prior
            && tokens <= old.0
        {
            assert_eq!(info, old, "scratch unexpectedly replaced");
        }
        if tokens == 257 {
            retained = Some(info);
        }
        if let Some(old) = retained {
            assert_eq!(info, old, "long-to-short workspace reuse");
        }
        if clone_each {
            let cloned = block.clone();
            assert_eq!(
                bits(
                    cloned
                        .forward_phase(&input, Execution::GroupedPrefill)
                        .unwrap()
                ),
                expected
            );
        }
    }
}

/// Cover the full shape/clone sweep under ordinary execution and memory tools.
#[test]
fn grouped_prefill_matches_reference_with_tails_and_workspace_reuse() {
    check_grouped(
        &[1, 257, 1, 15, 16, 17, 33, 128, 255, 256, 1],
        true,
        crate::MoePrefillTile::default(),
    );
}

/// Bound racecheck's instrumentation history while exercising the same masked
/// MMA, dispatch and long-to-short scratch synchronization as the full sweep.
#[test]
fn grouped_prefill_device_synchronization() {
    for profile in ["16,64,64,4,2", "32,32,32,2,1"] {
        check_grouped(&[1, 17, 1], false, profile.parse().unwrap());
    }
}

/// Every compiled profile must handle masked dimensions, prompt tails, workspace
/// growth/reuse and clones before any production-width tuning is attempted.
#[test]
fn every_prefill_tile_matches_reference_and_executes_requested_kernels() {
    for tile in crate::MoePrefillTile::supported() {
        check_grouped(&[1, 257, 1, 17, 33], true, tile);
    }
}

/// Aligned operands exercise asynchronous copies through multiple K-stage wraps
/// and a partial final stage. Different gate/up and down profiles must both run.
#[test]
fn cooperative_prefill_preserves_aligned_pipeline_and_asymmetric_tiles() {
    for (up, down) in [
        ("16,64,64,4,2", "16,32,32,2,2"),
        ("32,64,64,8,2", "32,32,32,4,1"),
    ] {
        let tiles =
            crate::MoePrefillConfig::new(up.parse().unwrap(), down.parse().unwrap())
                .unwrap();
        check_grouped_config(&[1, 17, 33, 1], true, tiles, 144, 80);
    }
}

/// K=128 must preserve the K=16 arithmetic while both projections wrap their
/// staging buffers and mask a final partial stage; receipts prove the new path.
#[test]
fn k128_prefill_preserves_aligned_wraps_and_projection_overrides() {
    for (up, down) in [
        ("16,64,128,4,2", "16,32,128,2,1"),
        ("32,64,128,8,1", "32,32,128,4,2"),
    ] {
        let tiles =
            crate::MoePrefillConfig::new(up.parse().unwrap(), down.parse().unwrap())
                .unwrap();
        check_grouped_config(&[1, 17, 33, 1], true, tiles, 272, 272);
    }
}
