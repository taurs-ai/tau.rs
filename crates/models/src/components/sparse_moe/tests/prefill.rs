//! Replays every reference token through actual sparse-block prefill operations.

use super::*;

use crate::test_support::tensor_capture::{self as capture, Value};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Workload {
    entries: Vec<Entry>,
}

#[derive(Clone, Deserialize, Serialize)]
struct Entry {
    id: String,
    layer: usize,
    tokens: usize,
    offset: usize,
    kind: String,
    input: PathBuf,
    router: Option<PathBuf>,
}

#[derive(Serialize)]
struct Observed {
    #[serde(flatten)]
    entry: Entry,
    observation_exact: bool,
    expert_assignments: usize,
    dispatch_block_size: usize,
    dispatch_padded: usize,
    native_prefill_executed: bool,
    executed_kernel_config: Option<serde_json::Value>,
    stages: BTreeMap<String, serde_json::Value>,
}

/// Reads an explicitly shaped, little-endian FP32 fixture without partial values.
fn read_input(path: &Path, rows: usize, width: usize, device: &Device) -> Tensor {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len(), rows * width * 4);
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(values.iter().all(|value| value.is_finite()));
    Tensor::from_vec(values, (rows, width), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

/// Converts observed storage to a shape and every FP32 bit for serialization.
fn values(value: Value) -> (Vec<usize>, Vec<f32>) {
    match value {
        Value::Host { shape, values } => {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            (shape, values)
        }
        Value::Tensor(tensor) => {
            let shape = tensor.dims().to_vec();
            let values = tensor
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(values.iter().all(|value| value.is_finite()));
            (shape, values)
        }
    }
}

/// Writes raw tensors; the Python owner adds integrity hashes and checks goldens.
fn dump(
    root: &Path,
    id: &str,
    name: &str,
    shape: Vec<usize>,
    values: &[f32],
) -> serde_json::Value {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    let file = format!("{id}-{name}.f32");
    let mut writer =
        std::io::BufWriter::new(fs::File::create_new(root.join(&file)).unwrap());
    for value in values {
        writer.write_all(&value.to_le_bytes()).unwrap();
    }
    writer.flush().unwrap();
    serde_json::json!({"file":file,"shape":shape})
}

/// Canonicalizes observed expert groups to token/rank slots without recomputing
/// an expert. This keeps different future dispatch layouts comparable to SGLang.
fn serialize(
    root: &Path,
    entry: Entry,
    mut observed: BTreeMap<String, Value>,
) -> Observed {
    let mut stages = BTreeMap::new();
    let mut add = |name: &str, shape, data: &[f32]| {
        stages.insert(name.to_owned(), dump(root, &entry.id, name, shape, data));
    };
    let (_, ids) = values(observed.remove("ids").expect("actual routing IDs missing"));
    assert_eq!(ids.len(), entry.tokens * 8);
    add("ids", vec![entry.tokens, 8], &ids);
    for name in ["input", "router", "weights", "routed", "output"] {
        let (shape, data) = values(observed.remove(name).expect(name));
        add(name, shape, &data);
    }
    for (name, captured) in [("shared", "shared-down"), ("shared_gate", "shared-gate")]
    {
        let (shape, data) = values(observed.remove(captured).expect(captured));
        add(name, shape, &data);
    }
    if observed.remove("grouped-executed").is_some() {
        let (shape, slots) = values(observed.remove("grouped-dispatch-slots").unwrap());
        add("dispatch_slots", shape, &slots);
        let (shape, experts) =
            values(observed.remove("grouped-dispatch-experts").unwrap());
        add("dispatch_experts", shape, &experts);
        for name in ["gate_up", "activation", "weighted_down"] {
            let (shape, data) =
                values(observed.remove(&format!("grouped-{name}")).unwrap());
            add(name, shape, &data);
        }
        let (_, receipt) = values(observed.remove("grouped-kernel-config").unwrap());
        let profiles = receipt.iter().map(|v| *v as i32).collect::<Vec<_>>();
        return Observed {
            expert_assignments: entry.tokens * 8,
            dispatch_block_size: profiles[0] as usize,
            executed_kernel_config: Some(
                serde_json::json!({"gate_up":profiles[..5], "down":profiles[5..]}),
            ),
            dispatch_padded: slots.len(),
            native_prefill_executed: true,
            entry,
            stages,
            observation_exact: true,
        };
    }
    let mut slots = Vec::new();
    let mut experts = Vec::new();
    let mut canonical = BTreeMap::from([
        ("gate_up", (1024, vec![0f32; entry.tokens * 8 * 1024])),
        ("activation", (512, vec![0f32; entry.tokens * 8 * 512])),
        ("weighted_down", (2048, vec![0f32; entry.tokens * 8 * 2048])),
    ]);
    let mut visited = vec![false; ids.len()];
    for expert in 0..256 {
        let Some(tokens) = observed.remove(&format!("expert-{expert}-tokens")) else {
            continue;
        };
        let (_, tokens) = values(tokens);
        let mut group_slots = Vec::new();
        for token in &tokens {
            let token = *token as usize;
            let rank = (0..8)
                .find(|rank| ids[token * 8 + rank] == expert as f32)
                .unwrap();
            let slot = token * 8 + rank;
            assert!(!std::mem::replace(&mut visited[slot], true));
            group_slots.push(slot);
            slots.push(slot as f32);
            experts.push(expert as f32);
        }
        for (stage, (width, output)) in &mut canonical {
            let (_, data) = values(
                observed
                    .remove(&format!("expert-{expert}-{stage}"))
                    .expect(stage),
            );
            assert_eq!(data.len(), tokens.len() * *width);
            for (group_row, slot) in group_slots.iter().enumerate() {
                output[slot * *width..(slot + 1) * *width].copy_from_slice(
                    &data[group_row * *width..(group_row + 1) * *width],
                );
            }
        }
    }
    assert!(
        visited.iter().all(|seen| *seen),
        "missing actual expert execution"
    );
    for (name, (width, output)) in canonical {
        add(name, vec![entry.tokens, 8, width], &output);
    }
    add("dispatch_slots", vec![slots.len()], &slots);
    add("dispatch_experts", vec![experts.len()], &experts);
    Observed {
        entry,
        stages,
        observation_exact: true,
        expert_assignments: slots.len(),
        dispatch_block_size: 1,
        dispatch_padded: slots.len(),
        native_prefill_executed: false,
        executed_kernel_config: None,
    }
}

/// Share checkpoint validation, mappings and private stream ownership between
/// the immutable SGLang corpus replay and the bounded configuration sweep.
fn load_fixture(root: &Path) -> (Device, SparseMoeConfig, VarBuilder<'static>) {
    let metadata = crate::load_model_config(root).unwrap();
    let cfg = crate::qwen3_5_moe::load_qwen3_5_moe_config(
        root,
        &metadata.architecture,
        &metadata.model_type,
    )
    .unwrap();
    let cfg = cfg.model.sparse_moe_config(SparseMoeWeightLayout::Packed);
    assert_eq!(
        (
            cfg.hidden_size,
            cfg.expert_intermediate_size,
            cfg.num_experts,
            cfg.experts_per_token
        ),
        (2048, 512, 256, 8)
    );
    let shards = crate::safetensors::load_safetensors_shards(root).unwrap();
    let paths = shards
        .iter()
        .map(|shard| shard.path().to_path_buf())
        .collect::<Vec<_>>();
    // SAFETY: Observations, eager replay and readback stay on this private stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    // SAFETY: The checkpoint mappings are read-only for this test's lifetime.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device) }
            .unwrap();
    (device, cfg, vb)
}

/// Executes observed/unobserved block calls on identical golden inputs and
/// requires exact output identity before exporting any candidate evidence.
#[test]
#[ignore = "requires the four-layer MoE model, CUDA and a prefill reference workload"]
fn capture_sglang_prefill_blocks() {
    let root = PathBuf::from(std::env::var("TAU_MOE_PREFILL_MODEL").unwrap());
    let workload: Workload = serde_json::from_slice(
        &fs::read(std::env::var("TAU_MOE_PREFILL_WORKLOAD").unwrap()).unwrap(),
    )
    .unwrap();
    let output = PathBuf::from(std::env::var("TAU_MOE_PREFILL_OUTPUT").unwrap());
    fs::create_dir(&output).unwrap();
    let (device, cfg, vb) = load_fixture(&root);
    let mut results = BTreeMap::new();
    for layer in 0..4 {
        let mut block = SparseMoeBlock::new(
            &cfg,
            true,
            vb.pp(format!("model.language_model.layers.{layer}.mlp")),
        )
        .unwrap();
        let profile = std::env::var("TAU_MOE_PREFILL_CONFIG").unwrap_or_default();
        if !profile.is_empty() {
            let up = profile.parse::<crate::MoePrefillTile>().unwrap();
            let down = std::env::var("TAU_MOE_PREFILL_DOWN_CONFIG").unwrap_or_default();
            let down = if down.is_empty() {
                up
            } else {
                down.parse().unwrap()
            };
            block = block
                .with_prefill_config(Some(
                    crate::MoePrefillConfig::new(up, down).unwrap(),
                ))
                .unwrap();
        }
        for entry in workload.entries.iter().filter(|entry| entry.layer == layer) {
            let input = read_input(&entry.input, entry.tokens, 2048, &device)
                .unsqueeze(0)
                .unwrap();
            let router = entry
                .router
                .as_ref()
                .map(|path| read_input(path, entry.tokens, 256, &device));
            capture::begin(router.clone());
            let backend = std::env::var("TAU_MOE_PREFILL_CAPTURE_BACKEND")
                .unwrap_or_else(|_| "auto".to_owned());
            let forward = || match backend.as_str() {
                "grouped" => block.forward_phase(&input, Execution::GroupedPrefill),
                "reference" => block.forward_phase(&input, Execution::ReferencePrefill),
                "auto" => block.forward(&input),
                _ => panic!("unknown capture backend"),
            };
            let before = block
                .prefill_runner
                .as_ref()
                .map_or(0, MoePrefillRunner::calls);
            let observed_output = forward().unwrap();
            if backend == "grouped" {
                assert_eq!(
                    block.prefill_runner.as_ref().unwrap().calls(),
                    before + 1,
                    "intended grouped prefill path did not execute"
                );
            }
            let observed = capture::finish();
            capture::begin(router);
            capture::pause();
            let replay = forward().unwrap();
            assert!(capture::finish().is_empty());
            let bits = |tensor| {
                values(Value::Tensor(tensor))
                    .1
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(observed_output),
                bits(replay),
                "observer changed {}",
                entry.id
            );
            results.insert(
                entry.id.clone(),
                serialize(&output, entry.clone(), observed),
            );
        }
    }
    let ordered = workload
        .entries
        .iter()
        .map(|entry| results.remove(&entry.id).unwrap())
        .collect::<Vec<_>>();
    fs::write(
        output.join("capture.json"),
        serde_json::to_vec_pretty(&ordered).unwrap(),
    )
    .unwrap();
}
