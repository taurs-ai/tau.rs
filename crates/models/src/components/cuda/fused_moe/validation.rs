//! Test-only expert scratch snapshots and artifact-backed single-token replay.
//! Scratch is copied only after a completed forward, outside graph capture.

mod observation;
mod projection_tests;
mod timing;

use super::*;
use crate::test_support::tensor_capture::Value;
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
struct Bank {
    layer: usize,
    file: String,
    sha256: String,
}
#[derive(serde::Deserialize)]
struct Case {
    id: String,
    layer: usize,
    file: String,
    sha256: String,
}
#[derive(serde::Deserialize)]
struct Corpus {
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
    banks: Vec<Bank>,
    entries: Vec<Case>,
}

fn file_hash(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn snapshot(runner: &FusedMoeRunner, output: &Tensor) -> BTreeMap<String, Value> {
    let mut result = runner.validation_snapshot();
    result.insert(
        "routed".to_owned(),
        Value::Tensor(output.to_device(&candle::Device::Cpu).unwrap()),
    );
    result
}

fn host_tensors(values: BTreeMap<String, Value>) -> BTreeMap<String, Tensor> {
    values
        .into_iter()
        .map(|(key, value)| {
            let t = match value {
                Value::Tensor(t) => t,
                Value::Host { shape, values } => {
                    Tensor::from_vec(values, shape, &candle::Device::Cpu).unwrap()
                }
            };
            let t = if ["gate-up-bf16", "activated", "weighted-bf16"]
                .contains(&key.as_str())
            {
                t.to_dtype(DType::BF16).unwrap()
            } else {
                t
            };
            (key, t)
        })
        .collect()
}

fn exact(actual: &BTreeMap<String, Tensor>, expected: &BTreeMap<String, Tensor>) {
    for (key, a) in actual {
        let b = &expected[key];
        let bits = |v: &Tensor| {
            v.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(a.shape(), b.shape());
        assert_eq!(bits(a), bits(b), "dynamic replay changed {key}");
    }
}

/// Hash immutable device operands in bounded rows, outside capture and timing.
/// Whole-bank coverage catches writes into unused expert rows too.
fn tensor_hash(tensor: &Tensor) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(format!("{:?}/{:?}", tensor.dtype(), tensor.dims()));
    for row in 0..tensor.dim(0).unwrap() {
        // A contiguous view still retains the entire backing allocation.
        // Materialize this row so host conversion transfers only the chunk.
        let slice = tensor
            .narrow(0, row, 1)
            .unwrap()
            .force_contiguous()
            .unwrap();
        hash.update(safetensors::View::data(&slice).as_ref());
    }
    format!("{:x}", hash.finalize())
}

/// Replays the real expert kernel with three 1.5 GiB banks plus small workspace.
/// Unselected experts are zero;
/// selected trained rows retain their original expert IDs and full-bank strides.
/// Hot and rotating-bank timing are separate because a lone layer can fit L2.
#[test]
#[ignore = "requires exported real decode fixtures; release CUDA replay/timing only"]
fn replay_captured_decode_fixture() {
    use crate::components::{CapturedGraph, DecodeProgram};
    use candle::{Device, Var};
    use std::path::PathBuf;
    if cfg!(debug_assertions) {
        panic!("use --release");
    }
    let root = PathBuf::from(std::env::var("TAU_MOE_DECODE_FIXTURE").unwrap());
    let output_dir = PathBuf::from(std::env::var("TAU_MOE_DECODE_OUTPUT").unwrap());
    std::fs::create_dir(&output_dir).unwrap();
    let manifest: Corpus =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        (
            manifest.hidden,
            manifest.intermediate,
            manifest.experts,
            manifest.top_k
        ),
        (2048, 512, 256, 8)
    );
    // SAFETY: All operations, updates, captures and readbacks use one stream.
    let device = unsafe { crate::components::new_capture_device(0) }.unwrap();
    let mut banks = BTreeMap::new();
    for spec in &manifest.banks {
        assert_eq!(file_hash(&root.join(&spec.file)), spec.sha256);
        let tensors =
            candle::safetensors::load(root.join(&spec.file), &Device::Cpu).unwrap();
        let ids = tensors["expert_ids"].to_vec1::<i64>().unwrap();
        let gu = Tensor::zeros((256, 1024, 2048), DType::BF16, &device).unwrap();
        let down = Tensor::zeros((256, 2048, 512), DType::BF16, &device).unwrap();
        let selected_gu = tensors["gate_up_proj"].to_device(&device).unwrap();
        let selected_down = tensors["down_proj"].to_device(&device).unwrap();
        for (rank, &id) in ids.iter().enumerate() {
            assert!((0..256).contains(&id));
            gu.slice_set(&selected_gu.narrow(0, rank, 1).unwrap(), 0, id as usize)
                .unwrap();
            down.slice_set(&selected_down.narrow(0, rank, 1).unwrap(), 0, id as usize)
                .unwrap();
        }
        device.synchronize().unwrap();
        assert!(banks.insert(spec.layer, (gu, down)).is_none());
    }
    let weight_hashes: Vec<_> = banks
        .values()
        .map(|(gu, down)| (tensor_hash(gu), tensor_hash(down)))
        .collect();
    struct Replay {
        layer: usize,
        id: String,
        runner: FusedMoeRunner,
        input: Var,
        logits: Var,
        output: Tensor,
    }
    let options = tau_cuda_kernels::MoeDecodeOptions {
        gate_up: std::env::var("TAU_MOE_DECODE_GATE_UP")
            .unwrap_or_else(|_| "baseline".to_owned())
            .parse()
            .unwrap(),
        down: std::env::var("TAU_MOE_DECODE_DOWN")
            .unwrap_or_else(|_| "baseline".to_owned())
            .parse()
            .unwrap(),
    };
    let mut program = DecodeProgram::new(&device).unwrap();
    let mut cases = Vec::new();
    for case in &manifest.entries {
        assert_eq!(file_hash(&root.join(&case.file)), case.sha256);
        assert!(
            case.id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-')
        );
        let tensors =
            candle::safetensors::load(root.join(&case.file), &device).unwrap();
        let input =
            Var::from_tensor(&tensors["input"].reshape((1, 1, 2048)).unwrap()).unwrap();
        let logits = Var::from_tensor(&tensors["logits"]).unwrap();
        let (gu, down) = &banks[&case.layer];
        let mut runner = FusedMoeRunner::new(gu.clone(), down.clone(), 8)
            .unwrap()
            .unwrap();
        runner.configure_decode(options).unwrap();
        let result = runner
            .prepare_decode_program(
                &mut program,
                &input.as_detached_tensor(),
                &logits.as_detached_tensor(),
            )
            .unwrap();
        cases.push(Replay {
            layer: case.layer,
            id: case.id.clone(),
            runner,
            input,
            logits,
            output: result,
        });
    }
    let input_hashes: Vec<_> = cases
        .iter()
        .map(|c| {
            (
                tensor_hash(&c.input.as_detached_tensor()),
                tensor_hash(&c.logits.as_detached_tensor()),
            )
        })
        .collect();
    program.run().unwrap();
    device.synchronize().unwrap();
    // SAFETY: Program/cases retain every buffer, runner and weight through graph teardown.
    let mut graph =
        unsafe { CapturedGraph::capture(&device, || program.run()) }.unwrap();
    assert_eq!(graph.allocation_nodes().unwrap(), 0);
    graph.replay().unwrap();
    device.synchronize().unwrap();
    let mut baseline = Vec::new();
    for case in &cases {
        let tensors = host_tensors(snapshot(&case.runner, &case.output));
        let saved = tensors
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<std::collections::HashMap<_, _>>();
        candle::safetensors::save(
            &saved,
            output_dir.join(format!("{}.safetensors", case.id)),
        )
        .unwrap();
        baseline.push(tensors);
    }
    let b = cases
        .iter()
        .position(|c| c.layer == cases[0].layer && c.id != cases[0].id)
        .unwrap();
    let original_input = cases[0].input.as_detached_tensor().copy().unwrap();
    let original_logits = cases[0].logits.as_detached_tensor().copy().unwrap();
    cases[0]
        .input
        .set(&cases[b].input.as_detached_tensor())
        .unwrap();
    cases[0]
        .logits
        .set(&cases[b].logits.as_detached_tensor())
        .unwrap();
    graph.replay().unwrap();
    device.synchronize().unwrap();
    exact(
        &host_tensors(snapshot(&cases[0].runner, &cases[0].output)),
        &baseline[b],
    );
    cases[0].input.set(&original_input).unwrap();
    cases[0].logits.set(&original_logits).unwrap();
    graph.replay().unwrap();
    device.synchronize().unwrap();
    exact(
        &host_tensors(snapshot(&cases[0].runner, &cases[0].output)),
        &baseline[0],
    );
    let selections:Vec<_>=cases.iter().map(|case| {
        let r=case.runner.resolution.unwrap();
        serde_json::json!({"id":case.id,"gate_up":{"requested":r[0].requested.to_string(),"effective":r[0].profile.to_string(),"fragment_columns":r[0].profile.fragment_columns(),"columns_per_warp":r[0].profile.columns_per_warp(),"shared_upper_bytes":r[0].profile.shared_memory_bytes(),"reason":r[0].reason},
        "down":{"requested":r[1].requested.to_string(),"effective":r[1].profile.to_string(),"fragment_columns":r[1].profile.fragment_columns(),"columns_per_warp":r[1].profile.columns_per_warp(),"shared_upper_bytes":r[1].profile.shared_memory_bytes(),"reason":r[1].reason}})
    }).collect();
    let receipts: Vec<_> = cases
        .iter()
        .map(|case| {
            let workspace = case.runner.workspace.lock().unwrap();
            let stream = device.as_cuda_device().unwrap().cuda_stream();
            let words = stream.clone_dtoh(&workspace.receipts).unwrap();
            assert!(
                words[..40]
                    .chunks_exact(5)
                    .all(|r| r == case.runner.config.gate_up.words())
            );
            assert!(
                words[40..]
                    .chunks_exact(5)
                    .all(|r| r == case.runner.config.down.words())
            );
            serde_json::json!({"id":case.id,"gate_up":words[..40],"down":words[40..]})
        })
        .collect();
    let executable_sha256 = file_hash(&std::env::current_exe().unwrap());
    let fixture_sha256 = file_hash(&root.join("manifest.json"));
    // Timing is a second invocation, authorized only by this exact executable's
    // independent CPU reference report. Validation-only runs have no samples.
    let approval = std::env::var("TAU_MOE_DECODE_APPROVAL").ok().map(|path| {
        let r: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(r["passed"], true);
        assert_eq!(r["mathematical_passed"], true);
        assert_eq!(r["exact_regression_passed"], true);
        assert_eq!(r["fixture_sha256"], fixture_sha256);
        assert_eq!(r["execution"]["executable_sha256"], executable_sha256);
        assert_eq!(r["execution"]["receipts"], serde_json::json!(receipts));
        r
    });
    let samples = timing::measure(
        &device,
        &cases[0].runner,
        &cases[0].input.as_detached_tensor(),
        &cases[0].logits.as_detached_tensor(),
        &mut graph,
        cases.len(),
        approval.is_some(),
    );
    let projections = timing::projections(
        &device,
        &cases.iter().map(|c| c.runner.clone()).collect::<Vec<_>>(),
        approval.is_some(),
    );
    for (case, expected) in cases.iter().zip(&baseline) {
        exact(
            &host_tensors(snapshot(&case.runner, &case.output)),
            expected,
        );
    }
    assert_eq!(
        weight_hashes,
        banks
            .values()
            .map(|(gu, down)| (tensor_hash(gu), tensor_hash(down)))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        input_hashes,
        cases
            .iter()
            .map(|c| (
                tensor_hash(&c.input.as_detached_tensor()),
                tensor_hash(&c.logits.as_detached_tensor())
            ))
            .collect::<Vec<_>>()
    );
    let report = serde_json::json!({"passed":true,"cases":cases.len(),"a_b_a_exact":true,
        "allocation_nodes":0,"immutable_operands":true,"samples":samples,"projections":projections,"banks":banks.len(),
        "device_uuid":format!("{:?}",device.as_cuda_device().unwrap().cuda_stream().context().uuid().unwrap()),
        "receipts":receipts,"selections":selections,"executable_sha256":executable_sha256,"fixture_sha256":fixture_sha256,
        "scope":"Routed MoE block only; actual M=1, top8, H2048/I512; zero unselected weights; hot/cycling working sets are not full-model throughput"});
    std::fs::write(
        output_dir.join("timing.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("MOE_DECODE_REPLAY {report}");
    drop(graph);
    drop(program);
    drop(cases);
    drop(banks);
}
