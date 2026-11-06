//! Prepared native decode state ownership and reset contracts.

use super::*;

/// Exercises the future capture boundary using ordinary CUDA execution first.
#[test]
fn prepared_cuda_decode_preserves_state_lifecycle() {
    assert_prepared_decode_lifecycle(false);
}

#[test]
fn prepared_cuda_decode_preserves_mixed_storage_state_lifecycle() {
    assert_prepared_decode_lifecycle(true);
}

fn assert_prepared_decode_lifecycle(mixed_storage: bool) {
    let device = Device::new_cuda(0).expect("GDN lifecycle test requires CUDA GPU 0");
    let config = cuda_test_config();
    let mut tensors = test_tensors(&config, &device);
    if mixed_storage {
        add_fp32_parameters(&mut tensors, &config, &device);
    }
    // Construct separate runners: cloning a layer would share its native state.
    let make_layer = || {
        GatedDeltaNet::new(
            &config,
            true,
            VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device),
        )
        .unwrap()
    };
    let mut prepared = make_layer();
    let mut eager = make_layer();
    let addresses = native_state_addresses(&prepared);
    let eager_addresses = native_state_addresses(&eager);
    assert_ne!(addresses.0, eager_addresses.0);
    assert_ne!(addresses.1, eager_addresses.1);
    let input = patterned_bf16((1, 1, config.hidden_size), 19, 0.018, &device);
    assert!(prepared.forward_prepared_decode(&input).is_err());

    let mut first_request_outputs = Vec::new();
    // A -> B -> A exercises request isolation and reseeding after different
    // prompts.
    for (request, phase) in [17, 5, 17].into_iter().enumerate() {
        prepared.clear_cache();
        eager.clear_cache();
        assert!(prepared.forward_prepared_decode(&input).is_err());
        let prefill = patterned_bf16((1, 3, config.hidden_size), phase, 0.02, &device);
        assert_tensor_exact(
            &prepared.forward(&prefill).unwrap(),
            &eager.forward(&prefill).unwrap(),
        );
        let initial_conv = prepared.conv_state.as_ref().unwrap().clone();
        let initial_recurrent = prepared.recurrent_state.as_ref().unwrap().clone();
        prepared.prepare_optimized_decode().unwrap();
        let (conv, recurrent) = native_state(&prepared);
        assert_tensor_exact(&conv, &initial_conv);
        assert_tensor_exact(&recurrent, &initial_recurrent);

        // Warm-up advances state. Restore the prompt state in place before use.
        prepared.forward_prepared_decode(&input).unwrap();
        prepared
            .seed_optimized_state(Some(&initial_conv), Some(&initial_recurrent))
            .unwrap();
        let (conv, recurrent) = native_state(&prepared);
        assert_tensor_exact(&conv, &initial_conv);
        assert_tensor_exact(&recurrent, &initial_recurrent);

        for step in 0..128 {
            let input =
                patterned_bf16((1, 1, config.hidden_size), 19 + step, 0.018, &device);
            let actual = prepared.forward_prepared_decode(&input).unwrap();
            let expected = eager.forward(&input).unwrap();
            assert_tensor_exact(&actual, &expected);
            // Preparation is idempotent even after native state has advanced.
            prepared.prepare_optimized_decode().unwrap();
            assert_native_state_exact(&prepared, &eager);
            assert_eq!(native_state_addresses(&prepared), addresses);
            match request {
                0 => first_request_outputs.push(actual),
                2 => assert_tensor_exact(&actual, &first_request_outputs[step]),
                _ => {}
            }
        }

        // Reference fallback must consume the state advanced through prepared
        // decode.
        let fallback = patterned_bf16((1, 2, config.hidden_size), 23, 0.017, &device);
        assert_tensor_exact(
            &prepared.forward(&fallback).unwrap(),
            &eager.forward(&fallback).unwrap(),
        );
        assert_tensor_exact(
            prepared.conv_state.as_ref().unwrap(),
            eager.conv_state.as_ref().unwrap(),
        );
        assert_tensor_exact(
            prepared.recurrent_state.as_ref().unwrap(),
            eager.recurrent_state.as_ref().unwrap(),
        );
        assert!(prepared.forward_prepared_decode(&input).is_err());
        prepared.prepare_optimized_decode().unwrap();
        assert_tensor_exact(
            &prepared.forward_prepared_decode(&input).unwrap(),
            &eager.forward(&input).unwrap(),
        );
        assert_native_state_exact(&prepared, &eager);
    }

    // An explicit zero reset also works while native state is active, without
    // prefill.
    prepared.seed_optimized_state(None, None).unwrap();
    eager.clear_cache();
    eager.prepare_optimized_decode().unwrap();
    assert_native_state_exact(&prepared, &eager);
    for state in [native_state(&prepared).0, native_state(&prepared).1] {
        assert_tensor_exact(&state, &state.zeros_like().unwrap());
    }
    assert_tensor_exact(
        &prepared.forward_prepared_decode(&input).unwrap(),
        &eager.forward(&input).unwrap(),
    );
    assert_native_state_exact(&prepared, &eager);
    assert_eq!(native_state_addresses(&prepared), addresses);

    // Invalid explicit calls must neither reset nor advance valid state.
    let (conv, recurrent) = native_state(&prepared);
    assert!(prepared.seed_optimized_state(Some(&conv), None).is_err());
    let wrong_dtype = recurrent.to_dtype(DType::BF16).unwrap();
    assert!(
        prepared
            .seed_optimized_state(Some(&conv), Some(&wrong_dtype))
            .is_err()
    );
    let multi_token = patterned_bf16((1, 2, config.hidden_size), 7, 0.02, &device);
    assert!(prepared.forward_prepared_decode(&multi_token).is_err());
    assert!(
        prepared
            .forward_prepared_decode(&input.to_dtype(DType::F32).unwrap())
            .is_err()
    );
    let (actual_conv, actual_recurrent) = native_state(&prepared);
    assert_tensor_exact(&actual_conv, &conv);
    assert_tensor_exact(&actual_recurrent, &recurrent);
    assert_tensor_exact(
        &prepared.forward_prepared_decode(&input).unwrap(),
        &eager.forward(&input).unwrap(),
    );
    assert_native_state_exact(&prepared, &eager);
}
