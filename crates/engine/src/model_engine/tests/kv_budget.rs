//! Kv budget contracts for the shared causal engine.

use super::*;

/// Integer-only policy can be exercised without loading a model or querying CUDA.
#[test]
fn request_limits_cap_before_addition_and_classify_failures() {
    use crate::model_engine::kv_budget::{RequestLimitError, effective_output_limit};
    assert_eq!(effective_output_limit(27, 128, 200, None), Ok(128));
    assert_eq!(
        effective_output_limit(27, 128, 150, None),
        Err(RequestLimitError::ContextOverflow {
            requested_tokens: 155,
            context_window: 150
        })
    );
    assert_eq!(
        effective_output_limit(1, usize::MAX, usize::MAX, None),
        Err(RequestLimitError::TokenBudgetOverflow)
    );
    let budget = KvCacheBudget {
        max_kv_cache_capacity: 200,
        concurrent_requests: 2,
        headroom_bytes: 0,
    };
    assert_eq!(
        effective_output_limit(27, usize::MAX, 80, Some(budget)),
        Ok(53)
    );
    assert_eq!(
        effective_output_limit(80, 1, 80, Some(budget)),
        Err(RequestLimitError::KvCapacityExceeded {
            prompt_tokens: 80,
            capacity: 80
        })
    );
    let largest = KvCacheBudget {
        max_kv_cache_capacity: usize::MAX,
        concurrent_requests: 1,
        headroom_bytes: 0,
    };
    assert_eq!(
        effective_output_limit(usize::MAX - 1, usize::MAX, usize::MAX, Some(largest)),
        Ok(1)
    );
}

/// Establish error ordering and instrumentation boundaries before extracting
/// request limits. These rejected requests must not reserve or reset model state.
#[test]
fn request_limit_errors_and_zero_output_preserve_request_boundaries() {
    let mut engine =
        sequence_engine(EngineOptions::new().with_tracing(true).with_metrics(true));
    let empty = engine.generate_from_prompt_with_metadata("", 0).unwrap();
    assert_eq!(empty.prompt_tokens(), 0);
    assert_eq!(empty.completion_tokens(), 0);
    assert_eq!(empty.finish_reason(), ChatFinishReason::Length);
    assert!(
        engine
            .generate_from_prompt("", 1)
            .unwrap_err()
            .to_string()
            .contains("prompt produced no tokens")
    );
    assert!(
        engine
            .generate_from_prompt("hello", usize::MAX)
            .unwrap_err()
            .to_string()
            .contains("request token budget overflow")
    );
    assert!(
        engine
            .generate_from_prompt("hello", 32)
            .unwrap_err()
            .to_string()
            .contains("request needs 33 tokens but model context is 32")
    );
    engine.options.kv_cache_budget = Some(KvCacheBudget {
        max_kv_cache_capacity: 1,
        concurrent_requests: 1,
        headroom_bytes: 0,
    });
    assert!(
        engine
            .generate_from_prompt("hello", usize::MAX)
            .unwrap_err()
            .to_string()
            .contains("prompt has 1 tokens but the KV cache limit is 1")
    );
    assert!(engine.model.calls.is_empty() && engine.model.reservations.is_empty());
    assert_eq!(engine.model.resets, 0);
    engine.options.kv_cache_budget = Some(KvCacheBudget {
        max_kv_cache_capacity: 5,
        concurrent_requests: 1,
        headroom_bytes: 0,
    });
    engine.model.eos_offset = usize::MAX;
    let capped = engine
        .generate_from_prompt_with_metadata("hello", usize::MAX)
        .unwrap();
    assert_eq!(capped.completion_tokens(), 4);
    assert_eq!(capped.finish_reason(), ChatFinishReason::Length);
    assert_eq!(engine.model.resets, 2);
}

#[test]
fn kv_budget_accounts_for_workspace_and_divides_concurrent_requests() {
    let gib = 1024 * 1024 * 1024;
    let budget =
        KvCacheBudget::from_memory(8 * gib, 16 * gib, 32768, 248832, 1).unwrap();
    let bytes_per_slot = 32768 + 248832;
    assert_eq!(budget.headroom_bytes, 16 * gib / 4);
    assert!(
        budget.headroom_bytes + budget.max_kv_cache_capacity * bytes_per_slot
            <= 8 * gib
    );
    assert!(
        budget.headroom_bytes + (budget.max_kv_cache_capacity + 1) * bytes_per_slot
            > 8 * gib
    );
    let shared =
        KvCacheBudget::from_memory(8 * gib, 16 * gib, 32768, 248832, 3).unwrap();
    assert_eq!(shared.max_kv_cache_capacity, budget.max_kv_cache_capacity);
    assert_eq!(
        shared.per_request_capacity(usize::MAX),
        budget.max_kv_cache_capacity / 3
    );
    assert_eq!(budget.per_request_capacity(32), 32);
    let without_workspace =
        KvCacheBudget::from_memory(8 * gib, 16 * gib, 32768, 0, 1).unwrap();
    assert!(without_workspace.max_kv_cache_capacity > budget.max_kv_cache_capacity);
}

#[test]
fn kv_budget_rejects_insufficient_memory_and_invalid_arithmetic() {
    let gib = 1024 * 1024 * 1024;
    assert!(KvCacheBudget::from_memory(gib, 16 * gib, 32768, 0, 1).is_err());
    assert!(KvCacheBudget::from_memory(8 * gib, 16 * gib, 32768, 0, 0).is_err());
    assert!(KvCacheBudget::from_memory(17 * gib, 16 * gib, 32768, 0, 1).is_err());
    assert!(KvCacheBudget::from_memory(8 * gib, 16 * gib, 0, 0, 1).is_err());
    assert!(KvCacheBudget::from_memory(8 * gib, 16 * gib, usize::MAX, 1, 1).is_err());
    let small = KvCacheBudget::from_memory(3 * gib, 4 * gib, 32768, 0, 1).unwrap();
    assert_eq!(small.headroom_bytes, 2 * gib);
}

#[test]
fn kv_capacity_caps_output_and_preserves_eos_and_smaller_requests() {
    let mut options = EngineOptions::new();
    options.kv_cache_budget = Some(KvCacheBudget {
        max_kv_cache_capacity: 10,
        concurrent_requests: 2,
        headroom_bytes: 0,
    });
    let mut engine = sequence_engine(options);
    engine.model.eos_offset = usize::MAX;
    // Clamp before adding: even a huge caller limit cannot overflow the sum.
    let completion = engine
        .generate_from_prompt_with_metadata("hello", usize::MAX)
        .unwrap();
    assert_eq!(completion.completion_tokens(), 4);
    assert_eq!(completion.finish_reason(), ChatFinishReason::Length);
    assert_eq!(engine.model.reservations, [(5, 1)]);
    assert_eq!(engine.model.calls, [0, 1, 2, 3]);
    let completion = engine
        .generate_from_prompt_with_metadata("hello", 2)
        .unwrap();
    assert_eq!(completion.completion_tokens(), 2);
    assert_eq!(engine.model.reservations.last(), Some(&(3, 3)));
    engine.model.eos_offset = 1;
    let completion = engine
        .generate_from_prompt_with_metadata("hello", 100)
        .unwrap();
    assert_eq!(completion.completion_tokens(), 1);
    assert_eq!(completion.finish_reason(), ChatFinishReason::Stop);
}

#[test]
fn kv_capacity_rejects_prompt_without_output_space_before_reservation() {
    let mut options = EngineOptions::new();
    options.kv_cache_budget = Some(KvCacheBudget {
        max_kv_cache_capacity: 1,
        concurrent_requests: 1,
        headroom_bytes: 0,
    });
    let mut engine = sequence_engine(options);
    let error = engine
        .generate_from_prompt_with_metadata("hello", 100)
        .unwrap_err();
    assert!(error.to_string().contains("shorten the prompt"));
    assert!(engine.model.reservations.is_empty());
    assert!(engine.model.calls.is_empty());
}
