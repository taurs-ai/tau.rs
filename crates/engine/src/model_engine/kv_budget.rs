//! Pure KV memory-budget arithmetic. Device queries and model mutation belong
//! to the factory; callers supply the measured memory and per-token estimates.

use crate::InferenceError;
use crate::model::ModelLoadError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KvCacheBudget {
    pub(super) max_kv_cache_capacity: usize,
    pub(super) concurrent_requests: usize,
    pub(super) headroom_bytes: usize,
}

impl KvCacheBudget {
    /// Calculates startup capacity from a supplied memory snapshot and per-token
    /// storage/workspace costs. This function performs no device queries.
    pub(super) fn from_memory(
        free_bytes: usize,
        total_bytes: usize,
        kv_bytes_per_token: usize,
        workspace_bytes_per_token: usize,
        concurrent_requests: usize,
    ) -> Result<Self, ModelLoadError> {
        if free_bytes > total_bytes
            || kv_bytes_per_token == 0
            || concurrent_requests == 0
        {
            return Err(ModelLoadError::InvalidConfig(
                "invalid KV memory budget inputs".to_owned(),
            ));
        }
        // The caller measures after loading weights. Leave the larger of 2 GiB
        // and one quarter of VRAM for activations, GDN state, capture and allocator
        // slack.
        // Additionally charge context-scaled attention workspace for each token.
        // A free-memory snapshot cannot protect against later external GPU users.
        let headroom_bytes = (2 * 1024 * 1024 * 1024).max(total_bytes / 4);
        let per_token = kv_bytes_per_token
            .checked_add(workspace_bytes_per_token)
            .ok_or_else(|| {
                ModelLoadError::InvalidConfig("KV memory estimate overflow".to_owned())
            })?;
        let max_kv_cache_capacity =
            free_bytes.saturating_sub(headroom_bytes) / per_token;
        if max_kv_cache_capacity / concurrent_requests == 0 {
            return Err(ModelLoadError::InvalidConfig(format!(
                "insufficient free VRAM for KV cache after reserving {headroom_bytes} \
                 bytes of workspace headroom"
            )));
        }
        Ok(Self {
            max_kv_cache_capacity,
            concurrent_requests,
            headroom_bytes,
        })
    }

    /// Divides the global token capacity and clips it to the model context limit.
    pub(super) fn per_request_capacity(self, context_window: usize) -> usize {
        (self.max_kv_cache_capacity / self.concurrent_requests).min(context_window)
    }
}

/// Pure request-limit failures. The engine owns failure logging and API reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestLimitError {
    KvCapacityExceeded {
        prompt_tokens: usize,
        capacity: usize,
    },
    TokenBudgetOverflow,
    ContextOverflow {
        requested_tokens: usize,
        context_window: usize,
    },
}

impl From<RequestLimitError> for InferenceError {
    fn from(error: RequestLimitError) -> Self {
        let message = match error {
            RequestLimitError::KvCapacityExceeded {
                prompt_tokens,
                capacity,
            } => format!(
                "prompt has {prompt_tokens} tokens but the KV cache limit is \
                 {capacity}; shorten the prompt to leave room for output"
            ),
            RequestLimitError::TokenBudgetOverflow => {
                "request token budget overflow".to_owned()
            }
            RequestLimitError::ContextOverflow {
                requested_tokens,
                context_window,
            } => format!(
                "request needs {requested_tokens} tokens but model context is \
                 {context_window}"
            ),
        };
        Self::Backend(message)
    }
}

/// Limits outputs while retaining the full prompt and checking the model context.
/// The caller handles zero-output requests and empty prompts before this policy.
/// Capping must precede addition: even a usize::MAX requested limit can fit when
/// an automatic KV budget reduces it. No device, model or instrumentation is used.
pub(super) fn effective_output_limit(
    prompt_tokens: usize,
    requested_limit: usize,
    context_window: usize,
    budget: Option<KvCacheBudget>,
) -> Result<usize, RequestLimitError> {
    let effective = if let Some(budget) = budget {
        let capacity = budget.per_request_capacity(context_window);
        let remaining = capacity
            .checked_sub(prompt_tokens)
            .filter(|&n| n > 0)
            .ok_or(RequestLimitError::KvCapacityExceeded {
                prompt_tokens,
                capacity,
            })?;
        requested_limit.min(remaining)
    } else {
        requested_limit
    };
    let requested_tokens = prompt_tokens
        .checked_add(effective)
        .ok_or(RequestLimitError::TokenBudgetOverflow)?;
    if requested_tokens > context_window {
        return Err(RequestLimitError::ContextOverflow {
            requested_tokens,
            context_window,
        });
    }
    Ok(effective)
}
