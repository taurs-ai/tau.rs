//! Common observation facts for tests; production callers supply their own metadata.

use crate::{GraphPolicy, RequestMetadata};

/// Gives fixture tests explicit metadata without importing any engine defaults.
pub(crate) fn request<'a>(
    model: &'a str,
    dtype: &'static str,
    max_new_tokens: usize,
    eos_token_ids: &'a [u32],
) -> RequestMetadata<'a> {
    RequestMetadata {
        model,
        dtype,
        max_new_tokens,
        eos_token_ids,
        optimize_kernel: true,
        prefill_chunk_size: 256,
        graph_policy: GraphPolicy::None,
    }
}
