//! Shared serialized configuration for the Qwen3.5 hybrid text decoder.

use crate::ModelLoadError;
use crate::components::{
    GatedAttentionConfig, GatedDeltaNetConfig, PartialRotaryConfig,
};
use candle_nn::Activation;
use serde::Deserialize;

/// Token mixer selected for one Qwen3.5 decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HybridLayerType {
    LinearAttention,
    FullAttention,
}

/// Serialized RoPE parameters shared by dense and MoE full-attention layers.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct RopeParameters {
    pub(crate) rope_type: String,
    pub(crate) rope_theta: f64,
    pub(crate) partial_rotary_factor: f64,
}

impl RopeParameters {
    /// Rejects RoPE behavior unsupported by the shared partial-RoPE component.
    fn validate(&self, family: &str) -> Result<(), ModelLoadError> {
        if self.rope_type != "default" {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} currently supports default RoPE, got {:?}",
                self.rope_type
            )));
        }
        if !(0.0..=1.0).contains(&self.partial_rotary_factor) {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} partial_rotary_factor must be between 0 and 1, got {}",
                self.partial_rotary_factor
            )));
        }
        Ok(())
    }
}

/// Serialized fields required by every Qwen3.5 hybrid text decoder.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct HybridTextConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) attention_bias: bool,
    pub(crate) attention_dropout: f64,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) use_cache: bool,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_parameters: RopeParameters,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) layer_types: Vec<HybridLayerType>,
}

impl HybridTextConfig {
    /// Translates serialized full-attention fields into component configuration.
    pub(crate) fn gated_attention_config(&self) -> GatedAttentionConfig {
        GatedAttentionConfig {
            hidden_size: self.hidden_size,
            num_query_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            projection_bias: self.attention_bias,
            norm_eps: self.rms_norm_eps,
        }
    }

    /// Translates serialized linear-attention fields into component configuration.
    pub(crate) fn gated_delta_net_config(&self) -> GatedDeltaNetConfig {
        GatedDeltaNetConfig {
            hidden_size: self.hidden_size,
            conv_kernel_size: self.linear_conv_kernel_dim,
            key_head_dim: self.linear_key_head_dim,
            value_head_dim: self.linear_value_head_dim,
            num_key_heads: self.linear_num_key_heads,
            num_value_heads: self.linear_num_value_heads,
            norm_eps: self.rms_norm_eps,
            conv_activation: self.hidden_act,
            gate_activation: self.hidden_act,
        }
    }

    /// Translates fractional RoPE metadata into concrete runtime dimensions.
    pub(crate) fn partial_rotary_config(&self) -> PartialRotaryConfig {
        PartialRotaryConfig {
            head_dim: self.head_dim,
            rotary_dim: (self.head_dim as f64
                * self.rope_parameters.partial_rotary_factor)
                as usize,
            max_position_embeddings: self.max_position_embeddings,
            theta: self.rope_parameters.rope_theta,
        }
    }

    /// Returns the width emitted by the gated query projection before splitting.
    pub(crate) fn query_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Returns the width of unexpanded full-attention keys and values.
    pub(crate) fn key_value_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Returns the width of linear-attention keys before head repetition.
    pub(crate) fn linear_key_width(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Returns the width of linear-attention values and output gates.
    pub(crate) fn linear_value_width(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Returns the packed Q/K/V width consumed by the depthwise convolution.
    pub(crate) fn linear_convolution_width(&self) -> usize {
        2 * self.linear_key_width() + self.linear_value_width()
    }

    /// Rejects malformed hybrid-decoder behavior shared by dense and MoE models.
    pub(crate) fn validate(&self, family: &str) -> Result<(), ModelLoadError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.max_position_embeddings == 0
        {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} dimensions and vocabulary must be non-zero"
            )));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} layer_types has {} entries for {} decoder layers",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if !self.layer_types.contains(&HybridLayerType::LinearAttention)
            || !self.layer_types.contains(&HybridLayerType::FullAttention)
        {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} requires both linear_attention and full_attention layers"
            )));
        }
        self.gated_attention_config()
            .validate()
            .map_err(ModelLoadError::InvalidConfig)?;
        self.gated_delta_net_config()
            .validate()
            .map_err(ModelLoadError::InvalidConfig)?;
        self.rope_parameters.validate(family)?;
        self.partial_rotary_config()
            .validate()
            .map_err(ModelLoadError::InvalidConfig)?;
        if self.attention_dropout != 0.0 {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} inference requires attention_dropout=0"
            )));
        }
        if !self.use_cache {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} inference requires use_cache=true"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a valid shared hybrid-text configuration for component tests.
    fn valid_config() -> HybridTextConfig {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "use_cache": true,
            "tie_word_embeddings": false,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5
            },
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ]
        }))
        .unwrap()
    }

    /// Verifies component configurations and dimensions derived from shared fields.
    #[test]
    fn derives_shared_component_configs_and_dimensions() {
        let config = valid_config();
        config.validate("test Qwen3.5").unwrap();

        assert_eq!(config.query_width(), 16);
        assert_eq!(config.key_value_width(), 8);
        assert_eq!(config.linear_key_width(), 8);
        assert_eq!(config.linear_value_width(), 16);
        assert_eq!(config.linear_convolution_width(), 32);

        let attention = config.gated_attention_config();
        assert_eq!(attention.hidden_size, 16);
        assert_eq!(attention.num_query_heads, 2);
        assert_eq!(attention.num_key_value_heads, 1);
        assert_eq!(attention.head_dim, 8);
        assert!(!attention.projection_bias);
        assert_eq!(attention.norm_eps, 1e-6);

        let delta_net = config.gated_delta_net_config();
        assert_eq!(delta_net.hidden_size, 16);
        assert_eq!(delta_net.conv_kernel_size, 4);
        assert_eq!(delta_net.key_head_dim, 4);
        assert_eq!(delta_net.value_head_dim, 4);
        assert_eq!(delta_net.num_key_heads, 2);
        assert_eq!(delta_net.num_value_heads, 4);
        assert_eq!(delta_net.norm_eps, 1e-6);
        assert_eq!(delta_net.conv_activation, Activation::Silu);
        assert_eq!(delta_net.gate_activation, Activation::Silu);

        let rotary = config.partial_rotary_config();
        assert_eq!(rotary.head_dim, 8);
        assert_eq!(rotary.rotary_dim, 4);
        assert_eq!(rotary.max_position_embeddings, 128);
        assert_eq!(rotary.theta, 10_000_000.0);
    }

    /// Verifies rejection of invalid schedules and rotary configuration.
    #[test]
    fn rejects_invalid_shared_schedule_and_rope() {
        let mut config = valid_config();
        config.layer_types.fill(HybridLayerType::LinearAttention);
        let error = config.validate("test Qwen3.5").unwrap_err();
        assert!(error.to_string().contains("requires both linear_attention"));

        let mut config = valid_config();
        config.rope_parameters.rope_type = "dynamic".to_owned();
        let error = config.validate("test Qwen3.5").unwrap_err();
        assert!(error.to_string().contains("supports default RoPE"));
    }
}
