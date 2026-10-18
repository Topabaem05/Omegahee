use uzu_engine_macros::{uzu_config, uzu_config_abstract};

use crate::config::{
    learned_scale::LearnedScaleConfig,
    linear::LinearConfig,
    mlp::AnyMLPConfig,
    normalization::NormalizationConfig,
    per_layer_embedding::PLELayerConfig,
    rope::AnyRoPEConfig,
    token_mixer::{AnyTokenMixerConfig, convolutions::SeparableCausalConvConfig},
};

#[uzu_config]
pub struct TransformerLayerConvConfig {
    pub conv_config: SeparableCausalConvConfig,
    pub kernel_projection_config: LinearConfig,
    pub conv_kernel_size: u32,
    pub conv_group_size: u32,
}

#[uzu_config_abstract(TransformerLayerConfig, NGPTAttentionLayerConfig, NGPTMLPLayerConfig)]
pub struct LayerConfig {
    pub hidden_dim: Option<u32>,
    pub kv_source_layer_index: Option<u32>,
    pub rope_config: Option<AnyRoPEConfig>,
}

#[uzu_config(self::LayerConfig)]
pub struct TransformerLayerConfig {
    pub pre_mixer_norm_config: Option<NormalizationConfig>,
    pub mixer_config: AnyTokenMixerConfig,
    pub mixer_conv_config: Option<TransformerLayerConvConfig>,
    pub post_mixer_norm_config: Option<NormalizationConfig>,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: AnyMLPConfig,
    pub mlp_conv_config: Option<TransformerLayerConvConfig>,
    pub post_mlp_norm_config: Option<NormalizationConfig>,
    pub ple_config: Option<PLELayerConfig>,
    pub has_post_layer_scalar: bool,
}

/// An attention-only layer with the normalized-GPT residual update
/// `outputs = norm(norm(inputs) + scale * (norm(mixer(inputs)) - norm(inputs)))`.
#[uzu_config(self::LayerConfig)]
pub struct NGPTAttentionLayerConfig {
    pub residual_norm_config: NormalizationConfig,
    pub residual_scale_config: LearnedScaleConfig,
    pub mixer_config: AnyTokenMixerConfig,
}

/// An MLP-only layer with the normalized-GPT residual update.
#[uzu_config(self::LayerConfig)]
pub struct NGPTMLPLayerConfig {
    pub residual_norm_config: NormalizationConfig,
    pub residual_scale_config: LearnedScaleConfig,
    pub mlp_config: AnyMLPConfig,
}

impl AnyLayerConfig {
    pub fn mixer_config(&self) -> Option<&AnyTokenMixerConfig> {
        match self {
            Self::TransformerLayerConfig(config) => Some(&config.mixer_config),
            Self::NGPTAttentionLayerConfig(config) => Some(&config.mixer_config),
            Self::NGPTMLPLayerConfig(_) => None,
        }
    }
}
