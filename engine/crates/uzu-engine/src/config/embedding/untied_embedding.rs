use uzu_engine_macros::uzu_config;

use crate::config::learned_scale::LearnedScaleConfig;

#[uzu_config(super::EmbeddingConfig)]
pub struct UntiedEmbeddingConfig {
    /// Per-vocabulary-entry logit scale applied after the readout projection.
    pub logit_scales_config: Option<LearnedScaleConfig>,
}
