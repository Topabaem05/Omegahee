use uzu_engine_macros::uzu_config;

use crate::config::{normalization::NormalizationConfig, transformer_layer::AnyLayerConfig};

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum RecurrentKVCacheMode {
    /// Every loop owns a separate cache for each recurrent layer.
    PerLoop,
    /// All loops share one cache per recurrent layer; every loop reads the cache as it was before the
    /// current forward pass, and the last loop's keys and values are kept for the current tokens.
    LastLoop,
}

/// Runs layers `[start_layer_index, end_layer_index)` `num_loops` times with shared weights.
///
/// With `h` the hidden state before the block, each loop starts from `norm(state + h)`
/// (`state = h` for the first loop) and ends with `state = norm(outputs)`; the block returns `state`.
#[uzu_config]
pub struct RecurrenceConfig {
    pub start_layer_index: u32,
    pub end_layer_index: u32,
    pub num_loops: u32,
    pub kv_cache_mode: RecurrentKVCacheMode,
    pub reinjection_norm_config: NormalizationConfig,
}

#[uzu_config]
pub struct TransformerConfig {
    pub layer_configs: Box<[AnyLayerConfig]>,
    pub output_norm_config: Option<NormalizationConfig>,
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub recurrence: Option<RecurrenceConfig>,
}
