use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MicrofloatScaleMode {
    Mxfp4,
    Nvfp4,
}

/// FP4 (e2m1) codes in groups of `group_size`. The engine reads BF16 group scales with the tensor scale
/// already folded in (`tools/nvfp4_to_omega.py`); packed FP8/E8M0 scales are not supported yet.
#[uzu_config(super::WeightMatrixSpec)]
pub struct MicrofloatSpec {
    pub bits: u32,
    pub group_size: u32,
    pub scale_mode: MicrofloatScaleMode,
    pub layout: Layout,
}
