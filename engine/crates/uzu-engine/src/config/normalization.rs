use uzu_engine_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum UpcastMode {
    OnlyNormalization,
    FullLayer,
}

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum NormReduction {
    Mean,
    Sum,
}

#[uzu_config]
pub struct NormalizationConfig {
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    pub subtract_mean: bool,
    pub has_scale: bool,
    pub has_biases: bool,
    pub reduction: NormReduction,
}

impl NormalizationConfig {
    /// `x * rsqrt(sum(x^2) + epsilon)`: a sum-reduction norm without scale, biases or mean subtraction.
    pub fn is_unit_l2(&self) -> bool {
        self.reduction == NormReduction::Sum && !self.has_scale && !self.has_biases && !self.subtract_mean
    }
}
