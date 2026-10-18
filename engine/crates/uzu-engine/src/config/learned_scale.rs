use uzu_engine_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum ScaleTransform {
    Identity,
    Square,
    Absolute,
}

/// A per-channel scale computed as `transform(values * multiplier)`.
#[uzu_config]
pub struct LearnedScaleConfig {
    pub multiplier: f32,
    pub transform: ScaleTransform,
}

impl LearnedScaleConfig {
    pub fn effective_scale(
        &self,
        values: &[f32],
    ) -> Box<[f32]> {
        values
            .iter()
            .map(|value| {
                let scaled = value * self.multiplier;
                match self.transform {
                    ScaleTransform::Identity => scaled,
                    ScaleTransform::Square => scaled * scaled,
                    ScaleTransform::Absolute => scaled.abs(),
                }
            })
            .collect()
    }
}
