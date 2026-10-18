use crate::{
    backends::common::{Backend, BufferMut, Context},
    config::learned_scale::LearnedScaleConfig,
    safetensors::{ParameterLoaderError, ParameterTree},
};

/// Reads the learned `values` of a per-channel scale and returns its effective scale.
pub fn read_effective_scale<B: Backend>(
    config: &LearnedScaleConfig,
    num_channels: u32,
    parameter_tree: &ParameterTree<B>,
) -> Result<Box<[f32]>, ParameterLoaderError<B>> {
    let values = parameter_tree.read_slice::<f32>("values", &[num_channels])?;
    Ok(config.effective_scale(&values))
}

pub fn create_f32_buffer<B: Backend>(
    values: &[f32],
    context: &B::Context,
) -> Result<B::GlobalBuffer, ParameterLoaderError<B>> {
    let mut buffer =
        context.create_buffer(std::mem::size_of_val(values)).map_err(ParameterLoaderError::BackendError)?;
    buffer.copyin(values);
    Ok(buffer)
}

/// Reads the learned `values` of a per-channel scale and returns its effective scale as an F32 buffer.
pub fn load_learned_scale<B: Backend>(
    config: &LearnedScaleConfig,
    num_channels: u32,
    parameter_tree: &ParameterTree<B>,
    context: &B::Context,
) -> Result<B::GlobalBuffer, ParameterLoaderError<B>> {
    create_f32_buffer(&read_effective_scale(config, num_channels, parameter_tree)?, context)
}
