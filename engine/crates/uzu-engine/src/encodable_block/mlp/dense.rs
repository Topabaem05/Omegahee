//! MLP block encodable.

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::activation_type::ActivationType,
        kernel::{NgptSwiGLUNormKernel, ScaleColumnsKernel},
    },
    config::{clipping::ClippingBounds, learned_scale::LearnedScaleConfig, mlp::dense_mlp::DenseMLPConfig},
    data_type::DataType,
    encodable_block::{
        learned_scale::{create_f32_buffer, read_effective_scale},
        linear::{Linear, LinearBlockError, LinearInput},
        mlp::{Mlp, MlpEncodeError, gate_act_mul::MlpGateActMulEncodable},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn Linear<B>>,
    hidden_dimension: u32,
    data_type: DataType,
    /// Effective per-channel scales of the fused [up, gate] projection output, applied in place.
    channel_scales: Option<(B::GlobalBuffer, <B::Kernels as Kernels>::ScaleColumnsKernel)>,
    hidden_norm: Option<Normalization<B>>,
    /// One kernel for channel scales, SwiGLU and the unit-L2 hidden normalization (nGPT MLPs).
    fused_hidden: Option<FusedHidden<B>>,
}

struct FusedHidden<B: Backend> {
    kernel: <B::Kernels as Kernels>::NgptSwiGLUNormKernel,
    epsilon: f32,
}

#[derive(Debug, Error)]
pub enum DenseMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("hidden_norm_config requires a full-precision down projection without input transforms")]
    HiddenNormWithTransformedDownInput,
}

fn read_effective_scale_or_ones<B: Backend>(
    config: Option<&LearnedScaleConfig>,
    num_channels: u32,
    parameter_tree: &ParameterTree<B>,
) -> Result<Box<[f32]>, ParameterLoaderError<B>> {
    match config {
        Some(config) => read_effective_scale(config, num_channels, parameter_tree),
        None => Ok(vec![1.0; num_channels as usize].into_boxed_slice()),
    }
}

impl<B: Backend> DenseMlp<B> {
    pub fn new(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &DenseMLPConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalBuffer>), DenseMlpNewError<B>> {
        let (up, up_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
            model_dimension,
            2 * hidden_dimension,
            data_type,
            config.has_up_biases,
            &parameter_tree.subtree("up_projection"),
            context,
        )?;

        let (down, down_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
            hidden_dimension,
            model_dimension,
            data_type,
            config.has_down_biases,
            &parameter_tree.subtree("down_projection"),
            context,
        )?;

        let hidden_norm = match &config.hidden_norm_config {
            Some(_) if down_input_preparation.is_some() => {
                return Err(DenseMlpNewError::HiddenNormWithTransformedDownInput);
            },
            Some(norm_config) => Some(Normalization::new(
                hidden_dimension,
                data_type,
                None,
                ShortcutMode::None,
                PostLayerScalar::None,
                norm_config,
                &parameter_tree.subtree("hidden_norm"),
                context,
            )?),
            None => None,
        };

        let channel_scales = if config.up_scale_config.is_some() || config.gate_scale_config.is_some() {
            let up_scales = read_effective_scale_or_ones(
                config.up_scale_config.as_ref(),
                hidden_dimension,
                &parameter_tree.subtree("up_scale"),
            )?;
            let gate_scales = read_effective_scale_or_ones(
                config.gate_scale_config.as_ref(),
                hidden_dimension,
                &parameter_tree.subtree("gate_scale"),
            )?;
            let kernel = <B::Kernels as Kernels>::ScaleColumnsKernel::new(context, data_type)
                .map_err(DenseMlpNewError::Backend)?;
            Some((create_f32_buffer(&[up_scales, gate_scales].concat(), context)?, kernel))
        } else {
            None
        };

        let fused_hidden = match &config.hidden_norm_config {
            Some(norm_config)
                if norm_config.is_unit_l2()
                    && config.activation.act_type() == ActivationType::SILU
                    && config.activation.custom_alpha().is_none()
                    && config.gate_clipping == ClippingBounds::default()
                    && config.up_clipping == ClippingBounds::default() =>
            {
                Some(FusedHidden {
                    kernel: <B::Kernels as Kernels>::NgptSwiGLUNormKernel::new(
                        context,
                        data_type,
                        channel_scales.is_some(),
                    )
                    .map_err(DenseMlpNewError::Backend)?,
                    epsilon: norm_config.epsilon,
                })
            },
            _ => None,
        };

        let gate = MlpGateActMulEncodable::new(
            hidden_dimension,
            data_type,
            config.activation.clone(),
            config.gate_clipping,
            config.up_clipping,
            down_input_preparation,
            context,
        )
        .map_err(DenseMlpNewError::Backend)?;

        Ok((
            Self {
                up,
                gate,
                down,
                hidden_dimension,
                data_type,
                channel_scales,
                hidden_norm,
                fused_hidden,
            },
            up_input_hadamard_factors,
        ))
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (dense)");

        let mut fused_up = self.up.encode(input, batch_dim, command_buffer)?;
        if let Some(fused_hidden) = &self.fused_hidden {
            let mut hidden = command_buffer
                .allocate_scratch_for_shape(&[batch_dim, self.hidden_dimension], self.data_type)
                .map_err(MlpEncodeError::Backend)?;
            fused_hidden.kernel.encode(
                &fused_up,
                self.channel_scales.as_ref().map(|(scales, _)| scales),
                &mut hidden,
                batch_dim,
                self.hidden_dimension,
                fused_hidden.epsilon,
                command_buffer,
            );
            return Ok(self.down.encode_input(LinearInput::FullPrecision(hidden), batch_dim, command_buffer)?);
        }
        if let Some((scales, scale_columns)) = &self.channel_scales {
            let fused_dimension = 2 * self.hidden_dimension;
            scale_columns.encode(&mut fused_up, scales, batch_dim, fused_dimension, fused_dimension, 0, command_buffer);
        }
        let act_format = self.down.select_activation_format(batch_dim, command_buffer.context());
        let down_input = self
            .gate
            .encode_for_linear(&fused_up, batch_dim, act_format, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        let down_input = match (&self.hidden_norm, down_input) {
            (None, down_input) => down_input,
            (Some(hidden_norm), LinearInput::FullPrecision(hidden)) => LinearInput::FullPrecision(
                hidden_norm
                    .encode(&hidden, 0, batch_dim, None::<&mut B::ScratchBuffer>, command_buffer)
                    .map_err(MlpEncodeError::Backend)?,
            ),
            (Some(_), _) => unreachable!("hidden normalization requires a full-precision down input"),
        };
        let output = self.down.encode_input(down_input, batch_dim, command_buffer)?;

        Ok(output)
    }
}
