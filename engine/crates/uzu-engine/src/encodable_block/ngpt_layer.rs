//! Normalized-GPT layers: a single sublayer (attention or MLP) followed by the residual update
//! `outputs = norm(norm(inputs) + scale * (norm(sublayer(inputs)) - norm(inputs)))`.
//!
//! Unlike [`TransformerLayer`](super::transformer_layer::TransformerLayer), the hidden state passed
//! between nGPT layers is the full residual stream, so no shortcut buffer is involved.

use std::any::Any;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, Buffer, CommandBuffer, CommandBufferEncoding, Kernels, kernel::NgptResidualMixKernel},
    config::{
        normalization::NormalizationConfig,
        transformer_layer::{NGPTAttentionLayerConfig, NGPTMLPLayerConfig},
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        learned_scale::load_learned_scale,
        mixer::{
            Mixer, MixerEncodeError, MixerNewError, MixerState,
            attention::{Attention, AttentionRowGroup, rope::PrecalculatedRoPE},
        },
        mlp::{Mlp, MlpBlockNewError, MlpEncodeError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

#[derive(Debug, Error)]
pub enum NgptLayerError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Mixer error: {0}")]
    Mixer(#[from] MixerNewError<B>),
    #[error("Mixer encoding error: {0}")]
    MixerEncode(#[from] MixerEncodeError<B>),
    #[error("MLP error: {0}")]
    MlpBlock(#[from] MlpBlockNewError<B>),
    #[error("MLP encoding error: {0}")]
    MlpEncode(#[from] MlpEncodeError<B>),
    #[error("Layer {layer_index}: the nGPT residual normalization must be a weightless sum-reduction norm")]
    UnsupportedResidualNorm {
        layer_index: u32,
    },
    #[error("Layer {layer_index}: nGPT layers do not support input Hadamard transforms")]
    UnsupportedInputTransform {
        layer_index: u32,
    },
}

pub enum NgptSublayer<B: Backend> {
    Mixer(Box<dyn Mixer<B>>),
    Mlp(Box<dyn Mlp<B>>),
}

pub struct NgptLayer<B: Backend> {
    pub layer_index: u32,
    pub sublayer: NgptSublayer<B>,
    model_dim: u32,
    epsilon: f32,
    residual_weights: B::GlobalBuffer,
    residual_mix: <B::Kernels as Kernels>::NgptResidualMixKernel,
    /// Writes the updated residual twice, so the next layer can consume one copy without a separate copy pass.
    residual_mix_with_copy: <B::Kernels as Kernels>::NgptResidualMixKernel,
}

fn residual_epsilon<B: Backend>(
    config: &NormalizationConfig,
    layer_index: u32,
) -> Result<f32, NgptLayerError<B>> {
    if !config.is_unit_l2() {
        return Err(NgptLayerError::UnsupportedResidualNorm {
            layer_index,
        });
    }
    Ok(config.epsilon)
}

impl<B: Backend> NgptLayer<B> {
    fn with_sublayer(
        layer_index: u32,
        model_dim: u32,
        data_type: DataType,
        sublayer: NgptSublayer<B>,
        residual_norm_config: &NormalizationConfig,
        residual_scale_config: &crate::config::learned_scale::LearnedScaleConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, NgptLayerError<B>> {
        let epsilon = residual_epsilon(residual_norm_config, layer_index)?;
        let residual_weights =
            load_learned_scale(residual_scale_config, model_dim, &parameter_tree.subtree("residual_scale"), context)?;
        let residual_mix = <B::Kernels as Kernels>::NgptResidualMixKernel::new(context, data_type, false)
            .map_err(NgptLayerError::Backend)?;
        let residual_mix_with_copy = <B::Kernels as Kernels>::NgptResidualMixKernel::new(context, data_type, true)
            .map_err(NgptLayerError::Backend)?;
        Ok(Self {
            layer_index,
            sublayer,
            model_dim,
            epsilon,
            residual_weights,
            residual_mix,
            residual_mix_with_copy,
        })
    }

    pub fn new_attention(
        model_dim: u32,
        layer_index: u32,
        data_type: DataType,
        config: &NGPTAttentionLayerConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, NgptLayerError<B>> {
        let (mixer, input_hadamard_factors) = <dyn Mixer<B>>::new(
            model_dim,
            data_type,
            config.rope_config.as_ref(),
            &config.mixer_config,
            &parameter_tree.subtree("mixer"),
            context,
        )?;
        if input_hadamard_factors.is_some() {
            return Err(NgptLayerError::UnsupportedInputTransform {
                layer_index,
            });
        }
        Self::with_sublayer(
            layer_index,
            model_dim,
            data_type,
            NgptSublayer::Mixer(mixer),
            &config.residual_norm_config,
            &config.residual_scale_config,
            parameter_tree,
            context,
        )
    }

    pub fn new_mlp(
        model_dim: u32,
        hidden_dim: u32,
        layer_index: u32,
        data_type: DataType,
        config: &NGPTMLPLayerConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, NgptLayerError<B>> {
        let (mlp, input_hadamard_factors) = <dyn Mlp<B>>::new(
            model_dim,
            config.hidden_dim.unwrap_or(hidden_dim),
            data_type,
            &config.mlp_config,
            &parameter_tree.subtree("mlp"),
            context,
        )?;
        if input_hadamard_factors.is_some() {
            return Err(NgptLayerError::UnsupportedInputTransform {
                layer_index,
            });
        }
        Self::with_sublayer(
            layer_index,
            model_dim,
            data_type,
            NgptSublayer::Mlp(mlp),
            &config.residual_norm_config,
            &config.residual_scale_config,
            parameter_tree,
            context,
        )
    }

    pub fn mixer(&self) -> Option<&dyn Mixer<B>> {
        match &self.sublayer {
            NgptSublayer::Mixer(mixer) => Some(mixer.as_ref()),
            NgptSublayer::Mlp(_) => None,
        }
    }

    /// Returns the updated residual stream.
    ///
    /// The sublayer consumes its input buffer while the residual update still needs the input, so the
    /// layer needs two copies of its input: `input_copy` from the previous layer, or a copy pass here.
    /// With `emit_copy` it also returns a second copy of its output for the next layer.
    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        input_copy: Option<B::ScratchBuffer>,
        emit_copy: bool,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, Option<B::ScratchBuffer>), NgptLayerError<B>> {
        let command_buffer = &mut *command_buffer.debug_group(&format!("ngpt layer {}", self.layer_index));
        let sublayer_input = Self::sublayer_input(&input, input_copy, command_buffer)?;
        let update = match &self.sublayer {
            NgptSublayer::Mixer(mixer) => {
                mixer.encode(sublayer_input, precalculated_rope, batch_dim, state, command_buffer)?
            },
            NgptSublayer::Mlp(mlp) => {
                assert!(state.is_none(), "nGPT MLP layers do not own a state");
                mlp.encode(sublayer_input, batch_dim.size(), command_buffer)?
            },
        };
        self.mix_residual(&input, &update, emit_copy, batch_dim.size(), command_buffer)
    }

    /// Like [`Self::encode`] for an attention layer whose consecutive row groups each read and append
    /// their own KV state at their own positions (rows from different loops of a loop wavefront).
    pub fn encode_row_groups(
        &self,
        input: B::ScratchBuffer,
        input_copy: Option<B::ScratchBuffer>,
        emit_copy: bool,
        groups: &mut [AttentionRowGroup<'_, '_, B>],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, Option<B::ScratchBuffer>), NgptLayerError<B>> {
        let command_buffer = &mut *command_buffer.debug_group(&format!("ngpt layer {} (row groups)", self.layer_index));
        let rows = groups.iter().map(|group| group.batch_dim.size()).sum::<u32>();
        let sublayer_input = Self::sublayer_input(&input, input_copy, command_buffer)?;
        let NgptSublayer::Mixer(mixer) = &self.sublayer else {
            panic!("row groups need an attention sublayer");
        };
        let attention =
            (mixer.as_ref() as &dyn Any).downcast_ref::<Attention<B>>().expect("row groups need an attention sublayer");
        let update =
            attention.attend_row_groups(sublayer_input, groups, command_buffer).map_err(MixerEncodeError::Attention)?;
        self.mix_residual(&input, &update, emit_copy, rows, command_buffer)
    }

    fn sublayer_input(
        input: &B::ScratchBuffer,
        input_copy: Option<B::ScratchBuffer>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, NgptLayerError<B>> {
        match input_copy {
            Some(input_copy) => Ok(input_copy),
            None => {
                let mut copy = command_buffer.allocate_scratch(input.size()).map_err(NgptLayerError::Backend)?;
                command_buffer.encode_copy(input, &mut copy);
                Ok(copy)
            },
        }
    }

    fn mix_residual(
        &self,
        input: &B::ScratchBuffer,
        update: &B::ScratchBuffer,
        emit_copy: bool,
        rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, Option<B::ScratchBuffer>), NgptLayerError<B>> {
        let mut output = command_buffer.allocate_scratch(input.size()).map_err(NgptLayerError::Backend)?;
        let mut output_copy = emit_copy
            .then(|| command_buffer.allocate_scratch(input.size()))
            .transpose()
            .map_err(NgptLayerError::Backend)?;
        let residual_mix = if emit_copy {
            &self.residual_mix_with_copy
        } else {
            &self.residual_mix
        };
        residual_mix.encode(
            input,
            update,
            &self.residual_weights,
            &mut output,
            output_copy.as_mut(),
            rows,
            self.model_dim,
            self.epsilon,
            command_buffer,
        );
        Ok((output, output_copy))
    }
}
