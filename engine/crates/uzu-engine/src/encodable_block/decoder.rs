use std::range::Range;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{
            Transformer, TransformerEncodeError, TransformerNewError, TransformerState, WavefrontCarry, WavefrontSlots,
        },
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Transformer encoding error: {0}")]
    TransformerEncode(#[from] TransformerEncodeError<B>),
}

pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    embedding_norm: Option<Normalization<B>>,
    per_layer_embedding: Option<PerLayerEmbedding<B>>,
    transformer: Transformer<B>,
}

pub struct DecoderEncodeOutput<B: Backend> {
    pub logits: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
    pub final_hidden: Option<B::ScratchBuffer>,
}

impl<B: Backend> Decoder<B> {
    pub(crate) fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }

    pub fn new(
        data_type: DataType,
        config: &DecoderConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, DecoderError<B>> {
        let (embedding, readout_input_hadamard_factors) = Embedding::new(
            config.vocab_size,
            config.transformer_config.model_dim,
            data_type,
            &config.embedding_config,
            &parameter_tree.subtree("embedding"),
            context,
        )?;

        let embedding_norm = config
            .embedding_norm_config
            .as_ref()
            .map(|norm_config| {
                Normalization::new(
                    config.transformer_config.model_dim,
                    data_type,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    norm_config,
                    &parameter_tree.subtree("embedding_norm"),
                    context,
                )
            })
            .transpose()?;

        let per_layer_embedding = if let Some(ple_config) = &config.ple_model_config {
            assert_eq!(
                ple_config.num_layers,
                config.transformer_config.layer_configs.len() as u32,
                "per-layer embedding num_layers must match transformer layer count"
            );
            Some(PerLayerEmbedding::new(
                config.transformer_config.model_dim,
                data_type,
                ple_config,
                &parameter_tree.subtree("per_layer_embedding"),
                context,
            )?)
        } else {
            None
        };

        let transformer = Transformer::new(
            data_type,
            readout_input_hadamard_factors,
            &config.transformer_config,
            &parameter_tree.subtree("transformer"),
            context,
        )?;

        Ok(Self {
            embedding,
            embedding_norm,
            per_layer_embedding,
            transformer,
        })
    }

    pub fn speculation_supported(&self) -> bool {
        self.transformer.speculation_supported()
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.transformer.max_context_length()
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.transformer.prefill_cache_skips_trailing_layers()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        self.transformer.create_empty_state(max_context_length, context)
    }

    pub fn wavefront_supported(&self) -> bool {
        self.per_layer_embedding.is_none() && self.transformer.wavefront_supported()
    }

    pub fn wavefront_slots(&self) -> WavefrontSlots {
        self.transformer.wavefront_slots()
    }

    pub fn create_wavefront_carry(
        &self,
        context: &B::Context,
    ) -> Result<WavefrontCarry<B>, B::Error> {
        self.transformer.create_wavefront_carry(context)
    }

    /// One loop-wavefront step (see `transformer::wavefront`): `token_id` enters at `position`, and
    /// with `has_carried_row` the row carried in `carry` finishes. Returns logits for
    /// `[carried row, new row]` or `[new row]`; the new row's logits are an early exit after one loop.
    pub fn encode_wavefront(
        &self,
        token_id: impl BufferRef<Backend = B>,
        carry: &mut WavefrontCarry<B>,
        position: u32,
        has_carried_row: bool,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, DecoderError<B>> {
        assert!(self.per_layer_embedding.is_none(), "wavefront does not support per-layer embeddings");
        let command_buffer = &mut *command_buffer.debug_group("decoder wavefront");
        let embedded = self.embedding.encode_lookup(token_id, 1, command_buffer)?;
        let embedded = match &self.embedding_norm {
            Some(embedding_norm) => embedding_norm
                .encode(&embedded, 0, 1, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DecoderError::Backend)?,
            None => embedded,
        };
        let hidden =
            self.transformer.encode_wavefront(embedded, carry, position, has_carried_row, state, command_buffer)?;
        let rows = 1 + has_carried_row as u32;
        Ok(self.embedding.encode_readout(
            &hidden,
            rows,
            self.embedding.vocab_size(),
            None::<&B::ScratchBuffer>,
            true,
            command_buffer,
        )?)
    }

    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("decoder");

        let embedded = self.embedding.encode_lookup(token_ids, batch_dim.size(), command_buffer)?;
        let embedded = if let Some(embedding_norm) = &self.embedding_norm {
            embedding_norm
                .encode(&embedded, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DecoderError::Backend)?
        } else {
            embedded
        };

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            Some(per_layer_embedding.encode(token_ids, &embedded, batch_dim.size(), command_buffer)?)
        } else {
            None
        };

        let transformer_output = self.transformer.encode(
            embedded,
            per_layer_inputs.as_ref(),
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            Some(state),
            command_buffer,
        )?;

        let logits = if let Some(output_range) = output_range {
            let output = transformer_output.output.as_ref().expect("decoder output range requires transformer output");
            Some(self.embedding.encode_readout(
                output,
                output_range.end - output_range.start,
                self.embedding.vocab_size(),
                None::<&B::ScratchBuffer>,
                true,
                command_buffer,
            )?)
        } else {
            None
        };
        let final_hidden = if hidden_feature_layer_indices.is_none() {
            None
        } else {
            transformer_output.output
        };

        Ok(DecoderEncodeOutput {
            logits,
            hidden_features: transformer_output.hidden_features,
            final_hidden,
        })
    }
}
