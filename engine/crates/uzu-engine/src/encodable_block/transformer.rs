use std::range::Range;

#[cfg(test)]
mod hybrid;
mod wavefront;
use derive_more::Debug;
use thiserror::Error;
pub use wavefront::{WavefrontCarry, WavefrontSlots};

use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, kernel::TensorAddScaleKernel,
    },
    config::{
        rope::AnyRoPEConfig,
        transformer::{RecurrentKVCacheMode, TransformerConfig},
        transformer_layer::AnyLayerConfig,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{Mixer, MixerState, attention::rope::PrecalculatedRoPE},
        ngpt_layer::{NgptLayer, NgptLayerError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    safetensors::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

enum TransformerLayerStateType<B: Backend> {
    Owned(Box<dyn MixerState<B>>),
    /// Reads the state of another slot.
    Shared(u32),
}

pub struct TransformerState<B: Backend> {
    layer_states: Box<[TransformerLayerStateType<B>]>,
    context_length: u32,
}

pub struct TransformerEncodeOutput<B: Backend> {
    pub output: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
}

impl<B: Backend> TransformerState<B> {
    pub fn context_length(&self) -> u32 {
        self.context_length
    }

    pub fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error> {
        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.prepare(context_length, suffix_length)?;
        }

        Ok(())
    }

    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        {
            let command_buffer = &mut *command_buffer.debug_group("transformer accept");

            for layer_state in &mut self.layer_states {
                let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                    continue;
                };

                layer_state.encode_accept(accepted_indices, command_buffer)?;
            }

            self.context_length += accepted_indices.len() as u32;
        }

        command_buffer.debug_point(&format!("accepted {} tokens", accepted_indices.len()));

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum TransformerNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("nGPT layer error: {0}")]
    NgptLayer(#[from] NgptLayerError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("A transformer must use either only nGPT layers or only standard layers")]
    MixedLayerKinds,
    #[error("Transformers with standard layers require an output normalization")]
    MissingOutputNorm,
    #[error("Invalid recurrent block [{start}, {end}) for {num_layers} layers with {num_loops} loops")]
    InvalidRecurrence {
        start: u32,
        end: u32,
        num_layers: u32,
        num_loops: u32,
    },
    #[error("Recurrence is supported only for nGPT layers")]
    RecurrenceWithStandardLayers,
}

#[derive(Debug, Error)]
pub enum TransformerEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("nGPT layer error: {0}")]
    NgptLayer(#[from] NgptLayerError<B>),
    #[error("Hidden feature capture is not supported for recurrent transformers")]
    HiddenFeaturesWithRecurrence,
}

pub enum Layer<B: Backend> {
    Standard(TransformerLayer<B>),
    Ngpt(NgptLayer<B>),
}

impl<B: Backend> Layer<B> {
    fn layer_index(&self) -> u32 {
        match self {
            Self::Standard(layer) => layer.layer_index,
            Self::Ngpt(layer) => layer.layer_index,
        }
    }

    fn mixer(&self) -> Option<&dyn Mixer<B>> {
        match self {
            Self::Standard(layer) => Some(layer.mixer.as_ref()),
            Self::Ngpt(layer) => layer.mixer(),
        }
    }

    fn kv_source_layer_index(&self) -> Option<u32> {
        match self {
            Self::Standard(layer) => layer.kv_source_layer_index,
            Self::Ngpt(_) => None,
        }
    }

    fn owns_state(&self) -> bool {
        self.mixer().is_some() && self.kv_source_layer_index().is_none()
    }
}

/// One execution of a physical layer; recurrent layers run once per loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExecutionStep {
    layer_index: u32,
    loop_index: u32,
}

/// The layers `[start, end)` run `num_loops` times. Each loop starts from `norm(state + inputs)`
/// (`state = inputs` for the first loop) and ends with `state = norm(outputs)`.
struct Recurrence<B: Backend> {
    start: u32,
    end: u32,
    num_loops: u32,
    kv_cache_mode: RecurrentKVCacheMode,
    reinjection_norm: Normalization<B>,
}

impl<B: Backend> Recurrence<B> {
    fn contains(
        &self,
        layer_index: u32,
    ) -> bool {
        (self.start..self.end).contains(&layer_index)
    }
}

pub struct Transformer<B: Backend> {
    ropes: Box<[AnyRoPEConfig]>,
    layers: Box<[(Layer<B>, Option<usize>)]>,
    recurrence: Option<Recurrence<B>>,
    steps: Box<[ExecutionStep]>,
    /// The state slot of every step, `None` for stateless layers.
    step_slots: Box<[Option<u32>]>,
    /// For every state slot, the layer whose mixer creates it and the slot it shares, if any.
    slot_layers: Box<[(u32, Option<u32>)]>,
    output_norm: Option<Normalization<B>>,
    is_ngpt: bool,
    model_dim: u32,
    data_type: DataType,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

fn build_layer<B: Backend>(
    transformer_config: &TransformerConfig,
    num_layers: u32,
    layer_index: u32,
    data_type: DataType,
    layer_config: &AnyLayerConfig,
    parameter_tree: &ParameterTree<B>,
    context: &B::Context,
) -> Result<Layer<B>, TransformerNewError<B>> {
    Ok(match layer_config {
        AnyLayerConfig::TransformerLayerConfig(config) => Layer::Standard(TransformerLayer::new(
            transformer_config.model_dim,
            transformer_config.hidden_dim,
            num_layers,
            layer_index,
            data_type,
            config,
            parameter_tree,
            context,
        )?),
        AnyLayerConfig::NGPTAttentionLayerConfig(config) => Layer::Ngpt(NgptLayer::new_attention(
            transformer_config.model_dim,
            layer_index,
            data_type,
            config,
            parameter_tree,
            context,
        )?),
        AnyLayerConfig::NGPTMLPLayerConfig(config) => Layer::Ngpt(NgptLayer::new_mlp(
            transformer_config.model_dim,
            transformer_config.hidden_dim,
            layer_index,
            data_type,
            config,
            parameter_tree,
            context,
        )?),
    })
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        data_type: DataType,
        output_norm_hadamard_factors: Option<B::GlobalBuffer>,
        transformer_config: &TransformerConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, TransformerNewError<B>> {
        let mut ropes: Vec<AnyRoPEConfig> = Vec::new();

        let num_layers = transformer_config.layer_configs.len() as u32;
        let layers = transformer_config
            .layer_configs
            .iter()
            .zip(0u32..)
            .map(|(layer_config, layer_index)| {
                let layer_loader = parameter_tree.subtree(&format!("layers.{}", layer_index));

                let rope = layer_config.rope_config().as_ref().map(|layer_rope_config| {
                    ropes.iter().position(|rope_config| rope_config == layer_rope_config).unwrap_or_else(|| {
                        ropes.push(layer_rope_config.clone());
                        ropes.len() - 1
                    })
                });

                let layer = build_layer(
                    transformer_config,
                    num_layers,
                    layer_index,
                    data_type,
                    layer_config,
                    &layer_loader,
                    context,
                )?;

                Ok((layer, rope))
            })
            .collect::<Result<Box<[_]>, TransformerNewError<B>>>()?;

        let ngpt_layer_count = layers.iter().filter(|(layer, _)| matches!(layer, Layer::Ngpt(_))).count();
        if ngpt_layer_count != 0 && ngpt_layer_count != layers.len() {
            return Err(TransformerNewError::MixedLayerKinds);
        }
        let is_ngpt = ngpt_layer_count != 0;

        let recurrence = transformer_config
            .recurrence
            .as_ref()
            .map(|recurrence_config| -> Result<_, TransformerNewError<B>> {
                let (start, end, num_loops) = (
                    recurrence_config.start_layer_index,
                    recurrence_config.end_layer_index,
                    recurrence_config.num_loops,
                );
                if start >= end || end > num_layers || num_loops == 0 {
                    return Err(TransformerNewError::InvalidRecurrence {
                        start,
                        end,
                        num_layers,
                        num_loops,
                    });
                }
                if !is_ngpt {
                    return Err(TransformerNewError::RecurrenceWithStandardLayers);
                }
                Ok(Recurrence {
                    start,
                    end,
                    num_loops,
                    kv_cache_mode: recurrence_config.kv_cache_mode.clone(),
                    reinjection_norm: Normalization::new(
                        transformer_config.model_dim,
                        data_type,
                        None,
                        ShortcutMode::None,
                        PostLayerScalar::None,
                        &recurrence_config.reinjection_norm_config,
                        &parameter_tree.subtree("reinjection_norm"),
                        context,
                    )?,
                })
            })
            .transpose()?;

        let steps =
            match &recurrence {
                None => (0..num_layers)
                    .map(|layer_index| ExecutionStep {
                        layer_index,
                        loop_index: 0,
                    })
                    .collect::<Box<[_]>>(),
                Some(recurrence) => (0..recurrence.start)
                    .map(|layer_index| (layer_index, 0))
                    .chain((0..recurrence.num_loops).flat_map(|loop_index| {
                        (recurrence.start..recurrence.end).map(move |layer| (layer, loop_index))
                    }))
                    .chain((recurrence.end..num_layers).map(|layer_index| (layer_index, 0)))
                    .map(|(layer_index, loop_index)| ExecutionStep {
                        layer_index,
                        loop_index,
                    })
                    .collect(),
            };

        // Standard layers keep one slot per layer (shared layers point at their KV source).
        // nGPT layers get one slot per stateful layer, and one per loop under per-loop KV caching.
        let mut slot_keys: Vec<(u32, u32)> = Vec::new();
        let mut slot_layers: Vec<(u32, Option<u32>)> = Vec::new();
        let step_slots = steps
            .iter()
            .map(|step| {
                let (layer, _) = &layers[step.layer_index as usize];
                let is_standard = matches!(layer, Layer::Standard(_));
                if !is_standard && !layer.owns_state() {
                    return None;
                }
                let has_loop_slot = recurrence.as_ref().is_some_and(|recurrence| {
                    recurrence.kv_cache_mode == RecurrentKVCacheMode::PerLoop && recurrence.contains(step.layer_index)
                });
                let key = (
                    step.layer_index,
                    if has_loop_slot {
                        step.loop_index
                    } else {
                        0
                    },
                );
                let slot = slot_keys.iter().position(|existing| *existing == key).unwrap_or_else(|| {
                    slot_keys.push(key);
                    let shared_slot = layer.kv_source_layer_index().map(|source_layer| {
                        slot_keys
                            .iter()
                            .position(|existing| *existing == (source_layer, 0))
                            .expect("a shared layer must come after its KV source layer") as u32
                    });
                    slot_layers.push((step.layer_index, shared_slot));
                    slot_keys.len() - 1
                });
                Some(slot as u32)
            })
            .collect::<Box<[_]>>();

        let output_norm = transformer_config
            .output_norm_config
            .as_ref()
            .map(|output_norm_config| {
                Normalization::new(
                    transformer_config.model_dim,
                    data_type,
                    output_norm_hadamard_factors,
                    if is_ngpt {
                        ShortcutMode::None
                    } else {
                        ShortcutMode::Add
                    },
                    PostLayerScalar::None,
                    output_norm_config,
                    &parameter_tree.subtree("output_norm"),
                    context,
                )
            })
            .transpose()?;
        if output_norm.is_none() && !is_ngpt {
            return Err(TransformerNewError::MissingOutputNorm);
        }

        let residual_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)
            .map_err(TransformerNewError::Backend)?;

        Ok(Self {
            ropes: ropes.into_boxed_slice(),
            layers,
            recurrence,
            steps,
            step_slots,
            slot_layers: slot_layers.into_boxed_slice(),
            output_norm,
            is_ngpt,
            model_dim: transformer_config.model_dim,
            data_type,
            residual_add,
        })
    }

    fn add_residual(
        &self,
        shortcut: impl BufferRef<Backend = B>,
        hidden: impl BufferRef<Backend = B>,
        batch_size: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = command_buffer.allocate_scratch(hidden.size())?;
        let elements = batch_size * self.model_dim;
        self.residual_add.encode(Some(shortcut), hidden, &mut output, elements, elements, 1.0, command_buffer);
        Ok(output)
    }

    /// The full residual stream after the last executed step.
    fn residual_stream(
        &self,
        shortcut: impl BufferRef<Backend = B>,
        hidden: B::ScratchBuffer,
        batch_size: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        if self.is_ngpt {
            let mut copy = command_buffer.allocate_scratch(hidden.size())?;
            command_buffer.encode_copy(&hidden, &mut copy);
            Ok(copy)
        } else {
            self.add_residual(shortcut, &hidden, batch_size, command_buffer)
        }
    }

    /// Whether the step after `step_index` is an nGPT layer that reads this step's output directly
    /// (not through the recurrent block's reinjection or loop-end normalization).
    fn next_step_consumes_copy(
        &self,
        step_index: usize,
        step_count: usize,
    ) -> bool {
        if step_index + 1 >= step_count {
            return false;
        }
        let (step, next_step) = (self.steps[step_index], self.steps[step_index + 1]);
        let crosses_recurrence_boundary = self.recurrence.as_ref().is_some_and(|recurrence| {
            step.layer_index == recurrence.end - 1 || next_step.layer_index == recurrence.start
        });
        !crosses_recurrence_boundary && matches!(self.layers[next_step.layer_index as usize].0, Layer::Ngpt(_))
    }

    pub fn speculation_supported(&self) -> bool {
        self.layers.iter().filter_map(|(layer, _rope)| layer.mixer()).all(|mixer| mixer.speculation_supported())
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.layers
            .iter()
            .filter_map(|(layer, _rope_index)| layer.mixer())
            .map(|mixer| mixer.max_context_length())
            .fold(None, |acc, el| match (acc, el) {
                (Some(a), Some(b)) => Some(u32::min(a, b)),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            })
    }

    /// The number of execution steps a cache-only prefill runs: steps after the last one that
    /// writes its own state only affect outputs.
    pub fn prefill_cache_layer_count(&self) -> usize {
        let Some(last_owned_state_step) =
            self.steps.iter().rposition(|step| self.layers[step.layer_index as usize].0.owns_state())
        else {
            return self.steps.len();
        };

        last_owned_state_step + 1
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.prefill_cache_layer_count() < self.steps.len()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let layer_states = self
            .slot_layers
            .iter()
            .map(|&(layer_index, shared_slot)| match shared_slot {
                None => self.layers[layer_index as usize]
                    .0
                    .mixer()
                    .expect("state slots belong to layers with a mixer")
                    .create_empty_state(max_context_length, context)
                    .map(TransformerLayerStateType::Owned),
                Some(shared_slot) => Ok(TransformerLayerStateType::Shared(shared_slot)),
            })
            .collect::<Result<_, B::Error>>()?;

        Ok(TransformerState {
            layer_states,
            context_length: 0,
        })
    }

    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        mut state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        if self.recurrence.is_some() && hidden_feature_layer_indices.is_some() {
            return Err(TransformerEncodeError::HiddenFeaturesWithRecurrence);
        }

        let mut hidden = input;
        let step_count = if output_range.is_none() && hidden_feature_layer_indices.is_none() {
            self.prefill_cache_layer_count()
        } else {
            self.steps.len()
        };

        let mut shortcut = command_buffer.allocate_scratch(hidden.size()).map_err(TransformerEncodeError::Backend)?;
        let mut hidden_features =
            hidden_feature_layer_indices.map(|indices| (0..indices.len()).map(|_| None).collect::<Vec<_>>());

        let context_length = state.as_ref().map(|state| state.context_length).unwrap_or(0);
        let token_positions = batch_dim.heights().map(|rel_pos| context_length + rel_pos).collect::<Box<[u32]>>();

        let precalculated_ropes = self
            .ropes
            .iter()
            .map(|rope_config| PrecalculatedRoPE::precalculate(rope_config, &token_positions, command_buffer))
            .collect::<Result<Box<[_]>, B::Error>>()
            .map_err(TransformerEncodeError::Backend)?;

        // The recurrent block's inputs and the state carried between its loops.
        let mut recurrent_inputs: Option<B::ScratchBuffer> = None;
        let mut loop_state: Option<B::ScratchBuffer> = None;
        // A second copy of `hidden` written by the previous nGPT layer for the next one to consume.
        let mut hidden_copy: Option<B::ScratchBuffer> = None;

        for (step_index, (step, step_slot)) in self.steps.iter().zip(&self.step_slots).take(step_count).enumerate() {
            let (layer, layer_rope_index) = &self.layers[step.layer_index as usize];
            let precalculated_rope = layer_rope_index.map(|i| &precalculated_ropes[i]);

            if let Some(recurrence) = &self.recurrence
                && step.layer_index == recurrence.start
            {
                if step.loop_index == 0 {
                    let mut inputs =
                        command_buffer.allocate_scratch(hidden.size()).map_err(TransformerEncodeError::Backend)?;
                    command_buffer.encode_copy(&hidden, &mut inputs);
                    recurrent_inputs = Some(inputs);
                    loop_state = Some(hidden);
                }
                let recurrent_inputs = recurrent_inputs.as_ref().expect("recurrent inputs are set by the first loop");
                let loop_state = loop_state.take().expect("each loop starts from the previous loop state");
                let combined = self
                    .add_residual(&loop_state, recurrent_inputs, batch_dim.size(), command_buffer)
                    .map_err(TransformerEncodeError::Backend)?;
                hidden = recurrence
                    .reinjection_norm
                    .encode(&combined, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                    .map_err(TransformerEncodeError::Backend)?;
            }

            let layer_state = match (&mut state, step_slot) {
                (Some(state), Some(slot)) => Some(match &mut state.layer_states[*slot as usize] {
                    TransformerLayerStateType::Owned(layer_state) => MaybeMut::Mut(layer_state.as_mut()),
                    TransformerLayerStateType::Shared(owner_slot) => {
                        let TransformerLayerStateType::Owned(owned_layer) = &state.layer_states[*owner_slot as usize]
                        else {
                            panic!("shared layer doesn't point to an owned layer");
                        };
                        MaybeMut::Const(owned_layer.as_ref())
                    },
                }),
                _ => None,
            };

            hidden = match layer {
                Layer::Standard(layer) => layer.encode(
                    hidden,
                    &mut shortcut,
                    per_layer_inputs,
                    precalculated_rope,
                    batch_dim,
                    layer_state,
                    command_buffer,
                )?,
                Layer::Ngpt(layer) => {
                    let (output, output_copy) = layer.encode(
                        hidden,
                        hidden_copy.take(),
                        self.next_step_consumes_copy(step_index, step_count),
                        precalculated_rope,
                        batch_dim,
                        layer_state,
                        command_buffer,
                    )?;
                    hidden_copy = output_copy;
                    output
                },
            };

            if let Some(recurrence) = &self.recurrence
                && step.layer_index == recurrence.end - 1
            {
                let state = recurrence
                    .reinjection_norm
                    .encode(&hidden, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                    .map_err(TransformerEncodeError::Backend)?;
                if step.loop_index == recurrence.num_loops - 1 {
                    hidden = state;
                } else {
                    loop_state = Some(state);
                }
            }

            if let (Some(hidden_features), Some(indices)) = (&mut hidden_features, hidden_feature_layer_indices) {
                for (feature_index, &layer_index) in indices.iter().enumerate() {
                    if layer_index == layer.layer_index() {
                        let mut feature =
                            command_buffer.allocate_scratch(hidden.size()).map_err(TransformerEncodeError::Backend)?;
                        command_buffer.encode_copy(&hidden, &mut feature);
                        let feature = self
                            .residual_stream(&shortcut, feature, batch_dim.size(), command_buffer)
                            .map_err(TransformerEncodeError::Backend)?;
                        hidden_features[feature_index] = Some(feature);
                    }
                }
            }
        }

        let hidden_features = hidden_features.map(|hidden_features| {
            hidden_features
                .into_iter()
                .enumerate()
                .map(|(feature_index, feature)| {
                    feature.unwrap_or_else(|| {
                        panic!(
                            "requested hidden feature for missing layer {}",
                            hidden_feature_layer_indices.unwrap()[feature_index]
                        )
                    })
                })
                .collect::<Box<[_]>>()
        });

        let Some(output_range) = output_range else {
            return Ok(TransformerEncodeOutput {
                output: None,
                hidden_features,
            });
        };
        let output_rows = output_range.end - output_range.start;

        let output = match &self.output_norm {
            Some(output_norm) => output_norm
                .encode(
                    &hidden,
                    output_range.start,
                    output_rows,
                    (!self.is_ngpt).then_some(&mut shortcut),
                    command_buffer,
                )
                .map_err(TransformerEncodeError::Backend)?,
            None => {
                let row_bytes = self.model_dim as usize * self.data_type.size_in_bytes();
                let mut output = command_buffer
                    .allocate_scratch_for_shape(&[output_rows, self.model_dim], self.data_type)
                    .map_err(TransformerEncodeError::Backend)?;
                command_buffer.encode_copy(
                    hidden.subrange(output_range.start as usize * row_bytes..output_range.end as usize * row_bytes),
                    &mut output,
                );
                output
            },
        };

        Ok(TransformerEncodeOutput {
            output: Some(output),
            hidden_features,
        })
    }
}
