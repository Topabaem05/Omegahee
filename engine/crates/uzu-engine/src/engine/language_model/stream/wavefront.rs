//! Greedy loop-wavefront decoding in the token stream (see `encodable_block::transformer::wavefront`).
//!
//! While the GPU runs a step, the CPU encodes the next step as if the running step's guess were
//! right: its entering token is the running step's early-exit draft, copied on the GPU. When the
//! running step finishes, the next step is submitted if the guess matched the final token. Otherwise
//! it is dropped unsubmitted, the KV bookkeeping taken before encoding it is restored, and a step
//! that enters the final token is submitted instead.
//!
//! Every returned token is followed by a submitted step that commits that token's KV, as the
//! sequential stream's in-flight pass does, so closing the stream leaves the same state.

use std::{mem::replace, sync::Arc};

use super::{DecodingState, LanguageModelStream};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
        Context, gpu_types::trie::TrieNode,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        sampling::SamplingMethod,
        transformer::{WavefrontCarry, WavefrontSlots},
    },
    engine::language_model::{LanguageModel, stream::LanguageModelStreamError},
};

const ONE_ROW: [TrieNode; 1] = [TrieNode {
    trie_start: 0,
    trie_end: 1,
    height: 0,
}];
const TWO_ROWS: [TrieNode; 2] = [
    TrieNode {
        trie_start: 0,
        trie_end: 2,
        height: 0,
    },
    TrieNode {
        trie_start: 1,
        trie_end: 2,
        height: 1,
    },
];

/// The token that enters a step.
enum EnteringToken<B: Backend> {
    Known(u64),
    /// The last row of a previous step's sampled tokens, read on the GPU.
    Draft(Arc<B::GlobalBuffer>, u32),
}

struct WavefrontStep<B: Backend> {
    output_tokens: Arc<B::GlobalBuffer>,
    has_carried_row: bool,
    /// Position of the entering row.
    position: u32,
    /// The entering token, once the CPU knows it.
    entering: Option<u64>,
    _entering_source: Option<Arc<B::GlobalBuffer>>,
}

pub(super) struct WavefrontDecoding<B: Backend> {
    carry: WavefrontCarry<B>,
    slots: WavefrontSlots,
    in_flight: Option<(WavefrontStep<B>, <B::CommandBuffer as CommandBuffer>::Pending)>,
}

impl<B: Backend> WavefrontDecoding<B> {
    /// Wavefront decoding is used when `OMEGA_WAVEFRONT` is set and nothing else needs the
    /// sequential path: greedy sampling, no grammar, no speculator.
    pub(super) fn create(
        model: &LanguageModel<B>,
        sampling_method: &SamplingMethod,
        has_grammar: bool,
    ) -> Result<Option<Self>, LanguageModelStreamError<B>> {
        let enabled = std::env::var_os("OMEGA_WAVEFRONT").is_some_and(|value| value != "0")
            && model.decoder.wavefront_supported()
            && model.speculator.is_none()
            && matches!(sampling_method, SamplingMethod::Greedy)
            && !has_grammar;
        if !enabled {
            return Ok(None);
        }
        Ok(Some(Self {
            carry: model
                .decoder
                .create_wavefront_carry(&model.engine.context)
                .map_err(LanguageModelStreamError::Backend)?,
            slots: model.decoder.wavefront_slots(),
            in_flight: None,
        }))
    }
}

fn read_tokens<B: Backend>(buffer: &B::GlobalBuffer) -> Vec<u64> {
    buffer.as_slice::<u32>().iter().map(|&token| token as u64).collect()
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub(super) async fn generate_wavefront(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        match replace(&mut self.decoding_state, DecodingState::Invalid) {
            DecodingState::Seeded {
                seed_token,
            } => {
                self.start_wavefront(seed_token)?;
                Ok(Some(seed_token))
            },
            DecodingState::ForwardPassPending(prefill) => {
                assert!(prefill.full_accept, "wavefront starts after a prefill");
                for pending in prefill.pending {
                    pending.wait_until_completed().await.map_err(LanguageModelStreamError::Backend)?;
                }
                let token = read_tokens::<B>(&prefill.output_tokens)[0];
                self.start_wavefront(token)?;
                Ok(Some(token))
            },
            DecodingState::Wavefront => {
                let token = self.advance_wavefront().await?;
                self.decoding_state = if token.is_some() {
                    DecodingState::Wavefront
                } else {
                    DecodingState::Halted
                };
                Ok(token)
            },
            state @ (DecodingState::Halted | DecodingState::Closed) => {
                self.decoding_state = state;
                Ok(None)
            },
            DecodingState::Accepting {
                ..
            }
            | DecodingState::Invalid => unreachable!("not used by wavefront decoding"),
        }
    }

    /// Records `token` (already sampled) as returned and submits the step that enters it.
    fn start_wavefront(
        &mut self,
        token: u64,
    ) -> Result<(), LanguageModelStreamError<B>> {
        self.model_state.tokens.push(token);
        self.metrics.num_tokens_returned += 1;
        let position = self.model_state.tokens.len() as u32 - 1;
        self.model_state.transformer_state.set_context_length(position);
        let (step, executable) = self.encode_wavefront_step(EnteringToken::Known(token), position, false, false)?;
        self.wavefront.as_mut().unwrap().in_flight = Some((step, executable.submit()));
        self.decoding_state = DecodingState::Wavefront;
        Ok(())
    }

    /// Runs steps until one returns a final token (`None` at the context limit).
    async fn advance_wavefront(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        loop {
            let (step, pending) = self.wavefront.as_mut().unwrap().in_flight.take().expect("a step is in flight");
            let next_position = step.position + 1;
            let can_continue = self.model_state.max_context_length.is_none_or(|max| next_position + 1 < max);

            // Encode the follow-up while the step runs, as if the step's guess were right.
            let bookkeeping = self.model_state.transformer_state.bookkeeping();
            let rows = 1 + step.has_carried_row as u32;
            let next = can_continue
                .then(|| {
                    self.encode_wavefront_step(
                        EnteringToken::Draft(step.output_tokens.clone(), rows - 1),
                        next_position,
                        true,
                        step.has_carried_row,
                    )
                })
                .transpose()?;

            pending.wait_until_completed().await.map_err(LanguageModelStreamError::Backend)?;
            let outputs = read_tokens::<B>(&step.output_tokens);
            let draft = *outputs.last().unwrap();

            if !step.has_carried_row {
                // The step entered a known token; its follow-up is always valid.
                let Some((mut next_step, executable)) = next else {
                    return Ok(None);
                };
                next_step.entering = Some(draft);
                self.wavefront.as_mut().unwrap().in_flight = Some((next_step, executable.submit()));
                continue;
            }

            let final_token = outputs[0];
            self.model_state.tokens.push(final_token);
            self.metrics.num_tokens_returned += 1;
            self.metrics.num_tokens_proposed += 1;
            if step.entering == Some(final_token) {
                self.metrics.num_tokens_accepted += 1;
                if let Some((mut next_step, executable)) = next {
                    next_step.entering = Some(draft);
                    self.wavefront.as_mut().unwrap().in_flight = Some((next_step, executable.submit()));
                }
            } else {
                // The entering row ran the wrong token: drop the follow-up and enter the final token.
                if next.is_some() {
                    self.metrics.num_decode_forward_passes -= 1;
                }
                drop(next);
                self.model_state.transformer_state.restore_bookkeeping(&bookkeeping);
                let (enter_step, executable) =
                    self.encode_wavefront_step(EnteringToken::Known(final_token), step.position, false, false)?;
                self.wavefront.as_mut().unwrap().in_flight = Some((enter_step, executable.submit()));
            }
            self.model_state.transformer_state.set_context_length(self.model_state.tokens.len() as u32 - 1);
            return Ok(Some(final_token));
        }
    }

    /// Encodes one step with its KV accepts; the caller submits or drops it.
    fn encode_wavefront_step(
        &mut self,
        entering: EnteringToken<B>,
        position: u32,
        has_carried_row: bool,
        accept_previous_new_row: bool,
    ) -> Result<(WavefrontStep<B>, <B::CommandBuffer as CommandBuffer>::Executable), LanguageModelStreamError<B>> {
        let model = self.model;
        let wavefront = self.wavefront.as_mut().expect("wavefront state");
        let transformer_state = &mut self.model_state.transformer_state;
        transformer_state.prepare(position + 1, 2).map_err(LanguageModelStreamError::Backend)?;
        let mut command_buffer = model
            .engine
            .context
            .create_command_buffer(
                Some(&format!("decode {} (wavefront)", self.metrics.num_decode_forward_passes)),
                Some(self.allocation_pool.clone()),
            )
            .map_err(LanguageModelStreamError::Backend)?;
        if accept_previous_new_row {
            transformer_state
                .encode_accept_slots(&wavefront.slots.new_row, &[0], &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;
        }
        let token_bytes = DataType::U32.size_in_bytes();
        let (token_ids, known, entering_source) = match entering {
            EnteringToken::Known(token) => (
                command_buffer
                    .allocate_constant_from_slice(&[token as u32])
                    .map_err(LanguageModelStreamError::Backend)?,
                Some(token),
                None,
            ),
            EnteringToken::Draft(source, row) => {
                let mut token_ids =
                    command_buffer.allocate_constant(token_bytes).map_err(LanguageModelStreamError::Backend)?;
                let offset = row as usize * token_bytes;
                command_buffer.encode_copy(source.as_ref().subrange(offset..offset + token_bytes), &mut token_ids);
                (token_ids, None, Some(source))
            },
        };
        let logits = model.decoder.encode_wavefront(
            &token_ids,
            &mut wavefront.carry,
            position,
            has_carried_row,
            transformer_state,
            &mut command_buffer,
        )?;
        let rows = 1 + has_carried_row as u32;
        let batch_dim = if has_carried_row {
            BatchTopology::new(&TWO_ROWS, true)
        } else {
            BatchTopology::new(&ONE_ROW, true)
        };
        let output_tokens = model
            .sampling
            .encode(
                &logits,
                None::<&B::ConstantBuffer>,
                None::<&B::ConstantBuffer>,
                None::<&B::GlobalBuffer>,
                None::<&B::ConstantBuffer>,
                &batch_dim,
                (0..rows).into(),
                &self.options.sampling_method,
                &mut command_buffer,
            )
            .map_err(LanguageModelStreamError::Backend)?;
        drop(logits);
        if has_carried_row {
            // The carried row's token is final: commit its second-loop and output-layer KV.
            transformer_state
                .encode_accept_slots(&wavefront.slots.carried_row, &[0], &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;
            transformer_state
                .encode_accept_slots(&wavefront.slots.output, &[0], &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;
        } else {
            // The entering token is known: commit its first-loop KV.
            transformer_state
                .encode_accept_slots(&wavefront.slots.new_row, &[0], &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;
        }
        drop(token_ids);
        self.metrics.num_decode_forward_passes += 1;
        Ok((
            WavefrontStep {
                output_tokens: Arc::new(output_tokens),
                has_carried_row,
                position,
                entering: known,
                _entering_source: entering_source,
            },
            command_buffer.end_encoding(),
        ))
    }

    /// Leaves the state as the sequential stream does: every returned token committed, and the token
    /// sampled after the last one as the next stream's seed.
    pub(super) async fn close_wavefront(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        let Some((step, pending)) = self.wavefront.as_mut().and_then(|wavefront| wavefront.in_flight.take()) else {
            return Ok(None);
        };
        pending.wait_until_completed().await.map_err(LanguageModelStreamError::Backend)?;
        let outputs = read_tokens::<B>(&step.output_tokens);
        let next_token = if step.has_carried_row {
            // The step committed the last returned token and sampled the token after it.
            outputs[0]
        } else {
            // The step entered the last returned token: finish it with one more step.
            let (step, executable) =
                self.encode_wavefront_step(EnteringToken::Known(outputs[0]), step.position + 1, true, false)?;
            executable.submit().wait_until_completed().await.map_err(LanguageModelStreamError::Backend)?;
            read_tokens::<B>(&step.output_tokens)[0]
        };
        self.model_state.transformer_state.set_context_length(self.model_state.tokens.len() as u32);
        Ok(Some(next_token))
    }
}
