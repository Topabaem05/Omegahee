//! Greedy loop-wavefront decoding in the token stream (see `encodable_block::transformer::wavefront`).
//!
//! Each step syncs with the CPU: whether the next step has a carried row depends on whether the
//! early-exit guess matched the final token. At every returned token the committed KV equals what
//! sequential decoding would have committed (all tokens before the returned one), so closing the
//! stream runs one ordinary decode pass for the returned token, like the sequential stream's
//! in-flight pass.

use std::mem::replace;

use super::{DecodingState, LanguageModelStream};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
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

enum WavefrontNext {
    /// A known token enters without a carried row (after the prefill or a wrong guess).
    Enter(u64),
    /// The last returned token is carried; the guess for the token after it enters.
    Guess(u64),
}

pub(super) struct WavefrontDecoding<B: Backend> {
    carry: WavefrontCarry<B>,
    slots: WavefrontSlots,
    next: WavefrontNext,
    /// The previous step's guess was right: accept its first-loop KV before the next step.
    accept_new_row: bool,
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
            next: WavefrontNext::Enter(0),
            accept_new_row: false,
        }))
    }
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub(super) async fn generate_wavefront(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        loop {
            match replace(&mut self.decoding_state, DecodingState::Invalid) {
                DecodingState::Seeded {
                    seed_token,
                } => {
                    self.model_state.tokens.push(seed_token);
                    self.metrics.num_tokens_returned += 1;
                    self.start_wavefront(seed_token);
                    return Ok(Some(seed_token));
                },
                DecodingState::ForwardPassPending(prefill) => {
                    assert!(prefill.full_accept, "wavefront starts after a prefill");
                    for pending in prefill.pending {
                        pending.wait_until_completed().await.map_err(LanguageModelStreamError::Backend)?;
                    }
                    let token = prefill.output_tokens.as_ref().as_slice::<u32>()[0] as u64;
                    self.model_state.tokens.push(token);
                    self.metrics.num_tokens_returned += 1;
                    self.start_wavefront(token);
                    return Ok(Some(token));
                },
                DecodingState::Wavefront => {
                    // The last returned token sits at `position`; its KV is not committed yet.
                    let position = self.model_state.tokens.len() as u32 - 1;
                    if self.model_state.max_context_length.is_some_and(|max| position + 1 >= max) {
                        self.decoding_state = DecodingState::Wavefront;
                        return Ok(None);
                    }
                    let wavefront = self.wavefront.as_mut().expect("wavefront state");
                    match wavefront.next {
                        WavefrontNext::Enter(token) => {
                            let outputs = self.wavefront_step(token, position, false).await?;
                            self.wavefront.as_mut().unwrap().next = WavefrontNext::Guess(outputs[0]);
                            self.decoding_state = DecodingState::Wavefront;
                        },
                        WavefrontNext::Guess(guess) => {
                            let outputs = self.wavefront_step(guess, position + 1, true).await?;
                            let (final_token, draft) = (outputs[0], outputs[1]);
                            self.model_state.tokens.push(final_token);
                            self.model_state
                                .transformer_state
                                .set_context_length(self.model_state.tokens.len() as u32 - 1);
                            self.metrics.num_tokens_returned += 1;
                            self.metrics.num_tokens_proposed += 1;
                            let wavefront = self.wavefront.as_mut().unwrap();
                            if guess == final_token {
                                self.metrics.num_tokens_accepted += 1;
                                wavefront.accept_new_row = true;
                                wavefront.next = WavefrontNext::Guess(draft);
                            } else {
                                wavefront.next = WavefrontNext::Enter(final_token);
                            }
                            self.decoding_state = DecodingState::Wavefront;
                            return Ok(Some(final_token));
                        },
                    }
                },
                state @ (DecodingState::Halted | DecodingState::Closed) => {
                    self.decoding_state = state;
                    return Ok(None);
                },
                DecodingState::Accepting {
                    ..
                }
                | DecodingState::Invalid => unreachable!("not used by wavefront decoding"),
            }
        }
    }

    fn start_wavefront(
        &mut self,
        token: u64,
    ) {
        let wavefront = self.wavefront.as_mut().expect("wavefront state");
        wavefront.next = WavefrontNext::Enter(token);
        wavefront.accept_new_row = false;
        self.model_state.transformer_state.set_context_length(self.model_state.tokens.len() as u32 - 1);
        self.decoding_state = DecodingState::Wavefront;
    }

    /// Runs one wavefront step with its KV accepts and returns the greedy token of every output row.
    async fn wavefront_step(
        &mut self,
        token: u64,
        position: u32,
        has_carried_row: bool,
    ) -> Result<Vec<u64>, LanguageModelStreamError<B>> {
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
        if replace(&mut wavefront.accept_new_row, false) {
            transformer_state
                .encode_accept_slots(&wavefront.slots.new_row, &[0], &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;
        }
        let token_ids =
            command_buffer.allocate_constant_from_slice(&[token as u32]).map_err(LanguageModelStreamError::Backend)?;
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
        command_buffer
            .end_encoding()
            .submit()
            .wait_until_completed()
            .await
            .map_err(LanguageModelStreamError::Backend)?;
        self.metrics.num_decode_forward_passes += 1;
        Ok(output_tokens.as_slice::<u32>().iter().map(|&token| token as u64).collect())
    }

    /// Commits the last returned token's KV with one sequential pass and returns the token sampled
    /// after it, which the next stream continues from.
    pub(super) async fn close_wavefront(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        let model = self.model;
        let Some(&token) = self.model_state.tokens.last() else {
            return Ok(None);
        };
        let transformer_state = &mut self.model_state.transformer_state;
        let position = self.model_state.tokens.len() as u32 - 1;
        transformer_state.set_context_length(position);
        transformer_state.prepare(position, 1).map_err(LanguageModelStreamError::Backend)?;
        let mut command_buffer = model
            .engine
            .context
            .create_command_buffer(Some("decode (wavefront close)"), Some(self.allocation_pool.clone()))
            .map_err(LanguageModelStreamError::Backend)?;
        let token_ids =
            command_buffer.allocate_constant_from_slice(&[token as u32]).map_err(LanguageModelStreamError::Backend)?;
        let batch_dim = BatchTopology::new(&ONE_ROW, true);
        let output = model.decoder.encode(
            &token_ids,
            &batch_dim,
            Some((0..1).into()),
            None,
            transformer_state,
            &mut command_buffer,
        )?;
        let logits = output.logits.expect("decoder logits");
        let output_tokens = model
            .sampling
            .encode(
                &logits,
                None::<&B::ConstantBuffer>,
                None::<&B::ConstantBuffer>,
                None::<&B::GlobalBuffer>,
                None::<&B::ConstantBuffer>,
                &batch_dim,
                (0..1).into(),
                &self.options.sampling_method,
                &mut command_buffer,
            )
            .map_err(LanguageModelStreamError::Backend)?;
        drop((logits, output.hidden_features, output.final_hidden));
        transformer_state.encode_accept(&[0], &mut command_buffer).map_err(LanguageModelStreamError::Backend)?;
        drop(token_ids);
        command_buffer
            .end_encoding()
            .submit()
            .wait_until_completed()
            .await
            .map_err(LanguageModelStreamError::Backend)?;
        Ok(Some(output_tokens.as_slice::<u32>()[0] as u64))
    }
}
