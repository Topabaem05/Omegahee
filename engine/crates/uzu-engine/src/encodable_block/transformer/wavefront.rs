//! Loop wavefront decoding for a recurrent block with two loops and per-loop KV caches.
//!
//! A step runs two tokens at different depths through one pass over the physical layers: the
//! carried row (position `p - 1`) runs the second loop and the output layers, and the new row
//! (position `p`) runs the input layers, the first loop and the output layers as an early exit. The
//! recurrent weights are read once for both rows. Each row reads and appends the KV state of its
//! own loop, so a step computes what two sequential decode passes would compute when the new row's
//! token is right; the caller checks that against the carried row's final token and accepts KV.
//!
//! KV is left in each state's suffix. `WavefrontSlots` names which states each row writes:
//! - `new_row`: input layers and the first loop, suffix index 0 (accept once the token is known)
//! - `carried_row`: the second loop, suffix index 0
//! - `output`: output layers, suffix index 0 for the carried row and 1 for the new row (the new
//!   row's early-exit entry is never accepted; its final pass rewrites it)

use super::{Layer, Transformer, TransformerEncodeError, TransformerLayerStateType, TransformerState};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, gpu_types::trie::TrieNode,
    },
    config::transformer::RecurrentKVCacheMode,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{
            MixerState,
            attention::{AttentionRowGroup, AttentionState, KVCacheBookkeeping, rope::PrecalculatedRoPE},
        },
    },
    utils::maybe_mut::MaybeMut,
};

/// The recurrent-block state of a row between its first and second loop.
pub struct WavefrontCarry<B: Backend> {
    loop_state: B::GlobalBuffer,
    recurrent_inputs: B::GlobalBuffer,
}

/// KV bookkeeping of every attention slot and the context length (see `TransformerState::bookkeeping`).
#[derive(Debug, Clone)]
pub struct TransformerBookkeeping {
    slots: Box<[Option<KVCacheBookkeeping>]>,
    context_length: u32,
}

/// State slots written by each row of a wavefront step.
#[derive(Debug, Clone)]
pub struct WavefrontSlots {
    pub new_row: Box<[u32]>,
    pub carried_row: Box<[u32]>,
    pub output: Box<[u32]>,
}

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

fn layer_rope<B: Backend>(
    ropes: &[PrecalculatedRoPE<B>],
    rope_index: Option<usize>,
) -> Option<&PrecalculatedRoPE<B>> {
    rope_index.map(|index| &ropes[index])
}

fn attention_state<B: Backend>(layer_state: &mut TransformerLayerStateType<B>) -> &mut AttentionState<B> {
    let TransformerLayerStateType::Owned(layer_state) = layer_state else {
        panic!("wavefront does not support shared KV slots");
    };
    (layer_state.as_mut() as &mut dyn std::any::Any)
        .downcast_mut::<AttentionState<B>>()
        .expect("wavefront recurrent mixers must be attention")
}

impl<B: Backend> TransformerState<B> {
    /// Accepts the same suffix indices in the given slots only.
    pub fn encode_accept_slots(
        &mut self,
        slots: &[u32],
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        for &slot in slots {
            let TransformerLayerStateType::Owned(layer_state) = &mut self.layer_states[slot as usize] else {
                continue;
            };
            layer_state.encode_accept(accepted_indices, command_buffer)?;
        }
        Ok(())
    }

    /// The bookkeeping of every attention slot, to undo the accepts of a command buffer that is
    /// encoded but never submitted (a speculatively encoded wavefront step).
    pub fn bookkeeping(&self) -> TransformerBookkeeping {
        TransformerBookkeeping {
            slots: self
                .layer_states
                .iter()
                .map(|layer_state| match layer_state {
                    TransformerLayerStateType::Owned(layer_state) => (layer_state.as_ref() as &dyn std::any::Any)
                        .downcast_ref::<AttentionState<B>>()
                        .map(AttentionState::cache_bookkeeping),
                    TransformerLayerStateType::Shared(_) => None,
                })
                .collect(),
            context_length: self.context_length,
        }
    }

    pub fn restore_bookkeeping(
        &mut self,
        bookkeeping: &TransformerBookkeeping,
    ) {
        for (layer_state, slot) in self.layer_states.iter_mut().zip(&bookkeeping.slots) {
            if let (TransformerLayerStateType::Owned(layer_state), Some(slot)) = (layer_state, slot) {
                (layer_state.as_mut() as &mut dyn std::any::Any)
                    .downcast_mut::<AttentionState<B>>()
                    .expect("bookkeeping belongs to an attention state")
                    .restore_cache_bookkeeping(*slot);
            }
        }
        self.context_length = bookkeeping.context_length;
    }

    /// Sets the context length after accepting slots directly.
    pub fn set_context_length(
        &mut self,
        context_length: u32,
    ) {
        self.context_length = context_length;
    }
}

impl<B: Backend> Transformer<B> {
    fn wavefront_recurrence(&self) -> Option<(u32, u32)> {
        let recurrence = self.recurrence.as_ref()?;
        (recurrence.num_loops == 2 && recurrence.kv_cache_mode == RecurrentKVCacheMode::PerLoop)
            .then_some((recurrence.start, recurrence.end))
    }

    /// Whether the model can decode with a loop wavefront: nGPT layers, two loops, per-loop KV.
    pub fn wavefront_supported(&self) -> bool {
        self.wavefront_recurrence().is_some() && self.slot_layers.iter().all(|(_, shared)| shared.is_none())
    }

    pub(super) fn slot_of(
        &self,
        layer_index: u32,
        loop_index: u32,
    ) -> Option<u32> {
        self.steps
            .iter()
            .position(|step| step.layer_index == layer_index && step.loop_index == loop_index)
            .and_then(|step_index| self.step_slots[step_index])
    }

    pub fn wavefront_slots(&self) -> WavefrontSlots {
        let (start, end) = self.wavefront_recurrence().expect("wavefront is not supported for this model");
        let num_layers = self.layers.len() as u32;
        WavefrontSlots {
            new_row: (0..end).filter_map(|layer| self.slot_of(layer, 0)).collect(),
            carried_row: (start..end).filter_map(|layer| self.slot_of(layer, 1)).collect(),
            output: (end..num_layers).filter_map(|layer| self.slot_of(layer, 0)).collect(),
        }
    }

    pub fn create_wavefront_carry(
        &self,
        context: &B::Context,
    ) -> Result<WavefrontCarry<B>, B::Error> {
        let row_bytes = self.model_dim as usize * self.data_type.size_in_bytes();
        Ok(WavefrontCarry {
            loop_state: context.create_buffer(row_bytes)?,
            recurrent_inputs: context.create_buffer(row_bytes)?,
        })
    }

    /// One wavefront step. `embedded` is the new row's input embedding (one row) at `position`;
    /// with `has_carried_row`, `carry` holds the row at `position - 1` after its first loop.
    /// Returns the output hidden rows (`[carried row, new row]` or `[new row]`) and writes the new
    /// row's state after its first loop into `carry`.
    pub fn encode_wavefront(
        &self,
        embedded: B::ScratchBuffer,
        carry: &mut WavefrontCarry<B>,
        position: u32,
        has_carried_row: bool,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("transformer wavefront");
        let (start, end) = self.wavefront_recurrence().expect("wavefront is not supported for this model");
        let num_layers = self.layers.len() as u32;
        let backend = TransformerEncodeError::Backend;
        let rows = if has_carried_row {
            2
        } else {
            1
        };
        let row_bytes = self.model_dim as usize * self.data_type.size_in_bytes();

        let precalculate = |positions: &[u32], command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding| {
            self.ropes
                .iter()
                .map(|rope_config| PrecalculatedRoPE::precalculate(rope_config, positions, command_buffer))
                .collect::<Result<Box<[_]>, B::Error>>()
        };
        let new_row_ropes = precalculate(&[position], command_buffer).map_err(backend)?;
        let carried_row_ropes =
            has_carried_row.then(|| precalculate(&[position - 1], command_buffer)).transpose().map_err(backend)?;
        let output_ropes = if has_carried_row {
            precalculate(&[position - 1, position], command_buffer).map_err(backend)?
        } else {
            precalculate(&[position], command_buffer).map_err(backend)?
        };
        let one_row = BatchTopology::new(&ONE_ROW, true);
        let all_rows = if has_carried_row {
            BatchTopology::new(&TWO_ROWS, true)
        } else {
            BatchTopology::new(&ONE_ROW, true)
        };

        // Runs `[first, last)` with every row in one attention state per layer.
        let run_segment = |hidden: B::ScratchBuffer,
                           layers: std::ops::Range<u32>,
                           ropes: &[PrecalculatedRoPE<B>],
                           batch_dim: &BatchTopology,
                           state: &mut TransformerState<B>,
                           command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding|
         -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
            let mut hidden = hidden;
            let mut hidden_copy = None;
            for layer_index in layers.clone() {
                let (Layer::Ngpt(layer), rope_index) = &self.layers[layer_index as usize] else {
                    unreachable!("wavefront needs nGPT layers");
                };
                let layer_state = self.slot_of(layer_index, 0).map(|slot| {
                    let TransformerLayerStateType::Owned(layer_state) = &mut state.layer_states[slot as usize] else {
                        panic!("wavefront does not support shared KV slots");
                    };
                    MaybeMut::Mut(layer_state.as_mut() as &mut dyn MixerState<B>)
                });
                let (output, output_copy) = layer.encode(
                    hidden,
                    hidden_copy.take(),
                    layer_index + 1 < layers.end,
                    rope_index.map(|index| &ropes[index]),
                    batch_dim,
                    layer_state,
                    command_buffer,
                )?;
                hidden = output;
                hidden_copy = output_copy;
            }
            Ok(hidden)
        };

        // Input layers: the new row only.
        let inputs = run_segment(embedded, 0..start, &new_row_ropes, &one_row, state, command_buffer)?;

        // Recurrent block entry, rows [carried, new]: norm(loop_state + recurrent_inputs). The new row
        // starts its first loop from its inputs (loop_state = inputs).
        let all_rows_scratch = |command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding| {
            command_buffer.allocate_scratch(rows as usize * row_bytes).map_err(backend)
        };
        let mut loop_state = all_rows_scratch(command_buffer)?;
        let mut recurrent_inputs = all_rows_scratch(command_buffer)?;
        let new_row_offset = (rows as usize - 1) * row_bytes;
        if has_carried_row {
            command_buffer.encode_copy(&carry.loop_state, (&mut loop_state).subrange_mut(..row_bytes));
            command_buffer.encode_copy(&carry.recurrent_inputs, (&mut recurrent_inputs).subrange_mut(..row_bytes));
        }
        command_buffer.encode_copy(&inputs, (&mut loop_state).subrange_mut(new_row_offset..));
        command_buffer.encode_copy(&inputs, (&mut recurrent_inputs).subrange_mut(new_row_offset..));
        let recurrence = self.recurrence.as_ref().expect("checked above");
        let combined = self.add_residual(&loop_state, &recurrent_inputs, rows, command_buffer).map_err(backend)?;
        let mut hidden = recurrence
            .reinjection_norm
            .encode(&combined, 0, rows, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(backend)?;

        // Recurrent layers: shared projections, attention per row in its own loop's state.
        let mut hidden_copy = None;
        for layer_index in start..end {
            let (Layer::Ngpt(layer), rope_index) = &self.layers[layer_index as usize] else {
                unreachable!("wavefront needs nGPT layers");
            };
            let emit_copy = layer_index + 1 < end;
            let (output, output_copy) = if layer.mixer().is_some() {
                let new_slot = self.slot_of(layer_index, 0).expect("attention layers own a state");
                let (new_state, carried_state) = if has_carried_row {
                    let carried_slot = self.slot_of(layer_index, 1).expect("per-loop KV has a second-loop slot");
                    let [new_state, carried_state] =
                        state.layer_states.get_disjoint_mut([new_slot as usize, carried_slot as usize]).unwrap();
                    (new_state, Some(carried_state))
                } else {
                    (&mut state.layer_states[new_slot as usize], None)
                };
                let mut groups = Vec::with_capacity(2);
                if let Some(carried_state) = carried_state {
                    groups.push(AttentionRowGroup {
                        batch_dim: BatchTopology::new(&ONE_ROW, true),
                        rope: layer_rope(carried_row_ropes.as_deref().expect("carried row ropes"), *rope_index),
                        state: attention_state(carried_state),
                    });
                }
                groups.push(AttentionRowGroup {
                    batch_dim: BatchTopology::new(&ONE_ROW, true),
                    rope: layer_rope(&new_row_ropes, *rope_index),
                    state: attention_state(new_state),
                });
                layer.encode_row_groups(hidden, hidden_copy.take(), emit_copy, &mut groups, command_buffer)?
            } else {
                layer.encode(hidden, hidden_copy.take(), emit_copy, None, &all_rows, None, command_buffer)?
            };
            hidden = output;
            hidden_copy = output_copy;
        }

        // Recurrent block exit: the carried row has finished its second loop, the new row its first.
        let loop_end = recurrence
            .reinjection_norm
            .encode(&hidden, 0, rows, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(backend)?;
        command_buffer.encode_copy(loop_end.subrange(new_row_offset..), &mut carry.loop_state);
        command_buffer.encode_copy(recurrent_inputs.subrange(new_row_offset..), &mut carry.recurrent_inputs);

        // Output layers: the carried row's final pass and the new row's early exit.
        let hidden = run_segment(loop_end, end..num_layers, &output_ropes, &all_rows, state, command_buffer)?;
        match &self.output_norm {
            Some(output_norm) => {
                output_norm.encode(&hidden, 0, rows, None::<&mut B::ScratchBuffer>, command_buffer).map_err(backend)
            },
            None => Ok(hidden),
        }
    }
}
