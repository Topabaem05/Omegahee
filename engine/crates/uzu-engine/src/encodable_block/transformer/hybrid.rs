//! Experimental core latent loop (`docs/dory_hybrid_loop_experiment_plan.md`, H-03 and H-04).
//!
//! A latent step runs only the recurrent block on a given input state `u` (the latent link's output): the
//! block starts from `norm(u + u) = u` and runs `loops` loops, and its final state `s` is the input of the
//! next latent step. Only the recurrent layers write KV at the latent position; the input and output layers
//! skip it, so their caches hold text positions only (RoPE still uses the global position). Text steps run
//! as usual and also return the block's final state.
//!
//! Every function leaves its KV in the suffix (index 0); the caller accepts the slots it commits.

use super::{Layer, Transformer, TransformerEncodeError, TransformerLayerStateType, TransformerState};
use crate::{
    backends::common::{Backend, Buffer, CommandBuffer, CommandBufferEncoding, gpu_types::trie::TrieNode},
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{MixerState, attention::rope::PrecalculatedRoPE},
    },
    utils::maybe_mut::MaybeMut,
};

const ONE_ROW: [TrieNode; 1] = [TrieNode {
    trie_start: 0,
    trie_end: 1,
    height: 0,
}];

impl<B: Backend> Transformer<B> {
    /// nGPT layers with a recurrent block and no shared KV slots.
    pub fn hybrid_supported(&self) -> bool {
        self.recurrence.is_some() && self.is_ngpt && self.slot_layers.iter().all(|(_, shared)| shared.is_none())
    }

    fn recurrent_range(&self) -> (u32, u32) {
        let recurrence = self.recurrence.as_ref().expect("the hybrid loop needs a recurrent block");
        (recurrence.start, recurrence.end)
    }

    /// The slots a latent step with `loops` loops writes (one per recurrent attention layer and loop slot).
    pub fn latent_slots(
        &self,
        loops: u32,
    ) -> Box<[u32]> {
        let (start, end) = self.recurrent_range();
        let mut slots = (0..loops)
            .flat_map(|loop_index| (start..end).filter_map(move |layer| self.slot_of(layer, loop_index)))
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        slots.into_boxed_slice()
    }

    fn rope_at(
        &self,
        position: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Box<[PrecalculatedRoPE<B>]>, TransformerEncodeError<B>> {
        self.ropes
            .iter()
            .map(|rope_config| PrecalculatedRoPE::precalculate(rope_config, &[position], command_buffer))
            .collect::<Result<Box<[_]>, B::Error>>()
            .map_err(TransformerEncodeError::Backend)
    }

    /// Layers `[first, last)` on one row, each reading and appending the KV slot of `loop_index`.
    fn run_layers(
        &self,
        hidden: B::ScratchBuffer,
        layers: std::ops::Range<u32>,
        loop_index: u32,
        ropes: &[PrecalculatedRoPE<B>],
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let batch_dim = BatchTopology::new(&ONE_ROW, true);
        let mut hidden = hidden;
        let mut hidden_copy = None;
        for layer_index in layers.clone() {
            let (Layer::Ngpt(layer), rope_index) = &self.layers[layer_index as usize] else {
                unreachable!("the hybrid loop needs nGPT layers");
            };
            let layer_state = self.slot_of(layer_index, loop_index).map(|slot| {
                let TransformerLayerStateType::Owned(layer_state) = &mut state.layer_states[slot as usize] else {
                    panic!("the hybrid loop does not support shared KV slots");
                };
                MaybeMut::Mut(layer_state.as_mut() as &mut dyn MixerState<B>)
            });
            let (output, output_copy) = layer.encode(
                hidden,
                hidden_copy.take(),
                layer_index + 1 < layers.end,
                rope_index.map(|index| &ropes[index]),
                &batch_dim,
                layer_state,
                command_buffer,
            )?;
            hidden = output;
            hidden_copy = output_copy;
        }
        Ok(hidden)
    }

    /// The recurrent block on `inputs` (one row) for `loops` loops: `norm(state + inputs)` before each loop,
    /// `norm` after it. Returns the block's final state.
    fn run_core(
        &self,
        inputs: &B::ScratchBuffer,
        loops: u32,
        ropes: &[PrecalculatedRoPE<B>],
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let (start, end) = self.recurrent_range();
        let recurrence = self.recurrence.as_ref().expect("checked by recurrent_range");
        let backend = TransformerEncodeError::Backend;
        let mut loop_state = command_buffer.allocate_scratch(inputs.size()).map_err(backend)?;
        command_buffer.encode_copy(inputs, &mut loop_state);
        for loop_index in 0..loops {
            let combined = self.add_residual(&loop_state, inputs, 1, command_buffer).map_err(backend)?;
            let hidden = recurrence
                .reinjection_norm
                .encode(&combined, 0, 1, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(backend)?;
            let hidden = self.run_layers(hidden, start..end, loop_index, ropes, state, command_buffer)?;
            loop_state = recurrence
                .reinjection_norm
                .encode(&hidden, 0, 1, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(backend)?;
        }
        Ok(loop_state)
    }

    /// `norm(state + update)` with the recurrent block's normalization (the latent link's residual output).
    pub fn encode_reinjected(
        &self,
        state: &B::ScratchBuffer,
        update: &B::ScratchBuffer,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let recurrence = self.recurrence.as_ref().expect("the hybrid loop needs a recurrent block");
        let combined = self.add_residual(state, update, 1, command_buffer).map_err(TransformerEncodeError::Backend)?;
        recurrence
            .reinjection_norm
            .encode(&combined, 0, 1, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(TransformerEncodeError::Backend)
    }

    /// One latent step at the global `position`: the recurrent block alone on `input`. Returns its final state.
    pub fn encode_latent_core(
        &self,
        input: &B::ScratchBuffer,
        position: u32,
        loops: u32,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("transformer latent step");
        let ropes = self.rope_at(position, command_buffer)?;
        self.run_core(input, loops, &ropes, state, command_buffer)
    }

    /// The output layers and output norm on a recurrent block state at `position` (their KV stays uncommitted).
    pub fn encode_output_readout(
        &self,
        core_state: &B::ScratchBuffer,
        position: u32,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("transformer output readout");
        let (_, end) = self.recurrent_range();
        let ropes = self.rope_at(position, command_buffer)?;
        let mut hidden = command_buffer.allocate_scratch(core_state.size()).map_err(TransformerEncodeError::Backend)?;
        command_buffer.encode_copy(core_state, &mut hidden);
        let hidden = self.run_layers(hidden, end..self.layers.len() as u32, 0, &ropes, state, command_buffer)?;
        self.output_rows(hidden, command_buffer)
    }

    /// One text step at the global `position` on an embedded row. Returns the output hidden row and the
    /// recurrent block's final state.
    pub fn encode_text_with_core_state(
        &self,
        embedded: B::ScratchBuffer,
        position: u32,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), TransformerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("transformer text step");
        let (start, end) = self.recurrent_range();
        let loops = self.recurrence.as_ref().expect("checked by recurrent_range").num_loops;
        let ropes = self.rope_at(position, command_buffer)?;
        let inputs = self.run_layers(embedded, 0..start, 0, &ropes, state, command_buffer)?;
        let core_state = self.run_core(&inputs, loops, &ropes, state, command_buffer)?;
        let mut hidden = command_buffer.allocate_scratch(core_state.size()).map_err(TransformerEncodeError::Backend)?;
        command_buffer.encode_copy(&core_state, &mut hidden);
        let hidden = self.run_layers(hidden, end..self.layers.len() as u32, 0, &ropes, state, command_buffer)?;
        Ok((self.output_rows(hidden, command_buffer)?, core_state))
    }

    fn output_rows(
        &self,
        hidden: B::ScratchBuffer,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        match &self.output_norm {
            Some(output_norm) => output_norm
                .encode(&hidden, 0, 1, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerEncodeError::Backend),
            None => Ok(hidden),
        }
    }
}
