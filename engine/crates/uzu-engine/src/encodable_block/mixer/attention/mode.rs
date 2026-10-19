use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{
            AttentionArguments, AttentionInput, AttentionKernel, AttentionPrepareInt8Kernel, AttentionPrepareKernel,
            FullPrecisionInput, Int8Input, NgptGateNormKernel, ScaleColumnsKernel,
        },
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::Linear,
        mixer::{
            MixerState,
            attention::{
                Attention, AttentionEncodeError, KVCacheView,
                kv_storage::{Int8KV, KVStorage},
                qkv_norm::QKVNorm,
                rope::PrecalculatedRoPE,
                state::AttentionState,
            },
        },
    },
    utils::maybe_mut::MaybeMut,
};

enum KVBuffers<'a, B: Backend> {
    BF16 {
        keys: MaybeMut<'a, dyn Buffer<Backend = B>>,
        values: MaybeMut<'a, dyn Buffer<Backend = B>>,
    },
    Int8(MaybeMut<'a, Int8KV<B>>),
}

/// Rows of an attention batch that read and append one KV state.
pub(crate) struct AttentionRowGroup<'a, 'b, B: Backend> {
    pub batch_dim: BatchTopology<'b>,
    pub rope: Option<&'a PrecalculatedRoPE<B>>,
    pub state: &'a mut AttentionState<B>,
}

pub(super) struct LinearProjection<B: Backend> {
    pub(super) lin: Box<dyn Linear<B>>,
    pub(super) norm: QKVNorm<B>,
}

impl<B: Backend> LinearProjection<B> {
    fn project(
        &self,
        hidden: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let mut projected = self.lin.encode(hidden, batch_dim, command_buffer)?;
        self.norm.encode(&mut projected, batch_dim, command_buffer);
        Ok(projected)
    }
}

impl<B: Backend> Attention<B> {
    pub(super) fn attend(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let mut qkvg = self.projection.project(hidden, batch_dim.size(), command_buffer)?;
        let q_dim = self.num_q_heads * self.head_dim;
        let gate_offset = (self.projection_dim - q_dim) as usize * self.data_type.size_in_bytes();
        if let (Some((gate_scales, scale_columns)), None) = (&self.gate_scales, &self.fused_gate_norm) {
            scale_columns.encode(
                &mut qkvg,
                gate_scales,
                batch_dim.size(),
                q_dim,
                self.projection_dim,
                self.projection_dim - q_dim,
                command_buffer,
            );
        }
        let gate = (self.has_gate && self.fused_gate_norm.is_none()).then(|| qkvg.subrange(gate_offset..));

        let attention_output = self.encode_core(&qkvg, gate, precalculated_rope, batch_dim, state, command_buffer)?;

        self.encode_output(attention_output, &qkvg, gate_offset, batch_dim.size(), command_buffer)
    }

    /// Prepare (RoPE, scales, KV append) and the attention kernel for the rows of `qkvg`.
    fn encode_core(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        gate: Option<impl BufferRef<Backend = B>>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let (mut keys, mut values);
        let (mut kv, cache) = match state {
            Some(MaybeMut::Mut(state)) => {
                let cache = state.view();
                let kv = match &mut state.kv {
                    KVStorage::BF16(kv, _) => KVBuffers::BF16 {
                        keys: MaybeMut::Mut(kv.keys.as_mut()),
                        values: MaybeMut::Mut(kv.values.as_mut()),
                    },
                    KVStorage::Int8(kv, _) => KVBuffers::Int8(MaybeMut::Mut(kv)),
                };
                (kv, cache)
            },
            Some(MaybeMut::Const(state)) => {
                let kv = match &state.kv {
                    KVStorage::BF16(kv, _) => KVBuffers::BF16 {
                        keys: MaybeMut::Const(kv.keys.as_ref()),
                        values: MaybeMut::Const(kv.values.as_ref()),
                    },
                    KVStorage::Int8(kv, _) => KVBuffers::Int8(MaybeMut::Const(kv)),
                };
                (kv, state.view())
            },
            None => {
                let Some(num_kv_heads) = self.num_kv_heads else {
                    panic!("stateless attention doesn't support query-only projection");
                };
                assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

                keys = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)
                    .map_err(AttentionEncodeError::Backend)?;
                values = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)
                    .map_err(AttentionEncodeError::Backend)?;
                let kv = KVBuffers::BF16 {
                    keys: MaybeMut::Mut(&mut keys),
                    values: MaybeMut::Mut(&mut values),
                };
                let cache = self.ring_capacity.map_or_else(|| KVCacheView::full(0), |_| KVCacheView::ring(0, 0));
                (kv, cache)
            },
        };
        let (queries, query_scales);
        let input = match &mut kv {
            KVBuffers::BF16 {
                keys,
                values,
            } => {
                queries = self
                    .prepare_bf16(
                        qkvg,
                        keys,
                        values,
                        cache.prefix_len(),
                        self.num_q_heads,
                        precalculated_rope,
                        batch_dim.size(),
                        command_buffer,
                    )
                    .map_err(AttentionEncodeError::Backend)?;
                AttentionInput::FullPrecision(FullPrecisionInput {
                    queries: &queries,
                    keys: &**keys,
                    values: &**values,
                })
            },
            KVBuffers::Int8(kv) => {
                (queries, query_scales) = self
                    .prepare_int8(
                        qkvg,
                        kv,
                        cache.prefix_len(),
                        self.num_q_heads,
                        precalculated_rope,
                        batch_dim.size(),
                        command_buffer,
                    )
                    .map_err(AttentionEncodeError::Backend)?;
                AttentionInput::Int8(Int8Input {
                    query_codes: &queries,
                    query_scales: &query_scales,
                    k_codes: kv.k_codes.as_ref(),
                    k_scales: kv.k_scales.as_ref(),
                    v_codes: kv.v_codes.as_ref(),
                    v_scales: kv.v_scales.as_ref(),
                })
            },
        };
        let trie = (!batch_dim.is_flat())
            .then(|| command_buffer.allocate_constant_from_slice(batch_dim.nodes()))
            .transpose()
            .map_err(AttentionEncodeError::Backend)?;
        self.kernel
            .encode(
                input,
                AttentionArguments {
                    suffix_length: batch_dim.size(),
                    trie: trie.as_ref(),
                    sinks: self.sinks.as_ref(),
                    cache,
                    gate,
                },
                command_buffer,
            )
            .map_err(AttentionEncodeError::Kernel)
    }

    /// The gate and output norm, then the output projection.
    fn encode_output(
        &self,
        attention_output: B::ScratchBuffer,
        qkvg: &B::ScratchBuffer,
        gate_offset: usize,
        rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let q_dim = self.num_q_heads * self.head_dim;
        let attention_output = match (&self.fused_gate_norm, &self.output_norm) {
            (Some(fused), _) => {
                let (gate_scales, _) = self.gate_scales.as_ref().expect("fused gate norm requires gate scales");
                let mut output =
                    command_buffer.allocate_scratch(attention_output.size()).map_err(AttentionEncodeError::Backend)?;
                fused.kernel.encode(
                    &attention_output,
                    qkvg.subrange(gate_offset..),
                    gate_scales,
                    &mut output,
                    rows,
                    q_dim,
                    self.projection_dim,
                    fused.epsilon,
                    command_buffer,
                );
                output
            },
            (None, Some(output_norm)) => output_norm
                .encode(&attention_output, 0, rows, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(AttentionEncodeError::Backend)?,
            (None, None) => attention_output,
        };

        Ok(self.out_projection.encode(attention_output, rows, command_buffer)?)
    }

    /// Attention over consecutive row groups that share the projections but each read and append
    /// their own KV state at their own positions (a loop wavefront batches rows from different loops).
    pub(crate) fn attend_row_groups(
        &self,
        hidden: B::ScratchBuffer,
        groups: &mut [AttentionRowGroup<'_, '_, B>],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        assert!(
            !self.has_gate || self.fused_gate_norm.is_some(),
            "row groups apply the gate after attention, which needs the fused gate norm"
        );
        let rows = groups.iter().map(|group| group.batch_dim.size()).sum::<u32>();
        let qkvg = self.projection.project(hidden, rows, command_buffer)?;
        let q_dim = self.num_q_heads * self.head_dim;
        let gate_offset = (self.projection_dim - q_dim) as usize * self.data_type.size_in_bytes();
        let qkvg_row_bytes = self.projection_dim as usize * self.data_type.size_in_bytes();
        let output_row_bytes = q_dim as usize * self.data_type.size_in_bytes();
        let mut attention_output = command_buffer
            .allocate_scratch_for_shape(&[rows, self.num_q_heads, self.head_dim], self.data_type)
            .map_err(AttentionEncodeError::Backend)?;
        let mut row = 0usize;
        for group in groups.iter_mut() {
            let group_rows = group.batch_dim.size() as usize;
            let group_output = self.encode_core(
                qkvg.subrange(row * qkvg_row_bytes..(row + group_rows) * qkvg_row_bytes),
                None::<&B::ScratchBuffer>,
                group.rope,
                &group.batch_dim,
                Some(MaybeMut::Mut(&mut *group.state)),
                command_buffer,
            )?;
            command_buffer.encode_copy(
                &group_output,
                (&mut attention_output).subrange_mut(row * output_row_bytes..(row + group_rows) * output_row_bytes),
            );
            row += group_rows;
        }
        self.encode_output(attention_output, &qkvg, gate_offset, rows, command_buffer)
    }

    pub fn append_projected_kv(
        &self,
        mut key_value: impl BufferMut<Backend = B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.projection.norm.encode_key_value(key_value.reborrow(), batch_dim, command_buffer);
        let prefix_len = state.view().prefix_len();
        match &mut state.kv {
            KVStorage::BF16(kv, _) => {
                self.prepare_bf16(
                    key_value.as_ref(),
                    &mut MaybeMut::Mut(kv.keys.as_mut()),
                    &mut MaybeMut::Mut(kv.values.as_mut()),
                    prefix_len,
                    0,
                    Some(precalculated_rope),
                    batch_dim,
                    command_buffer,
                )?;
            },
            KVStorage::Int8(kv, _) => {
                self.prepare_int8(
                    key_value.as_ref(),
                    &mut MaybeMut::Mut(kv),
                    prefix_len,
                    0,
                    Some(precalculated_rope),
                    batch_dim,
                    command_buffer,
                )?;
            },
        }
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn prepare_bf16(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        keys: &mut MaybeMut<'_, dyn Buffer<Backend = B>>,
        values: &mut MaybeMut<'_, dyn Buffer<Backend = B>>,
        kv_token_offset: u32,
        num_q_heads: u32,
        rope: Option<&PrecalculatedRoPE<B>>,
        suffix_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let (output_keys, output_values) = match (keys, values) {
            (MaybeMut::Mut(keys), MaybeMut::Mut(values)) => (Some(&mut **keys), Some(&mut **values)),
            _ => (None, None),
        };
        let append_kv = output_keys.is_some();
        let input_row_stride = if num_q_heads == 0 {
            2 * self.num_groups * self.head_dim
        } else {
            self.projection_dim
        };
        let query_count = num_q_heads as usize * suffix_length as usize;
        let mut queries = command_buffer
            .allocate_scratch((query_count * self.head_dim as usize).max(1) * self.data_type.size_in_bytes())?;
        self.prepare.encode(
            qkvg,
            &mut queries,
            output_keys,
            output_values,
            rope.map(|rope| &rope.cosines),
            rope.map(|rope| &rope.sines),
            self.query_scales.as_ref(),
            num_q_heads,
            append_kv.then_some(self.num_groups),
            self.head_dim,
            rope.map(|rope| rope.dim),
            append_kv.then_some(kv_token_offset),
            input_row_stride,
            suffix_length,
            command_buffer,
        );
        Ok(queries)
    }

    fn prepare_int8(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        kv: &mut MaybeMut<'_, Int8KV<B>>,
        kv_token_offset: u32,
        num_q_heads: u32,
        rope: Option<&PrecalculatedRoPE<B>>,
        suffix_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        let (k_codes, k_scales, v_codes, v_scales) = match kv {
            MaybeMut::Mut(kv) => (
                Some(kv.k_codes.as_mut()),
                Some(kv.k_scales.as_mut()),
                Some(kv.v_codes.as_mut()),
                Some(kv.v_scales.as_mut()),
            ),
            MaybeMut::Const(_) => (None, None, None, None),
        };
        let input_row_stride = if num_q_heads == 0 {
            2 * self.num_groups * self.head_dim
        } else {
            self.projection_dim
        };
        let query_count = num_q_heads as usize * suffix_length as usize;
        // K/V-only calls still need bound query buffers.
        let mut query_codes = command_buffer.allocate_scratch((query_count * self.head_dim as usize).max(1))?;
        let mut query_scales = command_buffer.allocate_scratch(query_count.max(1) * DataType::BF16.size_in_bytes())?;
        self.prepare_int8.as_ref().expect("INT8 prepare kernel").encode(
            qkvg,
            None::<&mut B::ScratchBuffer>,
            Some(&mut query_codes),
            Some(&mut query_scales),
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            rope.map(|rope| &rope.cosines),
            rope.map(|rope| &rope.sines),
            self.query_scales.as_ref(),
            num_q_heads,
            rope.map(|rope| rope.dim),
            kv_token_offset,
            input_row_stride,
            suffix_length,
            num_q_heads / self.num_groups * suffix_length,
            command_buffer,
        );
        Ok((query_codes, query_scales))
    }
}
