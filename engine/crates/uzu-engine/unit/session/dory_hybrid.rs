//! Core latent loop experiments on a real Dory checkpoint (`docs/dory_hybrid_loop_experiment_plan.md`).
//! Does nothing unless `UZU_DORY_HYBRID` names a JSON spec:
//!
//! ```json
//! {"model": "<converted model dir>", "out": "<results.json>",
//!  "speed": {"prompt_lengths": [128, 2048, 16384], "steps": 200, "text_steps": 64, "loops": [1, 2]},
//!  "continuity": {"prompts": "<prompts.json>", "segment": 16, "stride": 48, "max_prompts": 24}}
//! ```
//!
//! H-03 (`speed`): after a prefill, times text steps and latent steps (latent link, recurrent block, exit
//! probe) submitted back to back, and latent steps that read the probe on the CPU one step behind.
//!
//! H-04 (`continuity`): `prompts.json` entries are `{"id", "ids", "decode_ids"}`. A teacher pass feeds the
//! decode tokens one by one and keeps the recurrent block's final state `c_j` at each position. A second pass
//! replaces `segment` positions after every `stride` tokens by latent steps from `c_{j0}` with the identity
//! link (`W_down = 0`), and records the state cosine to the teacher, the output-layer readout's agreement with
//! the teacher's next token, and the KL of the first text step after the segment (against the teacher, and
//! for a control that skips the positions without latent steps).

use std::{fs::File, io::BufReader, path::PathBuf, time::Instant};

use futures::executor::block_on;
use half::bf16;
use serde::{Deserialize, Serialize};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        gpu_types::{activation_type::ActivationType, trie::TrieNode},
        kernel::{
            ActivationKernel,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
        },
    },
    config::model::language_model::LanguageModelConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology, decoder::Decoder, mixer::attention::ATTENTION_SUFFIX_CAPACITY,
        transformer::TransformerState,
    },
    safetensors::ParameterLoader,
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

const PROBE_ROWS: u32 = 32;

#[derive(Deserialize)]
struct Spec {
    model: PathBuf,
    out: Option<PathBuf>,
    speed: Option<SpeedSpec>,
    continuity: Option<ContinuitySpec>,
}

#[derive(Deserialize)]
struct SpeedSpec {
    prompt_lengths: Vec<usize>,
    steps: usize,
    text_steps: usize,
    loops: Vec<u32>,
}

#[derive(Deserialize)]
struct ContinuitySpec {
    prompts: PathBuf,
    segment: usize,
    stride: usize,
    max_prompts: usize,
}

#[derive(Deserialize)]
struct Prompt {
    id: String,
    ids: Vec<u32>,
    decode_ids: Vec<u32>,
}

#[derive(Serialize, Default)]
struct Results {
    speed: Vec<SpeedResult>,
    continuity: Option<ContinuityResult>,
}

#[derive(Serialize)]
struct SpeedResult {
    prompt_length: usize,
    text_steps_per_second: f64,
    latent: Vec<LatentSpeed>,
}

#[derive(Serialize)]
struct LatentSpeed {
    loops: u32,
    steps_per_second: f64,
    steps_per_second_with_probe_reads: f64,
}

#[derive(Serialize, Default)]
struct ContinuityResult {
    /// Largest logit difference between the hybrid text step and `Decoder::encode` on the first prompt.
    text_step_max_logit_difference: f32,
    segments: usize,
    /// Mean over segments, by latent step (1-based index k).
    cosine_to_teacher: Vec<f64>,
    cosine_to_previous_step: Vec<f64>,
    /// Baseline: the segment's start state against the teacher state at each step.
    start_state_cosine_to_teacher: Vec<f64>,
    readout_matches_teacher_top1: Vec<f64>,
    /// The same readout path on the teacher's own states (the output layers miss the segment's positions).
    teacher_state_readout_matches_top1: Vec<f64>,
    /// Sanity: the readout of the segment's start state at its own position against the teacher's top-1.
    start_readout_matches_top1: f64,
    exit_kl_latent: f64,
    exit_kl_skip: f64,
    exit_top1_latent: f64,
    exit_top1_skip: f64,
}

fn argmax(row: &[bf16]) -> u32 {
    let mut best = 0usize;
    for (index, value) in row.iter().enumerate() {
        if f32::from(*value) > f32::from(row[best]) {
            best = index;
        }
    }
    best as u32
}

fn log_softmax(row: &[bf16]) -> Vec<f64> {
    let values = row.iter().map(|value| f64::from(f32::from(*value))).collect::<Vec<_>>();
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = values.iter().map(|value| (value - max).exp()).sum::<f64>().ln() + max;
    values.into_iter().map(|value| value - log_sum).collect()
}

/// KL(teacher ‖ student) of two logit rows.
fn kl(
    teacher: &[bf16],
    student: &[bf16],
) -> f64 {
    let (teacher, student) = (log_softmax(teacher), log_softmax(student));
    teacher.iter().zip(&student).map(|(p, q)| p.exp() * (p - q)).sum()
}

fn cosine(
    a: &[bf16],
    b: &[bf16],
) -> f64 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(f32::from(*x)), f64::from(f32::from(*y)));
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Prefills `ids` in suffix-capacity chunks, accepting every chunk.
fn prefill<B: Backend>(
    context: &B::Context,
    decoder: &Decoder<B>,
    state: &mut TransformerState<B>,
    ids: &[u32],
) {
    for chunk in ids.chunks(ATTENTION_SUFFIX_CAPACITY as usize) {
        let length = chunk.len() as u32;
        state.prepare(state.context_length(), length).expect("prepare state");
        let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
        let token_buffer = command_buffer.allocate_constant_from_slice(chunk).expect("token ids");
        let nodes = (0..length)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: length,
                height: index,
            })
            .collect::<Box<[_]>>();
        let batch_dim = BatchTopology::new(&nodes, true);
        let output =
            decoder.encode(&token_buffer, &batch_dim, None, None, state, &mut command_buffer).expect("prefill");
        state.encode_accept(&(0..length).collect::<Box<[u32]>>(), &mut command_buffer).expect("accept");
        drop((output, token_buffer));
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("wait");
    }
}

/// The latent link (`u = norm(s + W_down silu(W_up s))`) and exit probe (`v s`) with BF16 weights.
/// `W_down` is zero, so the link is the identity up to the normalization's rounding.
struct LatentHead<B: Backend> {
    matmul: <B::Kernels as Kernels>::MatmulKernel,
    activation: <B::Kernels as Kernels>::ActivationKernel,
    up: B::GlobalBuffer,
    down: B::GlobalBuffer,
    probe: B::GlobalBuffer,
    dim: u32,
}

impl<B: Backend> LatentHead<B> {
    fn new(
        context: &B::Context,
        dim: u32,
    ) -> Self {
        let wave = |count: u32, scale: f32| {
            (0..count).map(|index| bf16::from_f32(((index as f32) * 0.618).sin() * scale)).collect::<Vec<_>>()
        };
        Self {
            matmul: <B::Kernels as Kernels>::MatmulKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
                .expect("matmul"),
            activation: <B::Kernels as Kernels>::ActivationKernel::new(context, DataType::BF16, true).expect("silu"),
            up: create_buffer_with_data::<B, bf16>(context, &wave(dim * dim, 0.02)),
            down: create_buffer_with_data::<B, bf16>(context, &vec![bf16::ZERO; (dim * dim) as usize]),
            probe: create_buffer_with_data::<B, bf16>(context, &wave(PROBE_ROWS * dim, 0.02)),
            dim,
        }
    }

    fn gemv(
        &mut self,
        weights_index: usize,
        input: impl crate::backends::common::BufferRef<Backend = B>,
        rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> B::ScratchBuffer {
        let mut output = command_buffer.allocate_scratch_for_shape(&[1, rows], DataType::BF16).expect("scratch");
        let weights = [&self.up, &self.down, &self.probe][weights_index];
        self.matmul
            .encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: input,
                        offset: 0,
                    },
                    b: MatmulB::FullPrecision {
                        b: weights,
                    },
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput::new(&mut output, MatmulDOps::none()),
                    gather_indices: None::<&B::GlobalBuffer>,
                    m: 1,
                    n: rows,
                    k: self.dim,
                },
                command_buffer,
            )
            .expect("gemv");
        output
    }

    fn encode_link(
        &mut self,
        decoder: &Decoder<B>,
        state: &B::ScratchBuffer,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> B::ScratchBuffer {
        let mut hidden = self.gemv(0, state, self.dim, command_buffer);
        self.activation.encode(None::<&B::ScratchBuffer>, &mut hidden, self.dim, ActivationType::SILU, command_buffer);
        let update = self.gemv(1, &hidden, self.dim, command_buffer);
        decoder.encode_reinjected(state, &update, command_buffer).expect("link norm")
    }
}

/// One latent step at `position` from the state in `carry` (updated in place); commits the recurrent KV.
/// Returns the probe output row (only row 0 is a probe; the rest pad the GEMV).
#[allow(clippy::too_many_arguments)]
fn encode_latent_step<B: Backend>(
    decoder: &Decoder<B>,
    head: &mut LatentHead<B>,
    carry: &mut B::GlobalBuffer,
    position: u32,
    loops: u32,
    latent_slots: &[u32],
    state: &mut TransformerState<B>,
    command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
) -> B::ScratchBuffer {
    state.prepare(position + 1, 1).expect("prepare state");
    let mut previous = command_buffer.allocate_scratch_for_shape(&[1, head.dim], DataType::BF16).expect("scratch");
    command_buffer.encode_copy(&*carry, &mut previous);
    let input = head.encode_link(decoder, &previous, command_buffer);
    let next = decoder.encode_latent_core(&input, position, loops, state, command_buffer).expect("latent core");
    let probe = head.gemv(2, &next, PROBE_ROWS, command_buffer);
    command_buffer.encode_copy(&next, carry);
    state.encode_accept_slots(latent_slots, &[0], command_buffer).expect("accept latent");
    probe
}

/// One text step at `position` that commits every slot; returns logits and the recurrent block state.
fn text_step<B: Backend>(
    context: &B::Context,
    decoder: &Decoder<B>,
    state: &mut TransformerState<B>,
    token: u32,
    position: u32,
    vocab_size: usize,
    dim: u32,
) -> (Vec<bf16>, Vec<bf16>) {
    state.prepare(position + 1, 1).expect("prepare state");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let token_buffer = command_buffer.allocate_constant_from_slice(&[token]).expect("token");
    let (logits, core_state) =
        decoder.encode_text_with_core_state(&token_buffer, position, state, &mut command_buffer).expect("text step");
    let mut host_logits = context.create_buffer(vocab_size * size_of::<bf16>()).expect("buffer");
    let mut host_state = context.create_buffer(dim as usize * size_of::<bf16>()).expect("buffer");
    command_buffer.encode_copy(&logits, &mut host_logits);
    command_buffer.encode_copy(&core_state, &mut host_state);
    state.encode_accept(&[0], &mut command_buffer).expect("accept");
    drop((logits, core_state, token_buffer));
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("wait");
    (buffer_to_vec::<B, bf16>(&host_logits), buffer_to_vec::<B, bf16>(&host_state))
}

/// The hybrid text step against `Decoder::encode` on the first 32 decode tokens of `prompt` (teacher-forced).
fn check_text_step<B: Backend>(
    context: &B::Context,
    decoder: &Decoder<B>,
    prompt: &Prompt,
    vocab_size: usize,
    dim: u32,
) -> f32 {
    let tokens = &prompt.decode_ids[..prompt.decode_ids.len().min(32)];
    let max_context = (prompt.ids.len() + tokens.len() + 8) as u32;
    let mut hybrid_state = decoder.create_empty_state(Some(max_context), context).expect("state");
    let mut plain_state = decoder.create_empty_state(Some(max_context), context).expect("state");
    prefill(context, decoder, &mut hybrid_state, &prompt.ids);
    prefill(context, decoder, &mut plain_state, &prompt.ids);
    let mut max_difference = 0.0f32;
    for (j, &token) in tokens.iter().enumerate() {
        let position = (prompt.ids.len() + j) as u32;
        let (hybrid, _) = text_step(context, decoder, &mut hybrid_state, token, position, vocab_size, dim);
        plain_state.prepare(plain_state.context_length(), 1).expect("prepare state");
        let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
        let token_buffer = command_buffer.allocate_constant_from_slice(&[token]).expect("token");
        let batch_dim = BatchTopology::new(
            &[TrieNode {
                trie_start: 0,
                trie_end: 1,
                height: 0,
            }],
            true,
        );
        let output = decoder
            .encode(&token_buffer, &batch_dim, Some((0..1).into()), None, &mut plain_state, &mut command_buffer)
            .expect("decoder encode");
        let logits = output.logits.expect("logits");
        let mut host_logits = context.create_buffer(vocab_size * size_of::<bf16>()).expect("buffer");
        command_buffer.encode_copy(&logits, &mut host_logits);
        plain_state.encode_accept(&[0], &mut command_buffer).expect("accept");
        drop((logits, output.hidden_features, output.final_hidden, token_buffer));
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("wait");
        let plain = buffer_to_vec::<B, bf16>(&host_logits);
        for (a, b) in hybrid.iter().zip(&plain) {
            max_difference = max_difference.max((f32::from(*a) - f32::from(*b)).abs());
        }
    }
    max_difference
}

fn run_speed<B: Backend>(
    context: &B::Context,
    decoder: &Decoder<B>,
    spec: &SpeedSpec,
    token_pool: &[u32],
    dim: u32,
) -> Vec<SpeedResult> {
    let mut head = LatentHead::<B>::new(context, dim);
    let mut results = Vec::new();
    for &prompt_length in &spec.prompt_lengths {
        let ids = token_pool.iter().cycle().take(prompt_length).copied().collect::<Vec<_>>();
        let max_context = (prompt_length + spec.text_steps + spec.steps * 2 + 16) as u32;
        let mut state = decoder.create_empty_state(Some(max_context), context).expect("state");
        prefill(context, decoder, &mut state, &ids);

        // Text steps submitted back to back (teacher-forced tokens, no CPU sync between steps).
        let mut carry = context.create_buffer(dim as usize * size_of::<bf16>()).expect("carry");
        let started = Instant::now();
        let mut pending = Vec::new();
        for step in 0..spec.text_steps {
            let position = (prompt_length + step) as u32;
            state.prepare(position + 1, 1).expect("prepare state");
            let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
            let token_buffer = command_buffer.allocate_constant_from_slice(&[ids[step % ids.len()]]).expect("token");
            let (logits, core_state) = decoder
                .encode_text_with_core_state(&token_buffer, position, &mut state, &mut command_buffer)
                .expect("text step");
            command_buffer.encode_copy(&core_state, &mut carry);
            state.encode_accept(&[0], &mut command_buffer).expect("accept");
            drop((logits, core_state, token_buffer));
            pending.push(command_buffer.end_encoding().submit());
        }
        for step in pending {
            block_on(step.wait_until_completed()).expect("wait");
        }
        let text_steps_per_second = spec.text_steps as f64 / started.elapsed().as_secs_f64();
        let first_latent = (prompt_length + spec.text_steps) as u32;
        let bookkeeping = state.bookkeeping();
        let carry_start = buffer_to_vec::<B, bf16>(&carry);

        let mut latent = Vec::new();
        for &loops in &spec.loops {
            let latent_slots = decoder.latent_slots(loops);
            let mut timings = [0.0f64; 2];
            for (mode, timing) in timings.iter_mut().enumerate() {
                state.restore_bookkeeping(&bookkeeping);
                let mut carry = create_buffer_with_data::<B, bf16>(context, &carry_start);
                let started = Instant::now();
                let mut pending = Vec::new();
                let mut previous: Option<(<B::CommandBuffer as CommandBuffer>::Pending, B::GlobalBuffer)> = None;
                for step in 0..spec.steps {
                    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
                    let probe = encode_latent_step(
                        decoder,
                        &mut head,
                        &mut carry,
                        first_latent + step as u32,
                        loops,
                        &latent_slots,
                        &mut state,
                        &mut command_buffer,
                    );
                    let mut host_probe = context.create_buffer(PROBE_ROWS as usize * size_of::<bf16>()).expect("probe");
                    command_buffer.encode_copy(&probe, &mut host_probe);
                    drop(probe);
                    let submitted = command_buffer.end_encoding().submit();
                    if mode == 0 {
                        pending.push(submitted);
                    } else {
                        // Read the previous step's probe while this step runs.
                        if let Some((done, host)) = previous.take() {
                            block_on(done.wait_until_completed()).expect("wait");
                            std::hint::black_box(f32::from(buffer_to_vec::<B, bf16>(&host)[0]));
                        }
                        previous = Some((submitted, host_probe));
                    }
                }
                for step in pending {
                    block_on(step.wait_until_completed()).expect("wait");
                }
                if let Some((done, _)) = previous {
                    block_on(done.wait_until_completed()).expect("wait");
                }
                *timing = spec.steps as f64 / started.elapsed().as_secs_f64();
            }
            let result = LatentSpeed {
                loops,
                steps_per_second: timings[0],
                steps_per_second_with_probe_reads: timings[1],
            };
            eprintln!(
                "prompt {prompt_length}: text {text_steps_per_second:.1} steps/s, latent loops {loops}: {:.1} steps/s \
                 ({:.1} with probe reads)",
                result.steps_per_second, result.steps_per_second_with_probe_reads
            );
            latent.push(result);
        }
        results.push(SpeedResult {
            prompt_length,
            text_steps_per_second,
            latent,
        });
    }
    results
}

/// Top-1 of the output-layer readout of `core_state` at `position` (output KV stays uncommitted).
fn readout_top1<B: Backend>(
    context: &B::Context,
    decoder: &Decoder<B>,
    state: &mut TransformerState<B>,
    core_state: &[bf16],
    position: u32,
    vocab_size: usize,
) -> u32 {
    state.prepare(position + 1, 1).expect("prepare state");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut scratch_state =
        command_buffer.allocate_scratch_for_shape(&[1, core_state.len() as u32], DataType::BF16).expect("scratch");
    let host_state = create_buffer_with_data::<B, bf16>(context, core_state);
    command_buffer.encode_copy(&host_state, &mut scratch_state);
    let logits = decoder.encode_latent_readout(&scratch_state, position, state, &mut command_buffer).expect("readout");
    let mut host_logits = context.create_buffer(vocab_size * size_of::<bf16>()).expect("buffer");
    command_buffer.encode_copy(&logits, &mut host_logits);
    drop((logits, scratch_state));
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("wait");
    argmax(&buffer_to_vec::<B, bf16>(&host_logits))
}

fn run_continuity<B: Backend>(
    context: &B::Context,
    decoder: &Decoder<B>,
    spec: &ContinuitySpec,
    prompts: &[Prompt],
    vocab_size: usize,
    dim: u32,
) -> ContinuityResult {
    let mut head = LatentHead::<B>::new(context, dim);
    let latent_slots = decoder.latent_slots(1);
    let segment = spec.segment;
    let mut cosine_sum = vec![0.0f64; segment];
    let mut step_cosine_sum = vec![0.0f64; segment];
    let mut readout_sum = vec![0.0f64; segment];
    let mut start_cosine_sum = vec![0.0f64; segment];
    let mut teacher_readout_sum = vec![0.0f64; segment];
    let mut start_readout_sum = 0.0f64;
    let (mut exit_kl_latent, mut exit_kl_skip, mut exit_top1_latent, mut exit_top1_skip) = (0.0, 0.0, 0.0, 0.0);
    let mut segments = 0usize;
    let text_step_max_logit_difference = check_text_step(context, decoder, &prompts[0], vocab_size, dim);
    eprintln!("hybrid text step against Decoder::encode: max logit difference {text_step_max_logit_difference}");

    for prompt in prompts.iter().take(spec.max_prompts) {
        let start = prompt.ids.len() as u32;
        let tokens = &prompt.decode_ids;
        let max_context = (prompt.ids.len() + tokens.len() + segment + 8) as u32;
        let segment_starts = (spec.stride..).step_by(spec.stride).take_while(|j0| j0 + segment + 1 < tokens.len());
        let segment_starts = segment_starts.collect::<Vec<_>>();
        let exit_positions = segment_starts.iter().map(|j0| j0 + segment + 1).collect::<Vec<_>>();

        // Teacher pass: every decode token as text.
        let mut state = decoder.create_empty_state(Some(max_context), context).expect("state");
        prefill(context, decoder, &mut state, &prompt.ids);
        let mut teacher_states = Vec::with_capacity(tokens.len());
        let mut teacher_top1 = Vec::with_capacity(tokens.len());
        let mut teacher_exit_logits = Vec::new();
        for (j, &token) in tokens.iter().enumerate() {
            let (logits, core_state) =
                text_step(context, decoder, &mut state, token, start + j as u32, vocab_size, dim);
            teacher_top1.push(argmax(&logits));
            teacher_states.push(core_state);
            if exit_positions.contains(&j) {
                teacher_exit_logits.push(logits);
            }
        }
        drop(state);

        // Latent pass.
        let mut state = decoder.create_empty_state(Some(max_context), context).expect("state");
        prefill(context, decoder, &mut state, &prompt.ids);
        let mut next_segment = 0;
        for (j, &token) in tokens.iter().enumerate() {
            text_step(context, decoder, &mut state, token, start + j as u32, vocab_size, dim);
            if next_segment >= segment_starts.len() || j != segment_starts[next_segment] {
                continue;
            }
            let bookkeeping = state.bookkeeping();
            // The start state's readout at its own position repeats the teacher's text step (sanity).
            let start_readout =
                readout_top1(context, decoder, &mut state, &teacher_states[j], start + j as u32, vocab_size);
            start_readout_sum += f64::from(u8::from(start_readout == teacher_top1[j]));
            let mut carry = create_buffer_with_data::<B, bf16>(context, &teacher_states[j]);
            let mut previous_state = teacher_states[j].clone();
            for k in 0..segment {
                let position = start + (j + 1 + k) as u32;
                let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
                let probe = encode_latent_step(
                    decoder,
                    &mut head,
                    &mut carry,
                    position,
                    1,
                    &latent_slots,
                    &mut state,
                    &mut command_buffer,
                );
                drop(probe);
                block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("wait");
                let latent_state = buffer_to_vec::<B, bf16>(&carry);

                // Readout through the output layers (uncommitted), of the latent state and of the teacher state.
                let readout = readout_top1(context, decoder, &mut state, &latent_state, position, vocab_size);
                let teacher_readout =
                    readout_top1(context, decoder, &mut state, &teacher_states[j + 1 + k], position, vocab_size);
                teacher_readout_sum[k] += f64::from(u8::from(teacher_readout == teacher_top1[j + 1 + k]));
                start_cosine_sum[k] += cosine(&teacher_states[j], &teacher_states[j + 1 + k]);

                cosine_sum[k] += cosine(&latent_state, &teacher_states[j + 1 + k]);
                step_cosine_sum[k] += cosine(&latent_state, &previous_state);
                readout_sum[k] += f64::from(u8::from(readout == teacher_top1[j + 1 + k]));
                previous_state = latent_state;
            }

            // The first text step after the segment, with latent KV in the recurrent layers.
            let exit = j + segment + 1;
            let teacher_logits = &teacher_exit_logits[next_segment];
            let (logits, _) =
                text_step(context, decoder, &mut state, tokens[exit], start + exit as u32, vocab_size, dim);
            exit_kl_latent += kl(teacher_logits, &logits);
            exit_top1_latent += f64::from(u8::from(argmax(&logits) == argmax(teacher_logits)));

            // Control: the same text step with the segment's positions skipped everywhere.
            state.restore_bookkeeping(&bookkeeping);
            let (logits, _) =
                text_step(context, decoder, &mut state, tokens[exit], start + exit as u32, vocab_size, dim);
            exit_kl_skip += kl(teacher_logits, &logits);
            exit_top1_skip += f64::from(u8::from(argmax(&logits) == argmax(teacher_logits)));

            state.restore_bookkeeping(&bookkeeping);
            segments += 1;
            next_segment += 1;
        }
        eprintln!("{}: {} segments so far", prompt.id, segments);
    }
    let mean = |values: Vec<f64>| values.into_iter().map(|value| value / segments.max(1) as f64).collect::<Vec<_>>();
    ContinuityResult {
        text_step_max_logit_difference,
        segments,
        cosine_to_teacher: mean(cosine_sum),
        cosine_to_previous_step: mean(step_cosine_sum),
        start_state_cosine_to_teacher: mean(start_cosine_sum),
        readout_matches_teacher_top1: mean(readout_sum),
        teacher_state_readout_matches_top1: mean(teacher_readout_sum),
        start_readout_matches_top1: start_readout_sum / segments.max(1) as f64,
        exit_kl_latent: exit_kl_latent / segments.max(1) as f64,
        exit_kl_skip: exit_kl_skip / segments.max(1) as f64,
        exit_top1_latent: exit_top1_latent / segments.max(1) as f64,
        exit_top1_skip: exit_top1_skip / segments.max(1) as f64,
    }
}

fn run<B: Backend>(spec: &Spec) {
    let config: LanguageModelConfig =
        serde_json::from_reader(BufReader::new(File::open(spec.model.join("config.json")).unwrap())).unwrap();
    let vocab_size = config.decoder_config.vocab_size as usize;
    let context = block_on(B::Context::new()).expect("context");
    let weights_file = File::open(spec.model.join("model.safetensors")).unwrap();
    let loader = ParameterLoader::<B>::new(&weights_file, &*context).unwrap();
    let decoder = Decoder::new(DataType::BF16, &config.decoder_config, &loader.tree().subtree("decoder"), &context)
        .expect("decoder");
    assert!(decoder.hybrid_supported(), "this model does not support the core latent loop");
    let dim = config.decoder_config.transformer_config.model_dim;

    let mut results = Results::default();
    let prompts: Vec<Prompt> = spec
        .continuity
        .as_ref()
        .map(|continuity| serde_json::from_reader(BufReader::new(File::open(&continuity.prompts).unwrap())).unwrap())
        .unwrap_or_default();
    if let Some(speed) = &spec.speed {
        let pool =
            prompts.iter().flat_map(|prompt| prompt.ids.iter().chain(&prompt.decode_ids)).copied().collect::<Vec<_>>();
        let pool = if pool.is_empty() {
            (1000..3000).collect()
        } else {
            pool
        };
        results.speed = run_speed(&*context, &decoder, speed, &pool, dim);
    }
    if let Some(continuity) = &spec.continuity {
        let result = run_continuity(&*context, &decoder, continuity, &prompts, vocab_size, dim);
        eprintln!("{}", serde_json::to_string(&result).unwrap());
        results.continuity = Some(result);
    }
    if let Some(out) = &spec.out {
        std::fs::write(out, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }
}

#[uzu_test]
fn dory_hybrid_loop() {
    let Ok(spec_path) = std::env::var("UZU_DORY_HYBRID") else {
        return;
    };
    let spec: Spec = serde_json::from_reader(BufReader::new(File::open(spec_path).unwrap())).unwrap();
    #[cfg(backend = "metal")]
    run::<crate::backends::metal::Metal>(&spec);
}
