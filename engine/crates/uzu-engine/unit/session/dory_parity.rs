//! Dumps uzu decoder logits for a fixed token set so they can be compared with the lalamo and
//! PyTorch references (omega-uzu/reference/compare_uzu.py). Does nothing unless `UZU_DORY_PARITY`
//! names a JSON spec:
//!
//! ```json
//! {"model": "<converted model dir>", "prompts": "<prompts.json>", "out_dir": "<dir>",
//!  "backend": "cpu" | "metal", "num_loops": 2, "kv_cache_mode": "per_loop" | "last_loop",
//!  "only": ["hello"]}
//! ```
//!
//! `prompts.json` entries are `{"id", "ids", "decode_ids"}`; `decode_ids` are teacher-forced one
//! token at a time after the prefill.

use std::{
    fs::File,
    io::{BufReader, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use futures::executor::block_on;
use half::bf16;
use serde::Deserialize;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
    config::{model::language_model::LanguageModelConfig, transformer::RecurrentKVCacheMode},
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::Decoder, mixer::attention::ATTENTION_SUFFIX_CAPACITY},
    safetensors::ParameterLoader,
    tests::helpers::buffer_to_vec,
};

#[derive(Deserialize)]
struct Spec {
    model: PathBuf,
    prompts: PathBuf,
    out_dir: PathBuf,
    backend: String,
    num_loops: Option<u32>,
    kv_cache_mode: Option<RecurrentKVCacheMode>,
    only: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct Prompt {
    id: String,
    ids: Vec<u32>,
    #[serde(default)]
    decode_ids: Vec<u32>,
}

fn write_f32(
    path: &Path,
    values: &[f32],
) {
    let mut file = File::create(path).expect("create logits file");
    file.write_all(bytemuck::cast_slice(values)).expect("write logits");
}

fn run_step<B: Backend>(
    context: &B::Context,
    decoder: &Decoder<B>,
    state: &mut crate::encodable_block::transformer::TransformerState<B>,
    token_ids: &[u32],
    vocab_size: usize,
) -> Vec<f32> {
    let length = token_ids.len() as u32;
    state.prepare(state.context_length(), length).expect("prepare state");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let token_buffer = command_buffer.allocate_constant_from_slice(token_ids).expect("token ids");
    let nodes = (0..length)
        .map(|index| TrieNode {
            trie_start: index,
            trie_end: length,
            height: index,
        })
        .collect::<Box<[_]>>();
    let batch_dim = BatchTopology::new(&nodes, true);
    let output = decoder
        .encode(&token_buffer, &batch_dim, Some((0..length).into()), None, state, &mut command_buffer)
        .expect("decoder encode");
    let logits = output.logits.expect("logits");
    let mut host_logits = context.create_buffer(token_ids.len() * vocab_size * size_of::<bf16>()).expect("buffer");
    command_buffer.encode_copy(&logits, &mut host_logits);
    state.encode_accept(&(0..length).collect::<Box<[u32]>>(), &mut command_buffer).expect("accept");
    drop((logits, output.hidden_features, output.final_hidden, token_buffer));
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("wait");
    let logits: Vec<bf16> = buffer_to_vec::<B, bf16>(&host_logits);
    assert_eq!(logits.len(), token_ids.len() * vocab_size);
    logits.into_iter().map(f32::from).collect()
}

fn run<B: Backend>(spec: &Spec) {
    let mut config: LanguageModelConfig =
        serde_json::from_reader(BufReader::new(File::open(spec.model.join("config.json")).unwrap())).unwrap();
    if let Some(recurrence) = config.decoder_config.transformer_config.recurrence.as_mut() {
        if let Some(num_loops) = spec.num_loops {
            recurrence.num_loops = num_loops;
        }
        if let Some(kv_cache_mode) = &spec.kv_cache_mode {
            recurrence.kv_cache_mode = kv_cache_mode.clone();
        }
    }
    let vocab_size = config.decoder_config.vocab_size as usize;
    let prompts: Vec<Prompt> = serde_json::from_reader(BufReader::new(File::open(&spec.prompts).unwrap())).unwrap();
    std::fs::create_dir_all(&spec.out_dir).unwrap();

    let context = block_on(B::Context::new()).expect("context");
    let started = Instant::now();
    let weights_file = File::open(spec.model.join("model.safetensors")).unwrap();
    let loader = ParameterLoader::<B>::new(&weights_file, &*context).unwrap();
    let decoder = Decoder::new(DataType::BF16, &config.decoder_config, &loader.tree().subtree("decoder"), &context)
        .expect("decoder");
    loader.assert_all_tensors_read().expect("all tensors read");
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    for prompt in prompts.iter().filter(|prompt| spec.only.as_ref().is_none_or(|only| only.contains(&prompt.id))) {
        let started = Instant::now();
        let mut state = decoder.create_empty_state(Some(4096), &context).expect("state");
        // The attention KV suffix holds at most ATTENTION_SUFFIX_CAPACITY tokens, so prefill in chunks like the stream.
        let prefill = prompt
            .ids
            .chunks(ATTENTION_SUFFIX_CAPACITY as usize)
            .flat_map(|chunk| run_step(&*context, &decoder, &mut state, chunk, vocab_size))
            .collect::<Vec<_>>();
        write_f32(&spec.out_dir.join(format!("{}_prefill.f32", prompt.id)), &prefill);
        let prefill_seconds = started.elapsed().as_secs_f32();
        let decode = prompt
            .decode_ids
            .iter()
            .flat_map(|&token| run_step(&*context, &decoder, &mut state, &[token], vocab_size))
            .collect::<Vec<_>>();
        write_f32(&spec.out_dir.join(format!("{}_decode.f32", prompt.id)), &decode);
        eprintln!(
            "{}: prefill {} tokens {:.1}s, decode {} tokens {:.1}s",
            prompt.id,
            prompt.ids.len(),
            prefill_seconds,
            prompt.decode_ids.len(),
            started.elapsed().as_secs_f32() - prefill_seconds,
        );
    }
}

#[uzu_test]
fn dory_parity_dump() {
    let Ok(spec_path) = std::env::var("UZU_DORY_PARITY") else {
        return;
    };
    let spec: Spec = serde_json::from_reader(BufReader::new(File::open(spec_path).unwrap())).unwrap();
    match spec.backend.as_str() {
        "cpu" => run::<crate::backends::cpu::Cpu>(&spec),
        #[cfg(backend = "metal")]
        "metal" => run::<crate::backends::metal::Metal>(&spec),
        backend => panic!("unsupported backend {backend}"),
    }
}
