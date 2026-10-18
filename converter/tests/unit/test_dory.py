import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import torch
from jaxtyping import Array

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.dory_loader import load_dory_decoder
from lalamo.model_import.model_configs.huggingface import HFDoryConfig
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    EmbeddingForwardPassConfig,
    Keychain,
    RecurrentKVCacheMode,
    State,
)
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.helpers import make_test_sharding_config

FORWARD_PASS_CONFIG = DecoderForwardPassConfig(
    embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
)

TINY_DORY_CONFIG = {
    "model_type": "dory",
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 8,
    "intermediate_size": 24,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 32,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "sliding_window": 3,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 100, "partial_rotary_factor": 1.0},
        "full_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0},
    },
    "layer_types": ["sliding_attention", "mlp"] * 3 + ["full_attention", "mlp"],
    "ngpt_alpha_init": 0.05,
    "attention_output_gate_logit_scale_init": None,
    "mlp_bias": False,
    "attention_bias": False,
    "n_input_layers": 2,
    "n_recurrent_layers": 4,
    "n_output_layers": 2,
    "n_recurrent_loops": 3,
}


def _justnorm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)


@dataclass
class _ReferenceCache:
    window: int | None
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, scale: float) -> torch.Tensor:
        self.keys = k if self.keys is None else torch.cat((self.keys[:, :start], k), dim=1)
        self.values = v if self.values is None else torch.cat((self.values[:, :start], v), dim=1)
        key_positions = torch.arange(self.keys.shape[1])
        query_positions = torch.arange(start, start + q.shape[1])[:, None]
        mask = key_positions <= query_positions
        if self.window is not None:
            mask &= key_positions > query_positions - self.window
        return torch.nn.functional.scaled_dot_product_attention(
            q, self.keys, self.values, attn_mask=mask, scale=scale, enable_gqa=True
        )


class ReferenceDory:
    """The math of vLLM PR #60149 `dory.py` in float32 PyTorch, with the PR test's reference attention."""

    def __init__(self, config: dict, weights: Mapping[str, torch.Tensor], kv_cache_mode: RecurrentKVCacheMode) -> None:
        self.config = config
        self.weights = weights
        self.per_loop = kv_cache_mode == RecurrentKVCacheMode.PER_LOOP
        self.caches: dict[tuple[int, int], _ReferenceCache] = {}

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, layer_type: str) -> torch.Tensor:
        head_dim = self.config["head_dim"]
        theta = self.config["rope_parameters"][layer_type]["rope_theta"]
        inverse_frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = positions[:, None].float() * inverse_frequencies[None, :]
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def _attention(self, index: int, x: torch.Tensor, positions: torch.Tensor, loop: int) -> torch.Tensor:
        c, w = self.config, self.weights
        prefix = f"backbone.layers.{index}.mixer"
        heads, groups, head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        tokens = x.shape[0]
        layer_type = c["layer_types"][index]
        q = self._rope((x @ w[f"{prefix}.q_proj.weight"].T).view(tokens, heads, head_dim), positions, layer_type)
        k = self._rope((x @ w[f"{prefix}.k_proj.weight"].T).view(tokens, groups, head_dim), positions, layer_type)
        v = (x @ w[f"{prefix}.v_proj.weight"].T).view(tokens, groups, head_dim)
        query_scale = (w[f"{prefix}.sqk"] * head_dim**0.5 / c["initializer_range"]).view(heads, head_dim)
        q = query_scale.square() * _justnorm(q)
        k = _justnorm(k)
        window = c["sliding_window"] if layer_type == "sliding_attention" else None
        cache = self.caches.setdefault((index, loop if self.per_loop else 0), _ReferenceCache(window))
        out = cache.attend(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            int(positions[0]),
            1 / math.sqrt(head_dim),
        )
        out = out.transpose(0, 1).reshape(tokens, heads * head_dim)
        gate_scale = (w[f"{prefix}.s_gate"] * c["hidden_size"] ** 0.25 / c["initializer_range"]).square()
        out = _justnorm(out * torch.sigmoid((x @ w[f"{prefix}.g_proj.weight"].T) * gate_scale))
        return out @ w[f"{prefix}.o_proj.weight"].T

    def _mlp(self, index: int, x: torch.Tensor) -> torch.Tensor:
        w = self.weights
        prefix = f"backbone.layers.{index}.mixer"
        factor = self.config["hidden_size"] ** 0.5
        gate = (x @ w[f"{prefix}.gate_proj.weight"].T) * (w[f"{prefix}.suv_gate"] * factor)
        up = (x @ w[f"{prefix}.up_proj.weight"].T) * (w[f"{prefix}.suv_up"] * factor)
        return _justnorm(torch.nn.functional.silu(gate) * up) @ w[f"{prefix}.down_proj.weight"].T

    def _layer(self, index: int, x: torch.Tensor, positions: torch.Tensor, loop: int = 0) -> torch.Tensor:
        c = self.config
        prefix = f"backbone.layers.{index}"
        if c["layer_types"][index] == "mlp":
            out, alpha = self._mlp(index, x), self.weights[f"{prefix}.mlp_alpha"]
        else:
            out, alpha = self._attention(index, x, positions, loop), self.weights[f"{prefix}.attn_alpha"]
        interpolation = (alpha * c["ngpt_alpha_init"] / c["initializer_range"]).abs()
        normalized_inputs, normalized_outputs = _justnorm(x), _justnorm(out)
        return _justnorm(normalized_inputs + interpolation * (normalized_outputs - normalized_inputs))

    @torch.inference_mode()
    def __call__(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        c = self.config
        start, end = c["n_input_layers"], c["n_input_layers"] + c["n_recurrent_layers"]
        h = self.weights["backbone.embeddings.weight"][token_ids]
        for index in range(start):
            h = self._layer(index, h, positions)
        input_h = state = h
        for loop in range(c["n_recurrent_loops"]):
            h = _justnorm(state + input_h)
            for index in range(start, end):
                h = self._layer(index, h, positions, loop)
            state = _justnorm(h)
        h = state
        for index in range(end, c["num_hidden_layers"]):
            h = self._layer(index, h, positions)
        logits = h @ self.weights["lm_head.weight"].T
        return logits * (self.weights["lm_head.logit_scale"] / c["initializer_range"])


def _random_weights(config: dict) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    hidden, inner = config["hidden_size"], config["intermediate_size"]
    q_dim = config["num_attention_heads"] * config["head_dim"]
    kv_dim = config["num_key_value_heads"] * config["head_dim"]
    shapes = {
        "backbone.embeddings.weight": (config["vocab_size"], hidden),
        "lm_head.weight": (config["vocab_size"], hidden),
        "lm_head.logit_scale": (config["vocab_size"],),
    }
    for index, layer_type in enumerate(config["layer_types"]):
        prefix = f"backbone.layers.{index}"
        if layer_type == "mlp":
            shapes |= {
                f"{prefix}.mixer.gate_proj.weight": (inner, hidden),
                f"{prefix}.mixer.up_proj.weight": (inner, hidden),
                f"{prefix}.mixer.down_proj.weight": (hidden, inner),
                f"{prefix}.mixer.suv_gate": (inner,),
                f"{prefix}.mixer.suv_up": (inner,),
                f"{prefix}.mlp_alpha": (hidden,),
            }
        else:
            shapes |= {
                f"{prefix}.mixer.q_proj.weight": (q_dim, hidden),
                f"{prefix}.mixer.k_proj.weight": (kv_dim, hidden),
                f"{prefix}.mixer.v_proj.weight": (kv_dim, hidden),
                f"{prefix}.mixer.g_proj.weight": (q_dim, hidden),
                f"{prefix}.mixer.o_proj.weight": (hidden, q_dim),
                f"{prefix}.mixer.sqk": (q_dim,),
                f"{prefix}.mixer.s_gate": (q_dim,),
                f"{prefix}.attn_alpha": (hidden,),
            }
    return {name: torch.randn(shape, generator=generator) * 0.02 for name, shape in shapes.items()}


class RecordingMapping(Mapping[str, Array]):
    def __init__(self, arrays: Mapping[str, Array]) -> None:
        self.arrays = arrays
        self.read: set[str] = set()

    def __getitem__(self, key: str) -> Array:
        result = self.arrays[key]
        self.read.add(key)
        return result

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def __len__(self) -> int:
        return len(self.arrays)


def _decoder_config(tmp_path: Path, **overrides: object) -> DecoderConfig:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(TINY_DORY_CONFIG | overrides))
    return HFDoryConfig.from_json(config_path).to_decoder_config({})


def _build(
    tmp_path: Path,
    weights: Mapping[str, torch.Tensor],
    **overrides: object,
) -> tuple[Decoder, RecordingMapping]:
    decoder = _decoder_config(tmp_path, **overrides).init(
        EmptyInitializer(default_dtype=jnp.float32, sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8])),
    )
    weights_dict = RecordingMapping({ParameterPath(name): torch_to_jax(tensor) for name, tensor in weights.items()})
    return load_dory_decoder(decoder, weights_dict), weights_dict


def _run_chunks(
    decoder: Decoder,
    token_ids: torch.Tensor,
    chunks: tuple[tuple[int, int], ...],
) -> list[Array]:
    state: State | None = None
    keychain = Keychain.init(0, sharding_config=make_test_sharding_config())
    results = []
    for start, end in chunks:
        result = decoder(
            torch_to_jax(token_ids[None, start:end].int()),
            jnp.arange(start, end, dtype=jnp.int32)[None, :],
            state=state,
            return_updated_state=True,
            forward_pass_config=FORWARD_PASS_CONFIG,
            keychain=keychain,
        )
        state = result.updated_state
        results.append(result.logits[0])
    return results


@pytest.mark.parametrize("kv_cache_mode", list(RecurrentKVCacheMode))
@pytest.mark.parametrize("num_loops", [1, 3])
def test_dory_matches_reference_through_prefill_and_decode(
    tmp_path: Path,
    kv_cache_mode: RecurrentKVCacheMode,
    num_loops: int,
) -> None:
    overrides = {"n_recurrent_loops": num_loops, "recurrent_kv_cache_mode": kv_cache_mode.value}
    weights = _random_weights(TINY_DORY_CONFIG)
    decoder, weights_dict = _build(tmp_path, weights, **overrides)
    assert weights_dict.read == set(weights_dict.arrays), "every checkpoint tensor must be loaded"

    reference = ReferenceDory(TINY_DORY_CONFIG | overrides, weights, kv_cache_mode)
    vocab_size = 32
    token_ids = torch.arange(7) * 5 % vocab_size
    # Prefill chunk, single-token decode, and history beyond the sliding window.
    chunks = ((0, 3), (3, 4), (4, 7))
    for (start, end), logits in zip(chunks, _run_chunks(decoder, token_ids, chunks), strict=True):
        assert_close(
            result=logits,
            reference=torch_to_jax(reference(token_ids[start:end], torch.arange(start, end))),
            atol=1e-4,
            rtol=1e-4,
            operation_name=f"logits for tokens [{start}, {end})",
        )


def test_loop_count_does_not_duplicate_weights(tmp_path: Path) -> None:
    single_loop = _decoder_config(tmp_path, n_recurrent_loops=1)
    many_loops = _decoder_config(tmp_path, n_recurrent_loops=27)
    assert single_loop.transformer_config.layer_configs == many_loops.transformer_config.layer_configs
    assert len(many_loops.transformer_config.state_slots) == 2 + 2 * 27
    last_loop = _decoder_config(tmp_path, n_recurrent_loops=27, recurrent_kv_cache_mode="last_loop")
    assert len(last_loop.transformer_config.state_slots) == 4


def test_dory_config_roundtrips_through_json(tmp_path: Path) -> None:
    config = _decoder_config(tmp_path)
    assert DecoderConfig.from_json(config.to_json()) == config
