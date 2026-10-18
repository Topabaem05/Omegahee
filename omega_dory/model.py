"""Minimal NumPy reference for NVIDIA Dory 1.0-4B.

This is a conversion/reference adapter, not a production Metal engine. It
loads the Q5_K GGUF, dequantizes tensors to float16, and implements the
physical-layer graph described by the Dory-enabled llama.cpp fork.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np


def _dequant_tensor(reader, name: str) -> np.ndarray:
    tensor = next(t for t in reader.tensors if t.name == name)
    array = gguf.dequantize(tensor.data, tensor.tensor_type)
    return np.asarray(array, dtype=np.float16)


def _bf16_vector(reader, name: str) -> np.ndarray:
    return _dequant_tensor(reader, name).astype(np.float32)


def l2_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Dory's build_gdn_l2_norm: unit L2 norm, not RMS norm."""
    x = x.astype(np.float32, copy=False)
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Full-rotary RoPE on the last dimension, split-half layout."""
    n, d = x.shape
    half = d // 2
    inv = theta ** (-np.arange(half, dtype=np.float32) / half)
    angles = positions[:, None].astype(np.float32) * inv[None, :]
    c, s = np.cos(angles), np.sin(angles)
    x0, x1 = x[:, :half], x[:, half:]
    return np.concatenate((x0 * c - x1 * s, x0 * s + x1 * c), axis=-1)


def rope_heads(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Apply RoPE independently to each head of [tokens, heads, head_dim]."""
    tokens, heads, dim = x.shape
    flat = x.reshape(tokens * heads, dim)
    rotated = rope(flat, np.repeat(positions, heads), theta)
    return rotated.reshape(tokens, heads, dim)


@dataclass
class DoryConfig:
    n_embd: int = 2560
    n_ff: int = 7168
    n_head: int = 16
    n_head_kv: int = 4
    head_dim: int = 256
    n_input: int = 16
    n_recurrent: int = 40
    n_output: int = 16
    n_loops: int = 2
    n_swa: int = 1024
    init_std: float = 0.0177
    alpha_init: float = 0.05
    gate_init: float = 50.596443
    theta_1: float = 1e7
    theta_2: float = 1e4
    pattern: str = ""
    profile: list[int] | None = None

    @property
    def n_physical(self) -> int:
        return self.n_input + self.n_recurrent + self.n_output

    def physical(self) -> list[int]:
        return (
            list(range(self.n_input))
            + [self.n_input + i for _ in range(self.n_loops) for i in range(self.n_recurrent)]
            + [self.n_input + self.n_recurrent + i for i in range(self.n_output)]
        )


class DoryModel:
    def __init__(self, config: DoryConfig, weights: dict[str, np.ndarray]):
        self.config = config
        self.w = weights
        self.physical = config.physical()
        if len(config.pattern) != config.n_physical:
            raise ValueError(f"pattern length {len(config.pattern)} != {config.n_physical}")
        if config.profile is None or len(config.profile) != config.n_physical:
            raise ValueError("profile length does not match physical layers")

    def _mm(self, x: np.ndarray, name: str) -> np.ndarray:
        w = self.w[name]
        return x.astype(np.float32) @ w.astype(np.float32).T

    def _attention(self, cur: np.ndarray, p: int) -> np.ndarray:
        c = self.config
        tokens = cur.shape[0]
        q = self._mm(cur, f"blk.{p}.attn_q.weight")
        k = self._mm(cur, f"blk.{p}.attn_k.weight")
        v = self._mm(cur, f"blk.{p}.attn_v.weight")
        q = q.reshape(tokens, c.n_head, c.head_dim)
        k = k.reshape(tokens, c.n_head_kv, c.head_dim)
        v = v.reshape(tokens, c.n_head_kv, c.head_dim)

        profile = c.profile[p]
        if profile:
            theta = c.theta_2 if profile == 2 else c.theta_1
            positions = np.arange(tokens, dtype=np.float32)
            q = rope_heads(q, positions, theta)
            k = rope_heads(k, positions, theta)
        q = l2_norm(q)
        k = l2_norm(k)
        sq = _vector(self.w, f"blk.{p}.attn_qk_scale.weight")
        sq = (sq * math.sqrt(c.head_dim) / c.init_std) ** 2
        q = q * sq.reshape(1, c.n_head, c.head_dim)

        rep = c.n_head // c.n_head_kv
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        scores = np.einsum("thd,shd->hts", q, k) / math.sqrt(c.head_dim)
        rows = np.arange(tokens)[:, None]
        cols = np.arange(tokens)[None, :]
        mask = cols <= rows
        if c.pattern[p] == "S":
            mask &= cols > rows - c.n_swa
        scores = np.where(mask[None, :, :], scores, -1e30)
        attn = softmax(scores)
        out = np.einsum("hts,shd->thd", attn, v).reshape(tokens, c.n_head * c.head_dim)
        gate = self._mm(cur, f"blk.{p}.attn_gate.weight")
        sg = _vector(self.w, f"blk.{p}.attn_gate_scale.weight")
        sg = (sg * math.sqrt(c.gate_init) / c.init_std) ** 2
        gate = sigmoid(gate * sg[None, :])
        out = l2_norm(out * gate)
        return self._mm(out, f"blk.{p}.attn_output.weight")

    def _ffn(self, cur: np.ndarray, p: int) -> np.ndarray:
        c = self.config
        g = self._mm(cur, f"blk.{p}.ffn_gate.weight")
        u = self._mm(cur, f"blk.{p}.ffn_up.weight")
        g = g * (_vector(self.w, f"blk.{p}.ffn_gate_scale.weight") * math.sqrt(c.n_embd))
        u = u * (_vector(self.w, f"blk.{p}.ffn_up_scale.weight") * math.sqrt(c.n_embd))
        h = l2_norm(silu(g) * u)
        return self._mm(h, f"blk.{p}.ffn_down.weight")

    def forward(self, input_ids) -> np.ndarray:
        c = self.config
        ids = np.asarray(input_ids, dtype=np.int64)
        cur = self.w["token_embd.weight"][ids].astype(np.float32)
        input_h = None
        recurrent_end = c.n_input + c.n_recurrent * c.n_loops

        for il, p in enumerate(self.physical):
            if il == c.n_input:
                input_h = cur.copy()
            if c.n_input <= il < recurrent_end and (il - c.n_input) % c.n_recurrent == 0:
                cur = l2_norm(cur + input_h)

            residual = cur.copy()
            if c.pattern[p] == "-":
                cur = self._ffn(cur, p)
            else:
                cur = self._attention(cur, p)

            a = l2_norm(residual)
            b = l2_norm(cur)
            alpha = np.abs(_vector(self.w, f"blk.{p}.residual_alpha.weight") * (c.alpha_init / c.init_std))
            cur = l2_norm(a + (b - a) * alpha[None, :])
            if c.n_input <= il < recurrent_end and (il + 1 - c.n_input) % c.n_recurrent == 0:
                cur = l2_norm(cur)

        logits = self._mm(cur[-1], "output.weight")
        logits = logits * (_vector(self.w, "output_scale.weight") / c.init_std)
        return logits

    def forward_token(self, token_id: int) -> np.ndarray:
        return self.forward([int(token_id)])

    def generate(self, input_ids, max_new_tokens: int) -> list[int]:
        ids = [int(x) for x in input_ids]
        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            ids.append(int(np.argmax(logits)))
        return ids


def _vector(weights: dict[str, np.ndarray], name: str) -> np.ndarray:
    return weights[name].astype(np.float32).reshape(-1)


def load_dory_gguf(path: str, loops: int = 2) -> DoryModel:
    reader = gguf.GGUFReader(path)
    fields = {name: field.contents() for name, field in reader.fields.items()}
    n_input = int(fields["dory.input_layer_count"])
    n_recurrent = int(fields["dory.recurrent_layer_count"])
    n_output = int(fields["dory.output_layer_count"])
    n_loops = int(loops)
    config = DoryConfig(
        n_embd=int(fields["dory.embedding_length"]),
        n_ff=int(fields["dory.feed_forward_length"]),
        n_head=int(fields["dory.attention.head_count"]),
        n_head_kv=int(fields["dory.attention.head_count_kv"]),
        head_dim=int(fields["dory.attention.key_length"]),
        n_input=n_input,
        n_recurrent=n_recurrent,
        n_output=n_output,
        n_loops=n_loops,
        n_swa=int(fields["dory.attention.sliding_window"]),
        init_std=float(fields["dory.init_std"]),
        alpha_init=float(fields["dory.alpha_init"]),
        gate_init=float(fields["dory.gate_init"]),
        theta_1=float(fields["dory.rope.freq_base"]),
        theta_2=float(fields["dory.rope.freq_base_2"]),
        pattern=str(fields["dory.layer_pattern"]),
        profile=[int(x) for x in fields["dory.rope.profile"]],
    )
    weights = {}
    for tensor in reader.tensors:
        weights[tensor.name] = _dequant_tensor(reader, tensor.name)
    return DoryModel(config, weights)
