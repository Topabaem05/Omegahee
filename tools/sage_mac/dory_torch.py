"""Dory 1.0 forward pass in PyTorch on any device (MPS on a Mac), for quantization calibration.

Follows `reference/dory_ref.py` (a port of vLLM PR #60149) for one full sequence at a time (prefill, no KV
cache across calls). Linear weights can be replaced (`set_linear`) and every linear's input can be observed
(`observer(layer, name, loop, x)`), which collects calibration statistics and block inputs.

Linear names per layer: attention layers `q_proj`, `k_proj`, `v_proj`, `g_proj` (one shared input) and `o_proj`;
MLP layers `gate_proj`, `up_proj` (one shared input) and `down_proj`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-6
ATTENTION_LINEARS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")


def justnorm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(dim=-1, keepdim=True) + EPS)).to(x.dtype)


def slerp_norm(a: torch.Tensor, b: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)


class DoryTorch:
    def __init__(self, model_dir: str | Path, device: str = "mps", dtype: torch.dtype = torch.bfloat16):
        cfg = json.loads(Path(model_dir, "config.json").read_text())
        self.cfg, self.device, self.dtype = cfg, torch.device(device), dtype
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.hidden, self.std = cfg["hidden_size"], cfg["initializer_range"]
        self.n_in, self.n_rec, self.n_layers = (
            cfg["n_input_layers"],
            cfg["n_recurrent_layers"],
            cfg["num_hidden_layers"],
        )
        self.loops = cfg["n_recurrent_loops"]
        self.layer_types = cfg["layer_types"]
        self.sqk_factor = self.hd**0.5 / self.std
        gate_init = cfg.get("attention_output_gate_logit_scale_init") or self.hidden**0.5
        self.s_gate_factor = gate_init**0.5 / self.std
        self.suv_factor = self.hidden**0.5
        self.alpha_scale = cfg["ngpt_alpha_init"] / self.std
        self.inv_freq = {}
        for kind, rope in cfg["rope_parameters"].items():
            base = float(rope["rope_theta"])
            self.inv_freq[kind] = (1.0 / (base ** (torch.arange(0, self.hd, 2, dtype=torch.float) / self.hd))).to(
                self.device
            )
        self.w: dict[str, torch.Tensor] = {}
        with safe_open(str(Path(model_dir, "model.safetensors")), "pt") as f:
            for name in f.keys():
                self.w[name] = f.get_tensor(name).to(device=self.device, dtype=dtype)
        self.observer: Callable[[int, str, int, torch.Tensor], None] | None = None

    def linear_name(self, layer: int, name: str) -> str:
        return f"backbone.layers.{layer}.mixer.{name}.weight"

    def set_linear(self, layer: int, name: str, weight: torch.Tensor) -> None:
        self.w[self.linear_name(layer, name)] = weight.to(device=self.device, dtype=self.dtype)

    def linears(self) -> list[tuple[int, str]]:
        return [
            (layer, name)
            for layer, kind in enumerate(self.layer_types)
            for name in (MLP_LINEARS if kind == "mlp" else ATTENTION_LINEARS)
        ]

    def _observe(self, layer: int, name: str, loop: int, x: torch.Tensor) -> None:
        if self.observer is not None:
            self.observer(layer, name, loop, x)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, kind: str) -> torch.Tensor:
        t = x.shape[0]
        freqs = positions.float()[:, None] * self.inv_freq[kind][None, :]
        cos, sin = freqs.cos().to(x.dtype)[:, None, :], freqs.sin().to(x.dtype)[:, None, :]
        x = x.view(t, -1, self.hd)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).reshape(t, -1)

    def _attend(self, q, k, v, kind: str) -> torch.Tensor:
        t = q.shape[0]
        q = q.view(t, self.nh, self.hd).transpose(0, 1)
        k = k.view(t, self.nkv, self.hd).transpose(0, 1)
        v = v.view(t, self.nkv, self.hd).transpose(0, 1)
        position = torch.arange(t, device=self.device)
        mask = position[None, :] <= position[:, None]
        if kind == "sliding_attention":
            mask &= position[None, :] > position[:, None] - self.cfg["sliding_window"]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(self.hd), enable_gqa=True)
        return out.transpose(0, 1).reshape(t, self.nh * self.hd)

    def attention_mixer(self, i: int, x: torch.Tensor, positions: torch.Tensor, loop: int) -> torch.Tensor:
        p = f"backbone.layers.{i}.mixer"
        self._observe(i, "qkvg", loop, x)
        q = x @ self.w[f"{p}.q_proj.weight"].T
        k = x @ self.w[f"{p}.k_proj.weight"].T
        v = x @ self.w[f"{p}.v_proj.weight"].T
        gate = x @ self.w[f"{p}.g_proj.weight"].T
        kind = self.layer_types[i]
        q, k = self._rope(q, positions, kind), self._rope(k, positions, kind)
        t = q.shape[0]
        sqk = (self.w[f"{p}.sqk"].float() * self.sqk_factor).view(1, self.nh, self.hd)
        q = (sqk.square() * justnorm(q.view(t, self.nh, self.hd)).float()).to(q.dtype).view(t, -1)
        k = justnorm(k.view(t, self.nkv, self.hd)).view(t, -1)
        out = self._attend(q, k, v, kind)
        sg = (self.w[f"{p}.s_gate"].float() * self.s_gate_factor).square()
        out = justnorm((out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype))
        self._observe(i, "o_proj", loop, out)
        return out @ self.w[f"{p}.o_proj.weight"].T

    def mlp_mixer(self, i: int, x: torch.Tensor, loop: int) -> torch.Tensor:
        p = f"backbone.layers.{i}.mixer"
        self._observe(i, "gate_up", loop, x)
        g = x @ self.w[f"{p}.gate_proj.weight"].T
        u = x @ self.w[f"{p}.up_proj.weight"].T
        g = (g.float() * (self.w[f"{p}.suv_gate"].float() * self.suv_factor)).to(g.dtype)
        u = (u.float() * (self.w[f"{p}.suv_up"].float() * self.suv_factor)).to(u.dtype)
        m = justnorm((F.silu(g.float()) * u.float()).to(x.dtype))
        self._observe(i, "down_proj", loop, m)
        return m @ self.w[f"{p}.down_proj.weight"].T

    def layer(self, i: int, h: torch.Tensor, positions: torch.Tensor, loop: int = 0) -> torch.Tensor:
        if self.layer_types[i] != "mlp":
            out, alpha = self.attention_mixer(i, h, positions, loop), self.w[f"backbone.layers.{i}.attn_alpha"]
        else:
            out, alpha = self.mlp_mixer(i, h, loop), self.w[f"backbone.layers.{i}.mlp_alpha"]
        return slerp_norm(h, out, (alpha.float() * self.alpha_scale).abs())

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Final hidden states (before `lm_head`) for one sequence of token ids."""
        ids = ids.to(self.device)
        positions = torch.arange(ids.shape[0], device=self.device)
        h = self.w["backbone.embeddings.weight"][ids]
        for i in range(self.n_in):
            h = self.layer(i, h, positions)
        input_h = h
        state = input_h
        for loop in range(self.loops):
            h = justnorm(state + input_h)
            for i in range(self.n_in, self.n_in + self.n_rec):
                h = self.layer(i, h, positions, loop)
            state = justnorm(h)
        h = state
        for i in range(self.n_in + self.n_rec, self.n_layers):
            h = self.layer(i, h, positions)
        return h

    @torch.inference_mode()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return (h @ self.w["lm_head.weight"].T).float() * (self.w["lm_head.logit_scale"].float() / self.std)
