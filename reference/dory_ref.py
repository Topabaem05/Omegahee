#!/usr/bin/env python3
"""Standalone CPU PyTorch reference for NVIDIA Dory 1.0-4B (BF16 safetensors).

This is a line-by-line port of vLLM PR #60149 `vllm/model_executor/models/dory.py`
at commit 5507695 (copy in `reference/vllm_pr60149_5507695/`). vLLM parallel
linears are plain matmuls and `Attention` is the PR test's `ReferenceAttention`
(single request, causal GQA, sliding window, KV cache addressed by start_pos).

Usage:
    python dory_ref.py --model DIR --out OUT.npz --loops 2 --kv-mode per_loop \
        --dtype float32 --decode 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-6
TOPK = 20


def justnorm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(dim=-1, keepdim=True) + EPS)).to(x.dtype)


def slerp_norm(a: torch.Tensor, b: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)


class KVCache:
    """ReferenceAttention from tests/models/test_dory.py."""

    def __init__(self, window: int | None):
        self.window = window
        self.keys = self.values = None

    def attend(self, q, k, v, start_pos: int, num_heads, num_kv_heads, head_dim, scale):
        q = q.view(-1, num_heads, head_dim).transpose(0, 1)
        k = k.view(-1, num_kv_heads, head_dim).transpose(0, 1)
        v = v.view(-1, num_kv_heads, head_dim).transpose(0, 1)
        offset = start_pos
        self.keys = k if self.keys is None else torch.cat((self.keys[:, :offset], k), dim=1)
        self.values = v if self.values is None else torch.cat((self.values[:, :offset], v), dim=1)
        key_pos = torch.arange(self.keys.shape[1])
        query_pos = torch.arange(offset, offset + q.shape[1])[:, None]
        mask = key_pos <= query_pos
        if self.window is not None:
            mask &= key_pos > query_pos - self.window
        out = F.scaled_dot_product_attention(
            q, self.keys, self.values, attn_mask=mask, scale=scale, enable_gqa=True
        )
        return out.transpose(0, 1).reshape(-1, num_heads * head_dim)


class Dory:
    def __init__(self, model_dir: str, dtype: torch.dtype, loops: int | None, kv_mode: str | None):
        cfg = json.loads(Path(model_dir, "config.json").read_text())
        if loops is not None:
            cfg["n_recurrent_loops"] = loops
        if kv_mode is not None:
            cfg["recurrent_kv_cache_mode"] = kv_mode
        cfg.setdefault("recurrent_kv_cache_mode", "per_loop")
        self.cfg = cfg
        self.dtype = dtype
        self.hidden = cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.std = cfg["initializer_range"]
        self.n_in = cfg["n_input_layers"]
        self.n_rec = cfg["n_recurrent_layers"]
        self.n_layers = cfg["num_hidden_layers"]
        self.loops = cfg["n_recurrent_loops"]
        self.per_loop = cfg["recurrent_kv_cache_mode"] == "per_loop"
        self.layer_types = cfg["layer_types"]
        self.sqk_factor = self.hd**0.5 / self.std
        g0 = cfg.get("attention_output_gate_logit_scale_init")
        if g0 is None:
            g0 = self.hidden**0.5
        self.s_gate_factor = g0**0.5 / self.std
        self.suv_factor = self.hidden**0.5
        self.alpha_scale = cfg["ngpt_alpha_init"] / self.std
        self.inv_freq = {}
        for kind, rope in cfg["rope_parameters"].items():
            rot = int(self.hd * rope.get("partial_rotary_factor", 1.0))
            assert rot == self.hd, "partial rotary not implemented"
            base = float(rope["rope_theta"])
            self.inv_freq[kind] = 1.0 / (base ** (torch.arange(0, rot, 2, dtype=torch.float) / rot))

        self.w: dict[str, torch.Tensor] = {}
        self.read: set[str] = set()
        with safe_open(str(Path(model_dir, "model.safetensors")), "pt") as f:
            for name in f.keys():
                self.w[name] = f.get_tensor(name).to(dtype)
        self.reset()

    def W(self, name: str) -> torch.Tensor:
        self.read.add(name)
        return self.w[name]

    def reset(self):
        self.caches: dict[tuple[int, int], KVCache] = {}

    def _cache(self, layer: int, loop_idx: int) -> KVCache:
        slot = loop_idx if self.per_loop else 0
        key = (layer, slot)
        if key not in self.caches:
            kind = self.layer_types[layer]
            window = self.cfg["sliding_window"] if kind == "sliding_attention" else None
            self.caches[key] = KVCache(window)
        return self.caches[key]

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, kind: str) -> torch.Tensor:
        # vLLM RotaryEmbedding.forward_native, neox style, cache cast to x dtype.
        t = x.shape[0]
        freqs = positions.float()[:, None] * self.inv_freq[kind][None, :]
        cos = freqs.cos().to(x.dtype)[:, None, :]
        sin = freqs.sin().to(x.dtype)[:, None, :]
        x = x.view(t, -1, self.hd)
        x1, x2 = x.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1).reshape(t, -1)

    def _attention(self, i: int, x: torch.Tensor, positions: torch.Tensor, loop_idx: int):
        p = f"backbone.layers.{i}.mixer"
        q = x @ self.W(f"{p}.q_proj.weight").T
        k = x @ self.W(f"{p}.k_proj.weight").T
        v = x @ self.W(f"{p}.v_proj.weight").T
        gate = x @ self.W(f"{p}.g_proj.weight").T
        kind = self.layer_types[i]
        q = self._rope(q, positions, kind)
        k = self._rope(k, positions, kind)
        t = q.shape[0]
        q = q.view(t, self.nh, self.hd)
        sqk = (self.W(f"{p}.sqk").float() * self.sqk_factor).view(1, self.nh, self.hd)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype).view(t, self.nh * self.hd)
        k = justnorm(k.view(t, self.nkv, self.hd)).view(t, self.nkv * self.hd)
        out = self._cache(i, loop_idx).attend(
            q, k, v, int(positions[0]), self.nh, self.nkv, self.hd, 1.0 / math.sqrt(self.hd)
        )
        sg = (self.W(f"{p}.s_gate").float() * self.s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        out = justnorm(out)
        return out @ self.W(f"{p}.o_proj.weight").T

    def _mlp(self, i: int, x: torch.Tensor):
        p = f"backbone.layers.{i}.mixer"
        g = x @ self.W(f"{p}.gate_proj.weight").T
        u = x @ self.W(f"{p}.up_proj.weight").T
        g = (g.float() * (self.W(f"{p}.suv_gate").float() * self.suv_factor)).to(g.dtype)
        u = (u.float() * (self.W(f"{p}.suv_up").float() * self.suv_factor)).to(u.dtype)
        m = (F.silu(g.float()) * u.float()).to(x.dtype)
        m = justnorm(m)
        return m @ self.W(f"{p}.down_proj.weight").T

    def _layer(self, i: int, h: torch.Tensor, positions: torch.Tensor, loop_idx: int = 0):
        if self.layer_types[i] != "mlp":
            out = self._attention(i, h, positions, loop_idx)
            alpha = self.W(f"backbone.layers.{i}.attn_alpha")
        else:
            out = self._mlp(i, h)
            alpha = self.W(f"backbone.layers.{i}.mlp_alpha")
        return slerp_norm(h, out, (alpha.float() * self.alpha_scale).abs())

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, positions: torch.Tensor, trace: dict | None = None):
        h = self.W("backbone.embeddings.weight")[ids]
        rec = lambda name, x: trace.__setitem__(name, x.float().numpy().copy()) if trace is not None else None
        rec("embed", h)
        step = 0
        for i in range(self.n_in):
            h = self._layer(i, h, positions)
            rec(f"x{step:03d}_p{i:02d}_l0", h)
            step += 1
        input_h = h
        state = input_h
        for loop in range(self.loops):
            h = justnorm(state + input_h)
            rec(f"loop{loop}_combined", h)
            for i in range(self.n_in, self.n_in + self.n_rec):
                h = self._layer(i, h, positions, loop_idx=loop)
                rec(f"x{step:03d}_p{i:02d}_l{loop}", h)
                step += 1
            state = justnorm(h)
            rec(f"loop{loop}_state", state)
        h = state
        for i in range(self.n_in + self.n_rec, self.n_layers):
            h = self._layer(i, h, positions)
            rec(f"x{step:03d}_p{i:02d}_l0", h)
            step += 1
        return h

    @torch.inference_mode()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = h @ self.W("lm_head.weight").T
        return logits.float() * (self.W("lm_head.logit_scale").float() / self.std)


def topk_logprobs(logits: torch.Tensor):
    lp = torch.log_softmax(logits.float(), dim=-1)
    vals, idx = lp.topk(TOPK, dim=-1)
    return idx.numpy().astype(np.int32), vals.numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSON list of {id, ids}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loops", type=int, required=True)
    ap.add_argument("--kv-mode", choices=["per_loop", "last_loop"], required=True)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--only", nargs="*", help="prompt ids to run")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    model = Dory(args.model, dtype, args.loops, args.kv_mode)
    load_s = time.time() - t0
    prompts = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for pr in prompts:
        if args.only and pr["id"] not in args.only:
            continue
        ids = list(pr["ids"])
        n = len(ids)
        model.reset()
        trace = None if args.no_trace else {}
        t1 = time.time()
        h = model.forward(torch.tensor(ids), torch.arange(n), trace)
        prefill_logits = model.logits(h)
        prefill_s = time.time() - t1
        p_idx, p_lp = topk_logprobs(prefill_logits)
        last_logits = prefill_logits[-1].numpy().copy()

        gen, d_idx, d_lp, d_logits = [], [], [], []
        nxt = int(prefill_logits[-1].argmax())
        t2 = time.time()
        for s in range(args.decode):
            gen.append(nxt)
            if s == args.decode - 1:
                break
            pos = n + s
            hd = model.forward(torch.tensor([nxt]), torch.tensor([pos]))
            lg = model.logits(hd)[0]
            i, v = topk_logprobs(lg)
            d_idx.append(i)
            d_lp.append(v)
            d_logits.append(lg.numpy().copy())
            nxt = int(lg.argmax())
        decode_s = time.time() - t2

        # Prefill/decode consistency: recompute the whole sequence without a cache.
        full = ids + gen[:-1]
        model.reset()
        hf = model.forward(torch.tensor(full), torch.arange(len(full)))
        full_logits = model.logits(hf)
        dec = np.stack([last_logits] + d_logits) if d_logits else last_logits[None]
        ref = full_logits[n - 1 :].numpy()
        recompute_maxabs = float(np.abs(dec - ref).max())
        recompute_argmax_eq = int((dec.argmax(-1) == ref.argmax(-1)).sum())

        name = f"{pr['id']}_L{args.loops}_{args.kv_mode}_{args.dtype}{args.tag}"
        arrays = {
            "prompt_ids": np.array(ids, dtype=np.int32),
            "gen_ids": np.array(gen, dtype=np.int32),
            "prefill_top_idx": p_idx,
            "prefill_top_lp": p_lp,
            "last_logits": last_logits.astype(np.float32),
            "decode_top_idx": np.array(d_idx, dtype=np.int32).reshape(-1, TOPK),
            "decode_top_lp": np.array(d_lp, dtype=np.float32).reshape(-1, TOPK),
            "decode_logits_argmax": dec.argmax(-1).astype(np.int32),
        }
        if trace is not None:
            arrays.update({f"trace/{k}": v for k, v in trace.items()})
        np.savez(out_dir / f"{name}.npz", **arrays)
        row = {
            "name": name,
            "prompt": pr["id"],
            "loops": args.loops,
            "kv_mode": args.kv_mode,
            "dtype": args.dtype,
            "prompt_tokens": n,
            "first_token": gen[0],
            "gen_ids": gen,
            "gen_sha1": hashlib.sha1(np.array(gen, dtype=np.int32).tobytes()).hexdigest(),
            "recompute_maxabs_logit": recompute_maxabs,
            "recompute_argmax_equal": f"{recompute_argmax_eq}/{len(dec)}",
            "kv_slots": len(model.caches),
            "prefill_s": round(prefill_s, 2),
            "decode_s": round(decode_s, 2),
        }
        summary.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "gen_ids"}), flush=True)

    unread = sorted(set(model.w) - model.read)
    meta = {"load_s": round(load_s, 1), "unread_tensors": unread, "runs": summary}
    tag = f"L{args.loops}_{args.kv_mode}_{args.dtype}{args.tag}"
    Path(out_dir, f"summary_{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"unread tensors: {len(unread)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
