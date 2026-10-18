# DRAFT — not posted. Proposed issue for trymirai/lalamo (cc trymirai/uzu)

**Title:** Proposal: config schema for NVIDIA Dory 1.0-4B (looped nGPT) support

Hi Mirai team — we plan to add support for
[nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
to lalamo and uzu. We could not find related issues or PRs. Before we write code, we want to agree on the
config schema, because uzu parses with `deny_unknown_fields` and every lalamo field must also exist in Rust.
Is anyone already working on this?

Reference implementation: vLLM PR #60149 (`dory.py`, commit `5507695`). We ported it to a CPU PyTorch
script and checked it against the Dory llama.cpp fork (same greedy first tokens).

## What Dory needs

- 72 physical layers. Each layer has **attention only or MLP only**.
- Layers 16..55 are a shared recurrent block, run `n_recurrent_loops` times (default 2). Each loop starts
  with `justnorm(state + input_h)`, and the loop ends with `state = justnorm(h)`.
- The residual is an nGPT interpolation: `justnorm(n(a) + w * (n(b) - n(a)))` with a per-channel vector `w`.
- `justnorm(x) = x * rsqrt(sum(x^2) + 1e-6)`.
- Attention: RoPE, then `justnorm` on q and k, then q times a per-channel vector `[heads * head_dim]`.
  The output is multiplied by a sigmoid gate, then `justnorm` over all heads, then `o_proj`.
- MLP: SwiGLU, then `justnorm` before `down_proj`.
- No final norm. Logits are multiplied by a per-vocab vector.
- KV cache: `per_loop` (one cache for each loop) or `last_loop` (each loop overwrites the same cache).

## What we can fold in the converter (no schema change)

| Dory op | Fold |
|---|---|
| `justnorm` | RMSNorm with `eps/n` and scale vector `1/sqrt(n)`. This is exactly equal. |
| MLP `suv_gate`, `suv_up` (× sqrt(d)) | Multiply the rows of `gate_proj` and `up_proj`. |
| Gate scale `(s_gate * f)^2` | Multiply the rows of `g_proj`. |
| `logit_scale / initializer_range` | Multiply the rows of `lm_head`. |
| `abs(alpha * alpha_init / std)` | Store the result vector. |
| k `justnorm` after RoPE | Same as before RoPE (L2 norm does not change under rotation). |

## Proposed new fields

1. `TransformerConfig.output_norm_config: NormalizationConfig | None` (Dory has no final norm).
2. `TransformerConfig.recurrence: RecurrenceConfig | None`
   ```
   RecurrenceConfig { start_layer: 16, end_layer: 56, num_loops: 2,
                      kv_cache_mode: "per_loop" | "last_loop" }
   ```
   The reinjection (`justnorm(state + input_h)`, `justnorm` at loop end) is fixed for this type. Weights are
   stored once. `num_loops` can be changed at load time.
3. A new layer config type (tagged, next to `TransformerLayerConfig`):
   ```
   NGPTLayerConfig { sublayer: AttentionConfig | MLPConfig,
                     residual_norm_config: NormalizationConfig }   # weight: alpha [model_dim]
   ```
   We think a separate type is cleaner than optional `mixer`/`mlp` fields on `TransformerLayerConfig`,
   because uzu fuses the residual add into the pre-norm kernel.
4. `AttentionConfig`: `post_rope_query_scale: bool` (weight `[heads * head_dim]`; it cannot be folded because
   the pairs `i, i + head_dim/2` have different values), and `output_norm_config: NormalizationConfig | None`
   (after the gate, before `o_proj`).
5. `DenseMLPConfig.hidden_norm_config: NormalizationConfig | None` (before `down_proj`).

Questions:
- Do you prefer `NGPTLayerConfig` or optional sublayers on `TransformerLayerConfig`?
- Are the names OK? Is there an existing pattern for recurrence (e.g. from DFlash or Gemma 4 KV sharing)?
- For uzu, should the state index be the execution step (loop × layer), and the physical index stay for weights?
