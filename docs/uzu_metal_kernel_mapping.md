# Ω-Uzu Metal SIMD-group mapping (uzu crates)

Reference pseudocode for the three integration points in the paper roadmap.
This is the kernel contract the Python reference already enforces numerically.
It is not a compiled `.metal` binary.

## 1. `quant_matmul` — trellis prologue + AWSRC epilogue

Crate: `crates/uzu/src/backends/metal/kernel/quant_matmul/`

Pack, per 64-column block, back-to-back (uzu #925):

- 2-bit native-basis codes from KSC-CurveTQ
- 14-bit start state `s0*`
- per-row scale `alpha_i`, per-column scale `beta_j` (fp16)
- AWSRC sidecar: `M_seed` of `(sigma_g,m : u16, gamma_g,m : bf16)` ≈ 0.162 bpw

```metal
// simdgroup, one output row tile
ushort s0 = header.s0 & 0x3FFF;          // 14-bit, no RHT unrotate
ushort state = s0;
float acc = 0.0;
for (ushort j = 0; j < 64; ++j) {
    ushort code = trellis_symbol(state, j);          // register shift
    float q = quantile_lut[code];                    // 128-entry constant
    acc += alpha * beta[j] * q * x[j];
    state = trellis_step(state, code);
}
// AWSRC: regenerate Phi(seed) in-register, never a shared LUT
for (ushort m = 0; m < M_SEED; ++m) {
    float phi = splitmix_sign(seeds[m], col) * rsqrt(float(block));
    acc += gamma[m] * phi * x[col];
}
```

Invariant already tested in Python: `rotation_applied == False`, and the curvature path cost is ≤ Euclidean rounding of the same codebook.

## 2. `session/speculator` — LiLiCorr + SparseSpec-L

Replace Weaver's per-slot autoregressive loop and the constant `tree_budget: 16`.

```text
if dflash_checkpoint_present and ram_budget_allows:
    marginals, U_out, U_in = dflash_single_pass(hidden)
    S_t[k, k'] = tau * cos(U_out[t, k], U_in[t+1, k'])
    tree = greedy_joint(marginals, S)          # lilicorr_joint_loglik
else:
    # zero extra bytes: read only the 2-bit StagQ prefix of the resident buffer
    draft_logits = forward(weights_prefix_2bit, kv_keep=0.15)
gamma, B_tree = sparsespec_controller(draft_probs)   # not shapes.json
```

Low-entropy drafts must receive a weakly larger `gamma*` than uniform drafts. The reference test locks `gamma*(peaked) > gamma*(uniform)`.

## 3. `attention` — ByteCross Split-K + ReadKV bit-plane load

Store K and V as 8 bit-planes, transposed so plane `p` of a channel is contiguous.

```text
if T < T_min:
    route = activation_only          # rho_act ≈ 0.72, do not launch KV sparsity
elif T < T_star:
    route = activation_primary
else:
    route = kv_sparse_splitk         # 4 sinks + recent window + top blocks
r_c^K = readkv_key_bits(|q_c|^2, sigma_k, B_read in [2, 4])
# load only planes 0 .. r_c^K-1
a = softmax(q_hat @ K_hat^T / sqrt(d))
r_tau^V = readkv_value_bits(a_tau * ||v_tau||, B_bar)
```

Decode (`T = 1` and tree width `< 16`) accumulates activations in BF16. Prefill and wide verify may use the INT8 neural-accelerator path. ODP streams prefill tiles through a ring of exactly two layers; the resident buffer is the 2.38 bpw decode checkpoint only.

## 4. GDN verify

```text
gather = static_parent_index(tree)          # fixed before the scan
S[v] = alpha[v] * S[parent[v]] * (I - beta[v] * k[v] k[v]^T) + beta[v] * v[v] k[v]^T
commit(S[accepted_leaf])                    # siblings are not published
```

The right-multiply matches a sequential scan of the accepted chain in float64 in the reference test. Pair with the existing Gumbel sampler clamped to `[2^{-24}, 1-2^{-24}]` on the top 24 bits.
