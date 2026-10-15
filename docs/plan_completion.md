# Plan completion — Ω-Uzu v2.0 (Omegahee)

Checked against `paper/uzu_omega_mathematical_algorithm_2026.tex` and the session log on 2026-10-16.

The GitHub repository previously contained only `LICENSE`. The bundle implemented LittleBit-2, ALoDLM, HybridLink, Dream-RSI, LoGRA, a ByteCross scalar, and a LiLiCorr greedy path. The four inference pillars were equations without kernels. Those kernels are now in `src/omegahee_pillars.py` and are gated by `tests/test_omegahee_pillars.py`.

| Plan item | Where | Status |
|---|---|---|
| LittleBit-2 Joint-ITQ + L2L QAT, 99.95% loss drop, 98.44% top-1 | `LittleBit2QATEngine` | reached |
| ALoDLM prelude / shared core / coda, 7 unique vs 22 eval, 84 KV pairs, prior E[d]=2.02 | `ALoDLMHybridARModel`, `DepthAwareKVCache` | reached |
| HybridLink residual adapter, cosine `</latent>` exit | `HybridLinkAdapter` (`W2` now zero-initialized) | reached |
| Continuous working memory concatenation | `ContinuousWorkingMemory` | filled |
| Two-pass HybridSFT = L_link + L_act | `hybrid_sft_two_pass` | filled |
| Dream-RSI monotonic score on the replay pool | `DreamRSILoGRAEngine.improve_policy_by_dreaming` | reached |
| LoGRA sketch + predicted-KL step | `consolidate_rsi_into_weights_logra` | reached |
| CASA closed-form bits, mean preserved, Q/K ≥ 3 | `casa_closed_form_bits`, `casa_discrete_assign` | filled |
| Schur complement + LDL curvature | `regularized_schur`, `ldl_factor` | filled |
| Rotation-free KSC-CurveTQ branch metric | `ksc_curvetq_quantize_row` (exact for block ≤ 6) | filled |
| 2.38075 bpw = 2 + 14/64 + 0.162, traffic −48.39% vs 4.613 | `mirai_s_effective_bpw` | filled |
| StagQ dyadic planes, monotone MSE | `stagq_dyadic_encode` | filled |
| AWSRC seed/scale sidecar, energy decreases | `awsrc_seeded_repair` | filled |
| DQ: INT8 prefill, BF16 decode, 2-layer ODP ring | `plan_disaggregated_quantization` | filled |
| ByteCross T* = 7633.5 and S_comp | `bytecross_crossover_length`, `bytecross_speedup` | filled |
| LiLiCorr joint surrogate log-likelihood | `lilicorr_joint_loglik` | filled |
| SparseSpec-L γ* and tree budget | `sparsespec_controller` | filled |
| GDN Tree-Scan, accepted-chain commit matches sequential scan | `gdn_tree_scan`, `gdn_commit_chain` | filled |
| 24-bit Gumbel clamp | `gumbel_clamp_uniforms` | filled |
| OptR centering + shared orthogonal U, softmax invariant | `optr_reparameterize` | filled |
| AATC reverse water-filling | `aatc_reverse_waterfill` | filled |
| ReadKV key/value prefix allocation | `readkv_key_bits`, `readkv_value_bits` | filled |
| Roofline single-pass 2.39× and 41.8% weight-byte fraction | `omega_roofline_factors` | filled |
| Metal SIMD-group mapping for uzu crates | `docs/uzu_metal_kernel_mapping.md` | spec filled, not a device binary |

Not claimed: a Metal/Rust port inside `trymirai/uzu`, or a measured TPS on Apple M5. Those remain integration targets. The numbers 148–310 tok/s in the paper are projections, not new device measurements.

Verification on this commit: `python -m unittest tests/test_omegahee_ci_cl.py tests/test_omegahee_pillars.py` — 13 tests, OK.
