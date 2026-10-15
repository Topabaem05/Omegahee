# Omegahee (Ω-Uzu v2.0)

Reference implementation of the Ω-Uzu inference math, zero-loss QAT recovery, weight-shared looped AR, and Dream-RSI weight consolidation.

The numerical kernels are Python. They are the specification check for a later Metal port into [trymirai/uzu](https://github.com/trymirai/uzu), not that port itself. Projected tok/s figures in the paper are roofline arithmetic, not new M5 measurements.

## Layout

- `src/omegahee_uzu_engine.py` — LittleBit-2 Joint-ITQ and L2L QAT, ALoDLM (prelude / shared core / coda, depth-aware KV, NELBO surrogate), HybridLink, Dream-RSI + LoGRA, ByteCross length, LiLiCorr greedy path.
- `src/omegahee_pillars.py` — the four pillars that the paper specified and the first engine did not implement:
  - **Pillar 1.** CASA closed-form bits, Schur complement, LDL curvature, rotation-free KSC-CurveTQ search, StagQ planes, AWSRC seeds. Effective width `2 + 14/64 + 0.162 = 2.38075` bpw (paper rounds the header to 0.22 bpw → 2.382), 48.4% under Mirai-M 4.613 bpw.
  - **Pillar 2.** DQ-ODP phase plan (BF16 decode activations, 2-layer ODP ring) and ByteCross speedup. Crossover at 7,633.5 tokens for the paper's constants.
  - **Pillar 3.** LiLiCorr joint score, SparseSpec-L `γ*` / tree budget, GDN Tree-Scan with accepted-chain commit, 24-bit Gumbel clamp.
  - **Pillar 4.** OptR key centering (softmax unchanged), AATC reverse water-filling, ReadKV prefix allocation.
- `tests/test_omegahee_ci_cl.py` — original 6 invariant tests.
- `tests/test_omegahee_pillars.py` — pillar closure tests.
- `paper/` — LaTeX specification and the 13-page PDF of the October typeset. The `.tex` source contains the later verification rows; the PDF does not, until recompiled with XeLaTeX and a CJK font.
- `docs/plan_completion.md` — equation-to-function checklist.
- `docs/uzu_metal_kernel_mapping.md` — SIMD-group contract for the uzu crates.
- `.github/workflows/ci_cl.yml` — unittest on `main` and `feat/omega-uzu-v2`.

## Check

```bash
pip install -r requirements.txt
python -m unittest tests/test_omegahee_ci_cl.py tests/test_omegahee_pillars.py -v
python src/omegahee_uzu_engine.py
```

13 tests, all required to pass. Latest local run: LittleBit-2 L2L loss −99.95% with 98.44% top-1 agreement; pillar audit traffic reduction 48.39%; single-pass roofline factor 2.391×; GDN chain match; OptR softmax error ~1e-16.
