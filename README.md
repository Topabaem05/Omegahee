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
- `tests/test_omegahee_spec_extra.py` — spec-conformance tests from the TeX source.
- `tests/test_omegahee_metal_reference.py` — Metal-pack reference semantics (bit packing, prefix budgets, correlation bounds).
- `src/omegahee_metal_reference.py` — numpy twin of the Metal kernels; also the vector generator's source of truth.
- `metal/` — Metal kernel pack and Swift parity/bandwidth harness (see `metal/README.md`).
- `paper/` — LaTeX specification and the compiled 14-page PDF. Source and PDF now agree: the CI/CL table carries all 16 pillar-closure rows. Rebuild with `tectonic` after installing the Noto CJK fonts.
- `docs/plan_completion.md` — equation-to-function checklist.
- `docs/design_omega_uzu_integration.md` — paper-to-uzu integration design with the measured pass-cost model and next steps.
- `docs/problem_report_20261017.md` — open blockers with measurements, and the questions
  that need a decision before more kernel work.
- `docs/uzu_metal_kernel_mapping.md` — SIMD-group contract for the uzu crates.
- `.github/workflows/ci_cl.yml` — unittest on `main` and `feat/omega-uzu-v2`.

## Check

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/omegahee_uzu_engine.py
```

38 tests, all required to pass. Latest local run: LittleBit-2 L2L loss −99.95% with 98.44% top-1 agreement; pillar audit traffic reduction 48.39%; single-pass roofline factor 2.391×; GDN chain match; OptR softmax error ~1e-16; Metal parity 4/4.

On an Apple Silicon machine the device layer also runs:

```bash
bash metal/run.sh        # Metal parity + measured bandwidth, writes metal/build/report.json
python metal/tools/roofline.py   # turns the measured bandwidth into paper accounting
```

The recompiled PDF in `paper/` now matches the `.tex` source: 14 pages and all 16
verification rows. Rebuild it with `tectonic -X compile paper/uzu_omega_mathematical_algorithm_2026.tex`
after installing the Noto CJK fonts; the source needs `bm` and `enumitem` in addition
to the original package list.

## Performance versus uzu

`docs/performance_comparison.md` records what is measured and what is not. The
short version: Omega-Uzu has no end-to-end decode path, so no tokens/s comparison
exists. At matched shape (N=12288, K=4096) the KSC GEMV reaches 51.6 GB/s against
uzu's measured 261-354 GB/s of weight traffic, a 5.1-6.9x gap. The cause is
arithmetic intensity, not the bit format: the GEMV makes about 2.7 MAC per code
byte and reaches 13% of the device peak, while uzu's tiled trellis GEMM feeds an
MMA tile from one load and reaches 69-93%.

```bash
bash metal/run.sh                 # parity, device streaming, KSC throughput, uzu-shape GEMV
python metal/tools/roofline.py    # paper accounting on the measured bandwidth
```

## Design work


`docs/design_omega_uzu_integration.md` turns the paper's Phase A/B spec into a
measured plan. Two tools back it:

```bash
python tools/spec_budget_sim.py     # entropy controller vs the fixed uzu budget
python tools/spec_budget_sweep.py   # budget sweep against measured pass costs
```

The pass-cost model is fitted to this machine's measurements: a single-token pass
takes 18.19 ms and a 16-slot tree verify takes 47.40 ms, so the fit is 16.10 ms
fixed plus 2.086 ms per draft candidate. On that model the optimal budget is 3--4,
not the shipped 16, worth 1.77--2.12x tokens/s.
