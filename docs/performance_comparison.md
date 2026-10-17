# Omega-Uzu vs uzu: what the numbers actually compare

Written 2026-10-17 on Apple M5 Pro (48 GB, macOS 27.0, Xcode 27.1).
Every number below is either measured on this machine or explicitly labelled a
projection. Nothing here is a marketing comparison.

## The two things being compared

| | uzu (`trymirai/uzu` 0.6.9) | Omega-Uzu (this repository) |
|---|---|---|
| What it is | shipped Metal/Rust inference engine | Python + Metal reference for the paper's math |
| Quantization | Mirai-M 4-bit asymmetric in production | KSC-CurveTQ 2-bit + 14-bit header + AWSRC, in reference kernels |
| Runs real models | yes (`trymirai/Qwen3.5-9B-M` etc.) | no end-to-end model path |
| Metal kernels | tiled trellis GEMM with SIMD-group MMA | contract/parity kernels |

This asymmetry matters. uzu is a product; Omega-Uzu is a specification with a
validated kernel pack. A single "ours is faster" number cannot exist yet because
Omega-Uzu has no end-to-end decode path to time.

## What is measured, side by side

| Quantity | uzu 0.6.9 (measured) | Omega-Uzu (measured) |
|---|---|---|
| 9B decode, speculation off | 50.3-68.2 tok/s, 6.22 GB | not applicable (no model path) |
| 9B decode, DFlash speculator | 55.1-71.2 tok/s, 5.79 GB | not applicable |
| 0.8B decode | 175.8-332.4 tok/s, 0.56 GB | not applicable |
| Weight bytes per token | 5.19 GB at 4.613 bpw (file is 5,189,121,136 B) | 2.678 GB at 2.38075 bpw (computed from paper constants) |
| Kernel weight throughput | 261-354 GB/s effective during decode | 36.7 GB/s best, 36.2 median (reference kernel) |
| Device streaming peak | - | 376-409 GB/s (probe on this machine) |
| Parity vs its own reference | n/a | 4/4 kernels within tolerance |

## Why the reference kernel sits about 3x below uzu, and why it is not a verdict

The reference `ksc_throughput` kernel decodes one 64-column block row per thread
through a 64-step dependent loop, reaching 36.7 GB/s. uzu's decode path moves
261-354 GB/s of weights, so the reference sits about 3.0-3.2x below it, not the
7-10x an earlier probe of mine implied (that probe undercounted the device peak). uzu's `gemm_trellis` does three things the
reference does not:

1. **SIMD-group cooperation.** The trellis state advances cooperatively across a
   SIMD group, so 32 lanes share one row instead of one thread walking it. The
   dependency chain is spread, not serialized.
2. **Weights derived from a hash, not a table.** `decode_biased_weights` turns a
   code state into biased weights with integer multiplies and folds, which avoids
   the 128-entry quantile LUT the paper warns about (bank conflicts).
3. **Matrix tiling with MMA.** Codes feed a tiled GEMM (`Tile64x64x256`, etc.), so
   the kernel does many output rows per load and uses the neural accelerator.

So the honest statement is: the reference pack proves the bit layout and the
error bounds, and it deliberately does not attempt production throughput. Any
future Omega-Uzu speed claim has to be made against `gemm_trellis`, not against
the reference kernel.

## Where Omega-Uzu does have a measured edge

1. **Bit budget.** 2.38075 bpw is arithmetic from the paper and matches the
   repository's audit output. If the KSC codebook holds quality at that width, the
   weight stream per token falls from 5.19 GB to 2.678 GB, a 48.4% cut.
2. **No runtime un-rotation.** The reference decode applies no Hadamard or
   orthogonal multiply (`rotation_applied == false`), which is a cost uzu pays
   with its block-diagonal RHT.
3. **No draft model.** The paper's self-speculative path needs no DFlash
   checkpoint. Measured: the 9B checkpoint ships a 755 MB drafter while the 0.8B
   ships none, so the gap is real and quantifiable.

## Where uzu is clearly ahead

1. **It runs.** 9B at ~55 tok/s and 0.8B at ~200 tok/s, with real models, today.
2. **Speculation already works,** even if it underuses bandwidth (109-119 GB/s
   against a 376-409 GB/s peak, i.e. 27-32% of the device).
3. **The trellis GEMM is real and fast.** Omega-Uzu has no equivalent yet.

## What would make the comparison meaningful

Two measurements, in order:

1. **Port one KSC decode into uzu's GEMM shape** and time it against
   `gemm_trellis` on the same model and prompt. This is the only path to a
   defensible tok/s comparison.
2. **Measure quality at 2.38 bpw.** Perplexity or task accuracy on a real
   checkpoint. A traffic reduction that costs accuracy is not a win.

Until both exist, the supportable claim is:

> Omega-Uzu reduces the weight bytes per token by 48.4% on paper, with verified
> bit layout and error bounds, and no end-to-end throughput or quality result yet.

## Reproduce

```bash
bash metal/run.sh                 # parity + device streaming + KSC throughput
python metal/tools/roofline.py    # paper accounting on the measured bandwidth
python tools/spec_budget_sweep.py # tree budget vs measured pass costs
```

Numbers used above come from `metal/build/report.json`,
`bench/results_uzu_mac.json`, and `bench/results_mlx.json`.
