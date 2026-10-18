# Omega-Uzu vs uzu: what the numbers actually compare

> **2026-10-18 정정.** 아래 51.6GB/s는 초기 측정이다. 최신 파리티 통과
> `ksc_gemv_coalesced`는 23.4–26.3GB/s(중간값 24.6)이고, 기존 1바이트 커널은
> 13.2–15.0GB/s다. 격차는 5.1–6.9배가 아니라 10.6–14.4배다. 최신 분석은
> `docs/methodology_revision_20261018.md`와
> `docs/step3_gemv_byteorder_20261018.md`를 따른다.

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
| Kernel weight throughput | 261-354 GB/s effective during decode | 51.6 GB/s best at uzu shape (K=4096 GEMV) |
| Device streaming peak | - | 376-409 GB/s (probe on this machine) |
| Parity vs its own reference | n/a | 4/4 kernels within tolerance |

## The direct experiment: same shape, same weight stream

Two kernels were built and measured, both with Metal parity against the numpy
reference.

| Kernel | Shape | Weight throughput |
|---|---|---|
| scalar, one thread per row | K=64 | 25.8-38.7 GB/s |
| SIMD group per row, uzu-shaped | N=12288, K=4096 | **51.6 GB/s best, 47.5 median** |
| uzu decode (measured, speculation off) | same model | 261-354 GB/s |

The uzu-shaped kernel reads the same 14.16 MB of KSC weight bytes per GEMV, with
one SIMD group owning one output row and each lane owning two columns per 64-column
block. Parity holds at 1.43e-06, and the spread is tight (49.7, 51.6, 50.8 GB/s
best over three runs).

**The gap is 5.1-6.9x.** That is the first apples-to-apples number this repository
can state, and it does not favour Omega-Uzu.

## Why, and what it would take to close it

The bottleneck is arithmetic intensity, not the decode logic. The KSC GEMV
produces 64 MACs per 16 code bytes, about 2.7 MAC/byte. At batch 1 that is
bandwidth-bound, and the kernel reaches 13% of the ~380 GB/s float4 peak.

uzu's trellis GEMM is a GEMM, not a GEMV. One code load feeds a full SIMD-group
MMA tile (64x64x256 and larger), so the same bytes produce hundreds to thousands
of MACs, and the kernel reaches 69-93% of peak. The advantage is the tiling and
MMA path, not the quantization format.

Closing the gap therefore needs one of:

1. **A batched KSC GEMM.** Port the format into uzu's tile shapes so the same
   codes feed MMA fragments. This is the honest port and the only one that
   produces a comparable number at production batch.
2. **Fewer bytes per MAC.** The KSC format stores 2 bits per weight. If decode
   could feed MMA fragments directly instead of scalar LUT lookups, the 48.4%
   byte reduction would show up as throughput. That is what the paper assumes and
   what this pack has not built.

The claim that survives measurement is narrower than the paper's table:
Omega-Uzu's bit layout and error bounds are verified, and its decode is 5.1-6.9x
behind the shipped engine until a tiled port exists.


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
