# Uzu Research Workspace (post-reset)

This repository is a clean research workspace for improving `trymirai/uzu` on
Apple Silicon. The pre-reset Omega-Uzu implementation, its kernels, benchmarks,
and documents are archived at the Git tag
`archive/omega-pre-reset-20261018`. They are not part of the current baseline.

Start here:

- `docs/research_plan_20261018.md` — new uzu-first research plan and go/no-go gates.
- `docs/uzu_stabilization_20261018.md` — pinned uzu version, clean-environment
  protocol, contamination audit, and current blocked status.
- `tools/clean_env_check.py` — refuses a benchmark run when the GPU is busy,
  swap is high, disk is low, or another inference/training job is active.

Hard rule: **no benchmark number is current unless it was produced after
`tools/clean_env_check.py` exited 0.** Historical numbers are evidence about the
old environment, not a baseline for the new work.

## Pinned baseline

- uzu source commit: `5baa58b` (2026-10-17)
- release binary: `cli 0.6.9`
- binary SHA-256: `00e3365fe560915decbb47182e1699c6e6f7861537ed278a993b840864189f47`
- machine: Apple M5 Pro, 48GB, macOS 27.0
- machine-readable pin: `uzu.lock`

## Layout

```text
docs/      current plan and stabilization record
tools/     measurement gates and small reproducible harnesses
bench/     new results only; old results are in the archive tag
paper/     historical mathematical specification, not a current measurement
```

`paper/` is kept for reference. Its projected tok/s and the old 2.38bpw claims
are not accepted as current results until the new plan's quality and kernel
gates pass.

The current 4B reference is `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`
(Q5_K_M). It needs the Dory-enabled llama.cpp fork and is measured as a
separate reference, not as a uzu checkpoint.
