# Benchmarks

| Script | What | Output |
|---|---|---|
| `bench_toks.py` | numpy engine: ALoDLM decode (KV cache on/off), LittleBit-2 linear (fp32 vs factored) | `results_toks.json` |
| `bench_qwen_dims.py` | numpy linear paths at Qwen2.5-0.5B shapes (896 / 128 / 4864), batch 1 & 64, plus labelled 24-layer linear-only ESTIMATE | `results_qwen_dims.json` |
| llama.cpp `llama-bench` (see RESUME.md) | Qwen2.5-0.5B-Instruct Q4_K_M / Q8_0, pp512 / tg128, 8 threads | `results_llamacpp.json` |
| `bench_mlx.py` | Apple Silicon: mlx-lm generation + mlx.core port of ALoDLM decode & LittleBit-2 linear | `results_mlx.json` |
| `bench_uzu.sh` / uzu release binary | uzu 0.6.9 on this M5 Pro: greedy bench and the bundled DFlash speculator path | `results_uzu_mac.json` |

## results_uzu_mac.json (2026-10-17, Apple M5 Pro)

Measured with the prebuilt `mirai-0.6.9-aarch64-apple-darwin` release binary, so no
local Rust build is needed. The file records the greedy `bench` numbers, the
speculative conversational numbers, and a cross-check against the paper's
projection table.

| Path | Qwen3.5-9B | Qwen3.5-0.8B |
|---|---|---|
| greedy `bench` (speculation off) | 50.3--68.2 tok/s, 6.22 GB | 175.8--332.4 tok/s, 0.56 GB |
| conversational (DFlash speculator on) | 55.1--71.2 tok/s, 2.6--3.1 tok/forward, 5.79 GB | 318.6 tok/s (no speculator ships) |

Three observations worth carrying forward. First, `model.safetensors` for the 9B is
5,189,121,136 bytes, matching the paper's 5.19 GB assumption. Second, speculative
runs move only 109--119 GB/s of weights against a measured 270.9 GB/s peak, so 56--60%
of the bandwidth is idle during decode. Third, `speculator/shapes.json` pins
`tree_budget: 16` for every Apple chip including M5 Pro, and the 9B checkpoint ships a
755 MB DFlash draft model while the 0.8B ships none.

## bench_mlx.py (Apple Silicon only, untested here — syntax-checked on Linux)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm numpy
python bench/bench_mlx.py                    # GPU (default), includes mlx-lm run on mlx-community/Qwen2.5-0.5B-Instruct-4bit
python bench/bench_mlx.py --device cpu       # mlx CPU backend
python bench/bench_mlx.py --skip-mlx-lm      # only the ported engine benchmarks
```
Notes: the decode port keeps the control flow on the host (numpy), as in the original engine, so it is dominated by
small-kernel launch / sync overhead; compare it with `results_toks.json` as such. LittleBit-2 "factored" path is
unpacked float matmuls (no bit-packed kernel). Sign factors come from rank-r SVD (timing depends only on shapes).

## results_mlx.json (2026-10-17, Apple M5 Pro)

First run of `bench_mlx.py` on this machine (mlx 0.32.4, Python 3.15). The mlx-lm
generation leg used `mlx-community/Qwen2.5-0.5B-Instruct-4bit`: 512 prompt tokens at
11,057 tok/s and 128 generated at 319.9 tok/s, peak memory 0.76 GB on the GPU.
