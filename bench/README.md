# Benchmarks

| Script | What | Output |
|---|---|---|
| `bench_toks.py` | numpy engine: ALoDLM decode (KV cache on/off), LittleBit-2 linear (fp32 vs factored) | `results_toks.json` |
| `bench_qwen_dims.py` | numpy linear paths at Qwen2.5-0.5B shapes (896 / 128 / 4864), batch 1 & 64, plus labelled 24-layer linear-only ESTIMATE | `results_qwen_dims.json` |
| llama.cpp `llama-bench` (see RESUME.md) | Qwen2.5-0.5B-Instruct Q4_K_M / Q8_0, pp512 / tg128, 8 threads | `results_llamacpp.json` |
| `bench_mlx.py` | Apple Silicon: mlx-lm generation + mlx.core port of ALoDLM decode & LittleBit-2 linear | `results_mlx.json` |

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
