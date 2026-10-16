#!/usr/bin/env bash
# Run uzu (trymirai/uzu) benchmark on an Apple Silicon Mac (Metal backend) for comparison with
# bench/results_uzu_vs_llamacpp.json (measured on x86 Linux CPU) and bench/bench_mlx.py.
# Requirements: Xcode (Metal toolchain), rustup. Usage: bash bench/bench_uzu.sh [repo_id] [tokens] [runs]
set -euo pipefail
REPO_ID="${1:-trymirai/Qwen3.5-0.8B-M}"; TOKENS="${2:-128}"; RUNS="${3:-3}"
WORK="${UZU_WORK:-$HOME/uzu-bench}"; OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"; cd "$WORK"
[ -d uzu ] || git clone --depth 1 https://github.com/trymirai/uzu
cd uzu && cargo build --release -p cli
# download model (one short message triggers download), then locate it
./target/release/cli --model "$REPO_ID" -m "hi" --reasoning-effort disabled >/dev/null
MODEL_DIR="$(dirname "$(find "$HOME" -path '*mirai*' -name model.safetensors -newer Cargo.toml 2>/dev/null | head -1)")"
echo "model dir: $MODEL_DIR"
cat > "$WORK/task.json" <<JSON
{"identifier":"omegahee-cmp","repo_id":"$REPO_ID","number_of_runs":$RUNS,"tokens_limit":$TOKENS,"greedy":true,
 "reasoning_effort":"disabled","messages":[{"role":"user","content":"Write a short paragraph about the sea."}]}
JSON
./target/release/cli bench "$MODEL_DIR" "$WORK/task.json" "$OUT_DIR/results_uzu_mac.json"
sysctl -n machdep.cpu.brand_string hw.memsize
echo "wrote $OUT_DIR/results_uzu_mac.json"
