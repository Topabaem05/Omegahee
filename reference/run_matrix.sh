#!/bin/sh
# Stage-0 reference matrix: loops 1/2/4 x per_loop/last_loop x fp32/bf16,
# a repeated fp32 run for determinism, and the long (>1024 token) prompt.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
py="$here/../.venv-ref/bin/python"
model=${MODEL:-/Users/topamini/code/uzu-omega/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16}
out="$here/out/matrix"
short="hello math gcd ko think"
for dtype in float32 bfloat16; do
  for loops in 1 2 4; do
    for kv in per_loop last_loop; do
      "$py" -I "$here/dory_ref.py" --model "$model" --prompts "$here/prompts.json" --out-dir "$out" \
        --loops "$loops" --kv-mode "$kv" --dtype "$dtype" --decode 32 --only $short
      "$py" -I "$here/dory_ref.py" --model "$model" --prompts "$here/prompts.json" --out-dir "$out" \
        --loops "$loops" --kv-mode "$kv" --dtype "$dtype" --decode 16 --only long --no-trace --tag _long
    done
  done
done
for loops in 1 2 4; do
  for kv in per_loop last_loop; do
    "$py" -I "$here/dory_ref.py" --model "$model" --prompts "$here/prompts.json" --out-dir "$out" \
      --loops "$loops" --kv-mode "$kv" --dtype float32 --decode 32 --only $short --tag _r2
  done
done
echo MATRIX_DONE
