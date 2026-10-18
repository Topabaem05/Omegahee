#!/bin/sh
# uzu Dory parity against the stage-0 reference for every loop count and KV mode.
set -u
here=$(cd "$(dirname "$0")" && pwd)
py="$here/../.venv-ref/bin/python"
backend=${1:-cpu}
out=${2:-$here/../bench/results/dory_uzu_${backend}_parity.json}
decode=${3:-31}
long_decode=${4:-16}
for loops in 2 1 4; do
  for kv in per_loop last_loop; do
    "$py" -I "$here/compare_uzu.py" --backend "$backend" --loops "$loops" --kv-mode "$kv" --decode "$decode" --out "$out"
  done
done
"$py" -I "$here/compare_uzu.py" --backend "$backend" --loops 2 --kv-mode per_loop --only long --decode "$long_decode" --out "$out"
echo UZU_MATRIX_DONE
