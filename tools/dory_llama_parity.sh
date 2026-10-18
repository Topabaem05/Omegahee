#!/bin/zsh
# Greedy text parity between two Dory llama.cpp builds: loops 1/2/4, 3 prompts, 96 tokens.
# Usage: BUILDS="build-ref build-m5" tools/dory_llama_parity.sh  (exit 0 = all match)
R=${DORY_LLAMA:-/Users/topamini/code/uzu-omega/llama.cpp-dory}
M=${DORY_MODEL:-/Users/topamini/models/dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf}
OUT=${PARITY_OUT:-${TMPDIR:-/tmp}/dory-parity}
mkdir -p $OUT
prompts=("Hello" "What is 2+2? Explain briefly." "Write a Python function that returns the greatest common divisor of two positive integers.")
fail=0
builds=(${=BUILDS:-build-ref build-m5})
for loops in 1 2 4; do
  for i in 1 2 3; do
    for b in $builds; do
      $R/$b/bin/llama-completion -m $M -p "${prompts[$i]}" -n 96 --temp 0 -c 4096 -fa on -no-cnv \
        --no-display-prompt --ignore-eos --no-warmup \
        --override-kv "dory.num_loops=int:$loops" </dev/null > $OUT/$b.l$loops.p$i.txt 2>/dev/null
    done
    set -- $builds
    if cmp -s $OUT/$1.l$loops.p$i.txt $OUT/$2.l$loops.p$i.txt; then r=MATCH; else r=DIFF; fail=1; fi
    echo "loops=$loops prompt=$i $r bytes=$(wc -c < $OUT/$1.l$loops.p$i.txt)"
  done
done
exit $fail
