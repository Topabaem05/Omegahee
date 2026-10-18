#!/usr/bin/env bash
# Fast local 27B invocation with the two settings that matter on this Mac:
#   1. a small KV cache (the model default is 262,144 tokens = 16 GiB of KV)
#   2. the 27B MTP head for speculative decoding
set -euo pipefail

TARGET="${TARGET:-$HOME/models/qwen38-27b/Qwen3.8-27B-UD-IQ1_S.gguf}"
MTP="${MTP:-$HOME/models/qwen38-27b/mtp-Qwen3.8-27B-Q4_0.gguf}"
CTX="${CTX:-4096}"
NMAX="${NMAX:-4}"

if [[ ! -f "$TARGET" ]]; then
  echo "missing target: $TARGET" >&2
  exit 1
fi

ARGS=(-m "$TARGET" -c "$CTX")
if [[ "${MODE:-mtp}" == "mtp" ]]; then
  if [[ ! -f "$MTP" ]]; then
    echo "missing MTP head: $MTP" >&2
    exit 1
  fi
  ARGS+=(-md "$MTP" --spec-type draft-mtp --spec-draft-n-max "$NMAX" -ngld 99)
fi

exec llama-cli "${ARGS[@]}" "$@"
