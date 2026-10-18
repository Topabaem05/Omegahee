#!/usr/bin/env python3
"""Extract the Dory GGUF metadata and tensor contract for an Omega adapter.

Requires the small `gguf` package. The repository keeps the model outside Git;
run this from the Dory venv, for example:

  /Users/topamini/models/dory/venv/bin/python tools/dory_gguf_manifest.py \\
    --model /Users/topamini/models/dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \\
    --out bench/results/dory_manifest_20261018.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader


LARGE_ARRAYS = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.scores",
}


def scalar(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [scalar(v) for v in value]
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-fields", action="store_true", help="include full tokenizer arrays")
    args = ap.parse_args()

    reader = GGUFReader(args.model)
    fields = {}
    for name, field in reader.fields.items():
        if name == "GGUF.version":
            continue
        value = scalar(field.contents())
        if not args.all_fields and name in LARGE_ARRAYS and isinstance(value, list):
            payload = json.dumps(value, separators=(",", ":")).encode()
            fields[name] = {
                "count": len(value),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "omitted": "large tokenizer array; use --all-fields to include",
            }
        else:
            fields[name] = value
    tensors = []
    for tensor in reader.tensors:
        tensors.append(
            {
                "name": tensor.name,
                "shape": [int(x) for x in tensor.shape],
                "type": getattr(tensor.tensor_type, "name", str(tensor.tensor_type)),
                "n_elements": int(np.prod(tensor.shape)),
            }
        )

    report = {
        "model": str(Path(args.model).resolve()),
        "size_bytes": Path(args.model).stat().st_size,
        "field_count": len(fields),
        "tensor_count": len(tensors),
        "fields": fields,
        "tensors": tensors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), "fields": len(fields), "tensors": len(tensors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
