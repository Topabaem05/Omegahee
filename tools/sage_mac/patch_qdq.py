#!/usr/bin/env python3
"""Put QDQ weights (Hugging Face names, from `quantize_init.py`) into a converted Ω-Hee BF16 Dory model.

Fused engine matrices take their rows from several Hugging Face matrices (qkvg = q, k, v, g; up = up, gate).
`--qdq` is one safetensors file or a directory of them (`blockwise.py` writes one per layer).
`--layers` picks the groups to replace: `input` (0..15), `recurrent` (16..55), `output` (56..71) or `all`;
the other layers keep the BF16 weights. Other files of the base model are copied.

    python3 tools/sage_mac/patch_qdq.py --base <omega BF16 model> --qdq qdq_weights.safetensors \
        --layers recurrent --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np

SOURCES = {
    "mixer.qkvg_projection": ["q_proj", "k_proj", "v_proj", "g_proj"],
    "mixer.out_projection": ["o_proj"],
    "mlp.up_projection": ["up_proj", "gate_proj"],
    "mlp.down_projection": ["down_proj"],
}
TENSOR = re.compile(
    r"^decoder\.transformer\.layers\.(\d+)\.(mixer\.qkvg_projection|mixer\.out_projection|mlp\.up_projection|"
    r"mlp\.down_projection)\.weights\.weights$"
)
GROUPS = {"input": range(0, 16), "recurrent": range(16, 56), "output": range(56, 72), "all": range(0, 72)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--qdq", required=True, type=Path)
    ap.add_argument("--layers", required=True, choices=sorted(GROUPS))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    layers = set(GROUPS[args.layers])

    base_path = args.base / "model.safetensors"
    with base_path.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
    metadata = header.pop("__metadata__", {})
    data = np.memmap(base_path, dtype=np.uint8, mode="r", offset=8 + header_size)
    qdq_tensors = {}
    for qdq_path in sorted(args.qdq.glob("*.safetensors")) if args.qdq.is_dir() else [args.qdq]:
        with qdq_path.open("rb") as file:
            qdq_header_size = struct.unpack("<Q", file.read(8))[0]
            qdq_header = json.loads(file.read(qdq_header_size))
        qdq_header.pop("__metadata__", None)
        qdq_data = np.memmap(qdq_path, dtype=np.uint8, mode="r", offset=8 + qdq_header_size)
        for name, info in qdq_header.items():
            qdq_tensors[name] = (qdq_data, info["data_offsets"])

    def qdq_bytes(name: str) -> bytes:
        qdq_data, (start, end) = qdq_tensors[name]
        return qdq_data[start:end].tobytes()

    replaced = 0
    out_header, blobs, offset = {}, [], 0
    for name in sorted(header):
        info = header[name]
        start, end = info["data_offsets"]
        blob = data[start:end].tobytes()
        match = TENSOR.match(name)
        if match and int(match.group(1)) in layers:
            layer, kind = int(match.group(1)), match.group(2)
            parts = [qdq_bytes(f"backbone.layers.{layer}.mixer.{source}.weight") for source in SOURCES[kind]]
            fused = b"".join(parts)
            if info["dtype"] != "BF16" or len(fused) != len(blob):
                raise SystemExit(f"{name}: shape or dtype differs from the QDQ weights")
            blob = fused
            replaced += 1
        out_header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    out_header["__metadata__"] = metadata
    encoded = json.dumps(out_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "model.safetensors").open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for blob in blobs:
            file.write(blob)
    for path in args.base.iterdir():
        if path.is_file() and path.name not in {"model.safetensors", "conversion.json"}:
            shutil.copy(path, args.out / path.name)
    (args.out / "conversion.json").write_text(
        json.dumps(
            {"base": str(args.base), "qdq": str(args.qdq), "layers": args.layers, "replaced": replaced}, indent=2
        )
        + "\n"
    )
    print(f"replaced {replaced} matrices ({args.layers}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
