#!/usr/bin/env python3
"""Put the FP4 codes of a ModelOpt NVFP4 checkpoint into a converted Ω-Hee (uzu) Dory model.

The base model (`--base`, from `tools/omega_convert.py`) supplies the config, embeddings, readout and
per-channel vectors; convert it from `tools/nvfp4_dequant.py` output so those tensors come from the
NVFP4 checkpoint too. Every transformer linear of the base is replaced, without re-quantizing, by:
  - `weights.weights`: the e2m1 codes, U8 [rows, columns / 2], two per byte, low nibble first
  - `weights.scales`: BF16 [groups, rows], the FP8 E4M3 block scale times the FP32 tensor scale
  - spec `MicrofloatSpec` (bits 4, group 16, nvfp4)
Fused base matrices take their rows from several checkpoint tensors (qkvg = q, k, v, g; up = up, gate),
each with its own tensor scale, which the BF16 group scales keep.

With `--fp8-scales` the block scales stay the checkpoint's E4M3 bytes (U8 [groups, rows], half the bytes
of BF16) and `weights.row_scales` (F32 [rows]) holds each row's tensor scale, so the values are exact.

    python3 tools/nvfp4_to_omega.py --base <omega model> --nvfp4 <nvfp4 checkout> --out <dir> [--fp8-scales]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_dequant import e4m3_to_float, load_raw, to_bf16_bits

BLOCK = 16
SPEC = {"type": "MicrofloatSpec", "bits": 4, "group_size": BLOCK, "scale_mode": "nvfp4", "layout": "output_input"}
SOURCES = {
    "mixer.qkvg_projection": ["mixer.q_proj", "mixer.k_proj", "mixer.v_proj", "mixer.g_proj"],
    "mixer.out_projection": ["mixer.o_proj"],
    "mlp.up_projection": ["mixer.up_proj", "mixer.gate_proj"],
    "mlp.down_projection": ["mixer.down_proj"],
}
LINEAR = re.compile(
    r"^decoder\.transformer\.layers\.(\d+)\."
    r"(mixer\.qkvg_projection|mixer\.out_projection|mlp\.up_projection|mlp\.down_projection)\.weights\."
)


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(size)), 8 + size


def fp4_matrix(
    nvfp4: dict, layer: int, sources: list[str], fp8_scales: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Codes [rows, columns / 2] and scales [groups, rows]: BF16 folded, or E4M3 bytes plus F32 row scales."""
    codes, scales, row_scales = [], [], []
    for source in sources:
        name = f"backbone.layers.{layer}.{source}.weight"
        packed = nvfp4[name][1]
        raw_scales = nvfp4[name + "_scale"][1].view(np.uint8)
        tensor_scale = np.float32(nvfp4[name + "_scale_2"][1])
        if raw_scales.shape != (packed.shape[0], packed.shape[1] * 2 // BLOCK):
            raise SystemExit(f"{name}: scale shape {raw_scales.shape} does not match codes {packed.shape}")
        codes.append(packed)
        if fp8_scales:
            scales.append(raw_scales)
            row_scales.append(np.full(packed.shape[0], tensor_scale, np.float32))
        else:
            scales.append(to_bf16_bits(e4m3_to_float(raw_scales) * tensor_scale))
    codes = np.ascontiguousarray(np.concatenate(codes, axis=0))
    scales = np.ascontiguousarray(np.concatenate(scales, axis=0).T)
    return codes, scales, np.concatenate(row_scales) if fp8_scales else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--nvfp4", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fp8-scales", action="store_true", help="keep E4M3 block scales and add F32 row scales")
    args = ap.parse_args()

    base_path = args.base / "model.safetensors"
    header, data_offset = read_header(base_path)
    metadata = dict(header.pop("__metadata__", {}))
    base_data = np.memmap(base_path, dtype=np.uint8, mode="r", offset=data_offset)
    nvfp4 = load_raw(args.nvfp4 / "model.safetensors")

    replaced: dict[str, tuple[str, list[int], bytes]] = {}
    prefixes = sorted({match.group(0)[: -len("weights.")] for name in header if (match := LINEAR.match(name))})
    for prefix in prefixes:
        layer, kind = LINEAR.match(prefix + "weights.").groups()
        codes, scales, row_scales = fp4_matrix(nvfp4, int(layer), SOURCES[kind], args.fp8_scales)
        old_codes = header[prefix + "weights.weights"]["shape"]
        rows = old_codes[0]
        if codes.shape[0] != rows:
            raise SystemExit(f"{prefix}: {codes.shape[0]} FP4 rows, base has {rows}")
        replaced[prefix + "weights.weights"] = ("U8", list(codes.shape), codes.tobytes())
        scale_dtype = "U8" if args.fp8_scales else "BF16"
        replaced[prefix + "weights.scales"] = (scale_dtype, list(scales.shape), scales.tobytes())
        if row_scales is not None:
            replaced[prefix + "weights.row_scales"] = ("F32", list(row_scales.shape), row_scales.tobytes())
        metadata[prefix + "weights.spec"] = json.dumps(SPEC)
    dropped = {
        name for name in header for prefix in prefixes if name.startswith(prefix + "weights.") and name not in replaced
    }

    names = sorted((set(header) - dropped) | set(replaced))
    out_header, blobs, offset = {}, [], 0
    for name in names:
        if name in replaced:
            dtype, shape, blob = replaced[name]
        else:
            info = header[name]
            start, end = info["data_offsets"]
            dtype, shape, blob = info["dtype"], info["shape"], base_data[start:end].tobytes()
        out_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    out_header["__metadata__"] = metadata
    encoded = json.dumps(out_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "model.safetensors"
    digest = hashlib.sha256()
    with out_path.open("wb") as file:
        for chunk in (struct.pack("<Q", len(encoded)), encoded, *blobs):
            file.write(chunk)
            digest.update(chunk)
    for path in args.base.iterdir():
        if path.is_file() and path.name not in {"model.safetensors", "conversion.json"}:
            shutil.copy(path, args.out / path.name)
    base_manifest = args.base / "conversion.json"
    base_conversion = json.loads(base_manifest.read_text()) if base_manifest.exists() else None
    manifest = {
        "base": str(args.base),
        "base_conversion": base_conversion,
        "nvfp4": str(args.nvfp4),
        "scales": "fp8_e4m3_with_f32_row_scales" if args.fp8_scales else "bf16_folded",
        "replaced_matrices": len(prefixes),
        "dropped_tensors": sorted(dropped)[:4] + (["..."] if len(dropped) > 4 else []),
        "output_sha256": digest.hexdigest(),
    }
    (args.out / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("replaced_matrices", "output_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
