#!/usr/bin/env python3
"""Dequantize a ModelOpt NVFP4 (W4A16) checkpoint into a BF16 checkout that lalamo can import.

NVFP4 stores two e2m1 codes per byte (`weight`, U8, low nibble first), one FP8 E4M3 scale per 16 values
(`weight_scale`) and one FP32 scale per tensor (`weight_scale_2`):
    w = e2m1(code) * float(weight_scale) * weight_scale_2
The other tensors are copied. `config.json` comes from `--config-from` (the BF16 checkout), because the
NVFP4 config carries a quantization section that importers for the BF16 layout do not expect.

    python3 tools/nvfp4_dequant.py --nvfp4 <nvfp4 checkout> --config-from <bf16 checkout> --out <dir>
    python3 tools/nvfp4_dequant.py --nvfp4 <nvfp4 checkout> --check <bf16 checkout>   # compare a few tensors
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
BLOCK = 16


def e4m3_to_float(raw: np.ndarray) -> np.ndarray:
    bits = raw.view(np.uint8).astype(np.int32)
    sign = np.where(bits & 0x80, -1.0, 1.0)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    normal = (1.0 + mantissa / 8.0) * np.exp2(exponent - 7.0)
    subnormal = (mantissa / 8.0) * np.exp2(-6.0)
    value = np.where(exponent == 0, subnormal, normal)
    value = np.where((exponent == 0xF) & (mantissa == 0x7), np.nan, value)
    return (sign * value).astype(np.float32)


def dequantize(packed: np.ndarray, scales: np.ndarray, global_scale: np.ndarray) -> np.ndarray:
    rows = packed.shape[0]
    codes = np.empty((rows, packed.shape[1] * 2), np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    values = E2M1[codes].reshape(rows, -1, BLOCK)
    block_scales = e4m3_to_float(scales)[:, :, None]
    return (values * block_scales * np.float32(global_scale)).reshape(rows, -1)


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    as_int = values.astype(np.float32).view(np.uint32)
    rounded = as_int + 0x7FFF + ((as_int >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def load_raw(path: Path) -> dict[str, np.ndarray]:
    # safetensors.numpy cannot express BF16/F8; read everything as raw bytes with the header shapes.
    import struct

    with path.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
        data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_size)
    dtypes = {"BF16": np.uint16, "F32": np.float32, "U8": np.uint8, "F8_E4M3": np.uint8, "I64": np.int64}
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors[name] = (
            info["dtype"],
            np.frombuffer(data[start:end], dtype=dtypes[info["dtype"]]).reshape(info["shape"]),
        )
    return tensors


def round_to_nvfp4(weights: np.ndarray) -> np.ndarray:
    """Round-to-nearest NVFP4 with an FP32 tensor scale and FP8 E4M3 block scales (absmax / 6)."""
    blocks = weights.reshape(weights.shape[0], -1, BLOCK)
    tensor_scale = np.abs(blocks).max() / (6.0 * 448.0)
    block_scales = np.abs(blocks).max(-1, keepdims=True) / 6.0 / tensor_scale
    block_scales = np.clip(block_scales, 2.0**-9, 448.0)
    # Round the block scales to E4M3 (3 mantissa bits).
    exponent = np.floor(np.log2(block_scales))
    block_scales = np.round(block_scales / np.exp2(exponent) * 8.0) / 8.0 * np.exp2(exponent)
    scale = block_scales * tensor_scale
    grid = E2M1[:8]
    magnitude = np.abs(blocks) / scale
    index = np.abs(magnitude[..., None] - grid).argmin(-1)
    return (np.sign(blocks) * grid[index] * scale).reshape(weights.shape).astype(np.float32)


def dequantized_tensors(nvfp4: dict, rtn_source: dict | None = None) -> dict[str, tuple[str, np.ndarray]]:
    out = {}
    for name, (dtype, value) in nvfp4.items():
        if name.endswith(".weight_scale") or name.endswith(".weight_scale_2") or name.endswith(".input_scale"):
            continue
        scale = nvfp4.get(name + "_scale")
        if dtype == "U8" and scale is not None:
            if rtn_source is not None:
                reference = (rtn_source[name][1].astype(np.uint32) << 16).view(np.float32)
                weights = round_to_nvfp4(reference)
            else:
                weights = dequantize(value, scale[1], nvfp4[name + "_scale_2"][1])
            out[name] = ("BF16", to_bf16_bits(weights))
        else:
            out[name] = (dtype, value)
    return out


def write_bf16_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    import struct

    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        dtype, value = tensors[name]
        blob = np.ascontiguousarray(value).tobytes()
        header[name] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for blob in blobs:
            file.write(blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nvfp4", required=True, type=Path)
    ap.add_argument("--config-from", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", type=Path, help="BF16 checkout to compare a few dequantized tensors against")
    ap.add_argument(
        "--rtn-from",
        type=Path,
        help="instead of the NVFP4 values, round the weights of this BF16 checkout to NVFP4 (per-block absmax / 6, "
        "FP8 block scales); the NVFP4 checkout only says which tensors are quantized",
    )
    args = ap.parse_args()
    nvfp4 = load_raw(args.nvfp4 / "model.safetensors")
    if args.check:
        bf16 = load_raw(args.check / "model.safetensors")
        for name in ["backbone.layers.16.mixer.q_proj.weight", "backbone.layers.17.mixer.down_proj.weight"]:
            weights = dequantize(nvfp4[name][1], nvfp4[name + "_scale"][1], nvfp4[name + "_scale_2"][1])
            reference = (bf16[name][1].astype(np.uint32) << 16).view(np.float32)
            error = np.abs(weights - reference).mean() / np.abs(reference).mean()
            correlation = np.corrcoef(weights.ravel(), reference.ravel())[0, 1]
            print(f"{name}: relative mean error {error:.4f}, correlation {correlation:.5f}")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    rtn_source = load_raw(args.rtn_from / "model.safetensors") if args.rtn_from else None
    write_bf16_safetensors(args.out / "model.safetensors", dequantized_tensors(nvfp4, rtn_source))
    for path in args.nvfp4.iterdir():
        if path.is_file() and path.name not in {"model.safetensors", "config.json", "hf_quant_config.json"}:
            shutil.copy(path, args.out / path.name)
    shutil.copy(args.config_from / "config.json", args.out / "config.json")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
