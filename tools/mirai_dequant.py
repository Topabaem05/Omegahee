"""Dequantize a Mirai-M 4-bit layer and measure KSC re-quantization error."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open


def find_weights(base: str) -> str:
    for entry in os.listdir(base):
        cand = os.path.join(base, entry, "model.safetensors")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(base)


def bf16_to_f32(raw: bytes) -> np.ndarray:
    """Little-endian bfloat16 bytes -> float32 (upper 16 bits of a float32)."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def dequant_layer(path: str, prefix: str) -> tuple[np.ndarray, dict]:
    """Rebuild the float32 matrix from Mirai-M packed nibbles, scales, zero points."""
    with safe_open(path, framework="numpy") as f:
        keys = set(f.keys())
        packed = f.get_tensor(prefix + ".weights")
        zinfo = f.get_slice(prefix + ".zero_points")
        zp_shape = zinfo.get_shape()
        zp_raw = zinfo[:]                     # bytes for bf16 dtypes
        s_raw = f.get_slice(prefix + ".scales")[:]
    # packed is (N, K/2) uint8: low nibble then high nibble
    n, half = packed.shape
    k = half * 2
    lo = (packed & 0x0F).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    q = np.empty((n, k), dtype=np.float32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi

    # scales shape is usually (N, K/group); infer group from the shape
    scales = bf16_to_f32(s_raw)
    zeros = bf16_to_f32(zp_raw)
    s_n, s_k = (scales.size // n, n) if scales.size % n == 0 else (0, 0)
    s_k = scales.size // n if n else 0
    group = k // s_k if s_k else k
    scales = scales.reshape(n, s_k)
    zeros = zeros.reshape(n, zeros.size // n)
    w = np.empty((n, k), dtype=np.float32)
    for g in range(s_k):
        seg = q[:, g * group:(g + 1) * group]
        w[:, g * group:(g + 1) * group] = (seg - zeros[:, g:g + 1]) * scales[:, g:g + 1]
    meta = {"shape": [n, k], "group": group, "scales_shape": list(scales.shape),
            "zero_points_shape": zp_shape}
    return w, meta


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.path.expanduser(
        "~/.cache/mirai/models/mirai/alibaba-qwen3.5-9b-mirai-mirai-m-4"))
    ap.add_argument("--prefix", default="decoder.transformer.layers.0.mlp.up_projection.weights.quantized")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv[1:])
    path = find_weights(args.base)
    w, meta = dequant_layer(path, args.prefix)
    stats = {
        "weights": path,
        "prefix": args.prefix,
        "meta": meta,
        "float_stats": {"min": float(w.min()), "max": float(w.max()),
                        "mean": float(w.mean()), "std": float(w.std())},
    }
    print(json.dumps(stats, indent=2))
    if args.out:
        np.save(args.out, w)
        print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
