"""Dequantize layer 0 up_projection and compare KSC re-quantization."""

import importlib.util
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mirai_raw_safetensors import RawSafeTensors, find_weights  # noqa: E402
import omegahee_ksc_encoder as E  # noqa: E402


def dequant(path: str, prefix: str) -> tuple:
    r = RawSafeTensors(path)
    packed = r.tensor(prefix + ".weights")      # (N, K/2) uint8
    scales = r.tensor(prefix + ".scales")       # (K/group, N)
    zeros = r.tensor(prefix + ".zero_points")   # (K/group, N/2)?
    r.close()
    n, half = packed.shape
    k = half * 2
    lo = (packed & 0x0F).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    q = np.empty((n, k), dtype=np.float32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    groups = scales.shape[0]
    group = k // groups
    s = scales.T                                 # (N, groups)
    # zero_points are (groups, N/2): output channels share a zero point in pairs.
    z = np.repeat(zeros, 2, axis=1).T             # (N, groups)
    if z.shape != s.shape:
        raise ValueError(f"zero_points {zeros.shape} does not expand to {s.shape}")
    w = np.empty((n, k), dtype=np.float32)
    for g in range(groups):
        seg = q[:, g * group:(g + 1) * group]
        w[:, g * group:(g + 1) * group] = (seg - z[:, g:g + 1]) * s[:, g:g + 1]
    return w, {"shape": (n, k), "groups": groups, "group": group}


def main():
    path = find_weights(os.path.expanduser(
        "~/.cache/mirai/models/mirai/alibaba-qwen3.5-9b-mirai-mirai-m-4"))
    prefix = "decoder.transformer.layers.0.mlp.up_projection.weights.quantized"
    w, meta = dequant(path, prefix)
    print("dequantised:", w.shape, "std", round(float(w.std()), 6),
          "min", round(float(w.min()), 4), "max", round(float(w.max()), 4))
    print("meta:", meta)

    # Compare 4-bit Mirai-M against a 2-bit KSC block encoding on the same rows.
    rows = 512
    sub = w[:rows].astype(np.float64)
    block = 64
    d = np.ones(block)
    L = np.tril(np.full((block, block), 0.01), -1)
    # Reference-style flat quantisation: 2-bit over the whole row.
    span = np.array([-3.0, -1.0, 1.0, 3.0])
    scale = np.std(sub, axis=1, keepdims=True)
    flat = 0.5 * scale * span[None, :]
    idx = np.argmin(np.abs(sub[:, :, None] - flat[:, None, :]), axis=-1)
    flat_hat = np.take_along_axis(flat[:, None, :], idx[:, :, None], axis=-1)[:, 0, :]
    flat_mse = float(np.mean((sub - flat_hat) ** 2))

    # KSC 2-bit with curvature, blockwise.
    ksc_hat = np.empty_like(sub)
    for start in range(0, sub.shape[1], block):
        blk = sub[:, start:start + block]
        out = E.ksc_curvetq_quantize_block_b64(blk, d, L, beam_width=8)
        ksc_hat[:, start:start + block] = out["hat"]
    ksc_mse = float(np.mean((sub - ksc_hat) ** 2))

    sqnr_flat = 10 * np.log10(np.mean(sub ** 2) / flat_mse)
    sqnr_ksc = 10 * np.log10(np.mean(sub ** 2) / ksc_mse)
    print()
    print(f"rows compared: {rows}")
    print(f"flat 2-bit MSE {flat_mse:.6e}  SQNR {sqnr_flat:.2f} dB")
    print(f"KSC  2-bit MSE {ksc_mse:.6e}  SQNR {sqnr_ksc:.2f} dB")
    print(f"KSC beats flat: {ksc_mse < flat_mse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
