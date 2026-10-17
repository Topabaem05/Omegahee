"""Calibration Hessian support for the KSC encoder."""

from __future__ import annotations

import importlib.util
import os
from typing import Dict, List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load("omegahee_pillars", "omegahee_pillars.py")
E = _load("omegahee_ksc_encoder", "omegahee_ksc_encoder.py")


def block_diag_covariance(X: np.ndarray, block: int = 64) -> List[np.ndarray]:
    """Per-block activation covariance H_b = X_b^T X_b / rows.

    The KSC branch metric is a curvature-weighted distance within one 64-column
    block, so only the block-diagonal part of the full Hessian is needed.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (rows, K)")
    rows, k = X.shape
    if block <= 0 or k % block != 0:
        raise ValueError("K must be divisible by block")
    out = []
    for start in range(0, k, block):
        seg = X[:, start:start + block]
        out.append(seg.T @ seg / float(rows))
    return out


def layer_curvature(X: np.ndarray, block: int = 64, lam: float = 1e-4) -> Dict[str, object]:
    """LDL factorisation per block: H = L diag(d) L^T.

    Returns {"blocks": [{"d": ndarray, "L": ndarray}, ...]}. `block` must match
    the encoder block size (64 for the production layout).
    """
    covs = block_diag_covariance(X, block)
    blocks = []
    for cov in covs:
        H = P.regularized_schur(cov, block=block, lam=lam)
        L, d = P.ldl_factor(H)
        blocks.append({"L": np.asarray(L, dtype=np.float64), "d": np.asarray(d, dtype=np.float64)})
    return {"blocks": blocks, "block": block, "lam": lam}


def quantize_layer_with_hessian(
    W: np.ndarray,
    X: Optional[np.ndarray],
    block: int = 64,
    beam_width: int = 8,
    lam: float = 1e-4,
    max_rows: Optional[int] = None,
) -> Dict[str, object]:
    """Quantize a layer with L from real activations, or L=0 when X is absent.

    With X=None this reduces to plain 2-bit rounding under the same codebook,
    which is the measured correct default: a fabricated L makes reconstruction
    worse (docs/step1_step2_results_20261018.md).
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be 2-D (N, K)")
    n, k = W.shape
    if k % block != 0:
        raise ValueError("K must be divisible by block")
    rows = n if max_rows is None else min(n, int(max_rows))
    W = W[:rows]

    curvature = None
    if X is not None:
        curvature = layer_curvature(X, block=block, lam=lam)

    nblocks = k // block
    hat = np.empty((rows, k), dtype=np.float64)
    codes = np.empty((rows, k), dtype=np.uint8)
    s0 = np.empty((rows, nblocks), dtype=np.uint16)
    scales = np.empty((rows, nblocks), dtype=np.float64)

    for b in range(nblocks):
        start = b * block
        seg = W[:, start:start + block]
        if curvature is None:
            L = np.zeros((block, block), dtype=np.float64)
            d = np.ones(block, dtype=np.float64)
        else:
            entry = curvature["blocks"][b]
            L = entry["L"]
            d = entry["d"]
        out = E.ksc_curvetq_quantize_block_b64(seg, d, L, beam_width=beam_width)
        hat[:, start:start + block] = out["hat"]
        codes[:, start:start + block] = out["codes"]
        packed = np.asarray(out["packed_codes"])
        if b == 0:
            packed_all = np.empty((rows, nblocks * 16), dtype=np.uint8)
        packed_all[:, b * 16:(b + 1) * 16] = packed
        s0[:, b] = np.asarray(out["s0_headers"])
        scales[:, b] = np.asarray(out["scales"])

    return {
        "hat": hat,
        "codes": codes,
        "packed_codes": packed_all,
        "s0_headers": s0,
        "scales": scales,
        "mse": float(np.mean((W - hat) ** 2)),
        "rows": rows,
        "used_hessian": X is not None,
    }
