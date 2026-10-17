"""Calibration-Hessian curvature for the Omegahee KSC encoder.

The KSC encoder consumes the diagonal ``d`` and unit-lower-triangular ``L``
from the LDL factorisation ``H = L diag(d) L^T`` of a block Hessian.  Given
calibration activations ``X`` with shape ``(rows, K)``, this module estimates
each block Hessian as ``X[:, b*block:(b+1)*block].T @ X[...] / rows`` and
factors it with the existing :mod:`omegahee_pillars` helpers.

``L = 0`` (and ``d = 1``) is the correct fallback when ``X`` is absent.  It
reduces the encoder to plain curvature-free 2-bit rounding.  Fabricating a
nonzero ``L`` is not a substitute for measured curvature: a constant ``L`` was
measured to make reconstruction worse than ``L = 0`` on a real Qwen3.5-9B
layer (see ``docs/step1_step2_results_20261018.md``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from omegahee_ksc_encoder import ksc_curvetq_quantize_block_b64
from omegahee_pillars import ldl_factor, regularized_schur


def _checked_block(block: int, columns: int) -> int:
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise ValueError("block must be a positive integer")
    block = int(block)
    if block <= 0:
        raise ValueError("block must be a positive integer")
    if columns % block != 0:
        raise ValueError(f"K={columns} must be divisible by block={block}")
    return block


def block_diag_covariance(X: np.ndarray, block: int = 64) -> List[np.ndarray]:
    """Return the per-block activation covariances of ``X``.

    ``X`` has shape ``(rows, K)``.  Entry ``b`` is the float64 matrix
    ``X_b.T @ X_b / rows`` where ``X_b`` is block ``b`` of ``X``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional (rows, K) array")
    rows, columns = X.shape
    if rows <= 0:
        raise ValueError("X must contain at least one calibration row")
    block = _checked_block(block, columns)

    covariances: List[np.ndarray] = []
    for start in range(0, columns, block):
        X_block = X[:, start : start + block]
        covariances.append((X_block.T @ X_block) / float(rows))
    return covariances


def layer_curvature(
    X: np.ndarray,
    block: int = 64,
    lam: float = 1e-4,
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Factor each calibration covariance block into KSC curvature.

    Each block first receives ridge regularisation through
    :func:`omegahee_pillars.regularized_schur`, then an LDL factorisation via
    :func:`omegahee_pillars.ldl_factor`.  The returned dictionary contains one
    ``{"d": ndarray, "L": ndarray}`` entry per column block, in column order.
    """
    covariances = block_diag_covariance(X, block)
    blocks: List[Dict[str, np.ndarray]] = []
    for covariance in covariances:
        width = covariance.shape[0]
        H = regularized_schur(covariance, width, lam)
        L, d = ldl_factor(H)
        blocks.append(
            {
                "d": np.asarray(d, dtype=np.float64),
                "L": np.asarray(L, dtype=np.float64),
            }
        )
    return {"blocks": blocks}


def quantize_layer_with_hessian(
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    block: int = 64,
    beam_width: int = 8,
    lam: float = 1e-4,
) -> Dict[str, np.ndarray | float]:
    """Quantize ``W`` using curvature estimated from calibration activations.

    Each ``(N, block)`` slice is encoded by the existing
    :func:`omegahee_ksc_encoder.ksc_curvetq_quantize_block_b64` with the real
    ``d`` and ``L`` for that block.  ``scales`` has shape
    ``(N, K / block)`` with one encoder scale per block.

    If ``X`` is ``None``, curvature is ``d = 1`` and ``L = 0``.  That is the
    correct fallback when calibration data is absent and reproduces plain
    curvature-free 2-bit rounding.  If ``X`` is supplied, its column count must
    match ``W``; ``K`` must also be divisible by ``block``.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be a two-dimensional (N, K) array")
    rows, columns = W.shape
    if rows <= 0 or columns <= 0:
        raise ValueError("W must have non-zero dimensions")
    block = _checked_block(block, columns)
    if block % 64 != 0:
        raise ValueError("block must be a multiple of 64 for KSC code packing")

    num_blocks = columns // block
    if X is None:
        curvature_blocks: List[Dict[str, np.ndarray]] = [
            {
                "d": np.ones(block, dtype=np.float64),
                "L": np.zeros((block, block), dtype=np.float64),
            }
            for _ in range(num_blocks)
        ]
    else:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional (rows, K) array")
        if X.shape[1] != columns:
            raise ValueError(
                f"X has {X.shape[1]} columns but W has {columns}; "
                "calibration activations must match the weight input dimension"
            )
        curvature_blocks = layer_curvature(X, block=block, lam=lam)["blocks"]

    hat = np.empty((rows, columns), dtype=np.float64)
    packed_codes = np.empty((rows, columns // 4), dtype=np.uint8)
    s0_headers = np.empty((rows, num_blocks), dtype=np.uint16)
    scales = np.empty((rows, num_blocks), dtype=np.float32)

    for b in range(num_blocks):
        c0 = b * block
        c1 = c0 + block
        curvature = curvature_blocks[b]
        result = ksc_curvetq_quantize_block_b64(
            W[:, c0:c1],
            curvature["d"],
            curvature["L"],
            beam_width=beam_width,
        )
        hat[:, c0:c1] = result["hat"]
        packed_codes[:, b * (block // 4) : (b + 1) * (block // 4)] = result[
            "packed_codes"
        ]
        s0_headers[:, b] = result["s0_headers"]
        scales[:, b] = result["scales"]

    return {
        "hat": hat,
        "packed_codes": packed_codes,
        "s0_headers": s0_headers,
        "scales": scales,
        "mse": float(np.mean((W - hat) ** 2)),
    }
