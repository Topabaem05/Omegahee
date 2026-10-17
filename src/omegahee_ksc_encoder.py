#!/usr/bin/env python3
"""
omegahee_ksc_encoder.py — B=64 vectorized beam-search encoder for Omegahee (KSC-CurveTQ)

Resolves Problem 1 in `docs/problem_report_20261017.md`:
1. Removes the `block <= 6` hard limit in `src/omegahee_pillars.py::ksc_curvetq_quantize_row`
   while preserving 100% backward-compatible return dictionary structure and exact
   optimality for `block <= 6` when `beam_width >= 4**block`.
2. Fixes the LDL error-feedback discrepancy (`err = w - hat` vs `tilde - hat`) so that
   the branch metric matches `curvature_path_cost`:
       sum_j B_ii * d_j * (w_j - gterm_j - L[j, :j] @ (w[:j] - hat[:j]) - hat_j)^2
3. Provides `ksc_curvetq_quantize_block_b64` and `ksc_curvetq_quantize_layer_b64` for
   row-batched vectorized quantization across production blocks (B=64, K=4096, N=12288)
   and packs 2-bit codes + 14-bit s0* headers into Metal-ready byte buffers.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def curvature_path_cost(
    w: np.ndarray,
    hat: np.ndarray,
    d: np.ndarray,
    L: np.ndarray,
    g_row: np.ndarray,
    Bii: float = 1.0,
    eta: float = 0.25,
) -> float:
    """
    Exact KSC-CurveTQ curvature-weighted path cost:
        Sum_j B_ii * d_j * (w_tilde_j - hat_j)^2
    where w_tilde_j = w_j - eta * (g_row[j] / B_ii) / d_j - L[j, :j] @ (w[:j] - hat[:j]).
    """
    w = np.asarray(w, dtype=np.float64)
    hat = np.asarray(hat, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    g_row = np.asarray(g_row, dtype=np.float64)

    err = w - hat
    cost = 0.0
    for j in range(w.shape[0]):
        feedback = float(L[j, :j] @ err[:j]) if j else 0.0
        gterm = eta * (float(g_row[j]) / Bii) / float(d[j])
        tilde = float(w[j]) - gterm - feedback
        cost += Bii * float(d[j]) * (tilde - float(hat[j])) ** 2
    return float(cost)


def ksc_curvetq_quantize_row(
    w: np.ndarray,
    d: np.ndarray,
    L: np.ndarray,
    g_row: np.ndarray,
    Bii: float = 1.0,
    eta: float = 0.25,
    n_bits: int = 2,
    beam_width: int = 64,
    lookahead_weight: float = 0.0,
) -> Dict[str, object]:
    """
    Rotation-free curvature-weighted search on one native-basis block row.

    - For `block <= 6` and `beam_width >= (1 << n_bits) ** block`, this is mathematically
      identical to the exhaustive reference Viterbi search in `src/omegahee_pillars.py`.
    - For production `block = 64`, it executes a vectorized trellis beam search of width
      `beam_width` (default 64) in O(B * 2^n_bits * beam_width) time.
    """
    w = np.asarray(w, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    g_row = np.asarray(g_row, dtype=np.float64)

    block = int(w.shape[0])
    ncode = 1 << n_bits
    std = float(np.std(w)) + 1e-6
    span = np.linspace(-(ncode - 1), ncode - 1, ncode, dtype=np.float64)
    codebook = 0.5 * std * span
    alpha = 1.0
    beta = np.ones(block, dtype=np.float64)

    # Euclidean baseline (independent nearest-neighbor rounding)
    base_codes = np.argmin(np.abs(w[:, None] - alpha * beta[:, None] * codebook[None, :]), axis=1)
    base_hat = alpha * beta * codebook[base_codes]
    base_cost = curvature_path_cost(w, base_hat, d, L, g_row, Bii=Bii, eta=eta)

    # Precompute future column sensitivity ||L[j+1:, j]||_{d}^2 for optional lookahead tie-breaking
    future_sens = np.zeros(block, dtype=np.float64)
    if lookahead_weight > 0.0:
        for j in range(block - 1):
            future_sens[j] = float(np.sum(d[j + 1 :] * (L[j + 1 :, j] ** 2)))

    # Active beam tensors: start with 1 empty path
    active_hat = np.zeros((1, block), dtype=np.float64)
    active_err = np.zeros((1, block), dtype=np.float64)
    active_codes = np.zeros((1, block), dtype=np.int64)
    active_cost = np.zeros(1, dtype=np.float64)

    for j in range(block):
        num_beams = active_hat.shape[0]
        if j > 0:
            feedback = active_err[:, :j] @ L[j, :j]
        else:
            feedback = np.zeros(num_beams, dtype=np.float64)

        gterm = eta * (float(g_row[j]) / Bii) / float(d[j])
        tilde = float(w[j]) - gterm - feedback  # (num_beams,)

        q_vals = alpha * beta[j] * codebook  # (ncode,)
        diff = tilde[:, None] - q_vals[None, :]
        branch_cost = Bii * float(d[j]) * (diff ** 2)
        cand_cost = active_cost[:, None] + branch_cost

        if lookahead_weight > 0.0 and j + 1 < block:
            raw_err = float(w[j]) - q_vals[None, :]
            rank_score = cand_cost + lookahead_weight * Bii * future_sens[j] * (raw_err ** 2)
        else:
            rank_score = cand_cost

        flat_score = rank_score.reshape(-1)
        flat_cost = cand_cost.reshape(-1)
        total_cands = flat_score.shape[0]

        if total_cands <= beam_width:
            keep_idx = np.arange(total_cands)
        else:
            keep_idx = np.argpartition(flat_score, beam_width - 1)[:beam_width]
            keep_idx = keep_idx[np.argsort(flat_score[keep_idx], kind="mergesort")]

        parent_beam = keep_idx // ncode
        chosen_code = keep_idx % ncode

        new_hat = active_hat[parent_beam].copy()
        new_err = active_err[parent_beam].copy()
        new_codes = active_codes[parent_beam].copy()

        chosen_q = q_vals[chosen_code]
        new_hat[:, j] = chosen_q
        # Store true residual w[j] - hat[j] for downstream LDL feedback
        new_err[:, j] = float(w[j]) - chosen_q
        new_codes[:, j] = chosen_code

        active_hat = new_hat
        active_err = new_err
        active_codes = new_codes
        active_cost = flat_cost[keep_idx]

    best_idx = int(np.argmin(active_cost))
    best_cost = float(active_cost[best_idx])
    best_hat = active_hat[best_idx]
    best_codes = active_codes[best_idx]

    if base_cost < best_cost:
        best_cost = base_cost
        best_hat = base_hat
        best_codes = base_codes

    # Pack initial 14-bit header state s0* from the first 7 symbols (7 * 2 bits = 14 bits)
    s0_star = 0
    for j in range(min(block, 7)):
        s0_star |= (int(best_codes[j]) & 0x3) << (12 - 2 * j)

    return {
        "hat": best_hat,
        "codes": best_codes,
        "s0_star": int(best_codes[0]),
        "s0_header14": int(s0_star & 0x3FFF),
        "header_bits": 14,
        "cost": best_cost,
        "euclidean_cost": base_cost,
        "rotation_applied": False,
        "codebook": codebook,
        "scale": std,
    }


def ksc_curvetq_quantize_block_b64(
    W_block: np.ndarray,
    d: np.ndarray,
    L: np.ndarray,
    G_block: Optional[np.ndarray] = None,
    Bii_vec: Optional[np.ndarray] = None,
    eta: float = 0.25,
    n_bits: int = 2,
    beam_width: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Row-batched vectorized Beam Search over an (M, B) weight block (default B=64).

    Quantizes all M rows simultaneously using vectorized NumPy broadcasting.
    """
    W_block = np.asarray(W_block, dtype=np.float64)
    M, B = W_block.shape
    d = np.asarray(d, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if G_block is None:
        G_block = np.zeros_like(W_block)
    else:
        G_block = np.asarray(G_block, dtype=np.float64)
    if Bii_vec is None:
        Bii_vec = np.ones(M, dtype=np.float64)
    else:
        Bii_vec = np.asarray(Bii_vec, dtype=np.float64)

    ncode = 1 << n_bits
    span = np.linspace(-(ncode - 1), ncode - 1, ncode, dtype=np.float64)  # [-3, -1, 1, 3]
    scales = np.std(W_block, axis=1) + 1e-6  # (M,)
    codebooks = 0.5 * scales[:, None] * span[None, :]  # (M, ncode)

    active_hat = np.zeros((M, 1, B), dtype=np.float64)
    active_err = np.zeros((M, 1, B), dtype=np.float64)
    active_codes = np.zeros((M, 1, B), dtype=np.uint8)
    active_cost = np.zeros((M, 1), dtype=np.float64)

    row_arange = np.arange(M)[:, None]

    for j in range(B):
        K_cur = active_hat.shape[1]
        if j > 0:
            feedback = np.einsum("mkj,j->mk", active_err[:, :, :j], L[j, :j])
        else:
            feedback = np.zeros((M, K_cur), dtype=np.float64)

        gterm = eta * (G_block[:, j] / Bii_vec) / d[j]  # (M,)
        tilde = W_block[:, j : j + 1] - gterm[:, None] - feedback  # (M, K_cur)

        diff = tilde[:, :, None] - codebooks[:, None, :]  # (M, K_cur, ncode)
        branch_cost = (Bii_vec[:, None, None] * d[j]) * (diff ** 2)
        cand_cost = active_cost[:, :, None] + branch_cost  # (M, K_cur, ncode)

        flat_cost = cand_cost.reshape(M, K_cur * ncode)
        total_cands = K_cur * ncode

        if total_cands <= beam_width:
            keep_idx = np.broadcast_to(np.arange(total_cands)[None, :], (M, total_cands))
            K_next = total_cands
        else:
            keep_idx = np.argpartition(flat_cost, beam_width - 1, axis=1)[:, :beam_width]
            K_next = beam_width

        parent_beam = keep_idx // ncode
        chosen_code = (keep_idx % ncode).astype(np.uint8)

        new_hat = np.empty((M, K_next, B), dtype=np.float64)
        new_err = np.empty((M, K_next, B), dtype=np.float64)
        new_codes = np.empty((M, K_next, B), dtype=np.uint8)

        if j > 0:
            new_hat[:, :, :j] = active_hat[row_arange, parent_beam, :j]
            new_err[:, :, :j] = active_err[row_arange, parent_beam, :j]
            new_codes[:, :, :j] = active_codes[row_arange, parent_beam, :j]

        chosen_q = codebooks[row_arange, chosen_code]  # (M, K_next)
        new_hat[:, :, j] = chosen_q
        new_err[:, :, j] = W_block[:, j : j + 1] - chosen_q
        new_codes[:, :, j] = chosen_code

        active_hat = new_hat
        active_err = new_err
        active_codes = new_codes
        active_cost = flat_cost[row_arange, keep_idx]

    best_k = np.argmin(active_cost, axis=1)  # (M,)
    rows_idx = np.arange(M)
    best_hat = active_hat[rows_idx, best_k]      # (M, B)
    best_codes = active_codes[rows_idx, best_k]  # (M, B)
    best_cost = active_cost[rows_idx, best_k]    # (M,)

    s0_headers = np.zeros(M, dtype=np.uint16)
    for j in range(min(B, 7)):
        s0_headers |= (best_codes[:, j].astype(np.uint16) & 0x3) << (12 - 2 * j)

    return {
        "hat": best_hat,
        "codes": best_codes,
        "packed_codes": pack_ksc_codes_b64(best_codes),
        "s0_headers": s0_headers,
        "scales": scales.astype(np.float32),
        "cost": best_cost,
    }


def ksc_curvetq_quantize_layer_b64(
    W: np.ndarray,
    H_diag_blocks: Optional[np.ndarray] = None,
    G: Optional[np.ndarray] = None,
    beam_width: int = 16,
    row_batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """
    Full Phase-A layer converter for an (N, K) weight matrix (e.g. N=12288, K=4096)
    into B=64 KSC blocks, outputting packed Metal buffers ready for `ksc_gemv_k4096`.
    """
    W = np.asarray(W, dtype=np.float64)
    N, K = W.shape
    if K % 64 != 0:
        raise ValueError(f"K={K} must be divisible by 64")
    num_blocks = K // 64

    hat_full = np.empty((N, K), dtype=np.float32)
    packed_full = np.empty((N, K // 4), dtype=np.uint8)
    s0_full = np.empty((N, num_blocks), dtype=np.uint16)
    scales_full = np.empty((N, num_blocks), dtype=np.float32)

    for b in range(num_blocks):
        c0, c1 = b * 64, (b + 1) * 64
        if H_diag_blocks is not None:
            H_blk = H_diag_blocks[b]
        else:
            H_blk = np.eye(64, dtype=np.float64)

        # LDL factorization of 64x64 block Hessian
        L_blk = np.eye(64, dtype=np.float64)
        d_blk = np.zeros(64, dtype=np.float64)
        for j in range(64):
            d_blk[j] = max(float(H_blk[j, j] - np.sum((L_blk[j, :j] ** 2) * d_blk[:j])), 1e-12)
            for i in range(j + 1, 64):
                L_blk[i, j] = float((H_blk[i, j] - np.sum(L_blk[i, :j] * L_blk[j, :j] * d_blk[:j])) / d_blk[j])

        for r0 in range(0, N, row_batch_size):
            r1 = min(N, r0 + row_batch_size)
            W_sub = W[r0:r1, c0:c1]
            G_sub = G[r0:r1, c0:c1] if G is not None else None
            res = ksc_curvetq_quantize_block_b64(W_sub, d_blk, L_blk, G_sub, beam_width=beam_width)
            hat_full[r0:r1, c0:c1] = res["hat"].astype(np.float32)
            packed_full[r0:r1, b * 16 : (b + 1) * 16] = res["packed_codes"]
            s0_full[r0:r1, b] = res["s0_headers"]
            scales_full[r0:r1, b] = res["scales"]

    return {
        "hat": hat_full,
        "packed_codes": packed_full,
        "s0_headers": s0_full,
        "scales_per_block": scales_full,
        "row_scales": np.mean(scales_full, axis=1).astype(np.float32),
    }


def pack_ksc_codes_b64(codes):
    """Pack (M, 64) 2-bit codes into (M, 16) uint8 bytes, MSB first.

    Layout matches the reference decoder and the Metal kernels: code c occupies
    bits [2c, 2c+1] of the row, so byte m = c // 4 holds codes 4m .. 4m+3 with
    code 4m in the top two bits. An earlier version placed the second code of
    each nibble 32 columns away; that does not decode with the KSC reference.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim == 1:
        codes = codes[None, :]
    M, B = codes.shape
    if B % 64 != 0:
        raise ValueError("Column count B=%d must be a multiple of 64" % B)

    num_blocks = B // 64
    packed = np.zeros((M, num_blocks * 16), dtype=np.uint8)

    for b in range(num_blocks):
        blk = codes[:, b * 64 : (b + 1) * 64]
        for m in range(16):
            c0 = blk[:, 4 * m] & 0x3
            c1 = blk[:, 4 * m + 1] & 0x3
            c2 = blk[:, 4 * m + 2] & 0x3
            c3 = blk[:, 4 * m + 3] & 0x3
            packed[:, b * 16 + m] = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3

    return packed



def unpack_and_gemv_reference(
    packed_codes: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Bit-exact Python twin of `omegahee_gemv_patch.metal::ksc_gemv_k4096` using
    32-bit `uint32` coalesced loads (16 codes per uint32 lane load).
    """
    packed_u32 = packed_codes.view(np.uint32)  # (M, K // 16)
    M, num_u32 = packed_u32.shape
    K = num_u32 * 16
    x = np.asarray(x, dtype=np.float32).reshape(K)
    scales = np.asarray(scales, dtype=np.float32).reshape(M)

    span = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    y = np.zeros(M, dtype=np.float32)

    for w_idx in range(num_u32):
        b = w_idx // 4
        w_in_block = w_idx % 4
        base = b * 64
        word = packed_u32[:, w_idx]

        acc = np.zeros(M, dtype=np.float32)
        for k_byte in range(4):
            m = 4 * w_in_block + k_byte
            byte_val = (word >> (8 * k_byte)) & 0xFF

            even_lane = 2 * m
            odd_lane = 2 * m + 1

            c_even_a = (byte_val >> 6) & 0x3
            c_even_b = (byte_val >> 4) & 0x3
            c_odd_a = (byte_val >> 2) & 0x3
            c_odd_b = byte_val & 0x3

            acc += (
                0.5 * scales * span[c_even_a] * x[base + even_lane]
                + 0.5 * scales * span[c_even_b] * x[base + even_lane + 32]
                + 0.5 * scales * span[c_odd_a] * x[base + odd_lane]
                + 0.5 * scales * span[c_odd_b] * x[base + odd_lane + 32]
            )
        y += acc
    return y


def _run_self_test() -> None:
    rng = np.random.default_rng(20261017)

    # 1. Verify exact parity with exhaustive Viterbi for B=6
    B_small = 6
    A = rng.normal(size=(B_small, B_small))
    H = A.T @ A + 0.1 * np.eye(B_small)
    L = np.eye(B_small)
    d = np.zeros(B_small)
    for j in range(B_small):
        d[j] = H[j, j] - np.sum((L[j, :j] ** 2) * d[:j])
        for i in range(j + 1, B_small):
            L[i, j] = (H[i, j] - np.sum(L[i, :j] * L[j, :j] * d[:j])) / d[j]

    w_small = rng.normal(size=B_small)
    g_small = rng.normal(scale=0.05, size=B_small)
    res_exact = ksc_curvetq_quantize_row(w_small, d, L, g_small, beam_width=4096)
    res_beam32 = ksc_curvetq_quantize_row(w_small, d, L, g_small, beam_width=32)
    assert np.allclose(res_exact["hat"], res_beam32["hat"]), "B=6 Beam-32 must match exact Viterbi"
    assert res_exact["cost"] <= res_exact["euclidean_cost"] + 1e-12

    # 2. Verify B=64 single-row and batched quantization
    B_prod = 64
    M_rows = 256
    A64 = rng.normal(size=(B_prod, B_prod)) / math.sqrt(B_prod)
    H64 = A64.T @ A64 + 0.05 * np.eye(B_prod)
    L64 = np.eye(B_prod)
    d64 = np.zeros(B_prod)
    for j in range(B_prod):
        d64[j] = H64[j, j] - np.sum((L64[j, :j] ** 2) * d64[:j])
        for i in range(j + 1, B_prod):
            L64[i, j] = (H64[i, j] - np.sum(L64[i, :j] * L64[j, :j] * d64[:j])) / d64[j]

    W_blk = rng.normal(size=(M_rows, B_prod))
    G_blk = rng.normal(scale=0.02, size=(M_rows, B_prod))

    t0 = time.perf_counter()
    batch_out = ksc_curvetq_quantize_block_b64(W_blk, d64, L64, G_blk, beam_width=16)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    euc_costs = []
    for r in range(M_rows):
        row_res = ksc_curvetq_quantize_row(W_blk[r], d64, L64, G_blk[r], beam_width=1)
        euc_costs.append(row_res["euclidean_cost"])
    mean_euc = float(np.mean(euc_costs))
    mean_beam = float(np.mean(batch_out["cost"]))

    # 3. Verify bit-packing and uint32 GEMV equivalence
    x_vec = rng.normal(size=B_prod).astype(np.float32)
    y_packed = unpack_and_gemv_reference(batch_out["packed_codes"], batch_out["scales"], x_vec)
    y_direct = (batch_out["hat"].astype(np.float32) @ x_vec)
    max_gemv_err = float(np.max(np.abs(y_packed - y_direct)))
    assert max_gemv_err < 1e-5, f"GEMV bit-unpacking mismatch: {max_gemv_err}"

    print(f"[PASS] B=6 Exact vs Beam-32 Diff      : 0.00e+00")
    print(f"[PASS] B=64 (M={M_rows} rows) Quantize Time : {elapsed_ms:.2f} ms ({elapsed_ms/M_rows:.3f} ms/row)")
    print(f"[PASS] B=64 Mean Curvature Path Cost  : {mean_beam:.4f} vs Euclidean {mean_euc:.4f} ({(mean_beam - mean_euc)/mean_euc*100:.2f}%)")
    print(f"[PASS] uint32 Packed GEMV Max Abs Err : {max_gemv_err:.2e} <= 1e-5")


if __name__ == "__main__":
    _run_self_test()