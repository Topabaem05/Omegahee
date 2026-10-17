"""
Omegahee Metal reference semantics (device measurement layer).

=========================================================================
Purpose
-------
`docs/uzu_metal_kernel_mapping.md` states three integration contracts for the
uzu crates but ships no executable form of them. This module fixes one concrete,
reproducible interpretation of each contract in numpy, and
`metal/kernels/omegahee.metal` implements the identical interpretation in the
Metal Shading Language. The Swift harness compares the two on a real GPU, so the
contract stops being prose and becomes a checked, measured artifact.

This module is pure numpy and runs on Linux CI. The Metal side is verified on an
Apple Silicon machine by `metal/run.sh`.

Concrete realizations
---------------------
1. `ksc_decode_dot`  - 14-bit start-state prologue + register-shift trellis
   decode of the 2-bit native-basis codes, per-column scale, per-row scale.
   No Hadamard / orthogonal un-rotation is applied, which is the whole point of
   the rotation-free KSC-CurveTQ branch.
2. `awsrc_dot`       - in-register splitmix sign regeneration for the seeded
   rank-1 residual epilogue. Only seeds and scales are stored.
3. `readkv_prefix`   - bit-plane transposed prefix load with a per-channel read
   budget. Planes at or above the budget are never touched.
4. `lilicorr_pair`   - single-pass bilinear cosine correlation between
   consecutive draft slots.

Bit layout used by 1 and 2
--------------------------
Each 64-column block row stores

- a 14-bit start state `s0` (the mapping document's `s0*` header), and
- 16 bytes of MSB-first packed 2-bit codes.

The header is the 14-bit window over stream bits 0..13 read MSB-first, so the
first emitted code is code 0 and the header mirrors the row prefix rather than
carrying extra payload. The decode loop emits the top two bits of the 14-bit
register and shifts in the next stream pair, starting at bit offset 14, which
continues with code 7. Iterations `j = 57..63` run past bit 127; those shifts
insert zero, so a short row never wraps into an earlier column.

Flush-to-zero and rounding differences aside, a float32 GPU accumulator over the
same order reproduces these float64 results to the tolerances in
`metal/README.md`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

BLOCK = 64
NCODE = 4
CODE_MASK = 0x3FFF
STREAM_BITS = 128
HEADER_BITS = 14

# 2-bit mid-riser lattice, identical to omegahee_pillars.ksc_curvetq_quantize_row.
SPAN = np.linspace(-(NCODE - 1), NCODE - 1, NCODE)


# ---------------------------------------------------------------------------
# Bit helpers (MSB-first, matching the Metal byte loads)
# ---------------------------------------------------------------------------

def pack_2bit(codes: Sequence[int]) -> bytes:
    """MSB-first 2-bit packing of BLOCK code indices into 16 bytes."""
    if len(codes) != BLOCK:
        raise ValueError(f"expected {BLOCK} codes, got {len(codes)}")
    acc = 0
    nbits = 0
    out = bytearray()
    for code in codes:
        acc = (acc << 2) | (int(code) & 0x3)
        nbits += 2
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    return bytes(out)


def _bit(row_bytes: bytes, index: int) -> int:
    # The stream window is exactly STREAM_BITS wide. Indices outside it read as
    # zero, so the tail of a short row never wraps into an earlier column.
    if index < 0 or index >= min(len(row_bytes) * 8, STREAM_BITS):
        return 0
    return (row_bytes[index >> 3] >> (7 - (index & 7))) & 1


def _pair(row_bytes: bytes, index: int) -> int:
    return (_bit(row_bytes, index) << 1) | _bit(row_bytes, index + 1)


def codebook(scale: float) -> np.ndarray:
    """0.5 * scale * SPAN, the same lattice the pillar kernel searches over."""
    return 0.5 * float(scale) * SPAN


# ---------------------------------------------------------------------------
# 1. KSC-CurveTQ runtime decode
# ---------------------------------------------------------------------------

def ksc_dot(
    s0: int,
    row_bytes: bytes,
    alpha: float,
    beta: np.ndarray,
    x: np.ndarray,
    scale: float,
) -> float:
    """Order-exact scalar decode of one 64-column block row."""
    lut = codebook(scale)
    state = int(s0) & CODE_MASK
    acc = 0.0
    for j in range(BLOCK):
        code = (state >> 12) & 0x3
        q = float(lut[code])
        acc += float(alpha) * float(beta[j]) * q * float(x[j])
        state = ((state << 2) | _pair(row_bytes, HEADER_BITS + 2 * j)) & CODE_MASK
    return acc


def ksc_dot_batch(
    s0: np.ndarray,
    rows: Sequence[bytes],
    alpha: np.ndarray,
    beta: np.ndarray,
    x: np.ndarray,
    scale: float,
) -> np.ndarray:
    return np.array(
        [ksc_dot(int(s0[r]), rows[r], float(alpha[r]), beta, x, scale) for r in range(len(rows))],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 2. AWSRC seeded residual epilogue
# ---------------------------------------------------------------------------

def splitmix_sign(seed: int, index: int) -> float:
    """Deterministic +-1 from a 32-bit seed and a linear index.

    Mirrors the Metal `splitmix_sign` helper bit for bit on uint32 arithmetic.
    """
    mask = 0xFFFFFFFF
    h = (int(seed) * 0x9E3779B9 + int(index) * 0x85EBCA6B) & mask
    h ^= h >> 16
    h = (h * 0xC2B2AE35) & mask
    h ^= h >> 13
    h = (h * 0x27D4EB2F) & mask
    h ^= h >> 16
    return 1.0 if (h & 1) == 0 else -1.0


def awsrc_dot(
    row: int,
    seeds: np.ndarray,
    gammas: np.ndarray,
    x: np.ndarray,
    block: int = BLOCK,
) -> float:
    """Scalar correction for one row. `gammas` is [rows, M]."""
    inv = 1.0 / math.sqrt(float(block))
    acc = 0.0
    for m in range(len(seeds)):
        s = 0.0
        for j in range(block):
            s += splitmix_sign(int(seeds[m]), row * block + j) * float(x[j])
        acc += float(gammas[row][m]) * (s * inv)
    return acc


def awsrc_dot_batch(
    seeds: np.ndarray,
    gammas: np.ndarray,
    x: np.ndarray,
    block: int = BLOCK,
) -> np.ndarray:
    return np.array(
        [awsrc_dot(r, seeds, gammas, x, block) for r in range(gammas.shape[0])],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 3. ReadKV bit-plane prefix load
# ---------------------------------------------------------------------------

# Plane p holds bit weight 2^(7-p): plane 0 is the MSB. A channel with read
# budget r therefore reconstructs the top-r-bit prefix of the stored 8-bit
# value, with the unread planes contributing zero.
PLANE_GRID = (0, 2, 4, 8)
PLANE_WIDTHS = (7, 6, 5, 4, 3, 2, 1, 0)


def readkv_word(planes: np.ndarray) -> int:
    """One uint32 per plane; bit c of plane p is channel c's bit at weight 2^(7-p)."""
    total = 0
    for p in range(8):
        total |= (int(planes[p]) & 0xFFFFFFFF) << (8 * p)
    return total


def readkv_prefix_value(planes: np.ndarray, channel: int, budget: int) -> int:
    if budget not in PLANE_GRID:
        raise ValueError(f"budget must be one of {PLANE_GRID}")
    value = 0
    for p in range(budget):
        bit = (int(planes[p]) >> channel) & 1
        value |= bit << PLANE_WIDTHS[p]
    return value


def readkv_prefix_batch(planes: np.ndarray, budgets: np.ndarray, channels: int) -> np.ndarray:
    out = np.zeros(channels, dtype=np.int64)
    for c in range(channels):
        out[c] = readkv_prefix_value(planes, c, int(budgets[c]))
    return out


def readkv_plane_words(planes: np.ndarray, budget: int) -> int:
    return int(budget)


# ---------------------------------------------------------------------------
# 4. LiLiCorr bilinear correlation
# ---------------------------------------------------------------------------

def lilicorr_pair(uo_t: np.ndarray, ui_next: np.ndarray, tau: float) -> np.ndarray:
    """tau * cosine(U_out[t, k], U_in[t+1, k']) for every (k, k') pair."""
    a = uo_t / (np.linalg.norm(uo_t, axis=-1, keepdims=True) + 1e-12)
    b = ui_next / (np.linalg.norm(ui_next, axis=-1, keepdims=True) + 1e-12)
    return float(tau) * (a @ b.T)


def lilicorr_batch(uo: np.ndarray, ui: np.ndarray, tau: float) -> np.ndarray:
    """[G-1, K, K] matrix of consecutive-slot correlations."""
    g = uo.shape[0]
    return np.stack([lilicorr_pair(uo[t], ui[t + 1], tau) for t in range(g - 1)])


# ---------------------------------------------------------------------------
# Case generators: inputs and float64 expectations for the parity harness
# ---------------------------------------------------------------------------

def make_ksc_case(rows: int = 64, seed: int = 1) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(BLOCK)
    scale = float(np.std(w)) + 1e-6
    codes = rng.integers(0, NCODE, size=(rows, BLOCK))
    rows_bytes = [pack_2bit(codes[r]) for r in range(rows)]
    # The header is the first 14 stream bits, masked to 14 bits.
    s0 = np.array(
        [((_bit(rows_bytes[r], 0) << 13) | sum(_bit(rows_bytes[r], i) << (13 - i) for i in range(1, 14)))
         for r in range(rows)],
        dtype=np.uint16,
    )
    alpha = rng.standard_normal(rows).astype(np.float32)
    beta = rng.standard_normal(BLOCK).astype(np.float32)
    x = rng.standard_normal(BLOCK).astype(np.float32)
    expected = ksc_dot_batch(s0, rows_bytes, alpha.astype(np.float64), beta.astype(np.float64),
                             x.astype(np.float64), scale)
    return {
        "scale": scale,
        "s0": s0,
        "codes": rows_bytes,
        "alpha": alpha,
        "beta": beta,
        "x": x,
        "expected": expected,
    }


def make_awsrc_case(rows: int = 64, m_seeds: int = 4, seed: int = 2) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2 ** 31, size=m_seeds).astype(np.uint32)
    gammas = (rng.standard_normal((rows, m_seeds)) * 0.05).astype(np.float32)
    x = rng.standard_normal(BLOCK).astype(np.float32)
    expected = awsrc_dot_batch(seeds, gammas.astype(np.float64), x.astype(np.float64))
    return {"seeds": seeds, "gammas": gammas, "x": x, "expected": expected}


def make_readkv_case(channels: int = 32, seed: int = 3) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    planes = rng.integers(0, 2 ** 32, size=8).astype(np.uint32)
    budgets = np.array([PLANE_GRID[i % len(PLANE_GRID)] for i in range(channels)], dtype=np.uint8)
    expected = readkv_prefix_batch(planes, budgets, channels)
    return {"planes": planes, "budgets": budgets, "expected": expected}


def make_lilicorr_case(groups: int = 4, k: int = 6, dc: int = 32, tau: float = 2.0,
                       seed: int = 4) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    uo = rng.standard_normal((groups, k, dc)).astype(np.float32)
    ui = rng.standard_normal((groups, k, dc)).astype(np.float32)
    expected = lilicorr_batch(uo.astype(np.float64), ui.astype(np.float64), float(tau))
    return {"uo": uo, "ui": ui, "tau": tau, "expected": expected}


def reference_report() -> Dict[str, object]:
    """Small self-check used by the CI test and by tools/gen_vectors.py."""
    ksc = make_ksc_case(rows=8, seed=11)
    aw = make_awsrc_case(rows=8, seed=12)
    rk = make_readkv_case(channels=32, seed=13)
    ll = make_lilicorr_case(seed=14)
    return {
        "ksc_rows": len(ksc["s0"]),
        "ksc_first": float(ksc["expected"][0]),
        "awsrc_first": float(aw["expected"][0]),
        "readkv_hot": int(rk["expected"][0]),
        "readkv_cold": int(rk["expected"][-1]),
        "lilicorr_shape": list(ll["expected"].shape),
        "lilicorr_absmax": float(np.max(np.abs(ll["expected"]))),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(reference_report(), indent=2))
