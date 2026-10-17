"""
Generate binary parity vectors and float64 expectations for the Metal harness.

Run from the repository root:

    python metal/tools/gen_vectors.py

Output (metal/build/vectors/):

  ksc.bin       header for the ksc_test case
  awsrc.bin     header for the awsrc_test case
  readkv.bin    header for the readkv_test case
  lilicorr.bin  header for the lilicorr_test case
  manifest.json sizes, seeds, tolerances, and the SHA-256 of each payload

Layouts (little-endian, all arrays tightly packed in the order listed):

  ksc:      u32 rows | f32 scale | u16 s0[rows] | u8 codes[rows][16]
            | f32 alpha[rows] | f32 beta[64] | f32 x[64] | f32 expected[rows]
  awsrc:    u32 rows | u32 m_seeds | f32 inv_block | u32 seeds[m]
            | f32 gammas[rows][m] | f32 x[64] | f32 expected[rows]
  s0 is the 14-bit window over stream bits 0..13; the decode loop continues
  from bit offset 14.

  readkv:   u32 channels | u8 budgets[channels] | u32 planes[8]
            | u32 expected[channels]
  lilicorr: u32 groups | u32 k | u32 d_corr | u32 pairs | f32 tau
            | f32 uo[groups][k][d] | f32 ui[groups][k][d]
            | f32 expected[pairs][k][k]

The single-file binary format keeps the Swift harness free of JSON parsing and
keeps byte order explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import omegahee_metal_reference as R  # noqa: E402

TOLERANCES = {
    "ksc_decode_dot": 5e-4,
    "awsrc_dot": 5e-4,
    "readkv_prefix_load": 0.0,
    "lilicorr_score": 1e-4,
}

BLOCK = R.BLOCK


def _write(path: str, payload: bytes) -> str:
    with open(path, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def gen_ksc(out_dir: str, rows: int = 64, seed: int = 1) -> dict:
    case = R.make_ksc_case(rows=rows, seed=seed)
    parts = [
        struct.pack("<I", rows),
        struct.pack("<f", float(case["scale"])),
        np.asarray(case["s0"], dtype="<u2").tobytes(),
        b"".join(case["codes"]),
        np.asarray(case["alpha"], dtype="<f4").tobytes(),
        np.asarray(case["beta"], dtype="<f4").tobytes(),
        np.asarray(case["x"], dtype="<f4").tobytes(),
        np.asarray(case["expected"], dtype="<f4").tobytes(),
    ]
    sha = _write(os.path.join(out_dir, "ksc.bin"), b"".join(parts))
    return {"file": "ksc.bin", "rows": rows, "seed": seed, "sha256": sha}


def gen_awsrc(out_dir: str, rows: int = 64, m_seeds: int = 4, seed: int = 2) -> dict:
    case = R.make_awsrc_case(rows=rows, m_seeds=m_seeds, seed=seed)
    inv_block = 1.0 / (BLOCK ** 0.5)
    parts = [
        struct.pack("<I", rows),
        struct.pack("<I", m_seeds),
        struct.pack("<f", inv_block),
        np.asarray(case["seeds"], dtype="<u4").tobytes(),
        np.asarray(case["gammas"], dtype="<f4").tobytes(),
        np.asarray(case["x"], dtype="<f4").tobytes(),
        np.asarray(case["expected"], dtype="<f4").tobytes(),
    ]
    sha = _write(os.path.join(out_dir, "awsrc.bin"), b"".join(parts))
    return {"file": "awsrc.bin", "rows": rows, "m_seeds": m_seeds, "seed": seed, "sha256": sha}


def gen_readkv(out_dir: str, channels: int = 32, seed: int = 3) -> dict:
    case = R.make_readkv_case(channels=channels, seed=seed)
    parts = [
        struct.pack("<I", channels),
        np.asarray(case["budgets"], dtype="<u1").tobytes(),
        np.asarray(case["planes"], dtype="<u4").tobytes(),
        np.asarray(case["expected"], dtype="<u4").tobytes(),
    ]
    sha = _write(os.path.join(out_dir, "readkv.bin"), b"".join(parts))
    return {"file": "readkv.bin", "channels": channels, "seed": seed, "sha256": sha}


def gen_lilicorr(out_dir: str, groups: int = 4, k: int = 6, d_corr: int = 32,
                 tau: float = 2.0, seed: int = 4) -> dict:
    case = R.make_lilicorr_case(groups=groups, k=k, dc=d_corr, tau=tau, seed=seed)
    pairs = groups - 1
    parts = [
        struct.pack("<I", groups),
        struct.pack("<I", k),
        struct.pack("<I", d_corr),
        struct.pack("<I", pairs),
        struct.pack("<f", float(tau)),
        np.asarray(case["uo"], dtype="<f4").tobytes(),
        np.asarray(case["ui"], dtype="<f4").tobytes(),
        np.asarray(case["expected"], dtype="<f4").tobytes(),
    ]
    sha = _write(os.path.join(out_dir, "lilicorr.bin"), b"".join(parts))
    return {
        "file": "lilicorr.bin", "groups": groups, "k": k, "d_corr": d_corr,
        "pairs": pairs, "tau": tau, "seed": seed, "sha256": sha,
    }


def main() -> int:
    out_dir = os.path.join(_ROOT, "metal", "build", "vectors")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "generator": "metal/tools/gen_vectors.py",
        "reference": "src/omegahee_metal_reference.py",
        "block": BLOCK,
        "tolerances": TOLERANCES,
        "cases": {
            "ksc_decode_dot": gen_ksc(out_dir),
            "awsrc_dot": gen_awsrc(out_dir),
            "readkv_prefix_load": gen_readkv(out_dir),
            "lilicorr_score": gen_lilicorr(out_dir),
        },
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
