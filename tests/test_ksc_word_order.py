"""Byte-order contract tests for the packed KSC word layout."""

import importlib.util
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load("omegahee_metal_reference", "src/omegahee_metal_reference.py")
E = _load("omegahee_ksc_encoder", "src/omegahee_ksc_encoder.py")


class TestWordByteOrder(unittest.TestCase):
    """Apple GPUs read a byte pointer as little-endian uint, KSC is MSB-first.

    A kernel that casts codes to uint* and shifts from bit 30 reads the bytes in
    reverse order unless it byte-swaps. That was the cause of a parity failure
    fixed on 2026-10-18, so the ordering is pinned here.
    """

    def test_big_endian_word_reproduces_the_reference_codes(self):
        codes = list(range(64))
        packed = R.pack_2bit([c % 4 for c in codes])
        for word_index in range(4):
            chunk = packed[word_index * 4:(word_index + 1) * 4]
            be = int.from_bytes(chunk, "big")
            le = int.from_bytes(chunk, "little")
            for c in range(16):
                want = (codes[word_index * 16 + c]) % 4
                self.assertEqual((be >> (30 - 2 * c)) & 3, want)
            if any(b != chunk[0] for b in chunk):
                self.assertNotEqual(
                    [(le >> (30 - 2 * c)) & 3 for c in range(16)],
                    [codes[word_index * 16 + c] % 4 for c in range(16)],
                    "little-endian reading must differ on a non-uniform word",
                )

    def test_packer_matches_the_decoder_bit_order(self):
        rng = np.random.default_rng(11)
        codes = rng.integers(0, 4, size=(3, 64))
        packed = E.pack_ksc_codes_b64(codes)
        for row in range(3):
            raw = bytes(packed[row])
            for c in range(16):
                self.assertEqual(
                    R._pair(raw, 2 * c), int(codes[row][c]),
                    msg=f"row {row} code {c}",
                )

    def test_bswap32_inverts_little_endian_to_big(self):
        raw = bytes([0xBA, 0xD4, 0xFA, 0x00])
        le = int.from_bytes(raw, "little")
        be = int.from_bytes(raw, "big")
        # __builtin_bswap32(le) == be for a 4-byte word
        swapped = int.from_bytes(le.to_bytes(4, "little"), "big")
        self.assertEqual(swapped, be)


if __name__ == "__main__":
    unittest.main()
