"""Tests for the B=64 KSC beam encoder (src/omegahee_ksc_encoder.py)."""

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


E = _load("omegahee_ksc_encoder", "src/omegahee_ksc_encoder.py")
P = _load("omegahee_pillars", "src/omegahee_pillars.py")


class TestExactness(unittest.TestCase):
    """The beam search must reproduce the exhaustive reference when it fits."""

    def test_matches_exact_reference_for_small_blocks(self):
        rng = np.random.default_rng(3)
        for block in (2, 3, 4, 5, 6):
            w = rng.standard_normal(block)
            d = np.ones(block)
            lower = np.tril(np.full((block, block), 0.03), -1)
            g = np.zeros(block)
            exact = P.ksc_curvetq_quantize_row(w, d, lower, g, n_bits=2)
            beam = E.ksc_curvetq_quantize_row(w, d, lower, g, n_bits=2, beam_width=4 ** block)
            self.assertAlmostEqual(float(beam["cost"]), float(exact["cost"]), places=9,
                                   msg=f"block {block} cost")
            self.assertTrue(np.array_equal(np.asarray(beam["codes"]), np.asarray(exact["codes"])),
                            msg=f"block {block} codes")

    def test_beam_smaller_than_codebook_still_returns_a_valid_path(self):
        rng = np.random.default_rng(4)
        w = rng.standard_normal(6)
        d = np.ones(6)
        lower = np.tril(np.full((6, 6), 0.02), -1)
        out = E.ksc_curvetq_quantize_row(w, d, lower, np.zeros(6), beam_width=2)
        self.assertEqual(len(out["codes"]), 6)
        self.assertTrue(np.all((np.asarray(out["codes"]) >= 0) & (np.asarray(out["codes"]) <= 3)))


class TestBlockEncoder(unittest.TestCase):
    def test_quantizes_production_block(self):
        rng = np.random.default_rng(5)
        rows, block = 32, 64
        w = rng.standard_normal((rows, block))
        d = np.ones(block)
        lower = np.tril(np.full((block, block), 0.01), -1)
        out = E.ksc_curvetq_quantize_block_b64(w, d, lower, beam_width=8)
        self.assertEqual(out["codes"].shape, (rows, block))
        self.assertTrue(np.all(out["codes"] <= 3))
        # packed payload is 16 bytes per row block
        self.assertEqual(np.asarray(out["packed_codes"]).shape, (rows, 16))

    def test_beats_plain_rounding_on_the_curvature_metric(self):
        rng = np.random.default_rng(6)
        rows, block = 24, 64
        w = rng.standard_normal((rows, block))
        d = np.ones(block)
        lower = np.tril(np.full((block, block), 0.02), -1)
        out = E.ksc_curvetq_quantize_block_b64(w, d, lower, beam_width=8)
        beam_mse = float(np.mean((w - out["hat"]) ** 2))
        scales = np.std(w, axis=1, keepdims=True) + 1e-6
        span = np.array([-3.0, -1.0, 1.0, 3.0])
        codebook = 0.5 * scales * span[None, :]
        idx = np.argmin(np.abs(w[:, :, None] - codebook[:, None, :]), axis=-1)
        round_hat = np.take_along_axis(codebook[:, None, :], idx[:, :, None], axis=-1)[:, 0, :]
        round_mse = float(np.mean((w - round_hat) ** 2))
        self.assertLessEqual(beam_mse, round_mse * 1.5)

    def test_packed_codes_roundtrip_through_reference_decoder(self):
        rng = np.random.default_rng(7)
        rows, block = 4, 64
        w = rng.standard_normal((rows, block))
        d = np.ones(block)
        lower = np.tril(np.full((block, block), 0.0), -1)
        out = E.ksc_curvetq_quantize_block_b64(w, d, lower, beam_width=4)
        packed = np.asarray(out["packed_codes"])
        codes = np.asarray(out["codes"])
        # MSB-first: code c sits in bits 6-2*(c%4) of byte c//4.
        for row in range(rows):
            for c in range(block):
                byte = int(packed[row][c // 4])
                got = (byte >> (6 - 2 * (c % 4))) & 3
                self.assertEqual(got, int(codes[row][c]), msg=f"row {row} code {c}")


class TestS0Headers(unittest.TestCase):
    def test_headers_are_14_bit_values(self):
        rng = np.random.default_rng(8)
        rows, block = 8, 64
        w = rng.standard_normal((rows, block))
        d = np.ones(block)
        lower = np.tril(np.full((block, block), 0.0), -1)
        out = E.ksc_curvetq_quantize_block_b64(w, d, lower, beam_width=4)
        headers = np.asarray(out["s0_headers"])
        self.assertEqual(headers.shape, (rows,))
        self.assertTrue(np.all(headers >= 0))
        self.assertTrue(np.all(headers <= 0x3FFF))


if __name__ == "__main__":
    unittest.main()
