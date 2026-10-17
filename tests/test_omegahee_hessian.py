"""Tests for calibration-Hessian support in the KSC encoder."""

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


H = _load("omegahee_hessian", "src/omegahee_hessian.py")


class TestCovariance(unittest.TestCase):
    def test_matches_analytic_gram(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((64, 128))
        covs = H.block_diag_covariance(X, block=64)
        self.assertEqual(len(covs), 2)
        want = X[:, :64].T @ X[:, :64] / 64
        self.assertTrue(np.allclose(covs[0], want))

    def test_rejects_indivisible_k(self):
        with self.assertRaises(ValueError):
            H.block_diag_covariance(np.zeros((4, 100)), block=64)

    def test_rejects_1d_input(self):
        with self.assertRaises(ValueError):
            H.block_diag_covariance(np.zeros(64), block=64)


class TestCurvature(unittest.TestCase):
    def test_one_entry_per_block_with_unit_lower_triangular(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((128, 128))
        curv = H.layer_curvature(X, block=64, lam=1e-2)
        self.assertEqual(len(curv["blocks"]), 2)
        for entry in curv["blocks"]:
            L = entry["L"]
            self.assertTrue(np.allclose(np.diag(L), 1.0))
            self.assertTrue(np.allclose(np.triu(L, 1), 0.0))
            self.assertEqual(entry["d"].shape, (64,))

    def test_diagonal_hessian_gives_diagonal_d(self):
        # Independent columns: covariance is diagonal, so LDL should be trivial.
        X = np.eye(64) * 2.0
        curv = H.layer_curvature(X, block=64, lam=1e-8)
        L = curv["blocks"][0]["L"]
        self.assertTrue(np.allclose(L, np.eye(64), atol=1e-6))


class TestQuantizeWithHessian(unittest.TestCase):
    def test_none_activations_gives_plain_rounding(self):
        rng = np.random.default_rng(3)
        W = rng.standard_normal((32, 128)) * 0.08
        out = H.quantize_layer_with_hessian(W, None, block=64, beam_width=4)
        self.assertFalse(out["used_hessian"])
        self.assertEqual(out["packed_codes"].shape, (32, 32))
        self.assertTrue(np.isfinite(out["mse"]))

    def test_shapes_and_headers(self):
        rng = np.random.default_rng(4)
        W = rng.standard_normal((16, 128)) * 0.08
        X = rng.standard_normal((64, 128))
        out = H.quantize_layer_with_hessian(W, X, block=64, beam_width=2)
        self.assertEqual(out["hat"].shape, (16, 128))
        self.assertEqual(out["s0_headers"].shape, (16, 2))
        self.assertTrue(np.all(out["s0_headers"] <= 0x3FFF))
        self.assertEqual(out["scales"].shape, (16, 2))

    def test_rejects_indivisible_k(self):
        with self.assertRaises(ValueError):
            H.quantize_layer_with_hessian(np.zeros((4, 100)), None, block=64)

    def test_documented_risk_high_condition_hessian_hurts_mse(self):
        """A near-singular Hessian makes curvature weighting worse, not better.

        This is a regression guard on a finding rather than a feature: with a
        condition number near 1e9 the LDL d spans orders of magnitude, so the
        weighted objective diverges from unweighted MSE. If a future change makes
        this case competitive, update the assertion and the documentation.
        """
        rng = np.random.default_rng(5)
        rows, k, block = 64, 128, 64
        W = rng.standard_normal((rows, k)) * 0.08
        A = rng.standard_normal((128, 4))
        X = A @ rng.standard_normal((4, k)) + 0.05 * rng.standard_normal((128, k))
        with_h = H.quantize_layer_with_hessian(W, X, block=block, beam_width=4, lam=1e-3)
        zero = H.quantize_layer_with_hessian(W, None, block=block, beam_width=4)
        self.assertGreater(with_h["mse"], zero["mse"],
                           "documented: ill-conditioned Hessian hurts plain MSE")


if __name__ == "__main__":
    unittest.main()
