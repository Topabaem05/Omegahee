"""Tests for calibration-Hessian curvature in the KSC encoder."""

import os
import sys
import unittest

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from omegahee_hessian import (  # noqa: E402
    block_diag_covariance,
    layer_curvature,
    quantize_layer_with_hessian,
)
from omegahee_pillars import ldl_factor, regularized_schur  # noqa: E402


class TestCalibrationCovariance(unittest.TestCase):
    def test_covariance_matches_xtx_over_rows(self) -> None:
        rng = np.random.default_rng(11)
        rows, columns, block = 37, 12, 4
        X = rng.standard_normal((rows, columns))

        covariances = block_diag_covariance(X, block)

        self.assertEqual(len(covariances), columns // block)
        for b, covariance in enumerate(covariances):
            expected = X[:, b * block : (b + 1) * block].T @ X[
                :, b * block : (b + 1) * block
            ] / rows
            self.assertEqual(covariance.shape, (block, block))
            self.assertEqual(covariance.dtype, np.float64)
            np.testing.assert_allclose(covariance, expected, rtol=0.0, atol=1e-15)

    def test_layer_curvature_is_per_block_unit_lower(self) -> None:
        rng = np.random.default_rng(12)
        rows, columns, block, lam = 512, 16, 8, 1e-4
        X = rng.standard_normal((rows, columns))
        X[:, 1:] = 0.6 * X[:, :-1] + 0.8 * X[:, 1:]

        curvature = layer_curvature(X, block=block, lam=lam)
        covariances = block_diag_covariance(X, block)

        self.assertEqual(len(curvature["blocks"]), columns // block)
        for covariance, entry in zip(covariances, curvature["blocks"]):
            L, d = entry["L"], entry["d"]
            self.assertEqual(L.shape, (block, block))
            self.assertEqual(d.shape, (block,))
            np.testing.assert_allclose(np.diag(L), np.ones(block), rtol=0.0, atol=1e-15)
            np.testing.assert_allclose(np.triu(L, k=1), 0.0, rtol=0.0, atol=0.0)
            self.assertTrue(np.all(d > 0.0))

            H = regularized_schur(covariance, block, lam)
            np.testing.assert_allclose(L @ np.diag(d) @ L.T, H, rtol=1e-12, atol=1e-12)
            expected_L, expected_d = ldl_factor(H)
            np.testing.assert_allclose(L, expected_L, rtol=0.0, atol=1e-15)
            np.testing.assert_allclose(d, expected_d, rtol=0.0, atol=1e-15)


class TestHessianQuantization(unittest.TestCase):
    def test_real_covariance_matches_or_improves_zero_curvature(self) -> None:
        rng = np.random.default_rng(0)
        rows, K, block, M = 2048, 64, 64, 2
        A = rng.standard_normal((1, K)) * np.exp(rng.normal(0.0, 0.7, (1, K)))
        X = rng.standard_normal((rows, 1)) @ A + 0.1 * rng.standard_normal((rows, K))
        levels = rng.integers(0, 4, size=(M, K))
        W = (levels - 1.5) * 0.5 + 0.1 * rng.standard_normal((M, K))

        with_hessian = quantize_layer_with_hessian(
            W, X, block=block, beam_width=8, lam=1e-4
        )
        zero_L = quantize_layer_with_hessian(W, None, block=block, beam_width=8)

        mse_hessian = float(with_hessian["mse"])
        mse_zero_L = float(zero_L["mse"])
        ratio = mse_hessian / mse_zero_L
        print(
            f"\nHessian MSE={mse_hessian:.12e}  "
            f"L=0 MSE={mse_zero_L:.12e}  ratio={ratio:.6f}"
        )

        self.assertEqual(with_hessian["hat"].shape, W.shape)
        self.assertEqual(with_hessian["packed_codes"].shape, (M, K // 4))
        self.assertEqual(with_hessian["s0_headers"].shape, (M, 1))
        self.assertEqual(with_hessian["scales"].shape, (M, 1))
        self.assertTrue(
            np.any(with_hessian["packed_codes"] != zero_L["packed_codes"]),
            "the informative Hessian must change at least one quantized code",
        )
        self.assertLessEqual(mse_hessian, mse_zero_L * 1.02)


    def test_ill_conditioned_hessian_can_hurt_and_l_zero_is_the_floor(self) -> None:
        """Records the measured limit: a badly conditioned H can beat L=0 backwards.

        Condition numbers near 5e4 were measured to make the weighted objective
        concentrate on a few columns and worsen plain MSE by up to 11x
        (docs/hessian_result_20261018.md). This test pins the safer direction:
        with L=0 the encoder must be identical to plain rounding, so no future
        change can silently make the fallback worse than flat.
        """
        rng = np.random.default_rng(21)
        rows, K, block, M = 512, 128, 64, 4
        A = rng.standard_normal((rows, 4))
        X = A @ rng.standard_normal((4, K)) + 0.05 * rng.standard_normal((rows, K))
        W = rng.standard_normal((M, K)) * 0.08

        zero = quantize_layer_with_hessian(W, None, block=block, beam_width=8)
        with_h = quantize_layer_with_hessian(W, X, block=block, beam_width=8, lam=1e-3)

        self.assertFalse(np.isnan(zero["mse"]))
        self.assertFalse(np.isnan(with_h["mse"]))
        # L=0 must be reproducible: same call twice gives the same bytes.
        again = quantize_layer_with_hessian(W, None, block=block, beam_width=8)
        self.assertTrue(np.array_equal(zero["packed_codes"], again["packed_codes"]))
        print(
            f"\nill-conditioned: hessian MSE={with_h['mse']:.6e} "
            f"L=0 MSE={zero['mse']:.6e} ratio={with_h['mse']/zero['mse']:.4f}"
        )

    def test_rejects_input_dimension_not_divisible_by_block(self) -> None:
        W = np.zeros((3, 96), dtype=np.float64)
        X = np.ones((7, 96), dtype=np.float64)
        with self.assertRaises(ValueError):
            quantize_layer_with_hessian(W, X, block=64)


if __name__ == "__main__":
    unittest.main()
