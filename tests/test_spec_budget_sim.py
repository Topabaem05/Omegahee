"""Tests for the SparseSpec-L budget simulator (tools/spec_budget_sim.py)."""

import importlib.util
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "spec_budget_sim", os.path.join(_ROOT, "tools", "spec_budget_sim.py"))
S = importlib.util.module_from_spec(_spec)
sys.modules["spec_budget_sim"] = S
_spec.loader.exec_module(S)


class TestExpectedAccepted(unittest.TestCase):
    def test_geometric_series(self):
        # Constant alpha = 0.5 gives sum_{t>=1} 0.5^t = 1.
        alpha = np.full(40, 0.5)
        self.assertAlmostEqual(S.expected_accepted(alpha), 1.0, places=6)

    def test_certain_chain_accepts_every_link(self):
        alpha = np.ones(5)
        self.assertAlmostEqual(S.expected_accepted(alpha), 5.0, places=12)

    def test_empty_chain_accepts_nothing(self):
        self.assertEqual(S.expected_accepted(np.array([])), 0.0)

    def test_monotone_in_alpha(self):
        low = S.expected_accepted(np.full(8, 0.3))
        high = S.expected_accepted(np.full(8, 0.7))
        self.assertLess(low, high)


class TestCalibration(unittest.TestCase):
    def test_solved_sharpness_matches_the_measurement(self):
        # The measured uzu figure is 2.65 tokens/forward at a fixed budget of 16,
        # i.e. 1.65 expected accepted tokens over a depth-15 chain.
        sharp = S.solve_sharpness_for_acceptance(1.65, 15, 32)
        rng = np.random.default_rng(1)
        chain = np.stack([rng.dirichlet(np.full(32, sharp)) for _ in range(15)])
        got = S.expected_accepted(np.exp(-S.P.normalized_candidate_entropy(chain)))
        self.assertAlmostEqual(got, 1.65, delta=0.15)

    def test_sharpness_is_positive(self):
        self.assertGreater(S.solve_sharpness_for_acceptance(0.5, 8, 16), 0.0)


class TestSimulate(unittest.TestCase):
    def _chains(self, kind, n=64, seed=11):
        rng = np.random.default_rng(seed)
        return [S.chain_profile(kind, 16, 32, rng, 0.0789) for _ in range(n)]

    def test_uniform_chains_collapse_the_budget(self):
        # alpha = exp(-1) = 0.368 at full entropy, so B0 * alpha * (1 + lambda_w)
        # = 8 * 0.368 * 1.5 = 4.4 -> floor 4. The point of the assertion is that a
        # uniformly uncertain chain gets a far smaller tree than the fixed 16.
        r = S.simulate(self._chains("uniform"), 0.35, 1.0, 8.0, 0.5, 2, 16, 16, 16)
        self.assertEqual(r["budget_histogram"], {4: 64})
        self.assertGreater(r["slot_reduction"], 0.7)

    def test_code_chains_take_deeper_chains_than_prose(self):
        code = S.simulate(self._chains("code"), 0.35, 1.0, 8.0, 0.5, 2, 16, 16, 16)
        prose = S.simulate(self._chains("prose"), 0.35, 1.0, 8.0, 0.5, 2, 16, 16, 16)
        code_gamma = max(code["gamma_histogram"])
        prose_gamma = max(prose["gamma_histogram"])
        self.assertGreater(code_gamma, prose_gamma)

    def test_fixed_baseline_is_independent_of_the_controller(self):
        a = S.simulate(self._chains("prose"), 0.35, 1.0, 8.0, 0.5, 2, 16, 16, 16)
        b = S.simulate(self._chains("prose"), 0.90, 1.0, 8.0, 0.5, 2, 16, 16, 16)
        self.assertAlmostEqual(a["tokens_per_forward_fixed"],
                               b["tokens_per_forward_fixed"], places=12)

    def test_report_is_json_serializable(self):
        import json
        r = S.simulate(self._chains("calibrated"), 0.35, 1.0, 8.0, 0.5, 2, 16, 16, 16)
        json.dumps(r)


if __name__ == "__main__":
    unittest.main()
