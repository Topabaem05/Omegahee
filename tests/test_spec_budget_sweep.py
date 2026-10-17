"""Tests for the measured pass-cost model and the budget sweep."""

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


S = _load("spec_budget_sim", "tools/spec_budget_sim.py")
W = _load("spec_budget_sweep", "tools/spec_budget_sweep.py")


class TestPassCostModel(unittest.TestCase):
    """The fit must reproduce both measured anchors before it predicts anything."""

    def test_single_token_pass_matches_measurement(self):
        # uzu with speculation off: 54.99 tok/s median -> 18.19 ms per pass.
        self.assertAlmostEqual(S.pass_time_ms(1), 18.19, delta=0.02)

    def test_budget_16_pass_matches_measurement(self):
        # uzu with the 16-slot tree: 47.4 ms per verify pass.
        self.assertAlmostEqual(S.pass_time_ms(16), 47.40, delta=0.02)

    def test_pass_time_grows_with_budget(self):
        times = [S.pass_time_ms(b) for b in (2, 4, 8, 16)]
        self.assertEqual(times, sorted(times))

    def test_tokens_per_second_uses_the_fit(self):
        # budget 4 with 1.21 accepted -> (2.21) / (22.36 ms) = 98.8 tok/s
        self.assertAlmostEqual(S.tokens_per_second(1.21, 4), 98.8, delta=1.0)


class TestBudgetSweep(unittest.TestCase):
    def _sweep(self, kind, seed=7):
        class Args:
            pass
        a = Args()
        a.tokens = 96
        a.k = 32
        a.seed = seed
        sharp = S.solve_sharpness_for_acceptance(1.65, 15, a.k)
        return W.sweep(kind, 16, a, sharp if kind == "calibrated" else None)

    def test_optimum_is_far_below_the_fixed_16(self):
        rows = self._sweep("calibrated")
        best = max(rows, key=lambda r: r["tokens_per_s"])
        self.assertLessEqual(best["budget"], 8)
        self.assertGreater(best["budget"], 1)

    def test_sixteen_is_not_the_optimum(self):
        rows = self._sweep("calibrated")
        at16 = [r for r in rows if r["budget"] == 16][0]
        best = max(rows, key=lambda r: r["tokens_per_s"])
        self.assertGreater(best["tokens_per_s"], at16["tokens_per_s"])

    def test_acceptance_saturates(self):
        rows = self._sweep("calibrated")
        # rows[0] is budget 2, so budget 4 is rows[2] and budget 16 is rows[-1].
        early = rows[2]["accepted"] - rows[0]["accepted"]
        late = rows[-1]["accepted"] - rows[-3]["accepted"]
        self.assertGreater(early, late * 5)

    def test_sweep_is_deterministic(self):
        a = self._sweep("code", seed=3)
        b = self._sweep("code", seed=3)
        self.assertEqual([r["tokens_per_s"] for r in a], [r["tokens_per_s"] for r in b])


if __name__ == "__main__":
    unittest.main()
