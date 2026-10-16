"""Additional spec-conformance tests derived from paper/uzu_omega_mathematical_algorithm_2026.tex."""
import math
import os
import sys
import importlib.util
import unittest
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "omegahee_uzu_engine", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "omegahee_uzu_engine.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["omegahee_uzu_engine"] = _mod
_spec.loader.exec_module(_mod)
E = _mod


class TestSpecExtra(unittest.TestCase):
    def test_hybridlink_w2_zero_init_identity(self):
        link = E.HybridLinkAdapter(d_model=16, latent_entry_id=1, latent_exit_id=2, seed=3)
        h = np.random.default_rng(0).standard_normal((5, 16))
        self.assertTrue(np.allclose(link.w2, 0.0))
        self.assertTrue(np.allclose(link.forward(h), h))  # R(h)=h at init

    def test_stick_breaking_halting_distribution_sums_to_one(self):
        g = np.array([0.3, 0.5, 0.2])  # g^(1..Dmax-1)
        q = [g[0], g[1] * (1 - g[0]), g[2] * (1 - g[0]) * (1 - g[1]), np.prod(1 - g)]
        self.assertAlmostEqual(float(np.sum(q)), 1.0, places=12)
        # cumulative halting mass equals 1 - prod(1-g) (the a_cum recursion used in decode)
        a = 0.0
        for k in range(3):
            a = a + (1 - a) * g[k]
            self.assertAlmostEqual(a, float(np.sum(q[: k + 1])), places=12)

    def test_decode_commits_on_halting_mass_even_with_high_entropy(self):
        model = E.ALoDLMHybridARModel(vocab_size=48, d_model=24, seed=99)
        model.exit_gate_b[0] = 40.0  # g ~ 1 -> halting mass >= q at depth 1
        seq = np.array([3, model.mask_token_id, model.mask_token_id], dtype=np.int64)
        out, stats = model.adaptive_block_ar_decode(seq, entropy_tau=-1.0, halt_threshold_q=0.5)
        self.assertFalse(np.any(out == model.mask_token_id))
        self.assertEqual(stats["mean_commit_depth"], 1.0)
        self.assertEqual(stats["total_inner_passes"], 1.0)

    def test_decode_with_and_without_kv_cache_completes(self):
        model = E.ALoDLMHybridARModel(vocab_size=48, d_model=24, seed=99)
        seq = np.array([3, 7, model.mask_token_id, 12, model.mask_token_id], dtype=np.int64)
        for flag in (True, False):
            out, stats = model.adaptive_block_ar_decode(seq, entropy_tau=3.6, halt_threshold_q=0.7, use_kv_cache=flag)
            self.assertFalse(np.any(out == model.mask_token_id))
            self.assertTrue(np.all(out[[0, 1, 3]] == seq[[0, 1, 3]]))  # unmasked tokens untouched

    def test_kv_buffer_pairs_formula(self):
        # L_start + L_coda non-recurrent + L_core * D_max
        for L, s, e, D in [(36, 10, 26, 4), (7, 2, 5, 4), (12, 3, 9, 2)]:
            c = E.DepthAwareKVCache(num_layers=L, start_rec=s, end_rec=e, max_depth=D)
            self.assertEqual(c.total_logical_buffer_pairs(), (L - (e - s)) + (e - s) * D)

    def test_kv_fallback_min_d_D_all_depths(self):
        c = E.DepthAwareKVCache(num_layers=7, start_rec=2, end_rec=5, max_depth=4)
        for d in range(1, 5):
            c.write(3, d, np.full((4, 8), float(d)), np.full((4, 8), float(d)))
        for pos, D in enumerate([1, 2, 3, 4]):
            c.deepest_available[(3, pos)] = D
        for d in range(1, 5):
            k, v = c.read_with_fallback(3, d, 4, query_vec=None, read_bits=8)
            for pos, D in enumerate([1, 2, 3, 4]):
                self.assertTrue(np.allclose(k[pos], min(d, D)))
                self.assertTrue(np.allclose(v[pos], min(d, D)))

    def test_joint_itq_procrustes_orthogonal_and_monotone(self):
        w = np.random.default_rng(1).standard_normal((24, 20))
        eng = E.LittleBit2QATEngine(rank=8, itq_iters=30)
        u, v, R, hist = eng.joint_itq(w)
        self.assertLess(np.linalg.norm(R @ R.T - np.eye(8)), 1e-10)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(hist, hist[1:])))
        # rotation is absorbed: U_rot V_rot^T equals rank-r SVD approximation
        U, S, Vt = np.linalg.svd(w, full_matrices=False)
        self.assertTrue(np.allclose(u @ v.T, (U[:, :8] * S[:8]) @ Vt[:8]))

    def test_smoothsign_hard_forward(self):
        x = np.array([-2.0, -0.1, 0.0, 0.3])
        self.assertTrue(np.array_equal(E.LittleBit2QATEngine.smooth_sign(x), np.array([-1, -1, 1, 1.0])))

    def test_bytecross_balance_equation(self):
        ra, rk, B, b = 0.72, 0.25, 2.68e9, 131072.0
        t = E.compute_bytecross_crossover_length(ra, rk, B, b)
        self.assertAlmostEqual((1 - ra) * B, (1 - rk) * b * t, delta=1e-3 * B)
        self.assertAlmostEqual(t, 7633.5, places=0)

    def test_lilicorr_matches_bruteforce_greedy(self):
        rng = np.random.default_rng(5)
        g, K, dc = 4, 6, 16
        uo, ui = rng.standard_normal((g, K, dc)), rng.standard_normal((g, K, dc))
        lp = E.log_softmax(rng.standard_normal((g, K)))
        S, chosen = E.lilicorr_bilinear_tree_score(uo, ui, lp, tau_corr=2.0)
        for t in range(g - 1):
            a, b = uo[t] / np.linalg.norm(uo[t], axis=-1, keepdims=True), ui[t + 1] / np.linalg.norm(ui[t + 1], axis=-1, keepdims=True)
            self.assertTrue(np.allclose(S[t], 2.0 * a @ b.T, atol=1e-6))
            self.assertLessEqual(np.max(np.abs(S[t])), 2.0 + 1e-9)  # |cos|<=1
        self.assertEqual(chosen[0], int(np.argmax(lp[0])))

    def test_predicted_kl_bound(self):
        rng = np.random.default_rng(2)
        w, dw, ctx = rng.standard_normal((10, 6)), rng.standard_normal((10, 6)), rng.standard_normal((5, 6))
        q, a = E.DreamRSILoGRAEngine.compute_predicted_kl_and_scale(w, dw, ctx, kl_budget_delta=1e-4)
        self.assertLessEqual(a * a * q, 1e-4 + 1e-12)
        real = E.kl_divergence_logits(ctx @ w.T, ctx @ (w + a * dw).T)
        self.assertLess(abs(real - a * a * q), 5e-5)  # 2nd-order approx is close for small step


if __name__ == "__main__":
    unittest.main()
