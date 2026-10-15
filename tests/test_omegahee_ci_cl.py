"""
CI/CL (Continuous Integration & Continuous Learning) Verification Suite
for Omegahee / Omega-Uzu v2.0
=======================================================================
Tests all mathematical invariants, convergence guarantees, QAT zero-loss
recovery, ALoDLM weight-sharing & depth-aware KV caching, HybridAgent
HybridLink, and Dream-RSI + LoGRA parametric self-improvement.
"""

import math
import os
import sys
import unittest
import numpy as np

import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "omegahee_uzu_engine.py"),
    os.path.join(_HERE, "..", "src", "omegahee_uzu_engine.py"),
]
_ENGINE_PATH = next(path for path in _CANDIDATES if os.path.isfile(path))
if os.path.dirname(_ENGINE_PATH) not in sys.path:
    sys.path.insert(0, os.path.dirname(_ENGINE_PATH))
_spec = importlib.util.spec_from_file_location("omegahee_uzu_engine", _ENGINE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["omegahee_uzu_engine"] = _mod
_spec.loader.exec_module(_mod)
from omegahee_uzu_engine import (
    ALoDLMHybridARModel,
    DepthAwareKVCache,
    DreamRSILoGRAEngine,
    ExplorationPolicy,
    HybridLinkAdapter,
    LittleBit2QATEngine,
    compute_bytecross_crossover_length,
    lilicorr_bilinear_tree_score,
    softmax,
)


class TestOmegaheeCICLSuite(unittest.TestCase):

    def test_01_littlebit2_joint_itq_and_zero_loss_qat(self) -> None:
        rng = np.random.default_rng(42)
        m, n, r = 32, 32, 16
        u0, _ = np.linalg.qr(rng.standard_normal((m, m)))
        v0, _ = np.linalg.qr(rng.standard_normal((n, n)))
        spectrum = np.exp(-np.linspace(0.0, 3.5, min(m, n)))
        w_teacher = (u0[:, :min(m, n)] * spectrum) @ v0[:, :min(m, n)].T

        x_calib = rng.standard_normal((64, n))
        engine = LittleBit2QATEngine(rank=r, itq_iters=25, beta_smooth=3.0, seed=42)

        u_rot, v_rot, rot, itq_losses = engine.joint_itq(w_teacher, rank=r)

        ident_err = np.linalg.norm(rot @ rot.T - np.eye(r), ord="fro")
        self.assertLess(ident_err, 1e-11, "Internal latent rotation R must be strictly orthogonal.")

        u_svd, s_svd, vt_svd = np.linalg.svd(w_teacher, full_matrices=False)
        l1 = u_svd[:, :r] * np.sqrt(s_svd[:r])[None, :]
        l2 = vt_svd[:r, :].T * np.sqrt(s_svd[:r])[None, :]
        prod_unrotated = l1 @ l2.T
        prod_rotated = u_rot @ v_rot.T
        self.assertLess(
            np.linalg.norm(prod_rotated - prod_unrotated, ord="fro"),
            1e-11,
            "Internal Latent Rotation must preserve exact low-rank matrix product.",
        )

        self.assertLessEqual(itq_losses[-1], itq_losses[0])

        w_recovered, qat_res = engine.run_qat_with_l2l_recovery(
            w=w_teacher,
            x_calib=x_calib,
            steps=35,
            lr=0.04,
            l2l_loss_scale=10.0,
            num_seeds=8,
        )
        self.assertLess(qat_res.qat_loss_history[-1], qat_res.qat_loss_history[0] * 0.05)

        y_teacher = x_calib @ w_teacher.T
        y_student = x_calib @ w_recovered.T
        top1_agreement = float(np.mean(np.argmax(y_teacher, axis=-1) == np.argmax(y_student, axis=-1)))
        self.assertGreaterEqual(top1_agreement, 0.98, f"Expected >=98% top-1 agreement, got {top1_agreement:.4f}")

    def test_02_alodlm_weight_sharing_and_depth_aware_kv_cache(self) -> None:
        cache_8b = DepthAwareKVCache(num_layers=36, start_rec=10, end_rec=26, max_depth=4)
        self.assertEqual(cache_8b.total_logical_buffer_pairs(), 84)

        model = ALoDLMHybridARModel(
            vocab_size=48,
            d_model=24,
            n_prelude=2,
            n_core_shared=3,
            n_coda=2,
            max_recurrent_depth=4,
            seed=99,
        )
        self.assertEqual(model.effective_layer_evaluations(4), 2 + 4 * 3 + 4 * 2)

        kv_cache = DepthAwareKVCache(num_layers=7, start_rec=2, end_rec=5, max_depth=4)
        k_d1 = np.ones((4, 24)) * 1.0
        v_d1 = np.ones((4, 24)) * 1.0
        k_d2 = np.ones((4, 24)) * 2.0
        v_d2 = np.ones((4, 24)) * 2.0
        kv_cache.write(layer=3, depth=1, k_mat=k_d1, v_mat=v_d1)
        kv_cache.write(layer=3, depth=2, k_mat=k_d2, v_mat=v_d2)
        kv_cache.deepest_available[(3, 0)] = 1
        kv_cache.deepest_available[(3, 1)] = 2

        k_read, _ = kv_cache.read_with_fallback(layer=3, depth=3, seq_len=2, query_vec=None, read_bits=8)
        self.assertTrue(np.allclose(k_read[0], 1.0))
        self.assertTrue(np.allclose(k_read[1], 2.0))

        seq = np.array([3, 7, model.mask_token_id, 12, model.mask_token_id, model.mask_token_id], dtype=np.int64)
        decoded, stats = model.adaptive_block_ar_decode(seq, entropy_tau=3.6, halt_threshold_q=0.7)
        self.assertFalse(np.any(decoded == model.mask_token_id))
        self.assertGreaterEqual(stats["mean_commit_depth"], 1.0)
        self.assertLessEqual(stats["mean_commit_depth"], 4.0)

    def test_03_alodlm_variational_nelbo_and_control_variate(self) -> None:
        prior = ALoDLMHybridARModel.truncated_geometric_prior(max_depth=4, c=0.4)
        expected_prior = np.array([0.4131, 0.2769, 0.1856, 0.1244])
        self.assertTrue(np.allclose(prior, expected_prior, atol=1e-3))
        mean_depth = float(np.sum(np.arange(1, 5) * prior))
        self.assertAlmostEqual(mean_depth, 2.02, places=2)

        model = ALoDLMHybridARModel(vocab_size=32, d_model=16, seed=17)
        y_clean = np.array([2, 5, 9, 11, 14, 3], dtype=np.int64)
        x_corr = np.array([2, model.mask_token_id, 9, model.mask_token_id, 14, model.mask_token_id], dtype=np.int64)
        losses = model.compute_alodlm_surrogate_loss(x_corr, y_clean, mask_prob_t=0.5)
        self.assertGreater(losses["L_den"], 0.0)
        self.assertGreater(losses["L_AR"], 0.0)
        self.assertTrue(math.isfinite(losses["L_impl"]))

    def test_04_hybrid_agent_link_and_working_memory(self) -> None:
        rng = np.random.default_rng(77)
        d_model = 24
        vocab_size = 40
        embeddings = rng.standard_normal((vocab_size, d_model))
        link = HybridLinkAdapter(d_model=d_model, latent_entry_id=37, latent_exit_id=38, seed=77)

        h_lat = rng.standard_normal((8, d_model))
        target_emb = embeddings[np.arange(8)]
        loss_init = link.compute_hybrid_link_loss(h_lat, target_emb)
        self.assertGreater(loss_init, 0.0)

        exit_vec = embeddings[38] * 5.0
        link_zero = HybridLinkAdapter(d_model=d_model, latent_entry_id=37, latent_exit_id=38, seed=77)
        link_zero.w2[:] = 0.0
        should_exit, nearest_id = link_zero.should_exit_latent_mode(exit_vec, embeddings)
        self.assertTrue(should_exit)
        self.assertEqual(nearest_id, 38)

    def test_05_cl_dream_rsi_and_logra_parametric_weight_consolidation(self) -> None:
        d_model = 32
        vocab_size = 48
        rsi_engine = DreamRSILoGRAEngine(d_model=d_model, vocab_size=vocab_size, sketch_rank=6, seed=2026)

        rng = np.random.default_rng(2026)
        init_w = rng.standard_normal(d_model)
        init_w /= np.linalg.norm(init_w)
        policy = ExplorationPolicy(policy_id="pi_init", weights=init_w, branch_width=3, stop_threshold=-0.1)
        w_lm_head = rng.standard_normal((vocab_size, d_model)) * (1.0 / math.sqrt(d_model))

        kl_budget = 5e-4
        round_gains = []

        for round_idx in range(1, 5):
            rsi_engine.online_explore_and_record_world(policy, num_nodes=20)
            new_policy, pre_dream_score, post_dream_score, elites = rsi_engine.improve_policy_by_dreaming(
                current_policy=policy,
                num_revisions=36,
                mutation_scale=0.3,
            )
            self.assertGreaterEqual(
                post_dream_score,
                pre_dream_score - 1e-12,
                f"Dream-RSI round {round_idx} violated monotonic improvement!",
            )
            round_gains.append(post_dream_score - pre_dream_score)
            policy = new_policy

            logra_res = rsi_engine.consolidate_rsi_into_weights_logra(
                w_lm_head=w_lm_head,
                elite_nodes=elites,
                step_size_eta=0.25,
                kl_budget_delta=kl_budget,
                alpha_max=1.0,
                seed=100 + round_idx,
            )
            self.assertLessEqual(float(logra_res["predicted_kl"]), kl_budget + 1e-9)
            self.assertLess(float(logra_res["post_rsi_loss"]), float(logra_res["initial_rsi_loss"]))
            self.assertLess(float(logra_res["memory_ratio_sketch_vs_dense"]), 0.25)
            w_lm_head = logra_res["w_updated"]

        self.assertEqual(len(rsi_engine.history_worlds), 4)
        self.assertGreater(sum(round_gains), 0.02)

    def test_06_lilicorr_and_bytecross_uzu_kernels(self) -> None:
        t_star = compute_bytecross_crossover_length(
            rho_act=0.72,
            rho_kv=0.25,
            b_proj_bytes=2.68e9,
            b_kv_per_token_bytes=131072.0,
        )
        self.assertGreater(t_star, 5000.0)

        rng = np.random.default_rng(11)
        u_out = rng.standard_normal((5, 8, 16))
        u_in = rng.standard_normal((5, 8, 16))
        marginals = np.log(softmax(rng.standard_normal((5, 8)), axis=-1))
        s_mat, chosen = lilicorr_bilinear_tree_score(u_out, u_in, marginals)
        self.assertEqual(s_mat.shape, (4, 8, 8))
        self.assertEqual(len(chosen), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
