"""CI checks for the four Omega-Uzu pillars that were specified but not coded."""

import math
import os
import sys
import unittest

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from omegahee_pillars import (  # noqa: E402
    aatc_distortion,
    aatc_reverse_waterfill,
    awsrc_seeded_repair,
    bytecross_crossover_length,
    bytecross_route,
    bytecross_speedup,
    casa_closed_form_bits,
    casa_discrete_assign,
    casa_kronecker_energy,
    ContinuousWorkingMemory,
    gdn_commit_chain,
    gdn_tree_scan,
    gumbel_argmax,
    gumbel_clamp_uniforms,
    hybrid_sft_two_pass,
    ksc_curvetq_quantize_row,
    ldl_factor,
    lilicorr_joint_loglik,
    memory_step_bytes,
    mirai_s_effective_bpw,
    omega_roofline_factors,
    optr_reparameterize,
    attention_probs,
    plan_disaggregated_quantization,
    readkv_key_bits,
    readkv_value_bits,
    regularized_schur,
    run_pillar_reference_audit,
    sparsespec_controller,
    stagq_dyadic_decode,
    stagq_dyadic_encode,
    weight_traffic_reduction,
)


class TestOmegaPillars(unittest.TestCase):
    def test_07_casa_mean_and_qk_protection(self) -> None:
        energies = [casa_kronecker_energy(4, 2, 8, 5), casa_kronecker_energy(12, 6, 40, 20)]
        params = [1.0e6, 4.0e6]
        bits = casa_closed_form_bits(energies, params, b_target=2.4)
        weighted = float(np.sum(np.array(params) * bits) / np.sum(params))
        self.assertAlmostEqual(weighted, 2.4, places=8)
        assigned = casa_discrete_assign([2.1, 1.4, 2.2], ["W_Q", "W_K", "W_down"])
        self.assertGreaterEqual(assigned[0], 3)
        self.assertGreaterEqual(assigned[1], 3)
        self.assertEqual(assigned[2], 2)

    def test_08_schur_ldl_and_rotation_free_viterbi(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.standard_normal((10, 10))
        cov = base @ base.T + 0.4 * np.eye(10)
        H = regularized_schur(cov, block=6, lam=1e-3)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(H))), -1e-8)
        L, d = ldl_factor(H)
        recon = L @ np.diag(d) @ L.T
        self.assertLess(np.linalg.norm(recon - H), 1e-8)
        w = rng.standard_normal(6)
        quant = ksc_curvetq_quantize_row(w, d, L, g_row=rng.standard_normal(6) * 0.01, Bii=1.2, eta=0.15)
        self.assertLessEqual(float(quant["cost"]), float(quant["euclidean_cost"]) + 1e-8)
        self.assertFalse(quant["rotation_applied"])
        self.assertEqual(quant["header_bits"], 14)
        bpw = mirai_s_effective_bpw()
        self.assertAlmostEqual(bpw, 2.0 + 14.0 / 64.0 + 0.162, places=6)
        self.assertAlmostEqual(weight_traffic_reduction(), (4.613 - bpw) / 4.613, places=10)
        self.assertGreater(weight_traffic_reduction(), 0.48)
        self.assertLess(weight_traffic_reduction(), 0.49)

    def test_09_stagq_monotone_and_awsrc_energy(self) -> None:
        rng = np.random.default_rng(1)
        residual = rng.standard_normal((5, 4))
        encoded = stagq_dyadic_encode(residual, n_planes=3)
        mses = encoded["mse_prefix"]
        self.assertGreaterEqual(mses[0] + 1e-12, mses[1])
        self.assertGreaterEqual(mses[1] + 1e-12, mses[2])
        self.assertGreaterEqual(mses[2] + 1e-12, mses[3])
        prefix = stagq_dyadic_decode(np.zeros_like(residual), encoded["planes"], encoded["gammas"], 1)
        full = stagq_dyadic_decode(np.zeros_like(residual), encoded["planes"], encoded["gammas"], 3)
        self.assertLess(np.mean((residual - full) ** 2), np.mean((residual - prefix) ** 2) + 1e-12)
        repaired = awsrc_seeded_repair(encoded["final_residual"], np.eye(4), num_seeds=4)
        self.assertLess(repaired["end_energy"], repaired["start_energy"])
        self.assertEqual(repaired["stored_bpw"], 0.162)

    def test_10_dq_and_bytecross(self) -> None:
        plan = plan_disaggregated_quantization(4096, resident_decode_bytes=8.11e9, layer_bytes=2e8, tree_width=1)
        self.assertEqual(plan["decode_activation"], "bf16")
        self.assertFalse(plan["activation_quant_on_decode"])
        self.assertEqual(plan["prefill_activation"], "int8")
        self.assertEqual(plan["odp_ring_layers"], 2)
        self.assertEqual(plan["odp_ring_bytes"], 4e8)
        narrow = plan_disaggregated_quantization(4, 1.0, 10.0, tree_width=1)
        self.assertEqual(narrow["prefill_activation"], "bf16")
        t_star = bytecross_crossover_length(0.72, 0.25, 2.68e9, 131072.0)
        self.assertAlmostEqual(t_star, 7633.5, places=1)
        self.assertEqual(bytecross_route(100, t_star, t_min=512), "activation_only")
        self.assertEqual(bytecross_route(9000, t_star, t_min=512), "kv_sparse_splitk")
        speed = bytecross_speedup(1024, 0.72, 0.25, 2.0e9, 1.0e5, 1.0e11, 0.0, 0.0, t_min=10_000)
        self.assertGreater(speed, 1.0)

    def test_11_sparsespec_lilicorr_gdn_gumbel(self) -> None:
        peaked = np.array([[0.97, 0.01, 0.01, 0.01]] * 4)
        flat = np.full((4, 4), 0.25)
        low = sparsespec_controller(peaked, c_draft=0.15, b_min=2, b_max=16)
        high = sparsespec_controller(flat, c_draft=0.15, b_min=2, b_max=16)
        self.assertGreater(low["gamma_star"], high["gamma_star"])
        self.assertGreaterEqual(low["tree_budget"], 2)
        self.assertLessEqual(low["tree_budget"], 16)
        s_mat = np.zeros((1, 3, 3))
        s_mat[0, 0, 1] = 5.0
        marginals = np.log(np.array([[0.5, 0.3, 0.2], [0.2, 0.7, 0.1]]))
        score_good = lilicorr_joint_loglik(marginals, s_mat, [0, 1])
        score_bad = lilicorr_joint_loglik(marginals, s_mat, [0, 2])
        self.assertGreater(score_good, score_bad)

        rng = np.random.default_rng(3)
        root = np.eye(3)
        k1 = rng.standard_normal(3); k1 /= np.linalg.norm(k1)
        v1 = rng.standard_normal(3)
        k2 = rng.standard_normal(3); k2 /= np.linalg.norm(k2)
        v2 = rng.standard_normal(3)
        k3 = rng.standard_normal(3); k3 /= np.linalg.norm(k3)
        nodes = {
            1: {"parent": 0, "alpha": 0.8, "beta": 0.3, "k": k1, "v": v1},
            2: {"parent": 1, "alpha": 0.7, "beta": 0.25, "k": k2, "v": v2},
            3: {"parent": 1, "alpha": 0.6, "beta": 0.1, "k": k3, "v": v2},
        }
        scanned = gdn_tree_scan(root, nodes)
        committed = gdn_commit_chain(root, [nodes[1], nodes[2]])
        self.assertTrue(np.allclose(scanned["states"][2], committed, atol=1e-12))
        self.assertEqual(scanned["gather_index"][0], (1, 0))
        # Rejected sibling is scanned but is not the committed chain state.
        self.assertFalse(np.allclose(scanned["states"][3], committed))

        u = np.array([1e-12, 0.5, 2.0])
        clamped = gumbel_clamp_uniforms(u)
        self.assertGreaterEqual(float(clamped.min()), 2.0 ** -24)
        self.assertLessEqual(float(clamped.max()), 1.0 - 2.0 ** -24)
        logits = np.array([0.0, 0.2, 0.1, -0.3])
        draw = np.array([0.15, 0.4, 0.7, 0.9])
        self.assertEqual(gumbel_argmax(logits, draw), gumbel_argmax(logits, draw + 1e-10))

    def test_12_optr_aatc_readkv_memory_roofline(self) -> None:
        rng = np.random.default_rng(4)
        keys = rng.standard_normal((16, 8)) + np.array([3.0, -2.0, 0, 0, 1, 0, 0, 0])
        queries = rng.standard_normal((2, 8))
        values = rng.standard_normal((16, 8))
        opt = optr_reparameterize(keys, queries, values)
        before = attention_probs(queries, keys)
        after = attention_probs(opt["queries"], opt["keys"])
        self.assertLess(float(np.max(np.abs(before - after))), 1e-10)
        self.assertLess(np.linalg.norm(opt["U"].T @ opt["U"] - np.eye(8)), 1e-10)

        xi = np.array([8.0, 2.0, 0.2, 0.02])
        sigma2 = np.ones(4)
        bits = aatc_reverse_waterfill(xi, sigma2, total_bits=6)
        uniform = np.array([1, 1, 2, 2])
        self.assertLessEqual(aatc_distortion(xi, sigma2, bits), aatc_distortion(xi, sigma2, uniform) + 1e-9)
        self.assertGreater(int(bits[0]), int(bits[-1]))

        key_bits = readkv_key_bits(np.array([4.0, 0.5, 0.1, 0.01]), np.ones(4), b_read=4.0)
        self.assertGreaterEqual(int(key_bits[0]), int(key_bits[-1]))
        self.assertLessEqual(float(np.mean(key_bits)), 4.0 + 1e-8)
        val_bits = readkv_value_bits(np.array([0.8, 0.15, 0.04, 0.01]), np.ones(4), b_bar=3.0)
        self.assertGreaterEqual(int(val_bits[0]), int(val_bits[-1]))
        self.assertTrue(set(int(x) for x in val_bits).issubset({0, 2, 4, 8}))

        mem = ContinuousWorkingMemory()
        mem.commit_turn(np.ones((3, 2)), np.ones((3, 2)))
        mem.commit_turn(np.ones((2, 2)) * 2, np.ones((2, 2)) * 2)
        cat_k, cat_v = mem.concatenated()
        self.assertEqual(cat_k.shape, (5, 2))
        self.assertEqual(mem.turn_count, 2)
        self.assertTrue(np.allclose(cat_v[3:], 2.0))

        step = memory_step_bytes(n_params=16, bits_per_weight=8, n_layers=1, context=1, d_kv=1, bits_per_kv=8)
        self.assertAlmostEqual(step, 16.0 + 2.0, places=8)
        roof = omega_roofline_factors()
        self.assertAlmostEqual(roof["traffic_reduction"], (4.613 - 2.382) / 4.613, places=6)
        self.assertAlmostEqual(roof["weight_byte_fraction"], 0.418, delta=0.002)
        self.assertAlmostEqual(roof["single_pass_speedup"], 2.39, delta=0.02)

        losses = hybrid_sft_two_pass(
            hidden=np.zeros(4),
            target_emb=np.ones(4),
            w1=np.eye(4),
            w2=np.zeros((4, 4)),
            action_logits=np.array([0.0, 2.0, 0.0]),
            action_target=1,
        )
        self.assertAlmostEqual(losses["L_HybridSFT"], losses["L_link"] + losses["L_act"], places=8)
        self.assertGreater(losses["L_link"], 0.0)

    def test_13_reference_audit_bundle(self) -> None:
        audit = run_pillar_reference_audit(seed=11)
        self.assertAlmostEqual(float(audit["casa_mean_bits"]), 2.4, places=6)
        self.assertTrue(audit["qk_protected"])
        self.assertTrue(audit["viterbi_beats_euclidean"])
        self.assertFalse(audit["rotation_applied"])
        self.assertGreater(float(audit["awsrc_energy_drop"]), 0.0)
        self.assertEqual(audit["decode_activation"], "bf16")
        self.assertAlmostEqual(float(audit["bytecross_T"]), 7633.5, places=1)
        self.assertGreater(audit["gamma_low_entropy"], audit["gamma_high_entropy"])
        self.assertTrue(audit["gdn_chain_match"])
        self.assertLess(float(audit["optr_softmax_err"]), 1e-10)
        self.assertGreater(audit["aatc_hot_channel_bits"], audit["aatc_cold_channel_bits"])
        self.assertGreaterEqual(audit["readkv_hot_key_bits"], audit["readkv_cold_key_bits"])
        self.assertEqual(audit["working_memory_tokens"], 7)
        self.assertTrue(audit["gumbel_stable"])
        self.assertTrue(math.isfinite(float(audit["bytecross_speedup_at_T"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
