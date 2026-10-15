"""
Omegahee / Omega-Uzu v2.0: Unified Mathematical & Algorithmic Engine
====================================================================
Integrates 2026 Research Methodologies for High-TPS & Large-Weight LLM Inference,
Zero-Loss QAT Recovery, Weight-Shared Adaptive AR/Diffusion Recurrence, and
Dream-RSI + LoGRA Parametric Self-Improvement:

1. LittleBit-2 (arXiv:2603.00042v2, ICML 2026):
   - Internal Latent Rotation via Joint Iterative Quantization (Joint-ITQ)
   - SmoothSign Differentiable Binarization + Residual Factorization + L2L QAT
2. HybridAgent (arXiv:2610.18861v1, Oct 2026):
   - HybridLink Residual Adapter R(h_t) = h_t + W_2 * GELU(W_1 * h_t)
   - Continuous Working Memory M^(t) across text & latent reasoning modes
   - Two-Pass Self-Conditioned HybridSFT & HybridRL Exact Rollout Replay
3. ALoDLM Weight-Shared Adaptive AR-Diffusion Architecture (arXiv:2610.04198v1, Oct 2026):
   - Prelude [0, L_pre) -> Weight-Shared Recurrent Core [L_pre, L_coda) (K loops) -> Coda [L_coda, L)
   - Dual Readout Heads: LMHead (entropy commitment <= tau) + ExitGate (halting >= q)
   - Depth-Aware KV Cache {K_l^(d), V_l^(d)} with deepest-available fallback min(d, D_{l,i})
   - Variational NELBO & Unbiased Score-Function Surrogate Loss with 1st-Pass Control Variate
4. Dream-RSI (dream-rsi.com / arXiv:2609.14858) + LoGRA (arXiv:2610.06647, Oct 2026):
   - History-as-Replay-Simulator Pool H_t = (T_1, ..., T_t) with zero-execution dreaming
   - Monotonic Policy Improvement Guarantee (pi_0 = pi_t included in candidate set)
   - Parametric RSI Weight Consolidation via LoGRA Low-Rank Gradient Sketches (S = G * A^T)
   - Predicted-KL Step Control: q(D) = (1 / (2|P|T^2)) * sum Var_{v ~ p(s)}[dot_z_v(s)],
     alpha = min(alpha_max, sqrt(delta / q(D)))
5. Omega-Uzu Kernel & KV Acceleration (2026.07-2026.10 arXiv):
   - KSC-CurveTQ (2610.09212 + 2608.15567 + 2607.07964) + StagQ (2610.05977) + AWSRC (2608.23144)
   - Disaggregated Quantization (2609.26333) + ByteCross Crossover (2609.33889)
   - LiLiCorr Single-Pass Bilinear Draft Correlation (2608.20530) + ReadKV (2610.11245)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import os
import sys
import numpy as np


def softmax(logits: np.ndarray, axis: int = -1, temp: float = 1.0) -> np.ndarray:
    scaled = logits / max(temp, 1e-8)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1, temp: float = 1.0) -> np.ndarray:
    scaled = logits / max(temp, 1e-8)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def rms_norm(x: np.ndarray, weight: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    normed = x / rms
    return normed * weight if weight is not None else normed


def kl_divergence_logits(logits_p: np.ndarray, logits_q: np.ndarray, temp: float = 1.0) -> float:
    p = softmax(logits_p, axis=-1, temp=temp)
    log_p = log_softmax(logits_p, axis=-1, temp=temp)
    log_q = log_softmax(logits_q, axis=-1, temp=temp)
    kl = np.sum(p * (log_p - log_q), axis=-1)
    return float(np.mean(kl))


@dataclass
class LittleBit2Result:
    u_rot: np.ndarray
    v_rot: np.ndarray
    rotation: np.ndarray
    row_scale: np.ndarray
    col_scale: np.ndarray
    latent_scale: np.ndarray
    res_u_rot: np.ndarray
    res_v_rot: np.ndarray
    res_scale: float
    awsrc_seeds: np.ndarray
    awsrc_scales: np.ndarray
    itq_loss_history: List[float]
    qat_loss_history: List[float]


class LittleBit2QATEngine:
    """
    Implements LittleBit-2 (arXiv:2603.00042v2) Internal Latent Rotation via Joint-ITQ,
    SmoothSign Quantization-Aware Training (QAT), Layer-to-Layer (L2L) output alignment,
    and AWSRC (arXiv:2608.23144) seeded residual repair for zero-loss recovery.
    """

    def __init__(self, rank: int = 16, itq_iters: int = 25, beta_smooth: float = 3.0, seed: int = 42):
        self.rank = rank
        self.itq_iters = itq_iters
        self.beta_smooth = beta_smooth
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def smooth_sign(x: np.ndarray, beta: float = 3.0, hard_forward: bool = True) -> np.ndarray:
        soft = np.tanh(beta * x)
        if not hard_forward:
            return soft
        hard = np.where(x >= 0.0, 1.0, -1.0)
        return soft + (hard - soft)

    def joint_itq(self, w: np.ndarray, rank: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        r = min(rank or self.rank, min(w.shape))
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        sqrt_s = np.sqrt(s[:r])
        l1 = u[:, :r] * sqrt_s[None, :]
        l2 = vt[:r, :].T * sqrt_s[None, :]

        l1_norm = l1 / (np.linalg.norm(l1, axis=0, keepdims=True) / math.sqrt(w.shape[0]) + 1e-8)
        l2_norm = l2 / (np.linalg.norm(l2, axis=0, keepdims=True) / math.sqrt(w.shape[1]) + 1e-8)
        stacked = np.vstack([l1_norm, l2_norm])

        # Start from identity so loss[0] is the unrotated SVD baseline distortion
        rot = np.eye(r, dtype=np.float64)
        loss_history: List[float] = []

        for _ in range(self.itq_iters):
            z = stacked @ rot
            b = np.where(z >= 0.0, 1.0, -1.0)
            loss = float(np.mean((b - z) ** 2))
            loss_history.append(loss)
            u_p, _, vt_p = np.linalg.svd(b.T @ stacked, full_matrices=False)
            rot = (u_p @ vt_p).T

        z_final = stacked @ rot
        loss_history.append(float(np.mean((np.where(z_final >= 0.0, 1.0, -1.0) - z_final) ** 2)))
        u_rot = l1 @ rot
        v_rot = l2 @ rot
        return u_rot, v_rot, rot, loss_history

    @staticmethod
    def _seeded_basis(seed_id: int, shape: Tuple[int, int]) -> np.ndarray:
        m, n = shape
        idx = np.arange(m * n, dtype=np.uint64)
        h = (idx ^ np.uint64(seed_id * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF))
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> np.uint64(31))
        signs = np.where((h & np.uint64(1)) == 0, 1.0, -1.0).reshape(m, n)
        return signs / math.sqrt(m * n)

    def run_qat_with_l2l_recovery(
        self,
        w: np.ndarray,
        x_calib: np.ndarray,
        steps: int = 40,
        lr: float = 0.05,
        l2l_loss_scale: float = 10.0,
        num_seeds: int = 8,
    ) -> Tuple[np.ndarray, LittleBit2Result]:
        m, n = w.shape
        r = min(self.rank, m, n)
        u_rot, v_rot, rot, itq_losses = self.joint_itq(w, rank=r)

        row_scale = np.mean(np.abs(u_rot), axis=1, keepdims=True) + 1e-6
        col_scale = np.mean(np.abs(v_rot), axis=1, keepdims=True).T + 1e-6
        latent_scale = np.ones(r, dtype=np.float64)

        u_lat = u_rot / row_scale
        v_lat = v_rot / col_scale.T

        b_u1 = np.where(u_lat >= 0.0, 1.0, -1.0)
        b_v1 = np.where(v_lat >= 0.0, 1.0, -1.0)
        w_stage1 = row_scale * ((b_u1 * latent_scale[None, :]) @ b_v1.T) * col_scale
        res1 = w - w_stage1
        res_u, res_v, _, _ = self.joint_itq(res1, rank=r)
        res_scale = float(np.mean(np.abs(res1))) + 1e-6

        y_teacher = x_calib @ w.T
        a_cov = (x_calib.T @ x_calib) / max(x_calib.shape[0], 1) + 1e-4 * np.eye(n)

        # Initial hard 1-bit quantization loss before QAT + StagQ + AWSRC recovery
        y_init_hard = x_calib @ w_stage1.T
        init_hard_loss = float(
            np.mean((w_stage1 - w) ** 2) + l2l_loss_scale * np.mean((y_init_hard - y_teacher) ** 2)
        )
        qat_losses: List[float] = [init_hard_loss]
        for step in range(steps):
            beta = self.beta_smooth * (1.0 + 0.05 * step)
            su = np.tanh(beta * u_lat)
            sv = np.tanh(beta * v_lat)
            s_res_u = np.tanh(beta * res_u)
            s_res_v = np.tanh(beta * res_v)

            core = (su * latent_scale[None, :]) @ sv.T
            res_core = s_res_u @ s_res_v.T
            w_hat_soft = row_scale * core * col_scale + res_scale * res_core

            y_pred = x_calib @ w_hat_soft.T
            diff_y = (y_pred - y_teacher) / max(x_calib.shape[0], 1)
            diff_w = w_hat_soft - w
            loss_l2l = float(np.mean((y_pred - y_teacher) ** 2))
            loss_w = float(np.mean(diff_w ** 2))
            total_loss = loss_w + l2l_loss_scale * loss_l2l
            qat_losses.append(total_loss)

            grad_w = (2.0 / (m * n)) * diff_w + l2l_loss_scale * (2.0 / (m * n)) * (diff_y.T @ x_calib)
            grad_row = np.sum(grad_w * (core * col_scale), axis=1, keepdims=True)
            grad_col = np.sum(grad_w * (row_scale * core), axis=0, keepdims=True)
            scaled_grad = grad_w * (row_scale * col_scale)
            grad_latent = np.sum((su.T @ scaled_grad) * sv.T, axis=1)

            d_su = (scaled_grad @ sv) * latent_scale[None, :] * beta * (1.0 - su ** 2)
            d_sv = (scaled_grad.T @ su) * latent_scale[None, :] * beta * (1.0 - sv ** 2)

            row_scale -= lr * grad_row
            col_scale -= lr * grad_col
            latent_scale -= lr * grad_latent
            u_lat -= lr * d_su
            v_lat -= lr * d_sv

        bu_final = np.where(u_lat >= 0.0, 1.0, -1.0)
        bv_final = np.where(v_lat >= 0.0, 1.0, -1.0)
        bres_u = np.where(res_u >= 0.0, 1.0, -1.0)
        bres_v = np.where(res_v >= 0.0, 1.0, -1.0)

        base_1 = (bu_final * latent_scale[None, :]) @ bv_final.T
        base_2 = bres_u @ bres_v.T
        w_raw = row_scale * base_1 * col_scale + res_scale * base_2
        # SchurOpt (arXiv:2608.15567) closed-form row-wise affine scale refitting under A_cov
        num_scale = np.sum((w @ a_cov) * w_raw, axis=1, keepdims=True)
        den_scale = np.sum((w_raw @ a_cov) * w_raw, axis=1, keepdims=True) + 1e-8
        w_base = (num_scale / den_scale) * w_raw

        residual = w - w_base
        awsrc_seeds = np.zeros(num_seeds, dtype=np.int64)
        awsrc_scales = np.zeros(num_seeds, dtype=np.float64)
        rem = residual.copy()
        for k_idx in range(num_seeds):
            best_seed = 0
            best_proj = 0.0
            best_basis = None
            for cand_seed in range(k_idx * 16 + 1, k_idx * 16 + 17):
                basis = self._seeded_basis(cand_seed, (m, n))
                proj = float(np.sum(rem * (basis @ a_cov)) / (np.sum(basis * (basis @ a_cov)) + 1e-8))
                if abs(proj) >= abs(best_proj):
                    best_proj = proj
                    best_seed = cand_seed
                    best_basis = basis
            awsrc_seeds[k_idx] = best_seed
            awsrc_scales[k_idx] = best_proj
            if best_basis is not None:
                rem = rem - best_proj * best_basis

        # StagQ dyadic refinement planes + sparse outlier side record (arXiv:2610.05977)
        step_size = np.max(np.abs(rem), axis=1, keepdims=True) / 31.0 + 1e-8
        q_plane = np.clip(np.round(rem / step_size), -31.0, 31.0) * step_size
        rem_after = rem - q_plane
        # Hold out top 2% worst-served weights in StagQ sparse side record
        thresh_out = np.percentile(np.abs(rem_after), 98.0)
        sparse_side = np.where(np.abs(rem_after) >= thresh_out, rem_after, 0.0)
        w_reconstructed = w - rem + q_plane + sparse_side

        y_final = x_calib @ w_reconstructed.T
        final_loss = float(
            np.mean((w_reconstructed - w) ** 2) + l2l_loss_scale * np.mean((y_final - y_teacher) ** 2)
        )
        qat_losses.append(final_loss)

        result = LittleBit2Result(
            u_rot=u_lat,
            v_rot=v_lat,
            rotation=rot,
            row_scale=row_scale,
            col_scale=col_scale,
            latent_scale=latent_scale,
            res_u_rot=res_u,
            res_v_rot=res_v,
            res_scale=res_scale,
            awsrc_seeds=awsrc_seeds,
            awsrc_scales=awsrc_scales,
            itq_loss_history=itq_losses,
            qat_loss_history=qat_losses,
        )
        return w_reconstructed, result


@dataclass
class DepthAwareKVCache:
    num_layers: int
    start_rec: int
    end_rec: int
    max_depth: int
    prelude_coda_kv: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    core_depth_kv: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    deepest_available: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def total_logical_buffer_pairs(self) -> int:
        m = self.end_rec - self.start_rec
        return self.num_layers + (self.max_depth - 1) * m

    def write(self, layer: int, depth: int, k_mat: np.ndarray, v_mat: np.ndarray) -> None:
        seq_len = k_mat.shape[0]
        if self.start_rec <= layer < self.end_rec:
            self.core_depth_kv[(layer, depth)] = (k_mat.copy(), v_mat.copy())
            for pos in range(seq_len):
                prev_d = self.deepest_available.get((layer, pos), 0)
                if depth > prev_d:
                    self.deepest_available[(layer, pos)] = depth
        else:
            self.prelude_coda_kv[layer] = (k_mat.copy(), v_mat.copy())

    def read_with_fallback(
        self,
        layer: int,
        depth: int,
        seq_len: int,
        query_vec: Optional[np.ndarray] = None,
        read_bits: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not (self.start_rec <= layer < self.end_rec):
            k_out, v_out = self.prelude_coda_kv[layer]
            k_res, v_res = k_out[:seq_len].copy(), v_out[:seq_len].copy()
        else:
            sample_k, _ = next(iter(self.core_depth_kv.values()))
            dim = sample_k.shape[-1]
            k_res = np.zeros((seq_len, dim), dtype=np.float64)
            v_res = np.zeros((seq_len, dim), dtype=np.float64)
            for pos in range(seq_len):
                avail_d = self.deepest_available.get((layer, pos), 1)
                use_d = min(depth, max(avail_d, 1))
                while (layer, use_d) not in self.core_depth_kv and use_d > 1:
                    use_d -= 1
                k_buf, v_buf = self.core_depth_kv[(layer, use_d)]
                k_res[pos] = k_buf[pos]
                v_res[pos] = v_buf[pos]

        if query_vec is not None and read_bits < 8:
            q_energy = np.abs(query_vec).mean(axis=0) if query_vec.ndim > 1 else np.abs(query_vec)
            median_e = float(np.median(q_energy))
            levels = np.where(q_energy >= median_e, 63.0, 3.0)[None, :]
            k_max = np.max(np.abs(k_res), axis=0, keepdims=True) + 1e-8
            k_res = np.round((k_res / k_max) * levels) / levels * k_max

        return k_res, v_res


class HybridLinkAdapter:
    def __init__(self, d_model: int, latent_entry_id: int, latent_exit_id: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.latent_entry_id = latent_entry_id
        self.latent_exit_id = latent_exit_id
        self.w1 = rng.standard_normal((d_model, d_model)) * (0.02 / math.sqrt(d_model))
        # Paper: W_2 is zero-initialized so R(h) = h at step 0.
        self.w2 = np.zeros((d_model, d_model), dtype=np.float64)

    def forward(self, h_t: np.ndarray) -> np.ndarray:
        return h_t + gelu(h_t @ self.w1.T) @ self.w2.T

    def should_exit_latent_mode(self, r_h: np.ndarray, embeddings: np.ndarray) -> Tuple[bool, int]:
        r_norm = r_h / (np.linalg.norm(r_h) + 1e-8)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        cos_sims = emb_norms @ r_norm
        nearest_token = int(np.argmax(cos_sims))
        return (nearest_token == self.latent_exit_id), nearest_token

    def compute_hybrid_link_loss(
        self,
        h_lat: np.ndarray,
        target_emb: np.ndarray,
        lambda_cos: float = 1.0,
        lambda_mse: float = 0.2,
    ) -> float:
        pred = self.forward(h_lat)
        pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        tgt_norm = target_emb / (np.linalg.norm(target_emb, axis=-1, keepdims=True) + 1e-8)
        cos_sim = np.sum(pred_norm * tgt_norm, axis=-1)
        loss_cos = float(np.mean(1.0 - cos_sim))
        loss_mse = float(np.mean(np.sum((pred - target_emb) ** 2, axis=-1)))
        return lambda_cos * loss_cos + lambda_mse * loss_mse


class SharedTransformerLayer:
    def __init__(self, d_model: int, rng: np.random.Generator):
        scale = 1.0 / math.sqrt(d_model)
        self.wq = rng.standard_normal((d_model, d_model)) * scale
        self.wk = rng.standard_normal((d_model, d_model)) * scale
        self.wv = rng.standard_normal((d_model, d_model)) * scale
        self.wo = rng.standard_normal((d_model, d_model)) * scale
        self.w_up = rng.standard_normal((d_model, d_model)) * scale
        self.w_down = rng.standard_normal((d_model, d_model)) * scale

    def forward(
        self,
        h: np.ndarray,
        layer_idx: int,
        depth: int,
        kv_cache: Optional[DepthAwareKVCache] = None,
        causal: bool = True,
        rho_act: float = 1.0,
    ) -> np.ndarray:
        seq_len, d_model = h.shape
        h_norm = rms_norm(h)
        q = h_norm @ self.wq.T
        k = h_norm @ self.wk.T
        v = h_norm @ self.wv.T

        if kv_cache is not None:
            kv_cache.write(layer_idx, depth, k, v)
            k_read, v_read = kv_cache.read_with_fallback(layer_idx, depth, seq_len, query_vec=q)
        else:
            k_read, v_read = k, v

        scores = (q @ k_read.T) / math.sqrt(d_model)
        if causal and seq_len > 1:
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores = scores + mask
        attn = softmax(scores, axis=-1)
        h = h + (attn @ v_read) @ self.wo.T

        h_ffn_in = rms_norm(h)
        act = gelu(h_ffn_in @ self.w_up.T)
        if rho_act < 1.0:
            k_keep = max(1, int(round(rho_act * d_model)))
            thresh = np.partition(np.abs(act), d_model - k_keep, axis=-1)[:, d_model - k_keep : d_model - k_keep + 1]
            act = np.where(np.abs(act) >= thresh, act, 0.0)
        h = h + act @ self.w_down.T
        return h


class ALoDLMHybridARModel:
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 32,
        n_prelude: int = 2,
        n_core_shared: int = 3,
        n_coda: int = 2,
        max_recurrent_depth: int = 4,
        seed: int = 123,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_prelude = n_prelude
        self.n_core_shared = n_core_shared
        self.n_coda = n_coda
        self.total_physical_layers = n_prelude + n_core_shared + n_coda
        self.max_depth = max_recurrent_depth
        self.mask_token_id = vocab_size - 1
        self.latent_entry_id = vocab_size - 3
        self.latent_exit_id = vocab_size - 2

        rng = np.random.default_rng(seed)
        self.embeddings = rng.standard_normal((vocab_size, d_model)) * (1.0 / math.sqrt(d_model))
        self.prelude_layers = [SharedTransformerLayer(d_model, rng) for _ in range(n_prelude)]
        self.core_layers = [SharedTransformerLayer(d_model, rng) for _ in range(n_core_shared)]
        self.coda_layers = [SharedTransformerLayer(d_model, rng) for _ in range(n_coda)]

        self.lm_head = rng.standard_normal((vocab_size, d_model)) * (1.0 / math.sqrt(d_model))
        self.exit_gate_w = rng.standard_normal((1, d_model)) * (1.0 / math.sqrt(d_model))
        self.exit_gate_b = np.zeros((1,), dtype=np.float64)
        self.hybrid_link = HybridLinkAdapter(d_model, self.latent_entry_id, self.latent_exit_id, seed=seed + 1)

    def effective_layer_evaluations(self, executed_passes: int) -> int:
        return self.n_prelude + executed_passes * self.n_core_shared + executed_passes * self.n_coda

    def run_prelude(self, x_ids: np.ndarray, kv_cache: Optional[DepthAwareKVCache] = None) -> np.ndarray:
        h = self.embeddings[x_ids].copy()
        for idx, layer in enumerate(self.prelude_layers):
            h = layer.forward(h, layer_idx=idx, depth=1, kv_cache=kv_cache)
        return h

    def run_core_pass(
        self,
        h_in: np.ndarray,
        depth: int,
        kv_cache: Optional[DepthAwareKVCache] = None,
        rho_act: float = 1.0,
    ) -> np.ndarray:
        h = h_in.copy()
        for offset, layer in enumerate(self.core_layers):
            layer_idx = self.n_prelude + offset
            h = layer.forward(h, layer_idx=layer_idx, depth=depth, kv_cache=kv_cache, rho_act=rho_act)
        return h

    def run_coda_readout(
        self,
        h_core: np.ndarray,
        depth: int,
        kv_cache: Optional[DepthAwareKVCache] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_state = h_core.copy()
        for offset, layer in enumerate(self.coda_layers):
            layer_idx = self.n_prelude + self.n_core_shared + offset
            r_state = layer.forward(r_state, layer_idx=layer_idx, depth=depth, kv_cache=kv_cache)
        logits = r_state @ self.lm_head.T
        probs = softmax(logits, axis=-1)
        if depth < self.max_depth:
            gate_logits = (r_state @ self.exit_gate_w.T).squeeze(-1) + self.exit_gate_b[0]
            lambdas = sigmoid(gate_logits)
        else:
            lambdas = np.ones(r_state.shape[0], dtype=np.float64)
        return r_state, probs, lambdas

    def adaptive_block_ar_decode(
        self,
        seq_ids: np.ndarray,
        entropy_tau: float = 3.2,
        halt_threshold_q: float = 0.75,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        x = seq_ids.copy()
        kv_cache = DepthAwareKVCache(
            num_layers=self.total_physical_layers,
            start_rec=self.n_prelude,
            end_rec=self.n_prelude + self.n_core_shared,
            max_depth=self.max_depth,
        )
        total_inner_passes = 0
        commit_depths: Dict[int, int] = {}

        while np.any(x == self.mask_token_id):
            unresolved = set(np.where(x == self.mask_token_id)[0].tolist())
            h = self.run_prelude(x, kv_cache=kv_cache)
            a_cum = np.zeros(len(x), dtype=np.float64)
            committed_in_step: Dict[int, int] = {}

            for s in range(1, self.max_depth + 1):
                total_inner_passes += 1
                h_tilde = self.run_core_pass(h, depth=s, kv_cache=kv_cache)
                _, probs, lambdas = self.run_coda_readout(h_tilde, depth=s, kv_cache=kv_cache)
                entropies = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)

                newly_committed = []
                for idx in list(unresolved):
                    a_cum[idx] = a_cum[idx] + (1.0 - a_cum[idx]) * lambdas[idx]
                    if entropies[idx] <= entropy_tau:
                        pred_tok = int(np.argmax(probs[idx, :-1]))
                        committed_in_step[idx] = pred_tok
                        commit_depths[idx] = s
                        newly_committed.append(idx)

                for idx in newly_committed:
                    unresolved.remove(idx)

                h = h_tilde.copy()
                for idx, tok_id in committed_in_step.items():
                    h[idx] = self.embeddings[tok_id]

                if len(unresolved) == 0:
                    break
                mean_a = float(np.mean([a_cum[i] for i in unresolved]))
                if mean_a >= halt_threshold_q:
                    break

            if len(committed_in_step) == 0 and len(unresolved) > 0:
                _, probs, _ = self.run_coda_readout(h, depth=self.max_depth, kv_cache=kv_cache)
                entropies = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
                best_idx = min(unresolved, key=lambda idx: float(entropies[idx]))
                committed_in_step[best_idx] = int(np.argmax(probs[best_idx, :-1]))
                commit_depths[best_idx] = self.max_depth

            for idx, tok_id in committed_in_step.items():
                x[idx] = tok_id

        stats = {
            "total_inner_passes": float(total_inner_passes),
            "mean_commit_depth": float(np.mean(list(commit_depths.values()))) if commit_depths else 1.0,
            "logical_kv_pairs": float(kv_cache.total_logical_buffer_pairs()),
        }
        return x, stats

    @staticmethod
    def truncated_geometric_prior(max_depth: int = 4, c: float = 0.4) -> np.ndarray:
        depths = np.arange(1, max_depth + 1, dtype=np.float64)
        unnorm = np.exp(-c * depths)
        return unnorm / np.sum(unnorm)

    def compute_alodlm_surrogate_loss(
        self,
        x_corrupted: np.ndarray,
        y_clean: np.ndarray,
        mask_prob_t: float = 0.5,
        alpha_reg: float = 0.1,
        beta_reg: float = 1.0,
    ) -> Dict[str, float]:
        masked_positions = np.where(x_corrupted == self.mask_token_id)[0]
        if len(masked_positions) == 0:
            masked_positions = np.array([len(x_corrupted) - 1])

        h = self.run_prelude(x_corrupted)
        all_ell = np.zeros((self.max_depth, len(x_corrupted)), dtype=np.float64)
        all_lambdas = np.zeros((self.max_depth, len(x_corrupted)), dtype=np.float64)

        for s in range(1, self.max_depth + 1):
            h = self.run_core_pass(h, depth=s)
            _, probs, lambdas = self.run_coda_readout(h, depth=s)
            all_lambdas[s - 1] = lambdas
            for pos in range(len(x_corrupted)):
                all_ell[s - 1, pos] = -math.log(max(float(probs[pos, y_clean[pos]]), 1e-12))

        prior = self.truncated_geometric_prior(self.max_depth, c=0.4)
        r_hat = np.zeros((len(masked_positions), self.max_depth), dtype=np.float64)
        z_sampled = np.zeros(len(masked_positions), dtype=np.int64)

        for idx_m, pos in enumerate(masked_positions):
            surv = 1.0
            for d in range(1, self.max_depth + 1):
                lam = all_lambdas[d - 1, pos] if d < self.max_depth else 1.0
                r_hat[idx_m, d - 1] = surv * lam if d < self.max_depth else surv
                surv *= (1.0 - lam)
            r_hat[idx_m] = np.clip(r_hat[idx_m], 1e-9, 1.0)
            r_hat[idx_m] /= np.sum(r_hat[idx_m])
            z_sampled[idx_m] = int(np.argmax(r_hat[idx_m])) + 1

        b_z = np.mean(r_hat, axis=0)
        l_den = 0.0
        r_seq_cost = 0.0
        b_seq_baseline = 0.0
        log_q_sum = 0.0

        for idx_m, pos in enumerate(masked_positions):
            z_i = z_sampled[idx_m]
            omega = np.zeros(z_i, dtype=np.float64)
            v_surv = 1.0
            for s in range(1, z_i + 1):
                lam = all_lambdas[s - 1, pos]
                if s < z_i:
                    omega[s - 1] = v_surv * lam
                    v_surv *= (1.0 - lam)
                else:
                    omega[s - 1] = v_surv
            l_den += float(np.sum(omega * all_ell[:z_i, pos])) / max(mask_prob_t, 1e-4)

            u_i = math.log(max(float(r_hat[idx_m, z_i - 1]), 1e-12))
            c_reg = (
                alpha_reg * (u_i - math.log(max(float(b_z[z_i - 1]), 1e-12)))
                + beta_reg * (math.log(max(float(b_z[z_i - 1]), 1e-12)) - math.log(float(prior[z_i - 1])))
            )
            r_seq_cost += float(all_ell[z_i - 1, pos]) + c_reg
            b_seq_baseline += float(all_ell[0, pos])
            log_q_sum += u_i

        m_count = float(len(masked_positions))
        l_gate = ((r_seq_cost - b_seq_baseline) * log_q_sum) / m_count
        l_ar = float(np.mean(all_ell[0]))
        return {
            "L_den": l_den / m_count,
            "L_gate": l_gate,
            "L_AR": l_ar,
            "L_impl": (l_den / m_count) + l_gate + l_ar,
            "first_pass_baseline": b_seq_baseline / m_count,
        }


@dataclass
class DiscoveryNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    feature_vec: np.ndarray
    target_token: int
    realized_reward: float


@dataclass
class DiscoveryTreeWorld:
    world_id: int
    nodes: Dict[int, DiscoveryNode]


@dataclass
class ExplorationPolicy:
    policy_id: str
    weights: np.ndarray
    branch_width: int = 3
    stop_threshold: float = -0.2


class DreamRSILoGRAEngine:
    def __init__(self, d_model: int, vocab_size: int, sketch_rank: int = 8, seed: int = 2026):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.sketch_rank = min(sketch_rank, d_model, vocab_size)
        self.rng = np.random.default_rng(seed)
        self.env_oracle_weights = self.rng.standard_normal(d_model)
        self.env_oracle_weights /= (np.linalg.norm(self.env_oracle_weights) + 1e-8)
        self.history_worlds: List[DiscoveryTreeWorld] = []

    def online_explore_and_record_world(
        self,
        policy: ExplorationPolicy,
        num_nodes: int = 18,
    ) -> DiscoveryTreeWorld:
        world_id = len(self.history_worlds) + 1
        nodes: Dict[int, DiscoveryNode] = {}
        root_feat = self.rng.standard_normal(self.d_model)
        root_feat /= (np.linalg.norm(root_feat) + 1e-8)
        root_score = float(root_feat @ policy.weights)
        nodes[0] = DiscoveryNode(
            node_id=0,
            parent_id=None,
            depth=0,
            feature_vec=root_feat,
            target_token=int(abs(hash((world_id, 0))) % (self.vocab_size - 3)),
            realized_reward=0.5 + 0.3 * math.tanh(root_score),
        )

        for nid in range(1, num_nodes):
            parent_id = (nid - 1) // max(policy.branch_width, 1)
            parent = nodes[parent_id]
            feat = 0.25 * parent.feature_vec + 0.75 * self.rng.standard_normal(self.d_model)
            feat /= (np.linalg.norm(feat) + 1e-8)
            env_alignment = float(feat @ self.env_oracle_weights)
            reward = float(env_alignment)
            tok = int(np.argmax(np.abs(feat[: min(len(feat), self.vocab_size - 3)])))
            nodes[nid] = DiscoveryNode(
                node_id=nid,
                parent_id=parent_id,
                depth=parent.depth + 1,
                feature_vec=feat,
                target_token=tok,
                realized_reward=reward,
            )

        world = DiscoveryTreeWorld(world_id=world_id, nodes=nodes)
        self.history_worlds.append(world)
        return world

    def dream_replay_score(
        self,
        candidate_policy: ExplorationPolicy,
        budget_nodes: int = 6,
    ) -> Tuple[float, List[DiscoveryNode]]:
        if not self.history_worlds:
            return 0.0, []

        world_scores: List[float] = []
        elite_nodes: List[DiscoveryNode] = []

        for world in self.history_worlds:
            ranked = sorted(
                world.nodes.values(),
                key=lambda n: float(n.feature_vec @ candidate_policy.weights) - 0.03 * n.depth,
                reverse=True,
            )
            selected = ranked[:budget_nodes]
            # Weighted average of top-ranked realized outcomes + peak discovery reward
            mean_outcome = float(np.mean([n.realized_reward for n in selected]))
            best_outcome = max(n.realized_reward for n in selected)
            world_scores.append(0.6 * mean_outcome + 0.4 * best_outcome)
            best_node = max(selected, key=lambda n: n.realized_reward)
            elite_nodes.append(best_node)

        return float(np.mean(world_scores)), elite_nodes

    def improve_policy_by_dreaming(
        self,
        current_policy: ExplorationPolicy,
        num_revisions: int = 32,
        mutation_scale: float = 0.25,
    ) -> Tuple[ExplorationPolicy, float, float, List[DiscoveryNode]]:
        base_score, best_elites = self.dream_replay_score(current_policy)
        best_policy = current_policy
        best_score = base_score

        for m_idx in range(1, num_revisions):
            cand_w = current_policy.weights + mutation_scale * self.rng.standard_normal(self.d_model)
            cand_w /= (np.linalg.norm(cand_w) + 1e-8)
            cand_policy = ExplorationPolicy(
                policy_id=f"{current_policy.policy_id}_rev{m_idx}",
                weights=cand_w,
                branch_width=current_policy.branch_width,
                stop_threshold=current_policy.stop_threshold,
            )
            score, elites = self.dream_replay_score(cand_policy)
            if score >= best_score:
                best_score = score
                best_policy = cand_policy
                best_elites = elites

        return best_policy, base_score, best_score, best_elites

    def sample_rademacher_projection(self, k_cols: int, seed: int) -> np.ndarray:
        rng_proj = np.random.default_rng(seed)
        signs = np.where(rng_proj.integers(0, 2, size=(self.sketch_rank, k_cols)) == 0, 1.0, -1.0)
        return signs / math.sqrt(self.sketch_rank)

    @staticmethod
    def compute_predicted_kl_and_scale(
        w: np.ndarray,
        proposed_dw: np.ndarray,
        contexts: np.ndarray,
        kl_budget_delta: float = 2e-4,
        alpha_max: float = 1.0,
        temperature: float = 1.0,
    ) -> Tuple[float, float]:
        logits = contexts @ w.T
        probs = softmax(logits, axis=-1, temp=temperature)
        dot_z = contexts @ proposed_dw.T

        mean_dot_z = np.sum(probs * dot_z, axis=-1, keepdims=True)
        var_dot_z = np.sum(probs * ((dot_z - mean_dot_z) ** 2), axis=-1)
        q_d = float(np.mean(var_dot_z) / (2.0 * (temperature ** 2)))

        if q_d <= 1e-16:
            alpha = alpha_max
        else:
            alpha = min(alpha_max, math.sqrt(kl_budget_delta / q_d))
        return q_d, alpha

    def consolidate_rsi_into_weights_logra(
        self,
        w_lm_head: np.ndarray,
        elite_nodes: List[DiscoveryNode],
        step_size_eta: float = 0.15,
        kl_budget_delta: float = 5e-4,
        alpha_max: float = 1.0,
        seed: int = 101,
    ) -> Dict[str, object]:
        d_out, k_in = w_lm_head.shape
        # Construct orthonormalized Rademacher sketch basis so low-rank update preserves descent direction
        raw_proj = self.sample_rademacher_projection(k_in, seed=seed)
        q_orth, _ = np.linalg.qr(raw_proj.T)
        a_proj = q_orth[:, :self.sketch_rank].T  # (r, k_in) with A @ A^T = I_r
        s_sketch = np.zeros((d_out, self.sketch_rank), dtype=np.float64)

        contexts = np.stack([n.feature_vec for n in elite_nodes], axis=0)
        initial_loss = 0.0

        for node in elite_nodes:
            x_m = node.feature_vec
            logits = w_lm_head @ x_m
            probs = softmax(logits)
            initial_loss += -node.realized_reward * math.log(max(float(probs[node.target_token]), 1e-12))

            grad_logits = probs.copy()
            grad_logits[node.target_token] -= 1.0
            grad_logits *= (node.realized_reward / max(len(elite_nodes), 1))

            x_projected = a_proj @ x_m
            s_sketch += np.outer(grad_logits, x_projected)

        initial_loss /= max(len(elite_nodes), 1)
        u_sketch = s_sketch.copy()
        proposed_dw = -step_size_eta * (u_sketch @ a_proj)

        q_d, alpha = self.compute_predicted_kl_and_scale(
            w=w_lm_head,
            proposed_dw=proposed_dw,
            contexts=contexts,
            kl_budget_delta=kl_budget_delta,
            alpha_max=alpha_max,
        )

        w_updated = w_lm_head + alpha * proposed_dw
        predicted_kl = (alpha ** 2) * q_d
        realized_kl = kl_divergence_logits(contexts @ w_lm_head.T, contexts @ w_updated.T)

        post_loss = 0.0
        for node in elite_nodes:
            probs_after = softmax(w_updated @ node.feature_vec)
            post_loss += -node.realized_reward * math.log(max(float(probs_after[node.target_token]), 1e-12))
        post_loss /= max(len(elite_nodes), 1)

        return {
            "w_updated": w_updated,
            "sketch_S": s_sketch,
            "projection_seed": seed,
            "alpha": alpha,
            "q_D": q_d,
            "predicted_kl": predicted_kl,
            "realized_kl": realized_kl,
            "initial_rsi_loss": initial_loss,
            "post_rsi_loss": post_loss,
            "memory_ratio_sketch_vs_dense": float((d_out * self.sketch_rank) / (d_out * k_in)),
        }


def compute_bytecross_crossover_length(
    rho_act: float,
    rho_kv: float,
    b_proj_bytes: float,
    b_kv_per_token_bytes: float,
) -> float:
    return ((1.0 - rho_act) / max(1.0 - rho_kv, 1e-8)) * (b_proj_bytes / max(b_kv_per_token_bytes, 1e-8))


def lilicorr_bilinear_tree_score(
    u_out: np.ndarray,
    u_in: np.ndarray,
    marginal_log_probs: np.ndarray,
    tau_corr: float = 2.0,
) -> Tuple[np.ndarray, List[int]]:
    gamma, k_cand, _ = u_out.shape
    out_norm = u_out / (np.linalg.norm(u_out, axis=-1, keepdims=True) + 1e-8)
    in_norm = u_in / (np.linalg.norm(u_in, axis=-1, keepdims=True) + 1e-8)

    s_matrices = np.zeros((gamma - 1, k_cand, k_cand), dtype=np.float64)
    for t in range(gamma - 1):
        s_matrices[t] = tau_corr * (out_norm[t] @ in_norm[t + 1].T)

    chosen = [int(np.argmax(marginal_log_probs[0]))]
    for t in range(gamma - 1):
        prev_k = chosen[-1]
        trans_log_probs = log_softmax(s_matrices[t, prev_k])
        combined = marginal_log_probs[t + 1] + trans_log_probs
        chosen.append(int(np.argmax(combined)))

    return s_matrices, chosen





def run_full_ci_cl_benchmark(seed: int = 42) -> Dict[str, Any]:
    import json
    rng = np.random.default_rng(seed)

    # 1. LittleBit-2 Joint-ITQ + Zero-Loss L2L QAT
    m, n, r = 32, 32, 16
    u0, _ = np.linalg.qr(rng.standard_normal((m, m)))
    v0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    spectrum = np.exp(-np.linspace(0.0, 3.5, min(m, n)))
    w_teacher = (u0[:, :min(m, n)] * spectrum) @ v0[:, :min(m, n)].T
    x_calib = rng.standard_normal((64, n))
    lb_engine = LittleBit2QATEngine(rank=r, itq_iters=25, beta_smooth=3.0, seed=seed)
    u_rot, v_rot, rot, itq_losses = lb_engine.joint_itq(w_teacher, rank=r)
    w_recovered, qat_res = lb_engine.run_qat_with_l2l_recovery(
        w=w_teacher, x_calib=x_calib, steps=35, lr=0.04, l2l_loss_scale=10.0, num_seeds=8
    )
    y_teacher = x_calib @ w_teacher.T
    y_student = x_calib @ w_recovered.T
    top1_agreement = float(np.mean(np.argmax(y_teacher, axis=-1) == np.argmax(y_student, axis=-1)))

    # 2. ALoDLM Weight-Shared Hybrid AR + Depth-Aware KV Cache
    cache_8b = DepthAwareKVCache(num_layers=36, start_rec=10, end_rec=26, max_depth=4)
    alodlm = ALoDLMHybridARModel(
        vocab_size=48, d_model=24, n_prelude=2, n_core_shared=3, n_coda=2, max_recurrent_depth=4, seed=99
    )
    seq = np.array([3, 7, alodlm.mask_token_id, 12, alodlm.mask_token_id, alodlm.mask_token_id], dtype=np.int64)
    decoded, ar_stats = alodlm.adaptive_block_ar_decode(seq, entropy_tau=3.6, halt_threshold_q=0.7)
    prior_p = ALoDLMHybridARModel.truncated_geometric_prior(max_depth=4, c=0.4)

    # 3. HybridAgent HybridLink
    embeddings = rng.standard_normal((40, 24))
    link = HybridLinkAdapter(d_model=24, latent_entry_id=37, latent_exit_id=38, seed=77)
    h_lat = rng.standard_normal((8, 24))
    link_loss = link.compute_hybrid_link_loss(h_lat, embeddings[np.arange(8)])

    # 4. Dream-RSI + LoGRA Weight Consolidation (4 Continuous Learning Rounds)
    rsi_engine = DreamRSILoGRAEngine(d_model=32, vocab_size=48, sketch_rank=6, seed=2026)
    rng_rsi = np.random.default_rng(2026)
    init_w = rng_rsi.standard_normal(32)
    init_w /= np.linalg.norm(init_w)
    policy = ExplorationPolicy(policy_id="pi_init", weights=init_w, branch_width=3, stop_threshold=-0.1)
    w_lm_head = rng_rsi.standard_normal((48, 32)) * (1.0 / math.sqrt(32))
    kl_budget = 5e-4
    rsi_history = []
    for round_idx in range(1, 5):
        rsi_engine.online_explore_and_record_world(policy, num_nodes=20)
        new_policy, pre_score, post_score, elites = rsi_engine.improve_policy_by_dreaming(
            current_policy=policy, num_revisions=36, mutation_scale=0.3
        )
        policy = new_policy
        logra_res = rsi_engine.consolidate_rsi_into_weights_logra(
            w_lm_head=w_lm_head, elite_nodes=elites, step_size_eta=0.25,
            kl_budget_delta=kl_budget, alpha_max=1.0, seed=100 + round_idx
        )
        w_lm_head = logra_res["w_updated"]
        rsi_history.append({
            "round": round_idx,
            "pre_dream_score": round(float(pre_score), 4),
            "post_dream_score": round(float(post_score), 4),
            "pre_logra_ce_loss": round(float(logra_res["initial_rsi_loss"]), 4),
            "post_logra_ce_loss": round(float(logra_res["post_rsi_loss"]), 4),
            "predicted_kl": round(float(logra_res["predicted_kl"]), 6),
            "alpha_step": round(float(logra_res.get("alpha", logra_res.get("alpha_scale", 0.0))), 4),
            "memory_ratio": round(float(logra_res["memory_ratio_sketch_vs_dense"]), 4),
        })

    # 5. ByteCross & LiLiCorr
    t_star = compute_bytecross_crossover_length(
        rho_act=0.72, rho_kv=0.25, b_proj_bytes=2.68e9, b_kv_per_token_bytes=131072.0
    )

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from omegahee_pillars import run_pillar_reference_audit
    pillar_audit = run_pillar_reference_audit(seed=seed)

    return {
        "littlebit2": {
            "itq_loss_start": round(float(itq_losses[0]), 4),
            "itq_loss_end": round(float(itq_losses[-1]), 4),
            "qat_l2l_loss_start": round(float(qat_res.qat_loss_history[0]), 5),
            "qat_l2l_loss_end": round(float(qat_res.qat_loss_history[-1]), 5),
            "qat_loss_reduction_pct": round(100.0 * (1.0 - qat_res.qat_loss_history[-1] / qat_res.qat_loss_history[0]), 2),
            "top1_logit_agreement": round(top1_agreement, 4),
        },
        "alodlm": {
            "kv_logical_buffer_pairs_8b": cache_8b.total_logical_buffer_pairs(),
            "unique_layers_stored": 2 + 3 + 2,
            "effective_layers_evaluated_d4": alodlm.effective_layer_evaluations(4),
            "mean_commit_depth": round(float(ar_stats["mean_commit_depth"]), 3),
            "truncated_geom_prior_c04": [round(float(x), 4) for x in prior_p],
        },
        "hybrid_agent": {
            "initial_hybrid_link_loss": round(float(link_loss), 4),
        },
        "dream_rsi_logra": rsi_history,
        "bytecross_t_star_tokens": round(float(t_star), 1),
        "pillar_audit": {
            "effective_bpw": round(float(pillar_audit["effective_bpw"]), 6),
            "traffic_reduction": round(float(pillar_audit["traffic_reduction"]), 4),
            "qk_protected": bool(pillar_audit["qk_protected"]),
            "viterbi_beats_euclidean": bool(pillar_audit["viterbi_beats_euclidean"]),
            "rotation_applied": bool(pillar_audit["rotation_applied"]),
            "decode_activation": pillar_audit["decode_activation"],
            "gamma_low_entropy": int(pillar_audit["gamma_low_entropy"]),
            "gamma_high_entropy": int(pillar_audit["gamma_high_entropy"]),
            "gdn_chain_match": bool(pillar_audit["gdn_chain_match"]),
            "optr_softmax_err": float(pillar_audit["optr_softmax_err"]),
            "single_pass_speedup": round(float(pillar_audit["single_pass_speedup"]), 3),
            "gumbel_stable": bool(pillar_audit["gumbel_stable"]),
        },
    }


if __name__ == "__main__":
    import json
    metrics = run_full_ci_cl_benchmark()
    print(json.dumps(metrics, indent=2))
