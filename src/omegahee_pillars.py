"""
Omega-Uzu reference kernels for the four mathematical pillars that the v2.0
engine left as paper equations only.

Each routine follows the closed forms in
paper/uzu_omega_mathematical_algorithm_2026.tex:

  Pillar 1  CASA bit allocation, Schur complement, LDL curvature, KSC-CurveTQ
            Viterbi metric, StagQ dyadic planes, AWSRC seeded residual, 2.382 bpw
  Pillar 2  Disaggregated prefill/decode plan (DQ-ODP) and ByteCross speedup
  Pillar 3  LiLiCorr joint score, SparseSpec-L horizon, GDN Tree-Scan
  Pillar 4  OptR key reparameterization, AATC reverse water-filling, ReadKV
  Plus     Hybrid working memory, 24-bit Gumbel clamp, roofline projection
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pillar 1 — CASA / Schur / KSC-CurveTQ / StagQ / AWSRC
# ---------------------------------------------------------------------------

def casa_kronecker_energy(tr_A: float, tr_B: float, kappa_A: float, kappa_B: float) -> float:
    """Omega_l = sqrt( tr(A) tr(B) psi(kappa_A, kappa_B) )."""
    psi = ((kappa_A + 1.0) * (kappa_B + 1.0)) / (4.0 * math.sqrt(kappa_A * kappa_B))
    return math.sqrt(max(tr_A, 0.0) * max(tr_B, 0.0) * psi)


def casa_closed_form_bits(
    energies: Sequence[float],
    param_counts: Sequence[float],
    b_target: float,
) -> np.ndarray:
    """Parameter-weighted closed form. Mean_P(b*) = b_target exactly."""
    omega = np.asarray(energies, dtype=np.float64)
    params = np.asarray(param_counts, dtype=np.float64)
    total = float(np.sum(params))
    log_ratio = np.log(np.maximum(omega / params, 1e-30))
    log_geo = float(np.sum((params / total) * log_ratio))
    return b_target + (1.0 / (2.0 * math.log(2.0))) * (log_ratio - log_geo)


def casa_discrete_assign(
    continuous_bits: Sequence[float],
    module_kinds: Sequence[str],
) -> List[int]:
    """Snap to {2, 3, 4}. Attention W_Q / W_K stay at >= 3 bits."""
    assigned: List[int] = []
    for raw, kind in zip(continuous_bits, module_kinds):
        snapped = int(np.clip(int(np.round(float(raw))), 2, 4))
        if kind in ("W_Q", "W_K"):
            snapped = max(snapped, 3)
        assigned.append(snapped)
    return assigned


def regularized_schur(A: np.ndarray, block: int, lam: float = 1e-4) -> np.ndarray:
    """H_{G/S} = A_GG - A_GS (A_SS + lam I)^{-1} A_SG. Full block => A_GG."""
    n = A.shape[0]
    if not (0 < block <= n):
        raise ValueError("block must lie in 1..n")
    Atil = np.asarray(A, dtype=np.float64) + lam * np.eye(n)
    Agg = Atil[:block, :block]
    if block == n:
        return 0.5 * (Agg + Agg.T)
    Ags = Atil[:block, block:]
    Ass = Atil[block:, block:]
    solved = np.linalg.solve(Ass, Ags.T)
    H = Agg - Ags @ solved
    return 0.5 * (H + H.T)


def ldl_factor(H: np.ndarray, jitter: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-lower L and diagonal d with H ≈ L diag(d) L^T."""
    n = H.shape[0]
    L = np.eye(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    for j in range(n):
        acc = 0.0
        for k in range(j):
            acc += (L[j, k] ** 2) * d[k]
        d[j] = float(H[j, j] - acc)
        if d[j] < jitter:
            d[j] = jitter
        inv_d = 1.0 / d[j]
        for i in range(j + 1, n):
            acc = 0.0
            for k in range(j):
                acc += L[i, k] * L[j, k] * d[k]
            L[i, j] = float((H[i, j] - acc) * inv_d)
    return L, d


def curvature_path_cost(
    w: np.ndarray,
    hat: np.ndarray,
    d: np.ndarray,
    L: np.ndarray,
    g_row: np.ndarray,
    Bii: float,
    eta: float,
) -> float:
    """Sum_j B_ii * d_j * (w_tilde_j - hat_j)^2 with LDL feedback and G^2PTQ shift."""
    err = w - hat
    cost = 0.0
    for j in range(w.shape[0]):
        feedback = float(L[j, :j] @ err[:j]) if j else 0.0
        gterm = eta * (float(g_row[j]) / Bii) / d[j]
        tilde = float(w[j]) - gterm - feedback
        cost += Bii * d[j] * (tilde - float(hat[j])) ** 2
    return float(cost)


def ksc_curvetq_quantize_row(
    w: np.ndarray,
    d: np.ndarray,
    L: np.ndarray,
    g_row: np.ndarray,
    Bii: float = 1.0,
    eta: float = 0.25,
    n_bits: int = 2,
) -> Dict[str, object]:
    """
    Rotation-free curvature-weighted search on one native-basis block row.

    The reference kernel exhausts the 2^{n_bits * B} codebook (B <= 6) so the
    returned path is the exact minimizer of the KSC-CurveTQ branch metric.
    No Hadamard / orthogonal multiply is applied. The winning first-symbol
    index is the packed start state s0*.
    """
    w = np.asarray(w, dtype=np.float64)
    block = int(w.shape[0])
    if block > 6:
        raise ValueError("reference Viterbi is exact only for block <= 6")
    ncode = 1 << n_bits
    std = float(np.std(w)) + 1e-6
    # Deterministic 2-bit (or n-bit) mid-riser lattice in the native basis.
    span = np.linspace(-(ncode - 1), ncode - 1, ncode)
    codebook = 0.5 * std * span
    alpha = 1.0
    beta = np.ones(block, dtype=np.float64)

    best_cost = math.inf
    best_hat = np.zeros(block, dtype=np.float64)
    best_codes = np.zeros(block, dtype=np.int64)
    hat = np.zeros(block, dtype=np.float64)
    codes = np.zeros(block, dtype=np.int64)

    def rec(j: int) -> None:
        nonlocal best_cost, best_hat, best_codes
        if j == block:
            cost = curvature_path_cost(w, hat, d, L, g_row, Bii, eta)
            if cost < best_cost:
                best_cost = cost
                best_hat = hat.copy()
                best_codes = codes.copy()
            return
        for c in range(ncode):
            codes[j] = c
            hat[j] = alpha * beta[j] * codebook[c]
            rec(j + 1)

    rec(0)

    # Euclidean rounding baseline, scored with the same curvature metric.
    base_hat = np.zeros(block, dtype=np.float64)
    for j in range(block):
        base_hat[j] = alpha * beta[j] * codebook[int(np.argmin(np.abs(codebook - w[j])))]
    base_cost = curvature_path_cost(w, base_hat, d, L, g_row, Bii, eta)

    return {
        "hat": best_hat,
        "codes": best_codes,
        "s0_star": int(best_codes[0]),
        "header_bits": 14,
        "cost": best_cost,
        "euclidean_cost": base_cost,
        "rotation_applied": False,
        "codebook": codebook,
    }


def mirai_s_effective_bpw(
    base_bits: float = 2.0,
    header_bits: int = 14,
    block: int = 64,
    awsrc_bpw: float = 0.162,
) -> float:
    """2 + 14/64 + 0.162 = 2.38075 bpw. The paper rounds the 14-bit header to 0.22 bpw (2.382)."""
    return base_bits + header_bits / float(block) + awsrc_bpw


def weight_traffic_reduction(baseline_bpw: float = 4.613, omega_bpw: Optional[float] = None) -> float:
    bw = mirai_s_effective_bpw() if omega_bpw is None else omega_bpw
    return (baseline_bpw - bw) / baseline_bpw


def stagq_dyadic_encode(residual: np.ndarray, n_planes: int = 3) -> Dict[str, object]:
    """
    Successive dyadic planes. Plane r stores P in {-1,+1} and a per-row
    least-squares scale gamma_r = 2^{-r} Delta_r. Each plane is the optimal
    rank-1 sign fit of the current residual, so row MSE is monotone.
    """
    rem = np.asarray(residual, dtype=np.float64).copy()
    planes: List[np.ndarray] = []
    gammas: List[np.ndarray] = []
    mses = [float(np.mean(rem ** 2))]
    for _r in range(n_planes):
        P = np.where(rem >= 0.0, 1.0, -1.0)
        gamma = np.mean(np.abs(rem), axis=1, keepdims=True)
        planes.append(P)
        gammas.append(gamma)
        rem = rem - gamma * P
        mses.append(float(np.mean(rem ** 2)))
    return {"planes": planes, "gammas": gammas, "mse_prefix": mses, "final_residual": rem}


def stagq_dyadic_decode(base: np.ndarray, planes: Sequence[np.ndarray], gammas: Sequence[np.ndarray], n_planes: int) -> np.ndarray:
    hat = np.array(base, dtype=np.float64, copy=True)
    for r in range(n_planes):
        hat = hat + gammas[r] * planes[r]
    return hat


def _splitmix_basis(seed_id: int, shape: Tuple[int, int]) -> np.ndarray:
    m, n = shape
    idx = np.arange(m * n, dtype=np.uint64)
    h = idx ^ np.uint64(seed_id * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    h = h ^ (h >> np.uint64(31))
    signs = np.where((h & np.uint64(1)) == 0, 1.0, -1.0).reshape(m, n)
    return signs / math.sqrt(m * n)


def awsrc_seeded_repair(
    residual: np.ndarray,
    a_cov: np.ndarray,
    num_seeds: int = 4,
    candidates_per_seed: int = 8,
) -> Dict[str, object]:
    """Codebook-free residual repair. Only seeds and scales are stored."""
    rem = np.asarray(residual, dtype=np.float64).copy()
    start = float(np.sum(rem * (rem @ a_cov)))
    seeds = np.zeros(num_seeds, dtype=np.int64)
    scales = np.zeros(num_seeds, dtype=np.float64)
    m, n = rem.shape
    for k_idx in range(num_seeds):
        best_seed = 0
        best_proj = 0.0
        best_basis = None
        for cand in range(k_idx * candidates_per_seed + 1, (k_idx + 1) * candidates_per_seed + 1):
            basis = _splitmix_basis(cand, (m, n))
            proj = float(np.sum(rem * (basis @ a_cov)) / (np.sum(basis * (basis @ a_cov)) + 1e-8))
            if abs(proj) >= abs(best_proj):
                best_proj = proj
                best_seed = cand
                best_basis = basis
        seeds[k_idx] = best_seed
        scales[k_idx] = best_proj
        if best_basis is not None:
            rem = rem - best_proj * best_basis
    end = float(np.sum(rem * (rem @ a_cov)))
    return {
        "seeds": seeds,
        "scales": scales,
        "residual": rem,
        "start_energy": start,
        "end_energy": end,
        "stored_bpw": 0.162,
    }


# ---------------------------------------------------------------------------
# Pillar 2 — DQ-ODP + ByteCross
# ---------------------------------------------------------------------------

def plan_disaggregated_quantization(
    prompt_tokens: int,
    resident_decode_bytes: float,
    layer_bytes: float,
    tree_width: int = 1,
) -> Dict[str, object]:
    """
    Prefill and wide tree-verify stay on INT8 activations.
    Single-token / small decode drops activation quantization (BF16 accumulate).
    ODP keeps a 2-layer ring and streams the rest from storage.
    """
    wide = prompt_tokens >= 16 or tree_width >= 16
    return {
        "prefill_activation": "int8" if wide else "bf16",
        "decode_activation": "bf16",
        "prefill_weight": "int8_or_fp4_stream",
        "decode_weight_bpw": mirai_s_effective_bpw(),
        "odp_ring_layers": 2,
        "odp_ring_bytes": 2.0 * float(layer_bytes),
        "resident_decode_bytes": float(resident_decode_bytes),
        "activation_quant_on_decode": False,
    }


def bytecross_crossover_length(
    rho_act: float,
    rho_kv: float,
    b_proj_bytes: float,
    b_kv_per_token_bytes: float,
) -> float:
    return ((1.0 - rho_act) / max(1.0 - rho_kv, 1e-8)) * (b_proj_bytes / max(b_kv_per_token_bytes, 1e-8))


def bytecross_speedup(
    context_tokens: float,
    rho_act: float,
    rho_kv: float,
    b_proj_bytes: float,
    b_kv_per_token_bytes: float,
    bandwidth_bytes_per_s: float,
    c_act_s: float,
    c_kv_s: float,
    t_min: float,
) -> float:
    """S_comp(T) from the ByteCross combined-kernel bound."""
    use_act = 1.0 if rho_act < 1.0 else 0.0
    use_kv = 1.0 if context_tokens > t_min else 0.0
    overhead = bandwidth_bytes_per_s * (c_act_s * use_act + c_kv_s * use_kv)
    numerator = b_proj_bytes + b_kv_per_token_bytes * context_tokens
    denominator = (
        rho_act * b_proj_bytes
        + rho_kv * b_kv_per_token_bytes * context_tokens
        + overhead
    )
    return float(numerator / max(denominator, 1e-8))


def bytecross_route(context_tokens: float, t_star: float, t_min: float) -> str:
    if context_tokens < t_min:
        return "activation_only"
    if context_tokens < t_star:
        return "activation_primary"
    return "kv_sparse_splitk"


# ---------------------------------------------------------------------------
# Pillar 3 — LiLiCorr joint, SparseSpec-L, GDN Tree-Scan
# ---------------------------------------------------------------------------

def normalized_candidate_entropy(probs: np.ndarray) -> np.ndarray:
    """H_t = -1/ln(K) sum_k p log p, shape (gamma,)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    k = p.shape[-1]
    h = -np.sum(p * np.log(p), axis=-1) / math.log(k)
    return h


def sparsespec_controller(
    probs: np.ndarray,
    c_draft: float = 0.35,
    beta_h: float = 1.0,
    gamma_max: Optional[int] = None,
    b0: float = 8.0,
    lambda_w: float = 0.5,
    b_min: int = 2,
    b_max: int = 16,
) -> Dict[str, object]:
    """
    gamma* = max { gamma | prod_{t=1..gamma} alpha_t > c_draft / (1 + gamma c_draft) }
    B_tree = clip(floor(B0 * sum alpha_t (1 + lambda_w H_t)), Bmin, Bmax)
    Empty feasible set falls back to gamma* = 1.
    """
    entropy = normalized_candidate_entropy(probs)
    alpha = np.exp(-beta_h * entropy)
    limit = int(entropy.shape[0] if gamma_max is None else min(gamma_max, entropy.shape[0]))
    gamma_star = 1
    prod = 1.0
    feasible = False
    for t in range(1, limit + 1):
        prod *= float(alpha[t - 1])
        thresh = c_draft / (1.0 + t * c_draft)
        if prod > thresh:
            gamma_star = t
            feasible = True
        else:
            break
    sl = slice(0, gamma_star)
    raw = b0 * float(np.sum(alpha[sl] * (1.0 + lambda_w * entropy[sl])))
    budget = int(np.clip(math.floor(raw), b_min, b_max))
    return {
        "entropy": entropy,
        "alpha": alpha,
        "gamma_star": gamma_star,
        "feasible": feasible,
        "tree_budget": budget,
    }


def lilicorr_joint_loglik(
    marginal_log_probs: np.ndarray,
    s_matrices: np.ndarray,
    path: Sequence[int],
) -> float:
    """J(y) = sum ln p_DFlash + sum ln softmax(S_t[k_t, k_{t+1}])."""
    score = float(marginal_log_probs[0, path[0]])
    for t in range(len(path) - 1):
        row = s_matrices[t, path[t]]
        row = row - np.max(row)
        log_z = math.log(float(np.sum(np.exp(row))))
        score += float(marginal_log_probs[t + 1, path[t + 1]])
        score += float(row[path[t + 1]] - log_z)
    return score


def gdn_step(state: np.ndarray, alpha: float, beta: float, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """S <- alpha * S @ (I - beta k k^T) + beta v k^T. Right-multiply form."""
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    eye = np.eye(state.shape[1], dtype=np.float64)
    return alpha * (state @ (eye - beta * np.outer(k, k))) + beta * np.outer(v, k)


def gdn_tree_scan(
    root_state: np.ndarray,
    nodes: Dict[int, Dict[str, object]],
    root_id: int = 0,
) -> Dict[str, object]:
    """
    Static parent gather, then path-parallel state restore.
    Committed state is published only for an accepted chain, never for a
    rejected sibling.
    """
    gather: List[Tuple[int, int]] = []
    states: Dict[int, np.ndarray] = {root_id: np.array(root_state, dtype=np.float64, copy=True)}
    pending = [nid for nid, node in nodes.items() if node.get("parent") == root_id]
    seen = {root_id}
    while pending:
        nid = pending.pop(0)
        parent = int(nodes[nid]["parent"])  # type: ignore[arg-type]
        if parent not in states:
            pending.append(nid)
            continue
        gather.append((nid, parent))
        node = nodes[nid]
        states[nid] = gdn_step(
            states[parent],
            float(node["alpha"]),
            float(node["beta"]),
            np.asarray(node["k"], dtype=np.float64),
            np.asarray(node["v"], dtype=np.float64),
        )
        seen.add(nid)
        for child, child_node in nodes.items():
            if child_node.get("parent") == nid and child not in seen:
                pending.append(child)
    return {"states": states, "gather_index": gather}


def gdn_commit_chain(root_state: np.ndarray, chain: Sequence[Dict[str, object]]) -> np.ndarray:
    state = np.array(root_state, dtype=np.float64, copy=True)
    for node in chain:
        state = gdn_step(
            state,
            float(node["alpha"]),
            float(node["beta"]),
            np.asarray(node["k"], dtype=np.float64),
            np.asarray(node["v"], dtype=np.float64),
        )
    return state


def gumbel_clamp_uniforms(uniforms: np.ndarray) -> np.ndarray:
    """Keep the top 24 bits and clamp into [2^{-24}, 1 - 2^{-24}]."""
    u = np.asarray(uniforms, dtype=np.float64)
    q = np.floor(u * float(2 ** 24)) / float(2 ** 24)
    return np.clip(q, 2.0 ** -24, 1.0 - 2.0 ** -24)


def gumbel_argmax(logits: np.ndarray, uniforms: np.ndarray) -> int:
    u = gumbel_clamp_uniforms(uniforms)
    gumbel = -np.log(-np.log(u))
    return int(np.argmax(np.asarray(logits, dtype=np.float64) + gumbel))


# ---------------------------------------------------------------------------
# Pillar 4 — OptR / AATC / ReadKV
# ---------------------------------------------------------------------------

def optr_reparameterize(
    keys: np.ndarray,
    queries: np.ndarray,
    values: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Center keys by mu_K and rotate K, Q by the same orthogonal U (eigh of the
    centered key covariance). Values get their own orthogonal V.
    Scores change only by a per-query constant, so softmax is invariant and
    no runtime rotation of the cache is required after offline fusion.
    """
    mu = np.mean(keys, axis=0)
    centered = keys - mu
    cov_k = centered.T @ centered / max(keys.shape[0], 1)
    _evals, u_rot = np.linalg.eigh(0.5 * (cov_k + cov_k.T))
    cov_v = values.T @ values / max(values.shape[0], 1)
    _evals_v, v_rot = np.linalg.eigh(0.5 * (cov_v + cov_v.T))
    return {
        "keys": centered @ u_rot,
        "queries": queries @ u_rot,
        "values": values @ v_rot,
        "U": u_rot,
        "V": v_rot,
        "mu": mu,
    }


def attention_probs(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    scale = math.sqrt(keys.shape[-1])
    scores = (queries @ keys.T) / scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores)
    return exp_s / np.sum(exp_s, axis=-1, keepdims=True)


def aatc_reverse_waterfill(
    channel_weight: np.ndarray,
    sigma2: np.ndarray,
    total_bits: int,
    b_max: int = 8,
) -> np.ndarray:
    """
    Greedy reverse water-filling. Giving one more bit to channel c reduces
    distortion by 0.75 * xi_c * sigma_c^2 * 4^{-b_c}.
    """
    xi = np.asarray(channel_weight, dtype=np.float64)
    var = np.asarray(sigma2, dtype=np.float64)
    bits = np.zeros(xi.shape[0], dtype=np.int64)
    for _ in range(int(total_bits)):
        gain = np.where(bits < b_max, 0.75 * xi * var * np.power(4.0, -bits.astype(np.float64)), -1.0)
        choice = int(np.argmax(gain))
        if gain[choice] < 0.0:
            break
        bits[choice] += 1
    return bits


def aatc_distortion(channel_weight: np.ndarray, sigma2: np.ndarray, bits: np.ndarray) -> float:
    return float(np.sum(channel_weight * sigma2 * np.power(4.0, -np.asarray(bits, dtype=np.float64))))


def readkv_key_bits(
    query: np.ndarray,
    sigma_k: np.ndarray,
    b_read: float,
    choices: Sequence[int] = (0, 2, 4, 8),
) -> np.ndarray:
    """Greedy discrete solution of sum |q_c|^2 sigma_c^2 2^{-2 r_c} s.t. mean r <= B_read."""
    q = np.asarray(query, dtype=np.float64)
    sig = np.asarray(sigma_k, dtype=np.float64)
    dim = q.shape[0]
    budget = float(b_read) * dim
    current = np.zeros(dim, dtype=np.int64)
    grid = list(choices)
    while True:
        best_gain = 0.0
        best_c = -1
        best_next = 0
        used = float(np.sum(current))
        for c in range(dim):
            higher = [g for g in grid if g > int(current[c])]
            if not higher:
                continue
            nxt = higher[0]
            if used - float(current[c]) + nxt > budget + 1e-8:
                continue
            gain = (q[c] ** 2) * (sig[c] ** 2) * (2.0 ** (-2 * int(current[c])) - 2.0 ** (-2 * nxt))
            if gain > best_gain:
                best_gain = gain
                best_c = c
                best_next = nxt
        if best_c < 0 or best_gain <= 0.0:
            break
        current[best_c] = best_next
    return current


def readkv_value_bits(
    attn: np.ndarray,
    value_norm: np.ndarray,
    b_bar: float,
    b_retain: int = 8,
) -> np.ndarray:
    """
    r_tau = clip(ceil(B_bar + log2(a ||v|| / geomean)), 0, B_retain), snapped up
    onto {0, 2, 4, 8}.
    """
    weight = np.clip(np.asarray(attn, dtype=np.float64) * np.asarray(value_norm, dtype=np.float64), 1e-12, None)
    geo = math.exp(float(np.mean(np.log(weight))))
    raw = b_bar + np.log2(weight / geo)
    snapped = np.zeros(raw.shape[0], dtype=np.int64)
    grid = (0, 2, 4, 8)
    for i, value in enumerate(raw):
        target = int(math.ceil(float(value)))
        target = int(np.clip(target, 0, b_retain))
        choice = grid[-1]
        for g in grid:
            if g >= target:
                choice = g
                break
        snapped[i] = min(choice, b_retain)
    return snapped


def readkv_mean_bits(key_bits: np.ndarray, value_bits: np.ndarray) -> float:
    return float(0.5 * (np.mean(key_bits) + np.mean(value_bits)))


# ---------------------------------------------------------------------------
# Hybrid working memory + roofline + end-to-end reference audit
# ---------------------------------------------------------------------------

class ContinuousWorkingMemory:
    """M^{(k)} = M^{(k-1)} concat [K_{<=t}, V_{<=t}]. No re-prefill of past turns."""

    def __init__(self) -> None:
        self._keys: List[np.ndarray] = []
        self._values: List[np.ndarray] = []

    def commit_turn(self, keys: np.ndarray, values: np.ndarray) -> None:
        self._keys.append(np.array(keys, dtype=np.float64, copy=True))
        self._values.append(np.array(values, dtype=np.float64, copy=True))

    @property
    def turn_count(self) -> int:
        return len(self._keys)

    def concatenated(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._keys:
            raise RuntimeError("working memory is empty")
        return np.concatenate(self._keys, axis=0), np.concatenate(self._values, axis=0)


def memory_step_bytes(
    n_params: float,
    bits_per_weight: float,
    n_layers: int,
    context: int,
    d_kv: int,
    bits_per_kv: float,
    meta_bytes: float = 0.0,
) -> float:
    return (
        n_params * bits_per_weight / 8.0
        + 2.0 * n_layers * context * d_kv * bits_per_kv / 8.0
        + meta_bytes
    )


def project_single_pass_tps(bandwidth_gbps: float, traffic_gb: float) -> float:
    return bandwidth_gbps / max(traffic_gb, 1e-12)


def omega_roofline_factors(baseline_bpw: float = 4.613, rho_act: float = 0.72) -> Dict[str, float]:
    """
    Paper accounting: 2.382 bpw is a 48.4% cut vs 4.613 bpw.
    Activation keep-rate 0.72 is modelled as 0.81x of remaining weight traffic,
    leaving 41.8% of the original weight bytes and a 2.39x single-pass TPS.
    """
    omega_bpw = 2.382
    reduction = (baseline_bpw - omega_bpw) / baseline_bpw
    byte_keep = 0.81 * (omega_bpw / baseline_bpw)
    return {
        "omega_bpw": omega_bpw,
        "traffic_reduction": reduction,
        "weight_byte_fraction": byte_keep,
        "single_pass_speedup": 1.0 / byte_keep,
    }


def hybrid_sft_two_pass(
    hidden: np.ndarray,
    target_emb: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    action_logits: np.ndarray,
    action_target: int,
) -> Dict[str, float]:
    """L_HybridSFT = L_Link + L_Act. Link uses the residual adapter; act is CE."""
    pre = np.tanh(hidden @ w1.T)
    pred = hidden + pre @ w2.T
    link = float(np.mean((pred - target_emb) ** 2))
    shifted = action_logits - np.max(action_logits)
    log_z = math.log(float(np.sum(np.exp(shifted))))
    nll = float(-(shifted[action_target] - log_z))
    return {"L_link": link, "L_act": nll, "L_HybridSFT": link + nll}


def run_pillar_reference_audit(seed: int = 7) -> Dict[str, object]:
    """One synthetic pass of Algorithm 1, Phase A + Phase B, at reference scale."""
    rng = np.random.default_rng(seed)

    kinds = ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]
    params = np.array([8e5, 8e5, 8e5, 8e5, 3.2e6, 3.2e6, 3.2e6])
    tr_A = np.array([4.0, 4.2, 3.5, 3.8, 12.0, 11.0, 9.5])
    tr_B = np.array([2.0, 2.1, 1.6, 1.8, 6.0, 5.5, 4.0])
    kappa_A = np.array([8.0, 9.0, 6.0, 7.0, 40.0, 35.0, 30.0])
    kappa_B = np.array([5.0, 5.5, 4.0, 4.5, 20.0, 18.0, 16.0])
    energies = [
        casa_kronecker_energy(a, b, ka, kb)
        for a, b, ka, kb in zip(tr_A, tr_B, kappa_A, kappa_B)
    ]
    cont = casa_closed_form_bits(energies, params, b_target=2.4)
    discrete = casa_discrete_assign(cont, kinds)

    dim = 10
    base = rng.standard_normal((dim, dim))
    cov = base @ base.T + 0.3 * np.eye(dim)
    block = 6
    H = regularized_schur(cov, block=block, lam=1e-3)
    L, d = ldl_factor(H)
    w = rng.standard_normal(block)
    g_row = rng.standard_normal(block) * 0.05
    quant = ksc_curvetq_quantize_row(w, d, L, g_row, Bii=1.4, eta=0.2)

    residual = rng.standard_normal((8, 6)) * 0.2
    stag = stagq_dyadic_encode(residual, n_planes=3)
    a_cov = np.eye(6)
    aw = awsrc_seeded_repair(stag["final_residual"], a_cov, num_seeds=3)  # type: ignore[arg-type]

    dq = plan_disaggregated_quantization(prompt_tokens=128, resident_decode_bytes=8.11e9, layer_bytes=1.8e8)
    t_star = bytecross_crossover_length(0.72, 0.25, 2.68e9, 131072.0)
    speed = bytecross_speedup(t_star, 0.72, 0.25, 2.68e9, 131072.0, 1.53e11, 1e-6, 2e-6, t_min=512.0)

    peaked = np.array([[0.92, 0.04, 0.03, 0.01], [0.9, 0.05, 0.03, 0.02], [0.88, 0.06, 0.04, 0.02]])
    flat = np.full((3, 4), 0.25)
    spec_low = sparsespec_controller(peaked, c_draft=0.2, b_max=16)
    spec_high = sparsespec_controller(flat, c_draft=0.2, b_max=16)

    root = rng.standard_normal((4, 4))
    root = 0.1 * root / np.linalg.norm(root)
    def _node(parent: int) -> Dict[str, object]:
        k = rng.standard_normal(4)
        k = k / np.linalg.norm(k)
        v = rng.standard_normal(4)
        return {"parent": parent, "alpha": 0.9, "beta": 0.2, "k": k, "v": v}

    nodes = {1: _node(0), 2: _node(1), 3: _node(1)}
    scanned = gdn_tree_scan(root, nodes, root_id=0)
    chain = [nodes[1], nodes[2]]
    committed = gdn_commit_chain(root, chain)

    keys = rng.standard_normal((12, 8))
    queries = rng.standard_normal((3, 8))
    values = rng.standard_normal((12, 8))
    opt = optr_reparameterize(keys, queries, values)
    probs_before = attention_probs(queries, keys)
    probs_after = attention_probs(opt["queries"], opt["keys"])

    xi = np.array([10.0, 1.0, 0.1, 0.01])
    sig = np.ones(4)
    bits = aatc_reverse_waterfill(xi, sig, total_bits=8)
    q = np.array([3.0, 0.2, 0.05, 0.01])
    key_bits = readkv_key_bits(q, np.ones(4), b_read=4.0)
    attn = np.array([0.7, 0.2, 0.08, 0.02])
    val_bits = readkv_value_bits(attn, np.ones(4), b_bar=3.0)

    memory = ContinuousWorkingMemory()
    memory.commit_turn(keys[:4], values[:4])
    memory.commit_turn(keys[4:7], values[4:7])
    cat_k, _cat_v = memory.concatenated()

    roof = omega_roofline_factors()
    uniforms = np.array([0.2, 0.4, 0.6, 0.8])
    tok_a = gumbel_argmax(np.array([0.1, 0.2, 1.5, 0.0]), uniforms)
    tok_b = gumbel_argmax(np.array([0.1, 0.2, 1.5, 0.0]), uniforms + 0.1 * 2.0 ** -24)

    return {
        "casa_mean_bits": float(np.sum(params * cont) / np.sum(params)),
        "casa_discrete": discrete,
        "qk_protected": discrete[0] >= 3 and discrete[1] >= 3,
        "schur_psd_min_eig": float(np.min(np.linalg.eigvalsh(H))),
        "ldl_recon_err": float(np.linalg.norm(L @ np.diag(d) @ L.T - H)),
        "viterbi_beats_euclidean": bool(quant["cost"] <= float(quant["euclidean_cost"]) + 1e-8),
        "rotation_applied": bool(quant["rotation_applied"]),
        "effective_bpw": mirai_s_effective_bpw(),
        "traffic_reduction": weight_traffic_reduction(),
        "stagq_mse_monotone": bool(stag["mse_prefix"][-1] <= stag["mse_prefix"][0] + 1e-12),
        "awsrc_energy_drop": float(aw["start_energy"] - aw["end_energy"]),
        "decode_activation": dq["decode_activation"],
        "odp_ring_layers": dq["odp_ring_layers"],
        "bytecross_T": t_star,
        "bytecross_speedup_at_T": speed,
        "gamma_low_entropy": spec_low["gamma_star"],
        "gamma_high_entropy": spec_high["gamma_star"],
        "gdn_chain_match": bool(np.allclose(scanned["states"][2], committed)),
        "optr_softmax_err": float(np.max(np.abs(probs_before - probs_after))),
        "aatc_hot_channel_bits": int(bits[0]),
        "aatc_cold_channel_bits": int(bits[-1]),
        "readkv_hot_key_bits": int(key_bits[0]),
        "readkv_cold_key_bits": int(key_bits[-1]),
        "readkv_value_bits": val_bits.tolist(),
        "working_memory_tokens": int(cat_k.shape[0]),
        "single_pass_speedup": roof["single_pass_speedup"],
        "weight_byte_fraction": roof["weight_byte_fraction"],
        "gumbel_stable": tok_a == tok_b,
    }
