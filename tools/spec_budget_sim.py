"""Compare the SparseSpec-L tree-budget controller with the fixed uzu budget."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load("omegahee_pillars", "src/omegahee_pillars.py")

# uzu decodes one tree per forward pass and accepts a chain from it, so the
# published figure of merit is tokens/forward = 1 + E[accepted chain]
# (crates/legacy/nagare/src/chat/token.rs). Every column below uses that figure.

FIXED_BUDGET = 16
OBSERVED_TOKENS_PER_FORWARD = 2.65  # median of the four measured 9B runs

# Measured pass costs on Apple M5 Pro (bench/results_uzu_mac.json):
#   a single-token pass takes 18.19 ms (54.99 tok/s with speculation off), and a
#   16-slot tree verify pass takes 47.40 ms. Fitting t = fixed + per_candidate *
#   candidates over the two anchors gives fixed = 16.10 ms and 2.086 ms per
#   candidate: 18.19 = 16.10 + 2.086*1, and 47.40 = 16.10 + 2.086*15.
#   A budget-16 tree is a root plus fifteen draft candidates.
PASS_FIXED_MS = 16.10
PASS_PER_CANDIDATE_MS = 2.086


def pass_time_ms(budget):
    # Verify-pass time for a tree of budget slots: one root plus drafts.
    candidates = max(budget - 1, 1)
    return PASS_FIXED_MS + PASS_PER_CANDIDATE_MS * candidates


def tokens_per_second(accepted, budget):
    return (1.0 + accepted) / (pass_time_ms(budget) / 1000.0)


def chain_profile(kind, steps, k, rng, calibrated_sharpness=None):
    """One draft chain as a (steps, K) matrix of per-step candidate marginals."""
    if kind == "code":
        sharpness = np.linspace(0.05, 0.30, steps)
    elif kind == "prose":
        sharpness = np.linspace(0.30, 1.80, steps)
    elif kind == "calibrated":
        if calibrated_sharpness is None:
            raise ValueError("calibrated profile needs the solved sharpness")
        sharpness = np.full(steps, calibrated_sharpness)
    elif kind == "uniform":
        return np.full((steps, k), 1.0 / k)
    else:
        raise ValueError("unknown profile: " + kind)
    return np.stack([rng.dirichlet(np.full(k, s)) for s in sharpness])


def expected_accepted(alpha):
    """Expected accepted speculative count: sum_t prod_{s<=t} alpha_s."""
    total = 0.0
    run = 1.0
    for a in alpha:
        run *= float(a)
        total += run
    return total


def solve_sharpness_for_acceptance(target_accepted, depth, k, lo=0.02, hi=3.0, iters=48):
    """Sharpness whose flat-alpha chain yields the requested accepted count."""

    def accepted_of(sharp):
        rng = np.random.default_rng(1)
        chain = np.stack([rng.dirichlet(np.full(k, sharp)) for _ in range(depth)])
        return expected_accepted(np.exp(-P.normalized_candidate_entropy(chain)))

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if accepted_of(mid) > target_accepted:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate(chains, c_draft, beta_h, b0, lambda_w, b_min, b_max, gamma_max, fixed_budget):
    fixed_depth = max(1, min(fixed_budget - 1, gamma_max))
    rows = len(chains)
    adapt_tokens = 0.0
    adapt_slots = 0.0
    fixed_tokens = 0.0
    filled_tokens = 0.0
    gamma_hist = {}
    budget_hist = {}
    for chain in chains:
        ctrl = P.sparsespec_controller(
            chain, c_draft=c_draft, beta_h=beta_h, gamma_max=gamma_max,
            b0=b0, lambda_w=lambda_w, b_min=b_min, b_max=b_max)
        gamma_star = int(ctrl["gamma_star"])
        budget = int(ctrl["tree_budget"])
        alpha = np.asarray(ctrl["alpha"], dtype=np.float64)[:gamma_star]
        adapt_tokens += 1.0 + expected_accepted(alpha)
        adapt_slots += budget
        ent = P.normalized_candidate_entropy(chain)
        alpha_fixed = np.exp(-beta_h * ent)[:fixed_depth]
        fixed_tokens += 1.0 + expected_accepted(alpha_fixed)
        # Same per-step marginals, full verify depth: isolates how much of the
        # adaptive loss comes from cutting the chain rather than from rejecting it.
        filled_tokens += 1.0 + expected_accepted(alpha_fixed)
        gamma_hist[gamma_star] = gamma_hist.get(gamma_star, 0) + 1
        budget_hist[budget] = budget_hist.get(budget, 0) + 1
    adapt_tokens /= rows
    fixed_tokens /= rows
    filled_tokens /= rows
    adapt_slots /= rows
    return {
        "tokens_per_forward_adaptive": adapt_tokens,
        "tokens_per_forward_fixed": fixed_tokens,
        "tokens_per_forward_filled": filled_tokens,
        "acceptance_gain": adapt_tokens / fixed_tokens,
        "slots_adaptive": adapt_slots,
        "slots_fixed": float(fixed_budget),
        "slot_reduction": 1.0 - adapt_slots / fixed_budget,
        "accepted_per_slot_adaptive": (adapt_tokens - 1.0) / adapt_slots,
        "accepted_per_slot_fixed": (fixed_tokens - 1.0) / fixed_budget,
        "gamma_histogram": dict(sorted(gamma_hist.items())),
        "budget_histogram": dict(sorted(budget_hist.items())),
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--c-draft", type=float, default=0.35)
    ap.add_argument("--beta-h", type=float, default=1.0)
    ap.add_argument("--b0", type=float, default=8.0)
    ap.add_argument("--lambda-w", type=float, default=0.5)
    ap.add_argument("--b-min", type=int, default=2)
    ap.add_argument("--b-max", type=int, default=16)
    ap.add_argument("--gamma-max", type=int, default=16)
    ap.add_argument("--fixed-budget", type=int, default=FIXED_BUDGET)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])
    sharp = solve_sharpness_for_acceptance(
        OBSERVED_TOKENS_PER_FORWARD - 1.0, max(1, args.fixed_budget - 1), args.k)
    out = {
        "config": vars(args),
        "calibration": {
            "sharpness": sharp,
            "target_tokens_per_forward": OBSERVED_TOKENS_PER_FORWARD,
            "source": "bench/results_uzu_mac.json, median of four measured 9B runs",
        },
        "profiles": {},
    }
    for kind in ("code", "calibrated", "prose", "uniform"):
        rng = np.random.default_rng(args.seed)
        chains = [chain_profile(kind, args.steps, args.k, rng, sharp) for _ in range(args.tokens)]
        out["profiles"][kind] = simulate(
            chains, args.c_draft, args.beta_h, args.b0, args.lambda_w, args.b_min,
            args.b_max, args.gamma_max, args.fixed_budget)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print("calibration: sharpness %.4f reproduces %.2f tokens/forward at fixed budget %d"
          % (sharp, OBSERVED_TOKENS_PER_FORWARD, args.fixed_budget))
    print("controller:  c_draft=%.2f beta_H=%.1f B0=%.1f lambda_w=%.1f B=[%d,%d] gamma_max=%d"
          % (args.c_draft, args.beta_h, args.b0, args.lambda_w, args.b_min, args.b_max, args.gamma_max))
    print("")
    hdr = "profile  tok/fwd A  tok/fwd F  gain  slots A  slots F  slot cut"
    print(hdr)
    for kind, r in out["profiles"].items():
        print("%8s %10.3f %10.3f %6.3f %8.2f %8.2f %8.1f%%"
              % (kind, r["tokens_per_forward_adaptive"], r["tokens_per_forward_fixed"],
                 r["acceptance_gain"], r["slots_adaptive"], r["slots_fixed"],
                 r["slot_reduction"] * 100.0))
    print("")
    print("A = entropy controller, F = uzu fixed budget. The controller trades a")
    print("little acceptance for far fewer verify slots, which is only a win if the")
    print("freed slots are reused or the freed bandwidth raises the tokens/forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
