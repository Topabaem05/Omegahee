"""Sweep the tree budget against measured pass costs (companion to spec_budget_sim)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "spec_budget_sim", os.path.join(_ROOT, "tools", "spec_budget_sim.py"))
S = importlib.util.module_from_spec(_spec)
sys.modules["spec_budget_sim"] = S
_spec.loader.exec_module(S)
P = S.P


def sweep(kind, budget_max, args, sharp=None):
    rng = np.random.default_rng(args.seed)
    chains = [S.chain_profile(kind, 16, args.k, rng, sharp) for _ in range(args.tokens)]
    rows = []
    for budget in range(2, budget_max + 1):
        depth = max(1, min(budget - 1, 15))
        acc = 0.0
        for chain in chains:
            ent = P.normalized_candidate_entropy(chain)
            acc += S.expected_accepted(np.exp(-ent)[:depth])
        acc /= len(chains)
        rows.append({
            "budget": budget,
            "candidates": depth,
            "accepted": acc,
            "pass_ms": S.pass_time_ms(budget),
            "tokens_per_s": S.tokens_per_second(acc, budget),
        })
    return rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget-max", type=int, default=16)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])
    sharp = S.solve_sharpness_for_acceptance(1.65, 15, args.k)
    out = {"calibrated_sharpness": sharp, "profiles": {}}
    for kind in ("code", "calibrated", "prose"):
        rows = sweep(kind, args.budget_max, args, sharp if kind == "calibrated" else None)
        best = max(rows, key=lambda r: r["tokens_per_s"])
        at16 = [r for r in rows if r["budget"] == 16][0]
        out["profiles"][kind] = {
            "rows": rows,
            "best_budget": best["budget"],
            "best_tokens_per_s": best["tokens_per_s"],
            "tokens_per_s_at_16": at16["tokens_per_s"],
            "gain_vs_16": best["tokens_per_s"] / at16["tokens_per_s"],
        }
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print("pass model: %.2f ms + %.3f ms per candidate" % (S.PASS_FIXED_MS, S.PASS_PER_CANDIDATE_MS))
    print("measured anchors: 18.19 ms single-token, 47.40 ms at budget 16")
    print("")
    print("%11s %8s %12s %12s %8s" % ("profile", "best B", "best tok/s", "tok/s at 16", "gain"))
    for kind, r in out["profiles"].items():
        print("%11s %8d %12.2f %12.2f %7.2fx"
              % (kind, r["best_budget"], r["best_tokens_per_s"], r["tokens_per_s_at_16"], r["gain_vs_16"]))
    print("")
    print("The optimum is small because a verify pass costs about 2.1 ms per extra")
    print("candidate against a 16.1 ms fixed cost, while acceptance saturates early.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
