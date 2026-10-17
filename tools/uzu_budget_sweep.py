"""Budget sweep with the instrumentation the first attempt lacked."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys


def run_once(cli: str, model_dir: str, task: str, out: str) -> dict:
    subprocess.run(
        [cli, "bench", model_dir, task, out],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with open(out, encoding="utf-8") as handle:
        rows = json.load(handle)
    speeds, tpf, passes, tokens = [], [], [], []
    for row in rows:
        if row.get("generate_tokens_per_second") is not None:
            speeds.append(row["generate_tokens_per_second"])
        tokens.append(row.get("tokens_count_output", 0))
        for reply in row.get("replies") or []:
            stats = reply.get("stats") or {}
            spec = stats.get("speculator_stats") or {}
            if "tokens_per_forward_pass" in spec:
                tpf.append(spec["tokens_per_forward_pass"])
            if "num_decode_forward_passes" in spec:
                passes.append(spec["num_decode_forward_passes"])
    return {
        "speeds": speeds,
        "tokens_per_forward": tpf,
        "decode_passes": passes,
        "output_tokens": tokens,
    }


def summarise(samples: dict) -> dict:
    speeds = samples["speeds"]
    tpf = samples["tokens_per_forward"]
    toks = samples["output_tokens"]
    out = {
        "n": len(speeds),
        "mean": statistics.mean(speeds) if speeds else 0.0,
        "median": statistics.median(speeds) if speeds else 0.0,
        "stdev": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "min": min(speeds) if speeds else 0.0,
        "max": max(speeds) if speeds else 0.0,
        "tpf_mean": statistics.mean(tpf) if tpf else None,
        "output_tokens_mean": statistics.mean(toks) if toks else 0.0,
    }
    if speeds and tpf:
        mean_tpf = statistics.mean(tpf)
        if mean_tpf:
            out["implied_pass_ms"] = 1000.0 * mean_tpf / statistics.mean(speeds)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--budgets", default="4,8,16")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--setter", required=True, help="tools/uzu_budget_override.py")
    ap.add_argument("--out", default="budget_sweep.json")
    args = ap.parse_args(argv[1:])
    budgets = [int(b) for b in args.budgets.split(",")]
    samples = {b: {"speeds": [], "tokens_per_forward": [], "decode_passes": [], "output_tokens": []} for b in budgets}
    order_rng = random.Random(20261018)
    tmp = os.path.join(os.path.dirname(args.out) or ".", "_sweep_row.json")
    for rnd in range(args.rounds):
        batch = budgets[:]
        order_rng.shuffle(batch)
        for budget in batch:
            subprocess.run([sys.executable, args.setter, args.model_dir, str(budget)],
                           check=True, stdout=subprocess.DEVNULL)
            got = run_once(args.cli, args.model_dir, args.task, tmp)
            for key in samples[budget]:
                samples[budget][key].extend(got[key])
        print(f"round {rnd + 1}/{args.rounds}", flush=True)
    report = {"budgets": budgets, "rounds": args.rounds,
              "per_budget": {str(b): summarise(samples[b]) for b in budgets},
              "raw": {str(b): samples[b] for b in budgets}}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["per_budget"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
