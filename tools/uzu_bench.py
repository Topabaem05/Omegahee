#!/usr/bin/env python3
"""Ten-sample uzu baseline with a hard clean-environment gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time


def swap_used_mb() -> float | None:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([0-9.]+)M", out)
    return float(match.group(1)) if match else None


def gate(path: str) -> dict:
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = {"ok": False, "stdout": result.stdout, "stderr": result.stderr}
    report["exit_code"] = result.returncode
    return report


def run_once(cli: str, model_dir: str, task: str, out_path: str) -> dict:
    before = swap_used_mb()
    started = time.time()
    proc = subprocess.run([cli, "bench", model_dir, task, out_path], capture_output=True, text=True)
    elapsed = time.time() - started
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "error": (proc.stderr or proc.stdout)[-2000:]}
    with open(out_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    speeds, tpf, passes, tokens, memory = [], [], [], [], []
    for row in rows:
        if row.get("generate_tokens_per_second") is not None:
            speeds.append(float(row["generate_tokens_per_second"]))
        if row.get("memory_used"):
            memory.append(float(row["memory_used"]) / 1e9)
        tokens.append(row.get("tokens_count_output", 0))
        for reply in row.get("replies") or []:
            spec = ((reply.get("stats") or {}).get("speculator_stats")) or {}
            if "tokens_per_forward_pass" in spec:
                tpf.append(float(spec["tokens_per_forward_pass"]))
            if "num_decode_forward_passes" in spec:
                passes.append(float(spec["num_decode_forward_passes"]))
    return {
        "returncode": 0,
        "elapsed_s": elapsed,
        "tok_s": speeds[0] if speeds else None,
        "tokens_per_forward": tpf[0] if tpf else None,
        "decode_passes": passes[0] if passes else None,
        "output_tokens": tokens[0] if tokens else None,
        "memory_gb": max(memory) if memory else None,
        "swap_before_mb": before,
        "swap_after_mb": swap_used_mb(),
    }


def summarise(samples: list[dict]) -> dict:
    speeds = [s["tok_s"] for s in samples if s.get("tok_s")]
    tpf = [s["tokens_per_forward"] for s in samples if s.get("tokens_per_forward")]
    mem = [s["memory_gb"] for s in samples if s.get("memory_gb")]
    out = {
        "n": len(samples),
        "speed_samples": len(speeds),
        "median_tok_s": statistics.median(speeds) if speeds else None,
        "mean_tok_s": statistics.mean(speeds) if speeds else None,
        "stdev_tok_s": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "min_tok_s": min(speeds) if speeds else None,
        "max_tok_s": max(speeds) if speeds else None,
        "median_tokens_per_forward": statistics.median(tpf) if tpf else None,
        "max_memory_gb": max(mem) if mem else None,
    }
    if speeds and tpf and statistics.median(speeds):
        out["implied_pass_ms"] = 1000.0 * statistics.median(tpf) / statistics.median(speeds)
    return out


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=os.path.join(root, "bench/results/uzu_baseline.json"))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--task", help="existing uzu task JSON")
    ap.add_argument("--prompt", default="Write one short paragraph about the sea.")
    ap.add_argument("--repo-id", default="trymirai/Qwen3.5-9B-M")
    ap.add_argument("--gate", default=os.path.join(root, "tools/clean_env_check.py"))
    ap.add_argument("--no-gate", action="store_true", help="debug only; results are not current")
    args = ap.parse_args()

    gate_reports = []
    if not args.no_gate:
        for run in range(1, args.runs + 1):
            report = gate(args.gate)
            gate_reports.append(report)
            if report.get("exit_code") != 0:
                print(json.dumps({"blocked_at_run": run, "gate": report}, indent=2))
                return 2

    if args.task:
        with open(args.task, encoding="utf-8") as handle:
            task_obj = json.load(handle)
    else:
        task_obj = {
            "identifier": "uzu-baseline",
            "repo_id": args.repo_id,
            "number_of_runs": 1,
            "tokens_limit": args.tokens,
            "greedy": True,
            "reasoning_effort": "disabled",
            "include_replies": True,
            "messages": [{"role": "user", "content": args.prompt}],
        }
    task_obj["number_of_runs"] = 1
    task_obj["include_replies"] = True

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    task_path = os.path.join(out_dir, "_uzu_task.json")
    row_path = os.path.join(out_dir, "_uzu_row.json")
    with open(task_path, "w", encoding="utf-8") as handle:
        json.dump(task_obj, handle, indent=2)

    samples = []
    for run in range(1, args.runs + 1):
        sample = run_once(args.cli, args.model_dir, task_path, row_path)
        sample["run"] = run
        samples.append(sample)
        print(
            f"run {run}/{args.runs}: rc={sample.get('returncode')} "
            f"tok_s={sample.get('tok_s')} tpf={sample.get('tokens_per_forward')} "
            f"mem={sample.get('memory_gb')} swap={sample.get('swap_before_mb')}",
            flush=True,
        )

    report = {
        "cli": os.path.abspath(args.cli),
        "model_dir": os.path.abspath(args.model_dir),
        "task": task_obj,
        "runs": args.runs,
        "gate": gate_reports,
        "summary": summarise(samples),
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["speed_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
