"""Repeatable 27B decode benchmark for the uzu/Mirai CLI.

Each sample is a separate ``bench`` process so model load, swap state and
thermal drift are recorded per run.  The task file sets ``include_replies`` so
the speculator's ``tokens_per_forward_pass`` is available when speculation is
enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time


def swap_used_gb() -> float | None:
    try:
        out = subprocess.run(
            ["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"used = ([0-9.]+)M", out)
    return float(match.group(1)) / 1024.0 if match else None


def default_task(args: argparse.Namespace) -> dict:
    return {
        "identifier": args.identifier,
        "repo_id": args.repo_id,
        "number_of_runs": 1,
        "tokens_limit": args.tokens,
        "greedy": True,
        "reasoning_effort": "disabled",
        "include_replies": True,
        "messages": [{"role": "user", "content": args.prompt}],
    }


def parse_rows(rows: list[dict], swap_before: float | None) -> dict:
    speeds, tpf, passes, memory = [], [], [], []
    for row in rows:
        if row.get("generate_tokens_per_second") is not None:
            speeds.append(float(row["generate_tokens_per_second"]))
        if row.get("memory_used"):
            memory.append(float(row["memory_used"]) / 1e9)
        for reply in row.get("replies") or []:
            spec = ((reply.get("stats") or {}).get("speculator_stats")) or {}
            if "tokens_per_forward_pass" in spec:
                tpf.append(float(spec["tokens_per_forward_pass"]))
            if "num_decode_forward_passes" in spec:
                passes.append(float(spec["num_decode_forward_passes"]))
    return {
        "tok_s": speeds[0] if speeds else None,
        "tokens_per_forward": tpf[0] if tpf else None,
        "decode_passes": passes[0] if passes else None,
        "memory_gb": max(memory) if memory else None,
        "swap_before_gb": swap_before,
        "swap_after_gb": swap_used_gb(),
    }


def summarise(samples: list[dict]) -> dict:
    speeds = [s["tok_s"] for s in samples if s["tok_s"] is not None]
    tpf = [s["tokens_per_forward"] for s in samples if s["tokens_per_forward"] is not None]
    mem = [s["memory_gb"] for s in samples if s["memory_gb"] is not None]
    swaps = [s["swap_before_gb"] for s in samples if s["swap_before_gb"] is not None]
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
        "median_swap_before_gb": statistics.median(swaps) if swaps else None,
    }
    if speeds and tpf:
        out["implied_pass_ms"] = 1000.0 * statistics.median(tpf) / statistics.median(speeds)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", help="existing task JSON; overrides prompt options")
    ap.add_argument("--prompt", default="Write one short paragraph about the sea.")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--identifier", default="q27b-bench")
    ap.add_argument("--repo-id", default="trymirai/Qwen3.8-27B-M")
    args = ap.parse_args(argv[1:])

    if args.task:
        with open(args.task, encoding="utf-8") as handle:
            task = json.load(handle)
        task["number_of_runs"] = 1
        task["include_replies"] = True
    else:
        task = default_task(args)

    workdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(workdir, exist_ok=True)
    task_path = os.path.join(workdir, "_q27b_task.json")
    row_path = os.path.join(workdir, "_q27b_row.json")
    with open(task_path, "w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2)

    samples = []
    for run in range(1, args.runs + 1):
        before = swap_used_gb()
        started = time.time()
        proc = subprocess.run(
            [args.cli, "bench", args.model_dir, task_path, row_path],
            capture_output=True, text=True,
        )
        elapsed = time.time() - started
        sample = {
            "run": run,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "swap_before_gb": before,
            "swap_after_gb": swap_used_gb(),
        }
        if proc.returncode != 0:
            sample["error"] = (proc.stderr or proc.stdout)[-2000:]
            samples.append(sample)
            print(f"run {run}/{args.runs}: failed rc={proc.returncode}", flush=True)
            continue
        with open(row_path, encoding="utf-8") as handle:
            rows = json.load(handle)
        sample.update(parse_rows(rows, before))
        samples.append(sample)
        print(
            f"run {run}/{args.runs}: {sample['tok_s']:.2f} tok/s, "
            f"tpf={sample['tokens_per_forward']}, "
            f"mem={sample['memory_gb']} GB, swap={before} GB",
            flush=True,
        )

    report = {
        "cli": os.path.abspath(args.cli),
        "model_dir": os.path.abspath(args.model_dir),
        "runs": args.runs,
        "task": task,
        "summary": summarise(samples),
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["speed_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
