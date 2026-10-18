#!/usr/bin/env python3
"""Dory decode speed per loop count through llama-completion, with the clean gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROMPT = "Write a Python function that returns the greatest common divisor of two positive integers."
EVAL_RE = re.compile(r"\beval time =\s*([0-9.]+) ms /\s*(\d+) runs.*?([0-9.]+) tokens per second")
PROMPT_RE = re.compile(r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens")


def gate(path: Path) -> dict:
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = {"ok": False, "stdout": result.stdout, "stderr": result.stderr}
    report["exit_code"] = result.returncode
    return report


def source_state(repo: Path) -> dict:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", str(repo), "diff"], capture_output=True, text=True).stdout
    return {
        "head": head,
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def run_once(binary: Path, model: str, loops: int, tokens: int, ctx: int) -> dict:
    cmd = [
        str(binary), "-m", model, "-p", PROMPT, "-n", str(tokens), "--temp", "0",
        "-c", str(ctx), "-fa", "on", "-no-cnv", "--no-display-prompt", "--ignore-eos",
        "--override-kv", f"dory.num_loops=int:{loops}",
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "error": proc.stderr[-2000:]}
    ev = EVAL_RE.search(proc.stderr)
    pp = PROMPT_RE.search(proc.stderr)
    if not ev:
        return {"returncode": 0, "error": "eval timing not found"}
    return {
        "returncode": 0,
        "elapsed_s": time.time() - started,
        "eval_ms": float(ev.group(1)),
        "eval_runs": int(ev.group(2)),
        "tok_s": float(ev.group(3)),
        "prompt_ms": float(pp.group(1)) if pp else None,
        "prompt_tokens": int(pp.group(2)) if pp else None,
    }


def summarize(samples: list[dict]) -> dict:
    speeds = [s["tok_s"] for s in samples if "tok_s" in s]
    if not speeds:
        return {"n": 0}
    return {
        "n": len(speeds),
        "median_tok_s": statistics.median(speeds),
        "mean_tok_s": statistics.fmean(speeds),
        "stdev_tok_s": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "min_tok_s": min(speeds),
        "max_tok_s": max(speeds),
        "median_ms_per_token": 1000.0 / statistics.median(speeds),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="Dory llama.cpp checkout")
    ap.add_argument("--build", default="build-m5")
    ap.add_argument("--model", required=True)
    ap.add_argument("--loops", default="1,2,4")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--label", required=True)
    ap.add_argument("--source-note", default=None,
                    help="what source the binary was built from, if not the working tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty-env", action="store_true",
                    help="run even if the clean gate fails; the record is marked provisional")
    args = ap.parse_args()

    repo = Path(args.repo)
    binary = repo / args.build / "bin" / "llama-completion"
    gate_path = root / "tools" / "clean_env_check.py"
    before = gate(gate_path)
    if not before.get("ok") and not args.allow_dirty_env:
        print(json.dumps(before, indent=2))
        return 1

    results = {}
    for loops in [int(x) for x in args.loops.split(",")]:
        samples = []
        for i in range(args.runs):
            sample = run_once(binary, args.model, loops, args.tokens, args.ctx)
            sample["run"] = i + 1
            samples.append(sample)
            print(f"loops={loops} run={i + 1} {sample.get('tok_s', sample.get('error'))}", flush=True)
        results[str(loops)] = {"summary": summarize(samples), "samples": samples}

    record = {
        "label": args.label,
        "provisional": not before.get("ok"),
        "binary": str(binary),
        "source": source_state(repo),
        "source_note": args.source_note,
        "model": args.model,
        "prompt": PROMPT,
        "tokens": args.tokens,
        "context": args.ctx,
        "flash_attn": "on",
        "gate_before": before,
        "gate_after": gate(gate_path),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for loops, r in results.items():
        s = r["summary"]
        if s["n"]:
            print(f"loops={loops} median={s['median_tok_s']:.2f} stdev={s['stdev_tok_s']:.2f} n={s['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
