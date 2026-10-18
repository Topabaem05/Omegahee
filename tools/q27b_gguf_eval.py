"""Measure decode speed and WikiText-2 perplexity for 27B GGUF variants.

Speed uses llama-bench's generation-only row (``-p 0``) so prompt processing
does not enter the tok/s number.  Perplexity uses the same raw WikiText-2 file
and context/chunk settings for every model.
"""

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


def swap_used_gb() -> float | None:
    try:
        out = subprocess.run(
            ["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"used = ([0-9.]+)M", out)
    return float(match.group(1)) / 1024.0 if match else None


def run_tracked(cmd: list[str], timeout: int | None = None) -> dict:
    """Run a command, recording wall time and peak RSS from ``ps``."""
    before = swap_used_gb()
    started = time.time()
    with tempfile.TemporaryFile(mode="w+") as out, tempfile.TemporaryFile(mode="w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        peak_kb = 0
        while proc.poll() is None:
            try:
                ps = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                if ps:
                    peak_kb = max(peak_kb, int(ps.split()[0]))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            if timeout and time.time() - started > timeout:
                proc.kill()
                break
            time.sleep(0.25)
        returncode = proc.wait()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    return {
        "cmd": cmd,
        "returncode": returncode,
        "elapsed_s": time.time() - started,
        "peak_rss_gb": peak_kb / 1024.0 / 1024.0 if peak_kb else None,
        "swap_before_gb": before,
        "swap_after_gb": swap_used_gb(),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "stdout": stdout,
    }


def speed(model: str, args: argparse.Namespace) -> dict:
    result = run_tracked(
        [args.llama_bench, "-m", model, "-p", "0", "-n", str(args.tokens),
         "-r", str(args.runs), "-o", "json"],
        timeout=args.timeout,
    )
    row = None
    if result["returncode"] == 0:
        try:
            rows = json.loads(result["stdout"])
            row = next(
                (r for r in rows if r.get("n_prompt") == 0 and r.get("n_gen") == args.tokens),
                rows[0] if rows else None,
            )
        except json.JSONDecodeError:
            row = None
    return {
        "mode": "speed",
        "model": model,
        "file_gb": os.path.getsize(model) / 1e9,
        "bench": row,
        **result,
        "stdout": None,
    }


def perplexity(model: str, args: argparse.Namespace) -> dict:
    result = run_tracked(
        [args.llama_ppl, "-m", model, "-f", args.wiki,
         "-c", str(args.ctx), "--chunks", str(args.chunks),
         "-ngl", "99"],
        timeout=args.timeout,
    )
    text = result["stdout"] + "\n" + result["stderr_tail"]
    match = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text)
    return {
        "mode": "ppl",
        "model": model,
        "file_gb": os.path.getsize(model) / 1e9,
        "ppl": float(match.group(1)) if match else None,
        "ppl_stderr": float(match.group(2)) if match else None,
        **result,
        "stdout": None,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--mode", choices=["speed", "ppl", "both"], default="speed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--llama-bench", default="llama-bench")
    ap.add_argument("--llama-ppl", default="llama-perplexity")
    ap.add_argument("--wiki", default=os.path.expanduser(
        "~/models/qwen38-27b/wikitext/wikitext-2-raw/wiki.test.raw"))
    args = ap.parse_args(argv[1:])

    results = []
    for model in args.models:
        if not os.path.exists(model):
            results.append({"model": model, "error": "missing"})
            continue
        if args.mode in ("speed", "both"):
            print(f"speed: {os.path.basename(model)}", flush=True)
            item = speed(model, args)
            bench = item.get("bench") or {}
            print(
                f"  {bench.get('avg_ts', float('nan')):.2f} tok/s "
                f"+/- {bench.get('stddev_ts', float('nan')):.2f}, "
                f"rss={item.get('peak_rss_gb')} GB, rc={item['returncode']}",
                flush=True,
            )
            results.append(item)
        if args.mode in ("ppl", "both"):
            print(f"ppl: {os.path.basename(model)}", flush=True)
            item = perplexity(model, args)
            print(
                f"  PPL={item.get('ppl')} +/- {item.get('ppl_stderr')}, "
                f"elapsed={item['elapsed_s']:.1f}s, rc={item['returncode']}",
                flush=True,
            )
            results.append(item)

    speeds = [r for r in results if r.get("mode") == "speed" and r.get("bench")]
    ppls = [r for r in results if r.get("mode") == "ppl" and r.get("ppl")]
    report = {
        "args": vars(args),
        "results": results,
        "speed_summary": [
            {
                "model": os.path.basename(r["model"]),
                "file_gb": r["file_gb"],
                "avg_ts": r["bench"].get("avg_ts"),
                "stddev_ts": r["bench"].get("stddev_ts"),
                "peak_rss_gb": r.get("peak_rss_gb"),
            }
            for r in speeds
        ],
        "ppl_summary": [
            {"model": os.path.basename(r["model"]), "file_gb": r["file_gb"],
             "ppl": r.get("ppl"), "ppl_stderr": r.get("ppl_stderr")}
            for r in ppls
        ],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in ("speed_summary", "ppl_summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
