"""A/B benchmark for llama.cpp MTP speculative decoding on Qwen3.8-27B.

The important trap this tool avoids: ``llama-cli`` defaults to the model's
training context (262,144 tokens for Qwen3.8-27B), which allocates a 16 GiB
KV cache.  Every run here passes an explicit ``--context`` (default 4096).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time


def run_once(cmd: list[str], timeout: int) -> dict:
    started = time.time()
    with tempfile.TemporaryFile(mode="w+") as out, tempfile.TemporaryFile(mode="w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        peak_kb = 0
        while proc.poll() is None:
            try:
                value = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                if value:
                    peak_kb = max(peak_kb, int(value.split()[0]))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            if time.time() - started > timeout:
                proc.kill()
                break
            time.sleep(0.2)
        returncode = proc.wait()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    gen = re.findall(r"Generation:\s*([0-9.]+)\s*t/s", stdout)
    prompt = re.findall(r"Prompt:\s*([0-9.]+)\s*t/s", stdout)
    acc = re.findall(r"#mean acc len = ([0-9.]+)", stderr)
    rates = re.findall(r"#acc rate/pos = \(([^)]*)\)", stderr)
    dur = re.findall(r"dur\(b,g,a\) = ([0-9.]+), ([0-9.]+), ([0-9.]+) ms", stderr)
    kv = re.findall(r"llama_kv_cache:\s+size = ([0-9.]+) MiB", stderr)
    return {
        "returncode": returncode,
        "elapsed_s": time.time() - started,
        "peak_rss_gb": peak_kb / 1024.0 / 1024.0 if peak_kb else None,
        "generation_tok_s": float(gen[-1]) if gen else None,
        "prompt_tok_s": float(prompt[-1]) if prompt else None,
        "mean_accept_length": float(acc[-1]) if acc else None,
        "accept_rates": [float(x) for x in rates[-1].split(",")] if rates else None,
        "draft_ms": float(dur[-1][1]) if dur else None,
        "kv_cache_mib": float(kv[0]) if kv else None,
        "stderr_tail": stderr[-1500:],
    }


def base_args(args: argparse.Namespace) -> list[str]:
    return [
        args.llama_cli, "-m", args.target, "-c", str(args.context),
        "-st", "--simple-io", "-p", args.prompt,
        "-n", str(args.tokens), "--temp", "0", "--top-k", "1",
        "--reasoning", "off", "-s", str(args.seed), "-lv", "4",
    ]


def mtp_args(args: argparse.Namespace, n_max: int) -> list[str]:
    return [
        args.llama_cli, "-m", args.target, "-md", args.mtp,
        "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max),
        "-ngld", "99", "-c", str(args.context),
        "-st", "--simple-io", "-p", args.prompt,
        "-n", str(args.tokens), "--temp", "0", "--top-k", "1",
        "--reasoning", "off", "-s", str(args.seed), "-lv", "4",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-cli", default="llama-cli")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-max", default="3,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument(
        "--prompt",
        default="Explain in detail how a transformer attention mechanism works.",
    )
    args = ap.parse_args()

    n_max_values = [int(x) for x in args.n_max.split(",") if x.strip()]
    configs: dict[str, list[str]] = {"base": base_args(args)}
    for n in n_max_values:
        configs[f"mtp{n}"] = mtp_args(args, n)

    results: dict[str, list[dict]] = {name: [] for name in configs}
    for rnd in range(args.rounds):
        for name, cmd in configs.items():
            item = run_once(cmd, args.timeout)
            item["round"] = rnd + 1
            results[name].append(item)
            print(
                f"round {rnd + 1} {name}: rc={item['returncode']} "
                f"gen={item['generation_tok_s']} acc={item['mean_accept_length']} "
                f"rss={item['peak_rss_gb']}",
                flush=True,
            )

    summary = {}
    for name, rows in results.items():
        speeds = [r["generation_tok_s"] for r in rows if r["generation_tok_s"]]
        accs = [r["mean_accept_length"] for r in rows if r["mean_accept_length"]]
        rss = [r["peak_rss_gb"] for r in rows if r["peak_rss_gb"]]
        summary[name] = {
            "n": len(rows),
            "speed_samples": len(speeds),
            "median_tok_s": statistics.median(speeds) if speeds else None,
            "mean_tok_s": statistics.mean(speeds) if speeds else None,
            "stdev_tok_s": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
            "median_accept_length": statistics.median(accs) if accs else None,
            "max_rss_gb": max(rss) if rss else None,
        }
    if summary.get("base", {}).get("median_tok_s") and summary.get("mtp3"):
        summary["mtp3"]["speedup_vs_base"] = (
            summary["mtp3"]["median_tok_s"] / summary["base"]["median_tok_s"]
        )
    if summary.get("base", {}).get("median_tok_s") and summary.get("mtp4"):
        summary["mtp4"]["speedup_vs_base"] = (
            summary["mtp4"]["median_tok_s"] / summary["base"]["median_tok_s"]
        )

    report = {"args": vars(args), "summary": summary, "runs": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
