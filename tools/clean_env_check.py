#!/usr/bin/env python3
"""Refuse to benchmark on a contended or low-memory machine."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys


BUSY_PATTERNS = (
    "train_yolo_seg",
    "train_dense_mps",
    "fuse_yolo_unet",
    "jadx",
    "llama-cli",
    "llama-server",
    "llama-bench",
    "mirai-aarch64",
)


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def gpu_utilization() -> int | None:
    out = run(["ioreg", "-r", "-c", "IOAccelerator", "-l"])
    values = [int(x) for x in re.findall(r'"Device Utilization %"\s*=\s*(\d+)', out)]
    return max(values) if values else None


def swap_used_mb() -> float | None:
    out = run(["sysctl", "vm.swapusage"])
    match = re.search(r"used = ([0-9.]+)M", out)
    return float(match.group(1)) if match else None


def busy_processes() -> list[str]:
    out = run(["ps", "-Ao", "pid=,command="])
    found = []
    for line in out.splitlines():
        if "clean_env_check" in line:
            continue
        if any(pattern in line for pattern in BUSY_PATTERNS):
            found.append(line.strip())
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-gpu", type=int, default=10)
    ap.add_argument("--max-swap-mb", type=float, default=1024.0)
    ap.add_argument("--min-disk-gb", type=float, default=20.0)
    ap.add_argument("--disk-path", default="/System/Volumes/Data")
    args = ap.parse_args()

    gpu = gpu_utilization()
    swap = swap_used_mb()
    free_gb = shutil.disk_usage(args.disk_path).free / 1e9
    busy = busy_processes()

    reasons = []
    if gpu is None:
        reasons.append("GPU utilization unavailable")
    elif gpu > args.max_gpu:
        reasons.append(f"GPU utilization {gpu}% > {args.max_gpu}%")
    if swap is None:
        reasons.append("swap usage unavailable")
    elif swap > args.max_swap_mb:
        reasons.append(f"swap {swap:.0f}MB > {args.max_swap_mb:.0f}MB")
    if free_gb < args.min_disk_gb:
        reasons.append(f"disk free {free_gb:.1f}GB < {args.min_disk_gb:.0f}GB")
    if busy:
        reasons.append(f"{len(busy)} busy process(es)")

    report = {
        "ok": not reasons,
        "gpu_utilization_pct": gpu,
        "swap_used_mb": swap,
        "disk_free_gb": round(free_gb, 2),
        "busy_processes": busy,
        "thresholds": {
            "max_gpu_pct": args.max_gpu,
            "max_swap_mb": args.max_swap_mb,
            "min_disk_gb": args.min_disk_gb,
        },
        "reasons": reasons,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
