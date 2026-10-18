#!/usr/bin/env python3
"""Ω-Hee decode benchmark runner (research plan stage 1).

For every model variant x request it waits until CPU and GPU are at or below 60 °C, runs
`benchmarks-uzu` (engine/scripts/benchmarks/engine-uzu) with `num_runs = runs + 1`, drops the first
run as warm-up, and appends one JSON row with the median, minimum and maximum of every metric, the
theoretical decode upper bound and the bandwidth efficiency.

Loop count and KV mode variants are staged as symlink directories with an edited config.json, so the
converted model is never copied or modified.

    python3 tools/omega_bench.py --model ~/models/Dory1.0-4B-lalamo-bf16 --loops 2 --kv per_loop \
        --requests bench/requests/prompt_2048.json --runs 5 --label baseline
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
BENCH_BINARY = ENGINE / "target" / "release" / "benchmarks-uzu"
SERVER_PROJECT = ENGINE / "scripts" / "benchmarks" / "server"
METRICS = (
    "decode_tps",
    "prompt_tps",
    "time_to_first_token",
    "tokens_per_forward_pass",
    "duration",
    "tokens_count",
    "memory_phys_footprint",
    "memory_resident",
    "memory_graphics_total",
)
MAX_TEMPERATURE = 60.0


def shell(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def temperatures() -> tuple[float, float]:
    output = shell(
        [
            "uv",
            "run",
            "-q",
            "python",
            "-c",
            "from mach import get_temp_sensors; t = get_temp_sensors(); print(t.cpu_avg, t.gpu_avg)",
        ],
        cwd=SERVER_PROJECT,
    )
    cpu, gpu = (float(value) for value in output.split())
    return cpu, gpu


def wait_cooldown(timeout_s: float) -> tuple[float, float]:
    started = time.time()
    cpu, gpu = temperatures()
    while max(cpu, gpu) > MAX_TEMPERATURE and time.time() - started < timeout_s:
        time.sleep(5)
        cpu, gpu = temperatures()
    return cpu, gpu


def stage_variant(model: Path, loops: int | None, kv_mode: str | None) -> Path:
    config = json.loads((model / "config.json").read_text())
    recurrence = config["decoder_config"]["transformer_config"].get("recurrence")
    if recurrence is not None:
        if loops is not None:
            recurrence["num_loops"] = loops
        if kv_mode is not None:
            recurrence["kv_cache_mode"] = kv_mode
    staged = Path(tempfile.mkdtemp(prefix=f"omega-bench-{model.name}-"))
    for path in model.iterdir():
        if path.name != "config.json":
            (staged / path.name).symlink_to(path.resolve())
    (staged / "config.json").write_text(json.dumps(config))
    return staged


def safetensors_tensor_bytes(path: Path) -> dict[str, int]:
    with path.open("rb") as file:
        (header_size,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(header_size))
    return {name: end - start for name, info in header.items() if name != "__metadata__" for start, end in [info["data_offsets"]]}


def decode_bytes_per_token(model: Path, loops: int, start: int, end: int) -> int:
    """Weight bytes one decode step reads: every layer once per execution, plus the readout matrix.
    The input embedding reads one row per token and is ignored."""
    total = 0
    for name, size in safetensors_tensor_bytes(model / "model.safetensors").items():
        if name.startswith("decoder.embedding.input_embedding"):
            continue
        if ".transformer.layers." in name:
            layer = int(name.split(".transformer.layers.")[1].split(".")[0])
            total += size * (loops if start <= layer < end else 1)
        else:
            total += size
    return total


def machine() -> dict[str, str]:
    return {
        "chip": shell(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": shell(["sysctl", "-n", "hw.memsize"]),
        "macos": platform.mac_ver()[0],
        "commit": shell(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT),
    }


def model_hash(model: Path) -> str:
    conversion = model / "conversion.json"
    if conversion.exists():
        return json.loads(conversion.read_text())["output_sha256"]
    digest = hashlib.sha256()
    with (model / "model.safetensors").open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_requests(model_dir: Path, requests: list[Path], runs: int) -> list[list[dict]]:
    process = subprocess.Popen(
        [str(BENCH_BINARY), "-m", str(model_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    results = []
    for request_path in requests:
        request = json.loads(request_path.read_text()) | {"num_runs": runs + 1}
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"benchmarks-uzu exited while running {request_path}")
        results.append(json.loads(line))
    process.stdin.close()
    process.wait()
    return results


def summarize(responses: list[dict]) -> dict:
    measured = responses[1:]
    return {
        metric: {
            "median": statistics.median(response[metric] for response in measured),
            "min": min(response[metric] for response in measured),
            "max": max(response[metric] for response in measured),
        }
        for metric in METRICS
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--loops", type=int, nargs="+", default=[2])
    ap.add_argument("--kv", nargs="+", default=["per_loop"], choices=["per_loop", "last_loop"])
    ap.add_argument("--requests", type=Path, nargs="+", default=[ROOT / "bench/requests/prompt_2048.json"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--label", required=True)
    ap.add_argument("--quantization", default="bf16")
    ap.add_argument("--bandwidth-gbs", type=float, default=307.0, help="peak memory bandwidth for the upper bound")
    ap.add_argument("--cooldown-timeout", type=float, default=900.0)
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/research_runs.jsonl")
    args = ap.parse_args()

    model = args.model.expanduser().resolve()
    config = json.loads((model / "config.json").read_text())
    recurrence = config["decoder_config"]["transformer_config"].get("recurrence") or {}
    info = machine() | {"model": str(model), "model_sha256": model_hash(model), "quantization": args.quantization}
    for loops in args.loops:
        for kv_mode in args.kv:
            variant = stage_variant(model, loops, kv_mode)
            bytes_per_token = decode_bytes_per_token(
                model, loops, recurrence.get("start_layer_index", 0), recurrence.get("end_layer_index", 0)
            )
            upper_bound = args.bandwidth_gbs * 1e9 / bytes_per_token
            cpu, gpu = wait_cooldown(args.cooldown_timeout)
            results = run_requests(variant, args.requests, args.runs)
            for request_path, responses in zip(args.requests, results, strict=True):
                summary = summarize(responses)
                row = info | {
                    "label": args.label,
                    "time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "loops": loops,
                    "kv_mode": kv_mode,
                    "request": request_path.name,
                    "runs": args.runs,
                    "start_temperature": {"cpu": round(cpu, 1), "gpu": round(gpu, 1)},
                    "decode_bytes_per_token": bytes_per_token,
                    "bandwidth_gbs": args.bandwidth_gbs,
                    "decode_upper_bound_tps": round(upper_bound, 2),
                    "bandwidth_efficiency": round(summary["decode_tps"]["median"] / upper_bound, 4),
                    "metrics": summary,
                    "text_head": responses[-1]["text"][:120],
                }
                with args.out.open("a") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                decode = summary["decode_tps"]
                print(
                    f"{args.label} L{loops} {kv_mode} {request_path.name}: decode {decode['median']:.2f} "
                    f"[{decode['min']:.2f}, {decode['max']:.2f}] tok/s, upper {upper_bound:.1f}, "
                    f"efficiency {decode['median'] / upper_bound:.1%}, prefill {summary['prompt_tps']['median']:.0f} tok/s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
