#!/usr/bin/env python3
"""Summarize the stage-0 reference matrix: determinism, BF16-vs-FP32 error, KV modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SHORT = ["hello", "math", "gcd", "ko", "think"]


def load(d: Path, name: str):
    p = d / f"{name}.npz"
    return np.load(p) if p.exists() else None


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def trace_keys(z):
    return sorted(k for k in z.files if k.startswith("trace/x"))


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def violation(a, b, atol, rtol):
    return float(np.mean(np.abs(a - b) > atol + rtol * np.abs(b)))


def top1_lp_gap(zt, zr):
    """Max |logprob of the reference top-1 token| difference over prefill positions."""
    ti, tl = zt["prefill_top_idx"], zt["prefill_top_lp"]
    ri, rl = zr["prefill_top_idx"], zr["prefill_top_lp"]
    same = ti[:, 0] == ri[:, 0]
    return float(same.mean()), float(np.abs(tl[:, 0] - rl[:, 0]).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    report = {"determinism": [], "bf16_vs_fp32": [], "kv_modes": [], "recompute": [], "long": []}

    for loops in (1, 2, 4):
        for kv in ("per_loop", "last_loop"):
            for pid in SHORT:
                base = f"{pid}_L{loops}_{kv}"
                a, b = load(d, f"{base}_float32"), load(d, f"{base}_float32_r2")
                if a is not None and b is not None:
                    keys = [k for k in a.files if k.startswith("trace/")]
                    maxdiff = max(float(np.abs(a[k] - b[k]).max()) for k in keys)
                    report["determinism"].append({
                        "run": base, "gen_equal": bool(np.array_equal(a["gen_ids"], b["gen_ids"])),
                        "logits_bitwise": bool(np.array_equal(a["last_logits"], b["last_logits"])),
                        "trace_maxabs": maxdiff,
                    })
                f32, bf = load(d, f"{base}_float32"), load(d, f"{base}_bfloat16")
                if f32 is not None and bf is not None:
                    keys = trace_keys(f32)
                    errs = [rel_err(bf[k], f32[k]) for k in keys]
                    viol = [violation(bf[k], f32[k], 1e-2, 1e-2) for k in keys]
                    t1, lpgap = top1_lp_gap(bf, f32)
                    report["bf16_vs_fp32"].append({
                        "run": base,
                        "layer_relerr_first": errs[0], "layer_relerr_max": max(errs),
                        "layer_relerr_last": errs[-1],
                        "viol_frac_max_atol_rtol_1e-2": max(viol),
                        "logits_maxabs": float(np.abs(bf["last_logits"] - f32["last_logits"]).max()),
                        "prefill_top1_agree": t1, "prefill_top1_lp_maxabs": lpgap,
                        "greedy_common_prefix": f"{common_prefix(bf['gen_ids'], f32['gen_ids'])}/{len(f32['gen_ids'])}",
                    })
        for pid in SHORT:
            pl, ll = load(d, f"{pid}_L{loops}_per_loop_float32"), load(d, f"{pid}_L{loops}_last_loop_float32")
            if pl is not None and ll is not None:
                report["kv_modes"].append({
                    "run": f"{pid}_L{loops}",
                    "prefill_logits_bitwise": bool(np.array_equal(pl["last_logits"], ll["last_logits"])),
                    "greedy_common_prefix": f"{common_prefix(pl['gen_ids'], ll['gen_ids'])}/{len(pl['gen_ids'])}",
                })

    for summ in sorted(d.glob("summary_*.json")):
        for r in json.loads(summ.read_text())["runs"]:
            row = {k: r[k] for k in ("name", "recompute_maxabs_logit", "recompute_argmax_equal", "kv_slots", "gen_sha1")}
            (report["long"] if r["prompt"] == "long" else report["recompute"]).append(row)
        unread = json.loads(summ.read_text())["unread_tensors"]
        if unread:
            report.setdefault("unread", {})[summ.name] = unread

    Path(args.out).write_text(json.dumps(report, indent=1))
    for section, rows in report.items():
        print(f"== {section} ({len(rows)})")
        for row in rows if isinstance(rows, list) else [rows]:
            print("  ", json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
