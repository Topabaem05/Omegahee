#!/usr/bin/env python3
"""Run uzu's Dory decoder (engine/ dory_parity_dump test) and compare it with the stage-0 reference.

The fp32 PyTorch reference is the ground truth; uzu computes in BF16, so the stage-0 BF16 tolerances
apply: last-position logit difference <= 0.5, and top-1 must match wherever the reference top-1/top-2
log-probability margin exceeds 0.5 nat.

    python3 compare_uzu.py --backend cpu --loops 2 --kv-mode per_loop --only hello math
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MODEL = Path("/Users/topamini/code/uzu-omega/models/Dory1.0-4B-lalamo-bf16")
VOCAB = 131072
MARGIN = 0.5
LOGIT_TOLERANCE = 0.5


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def run_uzu(args: argparse.Namespace, prompts: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "prompts.json"
    prompts_path.write_text(json.dumps(prompts))
    spec = {
        "model": str(args.model),
        "prompts": str(prompts_path),
        "out_dir": str(out_dir),
        "backend": args.backend,
        "num_loops": args.loops,
        "kv_cache_mode": args.kv_mode,
        "only": [prompt["id"] for prompt in prompts],
    }
    spec_path = out_dir / "spec.json"
    spec_path.write_text(json.dumps(spec))
    env = os.environ | {
        "UZU_DORY_PARITY": str(spec_path),
        "RUSTUP_TOOLCHAIN": "nightly",
        "PATH": "/opt/homebrew/opt/rustup/bin:" + os.environ["PATH"],
    }
    command = ["cargo", "test", "-q", "--release", "-p", "uzu-engine", "--lib", "--", "dory_parity_dump", "--nocapture"]
    subprocess.run(command, cwd=ROOT / "engine", env=env, check=True)


def bf16_envelope(reference: np.lib.npyio.NpzFile, bf16: np.lib.npyio.NpzFile | None) -> tuple[int, float]:
    """Margin violations and top-1 agreement of the BF16 run of the same reference (the BF16 noise floor)."""
    if bf16 is None:
        return 0, 1.0
    top_idx, top_lp = reference["prefill_top_idx"], reference["prefill_top_lp"]
    disagree = bf16["prefill_top_idx"][:, 0] != top_idx[:, 0]
    margins = top_lp[:, 0] - top_lp[:, 1]
    return int((disagree & (margins > MARGIN)).sum()), float(1 - disagree.mean())


def compare(prompt: dict, reference: np.lib.npyio.NpzFile, bf16: np.lib.npyio.NpzFile | None, out_dir: Path) -> dict:
    n = len(prompt["ids"])
    prefill = np.fromfile(out_dir / f"{prompt['id']}_prefill.f32", dtype=np.float32).reshape(n, VOCAB)
    decode = np.fromfile(out_dir / f"{prompt['id']}_decode.f32", dtype=np.float32).reshape(-1, VOCAB)

    def margin_ok(top_idx: np.ndarray, top_lp: np.ndarray, logits: np.ndarray) -> tuple[int, int, int]:
        margins = top_lp[:, 0] - top_lp[:, 1]
        agree = logits.argmax(-1) == top_idx[:, 0]
        confident = margins > MARGIN
        return int(agree.sum()), int((agree | ~confident).sum()), len(agree)

    p_agree, p_ok, p_total = margin_ok(reference["prefill_top_idx"], reference["prefill_top_lp"], prefill)
    steps = len(decode)
    d_agree, d_ok, d_total = margin_ok(reference["decode_top_idx"][:steps], reference["decode_top_lp"][:steps], decode)
    lp = log_softmax(decode) if steps else decode
    top1_lp_gap = 0.0 if not steps else float(np.abs(lp[np.arange(d_total), reference["decode_top_idx"][:steps, 0]] - reference["decode_top_lp"][:steps, 0]).max())
    last_maxabs = float(np.abs(prefill[-1] - reference["last_logits"]).max())
    allowed_violations, bf16_agreement = bf16_envelope(reference, bf16)
    prefill_ok = p_total - p_ok <= allowed_violations
    return {
        "prompt": prompt["id"],
        "last_logits_maxabs": last_maxabs,
        "prefill_top1_agree": f"{p_agree}/{p_total}",
        "prefill_margin_ok": f"{p_ok}/{p_total}",
        "decode_top1_agree": f"{d_agree}/{d_total}",
        "decode_margin_ok": f"{d_ok}/{d_total}",
        "decode_top1_lp_maxabs": top1_lp_gap,
        "bf16_reference_violations": allowed_violations,
        "bf16_reference_agreement": round(bf16_agreement, 4),
        "pass": last_maxabs <= LOGIT_TOLERANCE and prefill_ok and d_ok == d_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--backend", choices=["cpu", "metal"], default="cpu")
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--kv-mode", choices=["per_loop", "last_loop"], default="per_loop")
    ap.add_argument("--only", nargs="*", default=["hello", "math", "gcd", "ko", "think"])
    ap.add_argument("--decode", type=int, default=31)
    ap.add_argument("--ref-dir", type=Path, default=HERE / "out" / "matrix")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    all_prompts = {prompt["id"]: prompt for prompt in json.loads((HERE / "prompts.json").read_text())}
    tag = f"L{args.loops}_{args.kv_mode}"
    out_dir = HERE / "out" / f"uzu_{args.backend}_{tag}"
    references, prompts = {}, []
    for pid in args.only:
        suffix = "_long" if pid == "long" else ""
        reference = np.load(args.ref_dir / f"{pid}_{tag}_float32{suffix}.npz")
        bf16_path = args.ref_dir / f"{pid}_{tag}_bfloat16{suffix}.npz"
        references[pid] = (reference, np.load(bf16_path) if bf16_path.exists() else None)
        decode_ids = [int(token) for token in reference["gen_ids"][: min(args.decode, len(reference["gen_ids"]) - 1)]]
        prompts.append({"id": pid, "ids": all_prompts[pid]["ids"], "decode_ids": decode_ids})
    if not args.skip_run:
        run_uzu(args, prompts, out_dir)
    rows = [
        compare(prompt, *references[prompt["id"]], out_dir) | {"backend": args.backend, "config": tag} for prompt in prompts
    ]
    for row in rows:
        print(json.dumps(row))
    existing = json.loads(args.out.read_text()) if args.out.exists() else []
    existing = [row for row in existing if (row["backend"], row["config"], row["prompt"]) not in {(r["backend"], r["config"], r["prompt"]) for r in rows}]
    args.out.write_text(json.dumps(existing + rows, indent=1))
    return 0 if all(row["pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
