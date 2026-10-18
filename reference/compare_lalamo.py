#!/usr/bin/env python3
"""Compare lalamo's Dory (dory-support branch) with the stage-0 PyTorch reference outputs.

Run with the lalamo environment:
    uv run --project LALAMO python compare_lalamo.py --model DIR --prompts prompts.json \
        --ref-dir out/matrix --out result.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lalamo  # noqa: F401  (configures JAX before it is imported)
import jax
import jax.numpy as jnp
import numpy as np
from lalamo.modules import DecoderForwardPassConfig, Keychain

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lalamo_dory import import_dory  # noqa: E402

SHORT = ["hello", "math", "gcd", "ko", "think"]


def rel_err(result: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(result - reference) / max(np.linalg.norm(reference), 1e-30))


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


class Runner:
    def __init__(self, model_dir: Path, loops: int, kv_mode: str, capacity: int) -> None:
        started = time.time()
        imported = import_dory(model_dir, "float32", loops=loops, kv_mode=kv_mode)
        self.decoder = imported.model.decoder
        self.load_s = time.time() - started
        self.capacity = capacity
        self.config = DecoderForwardPassConfig.for_tracer_tests()
        self.keychain = Keychain.init(0, sharding_config=self.decoder.sharding_config)

    def prefill(self, ids: list[int], trace: bool):  # noqa: ANN201
        state = self.decoder.init_static_state(1, self.capacity, jnp.float32)
        return self.decoder(
            jnp.asarray([ids], dtype=jnp.int32),
            jnp.arange(len(ids), dtype=jnp.int32)[None, :],
            state=state,
            return_updated_state=True,
            return_activation_trace=trace,
            forward_pass_config=self.config,
            keychain=self.keychain,
        )

    def decode(self, state, token: int, position: int):  # noqa: ANN001, ANN201
        return self.decoder(
            jnp.asarray([[token]], dtype=jnp.int32),
            jnp.asarray([[position]], dtype=jnp.int32),
            state=state,
            return_updated_state=True,
            forward_pass_config=self.config,
            keychain=self.keychain,
        )


def compare_prefill(runner: Runner, prompt: dict, reference: np.lib.npyio.NpzFile, trace: bool) -> dict:
    started = time.time()
    result = runner.prefill(prompt["ids"], trace)
    logits = np.asarray(jax.device_get(result.logits[0]), dtype=np.float32)
    elapsed = time.time() - started
    reference_top1 = reference["prefill_top_idx"][:, 0]
    reference_top1_lp = reference["prefill_top_lp"][:, 0]
    lalamo_lp = log_softmax(logits)
    row = {
        "prefill_s": round(elapsed, 1),
        "last_logits_maxabs": float(np.abs(logits[-1] - reference["last_logits"]).max()),
        "prefill_top1_agree": float((logits.argmax(-1) == reference_top1).mean()),
        "prefill_top1_lp_maxabs": float(
            np.abs(lalamo_lp[np.arange(len(reference_top1)), reference_top1] - reference_top1_lp).max(),
        ),
        "first_token": int(logits[-1].argmax()),
        "reference_first_token": int(reference["gen_ids"][0]),
    }
    if trace:
        steps = sorted(key for key in reference.files if key.startswith("trace/x"))
        layer_results = result.activation_trace.layer_results
        assert len(layer_results) == len(steps), (len(layer_results), len(steps))
        errors = [
            rel_err(np.asarray(jax.device_get(layer.outputs[0]), dtype=np.float32), reference[key])
            for layer, key in zip(layer_results, steps, strict=True)
        ]
        row |= {
            "steps": len(steps),
            "step_relerr_max": max(errors),
            "step_relerr_last": errors[-1],
            "worst_step": steps[int(np.argmax(errors))].removeprefix("trace/"),
        }
    return row


def compare_decode(runner: Runner, prompt: dict, reference: np.lib.npyio.NpzFile) -> dict:
    """Teacher-force the reference greedy tokens and compare each decode step."""
    ids = list(prompt["ids"])
    generated = [int(token) for token in reference["gen_ids"]]
    result = runner.prefill(ids, trace=False)
    state = result.updated_state
    argmax_equal, lp_gaps, maxabs = 0, [], 0.0
    for step, token in enumerate(generated[:-1]):
        result = runner.decode(state, token, len(ids) + step)
        state = result.updated_state
        logits = np.asarray(jax.device_get(result.logits[0, 0]), dtype=np.float32)
        reference_index = reference["decode_top_idx"][step]
        reference_lp = reference["decode_top_lp"][step]
        lalamo_lp = log_softmax(logits)
        argmax_equal += int(logits.argmax() == reference["decode_logits_argmax"][step + 1])
        lp_gaps.append(float(np.abs(lalamo_lp[reference_index] - reference_lp).max()))
    return {
        "decode_steps": len(generated) - 1,
        "decode_argmax_equal": f"{argmax_equal}/{len(generated) - 1}",
        "decode_top20_lp_maxabs": max(lp_gaps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--ref-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--only", nargs="*", default=SHORT)
    ap.add_argument("--decode-loops", type=int, default=2)
    ap.add_argument("--long", action="store_true")
    args = ap.parse_args()
    prompts = {prompt["id"]: prompt for prompt in json.loads(args.prompts.read_text())}
    rows = []

    def emit(row: dict) -> None:
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(rows, indent=1))

    for loops in args.loops:
        runner = Runner(args.model, loops, "per_loop", capacity=64)
        for pid in args.only:
            reference = np.load(args.ref_dir / f"{pid}_L{loops}_per_loop_float32.npz")
            row = {"prompt": pid, "loops": loops, "kv_mode": "per_loop", "check": "prefill", "load_s": round(runner.load_s, 1)}
            emit(row | compare_prefill(runner, prompts[pid], reference, trace=True))
            if loops == args.decode_loops:
                emit({"prompt": pid, "loops": loops, "kv_mode": "per_loop", "check": "decode"} | compare_decode(runner, prompts[pid], reference))
        del runner
        gc.collect()

    runner = Runner(args.model, args.decode_loops, "last_loop", capacity=64)
    for pid in args.only:
        reference = np.load(args.ref_dir / f"{pid}_L{args.decode_loops}_last_loop_float32.npz")
        emit({"prompt": pid, "loops": args.decode_loops, "kv_mode": "last_loop", "check": "decode"} | compare_decode(runner, prompts[pid], reference))
    del runner
    gc.collect()

    if args.long:
        runner = Runner(args.model, args.decode_loops, "per_loop", capacity=2048)
        reference = np.load(args.ref_dir / f"long_L{args.decode_loops}_per_loop_float32_long.npz")
        emit({"prompt": "long", "loops": args.decode_loops, "kv_mode": "per_loop", "check": "prefill"} | compare_prefill(runner, prompts["long"], reference, trace=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
