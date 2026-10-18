#!/usr/bin/env python3
"""Run one Dory token through the NumPy reference adapter."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omega_dory import load_dory_gguf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--loops", type=int, default=2)
    args = ap.parse_args()
    started = time.time()
    model = load_dory_gguf(args.model, loops=args.loops)
    print(f"loaded in {time.time() - started:.1f}s, config={model.config}")
    started = time.time()
    logits = model.forward_token(args.token)
    elapsed = time.time() - started
    top = np.argsort(logits)[-5:][::-1]
    print(f"forward {elapsed:.2f}s, logits shape={logits.shape}, top={top.tolist()}")
    print("top values", [float(logits[i]) for i in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
