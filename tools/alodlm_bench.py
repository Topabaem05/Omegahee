#!/usr/bin/env python3
"""ALoDLM portable-decoder runner, including the unofficial MPS path.

Run with `/Users/topamini/models/alodlm/venv312/bin/python` because the
official package pins Python 3.12, PyTorch 2.8.0 and Transformers 4.56.1.
The official CLI only exposes cpu/cuda; this script calls the internal
`load_decoder` so MPS can be measured separately.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from alodlm.generate import decode_config, load_decoder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/Users/topamini/models/alodlm/ALoDLM-8B")
    ap.add_argument("--device", choices=("cpu", "mps"), required=True)
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument(
        "--prompt",
        default="Write a Python function that returns the greatest common divisor of two positive integers.",
    )
    args = ap.parse_args()

    model_args = SimpleNamespace(
        model=args.model,
        model_config=None,
        device=args.device,
        precision=args.precision,
        q=args.q,
        tau=args.tau,
        mode="entropy",
        window_size=args.window_size,
        max_new_tokens=args.max_new_tokens,
        position_penalty=None,
    )
    decoder = load_decoder(model_args)
    ids = decoder.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(json.dumps(decoder.generate(ids, decode_config(model_args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
