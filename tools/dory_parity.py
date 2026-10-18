#!/usr/bin/env python3
"""Compare Dory's chat prompt next-token prediction with a fixed prompt."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omega_dory import load_dory_gguf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--loops", type=int, default=2)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=False)
    template = Path(args.tokenizer_dir, "chat_template.jinja").read_text()
    tokenizer.chat_template = template
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    started = time.time()
    model = load_dory_gguf(args.model, loops=args.loops)
    print(f"loaded in {time.time() - started:.1f}s")
    started = time.time()
    logits = model.forward(ids)
    print(f"prompt tokens={len(ids)} ids={ids} forward={time.time() - started:.2f}s")
    top = int(np.argmax(logits))
    print("top token", top, repr(tokenizer.decode([top])))
    top5 = np.argsort(logits)[-5:][::-1]
    print("top5", [(int(i), tokenizer.decode([int(i)]), float(logits[i])) for i in top5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
