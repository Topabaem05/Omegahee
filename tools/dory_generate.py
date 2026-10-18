#!/usr/bin/env python3
"""Greedy Dory generation through the full-recompute NumPy adapter."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omega_dory import load_dory_gguf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=False)
    tokenizer.chat_template = Path(args.tokenizer_dir, "chat_template.jinja").read_text()
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    started = time.time()
    model = load_dory_gguf(args.model, loops=args.loops)
    print(f"loaded in {time.time() - started:.1f}s")
    started = time.time()
    ids = model.generate(prompt_ids, args.max_new_tokens)
    new_ids = ids[len(prompt_ids):]
    print(f"generated in {time.time() - started:.2f}s ids={new_ids}")
    print(repr(tokenizer.decode(new_ids, skip_special_tokens=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
