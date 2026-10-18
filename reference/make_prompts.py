#!/usr/bin/env python3
"""Build the fixed Dory prompt set (token ids) used by every parity stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

PROMPTS = [
    ("hello", "Hello", False),
    ("math", "What is 2+2?", False),
    ("gcd", "Write a Python function that returns the greatest common divisor of two integers.", False),
    ("ko", "대한민국의 수도는 어디인가요? 한 문장으로 답하세요.", False),
    ("think", "What is 17 * 23?", True),
]


def long_prompt() -> str:
    animals = ["cat", "dog", "owl", "fox", "eel", "yak", "ant", "bee"]
    lines = [f"Line {i}: the {animals[i % 8]} has code {i * 7919 % 10007}." for i in range(1, 121)]
    return "\n".join(lines) + "\n\nWhat is the code on line 3? Answer with the number only."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.model, "tokenizer.json")),
        chat_template=Path(args.model, "chat_template.jinja").read_text(),
    )
    rows = []
    for pid, text, think in PROMPTS + [("long", long_prompt(), False)]:
        s = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )
        ids = tok.encode(s, add_special_tokens=False)
        rows.append({"id": pid, "text": text, "enable_thinking": think, "template": s, "ids": ids})
        print(pid, len(ids), repr(s[:80]), "...", repr(s[-60:]))
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
