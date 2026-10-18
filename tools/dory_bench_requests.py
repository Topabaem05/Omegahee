#!/usr/bin/env python3
"""Build the fixed benchmark request set (research plan stage 1).

Prompt bodies are cut from open-source (MIT/Apache) text already in this repository, so no data is
downloaded. Each file holds one BenchRequest (scripts/benchmarks/engine-uzu/src/bench.rs) whose chat
prompt is about 128, 2,048 or 16,384 tokens long with the Dory chat template. The files and their
SHA-256 are written once and must not change afterwards (bench/requests/manifest.json).

    python3 tools/dory_bench_requests.py --tokenizer <model>/tokenizer.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "engine/README.md",
    *sorted(str(path.relative_to(ROOT)) for path in (ROOT / "engine/docs").rglob("*.mdx")),
    "converter/README.md",
    "engine/scripts/benchmarks/README.md",
    "reference/vllm_pr60149_5507695/vllm/model_executor/models/dory.py",
    "engine/LICENSE",
    "converter/LICENSE",
]
TARGETS = (128, 2048, 16384)
INSTRUCTION = "Read the following documentation and then summarize its main points.\n\n"
# <|im_start|>system\n<|im_end|>\n<|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n<think>\n
TEMPLATE_TOKENS = 17


def corpus() -> str:
    parts = [(ROOT / source).read_text(errors="replace") for source in SOURCES]
    text = "\n\n".join(parts)
    while len(text) < 400_000:
        text = text + "\n\n" + text
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "requests")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    body_ids = tokenizer.encode(corpus(), add_special_tokens=False).ids
    instruction_tokens = len(tokenizer.encode(INSTRUCTION, add_special_tokens=False).ids)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": SOURCES, "max_tokens": args.max_tokens, "files": {}}
    for target in TARGETS:
        body = tokenizer.decode(body_ids[: target - instruction_tokens - TEMPLATE_TOKENS])
        request = {
            "prompt_chat": [{"role": "user", "content": INSTRUCTION + body}],
            "max_tokens": args.max_tokens,
            "sampling": {"temp": 0.0},
            "num_runs": 1,
        }
        path = args.out / f"prompt_{target}.json"
        path.write_text(json.dumps(request, ensure_ascii=False) + "\n")
        prompt_tokens = len(tokenizer.encode(INSTRUCTION + body, add_special_tokens=False).ids) + TEMPLATE_TOKENS
        manifest["files"][path.name] = {
            "target_tokens": target,
            "approx_prompt_tokens": prompt_tokens,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
