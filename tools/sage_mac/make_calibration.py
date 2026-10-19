#!/usr/bin/env python3
"""Calibration sequences for Dory weight quantization on a Mac, kept apart from the evaluation suite.

Two sources, written as one JSON list of token-id sequences (`[{"source", "ids"}]`):
  - self-rollouts: the model's own replies (through `engine/target/release/cli server`, greedy, reasoning
    off) to prompts built here from task and topic templates; none comes from GSM8K, HumanEval, MMLU-Pro or
    the retrieval task
  - raw text: code and Markdown from local directories, cut into windows of `--window` tokens

    python3 tools/sage_mac/make_calibration.py --model <converted model> --tokenizer-dir <HF checkout> \
        --text-dir engine/crates --text-dir converter --out calibration.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "engine" / "target" / "release" / "cli"

TASKS = [
    "Explain {topic} to a first-year university student, with one concrete example.",
    "Write a short technical note on the common mistakes people make with {topic}.",
    "Compare two different approaches to {topic} and say when each one is better.",
    "Give a step-by-step plan for learning {topic} in four weeks.",
    "{topic}에 대해 핵심만 다섯 문장으로 설명해 주세요.",
    "Summarize the history of {topic} in one paragraph, then list three open problems.",
]
TOPICS = [
    "hash tables",
    "TCP congestion control",
    "photosynthesis",
    "compound interest",
    "the French Revolution",
    "garbage collection in managed runtimes",
    "linear regression",
    "plate tectonics",
    "vaccines",
    "public-key cryptography",
    "supply and demand",
    "sorting algorithms",
    "the water cycle",
    "neural networks",
    "git branching",
    "the Roman Empire",
    "probability distributions",
    "SQL joins",
    "climate models",
    "matrix multiplication",
]
CODE_TASKS = [
    "Write a Rust function that parses a semantic version string like '1.2.3-beta' into a struct, with tests.",
    "Write a Python script that counts word frequencies in a text file and prints the ten most common words.",
    "Implement an LRU cache in Python with get and put in O(1), and explain the design.",
    "Write a bash script that renames all .jpeg files in a directory to .jpg, skipping existing names.",
    "Write a TypeScript function that debounces another function, with a short usage example.",
    "Write a SQL query that finds the second highest salary per department, and explain it.",
    "Write a C function that reverses a singly linked list in place.",
    "Write a Python class for a 2D vector with addition, dot product and length, plus doctests.",
]
MATH_TASKS = [
    "A shop sells pencils in packs of {a} for ${b}. How much do {c} pencils cost if partial packs are not sold? "
    "Show your reasoning.",
    "A tank holds {a} liters and leaks {b} liters per hour while a pump adds {c} liters per hour. "
    "How long until it is full, starting empty? Explain each step.",
    "Find all integer solutions of x + y = {a} with x - y = {b}, or explain why there are none.",
]


def prompts(seed: int) -> list[str]:
    rng = random.Random(seed)
    items = [task.format(topic=topic) for task, topic in itertools.product(TASKS, TOPICS)]
    rng.shuffle(items)
    items = items[:80] + CODE_TASKS
    for template in MATH_TASKS:
        for _ in range(8):
            items.append(template.format(a=rng.randint(3, 40), b=rng.randint(2, 30), c=rng.randint(5, 60)))
    return items


def chat(port: int, model_id: str, content: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "reasoning_effort": "disabled",
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())["choices"][0]["message"].get("content") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="converted model for the cli server")
    ap.add_argument("--tokenizer-dir", required=True, type=Path, help="checkout with tokenizer.json and chat template")
    ap.add_argument("--text-dir", action="append", default=[], type=Path)
    ap.add_argument("--text-sequences", type=int, default=96)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.tokenizer_dir / "tokenizer.json"),
        chat_template=(args.tokenizer_dir / "chat_template.jinja").read_text(),
    )
    sequences = []

    # Raw text windows from code and Markdown files.
    files = sorted(
        path
        for directory in args.text_dir
        for path in directory.resolve().rglob("*")
        if path.suffix in {".rs", ".py", ".md", ".metal", ".h"} and path.is_file() and "target" not in path.parts
    )
    random.Random(args.seed).shuffle(files)
    for path in files:
        ids = tokenizer(path.read_text(errors="ignore"), add_special_tokens=False)["input_ids"]
        for start in range(0, len(ids) - args.window // 2, args.window):
            sequences.append({"source": f"text:{path.relative_to(ROOT)}", "ids": ids[start : start + args.window]})
        if len(sequences) >= args.text_sequences:
            break
    sequences = sequences[: args.text_sequences]

    # Self-rollouts through the engine server.
    server = subprocess.Popen(
        [str(CLI), "server", "--model", str(args.model), "--port", str(args.port), "--no-prefix-cache"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ),
    )
    try:
        model_id = None
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as response:
                    model_id = json.loads(response.read())["data"][0]["id"]
                break
            except OSError:
                time.sleep(1)
        for index, content in enumerate(prompts(args.seed)):
            reply = chat(args.port, model_id, content, args.max_tokens)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}, {"role": "assistant", "content": reply}],
                tokenize=False,
                enable_thinking=False,
            )
            sequences.append({"source": "rollout", "ids": tokenizer(text, add_special_tokens=False)["input_ids"]})
            if index % 20 == 0:
                print(f"rollout {index}: {len(sequences[-1]['ids'])} tokens", flush=True)
    finally:
        server.terminate()
        server.wait()
    args.out.write_text(json.dumps(sequences))
    tokens = sum(len(sequence["ids"]) for sequence in sequences)
    print(f"{len(sequences)} sequences, {tokens} tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
