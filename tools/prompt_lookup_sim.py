#!/usr/bin/env python3
"""Offline estimate of prompt-lookup drafts on greedy model outputs (tokens confirmed per forward pass).

A draft is the k tokens that followed the most recent earlier occurrence of the context's last n-gram (n from
`--max-n` down to `--min-n`), where the context is the prompt plus the output so far. Two schemes are counted:
  - sequential verification: one 2-loop pass checks the draft and confirms 1 + the matching draft prefix
  - loop wavefront with drafts: as above, but a rejected draft or a wrong 1-loop guess (agreement
    `--exit-agree`) costs one restart pass, because the next pass's rows were encoded speculatively
Outputs come from `tools/omega_eval.py` runs (`bench/eval_runs/<label>`) and benchmarks-uzu responses.

    python3 tools/prompt_lookup_sim.py --tokenizer <model>/tokenizer.json --eval-run bench/eval_runs/nvfp4-r8-e12 \
        --bench-output 2048=<benchmarks-uzu response json>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omega_eval import gsm8k_prompt, humaneval_prompt, load_items  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def draft(context: list[int], k: int, min_n: int, max_n: int) -> list[int]:
    for n in range(min(max_n, len(context) - 1), min_n - 1, -1):
        key = context[-n:]
        for start in range(len(context) - n - 1, -1, -1):
            if context[start : start + n] == key:
                follow = context[start + n : start + n + k]
                if follow:
                    return follow
    return []


def matched(prompt: list[int], reply: list[int], position: int, proposal: list[int]) -> int:
    count = 0
    for offset, token in enumerate(proposal):
        if position + 1 + offset < len(reply) and reply[position + 1 + offset] == token:
            count += 1
        else:
            break
    return count


def sequential(prompt: list[int], reply: list[int], k: int, min_n: int, max_n: int) -> tuple[int, int]:
    position = passes = 0
    while position < len(reply):
        passes += 1
        proposal = draft(prompt + reply[: position + 1], k, min_n, max_n) if k else []
        position += 1 + matched(prompt, reply, position, proposal)
    return len(reply), passes


def wavefront(
    prompt: list[int], reply: list[int], k: int, min_n: int, max_n: int, exit_agree: float, rng: random.Random
) -> tuple[int, int]:
    position = passes = 0
    while position < len(reply):
        passes += 1
        proposal = draft(prompt + reply[: position + 1], k, min_n, max_n) if k else []
        extra = matched(prompt, reply, position, proposal)
        position += 1 + extra
        if not (extra == len(proposal) and rng.random() < exit_agree):
            passes += 1
    return len(reply), passes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--eval-run", type=Path, help="omega_eval.py run directory with gsm8k.jsonl / humaneval.jsonl")
    ap.add_argument(
        "--bench-output",
        action="append",
        default=[],
        help="SIZE=PATH: benchmarks-uzu response JSON for bench/requests/prompt_SIZE.json",
    )
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--exit-agree", type=float, default=0.96)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    sets = []
    for spec in args.bench_output:
        size, path = spec.split("=", 1)
        request = json.loads((ROOT / f"bench/requests/prompt_{size}.json").read_text())
        reply = json.loads(Path(path).read_text())[-1]["text"]
        sets.append(
            (
                f"bench{size}",
                [(tokenizer.encode(request["prompt_chat"][0]["content"]).ids, tokenizer.encode(reply).ids)],
            )
        )
    if args.eval_run:
        for task, prompt_fn in (("gsm8k", gsm8k_prompt), ("humaneval", humaneval_prompt)):
            items = {item["id"]: item for item in load_items(task)}
            rows = [json.loads(line) for line in (args.eval_run / f"{task}.jsonl").read_text().splitlines()]
            sets.append(
                (
                    task,
                    [
                        (tokenizer.encode(prompt_fn(items[row["id"]])[0]).ids, tokenizer.encode(row["reply"]).ids)
                        for row in rows
                    ],
                )
            )
    for name, pairs in sets:
        cells = []
        for k in (2, 3):
            tokens = passes = 0
            for prompt, reply in pairs:
                t, p = sequential(prompt, reply, k, 1, args.max_n)
                tokens, passes = tokens + t, passes + p
            cells.append(f"sequential k={k} {tokens / passes:.3f}")
        for k, min_n in ((0, 1), (1, 3), (4, 3)):
            rng = random.Random(0)
            tokens = passes = 0
            for prompt, reply in pairs:
                t, p = wavefront(prompt, reply, k, min_n, args.max_n, args.exit_agree, rng)
                tokens, passes = tokens + t, passes + p
            cells.append(f"wavefront k={k} n>={min_n} {tokens / passes:.3f}")
        print(f"{name:10s} " + " | ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
