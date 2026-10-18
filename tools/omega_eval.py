#!/usr/bin/env python3
"""Ω-Hee quality evaluation suite (research plan stage 1, item 5).

Four tasks, scored through the engine's OpenAI-compatible server (`cli server`), greedy, reasoning disabled:
  - gsm8k: 200 test questions, final-answer exact match
  - humaneval: 164 problems, pass@1 (tests run under sandbox-exec: no network, writes only to a temp dir)
  - mmlu_pro: 500 test questions (seeded sample), letter match
  - long: 20 retrieval questions with one fact hidden in about 16K tokens of documentation
Suite score = mean of the four task scores; compare a variant against the BF16 baseline score.

The item sets are built once (`--build-items`) into bench/eval/ with SHA-256 in manifest.json and are
never changed afterwards.

    python3 tools/omega_eval.py --build-items --datasets ~/code/uzu-omega/datasets --tokenizer <tok.json>
    python3 tools/omega_eval.py --model ~/models/Dory1.0-4B-lalamo-bf16 --label bf16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "engine" / "target" / "release" / "cli"
EVAL_DIR = ROOT / "bench" / "eval"
TASKS = ("gsm8k", "humaneval", "mmlu_pro", "long")
LETTERS = "ABCDEFGHIJ"
SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (subpath "/private/var/folders") (literal "/dev/null"))
"""


# ---------------------------------------------------------------- item sets


def build_items(datasets: Path, tokenizer_path: Path) -> None:
    import pandas as pd
    from tokenizers import Tokenizer

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    gsm = pd.read_parquet(next((datasets / "gsm8k").rglob("test-*.parquet")))
    rows = gsm.to_dict("records")
    rng.shuffle(rows)
    gsm_items = [
        {"id": f"gsm8k-{i}", "question": row["question"], "answer": row["answer"].split("####")[-1].strip().replace(",", "")}
        for i, row in enumerate(rows[:200])
    ]

    humaneval = pd.read_parquet(next((datasets / "humaneval").rglob("test-*.parquet")))
    he_items = [
        {"id": row["task_id"], "prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]}
        for row in humaneval.to_dict("records")
    ]

    mmlu = pd.read_parquet(next((datasets / "mmlu_pro").rglob("test-*.parquet")))
    rows = mmlu.to_dict("records")
    rng.shuffle(rows)
    mmlu_items = [
        {
            "id": f"mmlu_pro-{row['question_id']}",
            "category": row["category"],
            "question": row["question"],
            "options": list(row["options"]),
            "answer": row["answer"],
        }
        for row in rows[:500]
    ]

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    haystack_sources = sorted((ROOT / "engine" / "docs").rglob("*.mdx")) + [ROOT / "engine/README.md", ROOT / "converter/README.md"]
    haystack = "\n\n".join(path.read_text(errors="replace") for path in haystack_sources)
    ids = tokenizer.encode(haystack, add_special_tokens=False).ids
    while len(ids) < 16000:
        ids = ids + ids
    chunk = ids[:15800]
    long_items = []
    for i in range(20):
        name = f"project-{rng.choice(['amber', 'cobalt', 'delta', 'ember', 'falcon', 'garnet', 'harbor', 'indigo'])}-{i}"
        code = rng.randrange(10000, 99999)
        position = int(len(chunk) * (i + 0.5) / 20)
        text = (
            tokenizer.decode(chunk[:position])
            + f"\n\nImportant: the access code for {name} is {code}.\n\n"
            + tokenizer.decode(chunk[position:])
        )
        long_items.append({"id": f"long-{i}", "context": text, "name": name, "answer": str(code)})

    manifest = {}
    for task, items in zip(TASKS, (gsm_items, he_items, mmlu_items, long_items), strict=True):
        path = EVAL_DIR / f"{task}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
        manifest[path.name] = {"items": len(items), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (EVAL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_items(task: str) -> list[dict]:
    return [json.loads(line) for line in (EVAL_DIR / f"{task}.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- prompts and scoring


def gsm8k_prompt(item: dict) -> tuple[str, int]:
    return (
        "Solve the following math problem step by step. At the end, write the final answer on its own line as "
        f"'#### <number>'.\n\n{item['question']}",
        768,
    )


def gsm8k_score(item: dict, reply: str) -> bool:
    match = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", reply)
    numbers = match or re.findall(r"-?[\d,]*\.?\d+", reply)
    if not numbers:
        return False
    try:
        return abs(float(numbers[-1].replace(",", "")) - float(item["answer"])) < 1e-6
    except ValueError:
        return False


def humaneval_prompt(item: dict) -> tuple[str, int]:
    return (
        "Complete the following Python function. Reply with the complete function (including the signature) in one "
        f"```python code block and nothing else.\n\n```python\n{item['prompt']}```",
        768,
    )


def humaneval_score(item: dict, reply: str) -> bool:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    code = blocks[0] if blocks else reply
    if f"def {item['entry_point']}" not in code:
        code = item["prompt"] + code
    program = f"{item['prompt']}\n{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix="omega-he-") as tmp:
        script = Path(tmp) / "solution.py"
        script.write_text(program)
        profile = Path(tmp) / "sandbox.sb"
        profile.write_text(SANDBOX_PROFILE.format(tmp=Path(tmp).resolve()))
        try:
            result = subprocess.run(
                ["sandbox-exec", "-f", str(profile), sys.executable, "-I", str(script)],
                cwd=tmp,
                capture_output=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0


def mmlu_prompt(item: dict) -> tuple[str, int]:
    options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(item["options"]))
    return (
        f"The following is a multiple choice question about {item['category']}. Answer with only the letter of the "
        f"correct option.\n\n{item['question']}\n\n{options}\n\nAnswer:",
        8,
    )


def mmlu_score(item: dict, reply: str) -> bool:
    match = re.search(r"\b([A-J])\b", reply.strip())
    return bool(match) and match.group(1) == item["answer"]


def long_prompt(item: dict) -> tuple[str, int]:
    return (f"{item['context']}\n\nWhat is the access code for {item['name']}? Reply with the number only.", 16)


def long_score(item: dict, reply: str) -> bool:
    return item["answer"] in reply


TASK_FUNCTIONS = {
    "gsm8k": (gsm8k_prompt, gsm8k_score),
    "humaneval": (humaneval_prompt, humaneval_score),
    "mmlu_pro": (mmlu_prompt, mmlu_score),
    "long": (long_prompt, long_score),
}


# ---------------------------------------------------------------- server


class Server:
    def __init__(self, model: Path, port: int) -> None:
        self.port = port
        self.log = tempfile.NamedTemporaryFile(prefix="omega-eval-server-", suffix=".log", delete=False)
        self.process = subprocess.Popen(
            [str(CLI), "server", "--model", str(model), "--port", str(port), "--no-prefix-cache"],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{port}/v1", timeout=1800)
        for _ in range(600):
            try:
                self.model_id = self.client.get("/models").json()["data"][0]["id"]
                return
            except httpx.HTTPError:
                time.sleep(1)
        raise RuntimeError(f"server did not start, see {self.log.name}")

    def chat(self, content: str, max_tokens: int) -> tuple[str, int]:
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "reasoning_effort": "disabled",
            },
        )
        response.raise_for_status()
        body = response.json()
        if "choices" not in body:
            raise RuntimeError(f"server error: {json.dumps(body)[:500]}")
        return body["choices"][0]["message"].get("content") or "", body["usage"]["completion_tokens"]

    def close(self) -> None:
        self.process.terminate()
        self.process.wait()


def evaluate(model: Path, label: str, tasks: list[str], port: int, limit: int | None) -> dict:
    server = Server(model, port)
    out_dir = ROOT / "bench" / "eval_runs" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    try:
        for task in tasks:
            prompt_fn, score_fn = TASK_FUNCTIONS[task]
            items = load_items(task)[:limit]
            started = time.time()
            correct, tokens, records = 0, 0, []
            for item in items:
                content, max_tokens = prompt_fn(item)
                reply, completion_tokens = server.chat(content, max_tokens)
                ok = score_fn(item, reply)
                correct += ok
                tokens += completion_tokens
                records.append({"id": item["id"], "correct": ok, "reply": reply})
            (out_dir / f"{task}.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
            scores[task] = {
                "score": correct / len(items),
                "correct": correct,
                "items": len(items),
                "completion_tokens": tokens,
                "seconds": round(time.time() - started, 1),
            }
            print(f"{label} {task}: {correct}/{len(items)} = {correct / len(items):.1%}", flush=True)
    finally:
        server.close()
    suite = sum(result["score"] for result in scores.values()) / len(scores)
    result = {"label": label, "model": str(model), "tasks": scores, "suite_score": suite, "limit": limit}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "bench" / "results" / "eval_runs.jsonl").open("a") as file:
        file.write(json.dumps(result) + "\n")
    print(f"{label} suite score {suite:.2%}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-items", action="store_true")
    ap.add_argument("--datasets", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--label")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--limit", type=int, help="first N items per task (smoke runs only)")
    args = ap.parse_args()
    if args.build_items:
        build_items(args.datasets.expanduser(), args.tokenizer.expanduser())
        return 0
    evaluate(args.model.expanduser().resolve(), args.label, args.tasks, args.port, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
