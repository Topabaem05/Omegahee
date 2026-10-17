"""Set a uzu checkpoint tree_budget without tripping its file guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def set_budget(model_dir: str, budget: int) -> dict:
    """Rewrite speculator/shapes.json in place, preserving its byte length.

    uzu deletes shapes.json and fails to load when the file length changes; it
    accepts a same-length rewrite and uses the new budget. Verified 2026-10-18
    against uzu 0.6.9 with Qwen3.5-9B-M.
    """
    path = os.path.join(model_dir, "speculator", "shapes.json")
    with open(path, "rb") as handle:
        raw = handle.read()
    original_len = len(raw)
    before = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    for entry in data.values():
        entry["tree_budget"] = int(budget)
    text = json.dumps(data, indent=2)
    if len(text) > original_len:
        raise ValueError(
            f"budget {budget} needs {len(text)} bytes, file is {original_len}; "
            "shrink the budget or the JSON formatting first")
    text = text + " " * (original_len - len(text))
    with open(path, "wb") as handle:
        handle.write(text.encode("utf-8"))
    with open(path, "rb") as handle:
        after = hashlib.sha256(handle.read()).hexdigest()
    return {
        "path": path,
        "budget": budget,
        "bytes": original_len,
        "sha256_before": before,
        "sha256_after": after,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="checkpoint directory containing speculator/")
    ap.add_argument("budget", type=int, help="target tree_budget")
    args = ap.parse_args(argv[1:])
    info = set_budget(args.model_dir, args.budget)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
