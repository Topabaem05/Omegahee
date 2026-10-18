#!/usr/bin/env python3
"""Convert or load the local Dory BF16 checkpoint with lalamo (dory-support branch).

Run with the lalamo environment:
    uv run --project LALAMO python lalamo_dory.py convert --model DIR --out OUT
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import lalamo  # noqa: F401  (configures JAX before it is imported)
import jax.numpy as jnp
from lalamo.model_import.common import import_model
from lalamo.model_import.model_specs.nvidia import NVIDIA_MODELS
from lalamo.model_import.origins import LocalOrigin
from lalamo.utils.sharding import ShardingConfig

MODEL_FILES = (
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def staged_model_dir(model_dir: Path, loops: int | None, kv_mode: str | None) -> Path:
    """A directory of symlinks to the checkpoint with config.json overrides for loops and KV mode."""
    config = json.loads((model_dir / "config.json").read_text())
    if loops is not None:
        config["n_recurrent_loops"] = loops
    if kv_mode is not None:
        config["recurrent_kv_cache_mode"] = kv_mode
    staged = Path(tempfile.mkdtemp(prefix="dory-lalamo-"))
    for name in MODEL_FILES:
        (staged / name).symlink_to((model_dir / name).resolve())
    (staged / "config.json").write_text(json.dumps(config))
    return staged


def import_dory(model_dir: Path, dtype: str, loops: int | None = None, kv_mode: str | None = None):  # noqa: ANN201
    root = staged_model_dir(model_dir, loops, kv_mode)
    (base_spec,) = NVIDIA_MODELS
    spec = replace(base_spec, origin=LocalOrigin(root=str(root), weight_files=("model.safetensors",)))
    return import_model(spec, sharding_config=ShardingConfig.replicated(), dtype=jnp.dtype(dtype))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["convert"])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    imported = import_dory(args.model, args.dtype)
    imported.model.save(args.out)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
