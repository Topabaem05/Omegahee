#!/usr/bin/env python3
"""Convert a local Hugging Face checkout into an Ω-Hee (uzu) model directory.

`lalamo convert` downloads by repository name. This tool converts a checkout that is already on disk
with the same code path (`import_model` -> `Model.save`), so large weights are not downloaded twice and
the original checkout is never modified. Config overrides go into a staged copy of `config.json`.

Run inside the converter environment:

    cd converter && uv run python ../tools/omega_convert.py \
        --spec NVIDIA-Nemotron-Labs-Dory1.0-4B \
        --checkout ~/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
        --out ~/models/Dory1.0-4B-bf16 \
        --encoding '{"type": "hanashi", "name": "dory"}' \
        --override n_recurrent_loops=2

`--spec` is a spec name or repository from `lalamo list-models`. See docs/converter_guide.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import lalamo  # noqa: F401  (configures JAX before it is imported)
import jax
import jax.numpy as jnp
from lalamo.model_import.common import import_model
from lalamo.model_import.model_specs import ALL_MODEL_LISTS
from lalamo.model_import.origins import LocalOrigin
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import FullPrecisionMatrix, WeightMatrixSpec


def find_spec(name: str):  # noqa: ANN201
    specs = [spec for model_list in ALL_MODEL_LISTS for spec in model_list]
    matches = [spec for spec in specs if name in (spec.name, spec.origin.description)]
    if len(matches) != 1:
        known = ", ".join(sorted(spec.name for spec in specs if name.lower() in spec.name.lower())) or "none"
        raise SystemExit(f"Expected one spec named {name!r}, found {len(matches)}. Similar: {known}")
    (spec,) = matches
    return spec


def parse_override(text: str) -> tuple[str, object]:
    key, sep, value = text.partition("=")
    if not sep:
        raise SystemExit(f"Overrides look like key=value, got {text!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def stage_checkout(checkout: Path, overrides: dict[str, object]) -> Path:
    """Symlinks every file of the checkout and writes an overridden config.json next to them."""
    staged = Path(tempfile.mkdtemp(prefix="omega-convert-"))
    for path in checkout.iterdir():
        if path.is_file() and path.name != "config.json":
            (staged / path.name).symlink_to(path.resolve())
    config = json.loads((checkout / "config.json").read_text())
    config.update(overrides)
    (staged / "config.json").write_text(json.dumps(config, indent=2))
    return staged


def quantize(model, linear_spec: WeightMatrixSpec | None, readout_spec: WeightMatrixSpec | None):  # noqa: ANN001, ANN201
    """Re-encodes the full-precision transformer linears and the readout matrix. Input embeddings and
    per-channel vectors stay in full precision."""

    def convert(path: tuple, leaf: object) -> object:
        if not isinstance(leaf, FullPrecisionMatrix):
            return leaf
        name = jax.tree_util.keystr(path)
        if ".transformer." in name:
            spec = linear_spec
        elif "output_embedding" in name:
            spec = readout_spec
        else:
            spec = None
        if spec is None:
            return leaf
        return spec.compress(leaf.decompress(), sharding_config=leaf.sharding_config, is_sharded=leaf.is_sharded)

    return jax.tree_util.tree_map_with_path(convert, model, is_leaf=lambda leaf: isinstance(leaf, FullPrecisionMatrix))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="lalamo model spec name or repository")
    ap.add_argument("--checkout", required=True, type=Path, help="local Hugging Face checkout")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--override", action="append", default=[], help="config.json key=value (JSON value)")
    ap.add_argument("--encoding", help="JSON for the engine's encoding.json (chat encoding), e.g. hanashi preset")
    ap.add_argument("--linear-spec", help='quantize transformer linears, e.g. {"type": "MLXSpec", "bits": 4, "group_size": 64}')
    ap.add_argument("--readout-spec", help='quantize the readout matrix, e.g. {"type": "MLXSpec", "bits": 8, "group_size": 64}')
    args = ap.parse_args()

    checkout = args.checkout.expanduser().resolve()
    spec = find_spec(args.spec)
    staged = stage_checkout(checkout, dict(parse_override(text) for text in args.override))
    weight_files = tuple(sorted(path.name for path in staged.glob(f"*{spec.origin.weight_format.value}")))
    if not weight_files:
        raise SystemExit(f"No {spec.origin.weight_format.value} weights in {checkout}")
    local_spec = replace(spec, origin=LocalOrigin(root=str(staged), weight_files=weight_files))

    imported = import_model(local_spec, sharding_config=ShardingConfig.replicated(), dtype=jnp.dtype(args.dtype))
    model = imported.model
    if args.linear_spec or args.readout_spec:
        linear_spec = WeightMatrixSpec.from_json(json.loads(args.linear_spec)) if args.linear_spec else None
        readout_spec = WeightMatrixSpec.from_json(json.loads(args.readout_spec)) if args.readout_spec else None
        model = quantize(model, linear_spec, readout_spec)
    out = args.out.expanduser()
    model.save(out)
    if args.encoding:
        (out / "encoding.json").write_text(json.dumps(json.loads(args.encoding)) + "\n")
    manifest = {
        "spec": spec.name,
        "checkout": str(checkout),
        "overrides": args.override,
        "dtype": args.dtype,
        "linear_spec": json.loads(args.linear_spec) if args.linear_spec else None,
        "readout_spec": json.loads(args.readout_spec) if args.readout_spec else None,
        "weights_sha256": {name: sha256(checkout / name) for name in weight_files},
        "output_sha256": sha256(out / "model.safetensors"),
    }
    (out / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
