#!/usr/bin/env python3
"""Block-wise reconstruction of SAGE-quantized Dory layers on a Mac (MPS), after `quantize_init.py`.

Every nGPT sublayer (attention or MLP mixer with its residual slerp) is trained in execution order. Its input
is the quantized model's hidden state so far (the student), its target the BF16 model's output at the same
point (the teacher), the loss the mean squared difference of the (unit) output rows. The trained parameters
are SAGE's per-vector radius logits and direction codebook (straight-through over the FP16 radius grid);
signs and direction indices stay fixed.

Dory runs the recurrent layers 16..55 twice. Each of them is trained once, at its first-loop position, on
first-loop student inputs and on second-loop inputs taken from the teacher (stored by a teacher pass before
training), both against teacher targets. The second loop then only propagates.

Each layer is written when it is done: `<out>/layers/layer_NN.safetensors` (QDQ BF16 weights under Hugging Face
names; `patch_qdq.py` reads the directory) and, with `--save-state`, `<out>/state/layer_NN.safetensors` (signs
packed eight per byte in row-major order, direction indices, radius logits, codebook logits, radius bits and
the channel unpermutation, for packing into a kernel format later). `--resume` loads finished layers instead
of training them again. `--input-layers-only` stops after layer 15 (for quick training-setting comparisons).

    python3 tools/sage_mac/blockwise.py --model <BF16 checkout> --calibration calibration.json \
        --energy <quantize_init out>/energy.pt --paroquant <paroquant checkout> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dory_torch import ATTENTION_LINEARS, MLP_LINEARS, DoryTorch, justnorm  # noqa: E402
from quantize_init import INPUT_KIND, load_sage  # noqa: E402


class LayerRunner:
    """Runs one layer of `model` in FP32 with the given linear weights (which may require gradients)."""

    def __init__(self, model: DoryTorch, layer: int):
        self.model, self.layer = model, layer
        prefix = f"backbone.layers.{layer}."
        self.saved = {name: tensor for name, tensor in model.w.items() if name.startswith(prefix)}

    def __call__(self, h: torch.Tensor, loop: int, weights: dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.model
        for name, tensor in self.saved.items():
            model.w[name] = tensor.float()
        for name, weight in weights.items():
            model.w[model.linear_name(self.layer, name)] = weight
        positions = torch.arange(h.shape[0], device=model.device)
        try:
            return model.layer(self.layer, h.float(), positions, loop)
        finally:
            model.w.update(self.saved)


def linear_names(model: DoryTorch, layer: int) -> tuple[str, ...]:
    return MLP_LINEARS if model.layer_types[layer] == "mlp" else ATTENTION_LINEARS


def sage_state(module) -> dict[str, torch.Tensor]:
    """The trained spherical state of one matrix, with the int8 signs packed to bits (row-major order)."""
    signs = module.signs.detach().cpu()
    return {
        "signs_packed": torch.from_numpy(np.packbits(signs.numpy().reshape(-1) > 0)),
        "direction_indices": module.direction_indices.detach().cpu(),
        "radius_logits": module.radius_logits.detach().float().cpu(),
        "angle_logits": module.angle_logits.detach().float().cpu(),
        "radius_bits": module.radius_bits.detach().cpu(),
        "channel_unpermutation": module.channel_unpermutation.detach().to(torch.int32).cpu(),
    }


def train_layer(
    model: DoryTorch,
    layer: int,
    pairs: list[tuple[torch.Tensor, int, torch.Tensor]],
    energy: dict,
    sage_class,
    args,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float, float]:
    """Trains the SAGE parameters of `layer` on (input, loop, target) rows; returns QDQ weights, state, losses."""
    runner = LayerRunner(model, layer)
    modules = {}
    for name in linear_names(model, layer):
        weight = model.w[model.linear_name(layer, name)].float()
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=model.device)
        with torch.no_grad():
            linear.weight.copy_(weight)
        modules[name] = sage_class(linear, channel_energy=energy[(layer, INPUT_KIND[name])].to(model.device))
    parameters = [p for module in modules.values() for p in (module.radius_logits, module.angle_logits)]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    def loss_on(batch, weights) -> torch.Tensor:
        total = 0.0
        for h, loop, target in batch:
            output = runner(h, loop, weights)
            total = total + (output - target.float()).square().sum(-1).mean()
        return total / len(batch)

    with torch.no_grad():
        first = float(loss_on(pairs[: args.batch], {n: m.materialized_weight() for n, m in modules.items()}))
    order = list(range(len(pairs)))
    rng = random.Random(layer)
    for _ in range(args.epochs):
        rng.shuffle(order)
        for start in range(0, len(order), args.batch):
            batch = [pairs[index] for index in order[start : start + args.batch]]
            optimizer.zero_grad(set_to_none=True)
            weights = {name: module.materialized_weight() for name, module in modules.items()}
            loss = loss_on(batch, weights)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        weights = {name: module.materialized_weight().detach() for name, module in modules.items()}
        last = float(loss_on(pairs[: args.batch], weights))
    state = {}
    for name, module in modules.items():
        for key, value in sage_state(module).items():
            state[f"{model.linear_name(layer, name)}.{key}"] = value.contiguous()
    return weights, state, first, last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--energy", required=True, type=Path)
    ap.add_argument("--paroquant", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sequences", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save-state", action="store_true", help="also write each layer's spherical state")
    ap.add_argument("--resume", action="store_true", help="load layers already written instead of training them")
    ap.add_argument("--index-radius-bits", type=int, default=10, help="as in quantize_init.py")
    ap.add_argument("--input-layers-only", action="store_true", help="train the input layers 0..15 and stop")
    args = ap.parse_args()
    SAGEPseudoQuantizedLinear = load_sage(args.paroquant, args.index_radius_bits)

    (args.out / "layers").mkdir(parents=True, exist_ok=True)
    if args.save_state:
        (args.out / "state").mkdir(exist_ok=True)
    model = DoryTorch(args.model, device=args.device, dtype=torch.bfloat16)
    energy = torch.load(args.energy)
    entries = json.loads(args.calibration.read_text())
    random.Random(0).shuffle(entries)
    sequences = [
        torch.tensor(entry["ids"][: args.max_tokens], device=model.device) for entry in entries[: args.sequences]
    ]
    n_in, n_rec, n_layers, loops = model.n_in, model.n_rec, model.n_layers, model.loops
    assert loops == 2, "the recurrent training below assumes two loops"
    recurrent = range(n_in, n_in + n_rec)

    # Teacher pass: second-loop inputs of the recurrent layers.
    started = time.time()
    second_loop_inputs: dict[int, list[torch.Tensor]] = {layer: [] for layer in recurrent}

    def store(layer: int, kind: str, loop: int, x: torch.Tensor) -> None:
        if loop == 1 and kind in ("qkvg", "gate_up") and not args.input_layers_only:
            second_loop_inputs[layer].append(x.detach().clone())

    model.observer = store
    for ids in sequences:
        model.forward(ids)
    model.observer = None
    print(f"teacher pass: {time.time() - started:.0f}s", flush=True)

    def embed(ids):
        return model.w["backbone.embeddings.weight"][ids]

    with torch.no_grad():
        teacher = [embed(ids) for ids in sequences]
        student = [h.clone() for h in teacher]
    log_path = args.out / "training.json"
    log = json.loads(log_path.read_text())["layers"] if args.resume and log_path.exists() else []

    def shard(layer: int) -> Path:
        return args.out / "layers" / f"layer_{layer:02d}.safetensors"

    def load_layer(layer: int) -> dict[str, torch.Tensor]:
        weights = load_file(str(shard(layer)))
        return {
            name: weights[model.linear_name(layer, name)].to(model.device).float()
            for name in linear_names(model, layer)
        }

    def run_layer_all(layer: int, states, loop: int, weights=None):
        runner = LayerRunner(model, layer)
        return [runner(h, loop, weights or {}).to(model.dtype) for h in states]

    def train_and_propagate(layer: int, loop: int, extra_pairs=()):
        nonlocal teacher, student
        started = time.time()
        with torch.no_grad():
            targets = run_layer_all(layer, teacher, loop)
        if args.resume and shard(layer).exists():
            weights = load_layer(layer)
            print(f"layer {layer}: loaded", flush=True)
        else:
            pairs = [(s, loop, t) for s, t in zip(student, targets)] + list(extra_pairs)
            weights, state, first, last = train_layer(model, layer, pairs, energy, SAGEPseudoQuantizedLinear, args)
            qdq = {model.linear_name(layer, n): w.to(torch.bfloat16).cpu().contiguous() for n, w in weights.items()}
            save_file(qdq, str(shard(layer)))
            weights = {n: qdq[model.linear_name(layer, n)].to(model.device).float() for n in weights}
            if args.save_state:
                save_file(state, str(args.out / "state" / f"layer_{layer:02d}.safetensors"))
            log.append({"layer": layer, "loss_before": first, "loss_after": last, "seconds": time.time() - started})
            log_path.write_text(
                json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "layers": log}, indent=1)
            )
            print(f"layer {layer}: loss {first:.5f} -> {last:.5f} ({time.time() - started:.0f}s)", flush=True)
        with torch.no_grad():
            student = run_layer_all(layer, student, loop, weights)
            teacher = targets

    for layer in range(n_in):
        train_and_propagate(layer, 0)
    if args.input_layers_only:
        return 0
    teacher_inputs, student_inputs = teacher, student
    with torch.no_grad():
        teacher = [justnorm(h + h) for h in teacher_inputs]
        student = [justnorm(h + h) for h in student_inputs]
    for layer in recurrent:
        with torch.no_grad():
            runner = LayerRunner(model, layer)
            extra = [(x, 1, runner(x, 1, {}).to(model.dtype)) for x in second_loop_inputs[layer]]
        train_and_propagate(layer, 0, extra)
        second_loop_inputs[layer] = []
    with torch.no_grad():
        teacher_state = [justnorm(h) for h in teacher]
        student_state = [justnorm(h) for h in student]
        teacher = [justnorm(s + i) for s, i in zip(teacher_state, teacher_inputs)]
        student = [justnorm(s + i) for s, i in zip(student_state, student_inputs)]
        for layer in recurrent:
            teacher = run_layer_all(layer, teacher, 1)
            student = run_layer_all(layer, student, 1, load_layer(layer))
        teacher = [justnorm(h) for h in teacher]
        student = [justnorm(h) for h in student]
    for layer in range(n_in + n_rec, n_layers):
        train_and_propagate(layer, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
