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

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dory_torch import ATTENTION_LINEARS, MLP_LINEARS, DoryTorch, justnorm  # noqa: E402
from quantize_init import INPUT_KIND  # noqa: E402


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


def train_layer(
    model: DoryTorch,
    layer: int,
    pairs: list[tuple[torch.Tensor, int, torch.Tensor]],
    energy: dict,
    sage_class,
    args,
) -> tuple[dict[str, torch.Tensor], float, float]:
    """Trains the SAGE parameters of `layer` on (input, loop, target) rows; returns QDQ weights and losses."""
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
    return weights, first, last


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
    args = ap.parse_args()
    sys.path.insert(0, str(args.paroquant))
    from paroquant.optim.sagelinear import SAGEPseudoQuantizedLinear

    args.out.mkdir(parents=True, exist_ok=True)
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
        if loop == 1 and kind in ("qkvg", "gate_up"):
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
    qdq: dict[str, torch.Tensor] = {}
    log = []

    def run_layer_all(layer: int, states, loop: int, weights=None):
        runner = LayerRunner(model, layer)
        return [runner(h, loop, weights or {}).to(model.dtype) for h in states]

    def train_and_propagate(layer: int, loop: int, extra_pairs=()):
        nonlocal teacher, student
        started = time.time()
        with torch.no_grad():
            targets = run_layer_all(layer, teacher, loop)
        pairs = [(s, loop, t) for s, t in zip(student, targets)] + list(extra_pairs)
        weights, first, last = train_layer(model, layer, pairs, energy, SAGEPseudoQuantizedLinear, args)
        for name, weight in weights.items():
            qdq[model.linear_name(layer, name)] = weight.to(torch.bfloat16).cpu().contiguous()
        with torch.no_grad():
            student = run_layer_all(layer, student, loop, weights)
            teacher = targets
        log.append({"layer": layer, "loss_before": first, "loss_after": last, "seconds": time.time() - started})
        print(f"layer {layer}: loss {first:.5f} -> {last:.5f} ({time.time() - started:.0f}s)", flush=True)

    for layer in range(n_in):
        train_and_propagate(layer, 0)
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
            student_weights = {
                name: qdq[model.linear_name(layer, name)].to(model.device).float()
                for name in linear_names(model, layer)
            }
            teacher = run_layer_all(layer, teacher, 1)
            student = run_layer_all(layer, student, 1, student_weights)
        teacher = [justnorm(h) for h in teacher]
        student = [justnorm(h) for h in student]
    for layer in range(n_in + n_rec, n_layers):
        train_and_propagate(layer, 0)

    save_file(qdq, str(args.out / "qdq_weights.safetensors"))
    (args.out / "training.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "layers": log}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
