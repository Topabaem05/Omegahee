#!/usr/bin/env python3
"""SAGE initialization (no block-wise training) of every Dory linear on a Mac.

1. Runs the calibration sequences through `DoryTorch` (MPS) and accumulates each linear input's per-channel
   energy, mean(x^2) over all tokens (both loops for the recurrent layers).
2. Builds SAGE's spherical representation per matrix with energy-stratified grouping and the adaptive
   direction/radius bit split (`SAGEPseudoQuantizedLinear` from a local paroquant checkout with the SAGE
   patches; that code is GPL-3.0 and stays outside this repository).
3. Writes the dense quantized-and-dequantized (QDQ) BF16 weights under their Hugging Face names, the chosen
   bit splits and errors, and optionally the spherical state of every matrix for later training or packing.

    python3 tools/sage_mac/quantize_init.py --model <BF16 checkout> --calibration calibration.json \
        --paroquant <paroquant checkout> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dory_torch import DoryTorch  # noqa: E402

INPUT_KIND = {
    "q_proj": "qkvg",
    "k_proj": "qkvg",
    "v_proj": "qkvg",
    "g_proj": "qkvg",
    "o_proj": "o_proj",
    "gate_proj": "gate_up",
    "up_proj": "gate_up",
    "down_proj": "down_proj",
}


def load_sage(paroquant: Path, index_radius_bits: int):
    """`SAGEPseudoQuantizedLinear` from a local paroquant checkout, with the given index plus radius bits."""
    sys.path.insert(0, str(paroquant))
    from paroquant.optim import sagelinear, sphlinear

    sphlinear.INDEX_AND_RADIUS_BITS = sagelinear.INDEX_AND_RADIUS_BITS = index_radius_bits
    return sagelinear.SAGEPseudoQuantizedLinear


def collect_energy(
    model: DoryTorch, sequences: list[list[int]], max_tokens: int
) -> dict[tuple[int, str], torch.Tensor]:
    sums: dict[tuple[int, str], torch.Tensor] = {}
    counts: dict[tuple[int, str], int] = {}

    def observe(layer: int, kind: str, _loop: int, x: torch.Tensor) -> None:
        key = (layer, kind)
        squares = x.float().square().sum(dim=0)
        sums[key] = squares if key not in sums else sums[key] + squares
        counts[key] = counts.get(key, 0) + x.shape[0]

    model.observer = observe
    for index, ids in enumerate(sequences):
        model.forward(torch.tensor(ids[:max_tokens]))
        if index % 32 == 0:
            print(f"energy: sequence {index}/{len(sequences)}", flush=True)
    model.observer = None
    return {key: (value / counts[key]).cpu() for key, value in sums.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="Hugging Face BF16 checkout")
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--paroquant", required=True, type=Path, help="paroquant checkout with the SAGE patches applied")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save-state", action="store_true", help="also save each matrix's spherical state")
    ap.add_argument("--index-radius-bits", type=int, default=10)
    ap.add_argument("--energy", type=Path, help="energy.pt from an earlier run (default: <out>/energy.pt)")
    args = ap.parse_args()
    SAGEPseudoQuantizedLinear = load_sage(args.paroquant, args.index_radius_bits)

    args.out.mkdir(parents=True, exist_ok=True)
    model = DoryTorch(args.model, device=args.device, dtype=torch.bfloat16)
    energy_path = args.energy or args.out / "energy.pt"
    if energy_path.exists():
        energy = torch.load(energy_path)
    else:
        sequences = [entry["ids"] for entry in json.loads(args.calibration.read_text())]
        started = time.time()
        energy = collect_energy(model, sequences, args.max_tokens)
        torch.save(energy, energy_path)
        print(f"energy collected in {time.time() - started:.0f}s", flush=True)

    weights: dict[str, torch.Tensor] = {}
    report: dict[str, dict] = {}
    states: dict[str, torch.Tensor] = {}
    started = time.time()
    for index, (layer, name) in enumerate(model.linears()):
        weight_name = model.linear_name(layer, name)
        weight = model.w[weight_name].float()
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=args.device)
        with torch.no_grad():
            linear.weight.copy_(weight)
        quantized = SAGEPseudoQuantizedLinear(
            linear, channel_energy=energy[(layer, INPUT_KIND[name])].to(args.device), esg=True, adaptive_bits=True
        )
        with torch.no_grad():
            qdq = quantized.materialized_weight().float()
            relative = float((qdq - weight).square().sum() / weight.square().sum())
        weights[weight_name] = qdq.to(torch.bfloat16).cpu().contiguous()
        report[weight_name] = {
            "codebook_bits": quantized.angle_logits.shape[0].bit_length() - 1,
            "candidate_errors": quantized.candidate_errors,
            "relative_error": relative,
        }
        if args.save_state:
            for key, value in quantized.state_dict().items():
                states[f"{weight_name}.{key}"] = value.detach().cpu().contiguous()
        if index % 24 == 0:
            print(f"{index}/{len(model.linears())} {weight_name}: relative error {relative:.4f}", flush=True)
    print(f"quantized in {time.time() - started:.0f}s", flush=True)
    save_file(weights, str(args.out / "qdq_weights.safetensors"))
    if args.save_state:
        save_file(states, str(args.out / "sage_state.safetensors"))
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
