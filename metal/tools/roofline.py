"""Turn a Metal harness report into the paper's roofline accounting.

    python metal/tools/roofline.py [metal/build/report.json]

Reads the measured bandwidth and prints the single-pass TPS ceiling for the
paper's 9B / 2.38075 bpw geometry. This is the check that keeps the paper's
projected numbers honest: the projection is a division by the measured
bandwidth, not a device measurement.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    pillars = _load("omegahee_pillars", "src/omegahee_pillars.py")
    report_path = argv[1] if len(argv) > 1 else os.path.join(_ROOT, "metal", "build", "report.json")
    if not os.path.isfile(report_path):
        print(f"no report at {report_path}; run: bash metal/run.sh", file=sys.stderr)
        return 1
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)

    bandwidth = report["bandwidth"]
    best = float(bandwidth["stream_gb_per_s"])

    bpw = pillars.mirai_s_effective_bpw()
    n_params = 9.0e9
    weight_bytes = n_params * bpw / 8.0
    traffic_gb = weight_bytes / 1e9
    ceiling = pillars.project_single_pass_tps(best, traffic_gb)

    kinds = ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]
    import numpy as np

    sample_bits = np.array([3, 3, 2, 2, 3, 2, 2], dtype=np.float64)
    readkv_mean = pillars.readkv_mean_bits(np.array([4, 2, 2, 1]), np.array([2, 2, 1, 1]))

    summary = {
        "device": report.get("device"),
        "parity_all_pass": report.get("parity_all_pass"),
        "measured_stream_gb_per_s_best": best,
        "measured_stream_gb_per_s_median": float(bandwidth.get("stream_gb_per_s_median", best)),
        "effective_bpw": bpw,
        "weight_bytes_9b": weight_bytes,
        "single_pass_tps_ceiling": ceiling,
        "paper_projection_tps": 64.4,
        "paper_projection_share_of_ceiling": 64.4 / ceiling,
        "qk_protected_bits": {k: int(v) for k, v in zip(kinds, sample_bits)},
        "readkv_mean_bits_sample": readkv_mean,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
