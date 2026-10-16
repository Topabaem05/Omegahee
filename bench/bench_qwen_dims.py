"""Linear-layer tok/s at Qwen2.5-0.5B shapes (hidden 896, MLP 4864, KV dim 128, 24 layers).
fp32 dense W vs LittleBit-2 factored sign path (rank r, unpacked fp32 numpy).
Weights are synthetic (decaying spectrum), NOT the real Qwen weights; timing depends only on shapes.
Sign factors come from LittleBit2QATEngine.joint_itq (full QAT/AWSRC skipped: too slow at these sizes, does not change shapes).
Writes bench/results_qwen_dims.json."""
import json, os, sys, time, platform, multiprocessing
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import omegahee_uzu_engine as E

N_LAYERS = 24
SHAPES = {"q_proj": (896, 896), "k_proj": (128, 896), "v_proj": (128, 896), "o_proj": (896, 896),
          "gate_proj": (4864, 896), "up_proj": (4864, 896), "down_proj": (896, 4864)}


def timeit(fn, min_time=1.0, warmup=3):
    for _ in range(warmup):
        fn()
    n, t0 = 0, time.perf_counter()
    while True:
        fn(); n += 1
        el = time.perf_counter() - t0
        if el >= min_time:
            return n, el


def main(rank=16):
    rng = np.random.default_rng(0)
    eng = E.LittleBit2QATEngine(rank=rank, itq_iters=10)
    rows = []
    for name, (m, n) in SHAPES.items():
        k = min(m, n)
        a, _ = np.linalg.qr(rng.standard_normal((m, k))); b, _ = np.linalg.qr(rng.standard_normal((n, k)))
        w = (a * np.exp(-np.linspace(0, 3.5, k))) @ b.T
        u, v, _, _ = eng.joint_itq(w, rank=rank)
        bu = np.where(u >= 0, 1, -1).astype(np.float32); bv = np.where(v >= 0, 1, -1).astype(np.float32)
        ru = bu.copy(); rv = bv.copy()  # residual path has the same shape
        rs = np.abs(u).mean(1).astype(np.float32); cs = np.abs(v).mean(1).astype(np.float32); g = np.ones(rank, np.float32)
        w32 = w.astype(np.float32)
        for batch in (1, 64):
            x = rng.standard_normal((batch, n)).astype(np.float32)
            paths = {"fp32_dense": lambda: x @ w32.T,
                     "littlebit2_factored": lambda: (((x * cs) @ bv) * g) @ bu.T * rs + 0.1 * ((x @ rv) @ ru.T)}
            for p, fn in paths.items():
                it, el = timeit(fn)
                rows.append({"matrix": name, "m": m, "n": n, "rank": rank, "batch": batch, "path": p,
                             "us_per_call": 1e6 * el / it, "tok_per_s": it * batch / el})
    est = []
    for batch in (1, 64):
        for p in ("fp32_dense", "littlebit2_factored"):
            us = sum(r["us_per_call"] for r in rows if r["batch"] == batch and r["path"] == p)
            est.append({"ESTIMATE": True, "batch": batch, "path": p, "layers": N_LAYERS,
                        "est_linear_only_ms_per_step": N_LAYERS * us / 1e3,
                        "est_linear_only_tok_per_s": batch / (N_LAYERS * us / 1e6),
                        "note": "sum of measured per-matrix times x 24 layers; excludes attention, norms, RoPE, embeddings, lm_head (151936x896), threading/overhead. Not a real model run."})
    hw = {"cpu_model": next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), ""),
          "logical_cpus": multiprocessing.cpu_count(), "python": platform.python_version(), "numpy": np.__version__}
    res = {"hardware": hw, "measured": rows, "estimate_full_linear_stack": est}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_qwen_dims.json"), "w"), indent=2)
    for r in rows:
        print(f"{r['matrix']:10s} b={r['batch']:<3d} {r['path']:20s} {r['tok_per_s']:12.0f} tok/s")
    for e in est:
        print("ESTIMATE", e["batch"], e["path"], round(e["est_linear_only_tok_per_s"], 1))


if __name__ == "__main__":
    main()
