"""CPU tokens/sec benchmark for Omegahee engine (numpy). Writes bench/results_toks.json."""
import json, os, platform, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import omegahee_uzu_engine as E


def timeit(fn, min_time=2.0, warmup=2):
    for _ in range(warmup):
        fn()
    n, t0 = 0, time.perf_counter()
    while True:
        r = fn(); n += 1
        el = time.perf_counter() - t0
        if el >= min_time:
            return n, el, r


def bench_decode(d_model, seq_len, n_mask, use_kv):
    model = E.ALoDLMHybridARModel(vocab_size=64, d_model=d_model, seed=123)
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 60, size=seq_len)
    seq[rng.choice(seq_len, n_mask, replace=False)] = model.mask_token_id
    n, el, (out, st) = timeit(lambda: model.adaptive_block_ar_decode(seq, entropy_tau=3.6, halt_threshold_q=0.7, use_kv_cache=use_kv))
    return {"path": "ALoDLM adaptive_block_ar_decode", "kv_cache": use_kv, "d_model": d_model, "seq_len": seq_len,
            "masked_tokens": n_mask, "runs": n, "sec_per_run": el / n, "tok_per_s": n * n_mask / el,
            "inner_passes": st["total_inner_passes"], "mean_commit_depth": st["mean_commit_depth"]}


def bench_littlebit(m, n_in, rank, batch):
    rng = np.random.default_rng(1)
    u0, _ = np.linalg.qr(rng.standard_normal((m, m))); v0, _ = np.linalg.qr(rng.standard_normal((n_in, n_in)))
    k = min(m, n_in); w = (u0[:, :k] * np.exp(-np.linspace(0, 3.5, k))) @ v0[:, :k].T
    x_cal = rng.standard_normal((64, n_in))
    eng = E.LittleBit2QATEngine(rank=rank, itq_iters=25, seed=42)
    w_rec, res = eng.run_qat_with_l2l_recovery(w, x_cal, steps=35, lr=0.04, num_seeds=8)
    # factored binary path (LittleBit-2 deploy form): diag(a) sign(U) diag(g) sign(V)^T diag(b) + res path
    bu = np.where(res.u_rot >= 0, 1.0, -1.0).astype(np.float32); bv = np.where(res.v_rot >= 0, 1.0, -1.0).astype(np.float32)
    bru = np.where(res.res_u_rot >= 0, 1.0, -1.0).astype(np.float32); brv = np.where(res.res_v_rot >= 0, 1.0, -1.0).astype(np.float32)
    a = res.row_scale.ravel().astype(np.float32); b = res.col_scale.ravel().astype(np.float32)
    g = res.latent_scale.astype(np.float32); rs = np.float32(res.res_scale)
    x = rng.standard_normal((batch, n_in)).astype(np.float32)
    w32 = w.astype(np.float32); wrec32 = w_rec.astype(np.float32)
    fp = lambda: x @ w32.T
    rec = lambda: x @ wrec32.T
    fac = lambda: (((x * b) @ bv) * g) @ bu.T * a + rs * ((x @ brv) @ bru.T)
    rows = []
    for name, fn in [("fp32 dense W", fp), ("LittleBit-2 reconstructed dense W_hat", rec), ("LittleBit-2 factored sign path (rank r, unpacked fp32)", fac)]:
        nr, el, _ = timeit(fn, min_time=1.5)
        rows.append({"path": name, "m": m, "n": n_in, "rank": rank, "batch_tokens": batch, "runs": nr,
                     "tok_per_s": nr * batch / el})
    return rows


if __name__ == "__main__":
    import multiprocessing
    hw = {"cpu": platform.processor() or "", "machine": platform.machine(), "logical_cpus": multiprocessing.cpu_count(),
          "python": platform.python_version(), "numpy": np.__version__}
    try:
        hw["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    results = {"hardware": hw, "decode": [], "littlebit2": []}
    for d in (32, 128):
        for kv in (True, False):
            results["decode"].append(bench_decode(d, 32, 16, kv))
    for m, batch in ((512, 1), (2048, 1), (2048, 64)):
        results["littlebit2"] += bench_littlebit(m, m, 16, batch)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_toks.json")
    json.dump(results, open(out, "w"), indent=2)
    print(json.dumps(results, indent=2))
