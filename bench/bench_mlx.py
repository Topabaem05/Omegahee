"""Apple Silicon benchmark (mlx / mlx-lm). NOT run on the Linux dev box; syntax-checked only.

(a) mlx_lm generation tok/s on mlx-community/Qwen2.5-0.5B-Instruct-4bit
(b) mlx.core ports of bench/bench_toks.py: ALoDLM adaptive block-AR decode (with/without depth-aware
    KV cache) and LittleBit-2 linear paths (fp32 dense vs factored sign path), same configs.
Prints JSON shaped like bench/results_toks.json and writes bench/results_mlx.json.
Usage: see bench/README.md
"""
import argparse
import json
import math
import os
import platform
import subprocess
import time

import numpy as np
import mlx.core as mx


# ----------------------------------------------------------------------------- helpers
def timeit(fn, min_time=2.0, warmup=2):
    for _ in range(warmup):
        mx.eval(fn())
    n, t0 = 0, time.perf_counter()
    while True:
        out = fn()
        mx.eval(out)  # mlx is lazy: force compute inside the timed region
        n += 1
        el = time.perf_counter() - t0
        if el >= min_time:
            return n, el, out


def hardware():
    hw = {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
          "mlx": getattr(mx, "__version__", "?"), "device": str(mx.default_device())}
    try:
        hw["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        hw["mem_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        pass
    return hw


def softmax(x, axis=-1):
    return mx.softmax(x, axis=axis)


def rms_norm(x, eps=1e-6):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def gelu(x):
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x):
    return mx.sigmoid(mx.clip(x, -40.0, 40.0))


# ----------------------------------------------------------------------------- (a) mlx-lm
def bench_mlx_lm(model_id, prompt_tokens=512, gen_tokens=128):
    from mlx_lm import load, stream_generate
    model, tok = load(model_id)
    prompt = " ".join(["hello"] * prompt_tokens)
    last = None
    for _ in range(2):  # first run = warmup
        for resp in stream_generate(model, tok, prompt, max_tokens=gen_tokens):
            last = resp
    return {"path": "mlx_lm.stream_generate", "model": model_id,
            "prompt_tokens": getattr(last, "prompt_tokens", None), "prompt_tok_per_s": getattr(last, "prompt_tps", None),
            "generation_tokens": getattr(last, "generation_tokens", None),
            "generation_tok_per_s": getattr(last, "generation_tps", None), "peak_memory_gb": getattr(last, "peak_memory", None)}


# ----------------------------------------------------------------------------- (b) ALoDLM port
class DepthAwareKVCache:
    def __init__(self, num_layers, start_rec, end_rec, max_depth):
        self.start_rec, self.end_rec, self.max_depth = start_rec, end_rec, max_depth
        self.prelude_coda = {}
        self.core = {}
        self.deepest = {}

    def write(self, layer, depth, k, v):
        if self.start_rec <= layer < self.end_rec:
            self.core[(layer, depth)] = (k, v)
            for pos in range(k.shape[0]):
                if depth > self.deepest.get((layer, pos), 0):
                    self.deepest[(layer, pos)] = depth
        else:
            self.prelude_coda[layer] = (k, v)

    def read(self, layer, depth, seq_len, q=None):
        if not (self.start_rec <= layer < self.end_rec):
            k, v = self.prelude_coda[layer]
            k, v = k[:seq_len], v[:seq_len]
        else:
            ks, vs = [], []
            for pos in range(seq_len):
                d = min(depth, max(self.deepest.get((layer, pos), 1), 1))
                while (layer, d) not in self.core and d > 1:
                    d -= 1
                kb, vb = self.core[(layer, d)]
                ks.append(kb[pos]); vs.append(vb[pos])
            k, v = mx.stack(ks), mx.stack(vs)
        if q is not None:  # ReadKV-style query-adaptive key precision (matches numpy engine, read_bits=4)
            e = mx.mean(mx.abs(q), axis=0)
            med = mx.sort(e)[e.shape[0] // 2]
            levels = mx.where(e >= med, 63.0, 3.0)[None, :]
            kmax = mx.max(mx.abs(k), axis=0, keepdims=True) + 1e-8
            k = mx.round(k / kmax * levels) / levels * kmax
        return k, v


class Layer:
    def __init__(self, d, rng):
        s = 1.0 / math.sqrt(d)
        self.wq, self.wk, self.wv, self.wo, self.wu, self.wd = [mx.array(rng.standard_normal((d, d)) * s, dtype=mx.float32) for _ in range(6)]

    def __call__(self, h, idx, depth, cache=None):
        T, d = h.shape
        hn = rms_norm(h)
        q, k, v = hn @ self.wq.T, hn @ self.wk.T, hn @ self.wv.T
        if cache is not None:
            cache.write(idx, depth, k, v)
            k, v = cache.read(idx, depth, T, q)
        sc = (q @ k.T) / math.sqrt(d)
        if T > 1:
            sc = sc + mx.triu(mx.full((T, T), -1e9), k=1)
        h = h + (softmax(sc) @ v) @ self.wo.T
        return h + gelu(rms_norm(h) @ self.wu.T) @ self.wd.T


class ALoDLM:
    def __init__(self, vocab=64, d=32, n_pre=2, n_core=3, n_coda=2, max_depth=4, seed=123):
        rng = np.random.default_rng(seed)
        self.vocab, self.max_depth, self.n_pre, self.n_core = vocab, max_depth, n_pre, n_core
        self.L = n_pre + n_core + n_coda
        self.mask = vocab - 1
        s = 1.0 / math.sqrt(d)
        self.emb = mx.array(rng.standard_normal((vocab, d)) * s, dtype=mx.float32)
        self.pre = [Layer(d, rng) for _ in range(n_pre)]
        self.core = [Layer(d, rng) for _ in range(n_core)]
        self.coda = [Layer(d, rng) for _ in range(n_coda)]
        self.lm = mx.array(rng.standard_normal((vocab, d)) * s, dtype=mx.float32)
        self.gw = mx.array(rng.standard_normal((1, d)) * s, dtype=mx.float32)
        self.gb = 0.0

    def readout(self, h, depth, cache):
        for o, l in enumerate(self.coda):
            h = l(h, self.n_pre + self.n_core + o, depth, cache)
        probs = softmax(h @ self.lm.T)
        lam = sigmoid((h @ self.gw.T)[:, 0] + self.gb) if depth < self.max_depth else mx.ones((h.shape[0],))
        return probs, lam

    def decode(self, seq, tau=3.6, q_halt=0.7, use_kv=True):
        x = np.array(seq).copy()
        cache = DepthAwareKVCache(self.L, self.n_pre, self.n_pre + self.n_core, self.max_depth) if use_kv else None
        passes, depths = 0, {}
        while np.any(x == self.mask):
            unresolved = set(np.where(x == self.mask)[0].tolist())
            h = self.emb[mx.array(x)]
            for i, l in enumerate(self.pre):
                h = l(h, i, 1, cache)
            a = np.zeros(len(x))
            committed = {}
            for s in range(1, self.max_depth + 1):
                passes += 1
                ht = h
                for o, l in enumerate(self.core):
                    ht = l(ht, self.n_pre + o, s, cache)
                probs, lam = self.readout(ht, s, cache)
                p_np = np.array(probs)
                lam_np = np.array(lam)
                ent = -np.sum(p_np * np.log(np.clip(p_np, 1e-12, 1.0)), axis=-1)
                for i in list(unresolved):
                    a[i] = a[i] + (1 - a[i]) * lam_np[i]
                    if ent[i] <= tau or a[i] >= q_halt:
                        committed[i] = int(np.argmax(p_np[i, :-1])); depths[i] = s
                        unresolved.discard(i)
                h_np = np.array(ht)
                emb_np = np.array(self.emb)
                for i, t in committed.items():
                    h_np[i] = emb_np[t]
                h = mx.array(h_np)
                if not unresolved:
                    break
            if not committed and unresolved:
                probs, _ = self.readout(h, self.max_depth, cache)
                p_np = np.array(probs)
                ent = -np.sum(p_np * np.log(np.clip(p_np, 1e-12, 1.0)), axis=-1)
                b = min(unresolved, key=lambda i: ent[i])
                committed[b] = int(np.argmax(p_np[b, :-1])); depths[b] = self.max_depth
            for i, t in committed.items():
                x[i] = t
        return mx.array(x), {"total_inner_passes": float(passes),
                              "mean_commit_depth": float(np.mean(list(depths.values()))) if depths else 1.0}


def bench_decode(d, seq_len, n_mask, use_kv):
    model = ALoDLM(d=d)
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 60, size=seq_len)
    seq[rng.choice(seq_len, n_mask, replace=False)] = model.mask
    stats = {}

    def run():
        out, st = model.decode(seq, use_kv=use_kv)
        stats.update(st)
        return out

    n, el, _ = timeit(run)
    return {"path": "ALoDLM adaptive_block_ar_decode (mlx)", "kv_cache": use_kv, "d_model": d, "seq_len": seq_len,
            "masked_tokens": n_mask, "runs": n, "sec_per_run": el / n, "tok_per_s": n * n_mask / el, **stats}


# ----------------------------------------------------------------------------- (b) LittleBit-2 linear port
def bench_littlebit(m, n_in, rank, batch, dtype=mx.float32):
    rng = np.random.default_rng(1)
    k = min(m, n_in)
    u0, _ = np.linalg.qr(rng.standard_normal((m, k))); v0, _ = np.linalg.qr(rng.standard_normal((n_in, k)))
    w = (u0 * np.exp(-np.linspace(0, 3.5, k))) @ v0.T
    U, S, Vt = np.linalg.svd(w, full_matrices=False)  # sign factors from rank-r SVD (timing depends only on shapes)
    u = U[:, :rank] * np.sqrt(S[:rank]); v = Vt[:rank].T * np.sqrt(S[:rank])
    bu = mx.array(np.sign(u + 1e-12), dtype=dtype); bv = mx.array(np.sign(v + 1e-12), dtype=dtype)
    a = mx.array(np.abs(u).mean(1), dtype=dtype); b = mx.array(np.abs(v).mean(1), dtype=dtype)
    g = mx.ones((rank,), dtype=dtype)
    W = mx.array(w, dtype=dtype)
    x = mx.array(rng.standard_normal((batch, n_in)), dtype=dtype)
    paths = {"fp32 dense W": lambda: x @ W.T,
             "LittleBit-2 factored sign path (rank r, unpacked fp32)":
                 lambda: ((((x * b) @ bv) * g) @ bu.T) * a + 0.1 * ((x @ bv) @ bu.T)}
    rows = []
    for name, fn in paths.items():
        nr, el, _ = timeit(fn, min_time=1.5)
        rows.append({"path": name, "m": m, "n": n_in, "rank": rank, "batch_tokens": batch, "runs": nr, "tok_per_s": nr * batch / el})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--skip-mlx-lm", action="store_true")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    args = ap.parse_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    res = {"hardware": hardware(), "mlx_lm": None, "decode": [], "littlebit2": [], "littlebit2_qwen_dims": []}
    if not args.skip_mlx_lm:
        res["mlx_lm"] = bench_mlx_lm(args.model)
    for d in (32, 128):  # same configs as bench_toks.py
        for kv in (True, False):
            res["decode"].append(bench_decode(d, 32, 16, kv))
    for m, batch in ((512, 1), (2048, 1), (2048, 64)):
        res["littlebit2"] += bench_littlebit(m, m, 16, batch)
    for (m, n) in ((896, 896), (128, 896), (4864, 896), (896, 4864)):  # Qwen2.5-0.5B shapes
        for batch in (1, 64):
            res["littlebit2_qwen_dims"] += bench_littlebit(m, n, 16, batch)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_mlx.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
