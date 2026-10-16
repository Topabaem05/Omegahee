# Resume / rebuild the benchmark environment

Cleaned after the 2026-10-16 run: llama.cpp clone+build, GGUF models, zig toolchain wrappers, pip/HF caches.
Kept: this repo, `bench/*.json`, `/workspace/venv`.

```bash
# 0) Python env (kept; recreate if missing)
python3 -m venv /workspace/venv && /workspace/venv/bin/pip install numpy pytest cmake ninja ziglang

# 1) Compiler: system g++ was not installable (broken apt deps), so zig is used as C/C++ compiler
mkdir -p /workspace/zigcc
printf '#!/bin/sh\nexec /workspace/venv/bin/python -m ziglang cc "$@"\n'  > /workspace/zigcc/cc
printf '#!/bin/sh\nexec /workspace/venv/bin/python -m ziglang c++ "$@"\n' > /workspace/zigcc/c++
chmod +x /workspace/zigcc/*

# 2) llama.cpp (measured build: commit f12aaa3). Static build is REQUIRED with zig (shared libs -> double free).
cd /workspace && git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout f12aaa3   # optional, for identical build (needs full clone: drop --depth)
PATH=/workspace/venv/bin:$PATH CC=/workspace/zigcc/cc CXX=/workspace/zigcc/c++ \
  cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF
PATH=/workspace/venv/bin:$PATH cmake --build build -j8 --target llama-bench

# 3) Models
mkdir -p /workspace/models && cd /workspace/models
for q in q4_k_m q8_0; do curl -L -o qwen2.5-0.5b-instruct-$q.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-$q.gguf; done

# 4) Run
/workspace/llama.cpp/build/bin/llama-bench -m /workspace/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -m /workspace/models/qwen2.5-0.5b-instruct-q8_0.gguf -t 8 -p 512 -n 128 -r 5 -o json > /tmp/llb.json
cd /workspace/att/omegahee_bundle
/workspace/venv/bin/python bench/bench_toks.py
/workspace/venv/bin/python bench/bench_qwen_dims.py
/workspace/venv/bin/pytest -q tests
```

## uzu (trymirai/uzu) on x86 Linux CPU — added 2026-10-16 (cleaned afterwards: /tmp/uzu, ~/.rustup, ~/.cargo, ~/.cache/mirai, /tmp/alsa)
Measured with uzu commit ee5ec57 (engine 0.6.8). x86_64 Linux is "in progress" upstream; CPU backend is single-threaded/reference-speed.
```bash
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal --no-modify-path
source ~/.cargo/env
git clone --depth 1 https://github.com/trymirai/uzu /tmp/uzu
# rodio->alsa-sys needs ALSA headers; system apt is broken, so extract them locally + fake pkg-config
mkdir -p /tmp/alsa && cd /tmp/alsa && apt-get download libasound2-dev && dpkg -x libasound2-dev*.deb root
ln -sf /usr/lib/x86_64-linux-gnu/libasound.so.2 root/usr/lib/x86_64-linux-gnu/libasound.so
cat > /tmp/alsa/pkg-config <<'SH'
#!/bin/sh
for a in "$@"; do [ "$a" = --modversion ] && { echo 1.2.13; exit 0; }; done
case "$*" in *--exists*) exit 0;; esac
o=""; case "$*" in *--cflags*) o="$o -I/tmp/alsa/root/usr/include";; esac
case "$*" in *--libs*) o="$o -L/tmp/alsa/root/usr/lib/x86_64-linux-gnu -lasound";; esac; echo $o
SH
chmod +x /tmp/alsa/pkg-config
cd /tmp/uzu && PKG_CONFIG=/tmp/alsa/pkg-config CXX=/workspace/zigcc/c++ PATH=/workspace/venv/bin:$PATH cargo build --release -p cli -j8
./target/release/cli --model trymirai/Qwen3.5-0.8B-M -m "hi" --reasoning-effort disabled   # downloads to ~/.cache/mirai
echo '{"identifier":"qwen3.5-0.8b-cpu","repo_id":"trymirai/Qwen3.5-0.8B-M","number_of_runs":2,"tokens_limit":16,"greedy":true,"reasoning_effort":"disabled","messages":[{"role":"user","content":"Write one short sentence about the sea."}]}' > /tmp/uzu_task.json
./target/release/cli bench ~/.cache/mirai/models/mirai/alibaba-qwen3.5-0.8b-mirai-mirai-m-4/*/ /tmp/uzu_task.json /tmp/uzu_bench_out.json
```
Matching llama.cpp model: `ggml-org/Qwen3.5-0.8B-GGUF` (Qwen3.5-0.8B-Q4_0.gguf, -Q8_0.gguf); llama.cpp build as in step 2 above (measured commit a8d9a7e).
Run with `-t 8 -p 512 -n 128 -r 5` and single-thread `-t 1 -p 20 -n 16 -r 2`.
On a Mac: `bash bench/bench_uzu.sh` (Metal backend).
