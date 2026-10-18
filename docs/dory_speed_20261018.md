# Dory 1.0-4B 디코드 속도: CPU 분할 제거 (2026-10-18)

## 결론

Dory 지원 llama.cpp(`98a74c9`)는 Dory의 BF16 스케일 벡터를 **CPU 버퍼**에 올렸다.
그래서 토큰마다 virtual layer별로 CPU↔Metal 동기화가 일어났다(그래프당 분할
8,085개). 가중치 배치 판정을 실제 첫 소비 연산(`CPY`, BF16→F32)으로 바꾸자
분할이 14개로 줄고 디코드가 **2.8–3.2배** 빨라졌다. greedy 출력은 수정 전과
바이트 단위로 같다(9/9).

| loops | 수정 전 (tok/s) | 수정 후 (tok/s) | 배수 | 토큰당 가중치 | 실효 대역폭(후) |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.29 ± 7.19 | **61.66 ± 0.88** | 3.20x | 2.994GB | 185GB/s |
| 2 | 14.71 ± 0.08 | **42.16 ± 0.31** | 2.87x | 4.276GB | 180GB/s |
| 4 (기본) | 8.95 ± 0.53 | **25.12 ± 0.23** | 2.81x | 6.841GB | 172GB/s |

중간값 ± 표준편차, 5표본, 128토큰, greedy, context 4096, flash-attn on.
15토큰 prompt 처리도 4 loops에서 264ms → 103ms로 줄었다.

**상태: 잠정.** 두 측정 모두 시작 시 GPU 0%였지만, 스왑 4.2GB와 디스크 여유
13.9GB 때문에 `clean_env_check.py`는 실패했다. 수정 후 측정의 끝 무렵에는
다른 GPU 작업(WindowServer/WebKit)이 시작됐다(`gate_after` 96%). 배수는
표준편차보다 훨씬 커서 결론은 유지되지만, 절대 tok/s는 청정 재측정이 필요하다.
같은 세션에서 GPU 유휴 상태로 잰 `llama-bench tg64`(4 loops)는 8.35 → 26.31 tok/s였다.

## 원인

`LLM_TENSOR_INFOS`가 Dory 스케일 텐서 6종(`residual_alpha`, `attn_qk_scale`,
`attn_gate_scale`, `ffn_gate_scale`, `ffn_up_scale`, `output_scale`)의 소비 연산을
`GGML_OP_MUL`로 적었다. 로더는 이 연산으로 버퍼를 고른다. Metal `MUL`은 BF16
입력을 받지 않으므로 이 텐서들은 CPU 버퍼로 갔다.

그러나 `dory.cpp` 그래프는 이 텐서를 먼저 `ggml_cast(..., F32)`(`CPY`)한다.
스케줄러는 이 `CPY`를 가중치가 있는 CPU에 배정했고, 결과를 Metal로 다시 복사했다.
레이어마다 스케일이 2–3개이고 virtual layer가 192개(4 loops)라 그래프가 약
4,000개의 CPU 구간과 4,000개의 Metal 구간으로 쪼개졌다.

`GGML_SCHED_DEBUG=1` 실측:

| | CPU 분할 | Metal 분할 | CPU 버퍼 | Metal 버퍼 |
|---|---:|---:|---:|---:|
| 수정 전 | 4,046 | 4,039 | 3,457MiB | 3,495MiB |
| 수정 후 | 7 | 7 | 640MiB | 3,496MiB |

남은 CPU 분할은 정상적인 `token_embd` 조회(BF16 임베딩, CPU 버퍼)다.

## 수정

`patches/dory-llama-cpy-placement.patch` (10줄 추가, 6줄 삭제):

1. `src/llama-arch.cpp`: 위 6개 텐서의 op를 `GGML_OP_MUL` → `GGML_OP_CPY`.
2. `src/llama-model-loader.cpp`: `weight_buft_supported`에 `GGML_OP_CPY` 테스트
   (`ggml_cast(w, F32)`) 추가. Metal은 BF16→F32 `CPY`를 지원한다.

그래프와 수치 경로는 바꾸지 않았다. BF16→F32 변환은 정확하므로 결과가 같아야
하고, 실제로 같았다.

적용:

```bash
cd llama.cpp-dory
git apply ../omega-uzu/patches/dory-llama-cpy-placement.patch
cmake -B build-m5 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF
cmake --build build-m5 -j 14 --target llama-completion llama-bench
```

## 검증

- **파리티**: `tools/dory_llama_parity.sh`로 수정 전 빌드(`build-ref`, 원본
  `98a74c9`)와 수정 후 빌드(`build-m5`)를 비교했다. loops 1/2/4 × 프롬프트 3개,
  96토큰 greedy 출력이 9/9 바이트 단위로 같다. 이 스크립트는 출력이 다르면
  DIFF와 exit 1을 낸다(아래 실패 실험에서 실제로 DIFF를 냈다).
- **분할 수**: 위 표.
- **속도**: `tools/dory_bench.py` 원자료
  - `bench/results/dory_speed_ref_20261018.json`
  - `bench/results/dory_speed_cpy_placement_20261018.json`

참고: 이전 기록 `dory_4b_loops_20261018.json`(1/2/4 loops = 20.0 / 8.9 / 10.3)은
2 loops가 4 loops보다 느린 이상값이었다. 새 수정 전 측정은 19.3 / 14.7 / 8.95로
loop 수에 따라 단조 감소한다. 이전 2 loops 값은 재현되지 않으므로 인용하지 않는다.

## 시도했으나 채택하지 않은 것

**스케일 체인 사전 계산(fold).** uzu가 상수 스케일을 epilogue에 접어 두는 방식을
따라, 로드 직후 `|α·s|`, `(x·s)²`, `x·s`를 물리 레이어당 한 번 F32로 계산하고
그래프에서 cast/scale/sqr/abs를 뺐다(토큰당 약 1,500 dispatch 감소).

- `llama-bench tg64` 4 loops: 26.31 ± 0.18 → 26.25 ± 0.11. **이득 없음.**
  이 연산들은 가중치에만 의존해 임계 경로 밖에서 병렬로 돌기 때문이다.
- `llama-completion`의 메모리 추정(no_alloc) 패스에서 미할당 텐서를 읽어
  abort했다(고칠 수 있음).
- 연구 계획의 채택 규칙(개선 > 표준편차)에 따라 되돌렸다.

## 남은 간격

`test-backend-ops perf`로 잰 Metal mat-vec 커널 대역폭:

| 커널 (m=4096, k=14336, n=1) | 시간 | 대역폭 |
|---|---:|---:|
| Q5_K | 161.5µs | 약 250GB/s |
| BF16 | 456.8µs | 약 257GB/s |

이 커널 대역폭이면 4 loops 상한은 약 36.5 tok/s(6.841GB / 250GB/s)이고, 현재는
그 69%다. loop 수별 시간을 맞추면 recurrent 구간의 한계 대역폭은 약 163GB/s,
virtual layer당 약 60–70µs가 mat-vec 밖에서 쓰인다. 후보는 레이어당 6–7번
직렬로 도는 L2 norm(`rms_norm` + `scale` 2커널)과 residual 혼합(sub/mul/add)이다.
다음 실험은 L2 norm을 한 커널(`ggml_l2_norm` 또는 Metal 융합)로 줄이는 것이다.
이는 eps 처리 방식이 달라 비트 동일성이 깨질 수 있으므로 파리티와 품질을 함께 본다.

## 재현

```bash
python3 tools/dory_bench.py --repo ../llama.cpp-dory --build build-m5 \
  --model /Users/topamini/models/dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
  --runs 5 --label dory-cpy-placement --out bench/results/<name>.json
```

청정 게이트가 실패하면 도구는 실행을 거부한다. `--allow-dirty-env`로 실행하면
기록에 `"provisional": true`가 붙는다.
