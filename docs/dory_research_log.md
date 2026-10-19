# Dory 4B tok/s 연구 기록

연구 계획: "Dory uzu tok/s 개선 연구 계획서"(2026-10-19). 이 문서는 측정 조건, 결과, 실험 판정을 쌓아 가는 기록입니다.

## 측정 환경

| 항목 | 값 |
|---|---|
| 기기 | Apple M5 Pro, 48GB, macOS 27.0 |
| 엔진 | Ω-Hee `engine/` (uzu `6671962` 기반), 커밋은 각 결과 행에 기록 |
| 모델 | Dory1.0-4B BF16 원본 `b366b87`(`model.safetensors` SHA-256 `20748366…`) |
| 측정 도구 | `tools/omega_bench.py`(benchmarks-uzu, 60°C 이하 대기, 워밍업 1회 제외 5회 중앙값), `tools/omega_eval.py`(품질 묶음) |
| 요청 묶음 | `bench/requests/prompt_{128,2048,16384}.json`(저장소 MIT/Apache 문서 발췌, SHA-256 `bench/requests/manifest.json`), 출력 256토큰, greedy |
| 품질 묶음 | `bench/eval/`: GSM8K 200, HumanEval 164(sandbox-exec), MMLU-Pro 500, 16K 검색 20. greedy, 추론 끔. 묶음 점수 = 네 점수 평균 |

품질 묶음은 추론(thinking)을 끈 상태로 잽니다. 추론을 켜면 변형 하나에 BF16으로 수 시간이 걸려 반복 비교가 어렵기
때문입니다. 양자화 간 상대 비교 기준으로 쓰고, 절대 점수는 모델 카드 수치와 비교하지 않습니다.

## 이론 상한

토큰당 읽는 가중치 바이트는 변환 결과 safetensors 헤더의 실제 텐서 크기에서 계산합니다(입력 임베딩은 토큰당 한 행만
읽으므로 제외, recurrent 층은 루프 수만큼). 대역폭 307GB/s는 연구 계획서의 M5 Pro 가정값입니다(별도 실측 아님).

| 모델 | 1루프 | 2루프 | 4루프 |
|---|---|---|---|
| BF16 | 7.28GB → 42.2 tok/s | 10.95GB → 28.0 tok/s | 18.30GB → 16.8 tok/s |
| MLX 4비트 g64 + INT8 readout | 2.21GB → 139.0 tok/s | 3.24GB → 94.7 tok/s | 5.31GB → 57.8 tok/s |

## 실험 기록

| ID | 변경 | 가설 | 전 decode tok/s | 후 decode tok/s | 커밋 | 판정 |
|---|---|---|---|---|---|---|
| E-00 | 기준선 BF16(2루프, per_loop, 2K) | — | — | 24.43 [24.36, 24.55] | `6e7ef86` | 기준 |
| E-00 | 기준선 INT4(MLX 4비트 g64, readout INT8 대칭 g64) | — | — | 66.02 [65.40, 66.10] | `6e7ef86` | 기준 |
| E-01 | 층 입력 복사를 잔차 커널 이중 출력으로 대체 | 토큰당 복사 dispatch 112개 제거 | BF16 24.43 / INT4 66.02 | BF16 24.65 [24.58, 24.74] / INT4 66.84 [66.75, 66.93] | `8117615` | 채택(출력 비트 동일, 개선폭 > 잡음) |
| E-02 | MLP의 채널 scale + SwiGLU + 은닉 L2 norm을 커널 하나로 합침 | 토큰당 dispatch 112개 제거 | BF16 24.65 / INT4 66.84 | BF16 24.63 [24.61, 24.75] / INT4 67.08 [66.84, 67.24] | (이 커밋) | 단독으로는 잡음 이내, E-03과 묶어 판정 |
| E-03 | attention의 gate scale + sigmoid gate + 출력 L2 norm을 커널 하나로 합침 | 토큰당 dispatch 약 112개 제거 | BF16 24.63 / INT4 67.08 | BF16 24.70 [24.65, 24.80] / INT4 67.47 [67.17, 67.58] | (이 커밋) | E-02+E-03 묶음 채택(출력 비트 동일, INT4 +0.9%, 범위 겹치지 않음) |
| E-04 | sliding window층 single-pass 대신 two-pass(임계 KV 1,024→256) | 16 threadgroup single-pass가 지연 병목 | INT4 67.47 | INT4 67.35 [66.77, 67.38] (128: 70.90) | (기록만) | 기각(개선 없음, 되돌림) |
| E-05 | readout INT8 대칭 → INT4 MLX g64(변환만 변경) | readout 읽기 0.35GB→0.19GB | INT4 67.47 | 69.02 [68.39, 69.92] | (변환 설정) | 속도는 +2.3%, 품질 묶음 판정 대기 |
| E-06 | 루프 wavefront decode(recurrent 층 batch 2, 1루프 조기 종료로 다음 토큰 추측), `OMEGA_WAVEFRONT=1` | recurrent 가중치와 고정 비용을 두 토큰이 공유 | BF16 24.70 / INT4 67.47 | BF16 30.96 [30.92, 30.97] / INT4 76.16 [75.56, 76.28] | `67e85fb`, (이 커밋) | 속도 채택(INT4 +12.9%, BF16 +25.3%), 근사 변경이라 품질 묶음 판정 후 기본값 결정 |
| E-07 | wavefront 다음 단계를 추측이 맞다고 보고 미리 인코딩(맞으면 제출, 틀리면 버리고 KV 장부 복원) | 단계마다 드러나던 CPU 인코딩 시간 숨김 | BF16 30.96 / INT4 76.16 | BF16 31.33 [31.06, 31.38] / INT4 78.44 [77.33, 79.16] | (이 커밋) | 채택(출력 토큰 E-06과 동일, INT4 +3.0%) |
| E-08 | MLX(ScaleBias) INT4 g64 m=2 GEMV에 입력 2행 타일(2 simdgroup, 출력 16행, lane 8) | batch 2가 가중치를 한 번만 읽음 | wavefront INT4 78.44 | wavefront INT4 82.27 [81.42, 82.35] (readout INT4: 83.78) | (이 커밋) | 채택(GEMV 정확도 동일, +4.9%) |
| E-09 | query scale이 있는 attention의 INT8 KV(전체 attention 9층), `OMEGA_INT8_KV=1` | 16K에서 전체 attention KV 읽기 절반 | wavefront INT4 r4 2K 83.99 / 16K 63.13 | 2K 83.01 [82.08, 84.05] / 16K 69.78 [69.07, 71.45] | (이 커밋) | 16K +10.5%(16K/2K 75%→84%), parity 근사 오차 증가, 품질 묶음 판정 대기 |
| E-10 | 잔차 mix의 두 내부 norm 합을 reduction 한 번으로(barrier 9→6, 출력 비트 동일) | 단일 threadgroup 커널의 barrier 지연 | wavefront INT4 r4 83.79(ABBA 평균) | 83.84(ABBA 평균) | (기록만) | 기각(차이 없음, 되돌림) |
| E-12 | 공식 NVFP4 부호를 엔진에서 직접 실행(FP4 GEMV, prefill은 BF16 전개 후 GEMM), 그룹 16 BF16 스케일 | INT4 바이트로 반복 층 INT8 혼합 품질 | 혼합(반복 층 INT8) 48.17 / wavefront 61.69 | 순차 60.72 [60.46, 60.88] / wavefront 69.07 [68.18, 69.31] | (이 커밋) | KL 0.0428(혼합 0.044, INT4 0.135), 품질 묶음 측정 중 |
| E-13 | FP4 그룹 16 m=2 GEMV를 32 lane 타일(8 simdgroup, 출력 32행, 입력 2행)로 | 8 lane 타일의 batch 2 비용 1.16~1.25배 → 1.02~1.05배 | NVFP4 wavefront 69.07 | 74.44 [74.06, 74.51] (128: 81.83, 16K: 57.53) | (이 커밋) | 채택(CPU 기준 오차 동일, +7.8%) |

### 기준선 전체 (E-00, 2루프 per_loop, 워밍업 제외 5회 중앙값 [최소, 최대])

| 모델 | 프롬프트 | decode tok/s | 효율 | prefill tok/s |
|---|---|---|---|---|
| BF16 | 128 | 25.15 [25.11, 25.24] | 89.8% | 1,042 |
| BF16 | 2,048 | 24.43 [24.36, 24.55] | 87.2% | 1,329 |
| BF16 | 16,384 | 22.77 [22.72, 22.79] | 81.2% | 1,181 |
| INT4 | 128 | 70.58 [70.24, 70.69] | 74.5% | 1,415 |
| INT4 | 2,048 | 66.02 [65.40, 66.10] | 69.7% | 1,335 |
| INT4 | 16,384 | 54.40 [54.24, 54.49] | 57.5% | 1,116 |

- 긴 문맥 유지율(16K ÷ 2K): BF16 93%, INT4 82%. 연구 계획의 긴 문맥 목표(80%)는 이미 넘습니다.
- 잡음 폭(5회 최대 − 최소): BF16 0.19, INT4 0.70 tok/s. 10회 반복 잡음 측정은 별도로 진행합니다.
- INT4 readout: uzu Metal GEMV의 MLX(scale+bias) 경로는 4비트만 있어 8비트 readout은 `IntSpec(bits=8, is_symmetric)`로 변환합니다.

### 루프 수 × KV 모드 (E-01 엔진, 2K, 워밍업 제외 5회 중앙값 [최소, 최대])

| 모델 | 루프 | per_loop | last_loop | 상한 | per_loop 효율 |
|---|---|---|---|---|---|
| INT4 | 1 | 97.59 [96.91, 98.50] | 97.03 [95.69, 97.68] | 139.0 | 70.2% |
| INT4 | 2 | 66.63 [66.50, 66.76] | 65.48 [65.36, 65.59] | 94.7 | 70.4% |
| INT4 | 4 | 40.14 [38.50, 40.30] | 39.35 [39.18, 39.59] | 57.8 | 69.5% |
| BF16 | 1 | 36.48 [36.15, 36.69] | 37.22 [37.13, 37.28] | 42.2 | 86.5% |
| BF16 | 2 | 24.63 [24.59, 24.73] | 24.61 [24.51, 24.73] | 28.0 | 87.9% |
| BF16 | 4 | 14.21 [14.18, 14.30] | 14.55 [14.50, 14.71] | 16.8 | 84.7% |

- 토큰당 시간을 루프 수에 직선으로 맞추면 INT4 per_loop는 루프당 약 4.9ms, 고정 부분(입력·출력 층과 readout)
  약 5.4ms입니다. 두 부분 모두 상한 대비 약 70%라서, 손실은 루프 구간에만 몰려 있지 않습니다.
- INT4에서 last_loop가 per_loop보다 1~2% 느립니다. BF16에서는 차이가 없거나 반대입니다. 원인은 아직 모릅니다.
- 같은 E-01 엔진의 INT4 2루프 per_loop가 이 행렬에서는 66.63, E-01 단독 측정에서는 66.84입니다. 실행
  사이 잡음은 약 0.2 tok/s로 봅니다.

### E-02, E-03 메모

- 두 실험 모두 출력 로짓이 비트 단위로 같습니다. 짧은 프롬프트 5개(single-pass attention)와 1,830토큰 프롬프트
  (two-pass attention, decode 15토큰)를 E-01 이전 덤프와 비교했습니다. 합친 커널의 행 합 순서가 기존 Normalization
  커널(1,024 스레드 strided 합)과 같아서 결과가 같습니다.
- 합친 경로가 실제로 쓰였는지는 임시 로그로 확인했습니다(프롬프트 5개 × 32 forward × 56층 = 8,960회씩).
- dispatch 224개를 없앴는데 INT4가 약 0.9%(약 0.14ms/토큰)만 빨라졌습니다. 작은 커널 하나의 비용은 약 0.6µs입니다.
  남은 30% 손실의 대부분은 작은 커널 수가 아니라 GEMV 자체의 대역폭 효율에 있다고 봅니다. 다음 측정은 Dory 모양
  GEMV 마이크로벤치(`unit/backends/metal/kernel/matmul/dory_bench.rs`)입니다.
- 처음 E-03 빌드는 2K 요청에서 멈췄습니다. two-pass attention 커널이 `gate_row_stride`로 gate 유무를 고정하는데,
  합친 경로가 gate 없이 호출했기 때문입니다. 짧은 parity 프롬프트는 KV 1,024 이하라 two-pass를 거치지 않아
  놓쳤습니다. 이제 합친 경로에서는 커널을 gate 없이 만들고, 출력 불변 변경은 긴 프롬프트 parity까지 확인합니다.

### 루프 조기 종료 일치율(단계 4 예비 측정)

teacher-forced 프롬프트 토큰 128개(짧은 프롬프트 5개)에서 1루프 출력과 2루프 출력의 다음 토큰을 비교했습니다.
top-1 일치 86.7%, 2루프 top-1이 1루프 top-3 안에 있는 비율 98.4%입니다. 표본이 작고 생성 문장이 아닌 프롬프트
문장이라 참고값입니다. 단순 비용 모형(INT4, 루프 4.9ms, 입력·출력 층 각 1.9ms, readout 1.6ms)으로는
- 첫 루프를 공유하는 자기 추측(k=3, 일치율 0.9): 약 73 tok/s
- 루프 wavefront(토큰 t의 2루프와 t+1의 1루프를 한 batch로, 일치율 0.9): 약 84 tok/s
입니다. GEMV 효율이 오르지 않으면 어느 방법도 100 tok/s에 닿지 않습니다.

### GEMV 실측과 대역폭 모형 수정 (2026-10-19)

`unit/backends/metal/kernel/matmul/dory_bench.rs`로 Dory decode 모양 GEMV를 따로 쟀습니다(256MiB 순환 버퍼로 캐시를
피함, 결과 `bench/results/gemv_dory_shapes_20261019.txt`).

| GEMV(INT4 MLX g64) | batch 1 | batch 2 | batch 4 |
|---|---|---|---|
| qkvg 2560→10240 | 49.9µs, 275GiB/s | 65.6µs (1.31배) | 121.4µs (2.43배) |
| out 4096→2560 | 20.1µs, 274GiB/s | 26.9µs (1.34배) | 49.3µs (2.46배) |
| up 2560→14336 | 70.5µs, 273GiB/s | 94.9µs (1.35배) | 173.7µs (2.46배) |
| down 7168→2560 | 37.0µs, 260GiB/s | 47.2µs (1.28배) | 84.5µs (2.28배) |
| readout INT8 대칭 g64 2560→131072 | 1.19ms, 270GiB/s | 1.33ms | 2.19ms |
| readout INT4 MLX g64 | 0.64ms, 273GiB/s | 0.81ms | 1.43ms |

- batch 1 GEMV는 이미 약 290GB/s(가정 307GB/s의 약 95%)입니다. GEMV 자체 효율은 남은 손실의 원인이 아닙니다.
- batch 2와 4는 GEMV 시간이 각각 약 1.3배, 2.4배입니다. 여러 토큰을 한 번에 검증하는 방법(추측 디코딩, 루프
  wavefront)은 이 비용을 넘는 이득이 있어야 합니다. M5 Pro(16코어, Apple10)는 m=1 타일 표에서 기본값
  `(8 simdgroup, 4 rows)`를 쓰고, m=2~7 표는 ZP 형식만 다루므로 MLX(ScaleBias) 형식은 기본 타일입니다.
- readout을 INT4로 바꾸면 토큰당 약 0.55ms(2K에서 약 3.7%)를 줄일 수 있습니다. 품질 묶음으로 판정합니다.

Dory의 attention 36층 중 27층은 sliding window 1,024이고 9층만 전체 attention입니다. 토큰당(2루프) 전체 attention
실행은 14회, sliding 실행은 42회입니다. KV를 포함한 토큰당 읽기량은 다음과 같습니다(KV 4KB/토큰/층, BF16).

| 문맥 | 가중치 | KV | 이상 시간(290GB/s) | 측정(E-03 INT4) | 나머지 |
|---|---|---|---|---|---|
| 128 | 3.24GB | 약 0.06GB | 약 11.4ms | 14.2ms(E-00) | 약 2.8ms |
| 2,048 | 3.24GB | 약 0.29GB | 약 12.2ms | 14.8ms | 약 2.6ms |
| 16,384 | 3.24GB | 약 1.12GB | 약 15.0ms | 18.4ms(E-00) | 약 3.4ms |

남은 약 2.6~2.8ms(토큰당 시간의 약 19%)는 문맥 길이와 거의 무관합니다. 디코딩은 이전 forward의 GPU 출력 토큰으로
다음 forward를 미리 인코딩하므로(`ForwardPassChaining::InFlight`) CPU 인코딩 시간은 대부분 가려집니다. 나머지는
GPU 쪽 비용(의존 dispatch 사이의 대기, attention 핵심부, 작은 커널)으로 봅니다. 다음 측정은 커널 종류를 하나씩 빼는
시간 ablation입니다(출력은 버리고 시간만 봄).

INT8 KV는 uzu에서 전체 attention, ring 없는 층에만 쓸 수 있어 Dory에서는 9층(토큰당 14회)만 해당합니다. 2K에서
약 1.4%, 16K에서 약 9%로 예상되어 우선순위를 낮춥니다.

### 커널 종류별 시간 ablation (E-03 엔진, INT4, 2026-10-19)

임시 패치(`OMEGA_ABLATE` 비트마스크, 저장소에 넣지 않음)로 커널 종류를 하나씩 빼고 decode 시간을 쟀습니다. 출력은
버리고 시간만 봅니다(정지 토큰 무시, 256토큰, 3회 중앙값). 결과 `bench/results/ablation_int4_e03_20261019.txt`.

| 뺀 커널 | 2K에서 줄어든 시간 | 128에서 줄어든 시간 | 토큰당 dispatch 수 |
|---|---|---|---|
| 선형층 GEMV(readout 제외) | 11.28ms | 11.51ms | 224 |
| attention 핵심부(prepare + attention) | 1.73ms | 0.66ms | 56 × 2~3 |
| readout | 0.97ms | 1.49ms | 1 |
| 잔차 mix | 0.59ms | 1.10ms | 112 |
| QKV norm | 0.21ms | 0.39ms | 56 |
| 합친 gate norm | 0.10ms | 0.41ms | 56 |
| 합친 SwiGLU norm | 0.13ms | 0.66ms | 56 |
| 작은 커널 넷 모두 | 1.28ms | 1.76ms | 280 |
| 위 전부 | 14.69ms(남은 0.14ms) | 14.04ms | — |

- 모델 안 GEMV는 11.28ms로, 같은 바이트를 마이크로벤치 대역폭(약 290GB/s)으로 읽는 9.98ms보다 약 1.3ms 깁니다.
  의존 dispatch마다 GEMV가 새로 차오르고 비는 비용(GEMV당 약 5.8µs)으로 봅니다.
- 작은 reduction 커널(행 하나에 threadgroup 하나)은 개당 약 4~6µs입니다. 대부분 지연 시간입니다.
- attention 핵심부는 실행당 고정 약 12µs에 KV 읽기가 더해집니다. E-04에서 single-pass를 two-pass로 바꿔도 빨라지지
  않았으므로 커널 선택보다 고정 비용(prepare, barrier) 쪽입니다.
- 결론: 2루프 INT4가 10ms(100 tok/s)에 닿으려면 GEMV만으로도 11.3ms가 걸리므로, 작은 커널을 줄이는 것으로는 부족합니다.
  토큰당 읽는 가중치 바이트를 줄이는 방법(루프 wavefront로 recurrent 가중치 공유, 더 낮은 비트)이 필요합니다.

### 생성 문장에서 루프 조기 종료 일치율

BF16 2루프 greedy로 생성한 200토큰(4개 프롬프트, 추론 켬)을 teacher-forced로 다시 넣고 1루프와 2루프의 다음 토큰을
비교했습니다(Metal parity 덤프).

| 프롬프트 | top-1 일치 | 2루프 top-1이 1루프 top-3 안 |
|---|---|---|
| 문서 요약(bench 128) | 90.0% | 99.0% |
| 코드 | 94.5% | 100% |
| GSM8K 수학 | 97.0% | 100% |
| 한국어 설명 | 88.5% | 99.5% |
| 전체 800단계 | 92.5% | 99.6% |

루프 wavefront(토큰 t의 2번째 루프와 토큰 t+1의 1번째 루프를 recurrent 층에서 batch 2로 함께 실행, t+1은 1루프 조기
종료 출력으로 추측)는 추측이 맞으면 단계당 1토큰, 틀리면 한 단계를 버립니다. 일치율 0.925면 단계당 약 0.93토큰입니다.
측정한 batch 2 GEMV 비용(1.3배)과 ablation 값으로 추정하면 INT4 약 74~78 tok/s(+10~13%), BF16 약 31 tok/s(+26%)입니다.
per_loop KV에서는 두 행이 서로 다른 루프의 KV를 쓰므로 attention만 행별로 따로 실행하면 순차 decode와 같은 계산입니다.

### E-06 루프 wavefront (엔진 API와 하네스 단계)

구현: `Transformer::encode_wavefront`(`encodable_block/transformer/wavefront.rs`), `Decoder::encode_wavefront`,
attention 행 묶음(`Attention::attend_row_groups`, `NgptLayer::encode_row_groups`). 한 단계에서
- 새 행(위치 p): 입력 층, 1번째 루프(1루프 KV), 출력 층(조기 종료 = 다음 토큰 추측)
- 이월 행(위치 p-1): 2번째 루프(2루프 KV), 출력 층(최종 logits)
을 실행합니다. recurrent 층의 GEMV, MLP, 잔차 mix는 batch 2로 한 번, attention은 행마다 자기 루프 KV로 따로
실행합니다. KV는 suffix에 남기고, 이월 행 몫(2루프, 출력 층 index 0)은 바로 accept, 새 행 몫(입력 층, 1루프)은 다음
단계에서 추측이 맞으면 accept합니다. ring KV(sliding window)는 accept 후 되돌릴 수 없으므로 이 지연 accept가
필요합니다. 추측이 틀리면 그 위치를 최종 토큰으로 다시 실행합니다(이월 행 없는 단계, 토큰 0개).

검증(`unit/session/dory_wavefront.rs`):
- teacher-forced(참조 토큰을 추측으로 넣음): parity 6개 프롬프트(1,830토큰 포함) 모두 순차 decode와 같은 판정
  통과(decode margin 31/31, 15/15, top-1 일치 수 동일). 순차 decode logits와 차이 maxabs 0.16~0.50, 평균 약 0.02,
  argmax 100% 일치. 결과 `bench/results/E-06-wavefront-parity.json`.
- greedy 자유 생성 256토큰(5개 프롬프트 × INT4/BF16): 10개 중 5개는 순차와 토큰이 완전히 같고, 나머지는 27~255번째
  토큰에서 갈라집니다. batch 2 GEMV와 출력 층 suffix 2 attention의 부동소수점 순서 차이로 근소한 동점에서 갈리는
  것으로 봅니다(teacher-forced 차이가 BF16 잡음 수준). 따라서 출력 불변 변경이 아니라 근사 변경으로 다루고 품질
  묶음으로 판정합니다.

하네스 속도(두 방법 모두 단계마다 CPU 동기화와 logits 읽기, eval 일시 정지 상태, `bench/results/E-06-wavefront-harness.json`):

| 모델 | 프롬프트 | 순차 tok/s | wavefront tok/s | 개선 | 추측 일치율 |
|---|---|---|---|---|---|
| INT4 | bench 2K | 63.31 | 75.38 | +19.1% | 97.3% |
| INT4 | 요약 128 | 68.51 | 78.50 | +14.6% | 91.0% |
| INT4 | 코드 | 69.00 | 82.32 | +19.3% | 96.1% |
| INT4 | 수학 | 67.71 | 81.73 | +20.7% | 96.5% |
| INT4 | 한국어 | 68.45 | 77.88 | +13.8% | 89.8% |
| BF16 | bench 2K | 23.66 | 30.99 | +31.0% | 97.3% |
| BF16 | 요약 128 | 24.95 | 31.38 | +25.8% | 90.6% |
| BF16 | 코드 | 24.98 | 32.59 | +30.5% | 95.3% |
| BF16 | 수학 | 24.58 | 32.81 | +33.5% | 97.3% |
| BF16 | 한국어 | 24.53 | 29.93 | +22.0% | 87.1% |

하네스 순차 decode는 stream(이전 forward 출력으로 다음 forward를 미리 인코딩)보다 느립니다(INT4 2K 63.3 대 67.5).
공식 수치는 stream에 넣은 뒤 `tools/omega_bench.py`로 잽니다.

### E-06 stream 통합과 공식 측정

`engine/language_model/stream/wavefront.rs`: `OMEGA_WAVEFRONT=1`이고 greedy, grammar 없음, speculator 없음, 모델이
wavefront를 지원할 때 stream이 wavefront로 decode합니다. 단계마다 CPU가 결과를 읽어 다음 단계(이월 행 유무, 추측
토큰)를 정합니다. 토큰을 돌려주는 시점마다 확정된 KV가 순차 decode와 같으므로, stream을 닫을 때 마지막 토큰을 순차
pass 한 번으로 넣어 다음 대화 턴이 순차 경로와 같은 상태에서 시작합니다.

`tools/omega_bench.py`(워밍업 제외 5회 중앙값, eval 일시 정지):

| 모델 | 프롬프트 | 이전 | wavefront | 개선 |
|---|---|---|---|---|
| INT4 | 128 | 70.58(E-00) | 79.52 [78.97, 79.68] | +12.7% |
| INT4 | 2,048 | 67.47(E-03) | 76.16 [75.56, 76.28] | +12.9% |
| INT4 | 16,384 | 54.40(E-00) | 58.00 [57.94, 58.05] | +6.6% |
| BF16 | 2,048 | 24.70(E-03) | 30.96 [30.92, 30.97] | +25.3% |

- 16K에서 개선이 작은 이유: 두 행이 각자 자기 루프 KV를 읽으므로 KV 읽기는 줄지 않고, 16K에서는 KV 비중이 큽니다.
- 순차 stream은 이전 forward 출력으로 다음 forward를 미리 인코딩해 CPU 시간을 가리지만, wavefront는 아직 단계마다
  동기화합니다. 하네스에서 같은 동기화가 순차 decode를 토큰당 약 0.97ms 늦췄으므로(63.31 대 67.47), 다음 실험은
  다음 단계를 추측이 맞다고 보고 미리 인코딩해 두고, 맞으면 제출하고 틀리면 버리는 방식입니다(KV 장부는 저장 후 복원).

### E-07 미리 인코딩한 wavefront 단계

GPU가 단계 k를 실행하는 동안 CPU가 단계 k+1을 인코딩합니다. k+1의 들어가는 토큰은 단계 k의 조기 종료 추측을 GPU에서
복사해 씁니다. 단계 k가 끝나 추측이 맞았으면 k+1을 바로 제출하고, 틀렸으면 k+1을 제출하지 않고 버린 뒤 인코딩 전에
저장한 KV 장부(`TransformerState::bookkeeping`, 각 attention 상태의 길이와 ring offset)를 되돌립니다. accept는 인코딩
시점에 CPU 장부만 바꾸고 데이터 복사는 command buffer 안에 있으므로, 제출하지 않은 buffer는 장부만 되돌리면 됩니다.

- 2K 출력 텍스트가 동기화 판(E-06)과 같습니다(INT4, 256토큰).
- 대화 두 턴(턴 1 생성 → 닫기 → 턴 2 입력)을 같은 토큰을 새 상태에 prefill한 결과와 비교했습니다
  (`dory_wavefront_turns`, BF16). 닫을 때 두 경로(마지막 토큰이 맞은 추측 / 틀린 추측 뒤)를 모두 지났습니다. 순차
  모드와 마찬가지로 대부분 끝까지 같고, 일부 길이에서 10~13번째 토큰에서 갈라집니다. decode로 계산한 KV와 prefill로
  계산한 KV의 수치 차이 때문으로 보며, 순차 모드도 같은 문장(n=21)에서 13번째 토큰에서 갈라졌습니다.

| 모델 | 프롬프트 | E-06(동기화) | E-07(미리 인코딩) |
|---|---|---|---|
| INT4 | 128 | 79.52 | 81.04 [80.39, 81.59] |
| INT4 | 2,048 | 76.16 | 78.44 [77.33, 79.16] |
| INT4 | 16,384 | 58.00 | 59.36 [58.98, 59.36] |
| BF16 | 2,048 | 30.96 | 31.33 [31.06, 31.38] |
| INT4 + readout INT4(E-05) | 2,048 | — | 79.82 [79.74, 79.87] |

단계당 시간은 약 12.3ms(토큰/단계 0.962)로, 1루프 decode(10.25ms)보다 약 2ms 깁니다. 대부분 batch 2 GEMV 비용입니다.
MLX(ScaleBias) 형식의 m=2~4 GEMV는 입력 행마다 가중치를 다시 읽는 타일(`input_row_tile = 1`)을 씁니다. 다음 실험은 m=2
타일 조정입니다.

### E-08 batch 2 GEMV 타일

MLX 형식(scale + bias) 4비트 GEMV는 batch 2~4에서 입력 행마다 가중치를 다시 읽는 타일(`lane_default`, 입력 1행)을
썼습니다. 커널 변형 제약(`gemv.metal`)이 입력 여러 행 타일을 zero-point와 대칭 형식에만 허용했기 때문입니다.
`QuantMetadata`는 ScaleBias도 입력 여러 행에서 처리하므로(행별 input_sum에 bias 곱) 제약에서 ScaleBias 제외를 풀고,
zero-point 경로가 쓰는 타일 계열을 m=2에 썼습니다(`policy::scale_bias_batch_tile`). 타일 비교
(`bench/results/gemv_dory_m2_tiles_20261019.txt`, m=1 시간 대비):

| GEMV | m=1 | m=2 기존 타일 | m=2 입력 2행 타일 |
|---|---|---|---|
| qkvg | 49.9µs | 64.3µs (1.29배) | 52.3µs (1.05배) |
| out | 20.1µs | 25.9µs (1.29배) | 20.7µs (1.03배) |
| up | 70.5µs | 93.3µs (1.32배) | 75.1µs (1.07배) |
| down | 37.0µs | 47.2µs (1.28배) | 37.9µs (1.02배) |
| readout INT4 | 643µs | 854µs (1.33배) | 684µs (1.06배) |

- 정확도: CPU 기준(f32 누산)과 비교한 오차가 세 타일 모두 BF16 출력 1ulp 안입니다(`dory_gemv_tile_accuracy`).
- 기존 m=2 타일은 m=1과 같은 커널 설정이라 행마다 m=1과 비트가 같았습니다. 새 타일은 반올림 경로가 달라 모델 전체
  logits에서 순차 decode와 차이가 커집니다(teacher-forced INT4 평균 차이 0.015→0.03~0.08, argmax 일치 96.8~100%).
  정확도가 낮아진 것이 아니라 순차 경로와 반올림이 달라진 것입니다.
- matmul 153개, quant 115개, gemv 66개 단위 테스트 통과. m=3~5는 측정하지 않아 기존 타일을 유지합니다.

| 모델 | 프롬프트 | E-07 | E-08 |
|---|---|---|---|
| INT4 wavefront | 128 | 81.04 | 85.77 [85.10, 86.14] |
| INT4 wavefront | 2,048 | 78.44 | 82.27 [81.42, 82.35] |
| INT4 wavefront | 16,384 | 59.36 | 60.27 [59.44, 60.69] |
| INT4 wavefront + readout INT4 | 2,048 | 79.82 | 83.78 [82.85, 84.97] |
| INT4 순차(확인용) | 2,048 | 67.47(E-03) | 66.49 [65.82, 67.24] |

순차 경로 값이 E-03보다 1.5% 낮습니다. m=1 경로 코드는 바뀌지 않았으므로 eval이 끝난 뒤 다시 잽니다.

### E-09 INT8 KV(query scale 지원)

uzu는 INT8 attention 커널이 지원하는 층(전체 attention, ring 없음, MXU)에서 INT8 KV를 기본으로 씁니다. Dory는 INT8
prepare 커널이 query scale을 적용하지 못해 빠져 있었습니다. `AttentionPrepareInt8`(Metal, CPU)에 BF16 prepare와 같은
위치(RoPE 뒤, 요소형으로 반올림)의 query scale을 넣었습니다. CPU와 Metal 출력이 비트 단위로 같습니다
(`test_attention_prepare_int8`). INT8 경로는 query도 INT8로 양자화하므로 근사입니다.

- parity(BF16 모델): 6개 중 5개 통과. "think"의 마지막 logits 차이가 0.634로 BF16 허용 오차(0.5)를 넘습니다(BF16 KV
  0.484). 1,830토큰 프롬프트는 prefill margin 1824/1830(허용 6), decode top-1 log-prob 차이 0.39(BF16 KV 0.18).
  결과 `bench/results/E-09-parity.json`.
- 속도(INT4, readout INT4, wavefront): 2K 83.99 → 83.01(잡음 안), 16K 63.13 → 69.78(+10.5%). 연구 계획의 긴 문맥
  목표(16K ≥ 2K의 80%)를 INT8 KV에서만 넘습니다(84%, BF16 KV 75%).
- query scale이 있는 모델에서는 품질을 재기 전까지 `OMEGA_INT8_KV=1`일 때만 켭니다.

### 순차 decode 회귀 확인

E-08 측정에서 순차 INT4가 E-03보다 낮게 나와(66.49 대 67.47) E-03 커밋 binary와 현재 binary를 번갈아(ABBA) 쟀습니다.
현재 67.30, E-03 66.83, E-03 66.49, 현재 65.72로 차이는 −0.2%이고, 순서에 따른 변동(먼저 잰 쪽이 빠름)이 더 큽니다.
회귀는 없습니다. 같은 세션 안에서도 시간이 떨어진 측정 사이에는 2~3%의 변동이 있으므로, 그보다 작은 차이는 ABBA로
판정합니다(`bench/results/E-09-abba.log`). `tools/omega_bench.py --binary`로 다른 빌드를 잴 수 있습니다.

### 품질 묶음 BF16 기준선(추론 끔, greedy)

| 과제 | 점수 |
|---|---|
| GSM8K 200 | 90.5% (181) |
| HumanEval 164 | 84.8% (139) |
| MMLU-Pro 500 | 43.6% (218) |
| 16K 검색 20 | 95.0% (19) |
| 묶음 점수 | 78.46% |

연구 계획의 품질 조건: 1차 목표 ≥ 97%(76.11%), 확장 목표 ≥ 95%(74.54%).

### wavefront 단계 ablation (E-09 엔진, INT4 + readout INT4, 2K)

ablation 중에는 추측을 항상 맞은 것으로 처리해 모든 pass가 두 행 단계가 되게 했습니다(출력을 버리면 추측 일치율이 바뀌어
pass 비용이 섞이기 때문). 결과 `bench/results/ablation_wavefront_r4_e09_20261019.txt`, 기준 11.17~11.29ms/pass.

| 뺀 커널 | 줄어든 시간 |
|---|---|
| 선형층 GEMV | 8.09ms(이상 약 6.7ms) |
| attention 핵심부 | 1.85ms |
| 잔차 mix(72회) | 0.72ms |
| readout | 0.71ms |
| SwiGLU norm / gate norm / QKV norm | 0.25 / 0.22 / 0.18ms |
| 행 묶음 복사 | 0.04ms |
| 작은 커널 모두 | 1.19ms |

E-10에서 잔차 mix 안의 barrier를 줄여도 빨라지지 않았으므로, 작은 커널 비용은 커널 안 계산보다 dispatch 경계와 그
앞뒤 대기입니다. 2루프 INT4 100 tok/s(토큰당 10ms)까지 남은 약 1.9ms를 줄일 후보와 추정 효과:

| 후보 | 추정 효과 | 비고 |
|---|---|---|
| 잔차 mix, gate norm, SwiGLU norm을 이웃 GEMV의 prologue/epilogue로 합침 | 0.5~0.8ms | GEMV 템플릿 변경이 큼 |
| 모델 안(의존 dispatch) 기준 m=1, m=2 GEMV 타일 재조정 | 0.3~0.6ms | 마이크로벤치는 독립 dispatch라 모델 안 차이를 놓침 |
| 행 묶음 attention을 prepare 둘 → attention 둘 순서로 인코딩(동시 실행 늘림), QKV norm과 prepare 합침 | 0.1~0.3ms | |
| 두 후보 추측(batch 3)으로 빈 단계(약 4%) 줄임 | 0.2~0.4ms | m=3 GEMV 타일 필요 |

### llama.cpp 비교 (참고, 형식이 달라 직접 비교 아님)

다른 작업 세션이 Dory 지원 llama.cpp(`98a74c9`, CPU 버퍼 배치 수정 후)를 같은 기기에서 쟀습니다
(`docs/dory_speed_20261018.md`, 잠정 표시, Q5_K_M, 128토큰, context 4096). 같은 양자화 형식이 아니므로 토큰당 바이트와
실효 대역폭으로 나란히 둡니다.

| 엔진 | 형식 | 루프 | 토큰당 가중치 | decode tok/s | 실효 대역폭 |
|---|---|---|---|---|---|
| llama.cpp | Q5_K_M | 2 | 4.28GB | 42.76 | 183GB/s |
| Ω-Hee 순차 | MLX INT4 g64 + INT8 readout | 2 | 3.24GB | 70.58(E-00, 128) | 229GB/s |
| Ω-Hee 순차 | BF16 | 2 | 10.95GB | 25.15(E-00, 128) | 275GB/s |
| Ω-Hee wavefront | MLX INT4 g64 + INT8 readout | 2 | (토큰당 약 1루프분) | 85.77(E-08, 128) | — |

같은 형식 비교(예: llama.cpp Q4_0과 Ω-Hee INT4)와 대조 모델(Qwen3.5-4B) 측정은 하지 않았습니다. 디스크 여유가
19GB(96% 사용)라 대조 모델 다운로드와 변환(약 16GB)을 미뤘습니다.

## 3단계 품질: 양자화 형식 탐색 (2026-10-19)

### 품질 묶음 결과

| 변형 | GSM8K | HumanEval | MMLU-Pro | 16K 검색 | 묶음 | BF16 대비 |
|---|---|---|---|---|---|---|
| BF16 | 90.5 | 84.8 | 43.6 | 95.0 | 78.46 | 100% |
| INT4 MLX g64 + readout INT8 | 83.0 | 74.4 | 41.0 | 95.0 | 73.35 | 93.5% |

INT4 MLX g64는 1차 품질 조건(97%)과 확장 조건(95%)을 모두 넘지 못합니다. 연구 계획 3단계 절차(그룹 크기 탐색 →
경로 B NVFP4)에 따라 형식을 탐색했습니다.

### 빠른 근사 지표: teacher-forced KL

품질 묶음 하나는 40분 이상 걸리므로, 후보를 먼저 BF16 출력 분포와의 KL로 거릅니다. BF16이 생성한 GSM8K 답 8개와
HumanEval 답 4개(2,845 위치)를 prefill로 넣고 위치마다 KL(BF16 ‖ 변형)과 top-1 일치율을 잽니다. 지정하지 않은 행렬은
BF16(원래 정밀도)입니다.

| 변형 | 평균 KL | top-1 일치 |
|---|---|---|
| INT4 MLX g64 전체(+ readout INT8) | 0.135 | 90.8% |
| INT4 MLX g64 전체 + readout INT4 | 0.136 | 91.0% |
| INT4 MLX g32 전체 | 0.099 | 91.7% |
| 입력 층 0~15만 INT4 | 0.017 | 96.6% |
| 반복 층 16~55만 INT4 | 0.096 | 91.5% |
| 출력 층 56~71만 INT4 | 0.030 | 95.4% |
| 반복 층 qkvg만 INT4 | 0.049 | 93.7% |
| 반복 층 up만 INT4 | 0.038 | 95.3% |
| 반복 층 down만 INT4 | 0.016 | 97.0% |
| 반복 층 out만 INT4 | 0.011 | 97.5% |
| 혼합: 반복 층 qkvg INT8, 나머지 INT4 | 0.090 | 92.1% |
| 혼합: 반복 층 qkvg·up INT8, 나머지 INT4 | 0.063 | 93.9% |
| 혼합: 반복 층 전체 INT8, 입력·출력 INT4 | 0.044 | 94.6% |
| 공식 NVFP4 체크포인트 값(INT8로 다시 저장) | 0.0435 | 95.8% |
| BF16을 NVFP4로 반올림(RTN) | 0.090 | 92.2% |

- 오차의 대부분은 반복 층에서 나옵니다(같은 가중치를 루프마다 다시 쓰므로 오차가 쌓임). 그 안에서는 qkvg와 up이
  민감합니다.
- readout INT4(E-05)는 KL을 거의 바꾸지 않습니다(0.135 → 0.136).
- 공식 NVFP4(`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4`, ModelOpt W4A16, 16값 블록, FP8 E4M3 블록 스케일 +
  FP32 텐서 스케일)는 4비트 그대로 반복 층 INT8 혼합과 같은 KL입니다. 같은 NVFP4 형식이라도 BF16을 단순 반올림하면
  0.090이므로, 차이는 NVIDIA의 보정에서 옵니다. 경로 B는 공식 부호를 손실 없이 가져와야 합니다.
- 가중치 평균 상대 오차(층 16 q, 층 17 down): 공식 NVFP4 0.083, RTN NVFP4 0.087, MLX g64 0.103, MLX g32 0.086.

혼합 정밀도 속도(2K, 순차 / wavefront): 반복 층 INT8 48.17 / 61.69, 반복 층 qkvg·up INT8 50.63 / 65.08
(`bench/results/E-11-speed.log`). 품질 묶음 측정 중입니다.

도구: `tools/omega_convert.py --spec-rule`(행렬별 형식), `tools/nvfp4_dequant.py`(ModelOpt NVFP4 → BF16 checkout,
`--rtn-from`으로 RTN 비교, `--check`로 값 확인).

### 품질 묶음: 반복 층 INT8 혼합

| 변형 | GSM8K | HumanEval | MMLU-Pro | 16K 검색 | 묶음 | BF16 대비 |
|---|---|---|---|---|---|---|
| 혼합: 반복 층 INT8, 입력·출력 INT4 g64, readout INT8 | 88.5 | 86.6 | 42.0 | 90.0 | 76.77 | 97.8% |

KL 0.044인 변형이 1차 품질 조건(97%)을 넘었습니다. 같은 KL의 공식 NVFP4 값으로 4비트 실행 경로를 만들었습니다(E-12).

### E-12 NVFP4 실행 경로 (경로 B)

- 형식: `MicrofloatSpec`(bits 4, group 16, `nvfp4`, `output_input`). 부호는 U8 [행, 열/2](바이트 하나에 두 개, 아래
  니블 먼저), 스케일은 BF16 [그룹, 행]으로 FP8 E4M3 블록 스케일과 FP32 텐서 스케일을 곱해 접은 값입니다. 융합
  행렬(qkvg = q, k, v, g; up = up, gate)은 원래 텐서마다 텐서 스케일이 다르므로 BF16로 접어야 손실이 없습니다
  (E4M3로 다시 맞추면 블록 스케일에 최대 약 6% 오차).
- 엔진: `QuantizationMethod::ScaleFp4`, `GemmBPrologueKind::ScaleFp4Dequant`. GEMV는 그룹 16 변형과 e2m1 해독을
  더했고, GEMM(prefill)은 FP4 행렬을 BF16 scratch로 전개(`Fp4Dequantize`)한 뒤 기존 BF16 GEMM을 씁니다. INT8 활성값
  GEMM 경로는 FP4에서 끕니다. CPU 기준 구현과 임베딩 조회도 지원합니다.
- 변환: `tools/nvfp4_dequant.py`(NVFP4 → BF16 checkout) → `tools/omega_convert.py`(기반 모델) →
  `tools/nvfp4_to_omega.py`(선형층 144개를 공식 부호로 교체). NVFP4 체크포인트의 BF16 텐서(α, 스케일, 임베딩) 219개 중
  170개가 BF16 체크포인트와 1~2ulp 다르므로, 기반 모델도 NVFP4 체크포인트에서 만듭니다.
- 검증: GEMV(m=1, 2)와 전개 + GEMM(m=16)이 CPU 기준과 BF16 1ulp 안(`dory_fp4_matmul_matches_cpu`, 두 파라미터
  배치). 모델 KL 0.0428, top-1 96.0%(INT8로 다시 저장한 대리 모델 0.0435). matmul 155, quant 115, gemv 67, embedding 12
  테스트 통과.
- 해독 비용: 처음 해독(지수·가수 분해와 선택)은 GEMV를 연산에 묶어 180GiB/s에 그쳤습니다. FP16 비트로 바로 만드는
  해독(`(c & 7) << 9 | (c & 8) << 12`를 FP16으로 읽으면 e2m1 × 2^-14, 2^14는 그룹 스케일에 접음)으로 260~271GiB/s가
  되었고, 모델 decode가 순차 44.74 → 60.72, wavefront 53.68 → 69.07 tok/s로 올랐습니다.
- 남은 차이: BF16 그룹 스케일이 MLX g64보다 선형층 바이트를 약 11% 늘립니다(FP8 스케일이면 같아짐). FP4 m=2 GEMV는
  m=1의 1.16~1.25배로 MLX(1.02~1.07배)보다 나빠 wavefront 이득이 작습니다.
